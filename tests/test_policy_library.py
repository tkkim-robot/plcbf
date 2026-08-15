from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from plcbf.benchmarking import (
    BenchmarkOutcome,
    BenchmarkResult,
    NumericSummary,
    aggregate_results,
    results_to_csv,
    results_to_json,
    results_to_markdown,
    write_benchmark_reports,
)
from plcbf.policy_library import (
    CBFHalfspace,
    PolicyCertificate,
    SelectionMode,
    box_halfspace_volume,
    cbf_halfspace,
    clip_rectangle_halfspaces,
    rectangle_halfspaces_area,
    select_policy,
    solve_box_halfspace_qp,
    solve_weighted_box_halfspaces_qp_2d,
)


class CBFHalfspaceTests(unittest.TestCase):
    def test_cbf_construction_matches_source_inequality(self) -> None:
        gradient = np.array([2.0, -1.0])
        drift = np.array([1.0, 3.0])
        control_matrix = np.array([[1.0, 2.0], [4.0, -1.0]])
        constraint = cbf_halfspace(
            gradient,
            drift,
            control_matrix,
            value=0.5,
            value_time_derivative=0.2,
            alpha=2.0,
            buffer=0.1,
            label="turn-left",
        )

        np.testing.assert_allclose(constraint.normal, [-2.0, 5.0])
        self.assertAlmostEqual(constraint.offset, 0.0)
        control = np.array([0.3, -0.4])
        source_lhs = (
            gradient @ (drift + control_matrix @ control)
            + 0.2
            + 2.0 * (0.5 - 0.1)
        )
        self.assertAlmostEqual(constraint.residual(control), source_lhs)
        self.assertEqual(constraint.label, "turn-left")

    def test_certificate_factory_and_readonly_arrays(self) -> None:
        certificate = PolicyCertificate.from_cbf(
            "hover",
            value=1.0,
            gradient=[1.0, 0.0],
            drift=[0.0, 0.0],
            control_matrix=np.eye(2),
            backup_control=[0.0, 0.0],
        )

        self.assertEqual(certificate.policy_id, "hover")
        self.assertEqual(len(certificate.halfspaces), 1)
        self.assertFalse(certificate.halfspaces[0].normal.flags.writeable)
        self.assertFalse(certificate.backup_control.flags.writeable)

    def test_invalid_shapes_and_negative_alpha_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cbf_halfspace(
                [1.0, 2.0],
                [0.0, 0.0],
                np.ones((3, 1)),
                value=0.0,
            )
        with self.assertRaises(ValueError):
            cbf_halfspace(
                [1.0],
                [0.0],
                [[1.0]],
                value=0.0,
                alpha=-1.0,
            )


class SingleHalfspaceQPTests(unittest.TestCase):
    def test_returns_clipped_reference_when_already_feasible(self) -> None:
        result = solve_box_halfspace_qp(
            [2.0, 0.25],
            [-1.0, -1.0],
            [1.0, 1.0],
            CBFHalfspace([1.0, 0.0], -0.5),
        )

        self.assertTrue(result.feasible)
        np.testing.assert_allclose(result.control, [1.0, 0.25])
        self.assertIn("upper[0]", result.active_constraints)
        self.assertAlmostEqual(result.objective, 0.5)

    def test_unclipped_projection(self) -> None:
        result = solve_box_halfspace_qp(
            [0.0, 0.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            1.0,
        )

        self.assertTrue(result.feasible)
        np.testing.assert_allclose(result.control, [0.5, 0.5], atol=1e-9)
        self.assertAlmostEqual(result.objective, 0.25, places=9)

    def test_monotone_clipping_handles_saturated_coordinate(self) -> None:
        result = solve_box_halfspace_qp(
            [0.0, 0.0],
            [-1.0, -1.0],
            [0.25, 1.0],
            ([1.0, 1.0], 1.0),
        )

        self.assertTrue(result.feasible)
        np.testing.assert_allclose(result.control, [0.25, 0.75], atol=1e-8)
        self.assertAlmostEqual(result.objective, 0.3125, places=8)
        self.assertLessEqual(result.max_violation, 1e-9)

    def test_reports_infeasible_box_intersection(self) -> None:
        result = solve_box_halfspace_qp(
            [0.0, 0.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            ([1.0, 1.0], 3.0),
        )

        self.assertFalse(result.feasible)
        self.assertIsNone(result.control)
        self.assertEqual(result.status, "infeasible")
        self.assertAlmostEqual(result.max_violation, 1.0)

    def test_zero_normal_can_be_tautological_or_infeasible(self) -> None:
        feasible = solve_box_halfspace_qp(
            [0.0], [-1.0], [1.0], ([0.0], 0.0)
        )
        infeasible = solve_box_halfspace_qp(
            [0.0], [-1.0], [1.0], ([0.0], 0.1)
        )

        self.assertTrue(feasible.feasible)
        self.assertFalse(infeasible.feasible)
        self.assertEqual(infeasible.status, "infeasible_zero_normal")


class VolumeAndPolygonTests(unittest.TestCase):
    def test_exact_volume_in_one_two_and_three_dimensions(self) -> None:
        one_dimensional = box_halfspace_volume(
            [0.0], [2.0], ([1.0], 0.5)
        )
        triangle = box_halfspace_volume(
            [0.0, 0.0], [1.0, 1.0], ([1.0, 1.0], 1.0)
        )
        half_cube = box_halfspace_volume(
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            ([1.0, 1.0, 1.0], 1.5),
        )

        self.assertAlmostEqual(one_dimensional, 1.5)
        self.assertAlmostEqual(triangle, 0.5)
        self.assertAlmostEqual(half_cube, 0.5)

    def test_volume_handles_negative_coefficients_and_degenerate_cases(self) -> None:
        x_at_most_quarter = box_halfspace_volume(
            [0.0], [1.0], ([-1.0], -0.25)
        )
        full = box_halfspace_volume(
            [-1.0, -1.0], [1.0, 1.0], ([0.0, 0.0], 0.0)
        )
        empty = box_halfspace_volume(
            [-1.0, -1.0], [1.0, 1.0], ([0.0, 0.0], 0.1)
        )
        zero_measure = box_halfspace_volume(
            [0.0, 0.0], [0.0, 1.0], ([1.0, 0.0], 0.0)
        )

        self.assertAlmostEqual(x_at_most_quarter, 0.25)
        self.assertAlmostEqual(full, 4.0)
        self.assertEqual(empty, 0.0)
        self.assertEqual(zero_measure, 0.0)

    def test_volume_dimension_cap_is_explicit(self) -> None:
        with self.assertRaises(ValueError):
            box_halfspace_volume(
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                ([1.0, 1.0, 1.0], 1.0),
                max_active_dimension=2,
            )

    def test_rectangle_clipping_returns_vertices_and_area(self) -> None:
        clipped = clip_rectangle_halfspaces(
            [0.0, 0.0],
            [1.0, 1.0],
            [
                CBFHalfspace([1.0, 0.0], 0.25),
                CBFHalfspace([0.0, 1.0], 0.5),
            ],
        )

        self.assertAlmostEqual(clipped.area, 0.375)
        self.assertEqual(clipped.vertices.shape, (4, 2))
        self.assertTrue(np.all(clipped.vertices[:, 0] >= 0.25 - 1e-10))
        self.assertTrue(np.all(clipped.vertices[:, 1] >= 0.5 - 1e-10))
        self.assertFalse(clipped.vertices.flags.writeable)

    def test_triangle_and_empty_rectangle(self) -> None:
        triangle_area = rectangle_halfspaces_area(
            [0.0, 0.0],
            [1.0, 1.0],
            [([1.0, 1.0], 1.0)],
        )
        empty = clip_rectangle_halfspaces(
            [0.0, 0.0],
            [1.0, 1.0],
            [([0.0, 0.0], 1.0)],
        )

        self.assertAlmostEqual(triangle_area, 0.5)
        self.assertTrue(empty.is_empty)
        self.assertEqual(empty.vertices.shape, (0, 2))


class WeightedQPTests(unittest.TestCase):
    def test_diagonal_weight_changes_boundary_projection(self) -> None:
        result = solve_weighted_box_halfspaces_qp_2d(
            [0.0, 0.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [([1.0, 1.0], 1.0)],
            weights=[1.0, 4.0],
        )

        self.assertTrue(result.feasible)
        np.testing.assert_allclose(result.control, [0.8, 0.2], atol=1e-10)
        self.assertAlmostEqual(result.objective, 0.4)

    def test_candidate_enumeration_finds_vertex_solution(self) -> None:
        result = solve_weighted_box_halfspaces_qp_2d(
            [0.0, 0.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [([1.0, 0.0], 0.8), ([0.0, 1.0], 0.7)],
            weights=np.array([[2.0, 0.5], [0.5, 1.0]]),
        )

        self.assertTrue(result.feasible)
        np.testing.assert_allclose(result.control, [0.8, 0.7])
        self.assertGreaterEqual(len(result.active_constraints), 2)

    def test_candidate_enumeration_reports_empty_intersection(self) -> None:
        result = solve_weighted_box_halfspaces_qp_2d(
            [0.0, 0.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [([1.0, 0.0], 0.8), ([-1.0, 0.0], -0.7)],
        )

        self.assertFalse(result.feasible)
        self.assertIsNone(result.control)

    def test_non_positive_definite_weights_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            solve_weighted_box_halfspaces_qp_2d(
                [0.0, 0.0],
                [-1.0, -1.0],
                [1.0, 1.0],
                weights=[1.0, 0.0],
            )


class PolicySelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.large_volume = PolicyCertificate(
            policy_id="large-volume",
            value=0.2,
            halfspaces=(CBFHalfspace([1.0, 0.0], 0.0),),
            backup_control=[-0.5, 0.0],
        )
        self.large_value = PolicyCertificate(
            policy_id="large-value",
            value=0.5,
            halfspaces=(CBFHalfspace([1.0, 0.0], 0.8),),
            backup_control=[0.5, 0.0],
        )

    def select(self, mode: SelectionMode | str):
        return select_policy(
            [self.large_value, self.large_volume],
            [0.0, 0.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            mode=mode,
        )

    def test_value_volume_and_intervention_modes(self) -> None:
        by_value = self.select(SelectionMode.VALUE)
        by_volume = self.select("input_space")
        by_intervention = self.select("per_policy_intervention")

        self.assertEqual(by_value.policy_id, "large-value")
        np.testing.assert_allclose(by_value.control, [0.8, 0.0], atol=1e-9)
        self.assertEqual(by_volume.policy_id, "large-volume")
        np.testing.assert_allclose(by_volume.control, [0.0, 0.0])
        self.assertEqual(by_intervention.policy_id, "large-volume")
        self.assertFalse(by_value.diagnostics.used_fallback)
        self.assertEqual(by_value.diagnostics.eligible_policy_count, 2)

    def test_all_unsafe_executes_least_unsafe_policy_backup(self) -> None:
        worse = PolicyCertificate(
            "worse",
            -0.5,
            (CBFHalfspace([1.0, 0.0], 0.0),),
            backup_control=[0.0, 0.0],
        )
        least_unsafe = PolicyCertificate(
            "least-unsafe",
            -0.1,
            (CBFHalfspace([1.0, 0.0], 0.0),),
            backup_control=[2.0, -2.0],
        )
        decision = select_policy(
            [worse, least_unsafe],
            [0.0, 0.0],
            [-1.0, -1.0],
            [1.0, 1.0],
        )

        self.assertEqual(decision.policy_id, "least-unsafe")
        np.testing.assert_allclose(decision.control, [1.0, -1.0])
        self.assertTrue(decision.diagnostics.used_fallback)
        self.assertEqual(
            decision.diagnostics.fallback_reason, "no_safe_policy"
        )
        self.assertEqual(
            decision.diagnostics.fallback_source, "selected_policy_backup"
        )

    def test_safe_but_infeasible_uses_global_fallback(self) -> None:
        infeasible = PolicyCertificate(
            "infeasible",
            1.0,
            (CBFHalfspace([1.0, 0.0], 2.0),),
        )
        decision = select_policy(
            [infeasible],
            [0.0, 0.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            fallback_control=[-0.25, 0.25],
        )

        np.testing.assert_allclose(decision.control, [-0.25, 0.25])
        self.assertEqual(
            decision.diagnostics.fallback_reason,
            "safe_policies_infeasible",
        )
        self.assertEqual(
            decision.diagnostics.fallback_source, "global_fallback"
        )

    def test_empty_library_explicitly_returns_clipped_nominal(self) -> None:
        decision = select_policy(
            [],
            [2.0, -2.0],
            [-1.0, -1.0],
            [1.0, 1.0],
        )

        self.assertIsNone(decision.policy_id)
        np.testing.assert_allclose(decision.control, [1.0, -1.0])
        self.assertEqual(
            decision.diagnostics.fallback_reason, "empty_policy_library"
        )
        self.assertEqual(
            decision.diagnostics.fallback_source, "clipped_nominal"
        )

    def test_ties_preserve_policy_library_order(self) -> None:
        second = PolicyCertificate("z-policy", 1.0)
        first = PolicyCertificate("a-policy", 1.0)
        decision = select_policy(
            [second, first],
            [0.0, 0.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            mode="value",
        )

        self.assertEqual(decision.policy_id, "z-policy")

    def test_value_tie_does_not_use_volume_or_intervention(self) -> None:
        first = PolicyCertificate(
            "first-small-volume",
            1.0,
            (CBFHalfspace([1.0, 0.0], 0.9),),
        )
        second = PolicyCertificate("second-large-volume", 1.0)
        decision = select_policy(
            [first, second],
            [0.0, 0.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            mode="value",
        )

        self.assertEqual(decision.policy_id, first.policy_id)

    def test_input_volume_and_value_tie_ignores_intervention(self) -> None:
        first = PolicyCertificate(
            "first-expensive",
            1.0,
            (CBFHalfspace([-1.0, 0.0], 0.8),),
        )
        second = PolicyCertificate(
            "second-cheap",
            1.0,
            (CBFHalfspace([1.0, 0.0], 0.8),),
        )
        decision = select_policy(
            [first, second],
            [0.7, 0.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            mode="input_volume",
        )

        self.assertEqual(decision.policy_id, first.policy_id)

    def test_polygon_reuse_matches_general_qp_on_random_certificates(self) -> None:
        rng = np.random.default_rng(20260809)
        lower = np.array([-1.7, -1.7])
        upper = np.array([1.7, 1.7])
        for sample in range(160):
            count = int(rng.integers(2, 7))
            halfspaces = tuple(
                CBFHalfspace(
                    rng.normal(size=2),
                    float(rng.uniform(-2.8, 2.8)),
                    f"random-{sample}-{index}",
                )
                for index in range(count)
            )
            reference = rng.uniform(-2.2, 2.2, size=2)
            weights = rng.uniform(0.4, 2.5, size=2)
            certificate = PolicyCertificate(
                f"candidate-{sample}",
                1.0,
                halfspaces,
                backup_control=np.zeros(2),
            )

            decision = select_policy(
                (certificate,),
                reference,
                lower,
                upper,
                weights=weights,
                mode="input_volume",
                tolerance=1e-9,
            )
            evaluation = decision.diagnostics.evaluations[0]
            expected = solve_weighted_box_halfspaces_qp_2d(
                reference,
                lower,
                upper,
                halfspaces,
                weights=weights,
                tolerance=1e-9,
            )
            self.assertEqual(evaluation.feasible, expected.feasible)
            self.assertAlmostEqual(
                evaluation.input_volume,
                rectangle_halfspaces_area(
                    lower,
                    upper,
                    halfspaces,
                    tolerance=1e-9,
                ),
                places=10,
            )
            if expected.feasible:
                self.assertIsNotNone(evaluation.control)
                np.testing.assert_allclose(
                    evaluation.control,
                    expected.control,
                    rtol=1e-9,
                    atol=2e-8,
                )
                self.assertAlmostEqual(
                    evaluation.intervention_cost,
                    expected.objective,
                    places=9,
                )
            else:
                self.assertIsNone(evaluation.control)

    def test_intervention_tie_does_not_use_value_or_volume(self) -> None:
        first = PolicyCertificate("first-low-value", 0.1)
        second = PolicyCertificate("second-high-value", 10.0)
        decision = select_policy(
            [first, second],
            [0.0, 0.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            mode="intervention",
        )

        self.assertEqual(decision.policy_id, first.policy_id)

    def test_duplicate_policy_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_policy(
                [PolicyCertificate("same", 1.0), PolicyCertificate("same", 2.0)],
                [0.0, 0.0],
                [-1.0, -1.0],
                [1.0, 1.0],
            )


class BenchmarkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = (
            BenchmarkResult(
                algorithm="A",
                case_id="case-b",
                seed=2,
                outcome=BenchmarkOutcome.COLLISION,
                min_clearance=-0.2,
                intervention=2.0,
                solve_times_s=(0.03,),
                case_metrics={"hid_in_room": False, "progress": 0.4},
            ),
            BenchmarkResult(
                algorithm="B",
                case_id="case-a",
                seed=1,
                outcome="success",
                min_clearance=0.8,
                intervention=0.25,
                solve_times_s=(),
                case_metrics={"label": "nominal"},
            ),
            BenchmarkResult(
                algorithm="A",
                case_id="case-a",
                seed=1,
                outcome="success",
                min_clearance=0.5,
                intervention=0.5,
                solve_times_s=(0.01, 0.02),
                case_metrics={"hid_in_room": True, "progress": 1.0},
            ),
        )

    def test_result_outcome_flags_and_solve_statistics(self) -> None:
        result = self.results[2]
        self.assertTrue(result.success)
        self.assertFalse(result.collision)
        self.assertAlmostEqual(result.solve_time_mean_s, 0.015)
        self.assertAlmostEqual(result.solve_time_p95_s, 0.0195)
        self.assertAlmostEqual(result.solve_time_max_s, 0.02)

    def test_numpy_scalar_case_metrics_are_normalized(self) -> None:
        result = BenchmarkResult(
            "A",
            "case",
            np.int64(3),
            "success",
            case_metrics={
                "boolean": np.bool_(True),
                "integer": np.int64(2),
                "floating": np.float64(0.5),
            },
        )

        self.assertIs(result.case_metrics["boolean"], True)
        self.assertEqual(result.case_metrics["integer"], 2)
        self.assertEqual(result.case_metrics["floating"], 0.5)

    def test_aggregation_covers_outcomes_and_numerical_metrics(self) -> None:
        aggregate_a, aggregate_b = aggregate_results(self.results)

        self.assertEqual(aggregate_a.algorithm, "A")
        self.assertEqual(aggregate_a.trial_count, 2)
        self.assertEqual(aggregate_a.success_count, 1)
        self.assertEqual(aggregate_a.collision_count, 1)
        self.assertAlmostEqual(aggregate_a.success_rate, 0.5)
        self.assertAlmostEqual(aggregate_a.clearance.mean, 0.15)
        self.assertAlmostEqual(aggregate_a.clearance.minimum, -0.2)
        self.assertAlmostEqual(aggregate_a.solve_time_mean_s, 0.02)
        self.assertAlmostEqual(aggregate_a.solve_time_p95_s, 0.029)
        self.assertAlmostEqual(aggregate_a.solve_time_max_s, 0.03)
        self.assertAlmostEqual(
            aggregate_a.case_metrics["hid_in_room"].mean, 0.5
        )
        self.assertEqual(aggregate_b.algorithm, "B")
        self.assertIsNone(aggregate_b.solve_time_s)

    def test_numeric_summary_linear_percentile(self) -> None:
        summary = NumericSummary.from_values([0.01, 0.02, 0.03])

        self.assertIsNotNone(summary)
        self.assertAlmostEqual(summary.p95, 0.029)
        summary_dict = summary.as_dict()
        self.assertEqual(summary_dict["count"], 3)
        self.assertAlmostEqual(summary_dict["mean"], 0.02)
        self.assertAlmostEqual(summary_dict["min"], 0.01)
        self.assertAlmostEqual(summary_dict["p95"], 0.029)
        self.assertAlmostEqual(summary_dict["max"], 0.03)

    def test_csv_is_sorted_and_contains_dynamic_metric_columns(self) -> None:
        csv_text = results_to_csv(reversed(self.results))
        rows = list(csv.DictReader(io.StringIO(csv_text)))

        self.assertEqual(
            [(row["algorithm"], row["case_id"]) for row in rows],
            [("A", "case-a"), ("A", "case-b"), ("B", "case-a")],
        )
        self.assertIn("metric.hid_in_room", rows[0])
        self.assertEqual(rows[0]["success"], "true")
        self.assertEqual(rows[1]["collision"], "true")
        self.assertEqual(rows[0]["metric.hid_in_room"], "true")

    def test_json_is_deterministic_and_contains_raw_and_aggregate_data(self) -> None:
        forward = results_to_json(
            self.results, metadata={"z": 2, "config": {"dt": 0.05}}
        )
        reverse = results_to_json(
            reversed(self.results), metadata={"config": {"dt": 0.05}, "z": 2}
        )
        document = json.loads(forward)

        self.assertEqual(forward, reverse)
        self.assertEqual(document["metadata"]["config"]["dt"], 0.05)
        self.assertEqual(document["aggregates"][0]["algorithm"], "A")
        self.assertEqual(document["results"][0]["case_id"], "case-a")
        self.assertTrue(document["results"][0]["success"])
        self.assertTrue(forward.endswith("\n"))

    def test_markdown_contains_all_outcomes_timing_and_case_metrics(self) -> None:
        report = results_to_markdown(self.results, title="Hospital | benchmark")

        self.assertIn("# Hospital \\| benchmark", report)
        self.assertIn("Collision", report)
        self.assertIn("Infeasible", report)
        self.assertIn("Timeout", report)
        self.assertIn("Solve mean / p95 ms", report)
        self.assertIn("hid_in_room mean", report)
        self.assertLess(report.index("| A |"), report.index("| B |"))

    def test_report_bundle_writes_the_three_stable_formats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "nested" / "benchmark"
            paths = write_benchmark_reports(
                prefix,
                self.results,
                metadata={"seed_count": 2},
                title="Test benchmark",
            )

            self.assertEqual(paths.csv.suffix, ".csv")
            self.assertEqual(paths.json.suffix, ".json")
            self.assertEqual(paths.markdown.suffix, ".md")
            self.assertEqual(
                paths.csv.read_text(encoding="utf-8"),
                results_to_csv(self.results),
            )
            self.assertEqual(
                paths.json.read_text(encoding="utf-8"),
                results_to_json(self.results, metadata={"seed_count": 2}),
            )
            self.assertEqual(
                paths.markdown.read_text(encoding="utf-8"),
                results_to_markdown(self.results, title="Test benchmark"),
            )

    def test_invalid_benchmark_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkResult("A", "case", 0, "unknown")
        with self.assertRaises(ValueError):
            BenchmarkResult(
                "A", "case", 0, "success", solve_times_s=(-0.1,)
            )
        with self.assertRaises(ValueError):
            BenchmarkResult(
                "A", "case", 0, "success", intervention=-0.1
            )
        with self.assertRaises(ValueError):
            BenchmarkResult(
                "A", "case", 0, "success", case_metrics={"metric": np.inf}
            )


if __name__ == "__main__":
    unittest.main()
