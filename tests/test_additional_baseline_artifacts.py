"""Artifact-key and audit-label regressions for additional baselines."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from examples.drift_car import benchmark_additional_baselines as drift
from examples.warehouse import benchmark_additional_baselines_quad as warehouse


EXPECTED_KEYS = {"multi_backup_cbf_mi", "library_pcbf_mi"}
DRIFT_TRIAL_FIELDS = {
    "algorithm",
    "seed",
    "run_idx",
    "obstacle_geometry",
    "library_size",
    "collision",
    "unrecoverable_infeasible",
    "historical_failure",
    "total_steps",
    "timed_steps",
    "solve_time_sum_sec",
}
DRIFT_SUMMARY_FIELDS = {
    "key",
    "label",
    "n",
    "fail_count",
    "fail_rate",
    "collision_count",
    "collision_rate",
    "unrecoverable_count",
    "unrecoverable_rate",
    "total_timed_steps",
    "mean_compute_ms",
}
WAREHOUSE_TRIAL_FIELDS = {
    "algorithm",
    "seed",
    "run_idx",
    "obstacle_geometry",
    "library_size",
    "collision",
    "unrecoverable_infeasible",
    "historical_failure",
    "solve_time_sum_sec",
    "timed_steps",
    "total_steps",
}
WAREHOUSE_SUMMARY_FIELDS = {
    "key",
    "label",
    "n_trials",
    "library_size",
    "collisions",
    "unrecoverable_infeasibles",
    "fail_count",
    "collision_rate_pct",
    "unrecoverable_infeasible_rate_pct",
    "fail_rate_pct",
    "avg_compute_ms",
    "total_timed_steps",
}
ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIRECTORY = (
    ROOT
    / "output/additional_baselines/projection_audited_baseline_only_2026-07-14"
)
PRE_PROJECTION_AUDIT_DIRECTORY = (
    ROOT / "output/additional_baselines/native_control_baseline_only_2026-07-13"
)


def _assert_projection_trial(trial):
    # Publication artifacts are required to have zero runtime/simulator errors.
    # Under that acceptance condition, every controller call has one recorded
    # simulator step and the per-step audit arrays must match ``total_steps``.
    assert trial["runtime_error"] is False
    assert trial["unrecoverable_infeasible"] is False
    count_fields = {
        "num_post_projection_audits": "num_post_projection_audits_per_step",
        "projection_event_count": "projection_event_count_per_step",
        "post_projection_rejection_count": (
            "post_projection_rejection_count_per_step"
        ),
    }
    maximum_fields = {
        "max_projection_delta_inf": "max_projection_delta_inf_per_step",
        "max_post_projection_constraint_violation": (
            "max_post_projection_constraint_violation_per_step"
        ),
        "max_post_projection_violation_ratio": (
            "max_post_projection_violation_ratio_per_step"
        ),
    }
    for scalar, per_step in count_fields.items():
        assert len(trial[per_step]) == trial["total_steps"]
        assert trial[scalar] == sum(trial[per_step])
    for scalar, per_step in maximum_fields.items():
        assert len(trial[per_step]) == trial["total_steps"]
        assert trial[scalar] == max(trial[per_step], default=0.0)

    assert trial["projection_occurred"] is (
        trial["projection_event_count"] > 0
    )
    assert (trial["max_projection_delta_inf"] > 0.0) is (
        trial["projection_event_count"] > 0
    )
    assert trial["max_projection_delta_inf"] <= 1e-5 + 1e-12
    assert trial["projection_event_count"] <= trial["num_post_projection_audits"]
    assert (
        trial["post_projection_rejection_count"]
        <= trial["num_post_projection_audits"]
    )
    assert trial["infeasible"] is trial["unrecoverable_infeasible"]
    assert trial["historical_failure"] is (
        trial["collision"] or trial["unrecoverable_infeasible"]
    )
    if trial["post_projection_rejection_count"] == 0:
        assert trial["max_post_projection_violation_ratio"] <= 1.0
    else:
        assert trial["max_post_projection_violation_ratio"] > 1.0


def _assert_projection_summary(summary, trials):
    trial_count_field = "n" if "n" in summary else "n_trials"
    assert summary[trial_count_field] == len(trials)
    assert summary["num_post_projection_audits"] > 0
    assert summary["projection_trial_count"] == sum(
        int(trial["projection_occurred"]) for trial in trials
    )
    for field in (
        "num_post_projection_audits",
        "projection_event_count",
        "post_projection_rejection_count",
    ):
        assert summary[field] == sum(trial[field] for trial in trials)
    for field in (
        "max_projection_delta_inf",
        "max_post_projection_constraint_violation",
        "max_post_projection_violation_ratio",
    ):
        assert summary[field] == max(
            (trial[field] for trial in trials), default=0.0
        )
    assert (summary["max_projection_delta_inf"] > 0.0) is (
        summary["projection_event_count"] > 0
    )
    if summary["post_projection_rejection_count"] == 0:
        assert summary["max_post_projection_violation_ratio"] <= 1.0
    else:
        assert summary["max_post_projection_violation_ratio"] > 1.0

    summary_to_trial = {
        "fail_count": "historical_failure",
        "collision_count": "collision",
        "collisions": "collision",
        "infeasible_count": "infeasible",
        "infeasibles": "infeasible",
        "unrecoverable_infeasibles": "unrecoverable_infeasible",
        "runtime_error_count": "runtime_error",
        "runtime_errors": "runtime_error",
        "certificate_lost_count": "certificate_lost",
        "certificate_losses": "certificate_lost",
        "qp_infeasible_count": "qp_infeasible",
        "qp_infeasibles": "qp_infeasible",
        "union_failure_count": "union_failure",
        "union_failures": "union_failure",
        "task_completed_count": "task_completed",
        "task_completions": "task_completed",
        "survived_horizon_count": "survived_horizon",
        "horizon_survivals": "survived_horizon",
        "successful_outcome_count": "completed_or_survived",
        "successful_outcomes": "completed_or_survived",
        "filter_failure_count": "filter_failure",
        "filter_failures": "filter_failure",
    }
    for summary_field, trial_field in summary_to_trial.items():
        if summary_field in summary:
            assert summary[summary_field] == sum(
                int(trial[trial_field]) for trial in trials
            )


def _digest(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _drift_result(variant, scenario):
    return drift.EpisodeResult(
        algorithm=variant.key,
        seed=scenario.seed,
        run_idx=scenario.run_idx,
        obstacle_geometry=list(scenario.obstacles),
        library_size=4,
        collision=False,
        unrecoverable_infeasible=False,
        historical_failure=False,
        total_steps=1,
        timed_steps=1,
        solve_time_sum_sec=0.001,
    )


def test_drift_writer_emits_only_two_baseline_keys(tmp_path, monkeypatch):
    scenario = drift.Scenario(0, 7, 1, ((80.0, "middle"),))
    monkeypatch.setattr(drift, "generate_scenarios", lambda *args: [scenario])
    monkeypatch.setattr(
        drift,
        "run_episode",
        lambda variant, item, config, verbose=False: _drift_result(variant, item),
    )
    paths = {
        suffix: tmp_path / f"drift.{suffix}" for suffix in ("md", "json", "csv")
    }
    drift.main(
        [
            "--num-runs",
            "1",
            "--output-md",
            str(paths["md"]),
            "--output-json",
            str(paths["json"]),
            "--output-csv",
            str(paths["csv"]),
        ]
    )

    payload = json.loads(paths["json"].read_text())
    assert set(payload["algorithms"]) == EXPECTED_KEYS
    assert set(payload["trials"]) == EXPECTED_KEYS
    assert {row["key"] for row in payload["summary"]} == EXPECTED_KEYS
    assert all(
        set(row) == DRIFT_SUMMARY_FIELDS for row in payload["summary"]
    )
    assert all(
        set(trial) == DRIFT_TRIAL_FIELDS
        for trials in payload["trials"].values()
        for trial in trials
    )
    with paths["csv"].open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert set(reader.fieldnames or ()) == DRIFT_TRIAL_FIELDS
    assert {row["algorithm"] for row in rows} == EXPECTED_KEYS


def _warehouse_result(algo, scenario):
    return warehouse.TrialResult(
        algorithm=algo,
        seed=scenario.seed,
        run_idx=scenario.run_idx,
        obstacle_geometry=list(scenario.ghosts),
        library_size=66,
        collision=False,
        unrecoverable_infeasible=False,
        historical_failure=False,
        solve_time_sum_sec=0.001,
        timed_steps=1,
        total_steps=1,
    )


def test_warehouse_writer_emits_only_two_baseline_keys(tmp_path, monkeypatch):
    scenario = warehouse.TrialScenario(0, 11, tuple())
    monkeypatch.setattr(
        warehouse, "generate_random_scenarios", lambda **kwargs: [scenario]
    )
    monkeypatch.setattr(
        warehouse,
        "run_trial",
        lambda algo, scenario, **kwargs: _warehouse_result(algo, scenario),
    )
    paths = {
        suffix: tmp_path / f"warehouse.{suffix}"
        for suffix in ("md", "json", "csv")
    }
    warehouse.main(
        [
            "--num-trials",
            "1",
            "--output-md",
            str(paths["md"]),
            "--output-json",
            str(paths["json"]),
            "--output-csv",
            str(paths["csv"]),
        ]
    )

    payload = json.loads(paths["json"].read_text())
    assert set(payload["algorithms"]) == EXPECTED_KEYS
    assert set(payload["trials"]) == EXPECTED_KEYS
    assert {row["key"] for row in payload["summaries"]} == EXPECTED_KEYS
    assert all(
        set(row) == WAREHOUSE_SUMMARY_FIELDS for row in payload["summaries"]
    )
    assert all(
        set(trial) == WAREHOUSE_TRIAL_FIELDS
        for trials in payload["trials"].values()
        for trial in trials
    )
    with paths["csv"].open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert set(reader.fieldnames or ()) == WAREHOUSE_TRIAL_FIELDS
    assert {row["algorithm"] for row in rows} == EXPECTED_KEYS


def test_superseded_directory_is_labeled_common_stop_audit():
    directory = (
        ROOT / "output/additional_baselines/reviewer_fix_2026-07-13"
    )
    manifest = json.loads((directory / "reproduction_manifest.json").read_text())
    readme = (directory / "README.md").read_text().lower()
    report = (directory / "corrected_results_report.md").read_text().lower()
    assert manifest["artifact_role"] == "common_stop_audit"
    assert manifest["not_for_publication"] is True
    assert "common-stop audit" in readme
    assert "not for publication" in readme
    assert "do not cite" in report


def test_publication_manifest_and_artifacts_are_baseline_only():
    manifest = json.loads(
        (PUBLICATION_DIRECTORY / "reproduction_manifest.json").read_text()
    )
    assert (
        manifest["artifact_role"]
        == "publication_projection_audited_baseline_only"
    )
    assert manifest["not_for_publication"] is False
    assert set(manifest["algorithm_keys"]) == EXPECTED_KEYS
    assert manifest["scope"]["standalone_historical_trial_executed"] is False
    assert manifest["scope"]["historical_solve_control_problem_called"] is False
    assert manifest["scope"]["drift_historical_reference_constructed"] is True
    assert (
        manifest["scope"]["superseded_pre_projection_audit"]
        == "../native_control_baseline_only_2026-07-13"
    )
    assert PRE_PROJECTION_AUDIT_DIRECTORY.is_dir()
    assert manifest["source_checkout"]["parent_status_porcelain"] == ""
    safe_control = manifest["source_checkout"]["submodules"]["safe_control"]
    assert safe_control["gitlink"] == safe_control["head"]
    assert safe_control["status_porcelain"] == ""
    projection_audit = manifest["projection_audit_contract"]
    assert projection_audit["all_original_qp_inequalities_rechecked"] is True
    assert projection_audit["failed_audit_rejects_candidate"] is True
    assert manifest["environment"]["timing_is_tentative"] is True

    expected_benchmarks = {
        "drift_outcome",
        "warehouse_outcome",
        "drift_timing_serial3",
        "warehouse_timing_serial3",
    }
    assert set(manifest["benchmarks"]) == expected_benchmarks
    timing_benchmarks = {
        name
        for name, benchmark in manifest["benchmarks"].items()
        if benchmark["publication_timing"]
    }
    assert timing_benchmarks == {
        "drift_timing_serial3",
        "warehouse_timing_serial3",
    }
    for name, benchmark in manifest["benchmarks"].items():
        assert "plcbf" not in json.dumps(benchmark["command"]).lower(), name
        if benchmark["publication_timing"]:
            assert benchmark["workers"] == 1
            assert benchmark["trials_per_algorithm"] == 3
    assert (
        manifest["benchmarks"]["drift_timing_serial3"][
            "warmup_calls_excluded_per_trial"
        ]
        == 5
    )
    assert (
        manifest["benchmarks"]["warehouse_timing_serial3"][
            "warmup_calls_excluded_per_trial"
        ]
        == 10
    )

    json_specs = {
        "drift_seed7_50.json": ("summary", 50),
        "drift_timing_serial3.json": ("summary", 3),
        "warehouse_seed11_100.json": ("summaries", 100),
        "warehouse_timing_serial3.json": ("summaries", 3),
    }
    loaded_payloads = {}
    for filename, (summary_key, trials_per_method) in json_specs.items():
        payload = json.loads((PUBLICATION_DIRECTORY / filename).read_text())
        loaded_payloads[filename] = payload
        assert set(payload["algorithms"]) == EXPECTED_KEYS
        assert set(payload["trials"]) == EXPECTED_KEYS
        assert {row["key"] for row in payload[summary_key]} == EXPECTED_KEYS
        assert all(
            len(payload["trials"][key]) == trials_per_method
            for key in EXPECTED_KEYS
        )
        summaries = {row["key"]: row for row in payload[summary_key]}
        for key in EXPECTED_KEYS:
            trials = payload["trials"][key]
            for trial in trials:
                _assert_projection_trial(trial)
            _assert_projection_summary(summaries[key], trials)

        paired = [
            sorted(payload["trials"][key], key=lambda row: row["run_idx"])
            for key in sorted(EXPECTED_KEYS)
        ]
        for left, right in zip(*paired):
            assert left["run_idx"] == right["run_idx"]
            assert left["seed"] == right["seed"]
            assert left["obstacle_geometry"] == right["obstacle_geometry"]

        audit = payload["projection_audit"]
        assert audit["scope"].startswith(
            "every finite, actuator-tolerance-valid successful-status candidate QP"
        )
        assert "every original affine QP inequality" in audit["action"]
        assert "reject the candidate" in audit["action"]
        assert audit["osqp_absolute_tolerance"] == 1e-5
        assert audit["osqp_relative_tolerance"] == 1e-5
        assert audit["multi_backup_scs_fallback_absolute_tolerance"] == 1e-4
        assert audit["multi_backup_scs_fallback_relative_tolerance"] == 1e-4
        if filename.startswith("drift_"):
            assert audit["drift_library_scs_absolute_tolerance"] == 1e-4
            assert audit["drift_library_scs_relative_tolerance"] == 1e-4

    drift_outcome = loaded_payloads["drift_seed7_50.json"]
    drift_rows = sorted(
        drift_outcome["trials"]["multi_backup_cbf_mi"],
        key=lambda row: row["run_idx"],
    )
    drift_geometry = [
        {
            "run_idx": row["run_idx"],
            "num_obstacles": len(row["obstacle_geometry"]),
            "obstacles": row["obstacle_geometry"],
        }
        for row in drift_rows
    ]
    assert _digest(drift_geometry) == (
        "b1462f165fd0dcfa811d8a334d3d8c1106c83d063e66e92b62a318e20b617e15"
    )

    warehouse_outcome = loaded_payloads["warehouse_seed11_100.json"]
    warehouse_rows = sorted(
        warehouse_outcome["trials"]["multi_backup_cbf_mi"],
        key=lambda row: row["run_idx"],
    )
    warehouse_geometry = [
        {"run_idx": row["run_idx"], "ghosts": row["obstacle_geometry"]}
        for row in warehouse_rows
    ]
    assert _digest(warehouse_geometry) == (
        "125b631d315b3c9debdc141caa22a1d5105757bb252255568f8cec845e656604"
    )

    for domain in ("drift", "warehouse"):
        outcome = loaded_payloads[f"{domain}_seed{'7_50' if domain == 'drift' else '11_100'}.json"]
        timing = loaded_payloads[f"{domain}_timing_serial3.json"]
        for key in EXPECTED_KEYS:
            outcome_prefix = sorted(
                outcome["trials"][key], key=lambda row: row["run_idx"]
            )[:3]
            timing_rows = sorted(
                timing["trials"][key], key=lambda row: row["run_idx"]
            )
            assert [
                (row["run_idx"], row["seed"], row["obstacle_geometry"])
                for row in timing_rows
            ] == [
                (row["run_idx"], row["seed"], row["obstacle_geometry"])
                for row in outcome_prefix
            ]

    drift_timing = loaded_payloads["drift_timing_serial3.json"]
    assert drift_timing["num_runs"] == 3
    assert "1 worker(s)" in drift_timing["timing_note"]
    assert "first five" in drift_timing["timing_note"]
    warehouse_timing = loaded_payloads["warehouse_timing_serial3.json"]
    assert warehouse_timing["config"]["num_trials"] == 3
    assert warehouse_timing["timing"]["worker_count"] == 1
    assert warehouse_timing["timing"]["warmup_calls_excluded"] == 10

    csv_specs = {
        "drift_seed7_50.csv": 50,
        "drift_timing_serial3.csv": 3,
        "warehouse_seed11_100.csv": 100,
        "warehouse_timing_serial3.csv": 3,
    }
    for filename, trials_per_method in csv_specs.items():
        csv_path = PUBLICATION_DIRECTORY / filename
        assert b"\r\n" not in csv_path.read_bytes()
        with csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 2 * trials_per_method
        assert {row["algorithm"] for row in rows} == EXPECTED_KEYS
        assert {
            "projection_occurred",
            "num_post_projection_audits",
            "projection_event_count",
            "post_projection_rejection_count",
            "max_projection_delta_inf",
            "max_post_projection_constraint_violation",
            "max_post_projection_violation_ratio",
        } <= rows[0].keys()

    for filename in (
        "drift_seed7_50.md",
        "drift_timing_serial3.md",
        "warehouse_seed11_100.md",
        "warehouse_timing_serial3.md",
    ):
        report = (PUBLICATION_DIRECTORY / filename).read_text()
        assert "## Post-projection QP audit" in report

    for filename, expected_digest in manifest["artifacts_sha256"].items():
        digest = hashlib.sha256(
            (PUBLICATION_DIRECTORY / filename).read_bytes()
        ).hexdigest()
        assert digest == expected_digest


def test_publication_report_preserves_submitted_plcbf_rows_verbatim():
    report = (PUBLICATION_DIRECTORY / "publication_report.md").read_text()
    normalized_report = " ".join(report.split())
    assert "No standalone PL-CBF trial was run" in report
    assert "solve_control_problem` was never called" in report
    assert "tentative" in report.lower()
    assert (
        "not directly comparable to the frozen PL-CBF timing"
        in normalized_report
    )
    assert (
        "| PL-CBF (ours) — unchanged submitted value | $\\Pi$ | "
        "0/50 (0.0%) | 10.00 | 7.522 |"
    ) in report
    assert (
        "| PL-CBF (ours) — unchanged submitted value | 64 | "
        "0/100 (0.0%) | 27.986 |"
    ) in report
