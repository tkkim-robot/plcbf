from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import examples.hospital.benchmark as benchmark
from examples.hospital.config import DEFAULT_CONFIG
from examples.hospital.obstacles import Human, Stretcher
from plcbf.baselines import BaselineDecision
from plcbf.benchmarking import BenchmarkOutcome


def _fast_config():
    return replace(
        DEFAULT_CONFIG,
        policies=replace(
            DEFAULT_CONFIG.policies,
            nominal_horizon=0.24,
            angle_horizon=0.24,
            reverse_horizon=0.24,
            stop_horizon=0.24,
            room_horizon=0.36,
            room_rollout_dt=0.18,
            num_angle_policies=2,
            room_policy_count=1,
        ),
    )


def _signature(simulation) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(np.r_[obstacle.center, obstacle.velocity])
        for obstacle in simulation.obstacles
    )


def test_seeded_cases_are_crowded_deterministic_and_keep_blockage() -> None:
    for case_id, blocker_count in benchmark.STRICT_HOSPITAL_CASES.items():
        first = benchmark.build_benchmark_scenario(case_id, seed=17)
        repeated = benchmark.build_benchmark_scenario(case_id, seed=17)
        changed = benchmark.build_benchmark_scenario(case_id, seed=18)

        humans = [
            obstacle
            for obstacle in first.obstacles
            if isinstance(obstacle, Human)
        ]
        stretchers = [
            obstacle
            for obstacle in first.obstacles
            if isinstance(obstacle, Stretcher)
        ]
        blockers = [
            obstacle
            for obstacle in stretchers
            if obstacle.identifier.startswith("blocking-stretcher-")
        ]
        assert len(humans) == 50
        assert len(stretchers) == 15 + blocker_count
        assert len(blockers) == blocker_count
        assert all(not item.reflect_at_route_bounds for item in blockers)
        assert all(
            item.cross_section_width == 7.1 for item in blockers
        )
        assert _signature(first) == _signature(repeated)
        assert _signature(first) != _signature(changed)
        metrics = first.benchmark_scenario_metrics
        assert metrics["dynamic_obstacle_count"] in {67, 68}
        assert metrics["full_width_blockade"] is True
        assert metrics["maximum_reverse_swept_before_west_junction"] is True


def test_randomization_metadata_describes_dense_paired_trials() -> None:
    metadata = benchmark.hospital_randomization_protocol_metadata()
    assert metadata["paired_seed_shared_across_methods"] is True
    assert metadata["every_seed_generates_dynamic_background_traffic"] is True
    assert metadata["human_count"] == 50
    assert metadata["ordinary_stretcher_count"] == 15
    assert metadata["total_stretcher_counts"] == [17, 18]


def test_default_benchmark_uses_full_plcbf_policy_library() -> None:
    parser = benchmark.build_parser()
    arguments = parser.parse_args([])
    config, period, compact = benchmark._resolve_cli_protocol(arguments)
    assert compact is False
    assert period == config.dt
    simulation = benchmark.build_benchmark_scenario(
        "blocked_3_stretchers",
        seed=0,
        config=config,
    )
    policies = simulation.controller.candidate_policies(simulation.state)
    assert len(policies) == 13
    assert sum(policy.kind == "angle" for policy in policies) == 7
    assert sum(policy.kind == "room" for policy in policies) == 3


def test_one_step_dense_benchmark_has_no_external_refuge_executor() -> None:
    result = benchmark.run_hospital_trial(
        "plcbf",
        "blocked_2_stretchers",
        seed=2,
        steps=1,
        config=_fast_config(),
        oracle_period_s=_fast_config().dt,
        raise_errors=True,
    )
    assert result.outcome in {
        BenchmarkOutcome.TIMEOUT,
        BenchmarkOutcome.INFEASIBLE,
    }
    assert result.case_metrics["dynamic_obstacle_count"] == 67
    assert result.case_metrics["external_refuge_state_machine"] is False
    assert result.case_metrics["external_room_policy_executor"] is False
    assert result.case_metrics["external_room_selector"] is False
    assert result.case_metrics["room_policy_available_to_method"] is True
    assert "refuge_minimum_hold_time_s" not in result.case_metrics
    assert "release_before_clear_violation" not in result.case_metrics


def test_stale_oracle_period_is_rejected() -> None:
    with pytest.raises(ValueError, match="every plant step"):
        benchmark.run_hospital_trial(
            "plcbf",
            "blocked_2_stretchers",
            steps=1,
            config=_fast_config(),
            oracle_period_s=0.12,
            raise_errors=True,
        )


def test_outcome_classification_has_no_hold_or_release_protocol() -> None:
    assert (
        benchmark._classify_outcome(
            physical_collision=False,
            operational_safety_violation=False,
            goal_reached=True,
            infeasible_count=0,
        )
        is BenchmarkOutcome.SUCCESS
    )
    assert (
        benchmark._classify_outcome(
            physical_collision=False,
            operational_safety_violation=True,
            goal_reached=False,
            infeasible_count=0,
        )
        is BenchmarkOutcome.INFEASIBLE
    )


@pytest.mark.parametrize(
    ("method", "status", "used_fallback", "feasible", "expected_backup"),
    [
        ("backup_cbf", "nominal_after_qp_failure:failed", True, False, False),
        ("backup_cbf", "fallback_after:failed", True, False, True),
        ("mi_mpc", "optimal", False, True, False),
        ("mi_mpc", "solver_failure", True, False, True),
    ],
)
def test_backup_execution_is_not_inferred_from_method_name(
    method: str,
    status: str,
    used_fallback: bool,
    feasible: bool,
    expected_backup: bool,
) -> None:
    decision = BaselineDecision(
        method=method,
        control=np.zeros(2),
        policy_id="test",
        feasible=feasible,
        status=status,
        used_fallback=used_fallback,
        objective=0.0,
        solve_time_s=0.0,
    )

    _, _, backup, _ = benchmark._decision_event_flags(
        benchmark.BenchmarkMethod(method),
        decision,
    )

    assert backup is expected_backup


def test_mi_mpc_relaxed_admission_is_not_requested_safety_feasible() -> None:
    suite = SimpleNamespace(
        last_mi_mpc_result=SimpleNamespace(
            safety_feasible=False,
            safety_threshold_relaxed=True,
        )
    )

    assert benchmark._mi_mpc_safety_flags(
        benchmark.BenchmarkMethod.MI_MPC,
        suite,
    ) == (False, True)


@pytest.mark.parametrize(
    "method",
    ["plcbf", "multi_backup_cbf_mi", "library_pcbf_mi"],
)
def test_room_policy_availability_is_true_only_for_library_methods(
    method: str,
) -> None:
    assert benchmark._room_policy_available_to_method(
        benchmark.BenchmarkMethod(method)
    )


@pytest.mark.parametrize(
    "method",
    ["pcbf", "backup_cbf", "mps", "gatekeeper", "mi_mpc"],
)
def test_room_policy_availability_excludes_native_fixed_and_mi_methods(
    method: str,
) -> None:
    assert not benchmark._room_policy_available_to_method(
        benchmark.BenchmarkMethod(method)
    )
