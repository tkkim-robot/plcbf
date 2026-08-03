from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import examples.hospital.benchmark as benchmark
from examples.hospital.config import DEFAULT_CONFIG
from examples.hospital.obstacles import Human, Stretcher
from examples.hospital.provenance import (
    HOSPITAL_BENCHMARK_SOURCE_FILES,
    HOSPITAL_BENCHMARK_SOURCE_MANIFEST_SCHEMA,
)
from examples.hospital.simulation import SweptTransitionSafety
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
    assert metadata["paired_world_shared_across_methods"] is True
    assert metadata["human_count"] == 50
    assert metadata["ordinary_stretcher_count"] == 0
    assert metadata["randomized_entities"] == ["human"]
    assert metadata["trial_count"] == 100


def test_source_manifest_hashes_ordered_relative_paths_and_exact_bytes() -> None:
    manifest = benchmark.hospital_benchmark_source_manifest()
    root = Path(benchmark.__file__).resolve().parents[2]
    combined = hashlib.sha256()

    def frame(value: bytes) -> bytes:
        return len(value).to_bytes(8, "big") + value

    combined.update(
        frame(HOSPITAL_BENCHMARK_SOURCE_MANIFEST_SCHEMA.encode("utf-8"))
    )
    expected_files = []
    for relative_path in HOSPITAL_BENCHMARK_SOURCE_FILES:
        content = (root / relative_path).read_bytes()
        combined.update(frame(relative_path.encode("utf-8")))
        combined.update(frame(content))
        expected_files.append(
            {
                "path": relative_path,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )

    assert manifest == {
        "schema": HOSPITAL_BENCHMARK_SOURCE_MANIFEST_SCHEMA,
        "hash_algorithm": "sha256",
        "combined_sha256": combined.hexdigest(),
        "files": expected_files,
    }
    assert list(HOSPITAL_BENCHMARK_SOURCE_FILES) == sorted(
        HOSPITAL_BENCHMARK_SOURCE_FILES
    )


def test_cli_report_metadata_embeds_source_manifest(
    monkeypatch, tmp_path
) -> None:
    captured: dict[str, object] = {}
    events: list[str] = []
    expected_manifest = {
        "schema": HOSPITAL_BENCHMARK_SOURCE_MANIFEST_SCHEMA,
        "combined_sha256": "test-startup-fingerprint",
        "files": [],
    }

    def fake_write(prefix, results, *, metadata, title):
        del results, title
        captured.update(metadata)
        return SimpleNamespace(
            csv=prefix.with_suffix(".csv"),
            json=prefix.with_suffix(".json"),
            markdown=prefix.with_suffix(".md"),
        )

    def fake_manifest():
        events.append("manifest")
        return expected_manifest

    def fake_run(**_):
        events.append("run")
        return ()

    monkeypatch.setattr(
        benchmark,
        "hospital_benchmark_source_manifest",
        fake_manifest,
    )
    monkeypatch.setattr(benchmark, "run_hospital_benchmark", fake_run)
    monkeypatch.setattr(benchmark, "write_benchmark_reports", fake_write)
    monkeypatch.setattr(
        benchmark,
        "write_hospital_benchmark_markdown",
        lambda *args, **kwargs: None,
    )

    exit_code = benchmark.main(
        ["--quick", "--output", str(tmp_path / "hospital-shard")]
    )

    assert exit_code == 0
    assert events == ["manifest", "run"]
    assert captured["implementation_source_manifest"] == expected_manifest


def test_default_benchmark_uses_full_plcbf_policy_library() -> None:
    parser = benchmark.build_parser()
    arguments = parser.parse_args([])
    config, period, compact = benchmark._resolve_cli_protocol(arguments)
    assert tuple(arguments.cases) == benchmark.HOSPITAL_BENCHMARK_STORIES
    assert tuple(arguments.seeds) == tuple(range(20))
    assert len(arguments.cases) * len(arguments.seeds) == 100
    assert (
        config.safety.max_obstacles
        == benchmark.PUBLICATION_MAX_SENSED_OBSTACLES
        == 53
    )
    assert (
        config.robot.sensing_range
        == benchmark.PUBLICATION_SENSING_RANGE_M
        == 24.0
    )
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


def test_publication_grid_prebuilds_each_world_once_for_all_methods(
    monkeypatch,
) -> None:
    built: dict[tuple[str, int], object] = {}
    observed: list[tuple[str, str, int, object]] = []

    def fake_build(story_id, *, traffic_seed, config):
        del config
        scenario = object()
        built[(story_id, traffic_seed)] = scenario
        return scenario

    def fake_trial(method, case_id, *, seed, _scenario, **kwargs):
        del kwargs
        observed.append((str(method), case_id, seed, _scenario))
        return SimpleNamespace()

    monkeypatch.setattr(
        benchmark,
        "build_hospital_story_scenario",
        fake_build,
    )
    monkeypatch.setattr(benchmark, "run_hospital_trial", fake_trial)

    results = benchmark.run_hospital_benchmark(
        methods=("pcbf", "plcbf"),
        cases=benchmark.HOSPITAL_BENCHMARK_STORIES,
        seeds=range(20),
        steps=1,
    )

    assert len(results) == len(observed) == 200
    assert len(built) == 100
    for _method, story_id, seed, scenario in observed:
        assert scenario is built[(story_id, seed)]


def test_narrative_trace_ignores_initial_room_but_counts_later_reentry() -> None:
    scenario = benchmark.build_hospital_story_scenario(
        "main_eastbound",
        traffic_seed=0,
        human_count=0,
    )
    simulation = scenario.to_simulation(
        benchmark.publication_benchmark_config()
    )
    simulation.benchmark_scenario_metrics.update(
        {
            "blockage_started_at_s": 0.1,
            "blockage_cleared_at_s": 0.3,
        }
    )
    trace = benchmark._HospitalNarrativeTrace.from_simulation(simulation)
    start_room = trace.initial_room
    assert start_room is not None
    assert trace.post_departure_room is None

    simulation.time = 0.12
    trace.update(
        simulation,
        transition_started_at_s=0.06,
        transition_safety_clearance=1.0,
    )
    assert not trace.start_room_left
    assert not trace.room_occupied_during_blockage

    outside = simulation.environment.room_door_path(
        start_room,
        simulation.config.robot.radius,
        simulation.config.refuge.inside_door_offset,
        simulation.config.refuge.outside_door_offset,
    )[0]
    simulation.state[:2] = outside
    simulation.time = 0.15
    trace.update(
        simulation,
        transition_started_at_s=0.12,
        transition_safety_clearance=1.0,
    )
    assert trace.start_room_left
    assert trace.post_departure_room is None

    simulation.state[:2] = start_room.center
    simulation.time = 0.2
    trace.update(
        simulation,
        transition_started_at_s=0.1,
        transition_safety_clearance=0.5,
    )
    assert trace.post_departure_room is start_room
    assert trace.room_occupied_during_blockage

    simulation.state[:2] = outside
    simulation.time = 0.4
    trace.update(
        simulation,
        transition_started_at_s=0.2,
        transition_safety_clearance=0.5,
    )
    assert trace.blockage_window_observed
    assert trace.safe_through_blockage
    assert trace.room_exited_after_clear


def test_every_method_receives_the_same_frozen_perception_snapshot(
    monkeypatch,
) -> None:
    captured: dict[str, tuple[str, ...]] = {}

    def build_sparse_case(case_id, *, seed, config):
        del case_id, seed
        return benchmark.build_blocked_main_hall_scenario(2, config=config)

    def frozen_snapshot(state, obstacles, config, *, environment=None):
        del state, config, environment
        return tuple(obstacles[:1])

    def record_solve(self, method, state, obstacles, nominal, **kwargs):
        del self, state, kwargs
        parsed = benchmark.BenchmarkMethod(method)
        captured[parsed.value] = tuple(
            obstacle.identifier for obstacle in obstacles
        )
        return BaselineDecision(
            method=parsed.value,
            control=np.asarray(nominal, dtype=float),
            policy_id="nominal",
            feasible=True,
            status="test",
            used_fallback=False,
            objective=0.0,
            solve_time_s=0.0,
        )

    monkeypatch.setattr(benchmark, "build_benchmark_scenario", build_sparse_case)
    monkeypatch.setattr(benchmark, "sensed_obstacles", frozen_snapshot)
    monkeypatch.setattr(
        benchmark.HospitalController,
        "build_policy_certificates",
        lambda *args, **kwargs: ((), ()),
    )
    monkeypatch.setattr(
        benchmark.HospitalBaselineSuite,
        "solve",
        record_solve,
    )

    for method in benchmark.BENCHMARK_METHODS:
        benchmark.run_hospital_trial(
            method,
            "blocked_2_stretchers",
            seed=0,
            steps=1,
            config=_fast_config(),
            oracle_period_s=_fast_config().dt,
            warmup=False,
            raise_errors=True,
        )

    assert set(captured) == set(benchmark.BENCHMARK_METHODS)
    assert set(captured.values()) == {("blocking-stretcher-0",)}


def test_one_step_dense_benchmark_has_no_external_refuge_executor() -> None:
    result = benchmark.run_hospital_trial(
        "plcbf",
        "blocked_2_stretchers",
        seed=2,
        steps=1,
        config=_fast_config(),
        oracle_period_s=_fast_config().dt,
        warmup=False,
        raise_errors=True,
    )
    assert result.outcome is BenchmarkOutcome.TIMEOUT
    assert result.case_metrics["dynamic_obstacle_count"] == 67
    assert result.case_metrics["external_refuge_state_machine"] is False
    assert result.case_metrics["external_room_policy_executor"] is False
    assert result.case_metrics["external_room_selector"] is False
    assert result.case_metrics["room_policy_available_to_method"] is True
    assert "refuge_minimum_hold_time_s" not in result.case_metrics
    assert "release_before_clear_violation" not in result.case_metrics


@pytest.mark.parametrize("method", ("plcbf", "library_pcbf_mi"))
def test_jax_library_warmup_is_excluded_and_runtime_has_no_cache_miss(
    method: str,
) -> None:
    config = replace(
        _fast_config(),
        safety=replace(_fast_config().safety, max_obstacles=4),
    )
    result = benchmark.run_hospital_trial(
        method,
        "blocked_2_stretchers",
        seed=0,
        steps=2,
        config=config,
        oracle_period_s=config.dt,
        warmup=True,
        raise_errors=True,
    )
    metrics = result.case_metrics

    assert metrics["jit_warmup_enabled"] is True
    assert metrics["jit_warmup_excluded_from_step_timing"] is True
    assert metrics["runtime_jit_cache_miss_delta"] == 0
    assert metrics["runtime_jit_compilation_detected"] is False
    assert (
        metrics["jit_cache_misses_after_trial"]
        == metrics["jit_cache_misses_after_warmup"]
    )


def test_trial_persists_synchronized_clearance_attribution() -> None:
    result = benchmark.run_hospital_trial(
        "mps",
        "blocked_2_stretchers",
        seed=0,
        steps=1,
        config=_fast_config(),
        oracle_period_s=_fast_config().dt,
        warmup=False,
        raise_errors=True,
    )
    metrics = result.case_metrics
    assert (
        metrics["clearance_diagnostic_schema_version"]
        == "hospital_clearance_witness_v1"
    )
    assert metrics["minimum_physical_clearance"] == result.min_clearance
    for prefix in (
        "minimum_physical_clearance",
        "minimum_safety_clearance",
    ):
        source = metrics[f"{prefix}_source_kind"]
        obstacle_identifier = metrics[f"{prefix}_obstacle_identifier"]
        assert source in {"static", "human", "stretcher"}
        assert (obstacle_identifier is None) is (source == "static")
        assert 0 <= metrics[f"{prefix}_step_index"] <= metrics["steps"]
        assert 0.0 <= metrics[f"{prefix}_time_s"] <= metrics["sim_time_s"]
        assert 0 <= metrics[f"{prefix}_substep_index"] <= 8
        assert 0.0 <= metrics[f"{prefix}_substep_fraction"] <= 1.0
        assert (
            0.0
            <= metrics[f"{prefix}_transition_elapsed_s"]
            <= metrics["plant_dt_s"]
        )


def test_stale_oracle_period_is_rejected() -> None:
    with pytest.raises(ValueError, match="every plant step"):
        benchmark.run_hospital_trial(
            "plcbf",
            "blocked_2_stretchers",
            steps=1,
            config=_fast_config(),
            oracle_period_s=0.12,
            warmup=False,
            raise_errors=True,
        )


def test_outcome_classification_has_no_hold_or_release_protocol() -> None:
    assert (
        benchmark._classify_outcome(
            physical_collision=False,
            goal_reached=True,
        )
        is BenchmarkOutcome.SUCCESS
    )
    assert (
        benchmark._classify_outcome(
            physical_collision=True,
            goal_reached=True,
        )
        is BenchmarkOutcome.COLLISION
    )
    assert (
        benchmark._classify_outcome(
            physical_collision=False,
            goal_reached=False,
        )
        is BenchmarkOutcome.TIMEOUT
    )


def test_deadlock_monitor_is_post_clear_observation_only_and_can_resolve() -> None:
    monitor = benchmark._DeadlockMonitor(
        eligible_after_s=5.0,
        window_s=2.0,
        max_path_length_m=0.1,
        max_goal_progress_m=0.1,
        max_speed_mps=0.05,
    )
    state = np.array([10.0, 10.0, 0.0, 0.0])
    goal = np.array([20.0, 10.0])
    for time_s in (0.0, 2.0, 4.9, 5.0, 6.0):
        assert monitor.update(time_s, state, goal) is False
    assert monitor.update(7.0, state, goal) is True
    assert monitor.detected is True
    assert monitor.first_detected_at_s == pytest.approx(7.0)
    assert monitor.episode_count == 1

    moving = np.array([10.2, 10.0, 0.2, 0.0])
    assert monitor.update(7.1, moving, goal) is False
    assert monitor.resolved_episode_count == 1


def test_hospital_default_horizon_is_three_minutes() -> None:
    assert benchmark.DEFAULT_HOSPITAL_SIMULATION_TIME_S == 180.0
    assert benchmark.default_hospital_benchmark_steps() == 3000


def test_operational_violation_is_diagnostic_and_trial_can_reach_goal(
    monkeypatch,
) -> None:
    config = _fast_config()
    simulation = benchmark.build_benchmark_scenario(
        "blocked_2_stretchers",
        seed=0,
        config=config,
    )
    transition_calls = 0
    plant_calls = 0

    def fake_transition(*args, **kwargs):
        nonlocal transition_calls
        del args, kwargs
        transition_calls += 1
        safety = -0.1 if transition_calls == 2 else 1.0
        physical_witness = benchmark.ClearanceWitness(
            value=1.0,
            source_kind="static",
            obstacle_identifier=None,
            elapsed_s=0.0,
            sample_index=0,
            sample_fraction=0.0,
            robot_position=(50.0, 47.5),
        )
        safety_witness = benchmark.ClearanceWitness(
            value=safety,
            source_kind="static",
            obstacle_identifier=None,
            elapsed_s=0.0,
            sample_index=0,
            sample_fraction=0.0,
            robot_position=(50.0, 47.5),
        )
        return SweptTransitionSafety(
            collision=False,
            minimum_clearance=1.0,
            minimum_safety_clearance=safety,
            minimum_clearance_witness=physical_witness,
            minimum_safety_clearance_witness=safety_witness,
        )

    def fake_step(state, control, dt, config):
        nonlocal plant_calls
        del control, dt, config
        plant_calls += 1
        result = np.asarray(state, dtype=float).copy()
        if plant_calls == 2:
            result[:2] = np.array([130.0, 47.5])
        return result

    def fake_solve(self, method, state, obstacles, nominal, **kwargs):
        del self, state, obstacles, kwargs
        return BaselineDecision(
            method=str(benchmark.BenchmarkMethod(method).value),
            control=np.asarray(nominal, dtype=float),
            policy_id="nominal",
            feasible=False,
            status="diagnostic_infeasible",
            used_fallback=True,
            objective=0.0,
            solve_time_s=0.0,
        )

    monkeypatch.setattr(benchmark, "evaluate_swept_transition", fake_transition)
    monkeypatch.setattr(benchmark, "step_double_integrator", fake_step)
    monkeypatch.setattr(benchmark.HospitalBaselineSuite, "solve", fake_solve)
    monkeypatch.setattr(
        benchmark,
        "build_benchmark_scenario",
        lambda *args, **kwargs: simulation,
    )

    result = benchmark.run_hospital_trial(
        "mps",
        "blocked_2_stretchers",
        seed=0,
        steps=3,
        config=config,
        warmup=False,
        raise_errors=True,
    )

    assert result.outcome is BenchmarkOutcome.SUCCESS
    assert result.case_metrics["steps"] == 2
    assert result.case_metrics["operational_safety_violation"] is True
    assert result.case_metrics["infeasible_count"] == 2
    assert result.case_metrics["termination_reason"] == "goal_reached"


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
