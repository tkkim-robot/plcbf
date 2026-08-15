from __future__ import annotations

from dataclasses import replace
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import examples.hospital.benchmark as benchmark
import examples.hospital.baselines as hospital_baselines
from examples.hospital.baselines import (
    HospitalBaselineConfig,
    HospitalBaselineSuite,
)
from examples.hospital.dynamics import waypoint_control
from examples.hospital.jax_rollout import (
    HospitalJaxCapacities,
    evaluate_policy_batch,
    pack_obstacle_batch,
    pack_parameters,
    pack_policy_batch,
    pack_static_geometry,
)
from examples.hospital.obstacles import (
    Human,
    Stretcher,
    obstacle_clearance,
)
from examples.hospital.policies import HospitalPolicy, rollout_policy
from examples.hospital.simulation import build_blocked_main_hall_scenario
from plcbf.baselines import BenchmarkMethod
from plcbf.big_m_mpc import build_big_m_trajectory_milp
from plcbf.policy_library import CBFHalfspace, PolicyCertificate


def _nominal(simulation) -> np.ndarray:
    target = simulation.controller._navigation_target(simulation.state)
    return waypoint_control(
        simulation.state,
        target,
        simulation.config.robot,
        simulation.config.policies.nominal_target_speed,
    )


def _short_suite():
    simulation = build_blocked_main_hall_scenario(2)
    algorithm_config = replace(
        HospitalBaselineConfig(),
        backup_horizon_s=2.0 * simulation.config.dt,
        multi_backup_maneuver_s=simulation.config.dt,
        gatekeeper_nominal_steps=2,
    )
    suite = HospitalBaselineSuite(
        simulation.controller,
        simulation.environment,
        simulation.config,
        algorithm_config=algorithm_config,
    )
    return simulation, suite


def test_default_backup_horizon_matches_warehouse_rounding() -> None:
    simulation = build_blocked_main_hall_scenario(2)
    suite = HospitalBaselineSuite(
        simulation.controller,
        simulation.environment,
        simulation.config,
    )

    assert suite._backup_steps == round(
        HospitalBaselineConfig().backup_horizon_s / simulation.config.dt
    )
    assert suite._backup_steps == 67


@pytest.mark.parametrize(
    "elapsed",
    (0.0, 0.015, 0.05, 0.06, 0.1, 0.135, 0.3, 1.17, 4.0),
)
def test_decision_local_prediction_cache_matches_scalar_predictors(
    elapsed: float,
) -> None:
    simulation, suite = _short_suite()
    human = Human(
        "cache-human",
        x=31.0,
        y=47.0,
        vx=1.1,
        vy=-0.35,
    )
    stretcher = Stretcher(
        "cache-stretcher",
        coordinate=52.0,
        lateral=47.0,
        speed=-1.3,
        axis="x",
        route_min=22.0,
        route_max=63.0,
    )
    suite._obstacles = (human, stretcher)
    suite._reset_prediction_cache()

    for index, obstacle in enumerate(suite._obstacles):
        expected = obstacle.predicted(elapsed, simulation.environment)
        actual = suite._predicted_obstacle(index, elapsed)
        np.testing.assert_allclose(actual.center, expected.center, atol=1e-12)
        np.testing.assert_allclose(
            actual.velocity,
            expected.velocity,
            atol=1e-12,
        )


def test_prediction_cache_reuses_human_checkpoints_without_cross_step_state(
    monkeypatch,
) -> None:
    simulation, suite = _short_suite()
    human = Human(
        "cache-human",
        x=31.0,
        y=47.0,
        vx=1.1,
        vy=-0.35,
    )
    suite._obstacles = (human,)
    suite._reset_prediction_cache()
    calls = 0
    original = Human.predicted

    def counted(self, elapsed, environment):
        nonlocal calls
        calls += 1
        return original(self, elapsed, environment)

    monkeypatch.setattr(Human, "predicted", counted)
    suite._predicted_obstacle(0, 4.0)
    first_pass_calls = calls
    suite._predicted_obstacle(0, 4.0)

    assert calls == first_pass_calls
    assert first_pass_calls < 90

    suite._reset_prediction_cache()
    suite._predicted_obstacle(0, 4.0)
    assert calls > first_pass_calls


def test_batched_human_advance_matches_every_scalar_collision_branch() -> None:
    simulation, suite = _short_suite()
    positions = np.asarray(
        [[10.0 * index, 1.0 + 10.0 * index] for index in range(5)]
    )
    velocities = np.asarray(
        [[1.0 + 0.1 * index, 0.5 + 0.1 * index] for index in range(5)]
    )
    elapsed = 0.2
    # full-free; x-only; y-only; both axial moves; neither axial move.
    branch_collisions = (
        (False, False, False),
        (True, False, True),
        (True, True, False),
        (True, False, False),
        (True, True, True),
    )
    collision_map: dict[tuple[float, float], bool] = {}
    for position, velocity, (full, x_only, y_only) in zip(
        positions, velocities, branch_collisions, strict=True
    ):
        collision_map[tuple(position + velocity * elapsed)] = full
        collision_map[
            (position[0] + velocity[0] * elapsed, position[1])
        ] = x_only
        collision_map[
            (position[0], position[1] + velocity[1] * elapsed)
        ] = y_only

    class BranchEnvironment:
        def is_collision(self, point, radius=0.0):
            del radius
            return collision_map[tuple(np.asarray(point, dtype=float))]

        def collisions(self, points, radii=0.0):
            del radii
            return np.asarray(
                [self.is_collision(point) for point in points],
                dtype=bool,
            )

    branch_environment = BranchEnvironment()
    suite.environment = branch_environment
    suite._batch_human_radii = np.full(len(positions), 0.52)
    actual_positions, actual_velocities = suite._advance_human_batch(
        positions,
        velocities,
        elapsed,
    )

    expected_humans = [
        Human(
            f"branch-{index}",
            x=float(position[0]),
            y=float(position[1]),
            vx=float(velocity[0]),
            vy=float(velocity[1]),
        )
        for index, (position, velocity) in enumerate(
            zip(positions, velocities, strict=True)
        )
    ]
    for human in expected_humans:
        human.advance(elapsed, branch_environment)
    np.testing.assert_array_equal(
        actual_positions,
        [human.center for human in expected_humans],
    )
    np.testing.assert_array_equal(
        actual_velocities,
        [human.velocity for human in expected_humans],
    )


@pytest.mark.parametrize(
    "elapsed",
    (
        0.0,
        0.015,
        0.04999999999,
        0.05,
        0.0500000002,
        0.06,
        0.135,
        0.1499999999,
        1.17,
        3.99,
    ),
)
def test_batched_human_and_stretcher_geometry_matches_scalar_prediction(
    elapsed: float,
) -> None:
    simulation, suite = _short_suite()
    obstacles = (
        Human(
            "batch-human-a",
            x=31.0,
            y=44.4,
            vx=1.1,
            vy=-0.75,
            radius=0.48,
        ),
        Stretcher(
            "batch-stretcher-x",
            coordinate=52.0,
            lateral=47.0,
            speed=-1.3,
            axis="x",
            route_min=22.0,
            route_max=63.0,
        ),
        Human(
            "batch-human-b",
            x=68.0,
            y=47.0,
            vx=-0.35,
            vy=0.55,
            radius=0.57,
        ),
        Stretcher(
            "batch-stretcher-y",
            coordinate=47.0,
            lateral=68.0,
            speed=1.05,
            axis="y",
            route_min=30.0,
            route_max=65.0,
            length=5.2,
            width=1.7,
        ),
    )
    suite._obstacles = obstacles
    suite._reset_prediction_cache()

    geometry = suite._predicted_geometry(elapsed)
    expected_humans = [
        obstacle.predicted(elapsed, simulation.environment)
        for obstacle in obstacles
        if isinstance(obstacle, Human)
    ]
    expected_stretchers = [
        obstacle.predicted(elapsed, simulation.environment)
        for obstacle in obstacles
        if isinstance(obstacle, Stretcher)
    ]
    np.testing.assert_allclose(
        geometry.human_centers,
        [obstacle.center for obstacle in expected_humans],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        geometry.human_radii,
        [obstacle.radius for obstacle in expected_humans],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        geometry.stretcher_centers,
        [obstacle.center for obstacle in expected_stretchers],
        rtol=0.0,
        atol=1e-12,
    )


def test_batched_point_margins_match_scalar_reference_at_swept_samples() -> None:
    simulation, suite = _short_suite()
    suite._obstacles = (
        Human(
            "swept-human-a",
            x=31.0,
            y=44.4,
            vx=1.1,
            vy=-0.75,
            radius=0.48,
        ),
        Human(
            "swept-human-b",
            x=68.0,
            y=47.0,
            vx=-0.35,
            vy=0.55,
            radius=0.57,
        ),
        Stretcher(
            "swept-stretcher-x",
            coordinate=52.0,
            lateral=47.0,
            speed=-1.3,
            axis="x",
            route_min=22.0,
            route_max=63.0,
        ),
        Stretcher(
            "swept-stretcher-y",
            coordinate=47.0,
            lateral=68.0,
            speed=1.05,
            axis="y",
            route_min=30.0,
            route_max=65.0,
            length=5.2,
            width=1.7,
        ),
    )
    suite._reset_prediction_cache()
    trajectory = np.array(
        [
            [29.0, 47.3, 1.0, 0.2],
            [29.06, 47.312, 1.0, 0.2],
            [29.12, 47.324, 1.0, 0.2],
            [29.18, 47.336, 1.0, 0.2],
        ]
    )
    substeps = suite.algorithm_config.trajectory_swept_substeps
    alphas = np.arange(substeps + 1, dtype=float) / substeps
    starts = trajectory[:-1, :2]
    ends = trajectory[1:, :2]
    swept_points = (
        starts[:, None, :]
        + alphas[None, :, None] * (ends - starts)[:, None, :]
    ).reshape(-1, 2)
    swept_times = (
        np.arange(len(starts), dtype=float)[:, None]
        + alphas[None, :]
    ).reshape(-1) * simulation.config.dt
    robot_radius = (
        simulation.config.robot.radius
        + simulation.config.safety.safety_margin
    )

    expected = []
    for point, elapsed in zip(
        swept_points, swept_times, strict=True
    ):
        values = [
            simulation.environment.static_clearance(
                point,
                simulation.config.robot.radius
                + simulation.config.safety.static_margin,
            )
        ]
        values.extend(
            obstacle_clearance(
                obstacle.predicted(elapsed, simulation.environment),
                point,
                robot_radius,
                simulation.config.safety.human_margin,
                simulation.config.safety.stretcher_margin,
            )
            for obstacle in suite._obstacles
        )
        expected.append(min(values))

    np.testing.assert_allclose(
        suite._point_margins(swept_points, swept_times),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_near_zero_vector_margin_uses_scalar_sign_fidelity(
    monkeypatch,
) -> None:
    simulation, suite = _short_suite()
    point = np.array([67.0, 47.5])
    combined_radius = (
        simulation.config.robot.radius
        + simulation.config.safety.safety_margin
        + 0.52
        + simulation.config.safety.human_margin
    )
    suite._obstacles = (
        Human(
            "boundary-human",
            x=float(point[0] + combined_radius),
            y=float(point[1]),
            vx=0.0,
            vy=0.0,
            radius=0.52,
        ),
    )
    suite._reset_prediction_cache()
    original = suite._scalar_point_margin
    scalar_calls = 0

    def counted(point_value, elapsed):
        nonlocal scalar_calls
        scalar_calls += 1
        return original(point_value, elapsed)

    monkeypatch.setattr(suite, "_scalar_point_margin", counted)
    offsets = np.asarray([-5e-13, 0.0, 5e-13])
    points = point[None, :] + np.column_stack(
        (offsets, np.zeros_like(offsets))
    )
    actual = suite._point_margins(points, np.zeros(len(points)))
    expected = np.asarray(
        [original(sample, 0.0) for sample in points]
    )

    assert scalar_calls == len(points)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "case_id,expected_obstacle_count",
    (
        ("blocked_2_stretchers", 67),
        ("blocked_3_stretchers", 68),
    ),
)
def test_crowded_seed_batch_margins_match_scalar_decision_signs(
    case_id: str,
    expected_obstacle_count: int,
) -> None:
    simulation = benchmark.build_benchmark_scenario(
        case_id,
        seed=7,
    )
    suite = HospitalBaselineSuite(
        simulation.controller,
        simulation.environment,
        simulation.config,
    )
    suite._obstacles = tuple(simulation.obstacles)
    suite._reset_prediction_cache()
    points = np.column_stack(
        (
            np.linspace(simulation.state[0], 78.0, 11),
            np.linspace(simulation.state[1], 47.5, 11),
        )
    )
    elapsed = np.linspace(0.0, 3.96, len(points))

    expected = np.asarray(
        [
            suite._scalar_point_margin(point, time_value)
            for point, time_value in zip(
                points, elapsed, strict=True
            )
        ]
    )
    actual = suite._point_margins(points, elapsed)

    assert len(simulation.obstacles) == expected_obstacle_count
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_fixed_warehouse_baselines_share_one_nonroom_retrace_backup() -> None:
    simulation, suite = _short_suite()
    retrace = suite.fixed_backup_policy(simulation.state)

    assert retrace.name == "retrace_waypoint"
    assert retrace.kind == "retrace"
    assert retrace.target_room is None
    assert (
        suite.algorithm_config.fixed_backup_policy_id
        == "retrace_waypoint"
    )
    assert suite.mps.backup_policy_id == "retrace_waypoint"
    assert suite.gatekeeper.backup_policy_id == "retrace_waypoint"
    assert all(
        policy.name != "retrace_waypoint"
        for policy in simulation.controller.candidate_policies(
            simulation.state
        )
    )
    assert not any(
        "room" in name or "refuge" in name
        for name in vars(suite)
    )


def test_retrace_waypoint_progress_is_monotone_and_rollout_local() -> None:
    simulation, suite = _short_suite()
    policy = HospitalPolicy(
        name="synthetic_retrace",
        kind="retrace",
        horizon=8.0,
        rollout_dt=simulation.config.dt,
        target_speed=2.8,
        waypoints=[
            np.array([20.0, 0.0]),
            np.array([10.0, 0.0]),
            np.array([0.0, 0.0]),
        ],
        feedback_gain=suite.algorithm_config.fixed_backup_gain,
    )
    state = np.array([20.0, 0.0, 0.0, 0.0])

    _, cursor = policy.control_with_cursor(state, simulation.config, 0)
    assert cursor == 1
    _, cursor = policy.control_with_cursor(
        np.array([18.0, 0.0, -1.0, 0.0]),
        simulation.config,
        cursor,
    )
    assert cursor == 1
    _, cursor = policy.control_with_cursor(
        np.array([10.0, 0.0, -1.0, 0.0]),
        simulation.config,
        cursor,
    )
    assert cursor == 2

    first = rollout_policy(policy, state, simulation.config)
    second = rollout_policy(policy, state, simulation.config)
    np.testing.assert_allclose(first, second)
    assert first[-1, 0] < 1.0
    assert np.all(np.diff(first[:, 0]) <= 1e-12)


def test_multibackup_and_mi_room_rollouts_match_scalar_and_jax_cursor() -> None:
    """Every hypothetical room branch gets an independent monotone cursor."""

    simulation = build_blocked_main_hall_scenario(2)
    horizon = 3.0
    suite = HospitalBaselineSuite(
        simulation.controller,
        simulation.environment,
        simulation.config,
        algorithm_config=replace(
            HospitalBaselineConfig(),
            backup_horizon_s=horizon,
            multi_backup_maneuver_s=horizon - simulation.config.dt,
        ),
    )
    state = np.array([50.0, 47.5, 0.0, 0.0])
    policy = HospitalPolicy(
        name="synthetic_room_cursor",
        kind="room",
        horizon=horizon,
        rollout_dt=simulation.config.dt,
        target_speed=simulation.config.robot.v_max,
        waypoints=[
            state[:2].copy(),
            np.array([52.0, 47.5]),
            np.array([55.0, 47.5]),
        ],
    )

    scalar = rollout_policy(policy, state, simulation.config)
    multibackup = suite._backup_rollout_states(
        policy,
        state,
        maneuver_steps=suite._backup_steps,
        multi_backup=True,
    )
    # Branch safety is irrelevant to this feedback-realization regression.
    suite._point_margins = lambda points, elapsed: np.ones(len(points))
    mi_branch, _, _ = suite._branch_rollout(state, policy)

    capacity = HospitalJaxCapacities(
        max_policies=1,
        max_obstacles=1,
        max_horizon_steps=suite._backup_steps,
        max_swept_samples=3,
        human_prediction_steps=64,
    )
    packed_policy = pack_policy_batch(
        (policy,), simulation.config, capacity
    )
    evaluated = evaluate_policy_batch(
        state,
        np.zeros(2),
        packed_policy,
        pack_obstacle_batch((), capacity),
        pack_static_geometry(simulation.environment),
        pack_parameters(simulation.config),
    )
    jax_trajectory = evaluated.trajectories[
        0, evaluated.trajectory_mask[0]
    ]

    np.testing.assert_allclose(multibackup, scalar, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(mi_branch, scalar, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        jax_trajectory,
        scalar,
        rtol=2.0e-6,
        atol=8.0e-6,
    )
    assert scalar[-1, 0] > 54.0
    # Repeating either baseline rollout starts a fresh hypothetical cursor;
    # no waypoint progress survives as controller or policy state.
    np.testing.assert_allclose(
        suite._backup_rollout_states(
            policy,
            state,
            maneuver_steps=suite._backup_steps,
            multi_backup=True,
        ),
        multibackup,
        rtol=0.0,
        atol=1e-12,
    )
    repeated_mi, _, _ = suite._branch_rollout(state, policy)
    np.testing.assert_allclose(repeated_mi, mi_branch, rtol=0.0, atol=1e-12)


def test_mps_and_gatekeeper_reset_retrace_before_each_candidate() -> None:
    simulation = build_blocked_main_hall_scenario(2)
    simulation.controller.navigation_path = [
        np.array([0.0, 0.0]),
        np.array([1.5, 0.0]),
        np.array([3.0, 0.0]),
        np.array([4.5, 0.0]),
    ]
    simulation.controller.navigation_index = 3
    algorithm_config = replace(
        HospitalBaselineConfig(),
        backup_horizon_s=2.0,
        multi_backup_maneuver_s=1.0,
        gatekeeper_nominal_steps=2,
    )
    suite = HospitalBaselineSuite(
        simulation.controller,
        simulation.environment,
        simulation.config,
        algorithm_config=algorithm_config,
    )
    state = np.array([3.0, 0.0, 0.0, 0.0])

    for shield in (suite.mps, suite.gatekeeper):
        first = shield._candidate(state, 0)
        suite._retrace_rollout_index = 2
        repeated = shield._candidate(state, 0)
        np.testing.assert_allclose(first.states, repeated.states)
        np.testing.assert_allclose(first.controls, repeated.controls)
        assert suite._retrace_rollout_index > 0


def test_pcbf_backup_cbf_mps_and_gatekeeper_execute_retrace_only() -> None:
    simulation, suite = _short_suite()
    state = simulation.state.copy()
    nominal = _nominal(simulation)
    retrace = suite.fixed_backup_policy(state)
    certificates, _ = simulation.controller.build_policy_certificates(
        state,
        (),
        (retrace,),
        nominal_control=nominal,
    )

    pcbf = suite.solve(
        BenchmarkMethod.POLICY_PCBF,
        state,
        (),
        nominal,
        certificates=certificates,
    )
    backup_cbf = suite.solve(
        BenchmarkMethod.BACKUP_CBF,
        state,
        (),
        nominal,
    )

    _, mps_suite = _short_suite()
    mps = mps_suite.solve(
        BenchmarkMethod.MPS,
        state,
        (),
        nominal,
    )
    _, gatekeeper_suite = _short_suite()
    gatekeeper = gatekeeper_suite.solve(
        BenchmarkMethod.GATEKEEPER,
        state,
        (),
        nominal,
    )

    decisions = (pcbf, backup_cbf, mps, gatekeeper)
    assert {decision.policy_id for decision in decisions} == {
        "retrace_waypoint"
    }
    assert all(
        decision.policy_id is not None
        and not decision.policy_id.startswith("room")
        for decision in decisions
    )
    assert mps_suite.mps.committed is not None
    assert (
        mps_suite.mps.committed.backup_policy_id
        == gatekeeper_suite.gatekeeper.backup_policy_id
        == "retrace_waypoint"
    )
    assert gatekeeper_suite.gatekeeper.committed is not None
    assert (
        gatekeeper_suite.gatekeeper.committed.backup_policy_id
        == "retrace_waypoint"
    )


def test_backup_cbf_keeps_current_margin_but_rows_start_after_it(
    monkeypatch,
) -> None:
    simulation, suite = _short_suite()
    state = simulation.state.copy()
    policy = suite.fixed_backup_policy(state)
    monkeypatch.setattr(
        suite,
        "_point_margin",
        lambda point, elapsed: (
            100.0 + float(point[0]) + float(elapsed)
        ),
    )
    oracle = suite._policy_rollout_margins(
        policy,
        maneuver_steps=suite._backup_steps,
        multi_backup=False,
    )

    path_values, terminal_value = oracle(state, 0.0)

    assert path_values.shape == (suite._backup_steps,)
    assert path_values[0] == pytest.approx(100.0 + state[0])

    candidate = suite._backup_candidate(
        policy,
        state,
        _nominal(simulation),
        maneuver_steps=suite._backup_steps,
    )
    path_rows = [
        row for row in candidate.halfspaces if ":path[" in row.label
    ]
    terminal_rows = [
        row for row in candidate.halfspaces if row.label.endswith(":terminal")
    ]
    assert candidate.path_values.shape == (suite._backup_steps,)
    assert candidate.path_values[0] == pytest.approx(path_values[0])
    assert len(path_rows) == suite._backup_steps - 1
    assert [row.label for row in path_rows] == [
        f"retrace_waypoint:path[{index}]"
        for index in range(1, suite._backup_steps)
    ]
    assert len(terminal_rows) <= 1
    assert candidate.terminal_value == pytest.approx(terminal_value)


def test_multi_backup_cbf_keeps_initial_through_horizon_path(
    monkeypatch,
) -> None:
    simulation, suite = _short_suite()
    state = simulation.state.copy()
    policy = suite.fixed_backup_policy(state)
    suite._point_margin = (
        lambda point, elapsed: (
            100.0 + float(point[0]) + float(elapsed)
        )
    )
    oracle = suite._policy_rollout_margins(
        policy,
        maneuver_steps=1,
        multi_backup=True,
    )
    path_values, _ = oracle(state, 0.0)
    expected_state_count = (
        int(
            np.ceil(
                suite.algorithm_config.backup_horizon_s
                / simulation.config.dt
            )
        )
        + 1
    )

    assert path_values.shape == (expected_state_count,)
    assert path_values[0] == pytest.approx(100.0 + state[0])
    trajectory = suite._backup_rollout_states(
        policy,
        state,
        maneuver_steps=1,
        multi_backup=True,
    )
    assert path_values[-1] == pytest.approx(
        100.0
        + trajectory[-1, 0]
        + (expected_state_count - 1) * simulation.config.dt
    )

    captured_flow_derivatives = []
    original_evaluator = hospital_baselines.evaluate_backup_cbf_candidate

    def capture_evaluator(**kwargs):
        captured_flow_derivatives.append(
            np.asarray(kwargs["path_flow_derivatives"]).copy()
        )
        return original_evaluator(**kwargs)

    monkeypatch.setattr(
        hospital_baselines,
        "evaluate_backup_cbf_candidate",
        capture_evaluator,
    )
    candidate = suite._backup_candidate(
        policy,
        state,
        _nominal(simulation),
        maneuver_steps=1,
        multi_backup=True,
    )
    path_rows = [
        row for row in candidate.halfspaces if ":path[" in row.label
    ]
    assert candidate.path_values.shape == (expected_state_count,)
    assert path_rows
    assert all(
        0 <= int(row.label.rsplit("[", 1)[1][:-1]) < expected_state_count
        for row in path_rows
    )
    assert sum(
        row.label == "retrace_waypoint:terminal"
        for row in candidate.halfspaces
    ) <= 1
    assert len(captured_flow_derivatives) == 1
    assert captured_flow_derivatives[0].shape == (expected_state_count,)
    np.testing.assert_allclose(
        captured_flow_derivatives[0],
        suite._backup_path_flow_derivatives(
            policy,
            state,
            maneuver_steps=1,
            multi_backup=True,
        ),
    )


def test_backup_terminal_margins_use_fixed_T_and_T_plus_dt() -> None:
    simulation = build_blocked_main_hall_scenario(2)
    horizon = 2.5 * simulation.config.dt
    algorithm_config = replace(
        HospitalBaselineConfig(),
        backup_horizon_s=horizon,
        multi_backup_maneuver_s=0.0,
        terminal_speed_mps=1.0,
    )
    suite = HospitalBaselineSuite(
        simulation.controller,
        simulation.environment,
        simulation.config,
        algorithm_config=algorithm_config,
    )
    suite._point_margin = (
        lambda point, elapsed: float(elapsed)
    )
    stop = next(
        policy
        for policy in simulation.controller.candidate_policies(
            simulation.state
        )
        if policy.kind == "stop"
    )

    single_path, single_terminal = suite._policy_rollout_margins(
        stop,
        maneuver_steps=0,
        multi_backup=False,
    )(simulation.state, 0.0)
    multi_path, multi_terminal = suite._policy_rollout_margins(
        stop,
        maneuver_steps=0,
        multi_backup=True,
    )(simulation.state, 0.0)

    np.testing.assert_allclose(single_path, [0.0, simulation.config.dt])
    np.testing.assert_allclose(
        multi_path,
        [
            0.0,
            simulation.config.dt,
            2.0 * simulation.config.dt,
            3.0 * simulation.config.dt,
        ],
    )
    assert single_terminal == pytest.approx(horizon)
    assert multi_terminal == pytest.approx(horizon)


def test_hospital_pcbf_and_library_fallbacks_match_warehouse() -> None:
    simulation, suite = _short_suite()
    state = simulation.state.copy()
    nominal = _nominal(simulation)
    impossible = (
        CBFHalfspace(np.zeros(2), 1.0, "impossible"),
    )
    retrace = PolicyCertificate(
        "retrace_waypoint",
        value=1.0,
        halfspaces=impossible,
        backup_control=np.array([-1.0, 1.0]),
    )
    room = PolicyCertificate(
        "room_0",
        value=2.0,
        halfspaces=impossible,
        backup_control=np.array([1.0, 1.0]),
    )
    stop_control = suite._terminal_stop_feedback(state)
    stop = PolicyCertificate(
        "stop",
        value=1.0,
        halfspaces=impossible,
        backup_control=stop_control,
    )

    pcbf = suite.solve(
        BenchmarkMethod.POLICY_PCBF,
        state,
        (),
        nominal,
        certificates=(retrace,),
    )
    library = suite.solve(
        BenchmarkMethod.LIBRARY_PCBF_MI,
        state,
        (),
        nominal,
        certificates=(room, stop),
    )

    assert pcbf.used_fallback is True
    np.testing.assert_allclose(pcbf.control, nominal)
    assert library.used_fallback is True
    np.testing.assert_allclose(library.control, stop_control)
    assert (
        library.policy_decision.diagnostics.fallback_source
        == "emergency_policy:stop"
    )


def test_benchmark_executes_method_control_without_cache_or_shared_projection() -> None:
    source = inspect.getsource(benchmark.run_hospital_trial)
    module_source = inspect.getsource(benchmark)

    assert "_OracleCache" not in module_source
    assert "phase_key" not in module_source
    assert "solve_control_halfspaces" not in source
    assert "current_hocbf_constraints" not in source
    assert "control = np.asarray(decision.control" in source
    assert "refresh_period = float(config.dt)" in source


def test_mi_mpc_has_directional_only_branches_and_full_x_u_milp() -> None:
    simulation, suite = _short_suite()
    policies = suite.mi_mpc_policies(simulation.state)

    assert len(policies) == 32
    assert all(policy.kind == "angle" for policy in policies)
    assert all(policy.target_room is None for policy in policies)
    assert all(policy.name.startswith("mi_angle_") for policy in policies)
    assert not {
        "nominal",
        "stop",
        "room",
        "retrace",
    } & {policy.kind for policy in policies}

    nominal = _nominal(simulation)
    problem, fallback_index = suite.build_mi_mpc_problem(
        simulation.state,
        nominal,
        policies,
    )
    branch_control = policies[fallback_index].control(
        simulation.state,
        simulation.config,
    )
    expected_fallback = np.clip(
        0.75 * branch_control + 0.25 * nominal,
        -simulation.config.robot.a_max,
        simulation.config.robot.a_max,
    )
    np.testing.assert_allclose(problem.fallback_control, expected_fallback)
    assert not np.allclose(problem.fallback_control, branch_control)
    mi_config = suite._mi_mpc_config()
    model = build_big_m_trajectory_milp(problem, mi_config)

    assert suite._backup_steps == 2
    assert model.layout.state.shape == (suite._backup_steps + 1, 4)
    assert model.layout.control.shape == (suite._backup_steps, 2)
    assert model.layout.selector.shape == (len(policies),)
    assert np.all(model.integrality[model.layout.selector] == 1)
    continuous = np.r_[
        model.layout.state.reshape(-1),
        model.layout.control.reshape(-1),
    ]
    assert np.all(model.integrality[continuous] == 0)
    assert {
        "dynamics",
        "one_hot",
        "position_tubes",
        "control_tubes",
    } <= set(model.row_groups)
    assert mi_config.position_tube == 3.0
    assert mi_config.early_control_tube == 6.0
    assert mi_config.early_control_steps == 2
    assert mi_config.tracking_weight == 8.0
    assert mi_config.terminal_weight == 16.0
    assert mi_config.velocity_weight == 0.15
    assert mi_config.control_weight == 0.02
    assert mi_config.nominal_weight == 0.5
    assert mi_config.time_limit_s == 1.0
    assert mi_config.mip_rel_gap == 0.05
    np.testing.assert_allclose(model.position_big_m, 400.0)
    np.testing.assert_allclose(model.control_big_m, 60.0)
    np.testing.assert_allclose(model.safety_big_m, 50.0)


def test_plcbf_numerical_fallback_does_not_erase_hocbf_feasibility(
    monkeypatch,
) -> None:
    simulation, suite = _short_suite()
    nominal = _nominal(simulation)
    policy_decision = SimpleNamespace(
        diagnostics=SimpleNamespace(
            used_fallback=True,
            fallback_reason="numerical_solver_failure",
        )
    )
    monkeypatch.setattr(
        suite.controller,
        "compute",
        lambda state, obstacles, time, **kwargs: SimpleNamespace(
            control=np.array([-0.3, 0.1]),
            selected_policy="room_0",
            feasible=True,
            decision=policy_decision,
        ),
    )

    decision = suite.solve(
        BenchmarkMethod.PLCBF,
        simulation.state,
        (),
        nominal,
    )

    assert decision.used_fallback is True
    assert decision.feasible is True
    assert decision.status == "fallback:numerical_solver_failure"
