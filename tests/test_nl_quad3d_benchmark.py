from __future__ import annotations

import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from examples.nl_quad3d import benchmark
from examples.nl_quad3d.baselines import NLQuad3DBaselineSuite
from examples.nl_quad3d.controller import (
    NLQuad3DControllerConfig,
    PLCBF_NLQuad3D,
)
from examples.nl_quad3d.dynamics import NLQuad3D
from examples.nl_quad3d.policies import PolicyCandidate
from examples.nl_quad3d.scenarios import (
    NLQuad3DScenario,
    PLAYGROUND_CROWDED_SCENARIO,
    PLAYGROUND_OBSTACLE_COUNT,
    PLAYGROUND_OBSTACLE_RADIUS,
    PLAYGROUND_OBSTACLE_SPEED_MAX,
    PLAYGROUND_OBSTACLE_SPEED_MIN,
    PLAYGROUND_REFERENCE_OBSTACLE_COUNT,
    PLAYGROUND_START_GOAL_PROTECTION,
    PLAYGROUND_STRESS_CORRIDOR_LOWER,
    PLAYGROUND_STRESS_CORRIDOR_UPPER,
    PLAYGROUND_STRESS_CROSS_FLOW_COUNT,
    PLAYGROUND_STRESS_OBSTACLE_COUNT,
    PLAYGROUND_STRESS_OBSTACLE_SPEED_MAX,
    PLAYGROUND_STRESS_OBSTACLE_SPEED_MIN,
    PLAYGROUND_STRESS_PAIR_CLEARANCE,
    PLAYGROUND_STRESS_PROTOCOL_VERSION,
    PLAYGROUND_STRESS_SCENARIO,
    PLAYGROUND_STRESS_STREAM_DIRECTIONS,
    WorldBounds,
    get_scenario,
    make_playground_crowded_scenario,
    make_playground_stress_scenario,
)
from plcbf.baselines import (
    BENCHMARK_METHODS,
    BaselineDecision,
    solve_policy_pcbf,
)
from plcbf.backup_cbf import (
    BackupCbfRolloutDerivatives,
    _rollout_derivatives,
    evaluate_backup_cbf_candidate,
    solve_fixed_backup_cbf,
    solve_multi_backup_cbf_min_intervention,
)
from plcbf.benchmarking import (
    BenchmarkOutcome,
    BenchmarkReportPaths,
    BenchmarkResult,
)
from plcbf.policy_library import CBFHalfspace, PolicyCertificate


class FakeModel:
    dt = 0.1

    def __init__(self, *, step_distance: float = 0.25) -> None:
        self.config = SimpleNamespace(
            robot_radius=0.1,
            attitude_bound=np.deg2rad(30.0),
            body_rate_max=6.0,
            nominal_yaw_slew_max=2.0,
            a_max_xy=4.0,
            a_max_z=4.0,
            v_max=3.5,
        )
        self.input_lower_bound = np.full(4, -1.0)
        self.input_upper_bound = np.full(4, 1.0)
        self.step_distance = step_distance

    def nominal_input(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        del state, goal
        return np.full(4, 0.5)

    def saturate_rotors(self, control: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(control, dtype=float), -1.0, 1.0)

    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        del control
        result = np.asarray(state, dtype=float).copy()
        result[0] += self.step_distance
        return result

    def safety_point(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float)[:3].copy()

    def stop_input(self, state: np.ndarray, gain: float = 3.0) -> np.ndarray:
        del state, gain
        return np.zeros(4)

    def acceleration_to_rotors(
        self,
        state: np.ndarray,
        desired_acceleration: np.ndarray,
    ) -> np.ndarray:
        del state, desired_acceleration
        return np.zeros(4)

    def f(self, state: np.ndarray) -> np.ndarray:
        del state
        return np.zeros(12)

    def g(self, state: np.ndarray) -> np.ndarray:
        del state
        return np.zeros((12, 4))


class StateInjectingModel(FakeModel):
    def __init__(
        self,
        *,
        phi: float = 0.0,
        theta: float = 0.0,
        yaw_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.phi = phi
        self.theta = theta
        self.yaw_rate = yaw_rate

    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        result = super().step(state, control)
        result[6] = self.phi
        result[7] = self.theta
        result[11] = self.yaw_rate
        return result


def _certificate(
    policy_id: str,
    *,
    value: float = 1.0,
    backup: float = 0.5,
    offset: float = -1.0,
    nominal_prefix_safe: int | None = None,
) -> PolicyCertificate:
    prefix_safe = (
        int(value >= 0.0)
        if nominal_prefix_safe is None
        else int(nominal_prefix_safe)
    )
    return PolicyCertificate(
        policy_id=policy_id,
        value=value,
        halfspaces=(CBFHalfspace(np.zeros(4), offset),),
        backup_control=np.full(4, backup),
        metadata={
            "rollout_safe": value >= 0.0,
            "terminal_safe": value >= 0.0,
            "nominal_prefix_safe": prefix_safe,
            "terminal_cost": 0.0 if policy_id == "nominal" else 1.0,
        },
    )


class FakeController:
    def __init__(
        self,
        model: FakeModel,
        config: NLQuad3DControllerConfig,
        bounds: object | None,
    ) -> None:
        del bounds
        self.model = model
        self.config = config
        self.calls = 0
        self.last_certificates: tuple[PolicyCertificate, ...] = ()
        self.last_decision = None
        self.last_status = "uninitialized"

    def policy_certificates(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
    ) -> tuple[PolicyCertificate, ...]:
        del state, goal, obstacles
        self.calls += 1
        return (
            _certificate("stop", backup=0.0),
            _certificate("nominal", backup=0.5),
        )

    def _active_obstacles(self, state, obstacles):
        del state
        return np.asarray(obstacles, dtype=float).reshape(-1, 7).copy()

    def candidates(self, goal):
        del goal
        return (
            PolicyCandidate(
                "radial_0",
                "radial",
                direction=(1.0, 0.0, 0.0),
                target_speed=1.0,
            ),
            PolicyCandidate("stop", "stop"),
            PolicyCandidate("nominal", "nominal"),
        )

    def _candidate_control(self, state, goal, candidate):
        del state, goal
        if candidate.kind == "nominal":
            return np.full(4, 0.5)
        return np.zeros(4)

    def solve_control_problem(self, state, goal, obstacles, control_ref=None):
        certificates = self.policy_certificates(state, goal, obstacles)
        self.last_certificates = certificates
        selected = max(certificates, key=lambda item: item.value)
        used_fallback = selected.value < 0.0
        control = (
            selected.backup_control
            if used_fallback
            else np.asarray(control_ref, dtype=float)
        )
        self.last_decision = SimpleNamespace(
            policy_id=selected.policy_id,
            diagnostics=SimpleNamespace(used_fallback=used_fallback),
        )
        self.last_status = "fallback" if used_fallback else "filtered"
        return np.asarray(control, dtype=float)


class SwitchingController(FakeController):
    def policy_certificates(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
    ) -> tuple[PolicyCertificate, ...]:
        del state, goal, obstacles
        self.calls += 1
        return (_certificate("first" if self.calls == 1 else "second"),)


class UnsafeController(FakeController):
    def policy_certificates(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
    ) -> tuple[PolicyCertificate, ...]:
        del state, goal, obstacles
        return (_certificate("unsafe", value=-1.0, backup=0.0),)


class ShieldController(FakeController):
    def policy_certificates(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
    ) -> tuple[PolicyCertificate, ...]:
        del state, goal, obstacles
        return (
            _certificate(
                "safe_backup",
                value=1.0,
                backup=0.0,
                nominal_prefix_safe=0,
            ),
        )


class FailingController(FakeController):
    def solve_control_problem(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
        control_ref=None,
    ) -> np.ndarray:
        del state, goal, obstacles, control_ref
        raise RuntimeError("oracle failed")


class DeterministicClock:
    def __init__(self, increment: float = 0.001) -> None:
        self.value = 0.0
        self.increment = increment

    def __call__(self) -> float:
        self.value += self.increment
        return self.value


def _scenario(
    *,
    goal_x: float = 0.5,
    obstacles: np.ndarray | None = None,
    reach_threshold: float = 1e-9,
) -> NLQuad3DScenario:
    return NLQuad3DScenario(
        name="synthetic",
        waypoints=np.array([[0.0, 0.0, 0.0], [goal_x, 0.0, 0.0]]),
        obstacles=(
            np.zeros((0, 7), dtype=float)
            if obstacles is None
            else obstacles
        ),
        bounds=None,
        reach_threshold=reach_threshold,
        default_steps=2,
    )


def _controller_factory(controller_type=FakeController):
    def factory(model, config, bounds):
        return controller_type(model, config, bounds)

    return factory


def test_default_config_contains_the_common_eight_methods() -> None:
    config = benchmark.NLQuad3DBenchmarkConfig(max_steps=1)

    assert config.methods == BENCHMARK_METHODS
    assert len(config.methods) == 8
    assert config.scenarios == benchmark.DEFAULT_BENCHMARK_SCENARIOS
    assert config.metadata()["dynamics"] == "nonlinear_12_state_quadrotor"
    assert "physical spheres" in config.metadata()["collision_geometry"]
    assert "without extra rho_z inflation" in config.metadata()["collision_geometry"]
    assert "not robot walls" in config.metadata()["collision_geometry"]
    assert config.metadata()["state_validity"]["tilt_max_deg"] == pytest.approx(
        60.0
    )
    assert "run_experiment.py" in (
        config.metadata()["state_validity"]["tilt_termination_reference"]
    )
    assert "Deprecated alias" in (
        config.metadata()["decision_metrics"]["fallback_count"]
    )
    headline = config.metadata()["headline_scenario"]
    assert headline["name"] == PLAYGROUND_CROWDED_SCENARIO
    assert headline["obstacle_count"] == PLAYGROUND_OBSTACLE_COUNT
    assert headline["reference_obstacle_count"] == 5
    assert headline["random_obstacle_count"] == 27
    assert headline["initial_velocity_mps"] == [1.0, 0.0, 0.0]
    stress = config.metadata()["stress_scenario"]
    assert stress["name"] == PLAYGROUND_STRESS_SCENARIO
    assert stress["protocol_version"] == PLAYGROUND_STRESS_PROTOCOL_VERSION
    assert stress["obstacle_count"] == PLAYGROUND_STRESS_OBSTACLE_COUNT
    assert (
        stress["structured_cross_flow_count"]
        == PLAYGROUND_STRESS_CROSS_FLOW_COUNT
    )
    assert stress["structured_stream_count"] == (
        PLAYGROUND_STRESS_CROSS_FLOW_COUNT
    )
    assert stress["corridor_random_count"] == 24
    assert stress["cross_flow_directions"] == list(
        PLAYGROUND_STRESS_STREAM_DIRECTIONS
    )
    assert stress["structured_stream_directions"] == list(
        PLAYGROUND_STRESS_STREAM_DIRECTIONS
    )
    assert stress["minimum_initial_pair_surface_clearance_m"] == (
        PLAYGROUND_STRESS_PAIR_CLEARANCE
    )
    assert "may overlap or pass through" in stress["obstacle_interaction"]


def test_nl_baseline_libraries_match_warehouse_roles() -> None:
    model = FakeModel()
    controller = FakeController(
        model,
        NLQuad3DControllerConfig(dt=model.dt),
        None,
    )
    suite = NLQuad3DBaselineSuite(
        controller,
        model,
        None,
        waypoints=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )

    assert suite.algorithm_config.fixed_backup_policy_id == "retrace_waypoint"
    assert "retrace_waypoint" not in {
        candidate.name for candidate in controller.candidates(np.zeros(3))
    }
    mi_candidates = suite.mi_mpc_candidates()
    assert len(mi_candidates) == 32
    assert all(candidate.kind == "radial" for candidate in mi_candidates)
    assert all(candidate.name.startswith("mi_radial_") for candidate in mi_candidates)
    assert not {
        "stop",
        "nominal",
        "retrace_waypoint",
    }.intersection(candidate.name for candidate in mi_candidates)


def test_single_and_multi_backup_cbf_keep_warehouse_rollout_indexing() -> None:
    class ControllablePositionModel(FakeModel):
        def g(self, state):
            matrix = super().g(state)
            matrix[0, 0] = 1.0
            return matrix

    model = ControllablePositionModel()
    controller = FakeController(
        model,
        NLQuad3DControllerConfig(
            dt=model.dt,
            backup_horizon=0.4,
            safety_margin=0.0,
            safety_scale=1.0,
        ),
        None,
    )
    suite = NLQuad3DBaselineSuite(
        controller,
        model,
        None,
        waypoints=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )
    state = np.zeros(12)
    obstacles = np.array([[4.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]])
    suite._goal = np.array([5.0, 0.0, 0.0])
    suite._obstacles = obstacles.copy()
    suite._altitude_reference = 0.0
    suite._active_waypoint_index = 1
    suite._active_retrace_index = 0

    # The warehouse strict multi-backup candidate uses N + 1 samples from
    # t=0 through t=T and constrains every one, including the initial state.
    policy = controller.candidates(suite._goal)[0]
    multi_path, _ = suite._policy_rollout_margins(
        policy,
        maneuver_steps=suite._backup_steps,
    )(state, 0.0)
    multi_flow = suite._policy_path_flow_derivatives(
        state,
        policy,
        maneuver_steps=suite._backup_steps,
    )
    assert multi_path.shape == (suite._backup_steps + 1,)
    assert multi_flow.shape == multi_path.shape
    np.testing.assert_allclose(multi_flow, -2.5, atol=1e-7)
    assert multi_path[0] == pytest.approx(
        suite._point_margin(state, obstacles)
    )

    # The warehouse single BackupCBF instead allocates exactly N states and
    # creates path rows only for phi[1], ..., phi[N-1].
    single_path, terminal_value = suite._retrace_rollout_margins()(
        state,
        0.0,
    )
    single_flow = suite._retrace_path_flow_derivatives(state)
    first_propagated = model.step(state, np.zeros(4))
    assert single_path.shape == (suite._backup_steps,)
    assert single_flow.shape == single_path.shape
    np.testing.assert_allclose(single_flow, -2.5, atol=1e-7)
    assert single_path[0] == pytest.approx(
        suite._point_margin(state, obstacles)
    )
    assert single_path[1] == pytest.approx(
        suite._point_margin(first_propagated, obstacles)
    )

    candidate = suite._retrace_backup_candidate(
        state,
        np.full(4, 0.5),
    )
    path_rows = [
        row for row in candidate.halfspaces if ":path[" in row.label
    ]
    terminal_rows = [
        row for row in candidate.halfspaces if row.label.endswith(":terminal")
    ]
    assert candidate.path_values.shape == (suite._backup_steps,)
    assert len(path_rows) == suite._backup_steps - 1
    assert [row.label for row in path_rows] == [
        f"retrace_waypoint:path[{index}]"
        for index in range(1, suite._backup_steps)
    ]
    assert len(terminal_rows) == 1
    assert candidate.terminal_value == pytest.approx(terminal_value)


def test_single_and_multi_backup_pass_explicit_warehouse_formulations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel()
    controller = FakeController(
        model,
        NLQuad3DControllerConfig(dt=model.dt, backup_horizon=0.4),
        None,
    )
    suite = NLQuad3DBaselineSuite(
        controller,
        model,
        None,
        waypoints=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )
    suite._goal = np.array([5.0, 0.0, 0.0])
    suite._obstacles = np.zeros((0, 7))
    suite._altitude_reference = 0.0
    suite._active_waypoint_index = 1
    suite._active_retrace_index = 0
    captured: list[dict[str, object]] = []

    def capture(**kwargs):
        captured.append(kwargs)
        return kwargs

    monkeypatch.setattr(
        "examples.nl_quad3d.baselines.evaluate_backup_cbf_candidate",
        capture,
    )

    state = np.zeros(12)
    nominal = np.full(4, 0.5)
    suite._retrace_backup_candidate(state, nominal)
    suite._backup_candidate(
        controller.candidates(suite._goal)[0],
        state,
        nominal,
        maneuver_steps=1,
    )

    assert captured[0]["formulation"] == "single"
    assert captured[0]["path_constraint_start_index"] == 1
    assert captured[1]["formulation"] == "strict_multi"
    assert captured[1]["path_constraint_start_index"] == 0


def test_multi_backup_uses_a_common_stop_tail_not_retrace() -> None:
    class RecordingModel(FakeModel):
        def __init__(self) -> None:
            super().__init__()
            self.applied: list[np.ndarray] = []

        def step(self, state, control):
            self.applied.append(np.asarray(control, dtype=float).copy())
            return super().step(state, control)

        def stop_input(self, state, gain=3.0):
            del state, gain
            return np.full(4, 0.25)

    model = RecordingModel()
    controller = FakeController(
        model,
        NLQuad3DControllerConfig(dt=model.dt, backup_horizon=0.4),
        None,
    )
    suite = NLQuad3DBaselineSuite(
        controller,
        model,
        None,
        waypoints=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )
    suite._goal = np.array([5.0, 0.0, 0.0])
    suite._obstacles = np.zeros((0, 7))
    policy = controller.candidates(suite._goal)[0]

    suite._policy_rollout_margins(policy, maneuver_steps=1)(
        np.zeros(12),
        0.0,
    )

    # Four rollout controls plus the terminal successor.  Only the first is
    # the maneuver policy; every later action is the shared stop policy.
    assert len(model.applied) == suite._backup_steps + 1
    np.testing.assert_array_equal(model.applied[0], np.zeros(4))
    for control in model.applied[1:]:
        np.testing.assert_array_equal(control, np.full(4, 0.25))


def test_crowded_batched_strict_multi_matches_all_scalar_candidates() -> None:
    """Batching changes evaluation order, never the strict warehouse method."""

    scenario = benchmark.seeded_scenario(
        get_scenario(PLAYGROUND_CROWDED_SCENARIO),
        1,
        position_perturbation=0.12,
        velocity_perturbation=0.08,
        playground_obstacle_count=PLAYGROUND_OBSTACLE_COUNT,
    )
    model = NLQuad3D()
    controller = PLCBF_NLQuad3D(
        model,
        NLQuad3DControllerConfig(dt=model.dt),
        bounds=scenario.bounds,
    )
    suite = NLQuad3DBaselineSuite(
        controller,
        model,
        scenario.bounds,
        waypoints=scenario.waypoints,
    )
    state = scenario.initial_state
    goal = scenario.waypoints[1]
    nominal = model.nominal_input(state, goal)
    suite._goal = goal.copy()
    suite._obstacles = scenario.obstacles.copy()
    suite._altitude_reference = float(state[2])
    candidates = controller.candidates(goal)
    assert len(candidates) == 14
    maneuver_steps = int(
        round(
            suite.algorithm_config.multi_backup_maneuver_fraction
            * suite._backup_steps
        )
    )
    batched_data = suite._batched_strict_multi_data(
        candidates,
        state,
        maneuver_steps=maneuver_steps,
    )
    batched_candidates = suite._backup_candidates_batched(
        candidates,
        state,
        nominal,
        maneuver_steps=maneuver_steps,
    )

    drift = model.f(state)
    control_matrix = model.g(state)
    scalar_candidates = []
    for index, policy in enumerate(candidates):
        oracle = suite._policy_rollout_margins(
            policy,
            maneuver_steps=maneuver_steps,
        )
        (
            path,
            terminal,
            path_gradients,
            path_time_derivatives,
            terminal_gradient,
        ) = _rollout_derivatives(
            state,
            oracle,
            np.full(
                12,
                suite.algorithm_config.backup_cbf_gradient_step,
            ),
            max(
                suite.algorithm_config.time_derivative_step_s,
                model.dt,
            ),
        )
        path_flow_derivatives = suite._policy_path_flow_derivatives(
            state,
            policy,
            maneuver_steps=maneuver_steps,
        )
        np.testing.assert_allclose(
            batched_data.path_values[index],
            path,
            rtol=0.0,
            atol=5e-13,
        )
        assert batched_data.terminal_values[index] == pytest.approx(
            terminal,
            rel=0.0,
            abs=5e-13,
        )
        np.testing.assert_allclose(
            batched_data.path_gradients[index],
            path_gradients,
            rtol=0.0,
            atol=5e-9,
        )
        np.testing.assert_allclose(
            batched_data.path_time_derivatives[index],
            path_time_derivatives,
            rtol=0.0,
            atol=5e-13,
        )
        np.testing.assert_allclose(
            batched_data.terminal_gradients[index],
            terminal_gradient,
            rtol=0.0,
            atol=5e-9,
        )
        np.testing.assert_allclose(
            batched_data.path_flow_derivatives[index],
            path_flow_derivatives,
            rtol=0.0,
            atol=5e-9,
        )

        direct_control = suite._candidate_feedback(state, policy)
        scalar_candidates.append(
            evaluate_backup_cbf_candidate(
                policy_id=policy.name,
                state=state,
                nominal_control=nominal,
                lower=model.input_lower_bound,
                upper=model.input_upper_bound,
                drift=drift,
                control_matrix=control_matrix,
                backup_closed_loop_drift=(
                    drift + control_matrix @ direct_control
                ),
                rollout_margins=oracle,
                path_flow_derivatives=path_flow_derivatives,
                formulation="strict_multi",
                path_constraint_start_index=0,
                gradient_steps=np.full(
                    12,
                    suite.algorithm_config.backup_cbf_gradient_step,
                ),
                time_derivative_step=max(
                    suite.algorithm_config.time_derivative_step_s,
                    model.dt,
                ),
                alpha=suite.algorithm_config.backup_cbf_alpha,
                terminal_alpha=(
                    suite.algorithm_config.backup_cbf_terminal_alpha
                ),
                precomputed_derivatives=BackupCbfRolloutDerivatives(
                    path_values=path,
                    terminal_value=terminal,
                    path_gradients=path_gradients,
                    path_time_derivatives=path_time_derivatives,
                    terminal_gradient=terminal_gradient,
                ),
            )
        )

    for batched, scalar in zip(
        batched_candidates,
        scalar_candidates,
        strict=True,
    ):
        assert batched.status == scalar.status
        assert batched.feasible is scalar.feasible
        assert batched.rollout_safe is scalar.rollout_safe
        assert batched.terminal_safe is scalar.terminal_safe
        assert [row.label for row in batched.halfspaces] == [
            row.label for row in scalar.halfspaces
        ]
        for batched_row, scalar_row in zip(
            batched.halfspaces,
            scalar.halfspaces,
            strict=True,
        ):
            np.testing.assert_allclose(
                batched_row.normal,
                scalar_row.normal,
                rtol=0.0,
                atol=5e-9,
            )
            assert batched_row.offset == pytest.approx(
                scalar_row.offset,
                rel=0.0,
                abs=5e-9,
            )
        if scalar.control is None:
            assert batched.control is None
        else:
            np.testing.assert_allclose(
                batched.control,
                scalar.control,
                rtol=0.0,
                atol=1e-8,
            )

    direct_controls = {
        policy.name: suite._candidate_feedback(state, policy)
        for policy in candidates
    }
    scalar_decision = solve_multi_backup_cbf_min_intervention(
        scalar_candidates,
        direct_backup_controls=direct_controls,
        lower=model.input_lower_bound,
        upper=model.input_upper_bound,
        emergency_policy_id="stop",
    )
    batched_decision = suite.solve(
        "multi_backup_cbf_mi",
        state,
        goal,
        scenario.obstacles,
        nominal,
        active_waypoint_index=1,
    )
    assert batched_decision.policy_id == scalar_decision.policy_id
    assert batched_decision.feasible is scalar_decision.feasible
    assert batched_decision.status == scalar_decision.status
    np.testing.assert_allclose(
        batched_decision.control,
        scalar_decision.control,
        rtol=0.0,
        atol=1e-8,
    )


def test_crowded_retrace_batches_match_scalar_decisions() -> None:
    scenario = benchmark.seeded_scenario(
        get_scenario(PLAYGROUND_CROWDED_SCENARIO),
        1,
        position_perturbation=0.12,
        velocity_perturbation=0.08,
        playground_obstacle_count=PLAYGROUND_OBSTACLE_COUNT,
    )

    def make_suite() -> tuple[
        NLQuad3DBaselineSuite,
        NLQuad3D,
    ]:
        model = NLQuad3D()
        controller = PLCBF_NLQuad3D(
            model,
            NLQuad3DControllerConfig(dt=model.dt),
            bounds=scenario.bounds,
        )
        suite = NLQuad3DBaselineSuite(
            controller,
            model,
            scenario.bounds,
            waypoints=scenario.waypoints,
        )
        suite._goal = scenario.waypoints[1].copy()
        suite._obstacles = scenario.obstacles.copy()
        suite._altitude_reference = float(scenario.initial_state[2])
        suite._active_waypoint_index = 1
        suite._active_retrace_index = 0
        return suite, model

    state = scenario.initial_state
    goal = scenario.waypoints[1]
    suite, model = make_suite()
    nominal = model.nominal_input(state, goal)

    scalar_certificate = suite._retrace_pcbf_certificate_scalar(state)
    batched_certificate = suite._retrace_pcbf_certificate_batched(state)
    assert batched_certificate.policy_id == scalar_certificate.policy_id
    assert batched_certificate.valid is scalar_certificate.valid
    assert batched_certificate.diagnostic == scalar_certificate.diagnostic
    assert batched_certificate.value == pytest.approx(
        scalar_certificate.value,
        rel=0.0,
        abs=5e-12,
    )
    np.testing.assert_allclose(
        batched_certificate.backup_control,
        scalar_certificate.backup_control,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        batched_certificate.halfspaces[0].normal,
        scalar_certificate.halfspaces[0].normal,
        rtol=0.0,
        atol=5e-9,
    )
    assert batched_certificate.halfspaces[0].offset == pytest.approx(
        scalar_certificate.halfspaces[0].offset,
        rel=0.0,
        abs=5e-9,
    )
    scalar_pcbf_decision = solve_policy_pcbf(
        (scalar_certificate,),
        nominal,
        model.input_lower_bound,
        model.input_upper_bound,
        backup_policy_id=suite.algorithm_config.fixed_backup_policy_id,
    )
    batched_pcbf_decision = suite.solve(
        "pcbf",
        state,
        goal,
        scenario.obstacles,
        nominal,
        active_waypoint_index=1,
    )
    assert batched_pcbf_decision.status == scalar_pcbf_decision.status
    assert batched_pcbf_decision.feasible is scalar_pcbf_decision.feasible
    np.testing.assert_allclose(
        batched_pcbf_decision.control,
        scalar_pcbf_decision.control,
        rtol=0.0,
        atol=1e-8,
    )

    scalar_candidate = suite._retrace_backup_candidate_scalar(
        state,
        nominal,
    )
    batched_candidate = suite._retrace_backup_candidate_batched(
        state,
        nominal,
    )
    assert batched_candidate.status == scalar_candidate.status
    assert batched_candidate.feasible is scalar_candidate.feasible
    np.testing.assert_allclose(
        batched_candidate.path_values,
        scalar_candidate.path_values,
        rtol=0.0,
        atol=5e-12,
    )
    assert batched_candidate.terminal_value == pytest.approx(
        scalar_candidate.terminal_value,
        rel=0.0,
        abs=5e-12,
    )
    assert [row.label for row in batched_candidate.halfspaces] == [
        row.label for row in scalar_candidate.halfspaces
    ]
    for batched_row, scalar_row in zip(
        batched_candidate.halfspaces,
        scalar_candidate.halfspaces,
        strict=True,
    ):
        np.testing.assert_allclose(
            batched_row.normal,
            scalar_row.normal,
            rtol=0.0,
            atol=5e-9,
        )
        assert batched_row.offset == pytest.approx(
            scalar_row.offset,
            rel=0.0,
            abs=5e-9,
        )
    np.testing.assert_allclose(
        batched_candidate.control,
        scalar_candidate.control,
        rtol=0.0,
        atol=1e-8,
    )
    scalar_backup_decision = solve_fixed_backup_cbf(
        scalar_candidate,
        direct_backup_control=suite._direct_retrace_control(state),
        lower=model.input_lower_bound,
        upper=model.input_upper_bound,
    )
    batched_backup_decision = suite.solve(
        "backup_cbf",
        state,
        goal,
        scenario.obstacles,
        nominal,
        active_waypoint_index=1,
    )
    assert batched_backup_decision.status == scalar_backup_decision.status
    assert (
        batched_backup_decision.feasible
        is scalar_backup_decision.feasible
    )
    np.testing.assert_allclose(
        batched_backup_decision.control,
        scalar_backup_decision.control,
        rtol=0.0,
        atol=1e-8,
    )

    batched_gatekeeper_suite, batched_gatekeeper_model = make_suite()
    scalar_gatekeeper_suite, scalar_gatekeeper_model = make_suite()
    scalar_gatekeeper_suite.gatekeeper._trajectory_is_safe = (
        scalar_gatekeeper_suite._trajectory_is_safe_scalar
    )
    batched_gatekeeper = batched_gatekeeper_suite.solve(
        "gatekeeper",
        state,
        goal,
        scenario.obstacles,
        batched_gatekeeper_model.nominal_input(state, goal),
        active_waypoint_index=1,
    )
    scalar_gatekeeper = scalar_gatekeeper_suite.solve(
        "gatekeeper",
        state,
        goal,
        scenario.obstacles,
        scalar_gatekeeper_model.nominal_input(state, goal),
        active_waypoint_index=1,
    )
    assert batched_gatekeeper.status == scalar_gatekeeper.status
    assert batched_gatekeeper.feasible is scalar_gatekeeper.feasible
    np.testing.assert_allclose(
        batched_gatekeeper.control,
        scalar_gatekeeper.control,
        rtol=0.0,
        atol=1e-12,
    )
    assert batched_gatekeeper_suite.gatekeeper.committed is not None
    assert scalar_gatekeeper_suite.gatekeeper.committed is not None
    np.testing.assert_allclose(
        batched_gatekeeper_suite.gatekeeper.committed.states,
        scalar_gatekeeper_suite.gatekeeper.committed.states,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        batched_gatekeeper_suite.gatekeeper.committed.controls,
        scalar_gatekeeper_suite.gatekeeper.committed.controls,
        rtol=0.0,
        atol=1e-12,
    )


def test_single_and_multi_backup_use_distinct_warehouse_terminal_sets() -> None:
    model = StateInjectingModel(phi=1.0, yaw_rate=2.0)
    controller = FakeController(
        model,
        NLQuad3DControllerConfig(dt=model.dt, backup_horizon=0.4),
        None,
    )
    suite = NLQuad3DBaselineSuite(
        controller,
        model,
        None,
        waypoints=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )
    state = np.zeros(12)
    state[3] = 1.0
    suite._goal = np.array([5.0, 0.0, 0.0])
    suite._obstacles = np.array(
        [[20.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]]
    )
    suite._altitude_reference = 0.0
    policy = controller.candidates(suite._goal)[0]

    _, single_terminal = suite._retrace_rollout_margins()(state, 0.0)
    _, strict_multi_terminal = suite._policy_rollout_margins(
        policy,
        maneuver_steps=1,
    )(state, 0.0)

    # Legacy single BackupCBF checks terminal safety and the model velocity
    # envelope, not the strict candidate's near-hover attitude/rate set.
    assert single_terminal > 0.0
    assert strict_multi_terminal < 0.0


def test_mi_mpc_problem_uses_actual_warehouse_constants_and_full_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel()
    controller = FakeController(
        model,
        NLQuad3DControllerConfig(dt=model.dt, backup_horizon=0.4),
        None,
    )
    suite = NLQuad3DBaselineSuite(
        controller,
        model,
        None,
        waypoints=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )
    captured: dict[str, object] = {}

    class CapturedSolve(RuntimeError):
        pass

    def capture(problem, config):
        captured["problem"] = problem
        captured["config"] = config
        raise CapturedSolve

    monkeypatch.setattr(
        "examples.nl_quad3d.baselines.solve_big_m_trajectory_mpc",
        capture,
    )

    with pytest.raises(CapturedSolve):
        suite.solve(
            "mi_mpc",
            np.zeros(12),
            np.array([5.0, 0.0, 0.0]),
            np.zeros((0, 7)),
            np.full(4, 0.5),
            active_waypoint_index=1,
        )

    problem = captured["problem"]
    config = captured["config"]
    assert problem.branch_states.shape == (
        suite.algorithm_config.mi_num_radial_policies,
        suite._backup_steps + 1,
        12,
    )
    assert problem.branch_controls.shape == (
        suite.algorithm_config.mi_num_radial_policies,
        suite._backup_steps,
        4,
    )
    assert problem.position_indices == (0, 1, 2)
    assert config.position_tube == pytest.approx(3.0)
    assert config.early_control_tube == pytest.approx(6.0)
    assert config.early_control_steps == 2
    assert config.big_m_position == pytest.approx(400.0)
    assert config.big_m_control == pytest.approx(60.0)
    assert config.big_m_safety == pytest.approx(50.0)
    assert config.tracking_weight == pytest.approx(8.0)
    assert config.terminal_weight == pytest.approx(16.0)
    assert config.velocity_weight == pytest.approx(0.15)
    assert config.control_weight == pytest.approx(0.02)
    assert config.nominal_weight == pytest.approx(0.5)
    assert config.time_limit_s == pytest.approx(1.0)
    assert config.mip_rel_gap == pytest.approx(0.05)


def test_native_plcbf_dispatch_does_not_use_generic_certificate_selector() -> None:
    class NativeOnlyController(FakeController):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.native_calls = 0

        def policy_certificates(self, state, goal, obstacles):
            del state, goal, obstacles
            raise AssertionError("native PL-CBF must not request generic certificates")

        def solve_control_problem(self, state, goal, obstacles, control_ref=None):
            del state, goal, obstacles
            self.native_calls += 1
            self.last_certificates = (_certificate("native_plcbf"),)
            self.last_decision = SimpleNamespace(
                policy_id="native_plcbf",
                diagnostics=SimpleNamespace(used_fallback=False),
            )
            self.last_status = "filtered"
            return np.asarray(control_ref, dtype=float)

    model = FakeModel()
    controller = NativeOnlyController(
        model,
        NLQuad3DControllerConfig(dt=model.dt),
        None,
    )
    suite = NLQuad3DBaselineSuite(
        controller,
        model,
        None,
        waypoints=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )

    decision = suite.solve(
        "plcbf",
        np.zeros(12),
        np.array([5.0, 0.0, 0.0]),
        np.zeros((0, 7)),
        np.full(4, 0.5),
        active_waypoint_index=1,
    )

    assert controller.native_calls == 1
    assert decision.policy_id == "native_plcbf"
    assert decision.policy_decision is controller.last_decision
    assert decision.feasible


def test_fixed_policy_pcbf_builds_baseline_only_retrace_certificate() -> None:
    class NoSharedCertificateController(FakeController):
        def policy_certificates(self, state, goal, obstacles):
            del state, goal, obstacles
            raise AssertionError("fixed PCBF must build its retrace certificate")

    model = FakeModel()
    controller = NoSharedCertificateController(
        model,
        NLQuad3DControllerConfig(dt=model.dt, backup_horizon=0.1),
        None,
    )
    suite = NLQuad3DBaselineSuite(
        controller,
        model,
        None,
        waypoints=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )

    decision = suite.solve(
        "pcbf",
        np.zeros(12),
        np.array([5.0, 0.0, 0.0]),
        np.zeros((0, 7)),
        np.full(4, 0.5),
        active_waypoint_index=1,
    )

    assert decision.policy_id == "retrace_waypoint"
    assert decision.feasible


def test_library_pcbf_unsafe_fallback_reports_executed_stop_policy() -> None:
    model = FakeModel()
    controller = FakeController(
        model,
        NLQuad3DControllerConfig(dt=model.dt, backup_horizon=0.1),
        None,
    )
    suite = NLQuad3DBaselineSuite(
        controller,
        model,
        None,
        waypoints=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )

    decision = suite.solve(
        "library_pcbf_mi",
        np.zeros(12),
        np.array([5.0, 0.0, 0.0]),
        np.zeros((0, 7)),
        np.full(4, 0.5),
        certificates=(
            _certificate("stop", value=-2.0, backup=0.0),
            _certificate("radial_0", value=-0.1, backup=0.75),
        ),
        active_waypoint_index=1,
    )

    assert decision.used_fallback
    assert not decision.feasible
    assert decision.policy_id == "stop"
    np.testing.assert_array_equal(decision.control, np.zeros(4))
    assert decision.policy_decision.certificate is None
    assert (
        decision.policy_decision.diagnostics.fallback_source
        == "emergency_policy:stop"
    )


def test_trajectory_safety_does_not_prefilter_tilt_or_body_rate() -> None:
    model = FakeModel()
    controller = FakeController(
        model,
        NLQuad3DControllerConfig(dt=model.dt),
        None,
    )
    suite = NLQuad3DBaselineSuite(
        controller,
        model,
        None,
        waypoints=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )
    obstacles = np.array([[4.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]])
    level = np.zeros(12)
    tilted = level.copy()
    tilted[6] = np.deg2rad(80.0)
    tilted[9:12] = [5.0, 5.0, 5.0]

    assert suite._point_margin(level, obstacles) == pytest.approx(
        suite._point_margin(tilted, obstacles)
    )


def test_near_wall_sphere_free_state_is_unconstrained_for_plcbf_and_baselines() -> None:
    """WorldBounds reflects hazards; it is not a method-specific robot wall."""

    model = NLQuad3D()
    bounds = WorldBounds((0.0, 0.0, 0.0), (20.0, 20.0, 10.0))
    controller = PLCBF_NLQuad3D(
        model,
        NLQuad3DControllerConfig(
            dt=model.dt,
            backup_horizon=0.1,
            num_radial_policies=2,
            max_obstacles=2,
            nominal_prefix_steps=0,
        ),
        bounds=bounds,
    )
    suite = NLQuad3DBaselineSuite(
        controller,
        model,
        bounds,
        waypoints=np.array([[0.05, 10.0, 5.0], [19.0, 10.0, 5.0]]),
    )
    near_wall = np.zeros(12)
    near_wall[:3] = [0.05, 10.0, 5.0]
    empty = np.zeros((0, 7))

    certificates = controller.decision_certificates(
        near_wall,
        np.array([19.0, 10.0, 5.0]),
        empty,
    )
    assert certificates
    assert all(certificate.valid for certificate in certificates)
    assert all(
        certificate.value == pytest.approx(100.0)
        for certificate in certificates
    )
    assert suite._point_margin(near_wall, empty) == pytest.approx(1e12)
    np.testing.assert_allclose(
        suite._batch_point_margins(near_wall[None, :], empty),
        [suite._point_margin(near_wall, empty)],
    )
    assert suite._trajectory_is_safe(near_wall[None, :])

    # Nonempty scalar and vectorized paths retain exact sphere-clearance parity.
    second = near_wall.copy()
    second[0] = 1.0
    obstacles = np.array([[2.0, 10.0, 5.25, 0.5, 0.0, 0.0, 0.0]])
    states = np.stack([near_wall, second])
    np.testing.assert_allclose(
        suite._batch_point_margins(states, obstacles),
        [suite._point_margin(state, obstacles) for state in states],
    )

    state_lower, state_upper = suite._state_bounds()
    np.testing.assert_array_equal(state_lower[:3], np.full(3, -100.0))
    np.testing.assert_array_equal(state_upper[:3], np.full(3, 100.0))


@pytest.mark.parametrize(
    "tilt_max_rad",
    [0.0, -1.0, float("nan"), float("inf"), np.pi + 0.01],
)
def test_tilt_max_must_be_a_finite_body_angle(
    tilt_max_rad: float,
) -> None:
    with pytest.raises(ValueError, match="tilt_max_rad"):
        benchmark.NLQuad3DBenchmarkConfig(tilt_max_rad=tilt_max_rad)


def test_benchmark_and_controller_rollout_tilt_limits_must_match() -> None:
    controller = NLQuad3DControllerConfig(
        rollout_tilt_max_rad=np.deg2rad(70.0)
    )

    with pytest.raises(ValueError, match="rollout_tilt_max_rad must equal"):
        benchmark.NLQuad3DBenchmarkConfig(controller_config=controller)

    with pytest.raises(ValueError, match="rollout_tilt_max_rad must equal"):
        benchmark.run_trial(
            "plcbf",
            _scenario(),
            max_steps=1,
            controller_config=controller,
            warmup=False,
            model_factory=FakeModel,
            controller_factory=_controller_factory(),
        )


def test_seeded_scenarios_are_bounded_reproducible_and_seed_zero_is_exact() -> None:
    source = NLQuad3DScenario(
        name="bounded",
        waypoints=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        obstacles=np.array(
            [
                [0.12, 0.15, 0.2, 0.1, 0.2, -0.1, 0.0],
                [0.88, 0.85, 0.8, 0.1, -0.2, 0.1, 0.0],
            ]
        ),
        bounds=WorldBounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )
    exact = benchmark.seeded_scenario(source, 0)
    first = benchmark.seeded_scenario(
        source,
        17,
        position_perturbation=0.2,
        velocity_perturbation=0.05,
    )
    repeated = benchmark.seeded_scenario(
        source,
        17,
        position_perturbation=0.2,
        velocity_perturbation=0.05,
    )
    different = benchmark.seeded_scenario(
        source,
        18,
        position_perturbation=0.2,
        velocity_perturbation=0.05,
    )

    np.testing.assert_array_equal(exact.obstacles, source.obstacles)
    np.testing.assert_array_equal(first.obstacles, repeated.obstacles)
    assert not np.array_equal(first.obstacles, different.obstacles)
    assert np.all(
        np.abs(first.obstacles[:, 4:7] - source.obstacles[:, 4:7])
        <= 0.05 + 1e-12
    )
    radii = first.obstacles[:, 3:4]
    assert np.all(first.obstacles[:, :3] >= radii)
    assert np.all(first.obstacles[:, :3] <= 1.0 - radii)


def test_crowded_playground_seed_is_replayable_bounded_and_protected() -> None:
    source = get_scenario(PLAYGROUND_CROWDED_SCENARIO)
    first = benchmark.seeded_scenario(source, 17)
    repeated = benchmark.seeded_scenario(source, 17)
    different = benchmark.seeded_scenario(source, 18)
    reference = get_scenario("playground_corridor")

    assert first.obstacles.shape == (PLAYGROUND_OBSTACLE_COUNT, 7)
    np.testing.assert_array_equal(first.obstacles, repeated.obstacles)
    np.testing.assert_array_equal(
        first.obstacles[:PLAYGROUND_REFERENCE_OBSTACLE_COUNT],
        reference.obstacles,
    )
    assert not np.array_equal(
        first.obstacles[PLAYGROUND_REFERENCE_OBSTACLE_COUNT:],
        different.obstacles[PLAYGROUND_REFERENCE_OBSTACLE_COUNT:],
    )
    assert first.bounds is not None
    lower = np.asarray(first.bounds.lower)
    upper = np.asarray(first.bounds.upper)
    positions = first.obstacles[:, :3]
    radii = first.obstacles[:, 3:4]
    assert np.all(positions >= lower + radii)
    assert np.all(positions <= upper - radii)
    random_positions = positions[PLAYGROUND_REFERENCE_OBSTACLE_COUNT:]
    for endpoint in first.waypoints[[0, -1]]:
        distances = np.linalg.norm(random_positions - endpoint, axis=1)
        assert np.all(distances >= PLAYGROUND_START_GOAL_PROTECTION)

    random_obstacles = first.obstacles[PLAYGROUND_REFERENCE_OBSTACLE_COUNT:]
    np.testing.assert_allclose(
        random_obstacles[:, 3],
        PLAYGROUND_OBSTACLE_RADIUS,
    )
    speeds = np.linalg.norm(random_obstacles[:, 4:7], axis=1)
    assert np.all(speeds >= PLAYGROUND_OBSTACLE_SPEED_MIN - 1e-12)
    assert np.all(speeds <= PLAYGROUND_OBSTACLE_SPEED_MAX + 1e-12)


def test_stress_playground_is_replayable_nonoverlapping_and_protected() -> None:
    source = get_scenario(PLAYGROUND_STRESS_SCENARIO)
    first = benchmark.seeded_scenario(source, 17)
    repeated = benchmark.seeded_scenario(source, 17)
    different = benchmark.seeded_scenario(source, 18)

    assert first.obstacles.shape == (PLAYGROUND_STRESS_OBSTACLE_COUNT, 7)
    np.testing.assert_array_equal(first.obstacles, repeated.obstacles)
    assert not np.array_equal(first.obstacles, different.obstacles)
    assert first.bounds is not None
    lower = np.asarray(first.bounds.lower)
    upper = np.asarray(first.bounds.upper)
    positions = first.obstacles[:, :3]
    radii = first.obstacles[:, 3:4]
    assert np.all(positions >= lower + radii)
    assert np.all(positions <= upper - radii)
    for endpoint in first.waypoints[[0, -1]]:
        distances = np.linalg.norm(positions - endpoint, axis=1)
        assert np.all(distances >= PLAYGROUND_START_GOAL_PROTECTION)

    pair_distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :],
        axis=2,
    )
    pair_clearances = pair_distances - radii - radii.T
    np.fill_diagonal(pair_clearances, np.inf)
    assert np.min(pair_clearances) >= (
        PLAYGROUND_STRESS_PAIR_CLEARANCE - 1e-12
    )
    speeds = np.linalg.norm(first.obstacles[:, 4:7], axis=1)
    assert np.all(speeds >= PLAYGROUND_STRESS_OBSTACLE_SPEED_MIN - 1e-12)
    assert np.all(speeds <= PLAYGROUND_STRESS_OBSTACLE_SPEED_MAX + 1e-12)


def test_stress_six_axis_streams_are_balanced_timed_and_corridor_bounded() -> None:
    scenario = make_playground_stress_scenario(9)
    structured = scenario.obstacles[:PLAYGROUND_STRESS_CROSS_FLOW_COUNT]
    background = scenario.obstacles[PLAYGROUND_STRESS_CROSS_FLOW_COUNT:]

    # Ordering is audit metadata only: every event cycles through all six
    # signed world axes.  A pair on each axis reaches its event ring at the
    # same seeded time without reading a controller trajectory.
    for lane, axis, sign in (
        (0, 0, 1.0),
        (1, 0, -1.0),
        (2, 1, 1.0),
        (3, 1, -1.0),
        (4, 2, 1.0),
        (5, 2, -1.0),
    ):
        lane_obstacles = structured[lane::6]
        assert lane_obstacles.shape[0] == 4
        assert np.all(sign * lane_obstacles[:, axis + 4] > 0.0)
        other_axes = [value for value in range(3) if value != axis]
        np.testing.assert_array_equal(
            lane_obstacles[:, np.asarray(other_axes) + 4],
            0.0,
        )

    for event_index, expected_time in enumerate((2.00, 2.55, 3.10, 3.65)):
        event = structured[event_index * 6 : (event_index + 1) * 6]
        pair_times = []
        for axis in range(3):
            positive = event[2 * axis]
            negative = event[2 * axis + 1]
            pair_times.append(
                (negative[axis] - positive[axis])
                / (positive[axis + 4] - negative[axis + 4])
            )
        np.testing.assert_allclose(pair_times, pair_times[0], atol=1e-12)
        assert pair_times[0] == pytest.approx(expected_time, abs=0.1)
        projected = event[:, :3] + pair_times[0] * event[:, 4:7]
        assert np.all((projected[:, 0] >= 5.7) & (projected[:, 0] <= 12.2))
        assert np.all((projected[:, 1] >= 9.2) & (projected[:, 1] <= 10.8))
        assert np.all((projected[:, 2] >= 4.7) & (projected[:, 2] <= 5.3))

    corridor_lower = np.asarray(PLAYGROUND_STRESS_CORRIDOR_LOWER)
    corridor_upper = np.asarray(PLAYGROUND_STRESS_CORRIDOR_UPPER)
    assert np.all(background[:, :3] >= corridor_lower)
    assert np.all(background[:, :3] <= corridor_upper)


def test_stress_generator_protocol_holds_across_benchmark_and_audit_seeds() -> None:
    for seed in (*range(1, 101), *range(1001, 1011)):
        scenario = make_playground_stress_scenario(seed)
        positions = scenario.obstacles[:, :3]
        radii = scenario.obstacles[:, 3]
        pair_distances = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :],
            axis=2,
        )
        pair_clearances = pair_distances - radii[:, None] - radii[None, :]
        np.fill_diagonal(pair_clearances, np.inf)

        assert np.min(pair_clearances) >= (
            PLAYGROUND_STRESS_PAIR_CLEARANCE - 1e-12
        ), seed
        assert np.all(
            np.linalg.norm(positions - scenario.waypoints[0], axis=1)
            >= PLAYGROUND_START_GOAL_PROTECTION
        ), seed
        assert np.all(
            np.linalg.norm(positions - scenario.waypoints[-1], axis=1)
            >= PLAYGROUND_START_GOAL_PROTECTION
        ), seed


def test_crowded_scenario_uses_playground_launch_velocity_and_count_override() -> None:
    scenario = make_playground_crowded_scenario(9, obstacle_count=12)

    assert scenario.obstacles.shape == (12, 7)
    np.testing.assert_array_equal(scenario.initial_state[3:6], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(scenario.initial_velocity, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="five reference"):
        make_playground_crowded_scenario(0, obstacle_count=4)


def test_stress_scenario_count_override_preserves_protocol_minimum() -> None:
    scenario = make_playground_stress_scenario(9, obstacle_count=12)

    assert scenario.obstacles.shape == (12, 7)
    np.testing.assert_array_equal(scenario.initial_state[3:6], [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="at least five"):
        make_playground_stress_scenario(0, obstacle_count=4)


def test_crowded_trial_row_records_realized_obstacle_count() -> None:
    result = benchmark.run_trial(
        "plcbf",
        PLAYGROUND_CROWDED_SCENARIO,
        seed=23,
        max_steps=1,
        controller_config=NLQuad3DControllerConfig(dt=0.1),
        playground_obstacle_count=12,
        warmup=False,
        model_factory=FakeModel,
        controller_factory=_controller_factory(),
        clock=DeterministicClock(),
    )

    assert result.case_metrics["obstacle_count"] == 12
    assert result.case_metrics["initial_velocity_x_mps"] == 1.0
    assert result.case_metrics["initial_velocity_y_mps"] == 0.0
    assert result.case_metrics["initial_velocity_z_mps"] == 0.0


def test_stress_trial_row_records_realized_generator_protocol() -> None:
    result = benchmark.run_trial(
        "plcbf",
        PLAYGROUND_STRESS_SCENARIO,
        seed=23,
        max_steps=1,
        controller_config=NLQuad3DControllerConfig(dt=0.1),
        playground_obstacle_count=12,
        warmup=False,
        model_factory=FakeModel,
        controller_factory=_controller_factory(),
        clock=DeterministicClock(),
    )

    metrics = result.case_metrics
    assert metrics["obstacle_count"] == 12
    assert metrics["scenario_generator"] == (
        "seeded_balanced_six_axis_stream_corridor"
    )
    assert metrics["scenario_protocol_version"] == (
        PLAYGROUND_STRESS_PROTOCOL_VERSION
    )
    assert metrics["structured_cross_flow_count"] == 12
    assert metrics["structured_stream_count"] == 12
    assert metrics["corridor_random_count"] == 0
    assert metrics["initial_min_pair_surface_clearance_m"] >= (
        PLAYGROUND_STRESS_PAIR_CLEARANCE - 1e-12
    )
    assert metrics["initial_obstacle_speed_min_mps"] >= (
        PLAYGROUND_STRESS_OBSTACLE_SPEED_MIN - 1e-12
    )
    assert metrics["initial_obstacle_speed_max_mps"] <= (
        PLAYGROUND_STRESS_OBSTACLE_SPEED_MAX + 1e-12
    )


def test_every_method_receives_identical_seeded_obstacles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[np.ndarray] = []

    class RecordingSuite:
        def __init__(self, controller, model, bounds, *, waypoints):
            del controller, model, bounds, waypoints

        def solve(self, method, state, goal, obstacles, nominal, **kwargs):
            del state, goal, kwargs
            observed.append(np.asarray(obstacles).copy())
            return BaselineDecision(
                method=str(method.value),
                control=nominal,
                policy_id="test",
                feasible=True,
                status="test",
                used_fallback=False,
                objective=0.0,
                solve_time_s=0.0,
            )

        def mi_mpc_candidates(self):
            return (SimpleNamespace(kind="radial"),)

    monkeypatch.setattr(benchmark, "NLQuad3DBaselineSuite", RecordingSuite)

    source = _scenario(
        goal_x=5.0,
        obstacles=np.array([[4.0, 1.0, 0.5, 0.1, -0.2, 0.0, 0.0]]),
    )
    config = benchmark.NLQuad3DBenchmarkConfig(
        methods=BENCHMARK_METHODS,
        scenarios=("synthetic",),
        seeds=(23,),
        max_steps=1,
        controller_config=NLQuad3DControllerConfig(dt=0.1),
        warmup=False,
    )
    benchmark.run_benchmark(
        config,
        model_factory=FakeModel,
        controller_factory=_controller_factory(),
        scenario_loader=lambda _: source,
        clock=DeterministicClock(),
    )

    assert len(observed) == len(BENCHMARK_METHODS)
    assert not np.array_equal(observed[0], source.obstacles)
    for obstacles in observed[1:]:
        np.testing.assert_array_equal(obstacles, observed[0])


def test_all_eight_methods_share_the_same_scenario_and_native_dispatch_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PassthroughSuite:
        def __init__(self, controller, model, bounds, *, waypoints):
            del controller, model, bounds, waypoints
            self.last_mi_mpc_result = None

        def solve(self, method, state, goal, obstacles, nominal, **kwargs):
            del state, goal, obstacles, kwargs
            if method is benchmark.BenchmarkMethod.MI_MPC:
                self.last_mi_mpc_result = SimpleNamespace(
                    safety_feasible=True,
                    safety_threshold_relaxed=False,
                )
            return BaselineDecision(
                method=str(method.value),
                control=nominal,
                policy_id="test",
                feasible=True,
                status="test",
                used_fallback=False,
                objective=0.0,
                solve_time_s=0.0,
            )

        def mi_mpc_candidates(self):
            return (SimpleNamespace(kind="radial"),)

    monkeypatch.setattr(benchmark, "NLQuad3DBaselineSuite", PassthroughSuite)
    scenario = _scenario()
    config = benchmark.NLQuad3DBenchmarkConfig(
        methods=BENCHMARK_METHODS,
        scenarios=("synthetic",),
        seeds=(7,),
        max_steps=2,
        controller_config=NLQuad3DControllerConfig(dt=0.1),
        warmup=False,
    )
    results = benchmark.run_benchmark(
        config,
        model_factory=FakeModel,
        controller_factory=_controller_factory(),
        scenario_loader=lambda _: scenario,
        clock=DeterministicClock(),
    )

    assert len(results) == 8
    assert {item.algorithm for item in results} == set(BENCHMARK_METHODS)
    assert {item.case_id for item in results} == {"synthetic/seed-7"}
    assert all(item.outcome is BenchmarkOutcome.SUCCESS for item in results)
    assert all(item.case_metrics["reached_goal"] is True for item in results)
    assert all(item.case_metrics["control_steps"] == 2 for item in results)
    assert all(item.case_metrics["infeasible_count"] == 0 for item in results)
    assert all(len(item.solve_times_s) == 2 for item in results)
    for item in results:
        np.testing.assert_allclose(item.solve_times_s, [0.002, 0.002])

    by_method = {item.algorithm: item for item in results}
    assert all(
        item.case_metrics["solver_fallback_count"] == 0 for item in results
    )
    assert by_method["mi_mpc"].case_metrics["backup_executed_count"] == 0
    assert by_method["mi_mpc"].case_metrics["shield_active_count"] == 0
    assert by_method["mi_mpc"].case_metrics["mi_mpc_result_count"] == 2
    assert (
        by_method["mi_mpc"]
        .case_metrics["mi_mpc_requested_safety_feasible_count"]
        == 2
    )
    assert (
        by_method["mi_mpc"]
        .case_metrics["mi_mpc_safety_threshold_relaxed_count"]
        == 0
    )
    for method in set(BENCHMARK_METHODS) - {"mi_mpc"}:
        assert by_method[method].case_metrics["backup_executed_count"] == 0
        assert by_method[method].case_metrics["mi_mpc_result_count"] is None


def test_physical_sphere_collision_and_clearance_are_measured() -> None:
    obstacle = np.array([[0.25, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]])
    result = benchmark.run_trial(
        "plcbf",
        _scenario(goal_x=2.0, obstacles=obstacle),
        max_steps=2,
        controller_config=NLQuad3DControllerConfig(dt=0.1),
        warmup=False,
        model_factory=FakeModel,
        controller_factory=_controller_factory(),
        clock=DeterministicClock(),
    )

    assert result.outcome is BenchmarkOutcome.COLLISION
    assert result.collision
    assert result.min_clearance == pytest.approx(-0.2)
    assert result.case_metrics["control_steps"] == 1


def test_fallback_infeasibility_and_policy_switches_are_counted() -> None:
    unsafe = benchmark.run_trial(
        "plcbf",
        _scenario(goal_x=5.0),
        max_steps=1,
        controller_config=NLQuad3DControllerConfig(dt=0.1),
        warmup=False,
        model_factory=FakeModel,
        controller_factory=_controller_factory(UnsafeController),
        clock=DeterministicClock(),
    )
    switching = benchmark.run_trial(
        "plcbf",
        _scenario(goal_x=5.0),
        max_steps=2,
        controller_config=NLQuad3DControllerConfig(dt=0.1),
        warmup=False,
        model_factory=FakeModel,
        controller_factory=_controller_factory(SwitchingController),
        clock=DeterministicClock(),
    )

    assert unsafe.outcome is BenchmarkOutcome.INFEASIBLE
    assert unsafe.case_metrics["infeasible_count"] == 1
    assert unsafe.case_metrics["selector_fallback_count"] == 1
    assert unsafe.case_metrics["solver_fallback_count"] == 1
    assert unsafe.case_metrics["backup_executed_count"] == 1
    assert unsafe.case_metrics["shield_active_count"] == 0
    assert unsafe.case_metrics["fallback_count"] == 1
    assert switching.outcome is BenchmarkOutcome.TIMEOUT
    assert switching.case_metrics["policy_switches"] == 1


@pytest.mark.parametrize("method", ["mps", "gatekeeper"])
def test_normal_shield_backup_is_not_a_solver_fallback(method: str) -> None:
    parsed = benchmark.BenchmarkMethod(method)
    decision = BaselineDecision(
        method=method,
        control=np.zeros(4),
        policy_id="retrace_waypoint",
        feasible=True,
        status="continued_committed_trajectory",
        used_fallback=True,
        objective=0.0,
        solve_time_s=0.0,
    )
    selector, solver, backup, shield = benchmark._decision_metric_flags(
        parsed,
        decision,
    )

    assert not selector
    assert not solver
    assert backup
    assert shield


@pytest.mark.parametrize(
    (
        "method",
        "status",
        "used_fallback",
        "selector_fallback",
        "feasible",
        "expected_backup",
    ),
    [
        ("pcbf", "fallback:qp_failed", True, True, False, False),
        ("plcbf", "fallback:qp_failed", True, True, False, True),
        (
            "backup_cbf",
            "nominal_after_qp_failure:infeasible",
            True,
            False,
            False,
            False,
        ),
        (
            "backup_cbf",
            "fallback_after:infeasible",
            True,
            False,
            False,
            True,
        ),
        (
            "multi_backup_cbf_mi",
            "no_feasible_backup_cbf_candidate",
            True,
            False,
            False,
            True,
        ),
        ("mi_mpc", "optimal", False, False, True, False),
        ("mi_mpc", "solver_failure", True, False, False, True),
    ],
)
def test_backup_execution_metric_counts_only_executed_backup_paths(
    method: str,
    status: str,
    used_fallback: bool,
    selector_fallback: bool,
    feasible: bool,
    expected_backup: bool,
) -> None:
    policy_decision = (
        SimpleNamespace(
            diagnostics=SimpleNamespace(used_fallback=selector_fallback)
        )
        if selector_fallback
        else None
    )
    decision = BaselineDecision(
        method=method,
        control=np.zeros(4),
        policy_id="test",
        feasible=feasible,
        status=status,
        used_fallback=used_fallback,
        objective=0.0,
        solve_time_s=0.0,
        policy_decision=policy_decision,
    )

    _, _, backup, _ = benchmark._decision_metric_flags(
        benchmark.BenchmarkMethod(method),
        decision,
    )

    assert backup is expected_backup


def test_mi_mpc_safety_diagnostics_distinguish_relaxed_admission() -> None:
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
    assert (
        benchmark._mi_mpc_safety_flags(
            benchmark.BenchmarkMethod.PLCBF,
            suite,
        )
        is None
    )


def test_completed_fallback_run_is_success_with_diagnostic_count() -> None:
    result = benchmark.run_trial(
        "plcbf",
        _scenario(goal_x=0.25),
        max_steps=1,
        controller_config=NLQuad3DControllerConfig(dt=0.1),
        warmup=False,
        model_factory=FakeModel,
        controller_factory=_controller_factory(UnsafeController),
        clock=DeterministicClock(),
    )

    assert result.outcome is BenchmarkOutcome.SUCCESS
    assert result.case_metrics["reached_goal"] is True
    assert result.case_metrics["infeasible_count"] == 1
    assert result.case_metrics["solver_fallback_count"] == 1
    assert result.case_metrics["backup_executed_count"] == 1
    assert result.case_metrics["fallback_count"] == 1


def test_excessive_reference_body_tilt_is_terminal_infeasible_without_clamp() -> None:
    result = benchmark.run_trial(
        "plcbf",
        _scenario(goal_x=5.0),
        max_steps=3,
        controller_config=NLQuad3DControllerConfig(dt=0.1),
        warmup=False,
        model_factory=lambda: StateInjectingModel(phi=np.deg2rad(65.0)),
        controller_factory=_controller_factory(),
        clock=DeterministicClock(),
    )

    assert result.outcome is BenchmarkOutcome.INFEASIBLE
    assert result.case_metrics["control_steps"] == 1
    assert result.case_metrics["max_tilt_deg"] == pytest.approx(65.0)
    assert result.case_metrics["tilt_max_deg"] == pytest.approx(60.0)
    assert result.case_metrics["tilt_max_violated"] is True
    assert result.case_metrics["state_bound_violation"] == "tilt_max"


def test_state_bound_violation_precedes_goal_on_the_same_transition() -> None:
    result = benchmark.run_trial(
        "plcbf",
        _scenario(goal_x=0.25),
        max_steps=1,
        controller_config=NLQuad3DControllerConfig(dt=0.1),
        warmup=False,
        model_factory=lambda: StateInjectingModel(phi=np.deg2rad(65.0)),
        controller_factory=_controller_factory(),
        clock=DeterministicClock(),
    )

    assert result.outcome is BenchmarkOutcome.INFEASIBLE
    assert result.case_metrics["reached_goal"] is False
    assert result.case_metrics["state_bound_violation"] == "tilt_max"


def test_tilt_uses_body_z_angle_formula_and_strict_reference_limit() -> None:
    result = benchmark.run_trial(
        "plcbf",
        _scenario(goal_x=5.0),
        max_steps=1,
        controller_config=NLQuad3DControllerConfig(dt=0.1),
        warmup=False,
        model_factory=lambda: StateInjectingModel(
            phi=np.deg2rad(45.0),
            theta=np.deg2rad(45.0),
        ),
        controller_factory=_controller_factory(),
        clock=DeterministicClock(),
    )

    assert result.outcome is BenchmarkOutcome.TIMEOUT
    assert result.case_metrics["max_tilt_deg"] == pytest.approx(60.0)
    assert result.case_metrics["attitude_bound_deg"] == pytest.approx(30.0)
    assert result.case_metrics["max_attitude_excess_deg"] == pytest.approx(30.0)
    assert result.case_metrics["tilt_max_violated"] is False


def test_desired_yaw_slew_cap_is_not_a_hard_body_yaw_rate_bound() -> None:
    result = benchmark.run_trial(
        "plcbf",
        _scenario(goal_x=5.0),
        max_steps=1,
        controller_config=NLQuad3DControllerConfig(dt=0.1),
        warmup=False,
        model_factory=lambda: StateInjectingModel(yaw_rate=-2.1),
        controller_factory=_controller_factory(),
        clock=DeterministicClock(),
    )

    assert result.outcome is BenchmarkOutcome.TIMEOUT
    assert result.case_metrics["control_steps"] == 1
    assert result.case_metrics[
        "max_abs_body_yaw_rate_rad_s"
    ] == pytest.approx(2.1)
    assert result.case_metrics[
        "nominal_yaw_slew_max_rad_s"
    ] == pytest.approx(2.0)
    assert result.case_metrics["max_body_rate_norm_rad_s"] == pytest.approx(2.1)
    assert result.case_metrics["body_rate_max_rad_s"] == pytest.approx(6.0)
    assert result.case_metrics["body_rate_bound_violated"] is False
    assert result.case_metrics["state_bound_violation"] == "none"


def test_oracle_errors_become_error_results_without_aborting_grid() -> None:
    result = benchmark.run_trial(
        "plcbf",
        _scenario(),
        max_steps=1,
        controller_config=NLQuad3DControllerConfig(dt=0.1),
        warmup=False,
        model_factory=FakeModel,
        controller_factory=_controller_factory(FailingController),
    )

    assert result.outcome is BenchmarkOutcome.ERROR
    assert result.error == "RuntimeError: oracle failed"
    assert result.case_metrics["control_steps"] == 0


def test_trial_seed_is_isolated_from_process_random_state() -> None:
    np.random.seed(123)
    random.seed(123)
    expected_numpy = np.random.random()
    expected_python = random.random()
    np.random.seed(123)
    random.seed(123)

    benchmark.run_trial(
        "plcbf",
        _scenario(),
        seed=999,
        max_steps=1,
        controller_config=NLQuad3DControllerConfig(dt=0.1),
        warmup=False,
        model_factory=FakeModel,
        controller_factory=_controller_factory(),
    )

    assert np.random.random() == expected_numpy
    assert random.random() == expected_python


def test_cli_accepts_repeated_method_scenario_seed_and_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured = {}
    paths = BenchmarkReportPaths(
        csv=tmp_path / "report.csv",
        json=tmp_path / "report.json",
        markdown=tmp_path / "report.md",
    )
    result = BenchmarkResult(
        "plcbf",
        "head_on/seed-4",
        4,
        "timeout",
        case_metrics={"control_steps": 1},
    )

    def fake_run_and_write(config, output):
        captured["config"] = config
        captured["output"] = output
        return (result,), paths

    monkeypatch.setattr(benchmark, "run_and_write", fake_run_and_write)
    return_code = benchmark.main(
        [
            "--method",
            "plcbf",
            "--scenario",
            "head_on",
            "--seed",
            "4",
            "--steps",
            "3",
            "--obstacle-count",
            "19",
            "--output",
            str(tmp_path / "report"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert return_code == 0
    assert captured["config"].methods == ("plcbf",)
    assert captured["config"].scenarios == ("head_on",)
    assert captured["config"].seeds == (4,)
    assert captured["config"].max_steps == 3
    assert captured["config"].playground_obstacle_count == 19
    assert captured["output"] == tmp_path / "report"
    assert payload["trials"] == 1


def test_cli_default_suite_and_quick_scenario_reduction() -> None:
    parser = benchmark.build_parser()
    normal = benchmark._config_from_args(parser.parse_args([]))
    quick = benchmark._config_from_args(parser.parse_args(["--quick"]))
    explicit = benchmark._config_from_args(
        parser.parse_args(["--quick", "--scenario", "head_on"])
    )

    assert normal.scenarios == benchmark.DEFAULT_BENCHMARK_SCENARIOS
    assert normal.scenarios == (PLAYGROUND_STRESS_SCENARIO,)
    assert normal.playground_obstacle_count == PLAYGROUND_STRESS_OBSTACLE_COUNT
    assert quick.scenarios == (PLAYGROUND_CROWDED_SCENARIO,)
    assert quick.playground_obstacle_count == PLAYGROUND_OBSTACLE_COUNT
    assert explicit.scenarios == ("head_on",)


def test_mixed_generated_scenarios_require_an_explicit_shared_count() -> None:
    with pytest.raises(ValueError, match="mixed crowded/stress"):
        benchmark.NLQuad3DBenchmarkConfig(
            methods=("plcbf",),
            scenarios=(
                PLAYGROUND_CROWDED_SCENARIO,
                PLAYGROUND_STRESS_SCENARIO,
            ),
        )


def test_tuned_controller_is_scoped_to_plcbf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_config = NLQuad3DControllerConfig(cbf_alpha=1.25)
    tuned_config = NLQuad3DControllerConfig(cbf_alpha=6.5)
    observed: dict[str, NLQuad3DControllerConfig] = {}

    def fake_run_trial(method, _scenario, **kwargs):
        observed[method] = kwargs["controller_config"]
        return BenchmarkResult(
            algorithm=method,
            case_id=f"head_on/{method}",
            seed=0,
            outcome="success",
        )

    monkeypatch.setattr(benchmark, "run_trial", fake_run_trial)
    config = benchmark.NLQuad3DBenchmarkConfig(
        methods=("plcbf", "backup_cbf"),
        scenarios=("head_on",),
        controller_config=baseline_config,
        plcbf_controller_config=tuned_config,
    )

    benchmark.run_benchmark(config)

    assert observed["plcbf"] == tuned_config
    assert observed["backup_cbf"] == baseline_config


def test_real_nonlinear_oracle_one_step_smoke() -> None:
    config = NLQuad3DControllerConfig(
        backup_horizon=0.1,
        max_obstacles=2,
        num_radial_policies=2,
        nominal_prefix_steps=0,
    )
    result = benchmark.run_trial(
        "plcbf",
        "playground_corridor",
        seed=0,
        max_steps=1,
        controller_config=config,
        warmup=False,
    )

    assert result.outcome is not BenchmarkOutcome.ERROR
    assert len(result.solve_times_s) == 1
    assert result.case_metrics["control_steps"] == 1
    assert result.case_metrics["mean_certificate_count"] == 4.0
