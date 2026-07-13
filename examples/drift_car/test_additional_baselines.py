"""Focused fairness/regression tests for the added drift-car baselines."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from safe_control.position_control.backup_cbf_qp import BackupCBF
from safe_control.robots.drifting_car import DriftingCar

from examples.drift_car import benchmark_black_ice as benchmark
from examples.drift_car.algorithms.library_pcbf_mi_drift import (
    LibraryPCBFMinInterventionDrift,
)
from examples.drift_car.algorithms.multi_backup_cbf_mi_drift import (
    DriftSampledStoppingTerminalEnvelope,
    MultiBackupCBFMinInterventionDrift,
)
from examples.drift_car.algorithms.multi_policy_baseline_common_drift import (
    CandidateCBFResult,
    assert_runtime_library_equal,
    runtime_library_signature,
    select_minimum_intervention,
)
from examples.drift_car.controllers.drift_policies_jax import (
    LaneChangeControllerJAX,
    StoppingControllerJAX,
)


def _system(backup_horizon: float = 0.1):
    cfg = replace(
        benchmark.SimConfig(),
        backup_horizon_time=backup_horizon,
        tf=0.1,
    )
    env, lanes = benchmark.setup_env_and_lanes(cfg)
    scenario = benchmark.Scenario(
        run_idx=0,
        seed=11,
        num_obstacles=1,
        obstacles=((80.0, "middle"),),
    )
    benchmark.add_black_ice_and_obstacles(env, lanes, cfg, scenario)
    spec = benchmark.build_vehicle_spec()
    state = benchmark.make_initial_state(lanes, cfg)
    car = DriftingCar(state, spec, cfg.dt, ax=None)
    reference = benchmark.create_reference_plcbf(car, env, lanes, cfg)
    states = np.repeat(state.reshape(1, -1), 10, axis=0)
    controls = np.zeros((3, 2), dtype=float)
    return cfg, env, lanes, car, reference, state, states, controls


def _library_baseline():
    cfg, env, lanes, car, reference, state, states, controls = _system()
    controller = LibraryPCBFMinInterventionDrift(
        robot=car,
        robot_spec=car.robot_spec,
        dt=cfg.dt,
        backup_horizon=cfg.backup_horizon_time,
        cbf_alpha=6.0,
        left_lane_y=lanes["left"],
        right_lane_y=lanes["right"],
        safety_margin=1.15,
        reference_plcbf=reference,
    )
    controller.set_environment(env)
    return controller, reference, state, states, controls


def _multi_backup_baseline(
    backup_horizon: float = 0.1,
    maneuver_prefix_time: float = 1.0,
):
    cfg, env, _, car, reference, state, states, controls = _system(backup_horizon)
    controller = MultiBackupCBFMinInterventionDrift(
        robot=car,
        robot_spec=car.robot_spec,
        dt=cfg.dt,
        backup_horizon=cfg.backup_horizon_time,
        reference_plcbf=reference,
        maneuver_prefix_time=maneuver_prefix_time,
    )
    controller.set_environment(env)
    controller.set_nominal_trajectory(states, controls)
    return controller, reference, state, states, controls


def _candidate(
    name: str,
    objective: float,
    u,
    feasible: bool = True,
    *,
    rollout_safe: bool | None = None,
    terminal_safe: bool | None = None,
    qp_solved: bool = True,
    solver_status: str | None = None,
):
    return CandidateCBFResult(
        policy_name=name,
        feasible=feasible,
        u=None if u is None else np.asarray(u, dtype=float),
        objective=objective,
        solver_status=(
            solver_status
            if solver_status is not None
            else ("optimal" if feasible else "infeasible")
        ),
        rollout_safe=feasible if rollout_safe is None else rollout_safe,
        terminal_safe=feasible if terminal_safe is None else terminal_safe,
        solve_time_sec=0.0,
        qp_solved=qp_solved,
    )


def test_default_registry_is_unchanged_and_new_keys_are_explicit():
    historical = benchmark.make_variants()
    assert len(historical) == 13
    assert all(
        variant.key not in {"multi_backup_cbf_mi", "library_pcbf_mi"}
        for variant in historical
    )
    additional = benchmark.make_variants(include_additional=True)
    assert {variant.key for variant in additional} >= {
        "multi_backup_cbf_mi",
        "library_pcbf_mi",
    }


def test_runtime_library_equality_is_exact_and_fails_loudly():
    library, reference, *_ = _library_baseline()
    multi_backup, _, *_ = _multi_backup_baseline()
    assert runtime_library_signature(library) == runtime_library_signature(reference)
    assert runtime_library_signature(multi_backup) == runtime_library_signature(reference)

    library.policy_configs["stop"]["params"] = library.policy_configs["stop"][
        "params"
    ]._replace(Kp_v=999.0)
    with pytest.raises(AssertionError):
        assert_runtime_library_equal(library, reference)


def test_multi_backup_uses_exact_maneuver_prefix_and_shared_stop_tail():
    controller, _, state, states, _ = _multi_backup_baseline(
        backup_horizon=0.3,
        maneuver_prefix_time=0.1,
    )
    frozen_controls = np.array(
        [
            [0.11, 101.0],
            [-0.12, -202.0],
            [0.13, 303.0],
            [-0.14, -404.0],
            [0.15, 505.0],
            [-0.16, -606.0],
        ],
        dtype=float,
    )
    controller.set_nominal_trajectory(states, frozen_controls)

    assert controller.total_rollout_steps == 6
    assert controller.maneuver_prefix_steps == 2
    assert controller.terminal_tail_steps == 4

    stop_expected = np.asarray(
        StoppingControllerJAX.compute(
            jnp.asarray(state), controller.policy_configs["stop"]["params"]
        ),
        dtype=float,
    )
    for name in ("lane_change_left", "lane_change_right"):
        adapter = controller.filters[name].policy_adapter
        maneuver_expected = np.asarray(
            LaneChangeControllerJAX.compute(
                jnp.asarray(state), controller.policy_configs[name]["params"]
            ),
            dtype=float,
        )
        for step in range(controller.maneuver_prefix_steps):
            np.testing.assert_array_equal(
                adapter.control_at_step(state, step), maneuver_expected
            )
        for step in range(
            controller.maneuver_prefix_steps, controller.total_rollout_steps
        ):
            np.testing.assert_array_equal(adapter.control_at_step(state, step), stop_expected)

    stop_adapter = controller.filters["stop"].policy_adapter
    for step in range(controller.total_rollout_steps):
        np.testing.assert_array_equal(
            stop_adapter.control_at_step(state, step), stop_expected
        )

    nominal_adapter = controller.filters["nominal"].policy_adapter
    for step in range(controller.maneuver_prefix_steps):
        np.testing.assert_array_equal(
            nominal_adapter.control_at_step(state, step), frozen_controls[step]
        )
    for step in range(
        controller.maneuver_prefix_steps, controller.total_rollout_steps
    ):
        np.testing.assert_array_equal(
            nominal_adapter.control_at_step(state, step), stop_expected
        )

    compound_status = controller.get_status()["compound_strategy"]
    assert compound_status == {
        "maneuver_prefix_steps": 2,
        "maneuver_prefix_time": pytest.approx(0.1),
        "terminal_tail_steps": 4,
        "terminal_tail_time": pytest.approx(0.2),
        "tail_policy": "stop",
        "hard_switch_to_exact_stop": True,
    }


@pytest.mark.parametrize("friction", [1.0, 0.3])
def test_default_compound_prefix_preserves_diversity_and_fits_sampled_envelope(
    friction,
):
    controller, _, state, states, _ = _multi_backup_baseline(
        backup_horizon=3.0,
        maneuver_prefix_time=1.0,
    )
    controller.set_friction(friction)
    controller.set_nominal_trajectory(states, np.zeros((60, 2), dtype=float))

    terminal_states = {}
    for name, candidate in controller.filters.items():
        rollout = candidate._integrate_policy_trajectory(state)
        terminal_states[name] = rollout[-1]
        status = candidate._evaluate_terminal_envelope(rollout[-1])
        assert status["satisfied"] is True, (name, friction, status)
        assert status["components"]["successor_abs_speed_nonincrease"] >= 0.0

    assert terminal_states["lane_change_left"][1] - state[1] > 5.0
    assert terminal_states["lane_change_right"][1] - state[1] < -5.0
    assert abs(terminal_states["stop"][1] - state[1]) < 1e-12
    assert abs(terminal_states["nominal"][1] - state[1]) < 1e-12


def _settled_stopping_state(state: np.ndarray) -> np.ndarray:
    value = np.array(state, dtype=float, copy=True)
    value[[3, 4, 5, 6, 7]] = 0.0
    return value


def test_multi_backup_terminal_envelope_is_named_auditable_sampled_proxy():
    controller, _, state, _, _ = _multi_backup_baseline()
    candidate = controller.filters["stop"]
    assert isinstance(candidate.terminal_envelope, DriftSampledStoppingTerminalEnvelope)
    state = _settled_stopping_state(state)
    status = candidate._evaluate_terminal_envelope(state, record=True)

    assert status["name"] == "drift_sampled_stop_tail_envelope_v1"
    assert status["satisfied"] is True
    assert status["is_formal_invariant_proof"] is False
    assert "sampled stopping-envelope proxy" in status["description"]
    assert status["tolerances"] == {
        "name": "drift_sampled_stop_tail_envelope_v1",
        "longitudinal_speed_abs_max": pytest.approx(7.5),
        "yaw_rate_abs_max": pytest.approx(0.05),
        "sideslip_abs_max": pytest.approx(0.05),
        "steering_abs_max": pytest.approx(np.deg2rad(5.0)),
        "stop_successor_abs_speed_increase_max": pytest.approx(1e-6),
        "safety_margin_min": pytest.approx(0.0),
    }
    assert status["components"]["terminal_safety"] >= 0.0
    assert status["components"]["successor_safety"] >= 0.0
    assert {
        "terminal_longitudinal_speed",
        "terminal_yaw_rate",
        "terminal_sideslip",
        "terminal_steering",
        "successor_longitudinal_speed",
        "successor_yaw_rate",
        "successor_sideslip",
        "successor_steering",
        "successor_abs_speed_nonincrease",
    } <= status["components"].keys()

    exposed = controller.get_status()
    assert exposed["terminal_envelope"]["is_formal_invariant_proof"] is False
    assert exposed["candidate_terminal_status"]["stop"] == status


@pytest.mark.parametrize(
    ("state_index", "tolerance_name", "component_name"),
    [
        (5, "longitudinal_speed_abs_max", "terminal_longitudinal_speed"),
        (3, "yaw_rate_abs_max", "terminal_yaw_rate"),
        (4, "sideslip_abs_max", "terminal_sideslip"),
        (6, "steering_abs_max", "terminal_steering"),
    ],
)
def test_multi_backup_terminal_envelope_rejects_each_state_bound_violation(
    state_index, tolerance_name, component_name
):
    controller, _, state, _, _ = _multi_backup_baseline()
    candidate = controller.filters["stop"]
    state = _settled_stopping_state(state)
    tolerance = getattr(candidate.terminal_envelope, tolerance_name)
    state[state_index] = tolerance + 1e-3

    status = candidate._evaluate_terminal_envelope(state)
    assert status["satisfied"] is False
    assert status["components"][component_name] < 0.0


def test_multi_backup_terminal_envelope_checks_one_exact_stop_successor(monkeypatch):
    controller, _, state, _, _ = _multi_backup_baseline()
    candidate = controller.filters["stop"]
    state = _settled_stopping_state(state)
    successor = state.copy()
    successor[5] = candidate.terminal_envelope.longitudinal_speed_abs_max + 0.1
    monkeypatch.setattr(candidate, "_stop_successor", lambda unused: successor.copy())

    status = candidate._evaluate_terminal_envelope(state)
    assert status["components"]["terminal_longitudinal_speed"] > 0.0
    assert status["components"]["successor_longitudinal_speed"] < 0.0
    assert status["satisfied"] is False


def test_multi_backup_terminal_envelope_requires_stop_successor_speed_nonincrease(
    monkeypatch,
):
    controller, _, state, _, _ = _multi_backup_baseline()
    candidate = controller.filters["stop"]
    state = _settled_stopping_state(state)
    state[5] = 1.0
    successor = state.copy()
    successor[5] = 1.1
    monkeypatch.setattr(candidate, "_stop_successor", lambda unused: successor.copy())

    status = candidate._evaluate_terminal_envelope(state)
    assert status["components"]["successor_longitudinal_speed"] > 0.0
    assert status["components"]["successor_abs_speed_nonincrease"] < 0.0
    assert status["satisfied"] is False


def test_minimum_intervention_selector_and_fixed_tie_break():
    names = ("lane_change_left", "lane_change_right", "stop", "nominal")
    results = [
        _candidate("lane_change_left", 2.0, [0.1, 0.0]),
        _candidate("lane_change_right", 0.5, [0.0, 1.0]),
        _candidate("stop", 1.0, [0.0, -1.0]),
        _candidate("nominal", 4.0, [0.0, 0.0]),
    ]
    assert select_minimum_intervention(results, names).policy_name == "lane_change_right"

    tied = [
        _candidate("stop", 1.0 + 0.5e-5, [0.0, -1.0]),
        _candidate("lane_change_left", 1.0, [0.1, 0.0]),
    ]
    assert select_minimum_intervention(tied, names).policy_name == "lane_change_left"


def test_library_pcbf_reuses_exact_values_gradients_and_trajectories():
    controller, reference, state, states, controls = _library_baseline()
    for candidate in (controller, reference):
        candidate.set_nominal_trajectory(states, controls)
        candidate._update_obstacles()

    values_new, gradients_new, trajectories_new = controller._compute_multi_value_and_grad(
        np.asarray(state)
    )
    values_reference, gradients_reference, trajectories_reference = (
        reference._compute_multi_value_and_grad(np.asarray(state))
    )
    assert values_new.keys() == values_reference.keys()
    for name in values_new:
        assert values_new[name] == pytest.approx(values_reference[name], abs=1e-9)
        np.testing.assert_allclose(gradients_new[name], gradients_reference[name], atol=1e-9)
        np.testing.assert_allclose(trajectories_new[name], trajectories_reference[name], atol=1e-9)


def test_library_selector_uses_realized_candidate_objective(monkeypatch):
    controller, _, state, states, controls = _library_baseline()
    values = {name: 1.0 for name in controller.policy_names}
    gradients = {name: np.zeros(8) for name in controller.policy_names}
    trajectories = {name: states.copy() for name in controller.policy_names}
    monkeypatch.setattr(
        controller,
        "_compute_multi_value_and_grad",
        lambda unused: (values, gradients, trajectories),
    )
    objectives = {
        "lane_change_left": 1.0,
        "lane_change_right": 0.25,
        "stop": 2.0,
        "nominal": 0.5,
    }
    outputs = {
        "lane_change_left": np.array([0.1, 0.0]),
        "lane_change_right": np.array([0.0, 1.0]),
        "stop": np.array([0.0, -1.0]),
        "nominal": np.array([0.0, 0.5]),
    }
    monkeypatch.setattr(
        controller,
        "_solve_policy_candidate",
        lambda name, *args: _candidate(name, objectives[name], outputs[name]),
    )
    result = controller.solve_control_problem(
        state,
        control_ref={"u_ref": np.zeros(2)},
        nominal_trajectory=states,
        nominal_controls=controls,
    )
    assert controller.best_policy_name == "lane_change_right"
    np.testing.assert_allclose(result.reshape(-1), outputs["lane_change_right"])


@pytest.mark.parametrize("kind", ["library", "multi_backup"])
def test_no_candidate_sets_explicit_failure_and_returns_bounded_emergency(
    monkeypatch, kind
):
    if kind == "library":
        controller, _, state, states, controls = _library_baseline()
        values = {name: -1.0 for name in controller.policy_names}
        gradients = {name: np.zeros(8) for name in controller.policy_names}
        trajectories = {name: states.copy() for name in controller.policy_names}
        monkeypatch.setattr(
            controller,
            "_compute_multi_value_and_grad",
            lambda unused: (values, gradients, trajectories),
        )
    else:
        controller, _, state, states, controls = _multi_backup_baseline()
        for name, candidate_filter in controller.filters.items():
            monkeypatch.setattr(
                candidate_filter,
                "solve_candidate",
                lambda unused_state, unused_u, name=name: _candidate(
                    name, float("inf"), None, feasible=False
                ),
            )

    result = controller.solve_control_problem(
        state,
        control_ref={"u_ref": np.zeros(2)},
        nominal_trajectory=states,
        nominal_controls=controls,
    )
    status = controller.get_status()
    assert status["certificate_lost"] is True
    if kind == "multi_backup":
        assert status["infeasible"] is False
        assert status["qp_infeasible"] is False
        assert status["fallback_applied"] is True
    else:
        assert status["infeasible"] is False
        assert status["qp_infeasible"] is False
        assert status["fallback_applied"] is True
    assert controller.get_metrics()["num_steps_with_no_safe_policy"] == 1
    assert np.all(np.isfinite(result))
    assert np.all(result.reshape(-1) >= controller.u_min - 1e-5)
    assert np.all(result.reshape(-1) <= controller.u_max + 1e-5)


def test_multi_backup_candidate_exception_is_runtime_error_not_certificate_loss(
    monkeypatch,
):
    controller, _, state, states, controls = _multi_backup_baseline()
    for name, candidate_filter in controller.filters.items():
        monkeypatch.setattr(
            candidate_filter,
            "solve_candidate",
            lambda unused_state, unused_u, name=name: _candidate(
                name,
                float("inf"),
                None,
                feasible=False,
                rollout_safe=None,
                terminal_safe=None,
                qp_solved=False,
                solver_status="error",
            ),
        )

    with pytest.raises(RuntimeError, match="certificate status is unknown"):
        controller.solve_control_problem(
            state,
            control_ref={"u_ref": np.zeros(2)},
            nominal_trajectory=states,
            nominal_controls=controls,
        )
    status = controller.get_status()
    assert status["runtime_error"] is True
    assert status["certificate_lost"] is False
    assert status["qp_infeasible"] is False
    assert controller.get_metrics()["candidate_evaluation_error_count"] == 4


@pytest.mark.parametrize(
    ("rollout_safe", "terminal_safe"),
    [(False, True), (True, False)],
)
def test_multi_backup_rejects_uncertified_rollout_before_qp_construction(
    monkeypatch, rollout_safe, terminal_safe
):
    controller, _, state, _, _ = _multi_backup_baseline()
    candidate_filter = controller.filters["stop"]
    phi = np.repeat(state.reshape(1, -1), candidate_filter.N, axis=0)
    monkeypatch.setattr(
        candidate_filter,
        "_integrate_policy_trajectory",
        lambda unused: phi,
    )
    monkeypatch.setattr(
        candidate_filter,
        "_compute_rollout_sensitivities",
        lambda unused: (_ for _ in ()).throw(
            AssertionError("sensitivities must not be built for an uncertified policy")
        ),
    )
    monkeypatch.setattr(
        candidate_filter,
        "_h_safety",
        lambda unused_state, unused_time=0.0: 1.0 if rollout_safe else -1.0,
    )
    terminal_value = 1.0 if terminal_safe else -1.0

    def terminal_status(unused_state, *, record=False):
        status = {"value": terminal_value, "satisfied": terminal_safe}
        if record:
            candidate_filter.last_terminal_envelope_status = status.copy()
        return status

    monkeypatch.setattr(candidate_filter, "_evaluate_terminal_envelope", terminal_status)
    dynamics_calls = []

    def forbidden_dynamics(unused_state):
        dynamics_calls.append(True)
        raise AssertionError("system matrices must not be built for an uncertified policy")

    monkeypatch.setattr(candidate_filter, "_dynamics_f", forbidden_dynamics)
    result = candidate_filter.solve_candidate(state, np.zeros(2))
    assert result.feasible is False
    assert result.qp_solved is False
    assert result.solver_status == "uncertified_rollout"
    assert result.rollout_safe is rollout_safe
    assert result.terminal_safe is terminal_safe
    assert dynamics_calls == []


def test_multi_backup_distinguishes_certified_qp_failure_from_certificate_loss(
    monkeypatch,
):
    controller, _, state, states, controls = _multi_backup_baseline()
    for name, candidate_filter in controller.filters.items():
        monkeypatch.setattr(
            candidate_filter,
            "solve_candidate",
            lambda unused_state, unused_u, name=name: _candidate(
                name,
                float("inf"),
                None,
                feasible=False,
                rollout_safe=True,
                terminal_safe=True,
                qp_solved=True,
            ),
        )

    result = controller.solve_control_problem(
        state,
        control_ref={"u_ref": np.zeros(2)},
        nominal_trajectory=states,
        nominal_controls=controls,
    )
    status = controller.get_status()
    assert status["certificate_lost"] is False
    assert status["qp_infeasible"] is True
    assert status["infeasible"] is True
    assert status["fallback_applied"] is True
    assert controller.get_metrics()["qp_infeasible_count"] == 1
    assert controller.get_metrics()["candidate_qp_failure_count"] == 4
    assert np.all(np.isfinite(result))


def test_multi_backup_candidate_order_cannot_mutate_frozen_context(monkeypatch):
    controller, _, state, _, _ = _multi_backup_baseline()
    nominal = np.array([0.05, 100.0])
    seen = []
    objectives = {name: float(index + 1) for index, name in enumerate(controller.policy_names)}

    for name, candidate_filter in controller.filters.items():
        def fake(candidate_state, candidate_nominal, name=name):
            seen.append((name, candidate_state.copy(), candidate_nominal.copy()))
            candidate_state[:] = -999.0
            candidate_nominal[:] = -999.0
            return _candidate(name, objectives[name], [0.0, 0.0])

        monkeypatch.setattr(candidate_filter, "solve_candidate", fake)

    forward = controller.evaluate_candidates(state, nominal, controller.policy_names)
    reverse = controller.evaluate_candidates(state, nominal, tuple(reversed(controller.policy_names)))
    for _, candidate_state, candidate_nominal in seen:
        np.testing.assert_allclose(candidate_state, state)
        np.testing.assert_allclose(candidate_nominal, nominal)
    assert select_minimum_intervention(forward, controller.policy_names).policy_name == (
        select_minimum_intervention(reverse, controller.policy_names).policy_name
    )


def test_single_policy_backup_candidate_matches_existing_backup_cbf():
    controller, _, state, _, _ = _multi_backup_baseline()
    strict = controller.filters["stop"]
    state = np.array(state, copy=True)
    state[[3, 4, 5, 6, 7]] = 0.0
    legacy = BackupCBF(
        robot=controller.robot,
        robot_spec=controller.robot_spec,
        dt=controller.dt,
        backup_horizon=controller.backup_horizon,
    )
    legacy.set_backup_controller(strict.policy_adapter, target=strict.backup_target)
    legacy.set_environment(controller.env)
    nominal = np.zeros((2, 2), dtype=float)
    legacy.set_nominal_trajectory(np.repeat(state.reshape(1, -1), 3, axis=0), nominal)

    strict.set_environment(controller.env)
    candidate = strict.solve_candidate(state, np.zeros(2))
    legacy_output = legacy.solve_control_problem(state).reshape(-1)
    assert candidate.feasible
    np.testing.assert_allclose(candidate.u, legacy_output, atol=2e-4, rtol=2e-4)
    legacy_rollout, _ = legacy._integrate_backup_trajectory(state)
    # The strict wrapper deliberately adds the terminal t=T sample locally so
    # it is reproducible from the clean safe_control gitlink, whose historical
    # BackupCBF rollout ends one sample earlier.  The shared rollout prefix
    # must remain numerically identical.
    assert strict.N == int(np.ceil(strict.backup_horizon / strict.dt)) + 1
    np.testing.assert_allclose(
        strict.latest_backup_trajectory[: len(legacy_rollout)],
        legacy_rollout,
        atol=1e-9,
        rtol=1e-9,
    )
    assert len(strict.latest_backup_trajectory) >= len(legacy_rollout)


def test_single_policy_library_qp_matches_plcbf_qp():
    controller, reference, state, states, controls = _library_baseline()
    for candidate in (controller, reference):
        candidate.set_nominal_trajectory(states, controls)
        candidate._update_obstacles()
    values, gradients, _ = controller._compute_multi_value_and_grad(np.asarray(state))
    state_jax = np.asarray(state)
    f = np.asarray(controller.dynamics_jax.f_full(state_jax, controller.current_friction))
    G = np.asarray(controller.dynamics_jax.g_full(state_jax))
    name = "lane_change_left"
    nominal = np.zeros(2)
    result = controller._solve_policy_candidate(
        name, nominal, values[name], gradients[name], f, G
    )

    reference.cbf_alpha = controller._policy_alpha(name)
    gradient = np.array(gradients[name], copy=True)
    gradient_norm = np.linalg.norm(gradient)
    if gradient_norm > 50.0:
        gradient *= 50.0 / gradient_norm
    expected = reference._solve_cbf_qp(
        nominal, values[name], gradient, f, G
    ).reshape(-1)
    assert result.feasible
    np.testing.assert_allclose(result.u, expected, atol=2e-4, rtol=2e-4)


def test_explicit_certificate_loss_increments_benchmark_failure(monkeypatch):
    variant = next(
        variant
        for variant in benchmark.make_variants(include_additional=True)
        if variant.key == "library_pcbf_mi"
    )
    scenario = benchmark.Scenario(0, 11, 1, ((80.0, "middle"),))
    failure = SimpleNamespace(
        collision=False,
        infeasible=True,
        certificate_lost=True,
        qp_infeasible=False,
        runtime_error=False,
        task_completed=True,
        survived_horizon=True,
        completed_or_survived=True,
        filter_failure=True,
        union_failure=True,
        nominal_tracking_pct=0.0,
        mean_compute_ms=1.0,
        timed_steps=1,
    )
    monkeypatch.setattr(benchmark, "run_episode", lambda *args, **kwargs: failure)
    summary, _ = benchmark.aggregate_results(
        [variant], [scenario], benchmark.SimConfig()
    )
    assert summary[0]["fail_count"] == 1
    assert summary[0]["infeasible_count"] == 1
    assert summary[0]["certificate_lost_count"] == 1


def test_benchmark_does_not_double_count_qp_failure_without_certificate():
    variant = benchmark.AlgoVariant("plcbf", "PLCBF", "plcbf", None)
    status = {
        "status": "qp_failed_after_certificate_loss",
        "certificate_lost": True,
        "qp_infeasible": True,
        "fallback_applied": True,
    }
    assert benchmark.classify_comparison_status(variant, object(), status) == (
        True,
        False,
        False,
        True,
    )


@pytest.mark.parametrize("event_kind", ["certificate_lost", "qp_infeasible"])
def test_benchmark_continues_after_filter_failure_with_bounded_fallback(
    monkeypatch, event_kind
):
    cfg = replace(benchmark.SimConfig(), tf=0.2, dt=0.05)
    n_steps = int(cfg.tf / cfg.dt)

    class StubEnv:
        track_length = cfg.track_length

        @staticmethod
        def get_friction_at_position(unused_pos, default_friction):
            return default_friction

    class StubCar:
        def __init__(self):
            self.state = np.zeros((8, 1), dtype=float)
            self.state[5, 0] = cfg.initial_velocity
            self.friction = 1.0

        def get_state(self):
            return self.state.copy()

        def get_position(self):
            return self.state[:2, 0].copy()

        def get_friction(self):
            return self.friction

        def set_friction(self, value):
            self.friction = float(value)

    class StubSimulator:
        def __init__(self, car):
            self.car = car
            self.calls = 0

        def step(self, control):
            assert np.all(np.isfinite(control))
            self.calls += 1
            self.car.state[0, 0] += 0.1
            return {"collision": False}

    class StubMPCC:
        @staticmethod
        def solve_control_problem(unused_state):
            return np.zeros((2, 1), dtype=float)

        @staticmethod
        def get_full_predictions():
            return None, None

    class StubShield:
        def __init__(self):
            self.calls = 0
            self.u_min = np.array([-1.0, -8000.0], dtype=float)
            self.u_max = np.array([1.0, 8000.0], dtype=float)

        def solve_control_problem(self, *args, **kwargs):
            self.calls += 1
            return np.zeros((2, 1), dtype=float)

        def get_status(self):
            is_certificate_loss = event_kind == "certificate_lost"
            return {
                "status": (
                    "certificate_lost_no_backup_candidate"
                    if is_certificate_loss
                    else "qp_infeasible_no_backup_candidate"
                ),
                "certificate_lost": is_certificate_loss,
                "qp_infeasible": not is_certificate_loss,
                "infeasible": not is_certificate_loss,
                "fallback_applied": True,
                "best_policy": None,
            }

        @staticmethod
        def get_metrics():
            return {}

        @staticmethod
        def _emergency_control(unused_state):
            return np.zeros(2, dtype=float)

    env = StubEnv()
    car = StubCar()
    simulator = StubSimulator(car)
    mpcc = StubMPCC()
    shield = StubShield()
    monkeypatch.setattr(benchmark, "setup_env_and_lanes", lambda unused: (env, {"middle": 0.0}))
    monkeypatch.setattr(benchmark, "add_black_ice_and_obstacles", lambda *args: None)
    monkeypatch.setattr(benchmark, "make_initial_state", lambda *args: car.state.copy())
    monkeypatch.setattr(benchmark, "DriftingCar", lambda *args, **kwargs: car)
    monkeypatch.setattr(
        benchmark,
        "DriftingCarSimulator",
        lambda *args, **kwargs: simulator,
    )
    monkeypatch.setattr(benchmark, "setup_mpcc", lambda *args: mpcc)
    monkeypatch.setattr(benchmark, "setup_shielding", lambda *args: shield)

    variant = next(
        item
        for item in benchmark.make_variants(include_additional=True)
        if item.key == "multi_backup_cbf_mi"
    )
    scenario = benchmark.Scenario(0, 7, 1, ((80.0, "middle"),))
    result = benchmark.run_episode(variant, scenario, cfg)
    assert shield.calls == n_steps
    assert simulator.calls == n_steps
    assert result.total_steps == n_steps
    assert result.certificate_lost is (event_kind == "certificate_lost")
    assert result.certificate_loss_steps == (
        n_steps if event_kind == "certificate_lost" else 0
    )
    assert result.qp_infeasible is (event_kind == "qp_infeasible")
    assert result.qp_infeasible_steps == (
        n_steps if event_kind == "qp_infeasible" else 0
    )
    assert result.collision is False
    assert result.survived_horizon is True
    assert result.task_completed is False
    assert result.completed_or_survived is True
    assert result.filter_failure is True
    assert result.union_failure is True
