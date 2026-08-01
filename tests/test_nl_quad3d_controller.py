from __future__ import annotations

import sys
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np

from examples.nl_quad3d.controller import (
    NLQuad3DControllerConfig,
    PLCBF_NLQuad3D,
    RolloutEvaluation,
    _policy_value_with_aux,
    _shared_compiled_evaluator,
)
from examples.nl_quad3d.dynamics import NLQuad3D, make_state
from examples.nl_quad3d.dynamics_jax import jax_params
from examples.nl_quad3d.policies import (
    POLICY_RADIAL,
    PolicyCandidate,
    fibonacci_directions,
)
from examples.nl_quad3d.scenarios import (
    NLQuad3DScenario,
    WorldBounds,
    advance_obstacles,
    get_scenario,
)
from examples.nl_quad3d.simulation import simulate
from plcbf.policy_library import (
    CBFHalfspace,
    PolicyCertificate,
    solve_box_halfspace_qp,
)
from plcbf.baselines import solve_library_pcbf_mi


def _small_controller(
    *,
    candidate_provider=None,
    max_operator: str = "value",
) -> PLCBF_NLQuad3D:
    model = NLQuad3D()
    config = NLQuad3DControllerConfig(
        dt=model.dt,
        backup_horizon=0.3,
        num_radial_policies=3,
        max_obstacles=2,
        sensing_radius=20.0,
        max_operator=max_operator,
    )
    return PLCBF_NLQuad3D(
        model,
        config,
        bounds=None,
        candidate_provider=candidate_provider,
    )


def test_fibonacci_library_is_full_3d_and_includes_stop_nominal() -> None:
    directions = fibonacci_directions(24)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.min(directions[:, 2]) < -0.8
    assert np.max(directions[:, 2]) > 0.8
    controller = _small_controller()
    candidates = controller.candidates(np.array([4.0, 0.0, 0.0]))
    assert [candidate.kind for candidate in candidates][-2:] == ["stop", "nominal"]
    assert sum(candidate.kind == "radial" for candidate in candidates) == 3


def test_controller_defaults_match_quad3d_playground() -> None:
    model = NLQuad3D()
    config = NLQuad3DControllerConfig()

    assert config.sensing_radius == 3.8
    assert config.backup_horizon == 1.25
    assert config.cbf_alpha == 2.2
    assert config.cbf_value_buffer == 0.2
    assert config.safety_margin == 0.18
    assert config.safety_scale == 1.15
    assert config.max_obstacles == 8
    assert config.num_radial_policies == 12
    assert config.target_speed == 2.8
    assert config.radial_gain == 2.6
    assert config.stop_gain == 3.0
    assert model.config.nominal_k_v == 0.65
    assert model.config.nominal_k_a == 1.70
    assert model.config.nominal_k_att == 12.5
    assert model.config.nominal_k_rate == 7.5


def test_bouncing_spherical_obstacles_remain_inside_world() -> None:
    bounds = WorldBounds((0.0, 0.0, 0.0), (4.0, 5.0, 6.0))
    obstacles = np.array([[3.7, 0.3, 5.7, 0.25, 2.0, -2.0, 3.0]])
    advanced = advance_obstacles(obstacles, 0.5, bounds)
    assert np.all(advanced[0, :3] >= np.asarray(bounds.lower) + 0.25)
    assert np.all(advanced[0, :3] <= np.asarray(bounds.upper) - 0.25)
    assert advanced[0, 4] < 0.0
    assert advanced[0, 5] > 0.0
    assert advanced[0, 6] < 0.0


def test_batched_rollout_values_have_correct_jax_gradients() -> None:
    def provider(**_):
        return (
            PolicyCandidate("right", "radial", (0.0, 1.0, 0.0), 1.0, 2.0),
            PolicyCandidate("stop", "stop", gain=2.0),
            PolicyCandidate("nominal", "nominal"),
        )

    controller = _small_controller(candidate_provider=provider)
    state = make_state([0.0, 0.0, 1.0], [0.3, 0.0, 0.0])
    goal = np.array([5.0, 0.0, 1.0])
    obstacles = np.array([[2.2, 0.3, 1.25, 0.35, -0.2, 0.0, 0.0]])
    evaluation = controller.evaluate_policies(state, goal, obstacles)
    assert evaluation.values.shape == (3,)
    assert evaluation.state_gradients.shape == (3, 12)
    assert evaluation.obstacle_gradients.shape == (3, 1, 7)
    assert evaluation.trajectories.shape == (3, 7, 12)
    assert np.all(np.isfinite(evaluation.state_gradients))

    epsilon = 1e-3
    plus = state.copy()
    minus = state.copy()
    plus[1] += epsilon
    minus[1] -= epsilon
    value_plus = controller.evaluate_policies(plus, goal, obstacles).values
    value_minus = controller.evaluate_policies(minus, goal, obstacles).values
    finite_difference = (value_plus - value_minus) / (2.0 * epsilon)
    np.testing.assert_allclose(
        evaluation.state_gradients[:, 1],
        finite_difference,
        rtol=2e-2,
        atol=2e-2,
    )


def test_padded_obstacle_mask_matches_all_active_reference_oracle() -> None:
    """Static padding must not change the certificate used by either QP."""

    model = NLQuad3D()
    common = dict(
        dt=model.dt,
        backup_horizon=0.1,
        num_radial_policies=2,
        sensing_radius=20.0,
        max_operator="input_space",
    )
    padded = PLCBF_NLQuad3D(
        model,
        NLQuad3DControllerConfig(max_obstacles=8, **common),
        bounds=None,
    )
    all_active = PLCBF_NLQuad3D(
        model,
        NLQuad3DControllerConfig(max_obstacles=3, **common),
        bounds=None,
    )
    state = make_state([0.0, 0.0, 1.0], [0.2, -0.1, 0.05])
    goal = np.asarray([6.0, 0.5, 1.5])
    obstacles = np.asarray(
        [
            [1.2, 0.1, 1.1, 0.25, -0.2, 0.1, 0.0],
            [1.5, -0.4, 1.4, 0.30, 0.0, -0.1, 0.1],
            [1.8, 0.5, 0.8, 0.20, -0.1, 0.0, -0.1],
        ]
    )

    padded_evaluation = padded.evaluate_policies(state, goal, obstacles)
    reference_evaluation = all_active.evaluate_policies(state, goal, obstacles)
    assert padded_evaluation.names == reference_evaluation.names
    assert padded_evaluation.selected_index == reference_evaluation.selected_index
    np.testing.assert_allclose(
        padded_evaluation.values,
        reference_evaluation.values,
        rtol=0.0,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        padded_evaluation.state_gradients,
        reference_evaluation.state_gradients,
        rtol=0.0,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        padded_evaluation.obstacle_gradients,
        reference_evaluation.obstacle_gradients,
        rtol=0.0,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        padded_evaluation.trajectories,
        reference_evaluation.trajectories,
        rtol=0.0,
        atol=2e-6,
    )
    padded_control = padded.solve_control_problem(state, goal, obstacles)
    reference_control = all_active.solve_control_problem(state, goal, obstacles)
    assert padded.last_decision is not None
    assert all_active.last_decision is not None
    assert padded.last_decision.policy_id == all_active.last_decision.policy_id
    assert padded.last_status == all_active.last_status
    np.testing.assert_allclose(
        padded_control, reference_control, rtol=0.0, atol=2e-5
    )


def test_one_warmup_covers_every_runtime_obstacle_count_and_fresh_controller() -> None:
    model = NLQuad3D()
    config = NLQuad3DControllerConfig(
        dt=model.dt,
        backup_horizon=0.1,
        num_radial_policies=2,
        max_obstacles=8,
        sensing_radius=20.0,
    )
    controller = PLCBF_NLQuad3D(model, config, bounds=None)
    state = make_state([0.0, 0.0, 1.0], [0.1, 0.0, 0.0])
    goal = np.asarray([5.0, 0.0, 1.0])
    obstacles = np.asarray(
        [
            [1.0 + 0.1 * index, 0.2 * (-1) ** index, 1.0, 0.2, 0, 0, 0]
            for index in range(8)
        ],
        dtype=float,
    )

    controller.warmup(state, goal, obstacles[:1])
    evaluator = next(iter(controller._compiled_evaluators.values()))
    for count in (1, 2, 4, 8):
        evaluation = controller.evaluate_policies(
            state, goal, obstacles[:count]
        )
        assert evaluation.obstacle_gradients.shape[1] == count
    assert len(controller._compiled_evaluators) == 1
    # Private JAX cache introspection is deliberately confined to this
    # compilation-regression test.
    if hasattr(evaluator, "_cache_size"):
        assert evaluator._cache_size() == 1

    fresh = PLCBF_NLQuad3D(NLQuad3D(), config, bounds=None)
    assert fresh._compiled_evaluator(4) is evaluator

    tuned_config = NLQuad3DControllerConfig(
        dt=model.dt,
        backup_horizon=0.1,
        num_radial_policies=2,
        max_obstacles=8,
        sensing_radius=20.0,
        safety_margin=config.safety_margin + 0.2,
        obstacle_temperature=config.obstacle_temperature + 5.0,
    )
    tuned = PLCBF_NLQuad3D(NLQuad3D(), tuned_config, bounds=None)
    assert tuned._compiled_evaluator(4) is evaluator
    baseline_values = controller.evaluate_policies(
        state, goal, obstacles[:2]
    ).values
    tuned_values = tuned.evaluate_policies(state, goal, obstacles[:2]).values
    assert np.max(np.abs(baseline_values - tuned_values)) > 1e-3
    if hasattr(evaluator, "_cache_size"):
        assert evaluator._cache_size() == 1


def test_shared_jit_structure_cache_is_bounded() -> None:
    _shared_compiled_evaluator.cache_clear()
    first = _shared_compiled_evaluator(1, 1, 1)
    for index in range(1, 17):
        _shared_compiled_evaluator(index + 1, 1, 1)
    info = _shared_compiled_evaluator.cache_info()
    assert info.maxsize == 16
    assert info.currsize == 16
    assert _shared_compiled_evaluator(1, 1, 1) is not first
    _shared_compiled_evaluator.cache_clear()


def test_decision_certificates_are_algorithmically_identical_to_rich_metadata() -> None:
    controller = _small_controller(max_operator="input_space")
    state = make_state([0.0, 0.0, 1.0], [0.3, -0.1, 0.0])
    goal = np.asarray([5.0, 0.5, 1.2])
    obstacles = np.asarray(
        [
            [1.7, 0.2, 1.1, 0.3, -0.2, 0.0, 0.0],
            [2.0, -0.5, 1.4, 0.25, 0.0, 0.1, 0.0],
        ]
    )

    rich = controller.policy_certificates(state, goal, obstacles)
    lean = controller.decision_certificates(state, goal, obstacles)
    for rich_item, lean_item in zip(rich, lean, strict=True):
        assert rich_item.policy_id == lean_item.policy_id
        assert rich_item.value == lean_item.value
        assert rich_item.valid is lean_item.valid
        np.testing.assert_array_equal(
            rich_item.backup_control, lean_item.backup_control
        )
        assert len(rich_item.halfspaces) == len(lean_item.halfspaces)
        for rich_row, lean_row in zip(
            rich_item.halfspaces, lean_item.halfspaces, strict=True
        ):
            np.testing.assert_array_equal(rich_row.normal, lean_row.normal)
            assert rich_row.offset == lean_row.offset

    nominal = controller.model.nominal_input(state, goal)
    bounds = (
        controller.model.input_lower_bound,
        controller.model.input_upper_bound,
    )
    rich_decision = solve_library_pcbf_mi(
        rich, nominal, *bounds, emergency_policy_id="stop"
    )
    lean_decision = solve_library_pcbf_mi(
        lean, nominal, *bounds, emergency_policy_id="stop"
    )
    assert rich_decision.policy_id == lean_decision.policy_id
    assert rich_decision.status == lean_decision.status
    assert rich_decision.feasible is lean_decision.feasible
    np.testing.assert_array_equal(rich_decision.control, lean_decision.control)


def test_controller_discovers_shared_policy_library_hook(monkeypatch) -> None:
    module = ModuleType("plcbf.policy_library")

    def build_nl_quad3d_candidates(**_):
        return [
            {
                "name": "shared_stop",
                "kind": "stop",
                "gain": 2.5,
            },
            {
                "name": "shared_nominal",
                "kind": "nominal",
            },
        ]

    module.build_nl_quad3d_candidates = build_nl_quad3d_candidates
    monkeypatch.setitem(sys.modules, "plcbf.policy_library", module)
    controller = PLCBF_NLQuad3D(
        NLQuad3D(),
        NLQuad3DControllerConfig(
            backup_horizon=0.1,
            num_radial_policies=1,
        ),
    )
    assert controller.policy_provider_source == "plcbf.policy_library"
    assert [item.name for item in controller.candidates(np.ones(3))] == [
        "shared_stop",
        "shared_nominal",
    ]


def test_public_certificates_match_rollout_halfspaces_and_plcbf_selection() -> None:
    def provider(**_):
        return (
            PolicyCandidate("up", "radial", (0.0, 0.0, 1.0), 1.2, 2.0),
            PolicyCandidate("stop", "stop", gain=2.5),
            PolicyCandidate("nominal", "nominal"),
        )

    controller = _small_controller(
        candidate_provider=provider,
        max_operator="input_space",
    )
    state = make_state([0.0, 0.0, 1.0], [0.4, 0.0, 0.0])
    goal = np.array([5.0, 0.0, 1.0])
    obstacles = np.array([[2.0, 0.1, 1.2, 0.35, -0.3, 0.0, 0.0]])
    certificates = controller.policy_certificates(state, goal, obstacles)
    evaluation = controller.last_evaluation
    assert evaluation is not None
    assert tuple(item.policy_id for item in certificates) == evaluation.names
    for index, certificate in enumerate(certificates):
        np.testing.assert_allclose(certificate.value, evaluation.values[index])
        assert len(certificate.halfspaces) == 1
        halfspace = certificate.halfspaces[0]
        np.testing.assert_allclose(
            halfspace.normal,
            evaluation.control_directions[index],
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            halfspace.offset,
            evaluation.constraint_rhs[index],
            rtol=1e-10,
            atol=1e-10,
        )
        assert certificate.backup_control is not None
        assert certificate.backup_control.shape == (4,)
        assert set(
            (
                "rollout_safe",
                "terminal_safe",
                "nominal_prefix_safe",
                "terminal_cost",
            )
        ).issubset(certificate.metadata)
        assert isinstance(certificate.metadata["rollout_safe"], bool)
        assert isinstance(certificate.metadata["terminal_safe"], bool)
        assert isinstance(certificate.metadata["nominal_prefix_safe"], int)
        assert certificate.metadata["terminal_cost"] >= 0.0

    nominal = controller.model.nominal_input(state, goal)
    control = controller.solve_control_problem(state, goal, obstacles)
    evaluation = controller.last_evaluation
    assert evaluation is not None
    selected = controller.last_certificates[evaluation.selected_index]
    expected = solve_box_halfspace_qp(
        nominal,
        controller.model.input_lower_bound,
        controller.model.input_upper_bound,
        selected.halfspaces[0],
    )
    if expected.feasible and selected.halfspaces[0].residual(expected.control) >= (
        -controller.config.constraint_tolerance
    ):
        np.testing.assert_allclose(control, expected.control, atol=1e-7)
    else:
        assert selected.backup_control is not None
        np.testing.assert_allclose(control, selected.backup_control, atol=1e-7)
    assert tuple(
        item.policy_id for item in controller.last_certificates
    ) == evaluation.names


def test_plcbf_filters_with_least_unsafe_policy_when_all_values_are_negative(
    monkeypatch,
) -> None:
    controller = _small_controller(max_operator="input_space")
    state = make_state([0.0, 0.0, 1.0])
    goal = np.array([5.0, 0.0, 1.0])
    nominal = controller.model.nominal_input(state, goal)
    normal = np.array([1.0, 0.0, 0.0, 0.0])
    offset = float(nominal[0] + 0.25)
    certificates = (
        PolicyCertificate(
            "least_unsafe",
            -0.1,
            (CBFHalfspace(normal, offset),),
            backup_control=np.zeros(4),
        ),
        PolicyCertificate(
            "worse",
            -1.0,
            (CBFHalfspace(-normal, -float(nominal[0]) + 0.25),),
            backup_control=np.zeros(4),
        ),
    )
    trajectories = np.repeat(state[None, None, :], 2, axis=0)
    evaluation = RolloutEvaluation(
        names=("least_unsafe", "worse"),
        values=np.array([-0.1, -1.0]),
        state_gradients=np.zeros((2, 12)),
        obstacle_gradients=np.zeros((2, 1, 7)),
        trajectories=trajectories,
        time_derivatives=np.zeros(2),
        control_directions=np.stack([normal, -normal]),
        constraint_rhs=np.array([offset, -float(nominal[0]) + 0.25]),
        scores=np.array([1.0, 0.5]),
        selected_index=0,
    )

    def fake_certificates(*_args, **_kwargs):
        controller.last_evaluation = evaluation
        controller.last_certificates = certificates
        return certificates

    monkeypatch.setattr(controller, "policy_certificates", fake_certificates)
    control = controller.solve_control_problem(
        state,
        goal,
        np.array([[2.0, 0.0, 1.0, 0.3, 0.0, 0.0, 0.0]]),
    )

    assert controller.last_decision is not None
    assert controller.last_decision.policy_id == "least_unsafe"
    assert not controller.last_decision.diagnostics.used_fallback
    assert control[0] >= offset - controller.config.constraint_tolerance
    assert not np.allclose(control, certificates[0].backup_control)


def test_plcbf_executes_selected_backup_when_halfspace_qp_is_infeasible(
    monkeypatch,
) -> None:
    controller = _small_controller(max_operator="input_space")
    state = make_state([0.0, 0.0, 1.0])
    goal = np.array([5.0, 0.0, 1.0])
    backup = controller.model.stop_input(state, gain=2.0)
    certificate = PolicyCertificate(
        "selected_stop",
        -0.1,
        (CBFHalfspace(np.zeros(4), 1.0),),
        backup_control=backup,
    )
    evaluation = RolloutEvaluation(
        names=("selected_stop",),
        values=np.array([-0.1]),
        state_gradients=np.zeros((1, 12)),
        obstacle_gradients=np.zeros((1, 1, 7)),
        trajectories=state.reshape(1, 1, 12),
        time_derivatives=np.zeros(1),
        control_directions=np.zeros((1, 4)),
        constraint_rhs=np.ones(1),
        scores=np.zeros(1),
        selected_index=0,
    )

    def fake_certificates(*_args, **_kwargs):
        controller.last_evaluation = evaluation
        controller.last_certificates = (certificate,)
        return (certificate,)

    monkeypatch.setattr(controller, "policy_certificates", fake_certificates)
    control = controller.solve_control_problem(
        state,
        goal,
        np.array([[2.0, 0.0, 1.0, 0.3, 0.0, 0.0, 0.0]]),
    )

    np.testing.assert_allclose(control, backup)
    assert controller.last_decision is not None
    assert controller.last_decision.policy_id == "selected_stop"
    assert controller.last_decision.diagnostics.used_fallback
    assert (
        controller.last_decision.diagnostics.fallback_reason
        == "selected_policy_qp_infeasible"
    )
    assert (
        controller.last_decision.diagnostics.fallback_source
        == "selected_policy_backup"
    )
    assert controller.last_status == "fallback"


def test_zero_velocity_stop_policy_has_finite_certificate_gradient() -> None:
    controller = _small_controller()
    state = make_state([0.0, 0.0, 1.0])
    goal = np.array([5.0, 0.0, 1.0])
    obstacles = np.array([[2.0, 0.0, 1.0, 0.35, 0.0, 0.0, 0.0]])

    certificates = controller.policy_certificates(state, goal, obstacles)
    evaluation = controller.last_evaluation

    assert evaluation is not None
    assert np.all(np.isfinite(evaluation.values))
    assert np.all(np.isfinite(evaluation.state_gradients))
    assert np.all(np.isfinite(evaluation.control_directions))
    assert all(
        np.all(np.isfinite(certificate.halfspaces[0].normal))
        and np.isfinite(certificate.halfspaces[0].offset)
        for certificate in certificates
    )


def test_full_jit_library_is_finite_at_production_gradient_regression_state() -> None:
    scenario = get_scenario("playground_corridor")
    controller = PLCBF_NLQuad3D(NLQuad3D(), bounds=scenario.bounds)
    state = np.array(
        [
            1.0018753851733151,
            10.0,
            5.0307695773355,
            0.07115596182464244,
            0.0,
            0.28510424842877696,
            0.0,
            0.12107864947036573,
            0.0,
            3.0963580102050645e-18,
            1.0008409231078599,
            0.0,
        ]
    )
    goal = np.array([19.0, 10.0, 5.0])
    obstacles = np.array(
        [
            [4.6625, 10.1125, 5.0625, 0.5, -0.15, 0.45, 0.25],
            [5.7125, 8.9875, 5.65, 0.5, -0.35, 0.35, -0.2],
            [6.3375, 10.9875, 4.45, 0.5, -0.25, -0.45, 0.2],
            [7.4875, 10.425, 5.9375, 0.5, -0.45, 0.1, -0.25],
            [8.325, 9.6875, 4.0625, 0.5, -0.3, 0.35, 0.25],
        ]
    )

    evaluation = controller.evaluate_policies(state, goal, obstacles)
    assert np.all(np.isfinite(evaluation.values))
    assert np.all(np.isfinite(evaluation.state_gradients))
    assert np.all(np.isfinite(evaluation.obstacle_gradients))
    assert np.all(np.isfinite(evaluation.trajectories))

    radial_index = evaluation.names.index("radial_5")
    radial = controller.candidates(goal)[radial_index]
    active = controller._active_obstacles(state, obstacles)
    params = jax_params(controller.model.config)
    assert scenario.bounds is not None
    lower = jnp.asarray(scenario.bounds.lower)
    upper = jnp.asarray(scenario.bounds.upper)

    def independent_value(
        initial_state: jnp.ndarray,
        initial_obstacles: jnp.ndarray,
    ):
        return _policy_value_with_aux(
            initial_state,
            initial_obstacles,
            jnp.asarray(POLICY_RADIAL),
            jnp.asarray(radial.direction),
            jnp.asarray(radial.target_speed),
            jnp.asarray(radial.gain),
            jnp.asarray(goal),
            params,
            controller.config.dt,
            controller.config.horizon_steps,
            lower,
            upper,
            jnp.asarray(True),
            controller.config.safety_margin,
            controller.config.safety_scale,
            controller.config.obstacle_temperature,
            controller.config.time_temperature,
        )

    (_, _), (state_gradient, obstacle_gradient) = jax.jit(
        jax.value_and_grad(
            independent_value,
            argnums=(0, 1),
            has_aux=True,
        )
    )(jnp.asarray(state), jnp.asarray(active))
    np.testing.assert_allclose(
        evaluation.state_gradients[radial_index],
        np.asarray(state_gradient),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        evaluation.obstacle_gradients[radial_index],
        np.asarray(obstacle_gradient),
        rtol=2e-5,
        atol=2e-5,
    )

    certificates = controller.policy_certificates(state, goal, obstacles)
    assert len(certificates) == len(evaluation.names)
    assert all(
        "nonfinite_cbf" not in certificate.diagnostic
        for certificate in certificates
    )


def test_out_of_envelope_rollouts_are_diagnostics_not_plcbf_prefilters() -> None:
    model = NLQuad3D()
    controller = PLCBF_NLQuad3D(
        model,
        NLQuad3DControllerConfig(
            dt=model.dt,
            backup_horizon=0.3,
            num_radial_policies=1,
            max_obstacles=1,
            sensing_radius=20.0,
        ),
        bounds=None,
    )
    state = make_state(
        [0.0, 0.0, 1.0],
        euler=[np.deg2rad(70.0), 0.0, 0.0],
    )
    goal = np.array([5.0, 0.0, 1.0])
    obstacles = np.array([[2.0, 0.0, 1.0, 0.3, 0.0, 0.0, 0.0]])

    certificates = controller.policy_certificates(state, goal, obstacles)

    assert certificates
    assert all(certificate.valid for certificate in certificates)
    assert all(
        certificate.metadata["rollout_domain_valid"] is False
        and certificate.metadata["rollout_safe"] is False
        and certificate.metadata["terminal_safe"] is False
        and certificate.metadata["nominal_prefix_safe"] == 0
        and "tilt_max" in certificate.diagnostic
        for certificate in certificates
    )


def test_terminal_one_step_domain_check_applies_without_obstacles() -> None:
    controller = _small_controller()
    terminal_state = make_state(
        [0.0, 0.0, 1.0],
        euler=[np.deg2rad(59.0), 0.0, 0.0],
        body_rates=[1.0, 0.0, 0.0],
    )
    stop = next(
        candidate
        for candidate in controller.candidates(np.array([5.0, 0.0, 1.0]))
        if candidate.kind == "stop"
    )

    assert not controller._terminal_safe(
        terminal_state[None, :],
        np.zeros((1, 0, 7)),
        np.array([5.0, 0.0, 1.0]),
        stop,
    )


def test_offset_point_clearance_matches_playground_formula() -> None:
    model = NLQuad3D()
    config = NLQuad3DControllerConfig(
        dt=model.dt,
        backup_horizon=0.1,
        num_radial_policies=1,
        safety_margin=0.0,
        safety_scale=1.0,
    )
    controller = PLCBF_NLQuad3D(model, config, bounds=None)
    state = make_state([0.0, 0.0, 0.0])
    obstacles = np.array([[0.0, 0.0, -0.7, 0.3, 0.0, 0.0, 0.0]])
    states = state[None, :]
    obstacle_states = obstacles[None, :, :]

    expected_playground_clearance = (
        np.linalg.norm(model.safety_point(state) - obstacles[0, :3])
        - (
            obstacles[0, 3]
            + model.config.robot_radius
            + config.safety_margin
        )
        * config.safety_scale
    )
    certified_clearance = controller._clearance_history(
        states,
        obstacle_states,
    )[0, 0]

    np.testing.assert_allclose(
        certified_clearance,
        expected_playground_clearance,
        atol=1e-12,
    )


def test_nominal_controller_reaches_goal_without_obstacles() -> None:
    scenario = NLQuad3DScenario(
        name="nominal_reach",
        waypoints=np.array([[0.0, 0.0, 1.0], [4.0, 1.0, 2.0]]),
        obstacles=np.zeros((0, 7)),
        bounds=None,
        reach_threshold=0.45,
        default_steps=400,
    )
    model = NLQuad3D()
    controller = _small_controller()
    result = simulate(
        scenario,
        model=model,
        controller=controller,
        max_steps=400,
        use_plcbf=True,
    )
    assert result.reached_goal
    assert not result.collision
    assert set(result.selected_policies) == {"nominal"}
    assert set(result.controller_statuses) == {"nominal_no_obstacles"}


def test_deterministic_head_on_avoidance_is_collision_free() -> None:
    scenario = get_scenario("head_on")
    model = NLQuad3D()
    config = NLQuad3DControllerConfig(
        dt=model.dt,
        backup_horizon=0.6,
        num_radial_policies=6,
        max_obstacles=2,
        max_operator="input_space",
        safety_margin=0.15,
        safety_scale=1.1,
    )
    controller = PLCBF_NLQuad3D(model, config, bounds=None)
    result = simulate(
        scenario,
        model=model,
        controller=controller,
        max_steps=300,
    )
    assert result.reached_goal
    assert not result.collision
    assert result.minimum_clearance > 0.0
    assert any(name.startswith("radial_") for name in result.selected_policies)
    assert "filtered" in result.controller_statuses
