from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from examples.nl_quad3d.dynamics import (
    NLQuad3D,
    NLQuad3DConfig,
    control_matrix,
    drift,
    euler_rate_matrix_inv,
    force_torque_allocation,
    make_state,
    mixing_matrix,
    rotate_to_input,
    rotation_matrix,
    safety_point,
    safety_velocity,
    saturate_rotors,
    step,
)
from examples.nl_quad3d.dynamics_jax import (
    control_matrix_jax,
    drift_jax,
    euler_rate_matrix_inv_jax,
    jax_params,
    rotation_matrix_jax,
    safety_point_jax,
    safety_velocity_jax,
    step_jax,
)


def test_reference_physical_parameters_and_hover() -> None:
    config = NLQuad3DConfig()
    assert config.mass == 1.5
    assert config.inertia_diag == (0.0347, 0.0347, 0.0977)
    assert config.arm_length == 0.25
    assert config.c_tau == 0.0245
    assert config.rho_z == 0.25
    assert config.gravity == 9.81
    assert config.w_min == 0.0
    assert np.isclose(config.hover_thrust, 3.67875)
    assert np.isclose(config.w_max, 7.3575)
    assert np.isclose(config.a_max_xy, 9.81 * np.tan(np.deg2rad(30.0)))
    assert np.isclose(config.a_max_z, 9.81)
    assert config.nominal_k_v == 0.65
    assert config.nominal_k_a == 1.70
    assert config.nominal_k_att == 12.5
    assert config.nominal_k_rate == 7.5

    model = NLQuad3D(config)
    derivative = model.dynamics(make_state(), model.hover_input)
    np.testing.assert_allclose(derivative, np.zeros(12), atol=1e-12)


def test_rotate_to_input_uses_bounded_reference_yaw_slew() -> None:
    config = NLQuad3DConfig(nominal_yaw_slew_max=2.0)
    state = make_state(euler=[0.0, 0.0, 0.0])
    control = rotate_to_input(state, np.pi / 2.0, config)
    _, torque = force_torque_allocation(config)

    np.testing.assert_allclose(np.mean(control), config.hover_thrust)
    np.testing.assert_allclose(
        torque @ control,
        [0.0, 0.0, config.inertia_diag[2] * 2.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rotate_to_input(state, 0.0, config),
        np.full(4, config.hover_thrust),
    )


def test_numpy_and_jax_dynamics_have_formula_level_parity() -> None:
    config = NLQuad3DConfig()
    params = jax_params(config)
    state = make_state(
        [1.2, -0.7, 3.1],
        [0.4, -0.3, 0.2],
        [0.21, -0.18, 0.37],
        [0.35, -0.22, 0.14],
    )
    control = np.array([3.1, 4.8, 4.0, 2.7])

    np.testing.assert_allclose(
        rotation_matrix(*state[6:9]),
        np.asarray(rotation_matrix_jax(*jnp.asarray(state[6:9]))),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        euler_rate_matrix_inv(state[6], state[7]),
        np.asarray(euler_rate_matrix_inv_jax(state[6], state[7])),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        drift(state, config),
        np.asarray(jax.jit(drift_jax)(jnp.asarray(state), params)),
        rtol=3e-6,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        control_matrix(state, config),
        np.asarray(jax.jit(control_matrix_jax)(jnp.asarray(state), params)),
        rtol=3e-6,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        step(state, control, 0.05, config),
        np.asarray(
            jax.jit(step_jax)(
                jnp.asarray(state),
                jnp.asarray(control),
                0.05,
                params,
            )
        ),
        rtol=4e-6,
        atol=4e-6,
    )
    np.testing.assert_allclose(
        safety_point(state, config),
        np.asarray(safety_point_jax(jnp.asarray(state), params)),
        rtol=3e-6,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        safety_velocity(state, config),
        np.asarray(safety_velocity_jax(jnp.asarray(state), params)),
        rtol=3e-6,
        atol=3e-6,
    )


def test_plus_configuration_allocation_and_torque_preserving_saturation() -> None:
    config = NLQuad3DConfig()
    force, torque = force_torque_allocation(config)
    expected_torque = np.array(
        [
            [0.0, 0.25, 0.0, -0.25],
            [0.25, 0.0, -0.25, 0.0],
            [0.0245, -0.0245, 0.0245, -0.0245],
        ]
    )
    np.testing.assert_allclose(force[2], np.ones(4))
    np.testing.assert_allclose(torque, expected_torque)
    np.testing.assert_allclose(
        mixing_matrix(config) @ np.linalg.solve(mixing_matrix(config), [12, 1, -2, 0.3]),
        [12, 1, -2, 0.3],
    )

    unsaturated = np.array([10.0, -2.0, 8.0, -1.0])
    saturated = saturate_rotors(unsaturated, config)
    assert np.all(saturated >= config.w_min)
    assert np.all(saturated <= config.w_max)
    input_torque = torque @ unsaturated
    output_torque = torque @ saturated
    nonzero = np.abs(input_torque) > 1e-10
    ratios = output_torque[nonzero] / input_torque[nonzero]
    np.testing.assert_allclose(ratios, np.full_like(ratios, ratios[0]), atol=1e-10)
    assert 0.0 <= ratios[0] <= 1.0


def test_step_clamps_speed_rates_angles_and_rotor_inputs() -> None:
    config = NLQuad3DConfig(v_max=1.0, body_rate_max=2.0)
    state = make_state(
        velocity=[4.0, 0.0, 0.0],
        euler=[3.13, -3.13, 3.13],
        body_rates=[8.0, 1.0, -1.0],
    )
    result = step(state, np.full(4, -100.0), 0.05, config)
    assert np.linalg.norm(result[3:6]) <= config.v_max + 1e-12
    assert np.linalg.norm(result[9:12]) <= config.body_rate_max + 1e-12
    assert np.all(result[6:9] >= -np.pi)
    assert np.all(result[6:9] < np.pi)


def test_safety_point_velocity_include_body_offset_kinematics() -> None:
    config = NLQuad3DConfig(rho_z=0.25)
    state = make_state(body_rates=[0.0, 2.0, 0.0])
    np.testing.assert_allclose(safety_point(state, config), [0.0, 0.0, 0.25])
    np.testing.assert_allclose(safety_velocity(state, config), [0.5, 0.0, 0.0])
