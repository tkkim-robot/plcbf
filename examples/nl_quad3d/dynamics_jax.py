"""Pure JAX twin of :mod:`examples.nl_quad3d.dynamics`.

Every function in this module is side-effect free and can be composed with
``jax.jit``, ``jax.vmap``, and automatic differentiation.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .dynamics import NLQuad3DConfig, mixing_matrix


class NLQuad3DJaxParams(NamedTuple):
    mass: jnp.ndarray
    inertia: jnp.ndarray
    arm_length: jnp.ndarray
    c_tau: jnp.ndarray
    rho_z: jnp.ndarray
    gravity: jnp.ndarray
    w_min: jnp.ndarray
    w_max: jnp.ndarray
    v_max: jnp.ndarray
    a_max_xy: jnp.ndarray
    a_max_z: jnp.ndarray
    body_rate_max: jnp.ndarray
    nominal_k_v: jnp.ndarray
    nominal_k_a: jnp.ndarray
    nominal_d_min: jnp.ndarray
    nominal_k_att: jnp.ndarray
    nominal_k_rate: jnp.ndarray
    robot_radius: jnp.ndarray
    mixing_inverse: jnp.ndarray


def jax_params(config: NLQuad3DConfig | None = None) -> NLQuad3DJaxParams:
    """Convert the immutable host configuration to a JAX pytree."""

    cfg = config or NLQuad3DConfig()
    return NLQuad3DJaxParams(
        mass=jnp.asarray(cfg.mass),
        inertia=jnp.asarray(cfg.inertia_diag),
        arm_length=jnp.asarray(cfg.arm_length),
        c_tau=jnp.asarray(cfg.c_tau),
        rho_z=jnp.asarray(cfg.rho_z),
        gravity=jnp.asarray(cfg.gravity),
        w_min=jnp.asarray(cfg.w_min),
        w_max=jnp.asarray(float(cfg.w_max)),
        v_max=jnp.asarray(cfg.v_max),
        a_max_xy=jnp.asarray(float(cfg.a_max_xy)),
        a_max_z=jnp.asarray(float(cfg.a_max_z)),
        body_rate_max=jnp.asarray(cfg.body_rate_max),
        nominal_k_v=jnp.asarray(cfg.nominal_k_v),
        nominal_k_a=jnp.asarray(cfg.nominal_k_a),
        nominal_d_min=jnp.asarray(cfg.nominal_d_min),
        nominal_k_att=jnp.asarray(cfg.nominal_k_att),
        nominal_k_rate=jnp.asarray(cfg.nominal_k_rate),
        robot_radius=jnp.asarray(cfg.robot_radius),
        mixing_inverse=jnp.asarray(jnp.linalg.inv(jnp.asarray(mixing_matrix(cfg)))),
    )


def rotation_matrix_jax(
    phi: jnp.ndarray,
    theta: jnp.ndarray,
    psi: jnp.ndarray,
) -> jnp.ndarray:
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    cth, sth = jnp.cos(theta), jnp.sin(theta)
    cpsi, spsi = jnp.cos(psi), jnp.sin(psi)
    return jnp.asarray(
        [
            [
                cpsi * cth,
                cpsi * sth * sphi - spsi * cphi,
                cpsi * sth * cphi + spsi * sphi,
            ],
            [
                spsi * cth,
                spsi * sth * sphi + cpsi * cphi,
                spsi * sth * cphi - cpsi * sphi,
            ],
            [-sth, cth * sphi, cth * cphi],
        ]
    )


def euler_rate_matrix_inv_jax(
    phi: jnp.ndarray,
    theta: jnp.ndarray,
) -> jnp.ndarray:
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    cth, tth = jnp.cos(theta), jnp.tan(theta)
    sec_th = 1.0 / cth
    return jnp.asarray(
        [
            [1.0, sphi * tth, cphi * tth],
            [0.0, cphi, -sphi],
            [0.0, sphi * sec_th, cphi * sec_th],
        ]
    )


def force_torque_allocation_jax(
    params: NLQuad3DJaxParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    force = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    length, drag = params.arm_length, params.c_tau
    torque = jnp.asarray(
        [
            [0.0, length, 0.0, -length],
            [length, 0.0, -length, 0.0],
            [drag, -drag, drag, -drag],
        ]
    )
    return force, torque


def drift_jax(state: jnp.ndarray, params: NLQuad3DJaxParams) -> jnp.ndarray:
    omega = state[9:12]
    euler_dot = euler_rate_matrix_inv_jax(state[6], state[7]) @ omega
    jx, jy, jz = params.inertia
    wx, wy, wz = omega
    gyro = jnp.asarray(
        [
            -(jz - jy) / jx * wy * wz,
            -(jx - jz) / jy * wz * wx,
            -(jy - jx) / jz * wx * wy,
        ]
    )
    return jnp.concatenate(
        [
            state[3:6],
            jnp.asarray([0.0, 0.0, -params.gravity]),
            euler_dot,
            gyro,
        ]
    )


def control_matrix_jax(
    state: jnp.ndarray,
    params: NLQuad3DJaxParams,
) -> jnp.ndarray:
    force, torque = force_torque_allocation_jax(params)
    rotation = rotation_matrix_jax(state[6], state[7], state[8])
    return jnp.vstack(
        [
            jnp.zeros((3, 4)),
            rotation @ force / params.mass,
            jnp.zeros((3, 4)),
            jnp.diag(1.0 / params.inertia) @ torque,
        ]
    )


def continuous_dynamics_jax(
    state: jnp.ndarray,
    control: jnp.ndarray,
    params: NLQuad3DJaxParams,
) -> jnp.ndarray:
    return drift_jax(state, params) + control_matrix_jax(state, params) @ control


def _safe_norm_jax(
    vector: jnp.ndarray,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> jnp.ndarray:
    """Differentiable Euclidean norm with a defined derivative at zero."""

    return jnp.sqrt(jnp.sum(jnp.square(vector), axis=axis) + 1e-12)


def _clamp_norm_jax(vector: jnp.ndarray, maximum: jnp.ndarray) -> jnp.ndarray:
    norm = _safe_norm_jax(vector)
    scale = jnp.minimum(1.0, maximum / jnp.maximum(norm, 1e-12))
    return vector * scale


def wrap_angle_jax(angle: jnp.ndarray) -> jnp.ndarray:
    return (angle + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def step_jax(
    state: jnp.ndarray,
    control: jnp.ndarray,
    dt: float | jnp.ndarray,
    params: NLQuad3DJaxParams,
    *,
    clip_control: bool = True,
) -> jnp.ndarray:
    applied = (
        jnp.clip(control, params.w_min, params.w_max) if clip_control else control
    )
    result = state + continuous_dynamics_jax(state, applied, params) * dt
    result = result.at[6:9].set(wrap_angle_jax(result[6:9]))
    result = result.at[3:6].set(_clamp_norm_jax(result[3:6], params.v_max))
    result = result.at[9:12].set(
        _clamp_norm_jax(result[9:12], params.body_rate_max)
    )
    return result


def cap_acceleration_jax(
    acceleration: jnp.ndarray,
    params: NLQuad3DJaxParams,
) -> jnp.ndarray:
    xy_norm = _safe_norm_jax(acceleration[:2])
    xy_scale = jnp.minimum(
        1.0,
        params.a_max_xy / jnp.maximum(xy_norm, 1e-12),
    )
    return acceleration.at[:2].set(acceleration[:2] * xy_scale).at[2].set(
        jnp.clip(acceleration[2], -params.a_max_z, params.a_max_z)
    )


def saturate_rotors_jax(
    rotor_thrusts: jnp.ndarray,
    params: NLQuad3DJaxParams,
) -> jnp.ndarray:
    collective = jnp.mean(rotor_thrusts)
    torque_component = rotor_thrusts - collective
    upper_room = params.w_max - collective
    lower_room = collective - params.w_min
    positive_scales = jnp.where(
        torque_component > 1e-9,
        upper_room / jnp.maximum(torque_component, 1e-9),
        jnp.inf,
    )
    negative_scales = jnp.where(
        torque_component < -1e-9,
        lower_room / jnp.maximum(-torque_component, 1e-9),
        jnp.inf,
    )
    torque_scale = jnp.minimum(
        1.0,
        jnp.minimum(jnp.min(positive_scales), jnp.min(negative_scales)),
    )
    torque_scale = jnp.where(
        (upper_room <= 0.0) | (lower_room <= 0.0),
        0.0,
        jnp.maximum(torque_scale, 0.0),
    )
    thrusts = collective + torque_scale * torque_component
    upper_shift = jnp.maximum(jnp.max(thrusts) - params.w_max, 0.0)
    thrusts = thrusts - upper_shift
    lower_shift = jnp.maximum(params.w_min - jnp.min(thrusts), 0.0)
    thrusts = thrusts + lower_shift
    return jnp.clip(thrusts, params.w_min, params.w_max)


def acceleration_to_rotors_jax(
    state: jnp.ndarray,
    desired_acceleration: jnp.ndarray,
    params: NLQuad3DJaxParams,
) -> jnp.ndarray:
    acceleration = cap_acceleration_jax(desired_acceleration, params)
    thrust_vector = acceleration + jnp.asarray([0.0, 0.0, params.gravity])
    thrust_norm = _safe_norm_jax(thrust_vector)
    desired_body_z = thrust_vector / jnp.maximum(thrust_norm, 1e-12)
    current_body_z = rotation_matrix_jax(state[6], state[7], state[8])[:, 2]
    attitude_error = jnp.cross(current_body_z, desired_body_z)
    omega = state[9:12]
    torque = params.inertia * (
        params.nominal_k_att * attitude_error - params.nominal_k_rate * omega
    ) + jnp.cross(omega, params.inertia * omega)
    wrench = jnp.concatenate(
        [jnp.asarray([params.mass * thrust_norm]), torque]
    )
    mixed = params.mixing_inverse @ wrench
    hover = jnp.full((4,), params.mass * params.gravity / 4.0)
    mixed = jnp.where(thrust_norm < 1e-6, hover, mixed)
    return saturate_rotors_jax(mixed, params)


def nominal_input_jax(
    state: jnp.ndarray,
    goal: jnp.ndarray,
    params: NLQuad3DJaxParams,
) -> jnp.ndarray:
    position_error = goal[:3] - state[:3]
    position_error = jnp.sign(position_error) * jnp.maximum(
        jnp.abs(position_error) - params.nominal_d_min,
        0.0,
    )
    desired_velocity = _clamp_norm_jax(
        params.nominal_k_v * position_error,
        params.v_max,
    )
    desired_acceleration = cap_acceleration_jax(
        params.nominal_k_a * (desired_velocity - state[3:6]),
        params,
    )
    return acceleration_to_rotors_jax(state, desired_acceleration, params)


def velocity_input_jax(
    state: jnp.ndarray,
    target_velocity: jnp.ndarray,
    gain: float | jnp.ndarray,
    params: NLQuad3DJaxParams,
) -> jnp.ndarray:
    acceleration = cap_acceleration_jax(
        gain * (target_velocity - state[3:6]),
        params,
    )
    return acceleration_to_rotors_jax(state, acceleration, params)


def safety_point_jax(
    state: jnp.ndarray,
    params: NLQuad3DJaxParams,
) -> jnp.ndarray:
    rho = jnp.asarray([0.0, 0.0, params.rho_z])
    rotation = rotation_matrix_jax(state[6], state[7], state[8])
    return state[:3] + rotation @ rho


def safety_velocity_jax(
    state: jnp.ndarray,
    params: NLQuad3DJaxParams,
) -> jnp.ndarray:
    rho = jnp.asarray([0.0, 0.0, params.rho_z])
    rotation = rotation_matrix_jax(state[6], state[7], state[8])
    return state[3:6] + rotation @ jnp.cross(state[9:12], rho)


# Short aliases make formula-level parity checks pleasant to read.
R_zyx_jax = rotation_matrix_jax
W_inv_jax = euler_rate_matrix_inv_jax
f_jax = drift_jax
g_jax = control_matrix_jax
