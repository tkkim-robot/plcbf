"""Full nonlinear rigid-body quadrotor dynamics (NumPy implementation).

State ordering is canonical throughout this case study::

    x = [p(3), v(3), phi, theta, psi, omega(3)]

Controls are the four non-negative rotor thrusts in a ``+`` configuration.
The implementation follows the nonlinear model used by the DPCBF reference
case, while keeping the API independent of CasADi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


STATE_DIM = 12
CONTROL_DIM = 4


@dataclass(frozen=True)
class NLQuad3DConfig:
    """Physical limits and nominal-controller gains."""

    mass: float = 1.5
    inertia_diag: tuple[float, float, float] = (0.0347, 0.0347, 0.0977)
    arm_length: float = 0.25
    c_tau: float = 0.0245
    rho_z: float = 0.25
    gravity: float = 9.81
    w_min: float = 0.0
    w_max: float | None = None
    v_max: float = 3.5
    attitude_bound: float = np.deg2rad(30.0)
    a_max_xy: float | None = None
    a_max_z: float | None = None
    body_rate_max: float = 6.0
    # Desired rotate-to yaw slew cap from the DPCBF reference. This is not a
    # hard bound on the physical body-z rate; ``body_rate_max`` bounds ||omega||.
    nominal_yaw_slew_max: float = 2.0
    # Defaults match the nonlinear Quad3D playground case exactly.
    nominal_k_v: float = 0.65
    nominal_k_a: float = 1.70
    nominal_d_min: float = 0.05
    nominal_k_att: float = 12.5
    nominal_k_rate: float = 7.5
    robot_radius: float = 0.5

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if len(self.inertia_diag) != 3 or min(self.inertia_diag) <= 0.0:
            raise ValueError("inertia_diag must contain three positive values")
        if self.arm_length <= 0.0 or self.c_tau <= 0.0:
            raise ValueError("arm_length and c_tau must be positive")
        hover = self.mass * self.gravity / 4.0
        if self.w_max is None:
            object.__setattr__(self, "w_max", 2.0 * hover)
        if self.a_max_xy is None:
            object.__setattr__(
                self,
                "a_max_xy",
                self.gravity * float(np.tan(self.attitude_bound)),
            )
        if self.a_max_z is None:
            object.__setattr__(
                self,
                "a_max_z",
                max(0.5, 4.0 * float(self.w_max) / self.mass - self.gravity),
            )
        if self.w_min < 0.0 or float(self.w_max) <= self.w_min:
            raise ValueError("rotor thrust limits must satisfy 0 <= w_min < w_max")
        if self.body_rate_max <= 0.0:
            raise ValueError("body_rate_max must be positive")
        if self.nominal_yaw_slew_max <= 0.0:
            raise ValueError("nominal_yaw_slew_max must be positive")

    @property
    def hover_thrust(self) -> float:
        """Thrust produced by each rotor at level hover."""

        return self.mass * self.gravity / 4.0

    @property
    def inertia(self) -> np.ndarray:
        return np.asarray(self.inertia_diag, dtype=float)


def _as_state(state: np.ndarray | Iterable[float]) -> np.ndarray:
    result = np.asarray(state, dtype=float).reshape(-1)
    if result.shape != (STATE_DIM,):
        raise ValueError(f"state must contain {STATE_DIM} values, got {result.shape}")
    return result


def _as_control(control: np.ndarray | Iterable[float]) -> np.ndarray:
    result = np.asarray(control, dtype=float).reshape(-1)
    if result.shape != (CONTROL_DIM,):
        raise ValueError(
            f"control must contain {CONTROL_DIM} rotor thrusts, got {result.shape}"
        )
    return result


def make_state(
    position: Iterable[float] = (0.0, 0.0, 0.0),
    velocity: Iterable[float] = (0.0, 0.0, 0.0),
    euler: Iterable[float] = (0.0, 0.0, 0.0),
    body_rates: Iterable[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build a canonical 12-vector without exposing slice bookkeeping."""

    state = np.concatenate(
        [
            np.asarray(tuple(position), dtype=float),
            np.asarray(tuple(velocity), dtype=float),
            np.asarray(tuple(euler), dtype=float),
            np.asarray(tuple(body_rates), dtype=float),
        ]
    )
    return _as_state(state)


def rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """World-from-body ZYX Euler rotation ``Rz(psi) Ry(theta) Rx(phi)``."""

    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array(
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
        ],
        dtype=float,
    )


def euler_rate_matrix_inv(phi: float, theta: float) -> np.ndarray:
    """Map body rates to ZYX Euler rates."""

    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, tth = np.cos(theta), np.tan(theta)
    sec_th = 1.0 / cth
    return np.array(
        [
            [1.0, sphi * tth, cphi * tth],
            [0.0, cphi, -sphi],
            [0.0, sphi * sec_th, cphi * sec_th],
        ],
        dtype=float,
    )


def force_torque_allocation(
    config: NLQuad3DConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return body force and torque allocation matrices."""

    force = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    length, drag = config.arm_length, config.c_tau
    torque = np.array(
        [
            [0.0, length, 0.0, -length],
            [length, 0.0, -length, 0.0],
            [drag, -drag, drag, -drag],
        ],
        dtype=float,
    )
    return force, torque


def mixing_matrix(config: NLQuad3DConfig) -> np.ndarray:
    """Map rotor thrusts to ``[collective thrust, body torque]``."""

    _, torque = force_torque_allocation(config)
    return np.vstack([np.ones((1, CONTROL_DIM)), torque])


def drift(state: np.ndarray, config: NLQuad3DConfig) -> np.ndarray:
    """Continuous-time drift ``f(x)``."""

    x = _as_state(state)
    omega = x[9:12]
    euler_dot = euler_rate_matrix_inv(x[6], x[7]) @ omega
    jx, jy, jz = config.inertia_diag
    wx, wy, wz = omega
    gyro = np.array(
        [
            -(jz - jy) / jx * wy * wz,
            -(jx - jz) / jy * wz * wx,
            -(jy - jx) / jz * wx * wy,
        ]
    )
    return np.concatenate(
        [x[3:6], np.array([0.0, 0.0, -config.gravity]), euler_dot, gyro]
    )


def control_matrix(state: np.ndarray, config: NLQuad3DConfig) -> np.ndarray:
    """Continuous-time control matrix ``g(x)``."""

    x = _as_state(state)
    force, torque = force_torque_allocation(config)
    rotation = rotation_matrix(x[6], x[7], x[8])
    return np.vstack(
        [
            np.zeros((3, CONTROL_DIM)),
            rotation @ force / config.mass,
            np.zeros((3, CONTROL_DIM)),
            np.diag(1.0 / config.inertia) @ torque,
        ]
    )


def continuous_dynamics(
    state: np.ndarray,
    control: np.ndarray,
    config: NLQuad3DConfig,
) -> np.ndarray:
    """Evaluate ``f(x) + g(x)u`` without modifying either argument."""

    x = _as_state(state)
    u = _as_control(control)
    return drift(x, config) + control_matrix(x, config) @ u


def _clamp_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > maximum:
        return vector * (maximum / norm)
    return vector


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def step(
    state: np.ndarray,
    control: np.ndarray,
    dt: float,
    config: NLQuad3DConfig,
    *,
    clip_control: bool = True,
) -> np.ndarray:
    """Explicit-Euler integration with angle, speed, and body-rate limits."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    x = _as_state(state)
    u = _as_control(control)
    if clip_control:
        u = np.clip(u, config.w_min, float(config.w_max))
    result = x + continuous_dynamics(x, u, config) * dt
    result[6:9] = wrap_angle(result[6:9])
    result[3:6] = _clamp_norm(result[3:6], config.v_max)
    result[9:12] = _clamp_norm(result[9:12], config.body_rate_max)
    return result


def cap_acceleration(
    acceleration: np.ndarray | Iterable[float],
    config: NLQuad3DConfig,
) -> np.ndarray:
    """Apply the physically derived lateral and vertical acceleration caps."""

    result = np.asarray(acceleration, dtype=float).reshape(3).copy()
    xy_norm = float(np.linalg.norm(result[:2]))
    if xy_norm > float(config.a_max_xy):
        result[:2] *= float(config.a_max_xy) / xy_norm
    result[2] = np.clip(result[2], -float(config.a_max_z), float(config.a_max_z))
    return result


def saturate_rotors(
    rotor_thrusts: np.ndarray | Iterable[float],
    config: NLQuad3DConfig,
) -> np.ndarray:
    """Saturate rotors while preserving torque ratios whenever possible.

    The collective component is separated from the zero-sum torque component.
    Torque is scaled only as much as needed to fit available headroom, followed
    by a uniform collective shift. The returned thrusts are always non-negative.
    """

    thrusts = _as_control(rotor_thrusts).copy()
    collective = float(np.mean(thrusts))
    torque_component = thrusts - collective
    upper_room = float(config.w_max) - collective
    lower_room = collective - config.w_min
    if upper_room <= 0.0 or lower_room <= 0.0:
        scale = 0.0
    else:
        scale = 1.0
        for component in torque_component:
            if component > 1e-9:
                scale = min(scale, upper_room / component)
            elif component < -1e-9:
                scale = min(scale, lower_room / -component)
        scale = max(0.0, scale)
    thrusts = collective + scale * torque_component
    if float(np.max(thrusts)) > float(config.w_max):
        thrusts -= float(np.max(thrusts)) - float(config.w_max)
    elif float(np.min(thrusts)) < config.w_min:
        thrusts += config.w_min - float(np.min(thrusts))
    return np.clip(thrusts, config.w_min, float(config.w_max))


def acceleration_to_rotors(
    state: np.ndarray,
    desired_acceleration: np.ndarray | Iterable[float],
    config: NLQuad3DConfig,
) -> np.ndarray:
    """Lee-style geometric map from desired world acceleration to rotors."""

    x = _as_state(state)
    acceleration = cap_acceleration(desired_acceleration, config)
    thrust_vector = acceleration + np.array([0.0, 0.0, config.gravity])
    thrust_norm = float(np.linalg.norm(thrust_vector))
    if thrust_norm < 1e-6:
        return np.full(CONTROL_DIM, config.hover_thrust)
    desired_body_z = thrust_vector / thrust_norm
    current_body_z = rotation_matrix(x[6], x[7], x[8])[:, 2]
    attitude_error = np.cross(current_body_z, desired_body_z)
    omega = x[9:12]
    inertia = config.inertia
    torque = inertia * (
        config.nominal_k_att * attitude_error - config.nominal_k_rate * omega
    ) + np.cross(omega, inertia * omega)
    wrench = np.concatenate([[config.mass * thrust_norm], torque])
    thrusts = np.linalg.solve(mixing_matrix(config), wrench)
    return saturate_rotors(thrusts, config)


def nominal_input(
    state: np.ndarray,
    goal: np.ndarray | Iterable[float],
    config: NLQuad3DConfig,
) -> np.ndarray:
    """Position/velocity cascade used by both the nominal and backup policies."""

    x = _as_state(state)
    goal_array = np.asarray(goal, dtype=float).reshape(-1)
    if goal_array.size < 3:
        goal_array = np.pad(goal_array, (0, 3 - goal_array.size))
    position_error = goal_array[:3] - x[:3]
    position_error = np.sign(position_error) * np.maximum(
        np.abs(position_error) - config.nominal_d_min,
        0.0,
    )
    desired_velocity = _clamp_norm(
        config.nominal_k_v * position_error,
        config.v_max,
    )
    desired_acceleration = cap_acceleration(
        config.nominal_k_a * (desired_velocity - x[3:6]),
        config,
    )
    return acceleration_to_rotors(x, desired_acceleration, config)


def velocity_input(
    state: np.ndarray,
    target_velocity: np.ndarray | Iterable[float],
    config: NLQuad3DConfig,
    *,
    gain: float,
) -> np.ndarray:
    """Track a world-frame velocity target with the same inner-loop cascade."""

    x = _as_state(state)
    target = np.asarray(target_velocity, dtype=float).reshape(3)
    acceleration = cap_acceleration(gain * (target - x[3:6]), config)
    return acceleration_to_rotors(x, acceleration, config)


def stop_input(
    state: np.ndarray,
    config: NLQuad3DConfig,
    *,
    gain: float = 3.0,
) -> np.ndarray:
    return velocity_input(state, np.zeros(3), config, gain=gain)


def rotate_to_input(
    state: np.ndarray,
    desired_yaw: float,
    config: NLQuad3DConfig,
    *,
    gain: float = 2.0,
) -> np.ndarray:
    """Hover while commanding the reference model's bounded yaw slew."""

    if gain <= 0.0:
        raise ValueError("gain must be positive")
    x = _as_state(state)
    yaw_error = float(wrap_angle(float(desired_yaw) - x[8]))
    desired_yaw_rate = float(
        np.clip(
            gain * yaw_error,
            -config.nominal_yaw_slew_max,
            config.nominal_yaw_slew_max,
        )
    )
    yaw_offset = (
        config.inertia[2] * desired_yaw_rate / (4.0 * config.c_tau)
    )
    thrusts = np.array(
        [
            config.hover_thrust + yaw_offset,
            config.hover_thrust - yaw_offset,
            config.hover_thrust + yaw_offset,
            config.hover_thrust - yaw_offset,
        ]
    )
    return saturate_rotors(thrusts, config)


def safety_point(state: np.ndarray, config: NLQuad3DConfig) -> np.ndarray:
    """C3BF-style point offset along the vehicle's body z-axis."""

    x = _as_state(state)
    rho = np.array([0.0, 0.0, config.rho_z])
    return x[:3] + rotation_matrix(x[6], x[7], x[8]) @ rho


def safety_velocity(state: np.ndarray, config: NLQuad3DConfig) -> np.ndarray:
    """World velocity of :func:`safety_point`."""

    x = _as_state(state)
    rho = np.array([0.0, 0.0, config.rho_z])
    rotation = rotation_matrix(x[6], x[7], x[8])
    return x[3:6] + rotation @ np.cross(x[9:12], rho)


class NLQuad3D:
    """Small object-oriented facade over the pure NumPy functions."""

    state_dim = STATE_DIM
    control_dim = CONTROL_DIM
    model_name = "nl_quad3d"

    def __init__(self, config: NLQuad3DConfig | None = None, dt: float = 0.05):
        self.config = config or NLQuad3DConfig()
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)

    @property
    def hover_input(self) -> np.ndarray:
        return np.full(CONTROL_DIM, self.config.hover_thrust)

    @property
    def input_lower_bound(self) -> np.ndarray:
        return np.full(CONTROL_DIM, self.config.w_min)

    @property
    def input_upper_bound(self) -> np.ndarray:
        return np.full(CONTROL_DIM, float(self.config.w_max))

    def f(self, state: np.ndarray) -> np.ndarray:
        return drift(state, self.config)

    def g(self, state: np.ndarray) -> np.ndarray:
        return control_matrix(state, self.config)

    def dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return continuous_dynamics(state, control, self.config)

    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return step(state, control, self.dt, self.config)

    def saturate_rotors(self, control: np.ndarray) -> np.ndarray:
        return saturate_rotors(control, self.config)

    def acceleration_to_rotors(
        self,
        state: np.ndarray,
        desired_acceleration: np.ndarray,
    ) -> np.ndarray:
        return acceleration_to_rotors(state, desired_acceleration, self.config)

    def nominal_input(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return nominal_input(state, goal, self.config)

    def stop_input(self, state: np.ndarray, gain: float = 3.0) -> np.ndarray:
        return stop_input(state, self.config, gain=gain)

    def rotate_to_input(
        self,
        state: np.ndarray,
        desired_yaw: float,
        gain: float = 2.0,
    ) -> np.ndarray:
        return rotate_to_input(
            state,
            desired_yaw,
            self.config,
            gain=gain,
        )

    def safety_point(self, state: np.ndarray) -> np.ndarray:
        return safety_point(state, self.config)

    def safety_velocity(self, state: np.ndarray) -> np.ndarray:
        return safety_velocity(state, self.config)
