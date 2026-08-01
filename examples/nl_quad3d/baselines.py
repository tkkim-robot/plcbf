"""Faithful comparison algorithms for the nonlinear Quad3D case study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from plcbf.backup_cbf import (
    BackupCbfDecision,
    BackupCbfRolloutDerivatives,
    evaluate_backup_cbf_candidate,
    solve_fixed_backup_cbf,
    solve_multi_backup_cbf_min_intervention,
)
from plcbf.big_m_mpc import (
    BigMTrajectoryMPCConfig,
    BigMTrajectoryMPCProblem,
    BigMTrajectoryMPCResult,
    solve_big_m_trajectory_mpc,
)
from plcbf.baselines import (
    BaselineDecision,
    BenchmarkMethod,
    solve_library_pcbf_mi,
    solve_policy_pcbf,
)
from plcbf.policy_library import CBFHalfspace, PolicyCertificate
from plcbf.dynamics_linearization import linearize_discrete_trajectory
from plcbf.trajectory_shielding import (
    GatekeeperShield,
    ModelPredictiveShield,
    ShieldDecision,
)

from .controller import PLCBF_NLQuad3D
from .dynamics import NLQuad3D, mixing_matrix
from .policies import PolicyCandidate, fibonacci_directions
from .scenarios import WorldBounds, advance_obstacles


class _PreparedModelPredictiveShield(ModelPredictiveShield):
    """Reset the retrace policy's rollout-local waypoint cursor."""

    def __init__(self, *, prepare_backup_rollout: Callable[[], None], **kwargs):
        self._prepare_backup_rollout = prepare_backup_rollout
        super().__init__(**kwargs)

    def _candidate(self, initial_state, nominal_steps):
        self._prepare_backup_rollout()
        return super()._candidate(initial_state, nominal_steps)


class _PreparedGatekeeperShield(GatekeeperShield):
    """Reset the retrace policy before every independently tested plan."""

    def __init__(self, *, prepare_backup_rollout: Callable[[], None], **kwargs):
        self._prepare_backup_rollout = prepare_backup_rollout
        super().__init__(**kwargs)

    def _candidate(self, initial_state, nominal_steps):
        self._prepare_backup_rollout()
        return super()._candidate(initial_state, nominal_steps)


def _smooth_min(values: np.ndarray, temperature: float) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return 100.0
    minimum = float(np.min(array))
    return float(
        minimum
        - np.log(np.sum(np.exp(-float(temperature) * (array - minimum))))
        / float(temperature)
    )


@dataclass(frozen=True)
class _BatchedStrictMultiBackupData:
    """Exact scalar-formulation samples evaluated in one NumPy batch."""

    path_values: np.ndarray
    terminal_values: np.ndarray
    path_gradients: np.ndarray
    path_time_derivatives: np.ndarray
    terminal_gradients: np.ndarray
    path_flow_derivatives: np.ndarray
    trajectories: np.ndarray


@dataclass(frozen=True)
class NLQuad3DBaselineConfig:
    """Warehouse baseline constants adapted to the nonlinear plant horizon."""

    gatekeeper_nominal_steps: int = 30
    gatekeeper_discount_steps: int = 1
    multi_backup_maneuver_fraction: float = 0.5
    backup_cbf_alpha: float = 2.0
    backup_cbf_terminal_alpha: float = 2.0
    terminal_linear_speed_mps: float = 0.5
    terminal_attitude_rad: float = 0.4
    terminal_angular_rate_rad_s: float = 0.5
    terminal_altitude_error_m: float = 0.25
    retrace_gain: float = 6.0
    retrace_target_speed_mps: float = 2.8
    retrace_waypoint_threshold_m: float = 0.8
    pcbf_alpha: float = 5.0
    time_derivative_step_s: float = 0.02
    backup_cbf_gradient_step: float = 1e-5
    gradient_steps: tuple[float, ...] = (
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.025,
        0.0025,
        0.0025,
        0.0025,
        0.01,
        0.01,
        0.01,
    )
    fixed_backup_policy_id: str = "retrace_waypoint"
    mi_num_radial_policies: int = 32
    mi_position_tube_m: float = 3.0
    mi_control_tube_thrust: float = 6.0
    mi_control_tube_steps: int = 2
    mi_big_m_position_m: float = 400.0
    mi_big_m_control_thrust: float = 60.0
    mi_big_m_safety: float = 50.0
    mi_tracking_weight: float = 8.0
    mi_terminal_weight: float = 16.0
    mi_velocity_weight: float = 0.15
    mi_control_weight: float = 0.02
    mi_nominal_weight: float = 0.5
    mi_time_limit_s: float = 1.0
    mi_mip_rel_gap: float = 0.05

    def __post_init__(self) -> None:
        if self.gatekeeper_nominal_steps < 0:
            raise ValueError("gatekeeper_nominal_steps must be nonnegative")
        if self.gatekeeper_discount_steps <= 0:
            raise ValueError("gatekeeper_discount_steps must be positive")
        if not 0.0 <= self.multi_backup_maneuver_fraction < 1.0:
            raise ValueError(
                "multi_backup_maneuver_fraction must lie in [0, 1)"
            )
        if len(self.gradient_steps) != 12 or min(self.gradient_steps) <= 0.0:
            raise ValueError("gradient_steps must contain 12 positive values")
        if (
            not np.isfinite(self.backup_cbf_gradient_step)
            or self.backup_cbf_gradient_step <= 0.0
        ):
            raise ValueError(
                "backup_cbf_gradient_step must be finite and positive"
            )
        if (
            self.retrace_gain <= 0.0
            or self.retrace_target_speed_mps <= 0.0
            or self.retrace_waypoint_threshold_m <= 0.0
            or self.pcbf_alpha < 0.0
        ):
            raise ValueError(
                "retrace gains, speed, and threshold must be positive and "
                "pcbf_alpha must be nonnegative"
            )
        if self.mi_position_tube_m < 0.0:
            raise ValueError("mi_position_tube_m must be nonnegative")
        if self.mi_num_radial_policies < 1:
            raise ValueError("mi_num_radial_policies must be positive")
        if self.mi_control_tube_thrust < 0.0:
            raise ValueError("mi_control_tube_thrust must be nonnegative")
        if self.mi_control_tube_steps < 0:
            raise ValueError("mi_control_tube_steps must be nonnegative")
        if (
            self.mi_big_m_position_m <= 0.0
            or self.mi_big_m_control_thrust <= 0.0
            or self.mi_big_m_safety <= 0.0
        ):
            raise ValueError("MI-MPC Big-M constants must be positive")
        if min(
            self.mi_tracking_weight,
            self.mi_terminal_weight,
            self.mi_velocity_weight,
            self.mi_control_weight,
            self.mi_nominal_weight,
        ) < 0.0:
            raise ValueError("MI-MPC objective weights must be nonnegative")


class NLQuad3DBaselineSuite:
    """Per-trial state for the nonlinear-quadrotor baselines."""

    def __init__(
        self,
        controller: PLCBF_NLQuad3D,
        model: NLQuad3D,
        bounds: WorldBounds | None,
        *,
        waypoints: np.ndarray | None = None,
        algorithm_config: NLQuad3DBaselineConfig = NLQuad3DBaselineConfig(),
    ) -> None:
        self.controller = controller
        self.model = model
        self.bounds = bounds
        self.algorithm_config = algorithm_config
        self._goal = np.zeros(3)
        self._obstacles = np.zeros((0, 7))
        self._altitude_reference = 0.0
        route = (
            np.zeros((1, 3), dtype=float)
            if waypoints is None
            else np.asarray(waypoints, dtype=float)
        )
        if (
            route.ndim != 2
            or route.shape[0] < 1
            or route.shape[1] != 3
            or not np.all(np.isfinite(route))
        ):
            raise ValueError("waypoints must be a non-empty finite (N, 3) array")
        self._waypoints = route.copy()
        self._active_waypoint_index = min(1, route.shape[0] - 1)
        self._active_retrace_index = max(0, self._active_waypoint_index - 1)
        self._retrace_rollout_index = self._active_retrace_index
        self._backup_steps = controller.config.horizon_steps
        self._trajectory_obstacle_history_cache: np.ndarray | None = None
        self.last_mi_mpc_result: BigMTrajectoryMPCResult | None = None
        common = dict(
            step=self.model.step,
            nominal_control=self._nominal_feedback,
            backup_control=self._fixed_backup_feedback,
            trajectory_is_safe=self._trajectory_is_safe,
            backup_horizon_steps=self._backup_steps,
            backup_policy_id=algorithm_config.fixed_backup_policy_id,
            prepare_backup_rollout=self._prepare_retrace_rollout,
        )
        self.mps = _PreparedModelPredictiveShield(**common)
        self.gatekeeper = _PreparedGatekeeperShield(
            **common,
            nominal_horizon_steps=algorithm_config.gatekeeper_nominal_steps,
            horizon_discount_steps=algorithm_config.gatekeeper_discount_steps,
        )

    def _nominal_feedback(self, state: np.ndarray) -> np.ndarray:
        return self.model.nominal_input(state, self._goal)

    def _fixed_backup_feedback(self, state: np.ndarray) -> np.ndarray:
        control, self._retrace_rollout_index = self._retrace_control(
            state,
            self._retrace_rollout_index,
        )
        return control

    def _terminal_stop_feedback(self, state: np.ndarray) -> np.ndarray:
        """Common terminal stop tail used by every strict multi-backup branch."""

        return self.model.stop_input(
            state,
            gain=self.controller.config.stop_gain,
        )

    def _prepare_retrace_rollout(self) -> None:
        self._retrace_rollout_index = self._active_retrace_index

    def _retrace_control(
        self,
        state: np.ndarray,
        waypoint_index: int,
    ) -> tuple[np.ndarray, int]:
        """Warehouse retrace-waypoint backup adapted to full 3-D motion."""

        value = np.asarray(state, dtype=float).reshape(12)
        index = int(np.clip(waypoint_index, 0, self._waypoints.shape[0] - 1))
        target = self._waypoints[index]
        distance = float(np.linalg.norm(target - value[:3]))
        if (
            distance < self.algorithm_config.retrace_waypoint_threshold_m
            and index > 0
        ):
            index -= 1
            target = self._waypoints[index]
            distance = float(np.linalg.norm(target - value[:3]))
        direction = (target - value[:3]) / (distance + 1e-6)
        acceleration_cap = min(
            float(self.model.config.a_max_xy),
            float(self.model.config.a_max_z),
        )
        braking_speed = np.sqrt(2.0 * acceleration_cap * max(distance, 0.0))
        target_speed = min(
            self.algorithm_config.retrace_target_speed_mps,
            braking_speed,
            float(self.model.config.v_max),
        )
        target_velocity = direction * target_speed
        desired_acceleration = self.algorithm_config.retrace_gain * (
            target_velocity - value[3:6]
        )
        return (
            self.model.acceleration_to_rotors(value, desired_acceleration),
            index,
        )

    def _direct_retrace_control(self, state: np.ndarray) -> np.ndarray:
        control, _ = self._retrace_control(
            state,
            self._active_retrace_index,
        )
        return control

    def _candidate_feedback(
        self,
        state: np.ndarray,
        candidate: PolicyCandidate,
    ) -> np.ndarray:
        return self.controller._candidate_control(
            state, self._goal, candidate
        )

    def mi_mpc_candidates(self) -> tuple[PolicyCandidate, ...]:
        """Warehouse-style MI-MPC branch library, separate from PL-CBF."""

        speed = min(
            float(self.controller.config.target_speed),
            float(self.model.config.v_max),
        )
        return tuple(
            PolicyCandidate(
                name=f"mi_radial_{index}",
                kind="radial",
                direction=tuple(float(item) for item in direction),
                target_speed=speed,
                gain=float(self.controller.config.radial_gain),
            )
            for index, direction in enumerate(
                fibonacci_directions(
                    self.algorithm_config.mi_num_radial_policies
                )
            )
        )

    def _obstacle_history(
        self,
        *,
        time_offset: float,
        count: int,
        initial_obstacles: np.ndarray | None = None,
    ) -> np.ndarray:
        source = self._obstacles if initial_obstacles is None else initial_obstacles
        obstacles = np.asarray(source, dtype=float).reshape(-1, 7).copy()
        remaining = max(0.0, float(time_offset))
        while remaining > 1e-12:
            step = min(self.model.dt, remaining)
            obstacles = advance_obstacles(obstacles, step, self.bounds)
            remaining -= step
        history = [obstacles.copy()]
        for _ in range(count - 1):
            obstacles = advance_obstacles(
                obstacles, self.model.dt, self.bounds
            )
            history.append(obstacles.copy())
        return np.asarray(history)

    def _point_margin(
        self,
        state: np.ndarray,
        obstacles: np.ndarray,
    ) -> float:
        """Return the playground's sphere-only clearance safe set.

        ``WorldBounds`` governs prescribed obstacle reflection; the playground
        does not model those visualization-box faces as robot walls.  Keeping
        them out of this margin also makes the path-wise baselines use the same
        collision geometry as PL-CBF, Library-PCBF-MI, and benchmark outcomes.
        """

        values: list[float] = []
        obstacle_array = np.asarray(obstacles, dtype=float).reshape(-1, 7)
        if obstacle_array.shape[0]:
            point = self.model.safety_point(state)
            distances = np.linalg.norm(
                point[None, :] - obstacle_array[:, :3],
                axis=1,
            )
            safe_radii = (
                obstacle_array[:, 3]
                + self.model.config.robot_radius
                + self.controller.config.safety_margin
            ) * self.controller.config.safety_scale
            values.append(float(np.min(distances - safe_radii)))
        # Path-wise baselines validate collision geometry, not a prospective
        # attitude heuristic.  The benchmark separately applies its explicit
        # episode-level tilt protocol.
        return float(min(values, default=1e12))

    def _point_margin_gradient(
        self,
        state: np.ndarray,
        obstacles: np.ndarray,
    ) -> np.ndarray:
        """Warehouse-style forward difference of the instantaneous margin."""

        value = np.asarray(state, dtype=float).reshape(12)
        base = self._point_margin(value, obstacles)
        gradient = np.zeros(12, dtype=float)
        epsilon = self.algorithm_config.backup_cbf_gradient_step
        for index in range(12):
            perturbed = value.copy()
            perturbed[index] += epsilon
            gradient[index] = (
                self._point_margin(perturbed, obstacles) - base
            ) / epsilon
        return gradient

    def _path_flow_derivatives(
        self,
        states: np.ndarray,
        obstacle_history: np.ndarray,
        path_indices: Sequence[int],
    ) -> np.ndarray:
        """Return ``grad h(phi_i) @ f_policy_i`` as in warehouse Backup-CBF."""

        trajectory = np.asarray(states, dtype=float).reshape(-1, 12)
        obstacles = np.asarray(obstacle_history, dtype=float)
        indices = tuple(int(index) for index in path_indices)
        if trajectory.shape[0] < 2 and indices:
            raise ValueError("path flow derivatives require two rollout states")
        result = []
        for index in indices:
            if index < 0 or index >= trajectory.shape[0]:
                raise IndexError("path-flow state index is outside the rollout")
            if index < trajectory.shape[0] - 1:
                policy_flow = (
                    trajectory[index + 1] - trajectory[index]
                ) / self.model.dt
            else:
                policy_flow = (
                    trajectory[index] - trajectory[index - 1]
                ) / self.model.dt
            gradient = self._point_margin_gradient(
                trajectory[index],
                obstacles[index],
            )
            result.append(float(gradient @ policy_flow))
        return np.asarray(result, dtype=float)

    def _trajectory_is_safe(self, states: np.ndarray) -> bool:
        trajectory = np.asarray(states, dtype=float)
        if (
            trajectory.ndim != 2
            or trajectory.shape[1] != 12
            or not np.all(np.isfinite(trajectory))
        ):
            return False
        if type(self.model) is NLQuad3D:
            cached_history = self._trajectory_obstacle_history_cache
            if (
                cached_history is None
                or cached_history.shape[0] < trajectory.shape[0]
            ):
                cached_history = self._obstacle_history(
                    time_offset=0.0,
                    count=trajectory.shape[0],
                )
                self._trajectory_obstacle_history_cache = cached_history
            obstacle_history = cached_history[: trajectory.shape[0]]
            margins = self._batch_point_margins(
                trajectory,
                obstacle_history,
            )
            return bool(np.all(margins >= 0.0))
        return self._trajectory_is_safe_scalar(trajectory)

    def _trajectory_is_safe_scalar(self, states: np.ndarray) -> bool:
        """Reference scalar trajectory validator for external dynamics."""

        trajectory = np.asarray(states, dtype=float)
        if (
            trajectory.ndim != 2
            or trajectory.shape[1] != 12
            or not np.all(np.isfinite(trajectory))
        ):
            return False
        obstacle_history = self._obstacle_history(
            time_offset=0.0,
            count=trajectory.shape[0],
        )
        return all(
            self._point_margin(state, obstacles) >= 0.0
            for state, obstacles in zip(
                trajectory, obstacle_history, strict=True
            )
        )

    def _retrace_rollout(
        self,
        initial_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        state = np.asarray(initial_state, dtype=float).reshape(12).copy()
        waypoint_index = self._active_retrace_index
        states = [state.copy()]
        controls = []
        for _ in range(self._backup_steps):
            control, waypoint_index = self._retrace_control(
                state,
                waypoint_index,
            )
            state = self.model.step(state, control)
            controls.append(np.asarray(control, dtype=float).copy())
            states.append(state.copy())
        return np.asarray(states), np.asarray(controls), waypoint_index

    def _retrace_value(
        self,
        initial_state: np.ndarray,
        active_obstacles: np.ndarray,
        *,
        time_offset: float,
    ) -> float:
        states, _, _ = self._retrace_rollout(initial_state)
        obstacle_history = self._obstacle_history(
            time_offset=time_offset,
            count=states.shape[0],
            initial_obstacles=active_obstacles,
        )
        path_values = np.asarray(
            [
                self._point_margin(state, obstacles)
                for state, obstacles in zip(
                    states, obstacle_history, strict=True
                )
            ],
            dtype=float,
        )
        return _smooth_min(
            path_values,
            self.controller.config.time_temperature,
        )

    def _retrace_pcbf_certificate(
        self,
        state: np.ndarray,
    ) -> PolicyCertificate:
        """Single-policy PCBF certificate for the fixed retrace backup."""

        if (
            type(self.model) is NLQuad3D
            and type(self.controller) is PLCBF_NLQuad3D
        ):
            return self._retrace_pcbf_certificate_batched(state)
        return self._retrace_pcbf_certificate_scalar(state)

    def _retrace_pcbf_certificate_scalar(
        self,
        state: np.ndarray,
    ) -> PolicyCertificate:
        """Reference scalar implementation retained for external models."""

        value = np.asarray(state, dtype=float).reshape(12)
        active = self.controller._active_obstacles(value, self._obstacles)

        def evaluate(candidate_state: np.ndarray, time_offset: float) -> float:
            return self._retrace_value(
                candidate_state,
                active,
                time_offset=time_offset,
            )

        base_value = evaluate(value, 0.0)
        gradient = np.empty(12, dtype=float)
        for index, step in enumerate(self.algorithm_config.gradient_steps):
            positive = value.copy()
            negative = value.copy()
            positive[index] += step
            negative[index] -= step
            gradient[index] = (
                evaluate(positive, 0.0) - evaluate(negative, 0.0)
            ) / (2.0 * step)
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm > self.controller.config.max_gradient_norm:
            gradient *= (
                self.controller.config.max_gradient_norm / gradient_norm
            )
        derivative_step = self.algorithm_config.time_derivative_step_s
        time_derivative = (
            evaluate(value, derivative_step) - base_value
        ) / derivative_step
        drift = self.model.f(value)
        control_matrix = self.model.g(value)
        normal = gradient @ control_matrix
        offset = (
            -float(gradient @ drift)
            - float(time_derivative)
            - self.algorithm_config.pcbf_alpha
            * (base_value - self.controller.config.cbf_value_buffer)
        )
        direct = self._direct_retrace_control(value)
        finite = bool(
            np.isfinite(base_value)
            and np.all(np.isfinite(gradient))
            and np.isfinite(time_derivative)
            and np.all(np.isfinite(normal))
            and np.isfinite(offset)
        )
        return PolicyCertificate(
            policy_id=self.algorithm_config.fixed_backup_policy_id,
            value=base_value if finite else -1e12,
            halfspaces=(
                CBFHalfspace(
                    normal if finite else np.zeros(4),
                    offset if finite else 1e12,
                    "retrace_waypoint:pcbf",
                ),
            ),
            backup_control=direct,
            valid=finite,
            diagnostic="" if finite else "nonfinite_retrace_pcbf",
            metadata={
                "rollout_safe": bool(finite and base_value >= 0.0),
                "fixed_backup": True,
                "backup_kind": "retrace_waypoint",
            },
        )

    def _batch_body_z(self, states: np.ndarray) -> np.ndarray:
        """Return the world-frame body-z axis for any leading batch shape."""

        values = np.asarray(states, dtype=float)
        phi = values[..., 6]
        theta = values[..., 7]
        psi = values[..., 8]
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        return np.stack(
            [
                cpsi * stheta * cphi + spsi * sphi,
                spsi * stheta * cphi - cpsi * sphi,
                ctheta * cphi,
            ],
            axis=-1,
        )

    @staticmethod
    def _batch_clamp_norm(
        vectors: np.ndarray,
        maximum: float,
    ) -> np.ndarray:
        values = np.asarray(vectors, dtype=float)
        norms = np.linalg.norm(values, axis=-1)
        scales = np.where(
            norms > float(maximum),
            float(maximum) / np.maximum(norms, 1e-300),
            1.0,
        )
        return values * scales[..., None]

    def _batch_cap_acceleration(
        self,
        accelerations: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(accelerations, dtype=float).copy()
        xy_norms = np.linalg.norm(values[..., :2], axis=-1)
        xy_scales = np.where(
            xy_norms > float(self.model.config.a_max_xy),
            float(self.model.config.a_max_xy)
            / np.maximum(xy_norms, 1e-300),
            1.0,
        )
        values[..., :2] *= xy_scales[..., None]
        values[..., 2] = np.clip(
            values[..., 2],
            -float(self.model.config.a_max_z),
            float(self.model.config.a_max_z),
        )
        return values

    def _batch_saturate_rotors(
        self,
        rotor_thrusts: np.ndarray,
    ) -> np.ndarray:
        """Vectorized twin of ``dynamics.saturate_rotors``."""

        thrusts = np.asarray(rotor_thrusts, dtype=float)
        collective = np.mean(thrusts, axis=-1)
        torque_component = thrusts - collective[..., None]
        upper_room = float(self.model.config.w_max) - collective
        lower_room = collective - float(self.model.config.w_min)
        positive_scales = np.where(
            torque_component > 1e-9,
            upper_room[..., None]
            / np.maximum(torque_component, 1e-300),
            np.inf,
        )
        negative_scales = np.where(
            torque_component < -1e-9,
            lower_room[..., None]
            / np.maximum(-torque_component, 1e-300),
            np.inf,
        )
        scales = np.minimum(
            1.0,
            np.minimum(
                np.min(positive_scales, axis=-1),
                np.min(negative_scales, axis=-1),
            ),
        )
        scales = np.where(
            (upper_room <= 0.0) | (lower_room <= 0.0),
            0.0,
            np.maximum(scales, 0.0),
        )
        result = collective[..., None] + scales[..., None] * torque_component
        maximum = np.max(result, axis=-1)
        minimum = np.min(result, axis=-1)
        exceeds_upper = maximum > float(self.model.config.w_max)
        below_lower = minimum < float(self.model.config.w_min)
        result = np.where(
            exceeds_upper[..., None],
            result
            - (
                maximum - float(self.model.config.w_max)
            )[..., None],
            np.where(
                below_lower[..., None],
                result
                + (
                    float(self.model.config.w_min) - minimum
                )[..., None],
                result,
            ),
        )
        return np.clip(
            result,
            float(self.model.config.w_min),
            float(self.model.config.w_max),
        )

    def _batch_acceleration_to_rotors(
        self,
        states: np.ndarray,
        desired_accelerations: np.ndarray,
    ) -> np.ndarray:
        """Vectorized twin of ``dynamics.acceleration_to_rotors``."""

        values = np.asarray(states, dtype=float)
        accelerations = self._batch_cap_acceleration(
            desired_accelerations
        )
        thrust_vectors = accelerations.copy()
        thrust_vectors[..., 2] += float(self.model.config.gravity)
        thrust_norms = np.linalg.norm(thrust_vectors, axis=-1)
        desired_body_z = thrust_vectors / np.maximum(
            thrust_norms[..., None],
            1e-300,
        )
        attitude_errors = np.cross(
            self._batch_body_z(values),
            desired_body_z,
        )
        omega = values[..., 9:12]
        inertia = np.asarray(self.model.config.inertia, dtype=float)
        torques = inertia * (
            float(self.model.config.nominal_k_att) * attitude_errors
            - float(self.model.config.nominal_k_rate) * omega
        ) + np.cross(omega, inertia * omega)
        wrenches = np.concatenate(
            [self.model.config.mass * thrust_norms[..., None], torques],
            axis=-1,
        )
        flat_wrenches = wrenches.reshape(-1, 4)
        mixed = np.linalg.solve(
            mixing_matrix(self.model.config),
            flat_wrenches.T,
        ).T.reshape(wrenches.shape)
        hover = np.full_like(mixed, self.model.config.hover_thrust)
        mixed = np.where(
            (thrust_norms < 1e-6)[..., None],
            hover,
            mixed,
        )
        return self._batch_saturate_rotors(mixed)

    def _batch_policy_controls(
        self,
        states: np.ndarray,
        candidates: Sequence[PolicyCandidate],
        *,
        force_stop: bool,
    ) -> np.ndarray:
        """Evaluate one policy per first-axis entry without Python loops."""

        values = np.asarray(states, dtype=float)
        if values.ndim < 2 or values.shape[-1] != 12:
            raise ValueError("batched policy states must end in 12 values")
        candidate_tuple = tuple(candidates)
        if values.shape[0] != len(candidate_tuple):
            raise ValueError(
                "batched policy state count must equal candidate count"
            )
        scalar_shape = (
            (len(candidate_tuple),)
            + (1,) * (values.ndim - 2)
        )
        if force_stop:
            target_velocities = np.zeros(values.shape[:-1] + (3,))
            gains = np.full(
                scalar_shape,
                float(self.controller.config.stop_gain),
            )
            accelerations = gains[..., None] * (
                target_velocities - values[..., 3:6]
            )
            return self._batch_acceleration_to_rotors(
                values,
                self._batch_cap_acceleration(accelerations),
            )

        directions = np.asarray(
            [candidate.direction for candidate in candidate_tuple],
            dtype=float,
        ).reshape(scalar_shape + (3,))
        target_speeds = np.asarray(
            [candidate.target_speed for candidate in candidate_tuple],
            dtype=float,
        ).reshape(scalar_shape)
        gains = np.asarray(
            [candidate.gain for candidate in candidate_tuple],
            dtype=float,
        ).reshape(scalar_shape)
        radial_mask = np.asarray(
            [candidate.kind == "radial" for candidate in candidate_tuple],
            dtype=bool,
        ).reshape(scalar_shape)
        nominal_mask = np.asarray(
            [candidate.kind == "nominal" for candidate in candidate_tuple],
            dtype=bool,
        ).reshape(scalar_shape)
        target_velocities = np.where(
            radial_mask[..., None],
            directions * target_speeds[..., None],
            0.0,
        )
        velocity_accelerations = self._batch_cap_acceleration(
            gains[..., None] * (
                target_velocities - values[..., 3:6]
            )
        )

        position_error = self._goal - values[..., :3]
        position_error = np.sign(position_error) * np.maximum(
            np.abs(position_error)
            - float(self.model.config.nominal_d_min),
            0.0,
        )
        desired_velocity = self._batch_clamp_norm(
            float(self.model.config.nominal_k_v) * position_error,
            float(self.model.config.v_max),
        )
        nominal_accelerations = self._batch_cap_acceleration(
            float(self.model.config.nominal_k_a)
            * (desired_velocity - values[..., 3:6])
        )
        accelerations = np.where(
            nominal_mask[..., None],
            nominal_accelerations,
            velocity_accelerations,
        )
        return self._batch_acceleration_to_rotors(values, accelerations)

    def _batch_step(
        self,
        states: np.ndarray,
        controls: np.ndarray,
    ) -> np.ndarray:
        """Vectorized twin of the nonlinear plant's explicit Euler step."""

        values = np.asarray(states, dtype=float)
        applied = np.clip(
            np.asarray(controls, dtype=float),
            float(self.model.config.w_min),
            float(self.model.config.w_max),
        )
        derivative = np.zeros_like(values)
        derivative[..., :3] = values[..., 3:6]
        derivative[..., 3:6] = (
            self._batch_body_z(values)
            * (
                np.sum(applied, axis=-1)
                / float(self.model.config.mass)
            )[..., None]
        )
        derivative[..., 5] -= float(self.model.config.gravity)

        phi = values[..., 6]
        theta = values[..., 7]
        omega = values[..., 9:12]
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta = np.cos(theta)
        ttheta = np.tan(theta)
        wx, wy, wz = np.moveaxis(omega, -1, 0)
        derivative[..., 6] = (
            wx + sphi * ttheta * wy + cphi * ttheta * wz
        )
        derivative[..., 7] = cphi * wy - sphi * wz
        derivative[..., 8] = (
            sphi / ctheta * wy + cphi / ctheta * wz
        )

        jx, jy, jz = self.model.config.inertia_diag
        gyro = np.stack(
            [
                -(jz - jy) / jx * wy * wz,
                -(jx - jz) / jy * wz * wx,
                -(jy - jx) / jz * wx * wy,
            ],
            axis=-1,
        )
        torque_allocation = mixing_matrix(self.model.config)[1:, :]
        control_torques = applied @ torque_allocation.T
        derivative[..., 9:12] = (
            gyro
            + control_torques
            / np.asarray(self.model.config.inertia_diag, dtype=float)
        )

        result = values + derivative * self.model.dt
        result[..., 6:9] = (
            result[..., 6:9] + np.pi
        ) % (2.0 * np.pi) - np.pi
        result[..., 3:6] = self._batch_clamp_norm(
            result[..., 3:6],
            float(self.model.config.v_max),
        )
        result[..., 9:12] = self._batch_clamp_norm(
            result[..., 9:12],
            float(self.model.config.body_rate_max),
        )
        return result

    def _batch_retrace_controls(
        self,
        states: np.ndarray,
        waypoint_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized twin of the fixed retrace-waypoint feedback."""

        values = np.asarray(states, dtype=float)
        if values.ndim != 2 or values.shape[1] != 12:
            raise ValueError("batched retrace states must have shape (N, 12)")
        indices = np.clip(
            np.asarray(waypoint_indices, dtype=int).reshape(-1),
            0,
            self._waypoints.shape[0] - 1,
        )
        if indices.shape[0] != values.shape[0]:
            raise ValueError("one waypoint index is required per retrace state")
        targets = self._waypoints[indices]
        distances = np.linalg.norm(targets - values[:, :3], axis=1)
        decrement = (
            distances
            < self.algorithm_config.retrace_waypoint_threshold_m
        ) & (indices > 0)
        indices = indices - decrement.astype(int)
        targets = self._waypoints[indices]
        distances = np.linalg.norm(targets - values[:, :3], axis=1)
        directions = (
            targets - values[:, :3]
        ) / (distances[:, None] + 1e-6)
        acceleration_cap = min(
            float(self.model.config.a_max_xy),
            float(self.model.config.a_max_z),
        )
        braking_speeds = np.sqrt(
            2.0 * acceleration_cap * np.maximum(distances, 0.0)
        )
        target_speeds = np.minimum.reduce(
            [
                np.full_like(
                    distances,
                    self.algorithm_config.retrace_target_speed_mps,
                ),
                braking_speeds,
                np.full_like(distances, float(self.model.config.v_max)),
            ]
        )
        target_velocities = directions * target_speeds[:, None]
        desired_accelerations = (
            self.algorithm_config.retrace_gain
            * (target_velocities - values[:, 3:6])
        )
        return (
            self._batch_acceleration_to_rotors(
                values,
                desired_accelerations,
            ),
            indices,
        )

    def _batch_retrace_rollout_states(
        self,
        initial_states: np.ndarray,
        *,
        control_steps: int,
    ) -> np.ndarray:
        """Roll out independent retrace trajectories in one NumPy batch."""

        current = np.asarray(initial_states, dtype=float)
        if current.ndim != 2 or current.shape[1] != 12:
            raise ValueError("batched initial states must have shape (N, 12)")
        if control_steps < 0:
            raise ValueError("control_steps must be nonnegative")
        current = current.copy()
        indices = np.full(
            current.shape[0],
            self._active_retrace_index,
            dtype=int,
        )
        states = [current.copy()]
        for _ in range(control_steps):
            controls, indices = self._batch_retrace_controls(
                current,
                indices,
            )
            current = self._batch_step(current, controls)
            states.append(current.copy())
        return np.stack(states, axis=1)

    @staticmethod
    def _batch_smooth_min(
        values: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        minimum = np.min(array, axis=-1)
        return minimum - np.log(
            np.sum(
                np.exp(
                    -float(temperature)
                    * (array - minimum[..., None])
                ),
                axis=-1,
            )
        ) / float(temperature)

    def _batch_point_margins(
        self,
        states: np.ndarray,
        obstacles: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the exact sphere-only collision margin in a batch."""

        values = np.asarray(states, dtype=float)
        if values.shape[-1] != 12:
            raise ValueError("batched margin states must end in 12 values")
        leading_shape = values.shape[:-1]
        points = values[..., :3] + (
            float(self.model.config.rho_z)
            * self._batch_body_z(values)
        )
        obstacle_array = np.asarray(obstacles, dtype=float)
        if obstacle_array.size:
            if obstacle_array.shape[-1] != 7:
                raise ValueError("batched obstacles must end in seven values")
            obstacle_count = obstacle_array.shape[-2]
            obstacle_array = np.broadcast_to(
                obstacle_array,
                leading_shape + (obstacle_count, 7),
            )
            distances = np.linalg.norm(
                points[..., None, :] - obstacle_array[..., :3],
                axis=-1,
            )
            safe_radii = (
                obstacle_array[..., 3]
                + float(self.model.config.robot_radius)
                + float(self.controller.config.safety_margin)
            ) * float(self.controller.config.safety_scale)
            margins = np.min(distances - safe_radii, axis=-1)
        else:
            margins = np.full(leading_shape, 1e12, dtype=float)
        return margins

    def _retrace_pcbf_certificate_batched(
        self,
        state: np.ndarray,
    ) -> PolicyCertificate:
        """Evaluate all fixed-PCBF central differences in one batch."""

        value = np.asarray(state, dtype=float).reshape(12)
        active = self.controller._active_obstacles(value, self._obstacles)
        variants = [value.copy()]
        for index, step in enumerate(self.algorithm_config.gradient_steps):
            positive = value.copy()
            negative = value.copy()
            positive[index] += step
            negative[index] -= step
            variants.extend([positive, negative])
        trajectories = self._batch_retrace_rollout_states(
            np.asarray(variants),
            control_steps=self._backup_steps,
        )
        obstacle_history = self._obstacle_history(
            time_offset=0.0,
            count=self._backup_steps + 1,
            initial_obstacles=active,
        )
        rollout_margins = self._batch_point_margins(
            trajectories,
            obstacle_history[None, :, :, :],
        )
        values = self._batch_smooth_min(
            rollout_margins,
            self.controller.config.time_temperature,
        )
        base_value = float(values[0])
        gradient = np.asarray(
            [
                (values[1 + 2 * index] - values[2 + 2 * index])
                / (2.0 * step)
                for index, step in enumerate(
                    self.algorithm_config.gradient_steps
                )
            ],
            dtype=float,
        )
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm > self.controller.config.max_gradient_norm:
            gradient *= (
                self.controller.config.max_gradient_norm / gradient_norm
            )
        derivative_step = self.algorithm_config.time_derivative_step_s
        advanced_history = self._obstacle_history(
            time_offset=derivative_step,
            count=self._backup_steps + 1,
            initial_obstacles=active,
        )
        advanced_margins = self._batch_point_margins(
            trajectories[:1],
            advanced_history[None, :, :, :],
        )
        advanced_value = float(
            self._batch_smooth_min(
                advanced_margins,
                self.controller.config.time_temperature,
            )[0]
        )
        time_derivative = (
            advanced_value - base_value
        ) / derivative_step
        drift = self.model.f(value)
        control_matrix = self.model.g(value)
        normal = gradient @ control_matrix
        offset = (
            -float(gradient @ drift)
            - float(time_derivative)
            - self.algorithm_config.pcbf_alpha
            * (base_value - self.controller.config.cbf_value_buffer)
        )
        direct = self._direct_retrace_control(value)
        finite = bool(
            np.isfinite(base_value)
            and np.all(np.isfinite(gradient))
            and np.isfinite(time_derivative)
            and np.all(np.isfinite(normal))
            and np.isfinite(offset)
        )
        return PolicyCertificate(
            policy_id=self.algorithm_config.fixed_backup_policy_id,
            value=base_value if finite else -1e12,
            halfspaces=(
                CBFHalfspace(
                    normal if finite else np.zeros(4),
                    offset if finite else 1e12,
                    "retrace_waypoint:pcbf",
                ),
            ),
            backup_control=direct,
            valid=finite,
            diagnostic="" if finite else "nonfinite_retrace_pcbf",
            metadata={
                "rollout_safe": bool(finite and base_value >= 0.0),
                "fixed_backup": True,
                "backup_kind": "retrace_waypoint",
            },
        )

    def _batched_strict_multi_data(
        self,
        candidates: Sequence[PolicyCandidate],
        state: np.ndarray,
        *,
        maneuver_steps: int,
    ) -> _BatchedStrictMultiBackupData:
        """Batch every policy/finite-difference rollout without approximation."""

        candidate_tuple = tuple(candidates)
        if not candidate_tuple:
            raise ValueError("strict multi-backup batching needs candidates")
        value = np.asarray(state, dtype=float).reshape(12)
        gradient_steps = np.full(
            12,
            self.algorithm_config.backup_cbf_gradient_step,
            dtype=float,
        )
        variants = [value.copy()]
        for index, step in enumerate(gradient_steps):
            positive = value.copy()
            negative = value.copy()
            positive[index] += step
            negative[index] -= step
            variants.extend([positive, negative])
        current = np.broadcast_to(
            np.asarray(variants, dtype=float)[None, :, :],
            (len(candidate_tuple), len(variants), 12),
        ).copy()
        trajectories = [current.copy()]
        for step_index in range(self._backup_steps):
            controls = self._batch_policy_controls(
                current,
                candidate_tuple,
                force_stop=step_index >= maneuver_steps,
            )
            current = self._batch_step(current, controls)
            trajectories.append(current.copy())
        trajectory_array = np.stack(trajectories, axis=2)

        base_history = self._obstacle_history(
            time_offset=0.0,
            count=self._backup_steps + 2,
        )
        path_values_all = self._batch_point_margins(
            trajectory_array,
            base_history[
                None,
                None,
                : self._backup_steps + 1,
                :,
                :,
            ],
        )
        terminal_controls = self._batch_policy_controls(
            trajectory_array[:, :, -1, :],
            candidate_tuple,
            force_stop=True,
        )
        terminal_successors = self._batch_step(
            trajectory_array[:, :, -1, :],
            terminal_controls,
        )
        successor_margins = self._batch_point_margins(
            terminal_successors,
            base_history[None, None, -1, :, :],
        )
        final_states = trajectory_array[:, :, -1, :]
        terminal_values_all = np.minimum.reduce(
            [
                path_values_all[:, :, -1],
                successor_margins,
                self.algorithm_config.terminal_linear_speed_mps
                - np.linalg.norm(final_states[..., 3:6], axis=-1),
                self.algorithm_config.terminal_attitude_rad
                - np.linalg.norm(final_states[..., 6:9], axis=-1),
                self.algorithm_config.terminal_angular_rate_rad_s
                - np.linalg.norm(final_states[..., 9:12], axis=-1),
                self.algorithm_config.terminal_altitude_error_m
                - np.abs(
                    final_states[..., 2] - self._altitude_reference
                ),
            ]
        )

        path_values = path_values_all[:, 0, :]
        terminal_values = terminal_values_all[:, 0]
        path_gradients = np.stack(
            [
                (
                    path_values_all[:, 1 + 2 * index, :]
                    - path_values_all[:, 2 + 2 * index, :]
                )
                / (2.0 * step)
                for index, step in enumerate(gradient_steps)
            ],
            axis=-1,
        )
        terminal_gradients = np.stack(
            [
                (
                    terminal_values_all[:, 1 + 2 * index]
                    - terminal_values_all[:, 2 + 2 * index]
                )
                / (2.0 * step)
                for index, step in enumerate(gradient_steps)
            ],
            axis=-1,
        )

        derivative_step = max(
            self.algorithm_config.time_derivative_step_s,
            self.model.dt,
        )
        advanced_history = self._obstacle_history(
            time_offset=derivative_step,
            count=self._backup_steps + 1,
        )
        advanced_path_values = self._batch_point_margins(
            trajectory_array[:, 0, :, :],
            advanced_history[None, :, :, :],
        )
        path_time_derivatives = (
            advanced_path_values - path_values
        ) / derivative_step

        base_trajectories = trajectory_array[:, 0, :, :]
        policy_flows = np.empty_like(base_trajectories)
        policy_flows[:, :-1, :] = (
            base_trajectories[:, 1:, :]
            - base_trajectories[:, :-1, :]
        ) / self.model.dt
        policy_flows[:, -1, :] = (
            base_trajectories[:, -1, :]
            - base_trajectories[:, -2, :]
        ) / self.model.dt
        point_variants = np.repeat(
            base_trajectories[:, :, None, :],
            13,
            axis=2,
        )
        for index in range(12):
            point_variants[:, :, index + 1, index] += (
                self.algorithm_config.backup_cbf_gradient_step
            )
        point_margins = self._batch_point_margins(
            point_variants,
            base_history[
                None,
                : self._backup_steps + 1,
                None,
                :,
                :,
            ],
        )
        point_gradients = (
            point_margins[:, :, 1:]
            - point_margins[:, :, :1]
        ) / self.algorithm_config.backup_cbf_gradient_step
        path_flow_derivatives = np.einsum(
            "ntd,ntd->nt",
            point_gradients,
            policy_flows,
        )
        return _BatchedStrictMultiBackupData(
            path_values=path_values,
            terminal_values=terminal_values,
            path_gradients=path_gradients,
            path_time_derivatives=path_time_derivatives,
            terminal_gradients=terminal_gradients,
            path_flow_derivatives=path_flow_derivatives,
            trajectories=base_trajectories,
        )

    def _backup_candidates_batched(
        self,
        candidates: Sequence[PolicyCandidate],
        state: np.ndarray,
        nominal: np.ndarray,
        *,
        maneuver_steps: int,
    ):
        """Build strict candidates from batched, scalar-equivalent samples."""

        candidate_tuple = tuple(candidates)
        data = self._batched_strict_multi_data(
            candidate_tuple,
            state,
            maneuver_steps=maneuver_steps,
        )
        drift = self.model.f(state)
        control_matrix = self.model.g(state)
        evaluated = []
        for index, candidate in enumerate(candidate_tuple):
            direct_control = self._candidate_feedback(state, candidate)
            evaluated.append(
                evaluate_backup_cbf_candidate(
                    policy_id=candidate.name,
                    state=state,
                    nominal_control=nominal,
                    lower=self.model.input_lower_bound,
                    upper=self.model.input_upper_bound,
                    drift=drift,
                    control_matrix=control_matrix,
                    backup_closed_loop_drift=(
                        drift + control_matrix @ direct_control
                    ),
                    rollout_margins=self._policy_rollout_margins(
                        candidate,
                        maneuver_steps=maneuver_steps,
                    ),
                    path_flow_derivatives=(
                        data.path_flow_derivatives[index]
                    ),
                    formulation="strict_multi",
                    path_constraint_start_index=0,
                    gradient_steps=np.full(
                        12,
                        self.algorithm_config.backup_cbf_gradient_step,
                    ),
                    time_derivative_step=max(
                        self.algorithm_config.time_derivative_step_s,
                        self.model.dt,
                    ),
                    alpha=self.algorithm_config.backup_cbf_alpha,
                    terminal_alpha=(
                        self.algorithm_config.backup_cbf_terminal_alpha
                    ),
                    precomputed_derivatives=BackupCbfRolloutDerivatives(
                        path_values=data.path_values[index],
                        terminal_value=data.terminal_values[index],
                        path_gradients=data.path_gradients[index],
                        path_time_derivatives=(
                            data.path_time_derivatives[index]
                        ),
                        terminal_gradient=data.terminal_gradients[index],
                    ),
                )
            )
        return tuple(evaluated)

    def _policy_rollout_margins(
        self,
        candidate: PolicyCandidate,
        *,
        maneuver_steps: int,
    ):
        def oracle(
            initial_state: np.ndarray,
            time_offset: float,
        ) -> tuple[np.ndarray, float]:
            state = np.asarray(initial_state, dtype=float).reshape(12).copy()
            obstacle_history = self._obstacle_history(
                time_offset=time_offset,
                count=self._backup_steps + 2,
            )
            path_values = [
                self._point_margin(state, obstacle_history[0])
            ]
            for step_index in range(self._backup_steps):
                if step_index < maneuver_steps:
                    control = self._candidate_feedback(state, candidate)
                else:
                    control = self._terminal_stop_feedback(state)
                state = self.model.step(state, control)
                path_values.append(
                    self._point_margin(
                        state, obstacle_history[step_index + 1]
                    )
                )

            successor = self.model.step(
                state, self._terminal_stop_feedback(state)
            )
            terminal_values = [
                path_values[-1],
                self._point_margin(successor, obstacle_history[-1]),
                self.algorithm_config.terminal_linear_speed_mps
                - float(np.linalg.norm(state[3:6])),
                self.algorithm_config.terminal_attitude_rad
                - float(np.linalg.norm(state[6:9])),
                self.algorithm_config.terminal_angular_rate_rad_s
                - float(np.linalg.norm(state[9:12])),
                self.algorithm_config.terminal_altitude_error_m
                - abs(float(state[2] - self._altitude_reference)),
            ]
            return np.asarray(path_values), float(min(terminal_values))

        return oracle

    def _retrace_rollout_margins(self):
        def oracle(
            initial_state: np.ndarray,
            time_offset: float,
        ) -> tuple[np.ndarray, float]:
            # Match the warehouse single BackupCBF indexing exactly.  Its
            # ``phi`` array contains N states: phi[0] is the current state and
            # phi[1:N] are N - 1 propagated backup states.  Only phi[1:N]
            # contributes path rows; phi[-1] also supplies the one distinct
            # terminal-set row.  MPS/Gatekeeper intentionally continue to use
            # the separate N-control rollout in ``_retrace_rollout``.
            terminal_state = (
                np.asarray(initial_state, dtype=float).reshape(12).copy()
            )
            waypoint_index = self._active_retrace_index
            obstacle_history = self._obstacle_history(
                time_offset=time_offset,
                count=self._backup_steps + 1,
            )
            path_values: list[float] = [
                self._point_margin(
                    terminal_state,
                    obstacle_history[0],
                )
            ]
            for step_index in range(1, self._backup_steps):
                control, waypoint_index = self._retrace_control(
                    terminal_state,
                    waypoint_index,
                )
                terminal_state = self.model.step(terminal_state, control)
                path_values.append(
                    self._point_margin(
                        terminal_state,
                        obstacle_history[step_index],
                    )
                )
            # Legacy warehouse single BackupCBF uses terminal safety at the
            # fixed horizon plus a velocity envelope.  Its terminal row does
            # not use the strict multi-backup method's near-hover/attitude/
            # altitude proxy or one-step stop-successor invariance check.
            terminal_safety_margin = self._point_margin(
                terminal_state,
                obstacle_history[-1],
            )
            terminal_values = [
                terminal_safety_margin,
                float(self.model.config.v_max)
                - float(np.linalg.norm(terminal_state[3:6])),
            ]
            return np.asarray(path_values, dtype=float), float(
                min(terminal_values)
            )

        return oracle

    def _retrace_path_flow_derivatives(
        self,
        initial_state: np.ndarray,
    ) -> np.ndarray:
        """Single-BackupCBF flow samples for phi[0], ..., phi[N - 1]."""

        current = np.asarray(initial_state, dtype=float).reshape(12).copy()
        waypoint_index = self._active_retrace_index
        states = [current.copy()]
        for _ in range(1, self._backup_steps):
            control, waypoint_index = self._retrace_control(
                current,
                waypoint_index,
            )
            current = self.model.step(current, control)
            states.append(current.copy())
        obstacles = self._obstacle_history(
            time_offset=0.0,
            count=self._backup_steps,
        )
        return self._path_flow_derivatives(
            np.asarray(states),
            obstacles,
            range(self._backup_steps),
        )

    def _policy_path_flow_derivatives(
        self,
        initial_state: np.ndarray,
        candidate: PolicyCandidate,
        *,
        maneuver_steps: int,
    ) -> np.ndarray:
        """Strict multi-backup flow rows from phi[0] through phi[N]."""

        current = np.asarray(initial_state, dtype=float).reshape(12).copy()
        states = [current.copy()]
        for step_index in range(self._backup_steps):
            if step_index < maneuver_steps:
                control = self._candidate_feedback(current, candidate)
            else:
                control = self._terminal_stop_feedback(current)
            current = self.model.step(current, control)
            states.append(current.copy())
        obstacles = self._obstacle_history(
            time_offset=0.0,
            count=self._backup_steps + 1,
        )
        return self._path_flow_derivatives(
            np.asarray(states),
            obstacles,
            range(self._backup_steps + 1),
        )

    def _backup_candidate(
        self,
        candidate: PolicyCandidate,
        state: np.ndarray,
        nominal: np.ndarray,
        *,
        maneuver_steps: int,
    ):
        direct_control = self._candidate_feedback(state, candidate)
        drift = self.model.f(state)
        control_matrix = self.model.g(state)
        return evaluate_backup_cbf_candidate(
            policy_id=candidate.name,
            state=state,
            nominal_control=nominal,
            lower=self.model.input_lower_bound,
            upper=self.model.input_upper_bound,
            drift=drift,
            control_matrix=control_matrix,
            backup_closed_loop_drift=(
                drift + control_matrix @ direct_control
            ),
            rollout_margins=self._policy_rollout_margins(
                candidate, maneuver_steps=maneuver_steps
            ),
            path_flow_derivatives=self._policy_path_flow_derivatives(
                state,
                candidate,
                maneuver_steps=maneuver_steps,
            ),
            formulation="strict_multi",
            path_constraint_start_index=0,
            gradient_steps=np.full(
                12,
                self.algorithm_config.backup_cbf_gradient_step,
            ),
            time_derivative_step=max(
                self.algorithm_config.time_derivative_step_s,
                self.model.dt,
            ),
            alpha=self.algorithm_config.backup_cbf_alpha,
            terminal_alpha=(
                self.algorithm_config.backup_cbf_terminal_alpha
            ),
        )

    def _batched_retrace_backup_derivatives(
        self,
        state: np.ndarray,
    ) -> tuple[BackupCbfRolloutDerivatives, np.ndarray]:
        """Batch the legacy single-Backup-CBF retrace differences."""

        if self._backup_steps < 2:
            raise ValueError(
                "path flow derivatives require two rollout states"
            )
        value = np.asarray(state, dtype=float).reshape(12)
        gradient_steps = np.full(
            12,
            self.algorithm_config.backup_cbf_gradient_step,
            dtype=float,
        )
        variants = [value.copy()]
        for index, step in enumerate(gradient_steps):
            positive = value.copy()
            negative = value.copy()
            positive[index] += step
            negative[index] -= step
            variants.extend([positive, negative])
        trajectories = self._batch_retrace_rollout_states(
            np.asarray(variants),
            control_steps=self._backup_steps - 1,
        )
        obstacle_history = self._obstacle_history(
            time_offset=0.0,
            count=self._backup_steps + 1,
        )
        path_values_all = self._batch_point_margins(
            trajectories,
            obstacle_history[
                None,
                : self._backup_steps,
                :,
                :,
            ],
        )
        terminal_safety = self._batch_point_margins(
            trajectories[:, -1, :],
            obstacle_history[None, -1, :, :],
        )
        terminal_values_all = np.minimum(
            terminal_safety,
            float(self.model.config.v_max)
            - np.linalg.norm(
                trajectories[:, -1, 3:6],
                axis=-1,
            ),
        )
        path_values = path_values_all[0]
        terminal_value = float(terminal_values_all[0])
        path_gradients = np.stack(
            [
                (
                    path_values_all[1 + 2 * index]
                    - path_values_all[2 + 2 * index]
                )
                / (2.0 * step)
                for index, step in enumerate(gradient_steps)
            ],
            axis=-1,
        )
        terminal_gradient = np.asarray(
            [
                (
                    terminal_values_all[1 + 2 * index]
                    - terminal_values_all[2 + 2 * index]
                )
                / (2.0 * step)
                for index, step in enumerate(gradient_steps)
            ],
            dtype=float,
        )
        derivative_step = max(
            self.algorithm_config.time_derivative_step_s,
            self.model.dt,
        )
        advanced_history = self._obstacle_history(
            time_offset=derivative_step,
            count=self._backup_steps + 1,
        )
        advanced_path = self._batch_point_margins(
            trajectories[:1],
            advanced_history[
                None,
                : self._backup_steps,
                :,
                :,
            ],
        )[0]
        path_time_derivatives = (
            advanced_path - path_values
        ) / derivative_step

        base_trajectory = trajectories[0]
        policy_flows = np.empty_like(base_trajectory)
        policy_flows[:-1] = (
            base_trajectory[1:] - base_trajectory[:-1]
        ) / self.model.dt
        policy_flows[-1] = (
            base_trajectory[-1] - base_trajectory[-2]
        ) / self.model.dt
        point_variants = np.repeat(
            base_trajectory[:, None, :],
            13,
            axis=1,
        )
        for index in range(12):
            point_variants[:, index + 1, index] += (
                self.algorithm_config.backup_cbf_gradient_step
            )
        point_margins = self._batch_point_margins(
            point_variants,
            obstacle_history[
                None,
                : self._backup_steps,
                None,
                :,
                :,
            ][0],
        )
        point_gradients = (
            point_margins[:, 1:] - point_margins[:, :1]
        ) / self.algorithm_config.backup_cbf_gradient_step
        path_flow_derivatives = np.einsum(
            "td,td->t",
            point_gradients,
            policy_flows,
        )
        return (
            BackupCbfRolloutDerivatives(
                path_values=path_values,
                terminal_value=terminal_value,
                path_gradients=path_gradients,
                path_time_derivatives=path_time_derivatives,
                terminal_gradient=terminal_gradient,
            ),
            path_flow_derivatives,
        )

    def _retrace_backup_candidate_batched(
        self,
        state: np.ndarray,
        nominal: np.ndarray,
    ):
        derivatives, path_flow_derivatives = (
            self._batched_retrace_backup_derivatives(state)
        )
        direct_control = self._direct_retrace_control(state)
        drift = self.model.f(state)
        control_matrix = self.model.g(state)
        return evaluate_backup_cbf_candidate(
            policy_id=self.algorithm_config.fixed_backup_policy_id,
            state=state,
            nominal_control=nominal,
            lower=self.model.input_lower_bound,
            upper=self.model.input_upper_bound,
            drift=drift,
            control_matrix=control_matrix,
            backup_closed_loop_drift=(
                drift + control_matrix @ direct_control
            ),
            rollout_margins=self._retrace_rollout_margins(),
            path_flow_derivatives=path_flow_derivatives,
            formulation="single",
            path_constraint_start_index=1,
            gradient_steps=np.full(
                12,
                self.algorithm_config.backup_cbf_gradient_step,
            ),
            time_derivative_step=max(
                self.algorithm_config.time_derivative_step_s,
                self.model.dt,
            ),
            alpha=self.algorithm_config.backup_cbf_alpha,
            terminal_alpha=(
                self.algorithm_config.backup_cbf_terminal_alpha
            ),
            precomputed_derivatives=derivatives,
        )

    def _retrace_backup_candidate(
        self,
        state: np.ndarray,
        nominal: np.ndarray,
    ):
        if (
            type(self.model) is NLQuad3D
            and type(self.controller) is PLCBF_NLQuad3D
        ):
            return self._retrace_backup_candidate_batched(
                state,
                nominal,
            )
        return self._retrace_backup_candidate_scalar(state, nominal)

    def _retrace_backup_candidate_scalar(
        self,
        state: np.ndarray,
        nominal: np.ndarray,
    ):
        """Reference scalar implementation retained for external models."""

        direct_control = self._direct_retrace_control(state)
        drift = self.model.f(state)
        control_matrix = self.model.g(state)
        return evaluate_backup_cbf_candidate(
            policy_id=self.algorithm_config.fixed_backup_policy_id,
            state=state,
            nominal_control=nominal,
            lower=self.model.input_lower_bound,
            upper=self.model.input_upper_bound,
            drift=drift,
            control_matrix=control_matrix,
            backup_closed_loop_drift=(
                drift + control_matrix @ direct_control
            ),
            rollout_margins=self._retrace_rollout_margins(),
            path_flow_derivatives=self._retrace_path_flow_derivatives(state),
            formulation="single",
            path_constraint_start_index=1,
            gradient_steps=np.full(
                12,
                self.algorithm_config.backup_cbf_gradient_step,
            ),
            time_derivative_step=max(
                self.algorithm_config.time_derivative_step_s,
                self.model.dt,
            ),
            alpha=self.algorithm_config.backup_cbf_alpha,
            terminal_alpha=(
                self.algorithm_config.backup_cbf_terminal_alpha
            ),
        )

    def _branch_rollout(
        self,
        state: np.ndarray,
        candidate: PolicyCandidate,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        current = np.asarray(state, dtype=float).reshape(12).copy()
        obstacle_history = self._obstacle_history(
            time_offset=0.0,
            count=self._backup_steps + 1,
        )
        states = [current.copy()]
        controls = []
        minimum_margin = self._point_margin(current, obstacle_history[0])
        for step_index in range(self._backup_steps):
            control = self._candidate_feedback(current, candidate)
            current = self.model.step(current, control)
            controls.append(control.copy())
            states.append(current.copy())
            minimum_margin = min(
                minimum_margin,
                self._point_margin(
                    current, obstacle_history[step_index + 1]
                ),
            )
        return (
            np.asarray(states),
            np.asarray(controls),
            float(minimum_margin),
        )

    def _nominal_reference(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        current = np.asarray(state, dtype=float).reshape(12).copy()
        states = [current.copy()]
        controls = []
        for _ in range(self._backup_steps):
            control = self._nominal_feedback(current)
            current = self.model.step(current, control)
            controls.append(control.copy())
            states.append(current.copy())
        return np.asarray(states), np.asarray(controls)

    def _state_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(12, -np.inf)
        upper = np.full(12, np.inf)
        # Big-M position tubes require finite numerical bounds.  These wide
        # limits are solver envelopes, independent of ``WorldBounds``; the
        # latter reflects hazards and is not a physical robot-wall safe set in
        # the nonlinear Quad3D playground protocol.
        # Keep the envelope comfortably inside the configured position Big-M
        # while remaining far beyond every registered NL-Quad3D route and any
        # physically reachable backup-horizon displacement.
        lower[:3] = -100.0
        upper[:3] = 100.0
        lower[3:6] = -self.model.config.v_max
        upper[3:6] = self.model.config.v_max
        lower[6:9] = -np.pi
        upper[6:9] = np.pi
        lower[9:12] = -self.model.config.body_rate_max
        upper[9:12] = self.model.config.body_rate_max
        return lower, upper

    def _solve_mi_mpc(
        self,
        state: np.ndarray,
        nominal: np.ndarray,
        candidates: Sequence[PolicyCandidate],
    ) -> BaselineDecision:
        import time

        started_at = time.perf_counter()
        candidate_tuple = tuple(candidates)
        if not candidate_tuple:
            raise ValueError("NL-Quad3D MI-MPC requires policy branches")
        branch_data = [
            self._branch_rollout(state, candidate)
            for candidate in candidate_tuple
        ]
        branch_states = np.asarray([item[0] for item in branch_data])
        branch_controls = np.asarray([item[1] for item in branch_data])
        branch_safety = np.asarray([item[2] for item in branch_data])
        nominal_states, nominal_controls = self._nominal_reference(state)
        matrices_a, matrices_b, affine = linearize_discrete_trajectory(
            self.model.step,
            nominal_states,
            nominal_controls,
            state_steps=np.asarray(
                self.algorithm_config.gradient_steps, dtype=float
            ),
            control_steps=np.full(4, 1e-4),
        )
        state_lower, state_upper = self._state_bounds()
        safest_branch = int(np.argmax(branch_safety))
        # Match the warehouse MI-MPC emergency action: the safest branch's
        # first control, mildly blended toward nominal.  This is not the fixed
        # retrace backup used by the single-backup comparison methods.
        fallback = self.model.saturate_rotors(
            0.75 * branch_controls[safest_branch, 0] + 0.25 * nominal
        )
        problem = BigMTrajectoryMPCProblem(
            x0=state,
            A=matrices_a,
            B=matrices_b,
            c=affine,
            branch_states=branch_states,
            branch_controls=branch_controls,
            branch_safety=branch_safety,
            state_lower=state_lower,
            state_upper=state_upper,
            control_lower=self.model.input_lower_bound,
            control_upper=self.model.input_upper_bound,
            position_indices=(0, 1, 2),
            velocity_indices=(3, 4, 5),
            tracking_target=self._goal,
            terminal_target=self._goal,
            nominal_control=nominal,
            fallback_control=fallback,
        )
        result = solve_big_m_trajectory_mpc(
            problem,
            BigMTrajectoryMPCConfig(
                safety_threshold=0.0,
                position_tube=self.algorithm_config.mi_position_tube_m,
                early_control_tube=(
                    self.algorithm_config.mi_control_tube_thrust
                ),
                early_control_steps=(
                    self.algorithm_config.mi_control_tube_steps
                ),
                big_m_position=(
                    self.algorithm_config.mi_big_m_position_m
                ),
                big_m_control=(
                    self.algorithm_config.mi_big_m_control_thrust
                ),
                big_m_safety=self.algorithm_config.mi_big_m_safety,
                tracking_weight=(
                    self.algorithm_config.mi_tracking_weight
                ),
                terminal_weight=(
                    self.algorithm_config.mi_terminal_weight
                ),
                velocity_weight=(
                    self.algorithm_config.mi_velocity_weight
                ),
                control_weight=(
                    self.algorithm_config.mi_control_weight
                ),
                nominal_weight=(
                    self.algorithm_config.mi_nominal_weight
                ),
                time_limit_s=self.algorithm_config.mi_time_limit_s,
                mip_rel_gap=self.algorithm_config.mi_mip_rel_gap,
            ),
        )
        self.last_mi_mpc_result = result
        control = (
            fallback if result.control is None else np.asarray(result.control)
        )
        selected_policy = (
            candidate_tuple[safest_branch].name
            if result.selected_branch is None
            else candidate_tuple[result.selected_branch].name
        )
        return BaselineDecision(
            method=BenchmarkMethod.MI_MPC.value,
            control=control,
            policy_id=selected_policy,
            feasible=result.feasible,
            status=result.status,
            used_fallback=result.used_fallback,
            objective=(
                float("inf")
                if result.objective is None
                else float(result.objective)
            ),
            solve_time_s=time.perf_counter() - started_at,
        )

    @staticmethod
    def _shield_decision(
        method: BenchmarkMethod,
        decision: ShieldDecision,
        nominal: np.ndarray,
    ) -> BaselineDecision:
        delta = decision.control - nominal
        return BaselineDecision(
            method=method.value,
            control=decision.control,
            policy_id=decision.committed_trajectory.backup_policy_id,
            feasible=decision.feasible,
            status=decision.status,
            used_fallback=decision.used_committed_backup,
            objective=float(delta @ delta),
            solve_time_s=decision.solve_time_s,
        )

    @staticmethod
    def _backup_decision(
        method: BenchmarkMethod,
        decision: BackupCbfDecision,
    ) -> BaselineDecision:
        return BaselineDecision(
            method=method.value,
            control=decision.control,
            policy_id=decision.policy_id,
            feasible=decision.feasible,
            status=decision.status,
            used_fallback=decision.used_fallback,
            objective=decision.objective,
            solve_time_s=decision.solve_time_s,
        )

    def solve(
        self,
        method: BenchmarkMethod | str,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
        nominal: np.ndarray,
        certificates: Sequence[PolicyCertificate] = (),
        candidates: Sequence[PolicyCandidate] | None = None,
        active_waypoint_index: int | None = None,
        safe_value_threshold: float = 0.0,
    ) -> BaselineDecision:
        """Solve one method; the only cross-step state is MPS/Gatekeeper state."""

        parsed = (
            method if isinstance(method, BenchmarkMethod) else BenchmarkMethod(method)
        )
        value = np.asarray(state, dtype=float).reshape(12)
        self._goal = np.asarray(goal, dtype=float).reshape(3)
        self._obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 7).copy()
        self._trajectory_obstacle_history_cache = None
        self._altitude_reference = float(value[2])
        if active_waypoint_index is not None:
            index = int(active_waypoint_index)
            if index < 0 or index >= self._waypoints.shape[0]:
                raise ValueError("active_waypoint_index is outside the route")
            self._active_waypoint_index = index
        self._active_retrace_index = max(0, self._active_waypoint_index - 1)
        self._prepare_retrace_rollout()
        nominal_array = self.model.saturate_rotors(nominal)
        lower = self.model.input_lower_bound
        upper = self.model.input_upper_bound

        if parsed is BenchmarkMethod.POLICY_PCBF:
            retrace_certificate = self._retrace_pcbf_certificate(value)
            return solve_policy_pcbf(
                (retrace_certificate,),
                nominal_array,
                lower,
                upper,
                backup_policy_id=self.algorithm_config.fixed_backup_policy_id,
            )
        if parsed is BenchmarkMethod.PLCBF:
            # Use the controller's exact playground-matched max operator and
            # selected-policy QP rather than the generic selector.
            control = self.controller.solve_control_problem(
                value,
                self._goal,
                self._obstacles,
                control_ref=nominal_array,
            )
            decision = self.controller.last_decision
            assert decision is not None
            delta = control - nominal_array
            return BaselineDecision(
                method=parsed.value,
                control=control,
                policy_id=decision.policy_id,
                feasible=not decision.diagnostics.used_fallback,
                status=self.controller.last_status,
                used_fallback=decision.diagnostics.used_fallback,
                objective=float(delta @ delta),
                solve_time_s=0.0,
                policy_decision=decision,
            )
        if parsed is BenchmarkMethod.LIBRARY_PCBF_MI:
            return solve_library_pcbf_mi(
                certificates,
                nominal_array,
                lower,
                upper,
                safe_value_threshold=safe_value_threshold,
                emergency_policy_id="stop",
            )
        if parsed is BenchmarkMethod.MPS:
            return self._shield_decision(
                parsed, self.mps.solve(value), nominal_array
            )
        if parsed is BenchmarkMethod.GATEKEEPER:
            return self._shield_decision(
                parsed, self.gatekeeper.solve(value), nominal_array
            )

        candidate_tuple = tuple(
            self.controller.candidates(self._goal)
            if candidates is None
            else candidates
        )
        if parsed is BenchmarkMethod.BACKUP_CBF:
            evaluated = self._retrace_backup_candidate(
                value,
                nominal_array,
            )
            decision = solve_fixed_backup_cbf(
                evaluated,
                direct_backup_control=self._direct_retrace_control(value),
                lower=lower,
                upper=upper,
            )
            return self._backup_decision(parsed, decision)
        if parsed is BenchmarkMethod.MULTI_BACKUP_CBF_MI:
            maneuver_steps = int(
                round(
                    self.algorithm_config.multi_backup_maneuver_fraction
                    * self._backup_steps
                )
            )
            if (
                type(self.model) is NLQuad3D
                and type(self.controller) is PLCBF_NLQuad3D
            ):
                evaluated = self._backup_candidates_batched(
                    candidate_tuple,
                    value,
                    nominal_array,
                    maneuver_steps=maneuver_steps,
                )
            else:
                # Test doubles and externally supplied dynamics retain the
                # generic scalar oracle path.
                evaluated = tuple(
                    self._backup_candidate(
                        candidate,
                        value,
                        nominal_array,
                        maneuver_steps=maneuver_steps,
                    )
                    for candidate in candidate_tuple
                )
            decision = solve_multi_backup_cbf_min_intervention(
                evaluated,
                direct_backup_controls={
                    candidate.name: self._candidate_feedback(
                        value, candidate
                    )
                    for candidate in candidate_tuple
                },
                lower=lower,
                upper=upper,
                emergency_policy_id="stop",
            )
            return self._backup_decision(parsed, decision)
        if parsed is BenchmarkMethod.MI_MPC:
            return self._solve_mi_mpc(
                value, nominal_array, self.mi_mpc_candidates()
            )
        raise AssertionError(f"unhandled nonlinear Quad3D method {parsed}")


__all__ = [
    "NLQuad3DBaselineConfig",
    "NLQuad3DBaselineSuite",
]
