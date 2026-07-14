"""Multi-backup CBF with minimum-intervention selection for Quad3D."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time
from typing import Dict, Iterable, Optional, Tuple

import cvxpy as cp
import jax.numpy as jnp
import numpy as np

from safe_control.position_control.backup_cbf_qp import BackupCBF

from examples.additional_baseline_control_utils import audit_cvxpy_inequalities
from examples.warehouse.controllers.policies_quad3d_jax import (
    WaypointPolicyParams,
)
from examples.warehouse.algorithms.additional_baseline_control_quad3d import (
    SOLVER_INPUT_TOL,
    project_quad3d_solver_control_with_diagnostics,
)
from examples.warehouse.algorithms.plcbf_quad3d import PLCBF_Quad3D


ACCEPTED_QP_STATUSES = frozenset({"optimal", "optimal_inaccurate"})
_OSQP_AUDIT_TOL = SOLVER_INPUT_TOL
_SCS_AUDIT_TOL = 1e-4


@dataclass(frozen=True)
class CandidateCBFResult:
    policy_name: str
    feasible: bool
    u: Optional[np.ndarray]
    objective: float
    solver_status: str
    rollout_safe: Optional[bool]
    terminal_safe: Optional[bool]
    solve_time_sec: float
    qp_solved: bool = False
    error: Optional[str] = None
    raw_u: Optional[np.ndarray] = None
    projected_u: Optional[np.ndarray] = None
    projection_occurred: bool = False
    projection_delta_inf: float = 0.0
    post_projection_constraints_satisfied: Optional[bool] = None
    max_post_projection_constraint_violation: Optional[float] = None
    max_post_projection_violation_ratio: Optional[float] = None
    constraint_audit_atol: Optional[float] = None
    constraint_audit_rtol: Optional[float] = None
    constraint_audit_count: int = 0
    solver_name: Optional[str] = None


class _FrozenGhostPredictor:
    def __init__(self, obstacles: Iterable[dict]):
        active = [dict(obs) for obs in obstacles if obs.get("active", True)]
        self._obstacles = tuple(active)
        self._x = np.asarray([float(obs["x"]) for obs in active], dtype=float)
        self._y = np.asarray([float(obs["y"]) for obs in active], dtype=float)
        self._vx = np.asarray(
            [float(obs.get("vx", 0.0)) for obs in active], dtype=float
        )
        self._vy = np.asarray(
            [float(obs.get("vy", 0.0)) for obs in active], dtype=float
        )
        self._radius = np.asarray(
            [float(obs["radius"]) for obs in active], dtype=float
        )

    def arrays_at(self, t: float):
        px = self._x + self._vx * float(t)
        py = self._y + self._vy * float(t)
        px = np.where(px < 2.0, 4.0 - px, np.where(px > 98.0, 196.0 - px, px))
        py = np.where(py < 2.0, 4.0 - py, np.where(py > 98.0, 196.0 - py, py))
        return px, py, self._radius

    def __call__(self, t: float):
        predicted = []
        for obs in self._obstacles:
            px = float(obs["x"]) + float(obs.get("vx", 0.0)) * float(t)
            py = float(obs["y"]) + float(obs.get("vy", 0.0)) * float(t)
            if px < 2.0:
                px = 4.0 - px
            elif px > 98.0:
                px = 196.0 - px
            if py < 2.0:
                py = 4.0 - py
            elif py > 98.0:
                py = 196.0 - py
            predicted.append(
                {
                    "x": px,
                    "y": py,
                    "radius": float(obs["radius"]),
                    "vx": float(obs.get("vx", 0.0)),
                    "vy": float(obs.get("vy", 0.0)),
                }
            )
        if not predicted:
            return None
        if len(predicted) == 1:
            return predicted[0]
        return predicted


class _QuadPolicyAdapter:
    def __init__(self, policy_name: str, policy_type: str, params):
        self.policy_name = policy_name
        self.policy_type = policy_type
        self.params = params
        self._stop_params = params if policy_type == "stop" else None
        self._maneuver_prefix_steps: Optional[int] = None
        self._rollout_step = 0

    def set_params(self, policy_type: str, params) -> None:
        if policy_type != self.policy_type:
            raise AssertionError(
                f"Policy type changed for {self.policy_name}: "
                f"{self.policy_type!r} -> {policy_type!r}"
            )
        self.params = params

    def configure_stop_tail(self, stop_params, maneuver_prefix_steps: int) -> None:
        self._stop_params = stop_params
        self._maneuver_prefix_steps = max(0, int(maneuver_prefix_steps))
        self._rollout_step = 0

    def set_rollout_step(self, step_index: int) -> None:
        self._rollout_step = max(0, int(step_index))

    @property
    def using_stop_tail(self) -> bool:
        return (
            self.policy_type == "stop"
            or (
                self._maneuver_prefix_steps is not None
                and self._rollout_step >= self._maneuver_prefix_steps
            )
        )

    def prepare_rollout(self, state) -> None:
        del state
        self._rollout_step = 0

    def compute_terminal_stop_control(self, state):
        if self._stop_params is None:
            raise RuntimeError(f"No stop-tail policy configured for {self.policy_name}")
        return self._compute_control_for("stop", self._stop_params, state)

    def compute_control(self, state, target=None):
        del target
        if self.using_stop_tail:
            if self._stop_params is None:
                raise RuntimeError(
                    f"No stop-tail policy configured for {self.policy_name}"
                )
            return self._compute_control_for("stop", self._stop_params, state)
        return self._compute_control_for(self.policy_type, self.params, state)

    @staticmethod
    def _compute_control_for(policy_type: str, params, state):
        x = np.asarray(state, dtype=float).reshape(-1)
        ctrl = params.ctrl

        def clip_xy(ax, ay):
            norm = np.sqrt(ax * ax + ay * ay + 1e-8)
            if norm > float(ctrl.a_max_xy):
                scale = float(ctrl.a_max_xy) / norm
                ax *= scale
                ay *= scale
            return ax, ay

        def accel_to_u(ax, ay, az):
            theta, phi, psi = x[3], x[4], x[5]
            q, p, r = x[9], x[10], x[11]
            theta_des = ax / float(ctrl.g)
            phi_des = -ay / float(ctrl.g)
            force_des = float(ctrl.m) * az
            tau_y = float(ctrl.Iy) * (
                float(ctrl.K_ang) * (theta_des - theta)
                + float(ctrl.Kd_ang) * (0.0 - q)
            )
            tau_x = float(ctrl.Ix) * (
                float(ctrl.K_ang) * (phi_des - phi)
                + float(ctrl.Kd_ang) * (0.0 - p)
            )
            tau_z = float(ctrl.Iz) * (
                float(ctrl.K_ang) * (0.0 - psi)
                + float(ctrl.Kd_ang) * (0.0 - r)
            )
            wrench = np.array([force_des, tau_y, tau_x, tau_z], dtype=float)
            u = np.asarray(ctrl.B2_inv, dtype=float) @ wrench
            return np.clip(u, float(ctrl.u_min), float(ctrl.u_max))

        if policy_type == "angle":
            vx_des = float(params.target_speed) * np.cos(
                float(params.target_angle)
            )
            vy_des = float(params.target_speed) * np.sin(
                float(params.target_angle)
            )
            ax = float(params.Kp_v) * (vx_des - x[6])
            ay = float(params.Kp_v) * (vy_des - x[7])
            ax, ay = clip_xy(ax, ay)
        elif policy_type == "stop":
            ax = -float(params.Kp_v) * x[6]
            ay = -float(params.Kp_v) * x[7]
            ax, ay = clip_xy(ax, ay)
        elif policy_type == "waypoint":
            waypoints = np.asarray(params.waypoints, dtype=float)
            index = int(np.clip(int(params.current_wp_idx), 0, len(waypoints) - 1))
            previous_index = max(index - 1, 0)
            target_xy = waypoints[index]
            previous_xy = waypoints[previous_index]
            segment = target_xy - previous_xy
            segment_norm = np.sqrt(float(segment @ segment) + 1e-8)
            if segment_norm > 1e-6:
                segment_direction = segment / segment_norm
            else:
                delta = target_xy - x[:2]
                segment_direction = delta / (np.sqrt(float(delta @ delta)) + 1e-6)
            perpendicular = np.array(
                [-segment_direction[1], segment_direction[0]], dtype=float
            )
            distance_along = float((target_xy - x[:2]) @ segment_direction)
            braking_speed = np.sqrt(
                2.0 * float(ctrl.a_max_xy) * abs(distance_along)
            )
            longitudinal_speed = min(float(params.v_max), braking_speed)
            longitudinal_sign = 1.0 if distance_along >= 0.0 else -1.0
            lateral_error = float((x[:2] - previous_xy) @ perpendicular)
            lateral_speed = np.clip(
                -float(params.K_lat) * lateral_error,
                -float(params.v_lat_max),
                float(params.v_lat_max),
            )
            desired_velocity = (
                longitudinal_sign * longitudinal_speed * segment_direction
                + lateral_speed * perpendicular
            )
            desired_norm = np.sqrt(float(desired_velocity @ desired_velocity) + 1e-8)
            if desired_norm > float(params.v_max):
                desired_velocity *= float(params.v_max) / desired_norm
            ax = float(params.Kp) * (desired_velocity[0] - x[6])
            ay = float(params.Kp) * (desired_velocity[1] - x[7])
            ax, ay = clip_xy(ax, ay)
        else:
            raise ValueError(
                f"Unsupported Quad3D PL-CBF policy type {policy_type!r}"
            )
        az = float(ctrl.Kp_z) * (float(ctrl.z_ref) - x[2]) - float(ctrl.Kd_z) * x[8]
        return accel_to_u(ax, ay, az).reshape(-1)


class _StrictBackupCBFCandidate(BackupCBF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = int(np.ceil(self.backup_horizon / self.dt)) + 1

    def configure_terminal_proxy(
        self,
        *,
        linear_speed_tol: float,
        attitude_tol: float,
        angular_rate_tol: float,
        altitude_error_tol: float,
    ) -> None:
        self.terminal_linear_speed_tol = float(linear_speed_tol)
        self.terminal_attitude_tol = float(attitude_tol)
        self.terminal_angular_rate_tol = float(angular_rate_tol)
        self.terminal_altitude_error_tol = float(altitude_error_tol)

    def _linear_rk4_step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_rk4_ad"):
            A = np.asarray(self.robot.A, dtype=float)
            B = np.asarray(self.robot.B, dtype=float)
            identity = np.eye(self.n_states)
            A2 = A @ A
            A3 = A2 @ A
            A4 = A3 @ A
            dt = float(self.dt)
            self._rk4_ad = (
                identity
                + dt * A
                + 0.5 * dt**2 * A2
                + (dt**3 / 6.0) * A3
                + (dt**4 / 24.0) * A4
            )
            self._rk4_bd = (
                dt * identity
                + 0.5 * dt**2 * A
                + (dt**3 / 6.0) * A2
                + (dt**4 / 24.0) * A3
            ) @ B
        next_state = self._rk4_ad @ state + self._rk4_bd @ control
        next_state = np.asarray(next_state, dtype=float).reshape(-1)
        next_state[3:6] = (next_state[3:6] + np.pi) % (2.0 * np.pi) - np.pi
        return next_state

    def _integrate_state_trajectory(self, x0):
        phi = np.zeros((self.N, self.n_states))
        state = np.asarray(x0, dtype=float).reshape(-1).copy()
        phi[0] = state

        for step in range(1, self.N):
            if hasattr(self.backup_controller, "set_rollout_step"):
                self.backup_controller.set_rollout_step(step - 1)
            control = np.asarray(self._backup_control(state), dtype=float).reshape(-1)
            state = self._linear_rk4_step(state, control)
            phi[step] = state
        return phi

    def _compute_rollout_sensitivities(self, phi):
        sensitivities = np.zeros((self.N, self.n_states, self.n_states))
        sensitivity = np.eye(self.n_states)
        sensitivities[0] = sensitivity
        eps = 1e-5

        for step in range(1, self.N):
            state = np.asarray(phi[step - 1], dtype=float)
            next_state = np.asarray(phi[step], dtype=float)
            if hasattr(self.backup_controller, "set_rollout_step"):
                self.backup_controller.set_rollout_step(step - 1)
            discrete_jacobian = np.zeros((self.n_states, self.n_states))
            for state_index in range(self.n_states):
                perturbed = state.copy()
                perturbed[state_index] += eps
                perturbed_control = np.asarray(
                    self._backup_control(perturbed), dtype=float
                ).reshape(-1)
                perturbed_next = self._linear_rk4_step(
                    perturbed, perturbed_control
                )
                discrete_jacobian[:, state_index] = (
                    perturbed_next - next_state
                ) / eps

            sensitivity = discrete_jacobian @ sensitivity
            sensitivities[step] = sensitivity
        return sensitivities

    def _integrate_backup_trajectory(self, x0):
        phi = self._integrate_state_trajectory(x0)
        return phi, self._compute_rollout_sensitivities(phi)

    def set_environment(self, env):
        super().set_environment(env)
        static_xy = []
        static_radius = []
        for obstacle in getattr(env, "obstacles", []):
            static_xy.append(
                [float(obstacle.get("x", 0.0)), float(obstacle.get("y", 0.0))]
            )
            if "spec" in obstacle:
                static_radius.append(float(obstacle["spec"].get("radius", 2.5)))
            else:
                static_radius.append(float(obstacle.get("radius", 1.0)))
        self._static_xy = np.asarray(static_xy, dtype=float).reshape(-1, 2)
        self._static_radius = np.asarray(static_radius, dtype=float)

    def _h_safety(self, x, t=0.0):
        if self.env is None or not (
            hasattr(self.env, "width") and hasattr(self.env, "height")
        ):
            return super()._h_safety(x, t)

        state = np.asarray(x, dtype=float).reshape(-1)
        position = state[:2]
        robot_radius = float(self.robot_spec.get("radius", 0.5))
        values = [
            position[0] - robot_radius,
            float(self.env.width) - position[0] - robot_radius,
            position[1] - robot_radius,
            float(self.env.height) - position[1] - robot_radius,
        ]

        if getattr(self, "_static_xy", np.empty((0, 2))).size:
            static_distance = np.linalg.norm(
                self._static_xy - position[None, :], axis=1
            )
            values.append(
                float(
                    np.min(
                        static_distance
                        - float(self.robot_spec.get("radius", 1.0))
                        - self._static_radius
                        - float(self.safety_margin)
                    )
                )
            )

        predictor = self.moving_obstacles
        if isinstance(predictor, _FrozenGhostPredictor):
            px, py, radii = predictor.arrays_at(t)
            if radii.size:
                dynamic_distance = np.sqrt(
                    np.square(position[0] - px) + np.square(position[1] - py)
                )
                values.append(
                    float(
                        np.min(
                            dynamic_distance
                            - robot_radius
                            - radii
                            - float(self.safety_margin)
                        )
                    )
                )
        elif predictor is not None:
            return super()._h_safety(x, t)

        return float(min(values))

    def _grad_h_safety(self, x, t=0.0, h0=None):
        eps = 1e-5
        state = np.asarray(x, dtype=float).reshape(-1)
        gradient = np.zeros(self.n_states)
        value = self._h_safety(state, t) if h0 is None else float(h0)
        for index in (0, 1):
            perturbed = state.copy()
            perturbed[index] += eps
            gradient[index] = (self._h_safety(perturbed, t) - value) / eps
        return gradient

    def _near_hover_components(
        self, state: np.ndarray, time_value: float, prefix: str = ""
    ) -> Dict[str, float]:
        angles = (state[3:6] + np.pi) % (2.0 * np.pi) - np.pi
        z_ref = float(self.robot_spec.get("z_ref", 0.0))
        return {
            f"{prefix}safety": float(self._h_safety(state, time_value)),
            f"{prefix}linear_speed": self.terminal_linear_speed_tol
            - float(np.linalg.norm(state[6:9])),
            f"{prefix}attitude": self.terminal_attitude_tol
            - float(np.linalg.norm(angles)),
            f"{prefix}angular_rate": self.terminal_angular_rate_tol
            - float(np.linalg.norm(state[9:12])),
            f"{prefix}altitude_error": self.terminal_altitude_error_tol
            - abs(float(state[2]) - z_ref),
        }

    def _terminal_envelope_components(self, x) -> Dict[str, float]:
        state = np.asarray(x, dtype=float).reshape(-1)
        if state.shape != (self.n_states,):
            raise ValueError(
                f"Terminal state shape {state.shape} != ({self.n_states},)"
            )

        terminal_time = float(self.backup_horizon)
        components = self._near_hover_components(state, terminal_time)

        if hasattr(self.backup_controller, "compute_terminal_stop_control"):
            terminal_control = np.asarray(
                self.backup_controller.compute_terminal_stop_control(state),
                dtype=float,
            ).reshape(-1)
        else:
            terminal_control = np.asarray(
                self._backup_control(state), dtype=float
            ).reshape(-1)
        u_min = float(self.robot_spec.get("u_min", -10.0))
        u_max = float(self.robot_spec.get("u_max", 10.0))
        control_valid = (
            terminal_control.shape == (self.n_controls,)
            and np.all(np.isfinite(terminal_control))
            and np.all(terminal_control >= u_min)
            and np.all(terminal_control <= u_max)
        )
        if not control_valid:
            components["successor_control"] = float("-inf")
            return components

        successor = self._linear_rk4_step(state, terminal_control)
        components.update(
            self._near_hover_components(
                successor, terminal_time + self.dt, prefix="successor_"
            )
        )
        return components

    def _h_terminal(self, x):
        components = self._terminal_envelope_components(x)
        values = np.asarray(tuple(components.values()), dtype=float)
        if not np.all(np.isfinite(values)):
            return float("-inf")
        return float(np.min(values))

    def _grad_h_terminal(self, x, h0=None):
        eps = 1e-5
        state = np.asarray(x, dtype=float).reshape(-1)
        gradient = np.zeros(self.n_states)
        value = self._h_terminal(state) if h0 is None else float(h0)
        for index in range(self.n_states):
            perturbed = state.copy()
            perturbed[index] += eps
            gradient[index] = (self._h_terminal(perturbed) - value) / eps
        return gradient

    def solve_candidate(self, robot_state, u_nom: np.ndarray) -> CandidateCBFResult:
        started = time.perf_counter()
        name = getattr(self.backup_controller, "policy_name", "unknown")
        qp_solved = False
        try:
            x0 = np.asarray(robot_state, dtype=float).reshape(-1)
            u_ref = np.asarray(u_nom, dtype=float).reshape(-1)
            if u_ref.shape != (self.n_controls,):
                raise ValueError(
                    f"Nominal control shape {u_ref.shape} != ({self.n_controls},)"
                )

            if hasattr(self.backup_controller, "prepare_rollout"):
                self.backup_controller.prepare_rollout(x0.copy())

            phi = self._integrate_state_trajectory(x0)
            h_values = [self._h_safety(phi[i], i * self.dt) for i in range(len(phi))]
            h_safety_min = float(np.min(h_values))
            h_terminal = float(self._h_terminal(phi[-1]))
            self._last_h_min = min(h_safety_min, h_terminal)
            self.latest_backup_trajectory = phi.copy()

            rollout_safe = bool(
                np.all(np.isfinite(h_values)) and h_safety_min >= 0.0
            )
            terminal_safe = bool(
                np.isfinite(h_terminal) and h_terminal >= 0.0
            )
            if not rollout_safe or not terminal_safe:
                if not np.all(np.isfinite(h_values)) or not np.isfinite(h_terminal):
                    status = "invalid_certificate"
                elif not rollout_safe:
                    status = "uncertified_rollout"
                else:
                    status = "uncertified_terminal"
                return CandidateCBFResult(
                    policy_name=name,
                    feasible=False,
                    u=None,
                    objective=float("inf"),
                    solver_status=status,
                    rollout_safe=rollout_safe,
                    terminal_safe=terminal_safe,
                    solve_time_sec=time.perf_counter() - started,
                    qp_solved=False,
                    error=None,
                )

            sensitivities = self._compute_rollout_sensitivities(phi)
            f0 = np.asarray(self._dynamics_f(x0), dtype=float).reshape(-1)
            g0 = np.asarray(self._dynamics_g(x0), dtype=float)
            lhs_rows = []
            rhs_values = []

            for i in range(self.N):
                x_i = phi[i]
                sensitivity = sensitivities[i]
                t_i = i * self.dt
                h_value = float(h_values[i])
                grad_h = np.asarray(
                    self._grad_h_safety(x_i, t_i, h_value), dtype=float
                )

                if self.moving_obstacles is not None:
                    h_next_t = float(self._h_safety(x_i, t_i + self.dt))
                    dh_dt = (h_next_t - h_value) / self.dt
                else:
                    dh_dt = 0.0

                if i < self.N - 1:
                    f_policy = (phi[i + 1] - phi[i]) / self.dt
                else:
                    f_policy = (phi[i] - phi[i - 1]) / self.dt

                lhs = grad_h @ sensitivity @ g0
                rhs = (
                    -(grad_h @ sensitivity @ f0)
                    + (grad_h @ f_policy)
                    - dh_dt
                    - self._alpha(h_value)
                )
                if np.linalg.norm(lhs) > 1e-6:
                    lhs_rows.append(lhs)
                    rhs_values.append(rhs)
                elif rhs > 1e-8:
                    return CandidateCBFResult(
                        policy_name=name,
                        feasible=False,
                        u=None,
                        objective=float("inf"),
                        solver_status="infeasible_constant_path_constraint",
                        rollout_safe=True,
                        terminal_safe=True,
                        solve_time_sec=time.perf_counter() - started,
                        qp_solved=False,
                        error="A zero-control-coefficient path row is violated",
                    )

            x_terminal = phi[-1]
            sensitivity_terminal = sensitivities[-1]
            grad_terminal = np.asarray(
                self._grad_h_terminal(x_terminal, h_terminal), dtype=float
            )
            lhs_terminal = grad_terminal @ sensitivity_terminal @ g0
            rhs_terminal = -(
                grad_terminal @ sensitivity_terminal @ f0
                + self._alpha_terminal(h_terminal)
            )
            if np.linalg.norm(lhs_terminal) > 1e-6:
                lhs_rows.append(lhs_terminal)
                rhs_values.append(rhs_terminal)
            elif rhs_terminal > 1e-8:
                return CandidateCBFResult(
                    policy_name=name,
                    feasible=False,
                    u=None,
                    objective=float("inf"),
                    solver_status="infeasible_constant_terminal_constraint",
                    rollout_safe=True,
                    terminal_safe=True,
                    solve_time_sec=time.perf_counter() - started,
                    qp_solved=False,
                    error="A zero-control-coefficient terminal row is violated",
                )

            u_limit = float(self.robot_spec.get("u_max", 10.0))
            u_scale = np.full(self.n_controls, u_limit, dtype=float)
            u_ref = np.clip(u_ref, -u_scale, u_scale)
            u_ref_scaled = u_ref / u_scale

            if not lhs_rows:
                realized_error = self.Q_u * (
                    u_ref_scaled
                    - np.asarray(u_nom, dtype=float).reshape(-1) / u_scale
                )
                return CandidateCBFResult(
                    policy_name=name,
                    feasible=True,
                    u=u_ref.copy(),
                    objective=float(realized_error @ realized_error),
                    solver_status="no_constraints",
                    rollout_safe=rollout_safe,
                    terminal_safe=terminal_safe,
                    solve_time_sec=time.perf_counter() - started,
                    qp_solved=False,
                    error=None,
                )

            lhs_matrix = np.asarray(lhs_rows, dtype=float)
            rhs_vector = np.asarray(rhs_values, dtype=float)
            if not (
                np.all(np.isfinite(lhs_matrix))
                and np.all(np.isfinite(rhs_vector))
            ):
                raise ValueError("Non-finite Backup-CBF constraint")

            u_scaled = cp.Variable(self.n_controls)
            weighted_error = np.diag(self.Q_u) @ (u_scaled - u_ref_scaled)
            objective = cp.Minimize(cp.sum_squares(weighted_error))
            constraints = [
                (lhs_matrix @ np.diag(u_scale)) @ u_scaled >= rhs_vector,
                u_scaled >= -1.0,
                u_scaled <= 1.0,
            ]
            problem = cp.Problem(objective, constraints)
            qp_solved = True

            status = "failure"
            try:
                problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
                status = str(problem.status)
            except Exception:
                try:
                    problem.solve(solver=cp.SCS, verbose=False)
                    status = str(problem.status)
                except Exception as exc:
                    return CandidateCBFResult(
                        policy_name=name,
                        feasible=False,
                        u=None,
                        objective=float("inf"),
                        solver_status="failure",
                        rollout_safe=rollout_safe,
                        terminal_safe=terminal_safe,
                        solve_time_sec=time.perf_counter() - started,
                        qp_solved=True,
                        error=str(exc),
                    )

            if status not in ACCEPTED_QP_STATUSES or u_scaled.value is None:
                return CandidateCBFResult(
                    policy_name=name,
                    feasible=False,
                    u=None,
                    objective=float("inf"),
                    solver_status=status,
                    rollout_safe=rollout_safe,
                    terminal_safe=terminal_safe,
                    solve_time_sec=time.perf_counter() - started,
                    qp_solved=True,
                    error=None,
                )

            scaled_solution = np.asarray(u_scaled.value, dtype=float).reshape(-1)
            raw_solution = u_scale * scaled_solution
            solver_name = str(problem.solver_stats.solver_name)
            audit_tolerance = (
                _SCS_AUDIT_TOL
                if solver_name.upper() == "SCS"
                else _OSQP_AUDIT_TOL
            )
            input_tol = SOLVER_INPUT_TOL
            valid = (
                raw_solution.shape == (self.n_controls,)
                and np.all(np.isfinite(raw_solution))
                and np.all(raw_solution <= u_scale + input_tol)
                and np.all(raw_solution >= -u_scale - input_tol)
            )
            if not valid:
                return CandidateCBFResult(
                    policy_name=name,
                    feasible=False,
                    u=None,
                    objective=float("inf"),
                    solver_status="invalid_solution",
                    rollout_safe=rollout_safe,
                    terminal_safe=terminal_safe,
                    solve_time_sec=time.perf_counter() - started,
                    qp_solved=True,
                    error="QP returned a non-finite or out-of-bounds input",
                    raw_u=raw_solution.copy(),
                    solver_name=solver_name,
                )

            projection = project_quad3d_solver_control_with_diagnostics(
                raw_solution,
                -u_scale,
                u_scale,
                expected_dimension=self.n_controls,
                tolerance=input_tol,
            )
            if projection.control is None:  # Kept explicit for static type checkers.
                raise AssertionError("validated Backup-CBF solution could not be projected")
            audit = audit_cvxpy_inequalities(
                constraints,
                [(u_scaled, projection.control / u_scale)],
                absolute_tolerance=audit_tolerance,
                relative_tolerance=audit_tolerance,
            )
            if not audit.passed:
                return CandidateCBFResult(
                    policy_name=name,
                    feasible=False,
                    u=None,
                    objective=float("inf"),
                    solver_status=status,
                    rollout_safe=rollout_safe,
                    terminal_safe=terminal_safe,
                    solve_time_sec=time.perf_counter() - started,
                    qp_solved=True,
                    error="post-projection constraint audit failed",
                    raw_u=raw_solution.copy(),
                    projected_u=projection.control.copy(),
                    projection_occurred=projection.projection_applied,
                    projection_delta_inf=projection.projection_delta_inf,
                    post_projection_constraints_satisfied=False,
                    max_post_projection_constraint_violation=audit.max_violation,
                    max_post_projection_violation_ratio=audit.max_violation_ratio,
                    constraint_audit_atol=audit.absolute_tolerance,
                    constraint_audit_rtol=audit.relative_tolerance,
                    constraint_audit_count=audit.constraint_count,
                    solver_name=solver_name,
                )
            u_solution = projection.control
            scaled_solution = u_solution / u_scale
            realized_error = self.Q_u * (scaled_solution - u_ref_scaled)
            realized_objective = float(realized_error @ realized_error)
            return CandidateCBFResult(
                policy_name=name,
                feasible=True,
                u=u_solution,
                objective=realized_objective,
                solver_status=status,
                rollout_safe=rollout_safe,
                terminal_safe=terminal_safe,
                solve_time_sec=time.perf_counter() - started,
                qp_solved=True,
                error=None,
                raw_u=raw_solution.copy(),
                projected_u=u_solution.copy(),
                projection_occurred=projection.projection_applied,
                projection_delta_inf=projection.projection_delta_inf,
                post_projection_constraints_satisfied=True,
                max_post_projection_constraint_violation=audit.max_violation,
                max_post_projection_violation_ratio=audit.max_violation_ratio,
                constraint_audit_atol=audit.absolute_tolerance,
                constraint_audit_rtol=audit.relative_tolerance,
                constraint_audit_count=audit.constraint_count,
                solver_name=solver_name,
            )
        except Exception as exc:
            return CandidateCBFResult(
                policy_name=name,
                feasible=False,
                u=None,
                objective=float("inf"),
                solver_status="error",
                rollout_safe=None,
                terminal_safe=None,
                solve_time_sec=time.perf_counter() - started,
                qp_solved=qp_solved,
                error=f"{type(exc).__name__}: {exc}",
            )


class MultiBackupCBFMinInterventionQuad3D(PLCBF_Quad3D):
    algorithm_key = "multi_backup_cbf_mi"
    table_name = "MB-CBF-MI"

    def __init__(
        self,
        robot,
        robot_spec: dict,
        dt: float = 0.05,
        backup_horizon: float = 4.0,
        cbf_alpha: float = 2.0,
        terminal_alpha: float = 2.0,
        safety_margin: float = 0.0,
        num_angle_policies: int = 64,
        tie_tolerance: float = 1e-8,
        maneuver_prefix_sec: float = 2.0,
        terminal_linear_speed_tol: float = 0.5,
        terminal_attitude_tol: float = 0.4,
        terminal_angular_rate_tol: float = 0.5,
        terminal_altitude_error_tol: float = 0.25,
        ax=None,
    ):
        if not (0.0 <= float(maneuver_prefix_sec) < float(backup_horizon)):
            raise ValueError(
                "maneuver_prefix_sec must be nonnegative and shorter than "
                "backup_horizon"
            )
        terminal_tolerances = {
            "terminal_linear_speed_tol": terminal_linear_speed_tol,
            "terminal_attitude_tol": terminal_attitude_tol,
            "terminal_angular_rate_tol": terminal_angular_rate_tol,
            "terminal_altitude_error_tol": terminal_altitude_error_tol,
        }
        for name, value in terminal_tolerances.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        super().__init__(
            robot_spec=robot_spec,
            dt=dt,
            backup_horizon=backup_horizon,
            cbf_alpha=cbf_alpha,
            safety_margin=safety_margin,
            num_angle_policies=num_angle_policies,
            max_operator="input_space",
            ax=ax,
        )
        self.robot = robot
        self.tie_tolerance = float(tie_tolerance)
        self.terminal_alpha = float(terminal_alpha)
        self.maneuver_prefix_steps = int(round(float(maneuver_prefix_sec) / dt))
        self.maneuver_prefix_sec = self.maneuver_prefix_steps * float(dt)
        if self.maneuver_prefix_sec >= float(backup_horizon):
            raise ValueError(
                "Rounded maneuver prefix leaves no common terminal stop tail"
            )
        self.terminal_tail_sec = float(backup_horizon) - self.maneuver_prefix_sec
        self.terminal_linear_speed_tol = float(terminal_linear_speed_tol)
        self.terminal_attitude_tol = float(terminal_attitude_tol)
        self.terminal_angular_rate_tol = float(terminal_angular_rate_tol)
        self.terminal_altitude_error_tol = float(terminal_altitude_error_tol)
        self.certificate_lost = False
        self.qp_infeasible = False
        self.runtime_error = False
        self.infeasible = False
        self.fallback_applied = False

        self._candidate_adapters: "OrderedDict[str, _QuadPolicyAdapter]" = OrderedDict()
        self._candidate_filters: "OrderedDict[str, _StrictBackupCBFCandidate]" = OrderedDict()
        stop_type, stop_params = self.policy_configs["stop"]
        if stop_type != "stop":
            raise AssertionError("PL-CBF runtime stop entry is not a stop policy")
        for name, (policy_type, params) in self.policy_configs.items():
            adapter = _QuadPolicyAdapter(name, policy_type, params)
            adapter.configure_stop_tail(stop_params, self.maneuver_prefix_steps)
            candidate = _StrictBackupCBFCandidate(
                robot=robot,
                robot_spec=robot_spec,
                dt=dt,
                backup_horizon=backup_horizon,
                ax=None,
            )
            candidate.alpha = float(cbf_alpha)
            candidate.alpha_terminal = float(terminal_alpha)
            candidate.safety_margin = float(safety_margin)
            candidate.configure_terminal_proxy(
                linear_speed_tol=self.terminal_linear_speed_tol,
                attitude_tol=self.terminal_attitude_tol,
                angular_rate_tol=self.terminal_angular_rate_tol,
                altitude_error_tol=self.terminal_altitude_error_tol,
            )
            candidate.set_backup_controller(adapter)
            self._candidate_adapters[name] = adapter
            self._candidate_filters[name] = candidate

        self._assert_library_alignment()

    @property
    def policy_names(self) -> Tuple[str, ...]:
        return tuple(self.policy_configs.keys())

    def _assert_library_alignment(self) -> None:
        expected = tuple(self.policy_configs.keys())
        if tuple(self._candidate_adapters.keys()) != expected:
            raise AssertionError("MB-CBF-MI policy ordering differs from PL-CBF")
        if tuple(self._candidate_filters.keys()) != expected:
            raise AssertionError("MB-CBF-MI candidate ordering differs from PL-CBF")
        for name, (policy_type, params) in self.policy_configs.items():
            adapter = self._candidate_adapters[name]
            if adapter.policy_type != policy_type or adapter.params is not params:
                raise AssertionError(f"Policy parameters were reconstructed for {name}")
            if adapter._maneuver_prefix_steps != self.maneuver_prefix_steps:
                raise AssertionError(f"Stop-tail switch differs for {name}")

    def set_environment(self, env) -> None:
        super().set_environment(env)
        for candidate in self._candidate_filters.values():
            candidate.set_environment(env)

    def _runtime_policy_params(self, control_ref: Optional[dict]):
        runtime = OrderedDict(self.policy_configs)
        if control_ref is not None and "waypoints" in control_ref:
            policy_type, base = runtime["nominal"]
            runtime["nominal"] = (
                policy_type,
                WaypointPolicyParams(
                    waypoints=jnp.asarray(control_ref["waypoints"]),
                    v_max=float(self.robot_spec.get("v_max", 5.0)),
                    Kp=float(self.robot_spec.get("nominal_Kp_v", 6.0)),
                    K_lat=float(self.robot_spec.get("nominal_K_lat", 1.0)),
                    v_lat_max=float(
                        self.robot_spec.get(
                            "nominal_v_lat_max", self.robot_spec.get("v_ref", 4.0)
                        )
                    ),
                    dist_threshold=float(
                        self.robot_spec.get("nominal_dist_threshold", 0.8)
                    ),
                    current_wp_idx=int(control_ref.get("wp_idx", 0)),
                    ctrl=base.ctrl,
                ),
            )
        return runtime

    def _select_minimum_intervention(
        self, results: Iterable[CandidateCBFResult]
    ) -> Optional[CandidateCBFResult]:
        policy_index = {name: index for index, name in enumerate(self.policy_names)}
        feasible = [result for result in results if result.feasible]
        if not feasible:
            return None
        best = min(feasible, key=lambda result: policy_index[result.policy_name])
        for result in feasible:
            objective_better = result.objective < best.objective - self.tie_tolerance
            objective_tied = abs(result.objective - best.objective) <= self.tie_tolerance
            index_better = (
                policy_index[result.policy_name] < policy_index[best.policy_name]
            )
            if objective_better or (objective_tied and index_better):
                best = result
        return best

    def solve_control_problem(self, state, control_ref=None):
        self.runtime_error = False
        u_nom = np.zeros(4, dtype=float)
        if control_ref is not None and "u_ref" in control_ref:
            u_nom = np.asarray(control_ref["u_ref"], dtype=float).reshape(-1)
        if u_nom.shape != (4,) or not np.all(np.isfinite(u_nom)):
            raise ValueError("MB-CBF-MI received an invalid nominal input")

        runtime_configs = self._runtime_policy_params(control_ref)
        if tuple(runtime_configs.keys()) != self.policy_names:
            raise AssertionError("Runtime policy library changed unexpectedly")

        frozen_predictor = _FrozenGhostPredictor(self.dynamic_obstacles)
        candidate_results = []
        for name, (policy_type, params) in runtime_configs.items():
            adapter = self._candidate_adapters[name]
            adapter.set_params(policy_type, params)
            candidate = self._candidate_filters[name]
            candidate.set_nominal_trajectory(None, np.asarray([u_nom]))
            candidate.set_moving_obstacles(frozen_predictor)
            candidate_results.append(
                candidate.solve_candidate(state.copy(), u_nom.copy())
            )

        certified = [
            result
            for result in candidate_results
            if result.rollout_safe is True and result.terminal_safe is True
        ]
        feasible = [result for result in candidate_results if result.feasible]
        candidate_evaluation_error_count = sum(
            int(result.solver_status == "error") for result in candidate_results
        )
        self.runtime_error = bool(
            candidate_evaluation_error_count and not feasible
        )
        self.certificate_lost = bool(not certified and not self.runtime_error)
        self.qp_infeasible = bool(
            certified and not feasible and not self.runtime_error
        )
        self.infeasible = self.qp_infeasible
        self.fallback_applied = not feasible

        if not feasible:
            if self.runtime_error:
                raise RuntimeError(
                    "MB-CBF-MI candidate evaluation failed; certificate status "
                    "is unknown"
                )
            emergency = self._candidate_adapters["stop"].compute_control(state)
            limit = float(self.robot_spec.get("u_max", 10.0))
            return np.clip(np.asarray(emergency, dtype=float), -limit, limit)

        best = self._select_minimum_intervention(candidate_results)
        if best is None:
            raise AssertionError("feasible candidate set unexpectedly became empty")
        return np.asarray(best.u, dtype=float).reshape(-1)


__all__ = [
    "CandidateCBFResult",
    "MultiBackupCBFMinInterventionQuad3D",
]
