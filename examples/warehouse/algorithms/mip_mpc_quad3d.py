"""
Mixed-integer MPC baseline for Warehouse Quad3D.

This is a deliberately heavy baseline for runtime comparison with PLCBF:
- Build fallback rollouts for many angle policies (32/64 tunable)
- Compute per-policy safety value V_i = min_t h_i(t)
- Solve a mixed-integer MPC with OR-style policy-selection constraints
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


def _angle_normalize(x: np.ndarray | float) -> np.ndarray | float:
    return ((x + np.pi) % (2.0 * np.pi)) - np.pi


def _reflect_1d(value: float, low: float, high: float) -> float:
    """Reflect a scalar coordinate at box boundaries."""
    x = float(value)
    if high <= low:
        return low
    while x < low or x > high:
        if x < low:
            x = 2.0 * low - x
        elif x > high:
            x = 2.0 * high - x
    return x


class MIPMPC_Quad3D:
    """Mixed-integer MPC with OR constraints over fallback policy rollouts."""

    def __init__(
        self,
        robot_spec: dict,
        dt: float = 0.05,
        mpc_horizon_steps: int = 8,
        backup_horizon: float = 4.0,
        num_angle_policies: int = 32,
        safety_margin: float = 0.0,
        safety_threshold: float = 0.0,
        xy_tube: float = 3.0,
        control_tube: float = 6.0,
        control_tube_steps: int = 2,
        goal_weight: float = 8.0,
        terminal_goal_weight: float = 16.0,
        velocity_weight: float = 0.15,
        control_weight: float = 0.02,
        nominal_weight: float = 0.5,
        big_m_xy: float = 400.0,
        big_m_u: float = 60.0,
        big_m_h: float = 50.0,
        mip_solver: str = "ECOS_BB",
    ):
        self.robot_spec = dict(robot_spec)
        self.dt = float(dt)
        self.mpc_horizon_steps = int(mpc_horizon_steps)
        self.backup_horizon = float(backup_horizon)
        self.backup_horizon_steps = max(1, int(round(self.backup_horizon / self.dt)))
        self.num_angle_policies = int(num_angle_policies)
        self.safety_margin = float(safety_margin)
        self.safety_threshold = float(safety_threshold)
        self.xy_tube = float(xy_tube)
        self.control_tube = float(control_tube)
        self.control_tube_steps = max(1, int(control_tube_steps))
        self.goal_weight = float(goal_weight)
        self.terminal_goal_weight = float(terminal_goal_weight)
        self.velocity_weight = float(velocity_weight)
        self.control_weight = float(control_weight)
        self.nominal_weight = float(nominal_weight)
        self.big_m_xy = float(big_m_xy)
        self.big_m_u = float(big_m_u)
        self.big_m_h = float(big_m_h)
        self.mip_solver = mip_solver

        self.dynamic_obstacles: List[dict] = []
        self.static_obstacles: List[dict] = []
        self.env = None

        self.robot_radius = float(self.robot_spec.get("radius", 1.0)) + self.safety_margin
        self.world_width = 100.0
        self.world_height = 100.0
        self.boundary_low = 2.0
        self.boundary_high_x = 98.0
        self.boundary_high_y = 98.0

        self.u_min = float(self.robot_spec.get("u_min", -10.0))
        self.u_max = float(self.robot_spec.get("u_max", 10.0))
        self.v_ref = float(self.robot_spec.get("v_ref", 3.0))
        self.kp_v_angle = float(self.robot_spec.get("angle_Kp_v", 7.0))
        self.a_max_xy = float(self.robot_spec.get("a_max_xy", 8.0))
        self.z_ref = float(self.robot_spec.get("z_ref", 0.0))
        self.kp_z = float(self.robot_spec.get("Kp_z", 4.0))
        self.kd_z = float(self.robot_spec.get("Kd_z", 3.0))
        self.k_ang = float(self.robot_spec.get("K_ang", 10.0))
        self.kd_ang = float(self.robot_spec.get("Kd_ang", 4.0))
        self.mass = float(self.robot_spec.get("mass", 3.0))
        self.ix = float(self.robot_spec.get("Ix", 0.5))
        self.iy = float(self.robot_spec.get("Iy", 0.5))
        self.iz = float(self.robot_spec.get("Iz", 0.5))
        self.g = float(self.robot_spec.get("g", 9.8))
        self.l = float(self.robot_spec.get("L", 0.3))
        self.nu = float(self.robot_spec.get("nu", 0.1))

        self.A, self.B, self.B2_inv = self._build_quad3d_matrices()
        self.Ad = np.eye(12) + self.dt * self.A
        self.Bd = self.dt * self.B

        # Runtime stats
        self.last_policy_eval_time_sec = 0.0
        self.last_mip_solve_time_sec = 0.0
        self.last_total_time_sec = 0.0
        self.last_solver_status = "not_run"
        self.last_solver_name = ""
        self.last_selected_policy_idx = -1
        self.last_selected_policy_safety = -np.inf
        self.last_policy_safety_values = np.array([])

    def _build_quad3d_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = np.zeros((12, 12))
        A[0, 6] = 1.0
        A[1, 7] = 1.0
        A[2, 8] = 1.0
        A[3, 9] = 1.0
        A[4, 10] = 1.0
        A[5, 11] = 1.0
        A[6, 3] = self.g
        A[7, 4] = -self.g

        B2 = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, self.l, 0.0, -self.l],
                [self.l, 0.0, -self.l, 0.0],
                [self.nu, -self.nu, self.nu, -self.nu],
            ],
            dtype=float,
        )

        B1 = np.zeros((12, 4))
        B1[8, 0] = 1.0 / self.mass
        B1[9, 1] = 1.0 / self.iy
        B1[10, 2] = 1.0 / self.ix
        B1[11, 3] = 1.0 / self.iz
        B = B1 @ B2
        B2_inv = np.linalg.pinv(B2)
        return A, B, B2_inv

    def set_environment(self, env):
        self.env = env
        if env is not None:
            self.world_width = float(getattr(env, "width", self.world_width))
            self.world_height = float(getattr(env, "height", self.world_height))

    def update_obstacles(self, dynamic_obstacles: List[dict], static_obstacles: List[dict]):
        self.dynamic_obstacles = list(dynamic_obstacles) if dynamic_obstacles else []
        self.static_obstacles = list(static_obstacles) if static_obstacles else []

    def update_visualization(self):
        # Kept for API parity with other algorithms.
        return

    def _rk4_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        u = np.clip(u, self.u_min, self.u_max)
        k1 = self.A @ x + self.B @ u
        k2 = self.A @ (x + 0.5 * self.dt * k1) + self.B @ u
        k3 = self.A @ (x + 0.5 * self.dt * k2) + self.B @ u
        k4 = self.A @ (x + self.dt * k3) + self.B @ u
        x_next = x + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x_next[3] = _angle_normalize(x_next[3])
        x_next[4] = _angle_normalize(x_next[4])
        x_next[5] = _angle_normalize(x_next[5])
        return x_next

    def _clip_xy_accel(self, ax: float, ay: float) -> Tuple[float, float]:
        norm = np.hypot(ax, ay)
        if norm > self.a_max_xy:
            scale = self.a_max_xy / (norm + 1e-8)
            ax *= scale
            ay *= scale
        return ax, ay

    def _accel_to_u(self, state: np.ndarray, ax_des: float, ay_des: float, az_des: float) -> np.ndarray:
        theta = float(state[3])
        phi = float(state[4])
        psi = float(state[5])
        q = float(state[9])
        p = float(state[10])
        r = float(state[11])

        theta_des = ax_des / self.g
        phi_des = -ay_des / self.g
        F_des = self.mass * az_des

        tau_y = self.iy * (self.k_ang * (theta_des - theta) + self.kd_ang * (0.0 - q))
        tau_x = self.ix * (self.k_ang * (phi_des - phi) + self.kd_ang * (0.0 - p))
        tau_z = self.iz * (self.k_ang * (0.0 - psi) + self.kd_ang * (0.0 - r))

        wrench = np.array([F_des, tau_y, tau_x, tau_z], dtype=float)
        u = self.B2_inv @ wrench
        return np.clip(u, self.u_min, self.u_max)

    def _angle_policy_u(self, state: np.ndarray, angle: float) -> np.ndarray:
        vx, vy, vz = state[6], state[7], state[8]
        vx_des = self.v_ref * np.cos(angle)
        vy_des = self.v_ref * np.sin(angle)

        ax = self.kp_v_angle * (vx_des - vx)
        ay = self.kp_v_angle * (vy_des - vy)
        ax, ay = self._clip_xy_accel(ax, ay)
        az = self.kp_z * (self.z_ref - state[2]) - self.kd_z * vz
        return self._accel_to_u(state, ax, ay, az)

    def _dynamic_obs_pos(self, obs: dict, t: float) -> Tuple[float, float]:
        ox = float(obs["x"]) + float(obs.get("vx", 0.0)) * t
        oy = float(obs["y"]) + float(obs.get("vy", 0.0)) * t
        ox = _reflect_1d(ox, self.boundary_low, self.boundary_high_x)
        oy = _reflect_1d(oy, self.boundary_low, self.boundary_high_y)
        return ox, oy

    def _h_value(self, state: np.ndarray, t: float) -> float:
        x, y = float(state[0]), float(state[1])

        h_min = min(
            x - self.robot_radius,
            self.world_width - x - self.robot_radius,
            y - self.robot_radius,
            self.world_height - y - self.robot_radius,
        )

        for obs in self.static_obstacles:
            ox = float(obs["x"])
            oy = float(obs["y"])
            r = float(obs["radius"]) + self.robot_radius
            h_min = min(h_min, np.hypot(x - ox, y - oy) - r)

        for obs in self.dynamic_obstacles:
            ox, oy = self._dynamic_obs_pos(obs, t)
            r = float(obs["radius"]) + self.robot_radius
            h_min = min(h_min, np.hypot(x - ox, y - oy) - r)

        return float(h_min)

    def _rollout_angle_policy(
        self, x0: np.ndarray, angle: float, rollout_steps: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        states = np.zeros((rollout_steps + 1, 12), dtype=float)
        controls = np.zeros((rollout_steps, 4), dtype=float)
        states[0] = x0

        safety_val = self._h_value(states[0], 0.0)
        for k in range(rollout_steps):
            u = self._angle_policy_u(states[k], angle)
            controls[k] = u
            states[k + 1] = self._rk4_step(states[k], u)

            if k + 1 <= self.backup_horizon_steps:
                t_k = (k + 1) * self.dt
                safety_val = min(safety_val, self._h_value(states[k + 1], t_k))

        return states, controls, float(safety_val)

    def _compute_policy_data(self, x0: np.ndarray) -> Dict[str, np.ndarray]:
        rollout_steps = max(self.mpc_horizon_steps, self.backup_horizon_steps)
        angles = np.linspace(0.0, 2.0 * np.pi, self.num_angle_policies, endpoint=False)

        ref_xy = np.zeros((self.num_angle_policies, self.mpc_horizon_steps + 1, 2))
        ref_u = np.zeros((self.num_angle_policies, self.mpc_horizon_steps, 4))
        safety = np.zeros(self.num_angle_policies)

        for i, angle in enumerate(angles):
            states_i, controls_i, v_i = self._rollout_angle_policy(x0, angle, rollout_steps)
            ref_xy[i] = states_i[: self.mpc_horizon_steps + 1, :2]
            ref_u[i] = controls_i[: self.mpc_horizon_steps]
            safety[i] = v_i

        return {"angles": angles, "ref_xy": ref_xy, "ref_u": ref_u, "safety": safety}

    def _target_xy(self, state: np.ndarray, control_ref: Optional[dict]) -> np.ndarray:
        if control_ref and "waypoints" in control_ref and control_ref["waypoints"] is not None:
            wps = np.array(control_ref["waypoints"], dtype=float)
            if wps.ndim == 2 and wps.shape[0] > 0:
                idx = int(np.clip(control_ref.get("wp_idx", 0), 0, wps.shape[0] - 1))
                return wps[idx]
        if self.env is not None and hasattr(self.env, "goal_pos"):
            return np.array(self.env.goal_pos, dtype=float)
        return np.array(state[:2], dtype=float)

    def _goal_xy(self) -> Optional[np.ndarray]:
        if self.env is not None and hasattr(self.env, "goal_pos"):
            return np.array(self.env.goal_pos, dtype=float)
        return None

    def _fallback_control(self, policy_data: Dict[str, np.ndarray], u_ref: Optional[np.ndarray]) -> np.ndarray:
        safety = policy_data["safety"]
        best_idx = int(np.argmax(safety))
        u_fb = policy_data["ref_u"][best_idx, 0].copy()
        if u_ref is not None:
            u_fb = 0.75 * u_fb + 0.25 * u_ref
        self.last_selected_policy_idx = best_idx
        self.last_selected_policy_safety = float(safety[best_idx])
        return np.clip(u_fb, self.u_min, self.u_max)

    def _choose_solvers(self) -> List[str]:
        installed = set(cp.installed_solvers())
        candidates = [self.mip_solver, "GUROBI", "MOSEK", "SCIP", "ECOS_BB"]
        uniq = []
        for name in candidates:
            if name in installed and name not in uniq:
                uniq.append(name)
        return uniq

    def _solve_milp_scipy(
        self,
        x0: np.ndarray,
        u_ref: Optional[np.ndarray],
        target_xy: np.ndarray,
        goal_xy: Optional[np.ndarray],
        policy_data: Dict[str, np.ndarray],
    ) -> Tuple[Optional[np.ndarray], str, str, int]:
        """
        Pure MILP fallback using scipy.optimize.milp.
        The objective is linearized with L1 auxiliary variables.
        """
        n = 12
        m = 4
        N = self.mpc_horizon_steps
        P = self.num_angle_policies

        nx = n * (N + 1)
        nu = m * N
        nz = P
        n_goal_abs = 2 * N
        n_vel_abs = 2 * N
        n_u_abs = m * N
        n_nom_abs = m if u_ref is not None else 0
        n_term_abs = 2 if goal_xy is not None else 0

        offset = 0
        x_start = offset
        offset += nx
        u_start = offset
        offset += nu
        z_start = offset
        offset += nz
        goal_abs_start = offset
        offset += n_goal_abs
        vel_abs_start = offset
        offset += n_vel_abs
        u_abs_start = offset
        offset += n_u_abs
        nom_abs_start = offset
        offset += n_nom_abs
        term_abs_start = offset
        offset += n_term_abs
        n_var = offset

        def x_idx(k: int, d: int) -> int:
            return x_start + k * n + d

        def u_idx(k: int, j: int) -> int:
            return u_start + k * m + j

        def z_idx(i: int) -> int:
            return z_start + i

        def goal_abs_idx(k: int, axis: int) -> int:
            return goal_abs_start + (k - 1) * 2 + axis

        def vel_abs_idx(k: int, axis: int) -> int:
            return vel_abs_start + (k - 1) * 2 + axis

        def u_abs_idx(k: int, j: int) -> int:
            return u_abs_start + k * m + j

        def nom_abs_idx(j: int) -> int:
            return nom_abs_start + j

        def term_abs_idx(axis: int) -> int:
            return term_abs_start + axis

        lb_var = np.full(n_var, -np.inf)
        ub_var = np.full(n_var, np.inf)
        integrality = np.zeros(n_var, dtype=int)

        # Input bounds
        for k in range(N):
            for j in range(m):
                idx = u_idx(k, j)
                lb_var[idx] = self.u_min
                ub_var[idx] = self.u_max

        # Binary policy selectors
        for i in range(P):
            idx = z_idx(i)
            lb_var[idx] = 0.0
            ub_var[idx] = 1.0
            integrality[idx] = 1

        # Nonnegative absolute-value helper variables
        lb_var[goal_abs_start : goal_abs_start + n_goal_abs] = 0.0
        lb_var[vel_abs_start : vel_abs_start + n_vel_abs] = 0.0
        lb_var[u_abs_start : u_abs_start + n_u_abs] = 0.0
        if n_nom_abs > 0:
            lb_var[nom_abs_start : nom_abs_start + n_nom_abs] = 0.0
        if n_term_abs > 0:
            lb_var[term_abs_start : term_abs_start + n_term_abs] = 0.0

        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        lb_con: List[float] = []
        ub_con: List[float] = []

        def add_row(coeff: Dict[int, float], lb_value: float = -np.inf, ub_value: float = np.inf):
            row_id = len(lb_con)
            for c_idx, v in coeff.items():
                if abs(v) > 1e-12:
                    rows.append(row_id)
                    cols.append(c_idx)
                    vals.append(float(v))
            lb_con.append(float(lb_value))
            ub_con.append(float(ub_value))

        # x0 equality
        for d in range(n):
            add_row({x_idx(0, d): 1.0}, lb_value=float(x0[d]), ub_value=float(x0[d]))

        # Dynamics equalities
        for k in range(N):
            for d in range(n):
                coeff = {x_idx(k + 1, d): 1.0}
                for j in range(n):
                    a = -self.Ad[d, j]
                    if abs(a) > 1e-12:
                        coeff[x_idx(k, j)] = coeff.get(x_idx(k, j), 0.0) + a
                for j in range(m):
                    b = -self.Bd[d, j]
                    if abs(b) > 1e-12:
                        coeff[u_idx(k, j)] = coeff.get(u_idx(k, j), 0.0) + b
                add_row(coeff, lb_value=0.0, ub_value=0.0)

        # Sum z = 1
        add_row({z_idx(i): 1.0 for i in range(P)}, lb_value=1.0, ub_value=1.0)

        # Position bounds
        x_low = self.robot_radius
        x_high = self.world_width - self.robot_radius
        y_low = self.robot_radius
        y_high = self.world_height - self.robot_radius
        for k in range(1, N + 1):
            add_row({x_idx(k, 0): 1.0}, lb_value=x_low, ub_value=x_high)
            add_row({x_idx(k, 1): 1.0}, lb_value=y_low, ub_value=y_high)

        # OR constraints tied to precomputed fallback rollouts
        safety_vals = policy_data["safety"]
        ref_xy = policy_data["ref_xy"]
        ref_u = policy_data["ref_u"]
        admissible = safety_vals >= self.safety_threshold
        effective_threshold = self.safety_threshold
        if not np.any(admissible):
            best_idx = int(np.argmax(safety_vals))
            admissible[:] = False
            admissible[best_idx] = True
            effective_threshold = float(safety_vals[best_idx]) - 1e-6

        for i in range(P):
            zi = z_idx(i)
            if not admissible[i]:
                ub_var[zi] = 0.0

            # safety_i + M*(1-z_i) >= threshold
            # equivalent linear inequality: M*z_i <= safety_i + M - threshold
            add_row(
                {zi: self.big_m_h},
                ub_value=float(safety_vals[i] + self.big_m_h - effective_threshold),
            )

            for k in range(1, N + 1):
                rx = float(ref_xy[i, k, 0])
                ry = float(ref_xy[i, k, 1])

                # x_k - rx <= tube + M*(1-z_i)
                add_row(
                    {x_idx(k, 0): 1.0, zi: self.big_m_xy},
                    ub_value=self.xy_tube + self.big_m_xy + rx,
                )
                # rx - x_k <= tube + M*(1-z_i)
                add_row(
                    {x_idx(k, 0): -1.0, zi: self.big_m_xy},
                    ub_value=self.xy_tube + self.big_m_xy - rx,
                )
                # y_k - ry <= tube + M*(1-z_i)
                add_row(
                    {x_idx(k, 1): 1.0, zi: self.big_m_xy},
                    ub_value=self.xy_tube + self.big_m_xy + ry,
                )
                # ry - y_k <= tube + M*(1-z_i)
                add_row(
                    {x_idx(k, 1): -1.0, zi: self.big_m_xy},
                    ub_value=self.xy_tube + self.big_m_xy - ry,
                )

            for k in range(min(self.control_tube_steps, N)):
                for j in range(m):
                    ru = float(ref_u[i, k, j])
                    add_row(
                        {u_idx(k, j): 1.0, zi: self.big_m_u},
                        ub_value=self.control_tube + self.big_m_u + ru,
                    )
                    add_row(
                        {u_idx(k, j): -1.0, zi: self.big_m_u},
                        ub_value=self.control_tube + self.big_m_u - ru,
                    )

        # Absolute-value linearization for objective terms
        tx, ty = float(target_xy[0]), float(target_xy[1])
        for k in range(1, N + 1):
            agx = goal_abs_idx(k, 0)
            agy = goal_abs_idx(k, 1)
            avx = vel_abs_idx(k, 0)
            avy = vel_abs_idx(k, 1)

            # |x-target_x|
            add_row({x_idx(k, 0): 1.0, agx: -1.0}, ub_value=tx)
            add_row({x_idx(k, 0): -1.0, agx: -1.0}, ub_value=-tx)
            # |y-target_y|
            add_row({x_idx(k, 1): 1.0, agy: -1.0}, ub_value=ty)
            add_row({x_idx(k, 1): -1.0, agy: -1.0}, ub_value=-ty)

            # |vx|, |vy|
            add_row({x_idx(k, 6): 1.0, avx: -1.0}, ub_value=0.0)
            add_row({x_idx(k, 6): -1.0, avx: -1.0}, ub_value=0.0)
            add_row({x_idx(k, 7): 1.0, avy: -1.0}, ub_value=0.0)
            add_row({x_idx(k, 7): -1.0, avy: -1.0}, ub_value=0.0)

        # |u|
        for k in range(N):
            for j in range(m):
                au = u_abs_idx(k, j)
                add_row({u_idx(k, j): 1.0, au: -1.0}, ub_value=0.0)
                add_row({u_idx(k, j): -1.0, au: -1.0}, ub_value=0.0)

        # |u0 - u_ref|
        if u_ref is not None:
            for j in range(m):
                an = nom_abs_idx(j)
                ur = float(u_ref[j])
                add_row({u_idx(0, j): 1.0, an: -1.0}, ub_value=ur)
                add_row({u_idx(0, j): -1.0, an: -1.0}, ub_value=-ur)

        # Terminal |x_N - goal|
        if goal_xy is not None:
            gx, gy = float(goal_xy[0]), float(goal_xy[1])
            atx = term_abs_idx(0)
            aty = term_abs_idx(1)
            add_row({x_idx(N, 0): 1.0, atx: -1.0}, ub_value=gx)
            add_row({x_idx(N, 0): -1.0, atx: -1.0}, ub_value=-gx)
            add_row({x_idx(N, 1): 1.0, aty: -1.0}, ub_value=gy)
            add_row({x_idx(N, 1): -1.0, aty: -1.0}, ub_value=-gy)

        # Linear objective
        c = np.zeros(n_var)
        for k in range(1, N + 1):
            c[goal_abs_idx(k, 0)] = self.goal_weight
            c[goal_abs_idx(k, 1)] = self.goal_weight
            c[vel_abs_idx(k, 0)] = self.velocity_weight
            c[vel_abs_idx(k, 1)] = self.velocity_weight
        for k in range(N):
            for j in range(m):
                c[u_abs_idx(k, j)] = self.control_weight
        if u_ref is not None:
            for j in range(m):
                c[nom_abs_idx(j)] = self.nominal_weight
        if goal_xy is not None:
            c[term_abs_idx(0)] = self.terminal_goal_weight
            c[term_abs_idx(1)] = self.terminal_goal_weight

        # Encourage choosing safer branch when ties occur.
        for i in range(P):
            c[z_idx(i)] = -0.01 * float(safety_vals[i])

        A = sparse.coo_matrix((vals, (rows, cols)), shape=(len(lb_con), n_var))
        linear_constraint = LinearConstraint(A, np.array(lb_con), np.array(ub_con))
        bounds = Bounds(lb_var, ub_var)

        try:
            result = milp(
                c=c,
                integrality=integrality,
                bounds=bounds,
                constraints=linear_constraint,
                options={"disp": False, "time_limit": 1.0, "mip_rel_gap": 0.05},
            )
        except Exception as exc:
            return None, f"scipy_exception:{exc.__class__.__name__}", "SCIPY", -1

        status = f"scipy_status_{result.status}"
        if result.x is not None:
            u0 = np.array([result.x[u_idx(0, j)] for j in range(m)], dtype=float)
            if np.all(np.isfinite(u0)) and result.status in (0, 1):
                z_vals = np.array([result.x[z_idx(i)] for i in range(P)], dtype=float)
                chosen_policy_idx = int(np.argmax(z_vals))
                if result.status == 0:
                    status = "optimal"
                else:
                    status = "time_limited_feasible"
                return u0, status, "SCIPY", chosen_policy_idx

        return None, status, "SCIPY", -1

    def _solve_mip(
        self,
        x0: np.ndarray,
        u_ref: Optional[np.ndarray],
        target_xy: np.ndarray,
        goal_xy: Optional[np.ndarray],
        policy_data: Dict[str, np.ndarray],
    ) -> Tuple[Optional[np.ndarray], str, str, int]:
        solver_candidates = self._choose_solvers()
        if not solver_candidates:
            return self._solve_milp_scipy(x0, u_ref, target_xy, goal_xy, policy_data)

        n = 12
        m = 4
        N = self.mpc_horizon_steps
        P = self.num_angle_policies

        x = cp.Variable((n, N + 1))
        u = cp.Variable((m, N))
        z = cp.Variable(P, boolean=True)

        constraints = [x[:, 0] == x0, cp.sum(z) == 1]

        x_low = self.robot_radius
        x_high = self.world_width - self.robot_radius
        y_low = self.robot_radius
        y_high = self.world_height - self.robot_radius

        for k in range(N):
            constraints += [
                x[:, k + 1] == self.Ad @ x[:, k] + self.Bd @ u[:, k],
                u[:, k] >= self.u_min,
                u[:, k] <= self.u_max,
            ]

        for k in range(1, N + 1):
            constraints += [
                x[0, k] >= x_low,
                x[0, k] <= x_high,
                x[1, k] >= y_low,
                x[1, k] <= y_high,
            ]

        safety_vals = policy_data["safety"]
        ref_xy = policy_data["ref_xy"]
        ref_u = policy_data["ref_u"]
        admissible = safety_vals >= self.safety_threshold
        effective_threshold = self.safety_threshold
        if not np.any(admissible):
            best_idx = int(np.argmax(safety_vals))
            admissible[:] = False
            admissible[best_idx] = True
            effective_threshold = float(safety_vals[best_idx]) - 1e-6

        for i in range(P):
            if not admissible[i]:
                constraints += [z[i] == 0]
            constraints += [safety_vals[i] + self.big_m_h * (1.0 - z[i]) >= effective_threshold]

            for k in range(1, N + 1):
                rx = float(ref_xy[i, k, 0])
                ry = float(ref_xy[i, k, 1])
                constraints += [
                    x[0, k] - rx <= self.xy_tube + self.big_m_xy * (1.0 - z[i]),
                    rx - x[0, k] <= self.xy_tube + self.big_m_xy * (1.0 - z[i]),
                    x[1, k] - ry <= self.xy_tube + self.big_m_xy * (1.0 - z[i]),
                    ry - x[1, k] <= self.xy_tube + self.big_m_xy * (1.0 - z[i]),
                ]

            for k in range(min(self.control_tube_steps, N)):
                for j in range(m):
                    ru = float(ref_u[i, k, j])
                    constraints += [
                        u[j, k] - ru <= self.control_tube + self.big_m_u * (1.0 - z[i]),
                        ru - u[j, k] <= self.control_tube + self.big_m_u * (1.0 - z[i]),
                    ]

        cost = 0.0
        for k in range(1, N + 1):
            cost += self.goal_weight * cp.sum_squares(x[0:2, k] - target_xy)
            cost += self.velocity_weight * cp.sum_squares(x[6:8, k])
        for k in range(N):
            cost += self.control_weight * cp.sum_squares(u[:, k])
        if u_ref is not None:
            cost += self.nominal_weight * cp.sum_squares(u[:, 0] - u_ref)
        if goal_xy is not None:
            cost += self.terminal_goal_weight * cp.sum_squares(x[0:2, N] - goal_xy)

        problem = cp.Problem(cp.Minimize(cost), constraints)

        chosen_u = None
        chosen_status = "solver_not_run"
        chosen_solver = ""
        chosen_policy_idx = -1

        for solver_name in solver_candidates:
            try:
                kwargs = {"solver": solver_name, "verbose": False}
                if solver_name == "ECOS_BB":
                    kwargs.update(
                        {
                            "max_iters": 300,
                            "mi_max_iters": 300,
                            "abstol": 1e-3,
                            "reltol": 1e-3,
                            "feastol": 1e-3,
                        }
                    )
                elif solver_name == "GUROBI":
                    kwargs.update({"MIPGap": 0.05, "TimeLimit": 1.0})
                elif solver_name == "MOSEK":
                    kwargs.update({"mosek_params": {"MSK_DPAR_MIO_TOL_REL_GAP": 0.05}})

                problem.solve(**kwargs)
            except Exception:
                continue

            chosen_status = str(problem.status)
            chosen_solver = solver_name
            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                if u.value is not None and np.all(np.isfinite(u.value[:, 0])):
                    chosen_u = np.array(u.value[:, 0], dtype=float).flatten()
                    if z.value is not None:
                        chosen_policy_idx = int(np.argmax(np.array(z.value).flatten()))
                    break

        if chosen_u is not None:
            return chosen_u, chosen_status, chosen_solver, chosen_policy_idx

        # CVXPY path did not produce a feasible MIP solution; try SciPy MILP fallback.
        u_scipy, status_scipy, solver_scipy, policy_idx_scipy = self._solve_milp_scipy(
            x0, u_ref, target_xy, goal_xy, policy_data
        )
        if u_scipy is not None:
            return u_scipy, status_scipy, solver_scipy, policy_idx_scipy

        return None, f"{chosen_status}|{status_scipy}", chosen_solver or solver_scipy, -1

    def solve_control_problem(self, state: np.ndarray, control_ref: Optional[dict] = None) -> np.ndarray:
        t0 = time.perf_counter()

        x0 = np.array(state, dtype=float).flatten()
        u_ref = None
        if control_ref and control_ref.get("u_ref", None) is not None:
            u_ref = np.array(control_ref["u_ref"], dtype=float).flatten()

        t_policy0 = time.perf_counter()
        policy_data = self._compute_policy_data(x0)
        t_policy1 = time.perf_counter()

        target_xy = self._target_xy(x0, control_ref)
        goal_xy = self._goal_xy()

        t_mip0 = time.perf_counter()
        u_opt, status, solver_name, policy_idx = self._solve_mip(x0, u_ref, target_xy, goal_xy, policy_data)
        t_mip1 = time.perf_counter()

        self.last_policy_eval_time_sec = t_policy1 - t_policy0
        self.last_mip_solve_time_sec = t_mip1 - t_mip0
        self.last_solver_status = status
        self.last_solver_name = solver_name
        self.last_policy_safety_values = policy_data["safety"].copy()

        if u_opt is None:
            u_opt = self._fallback_control(policy_data, u_ref)
            self.last_solver_status = f"fallback_after_{self.last_solver_status}"
        else:
            if policy_idx >= 0:
                self.last_selected_policy_idx = policy_idx
                self.last_selected_policy_safety = float(policy_data["safety"][policy_idx])
            else:
                self.last_selected_policy_idx = int(np.argmax(policy_data["safety"]))
                self.last_selected_policy_safety = float(policy_data["safety"][self.last_selected_policy_idx])

        self.last_total_time_sec = time.perf_counter() - t0
        return np.clip(u_opt, self.u_min, self.u_max)
