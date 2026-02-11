"""
Created on February 11th, 2026
@author: Taekyung Kim

@description:
MPCBF implementation for Quad3D (MPCBF_Quad3D).
"""

from typing import Dict
import numpy as np
import jax
import jax.numpy as jnp
import cvxpy as cp

from .pcbf_quad3d import _compute_value_pure_quad3d
from examples.inventory.dynamics.dynamics_quad3d_jax import Quad3DDynamicsJAX, Quad3DDynamicsParams, _build_quad3d_matrices
from examples.inventory.controllers.policies_quad3d_jax import (
    AnglePolicyJAX, StopPolicyJAX, WaypointPolicyJAX,
    AnglePolicyParams, StopPolicyParams, WaypointPolicyParams, Quad3DControlParams
)
from .pcbf_quad3d import PCBF_Quad3D


def _halfspace_box_volume(a, b, u_min, u_max, eps=1e-9):
    """
    Volume of intersection between axis-aligned box [u_min, u_max] and half-space a·u >= b.
    Uses inclusion-exclusion formula for weighted sum of uniforms (exact for hyper-rectangle).
    """
    a = np.array(a, dtype=float).flatten()
    u_min = np.array(u_min, dtype=float).flatten()
    u_max = np.array(u_max, dtype=float).flatten()

    # Full volume
    widths = u_max - u_min
    full_vol = float(np.prod(widths))

    if np.linalg.norm(a) < eps:
        return full_vol if b <= 0.0 else 0.0

    # Flip variables to make a nonnegative
    for i in range(a.size):
        if a[i] < 0:
            a[i] = -a[i]
            u_min[i], u_max[i] = -u_max[i], -u_min[i]
            widths[i] = u_max[i] - u_min[i]

    active = (a > eps) & (widths > eps)
    if not np.any(active):
        return full_vol if b <= 0.0 else 0.0

    a_act = a[active]
    u_min_act = u_min[active]
    u_max_act = u_max[active]
    widths_act = u_max_act - u_min_act

    inactive_vol = float(np.prod(widths[~active])) if np.any(~active) else 1.0

    w = a_act * widths_act
    t = b - np.dot(a_act, u_min_act)

    total_w = float(np.sum(w))
    if t <= 0.0:
        return full_vol
    if t >= total_w:
        return 0.0

    m = int(a_act.size)
    import math
    denom = float(math.factorial(m) * np.prod(w))
    cdf = 0.0
    for mask in range(1 << m):
        s = t
        bits = 0
        for i in range(m):
            if mask & (1 << i):
                s -= w[i]
                bits += 1
        if s > 0.0:
            cdf += ((-1.0) ** bits) * (s ** m)
    cdf = cdf / denom
    prob = 1.0 - cdf
    prob = float(np.clip(prob, 0.0, 1.0))
    return prob * float(np.prod(widths_act)) * inactive_vol


MAX_OPERATOR_TYPES = ['v', 'input_space']


class MPCBF_Quad3D(PCBF_Quad3D):
    """
    MPCBF for Quad3D.
    Evaluates multiple policies and selects the best one.
    """

    def __init__(
        self,
        robot_spec: dict,
        dt: float = 0.05,
        backup_horizon: float = 2.0,
        cbf_alpha: float = 5.0,
        safety_margin: float = 0.0,
        num_angle_policies: int = 10,
        max_operator: str = 'input_space',
        ax=None
    ):
        self.num_angle_policies = num_angle_policies
        self.max_operator = max_operator
        if self.max_operator not in MAX_OPERATOR_TYPES:
            raise ValueError(f"max_operator must be one of {MAX_OPERATOR_TYPES}")

        super().__init__(robot_spec, dt, backup_horizon, cbf_alpha, safety_margin, ax=None)
        self.ax = ax
        self.eval_horizon_steps = int(self.backup_horizon / self.dt)

        self.policy_configs = {}
        self._setup_policies()

        self.nominal_trajectory = None
        self.policy_lines = {}
        self._last_results = None
        self._last_best_name = None
        self._jit_val_grad_fn = None
        self._jit_val_grad_obs = None
        self.curr_step = 0
        self.debug = False
        if self.ax is not None:
            self._setup_visualization()

    def _setup_dynamics(self):
        self.dynamics_jax = Quad3DDynamicsJAX(self.robot_spec, self.dt)

        m = float(self.robot_spec.get('mass', 3.0))
        Ix = float(self.robot_spec.get('Ix', 0.5))
        Iy = float(self.robot_spec.get('Iy', 0.5))
        Iz = float(self.robot_spec.get('Iz', 0.5))
        L = float(self.robot_spec.get('L', 0.3))
        nu = float(self.robot_spec.get('nu', 0.1))
        g = float(self.robot_spec.get('g', 9.8))
        u_min = float(self.robot_spec.get('u_min', -10.0))
        u_max = float(self.robot_spec.get('u_max', 10.0))

        A, B, B2, B2_inv = _build_quad3d_matrices(m, Ix, Iy, Iz, L, nu, g)
        self.dynamics_params = Quad3DDynamicsParams(
            m=m, Ix=Ix, Iy=Iy, Iz=Iz, L=L, nu=nu, g=g,
            u_min=u_min, u_max=u_max, A=A, B=B, B2=B2, B2_inv=B2_inv
        )

    def _setup_policies(self):
        ctrl = Quad3DControlParams(
            m=float(self.robot_spec.get('mass', 3.0)),
            Ix=float(self.robot_spec.get('Ix', 0.5)),
            Iy=float(self.robot_spec.get('Iy', 0.5)),
            Iz=float(self.robot_spec.get('Iz', 0.5)),
            g=float(self.robot_spec.get('g', 9.8)),
            B2_inv=self.dynamics_params.B2_inv,
            u_min=float(self.robot_spec.get('u_min', -10.0)),
            u_max=float(self.robot_spec.get('u_max', 10.0)),
            K_ang=float(self.robot_spec.get('K_ang', 5.0)),
            Kd_ang=float(self.robot_spec.get('Kd_ang', 2.0)),
            z_ref=float(self.robot_spec.get('z_ref', 0.0)),
            Kp_z=float(self.robot_spec.get('Kp_z', 4.0)),
            Kd_z=float(self.robot_spec.get('Kd_z', 3.0)),
            a_max_xy=float(self.robot_spec.get('a_max_xy', 3.0))
        )

        Kp_v_nom = float(self.robot_spec.get('nominal_Kp_v', 6.0))
        K_lat_nom = float(self.robot_spec.get('nominal_K_lat', 1.0))
        v_lat_max_nom = float(self.robot_spec.get('nominal_v_lat_max', self.robot_spec.get('v_ref', 4.0)))
        dist_threshold_nom = float(self.robot_spec.get('nominal_dist_threshold', 0.8))
        Kp_v_angle = float(self.robot_spec.get('angle_Kp_v', Kp_v_nom))
        Kp_v_stop = float(self.robot_spec.get('stop_Kp_v', 3.0))

        for i in range(self.num_angle_policies):
            angle = i * (2 * np.pi / self.num_angle_policies)
            name = f'angle_{i}'
            v_ref = float(self.robot_spec.get('v_ref', 4.0))
            self.policy_configs[name] = ('angle', AnglePolicyParams(
                target_angle=angle, target_speed=v_ref, Kp_v=Kp_v_angle, ctrl=ctrl
            ))

        self.policy_configs['stop'] = ('stop', StopPolicyParams(
            Kp_v=Kp_v_stop, ctrl=ctrl
        ))

        self.policy_configs['nominal'] = ('waypoint', WaypointPolicyParams(
            waypoints=jnp.zeros((1, 2)),
            v_max=v_ref,
            Kp=Kp_v_nom,
            K_lat=K_lat_nom,
            v_lat_max=v_lat_max_nom,
            dist_threshold=dist_threshold_nom,
            current_wp_idx=0,
            ctrl=ctrl
        ))

    def set_nominal_traj(self, traj):
        self.nominal_trajectory = traj

    def _get_control_dim(self) -> int:
        return 4

    def _add_input_constraints(self, u, constraints):
        u_min = float(self.robot_spec.get('u_min', -10.0))
        u_max = float(self.robot_spec.get('u_max', 10.0))
        for i in range(4):
            constraints.append(u[i] <= u_max)
            constraints.append(u[i] >= u_min)

    def _get_system_matrices(self, state):
        f = np.array(self.dynamics_params.A @ np.array(state))
        g = np.array(self.dynamics_params.B)
        return f, g

    def _get_jit_val_grad(self):
        if self._jit_val_grad_fn is None:
            def val_fn(x0, dyn_p, pol_p, dyn_obs, stat_obs, p_type, hor, r_rad, rr_base, dt_val):
                V, _ = _compute_value_pure_quad3d(x0, dyn_p, pol_p, dyn_obs, stat_obs, p_type, hor, r_rad, rr_base, dt_val)
                return V

            self._jit_val_grad_fn = jax.jit(
                jax.value_and_grad(val_fn),
                static_argnums=(5, 6)
            )

            self._jit_traj = jax.jit(
                lambda x0, dp, pp, do, so, pt, h, rr, rb, dt: _compute_value_pure_quad3d(x0, dp, pp, do, so, pt, h, rr, rb, dt)[1],
                static_argnums=(5, 6)
            )
        return self._jit_val_grad_fn, self._jit_traj

    def _get_jit_val_grad_obs(self):
        if self._jit_val_grad_obs is None:
            def val_fn(x0, dyn_p, pol_p, dyn_obs, stat_obs, p_type, hor, r_rad, rr_base, dt_val):
                V, _ = _compute_value_pure_quad3d(x0, dyn_p, pol_p, dyn_obs, stat_obs, p_type, hor, r_rad, rr_base, dt_val)
                return V

            self._jit_val_grad_obs = jax.jit(
                jax.grad(val_fn, argnums=3),
                static_argnums=(5, 6)
            )
        return self._jit_val_grad_obs

    def _setup_visualization(self):
        if self.ax is None:
            return
        if self.policy_lines:
            return
        import matplotlib.cm as cm
        cmap = cm.get_cmap('hsv', self.num_angle_policies + 1)
        for i in range(self.num_angle_policies):
            name = f'angle_{i}'
            self.policy_lines[name], = self.ax.plot([], [], color=cmap(i), alpha=0.3, linewidth=1)
        self.policy_lines['nominal'], = self.ax.plot([], [], color='k', linestyle='--', alpha=0.5, linewidth=1)

    def _setup_multi_visualization(self):
        self._setup_visualization()

    def update_visualization(self):
        if self.ax is None or self._last_results is None:
            return
        for name, (V, g, traj) in self._last_results.items():
            if name in self.policy_lines:
                self.policy_lines[name].set_data(traj[:, 0], traj[:, 1])
                if name == self._last_best_name:
                    self.policy_lines[name].set_linewidth(3)
                    self.policy_lines[name].set_alpha(1.0)
                else:
                    self.policy_lines[name].set_linewidth(1)
                    self.policy_lines[name].set_alpha(0.3)

    def solve_control_problem(self, state, control_ref=None):
        if control_ref and 'u_ref' in control_ref:
            u_nom = np.array(control_ref['u_ref']).flatten()
        else:
            u_nom = np.zeros(4)

        results = {}

        if self.dynamic_obstacles:
            obs_array = jnp.array([
                (o['x'], o['y'], o['radius'], o.get('vx', 0.0), o.get('vy', 0.0))
                for o in self.dynamic_obstacles
            ])
        else:
            obs_array = jnp.zeros((0, 5))

        if self.static_obstacles:
            stat_obs_array = jnp.array([(o['x'], o['y'], o['radius']) for o in self.static_obstacles])
        else:
            stat_obs_array = jnp.zeros((0, 3))

        val_grad_fn, traj_fn = self._get_jit_val_grad()
        grad_obs_fn = self._get_jit_val_grad_obs()

        state_jax = jnp.array(state)
        policy_params_used = {}
        time_derivatives = {}

        for name, (ptype, params) in self.policy_configs.items():
            if name == 'nominal' and control_ref is not None and 'waypoints' in control_ref:
                Kp_v_nom = float(self.robot_spec.get('nominal_Kp_v', 6.0))
                K_lat_nom = float(self.robot_spec.get('nominal_K_lat', 1.0))
                v_lat_max_nom = float(self.robot_spec.get('nominal_v_lat_max', self.robot_spec.get('v_ref', 4.0)))
                dist_threshold_nom = float(self.robot_spec.get('nominal_dist_threshold', 0.8))
                params = WaypointPolicyParams(
                    waypoints=jnp.array(control_ref['waypoints']),
                    v_max=float(self.robot_spec.get('v_max', 5.0)),
                    Kp=Kp_v_nom,
                    K_lat=K_lat_nom,
                    v_lat_max=v_lat_max_nom,
                    dist_threshold=dist_threshold_nom,
                    current_wp_idx=control_ref.get('wp_idx', 0),
                    ctrl=params.ctrl
                )
            policy_params_used[name] = (ptype, params)

            V_jax, grad_jax = val_grad_fn(
                state_jax, self.dynamics_params, params, obs_array, stat_obs_array, ptype,
                self.eval_horizon_steps, self.robot_spec.get('radius', 1.0) + self.safety_margin,
                self.robot_spec.get('radius', 1.0), self.dt
            )
            traj = traj_fn(
                state_jax, self.dynamics_params, params, obs_array, stat_obs_array, ptype,
                self.eval_horizon_steps, self.robot_spec.get('radius', 1.0) + self.safety_margin,
                self.robot_spec.get('radius', 1.0), self.dt
            )
            results[name] = (float(V_jax), np.array(grad_jax), np.array(traj))

            if obs_array.shape[0] > 0:
                grad_obs = grad_obs_fn(
                    state_jax, self.dynamics_params, params, obs_array, stat_obs_array, ptype,
                    self.eval_horizon_steps, self.robot_spec.get('radius', 1.0) + self.safety_margin,
                    self.robot_spec.get('radius', 1.0), self.dt
                )
                obs_vel = obs_array[:, 3:5]
                time_derivatives[name] = float(jnp.sum(grad_obs[:, 0:2] * obs_vel))
            else:
                time_derivatives[name] = 0.0

        best_name = None
        best_score = -np.inf
        best_V = -np.inf

        u_min = np.array([self.dynamics_params.u_min] * 4)
        u_max = np.array([self.dynamics_params.u_max] * 4)

        f = np.array(self.dynamics_params.A @ np.array(state))
        g = np.array(self.dynamics_params.B)

        for name, (V, grad_V, traj) in results.items():
            Lf_V = np.dot(grad_V, f)
            Lg_V = grad_V @ g

            if self.max_operator == 'v':
                score = V
            else:
                if np.linalg.norm(Lg_V) < 1e-6:
                    score = V
                else:
                    cbf_rhs = Lf_V + time_derivatives.get(name, 0.0) + self.cbf_alpha * V
                    score = _halfspace_box_volume(Lg_V, -cbf_rhs, u_min, u_max)

            if V > 0 and best_V > 0:
                if score > best_score + 1e-6:
                    best_name, best_score, best_V = name, score, V
            elif V > 0 and best_V <= 0:
                best_name, best_score, best_V = name, score, V
            elif V <= 0 and best_V <= 0:
                # When all policies are unsafe, prefer the least-negative V
                if V > best_V + 1e-6:
                    best_name, best_score, best_V = name, score, V
                elif abs(V - best_V) <= 1e-6 and score > best_score + 1e-6:
                    best_name, best_score, best_V = name, score, V

        if best_name is None:
            best_name = 'nominal'

        V_best, grad_best, _ = results[best_name]
        self._last_results = results
        self._last_best_name = best_name

        if self.ax:
            for name, (V, g, traj) in results.items():
                if name in self.policy_lines:
                    self.policy_lines[name].set_data(traj[:, 0], traj[:, 1])
                    if name == best_name:
                        self.policy_lines[name].set_linewidth(3)
                        self.policy_lines[name].set_alpha(1.0)
                    else:
                        self.policy_lines[name].set_linewidth(1)
                        self.policy_lines[name].set_alpha(0.3)
        self._last_time_derivative = time_derivatives.get(best_name, 0.0)
        if self.debug and self.curr_step % 50 == 0:
            lg_norm = float(np.linalg.norm(grad_best @ g))
            g_norm = float(np.linalg.norm(grad_best))
            print(f"[MPCBF_Quad3D] step={self.curr_step} best={best_name} V={V_best:.3f} time_dV={self._last_time_derivative:.3f} |g|={g_norm:.3f} |Lg|={lg_norm:.3f}")

        u = cp.Variable(4)
        cost = cp.sum_squares(u - u_nom)
        constraints = []
        self._add_input_constraints(u, constraints)
        self._add_cbf_constraints(u, constraints, state, V_best, grad_best)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        res = None
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
            if prob.status in ['optimal', 'optimal_inaccurate'] and u.value is not None:
                res = u.value
        except Exception:
            res = None

        if res is None:
            ptype, pparams = policy_params_used.get(best_name, ('waypoint', self.policy_configs['nominal'][1]))
            if ptype == 'angle':
                res = np.array(AnglePolicyJAX.compute(jnp.array(state), pparams))
            elif ptype == 'stop':
                res = np.array(StopPolicyJAX.compute(jnp.array(state), pparams))
            elif ptype == 'waypoint':
                res = np.array(WaypointPolicyJAX.compute(jnp.array(state), pparams))
            else:
                res = u_nom

        self.curr_step += 1
        return res

    def _add_cbf_constraints(self, u, constraints, state, V, grad_V):
        if np.isfinite(V) and np.all(np.isfinite(grad_V)):
            f, g = self._get_system_matrices(state)
            Lf_V = np.dot(grad_V, f)
            Lg_V = grad_V @ g
            time_term = getattr(self, "_last_time_derivative", 0.0)
            constraints.append(Lg_V @ u >= -self.cbf_alpha * V - Lf_V - time_term)

        self._add_static_hocbf_constraints(u, constraints, state)
