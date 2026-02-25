"""
Created on February 11th, 2026
@author: Taekyung Kim

@description:
PCBF implementation for Quad3D (PCBF_Quad3D).
"""

from typing import Tuple
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import cvxpy as cp

from examples.warehouse.dynamics.dynamics_quad3d_jax import Quad3DDynamicsJAX, Quad3DDynamicsParams, _build_quad3d_matrices
from examples.warehouse.controllers.policies_quad3d_jax import (
    AnglePolicyJAX, StopPolicyJAX, WaypointPolicyJAX,
    AnglePolicyParams, StopPolicyParams, WaypointPolicyParams, RetracePolicyParams, Quad3DControlParams,
    _clip_xy_accel, _accel_to_u
)
from examples.drift_car.algorithms.pcbf_drift import smooth_min
from plcbf.pcbf import PCBFBase


# =============================================================================
# Value Function Computation (JAX)
# =============================================================================

def _rollout_trajectory_quad3d(
    dynamics_params: Quad3DDynamicsParams,
    policy_params,
    policy_type: str,
    x0: jnp.ndarray,
    horizon: int,
    dt: float
) -> jnp.ndarray:
    """Rollout trajectory for Quad3D."""

    A = dynamics_params.A
    B = dynamics_params.B
    u_min = dynamics_params.u_min
    u_max = dynamics_params.u_max

    def step_fn(x, u):
        u = jnp.clip(u, u_min, u_max)
        k1 = A @ x + B @ u
        k2 = A @ (x + 0.5 * dt * k1) + B @ u
        k3 = A @ (x + 0.5 * dt * k2) + B @ u
        k4 = A @ (x + dt * k3) + B @ u
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # normalize angles
        x_next = x_next.at[3].set(((x_next[3] + jnp.pi) % (2.0 * jnp.pi)) - jnp.pi)
        x_next = x_next.at[4].set(((x_next[4] + jnp.pi) % (2.0 * jnp.pi)) - jnp.pi)
        x_next = x_next.at[5].set(((x_next[5] + jnp.pi) % (2.0 * jnp.pi)) - jnp.pi)
        return x_next

    def policy_fn(state):
        if policy_type == 'angle':
            return AnglePolicyJAX.compute(state, policy_params)
        elif policy_type == 'stop':
            return StopPolicyJAX.compute(state, policy_params)
        elif policy_type == 'waypoint':
            return WaypointPolicyJAX.compute(state, policy_params)
        else:
            return jnp.zeros(4)

    def body_fn(carry, _):
        x = carry
        u = policy_fn(x)
        x_next = step_fn(x, u)
        return x_next, x_next

    if policy_type != 'retrace_waypoint':
        _, trajectory = lax.scan(body_fn, x0, None, length=horizon)
        return jnp.vstack([x0[None, :], trajectory])

    # Retrace-waypoint policy: update waypoint index during rollout (match Python retrace behavior)
    params: RetracePolicyParams = policy_params
    ctrl = params.ctrl

    def retrace_step(carry, _):
        x, idx = carry
        pos = x[0:2]
        vel = x[6:8]

        idx = jnp.clip(idx, 0, params.waypoints.shape[0] - 1)
        target = params.waypoints[idx]
        dist = jnp.sqrt(jnp.sum((target - pos) ** 2) + 1e-8)

        # Decrement retrace index when close to target
        idx_next = jax.lax.cond(
            jnp.logical_and(dist < params.dist_threshold, idx > 0),
            lambda i: i - 1,
            lambda i: i,
            idx
        )

        v_des_dir = (target - pos) / (dist + 1e-6)
        braking_speed = jnp.sqrt(2.0 * ctrl.a_max_xy * jnp.maximum(dist, 0.0))
        v_des_speed = jnp.minimum(params.v_max, braking_speed)
        v_des = v_des_dir * v_des_speed

        ax = params.Kp * (v_des[0] - vel[0])
        ay = params.Kp * (v_des[1] - vel[1])
        ax, ay = _clip_xy_accel(ax, ay, ctrl.a_max_xy)

        az = ctrl.Kp_z * (ctrl.z_ref - x[2]) - ctrl.Kd_z * x[8]
        u = _accel_to_u(x, ax, ay, az, ctrl)

        x_next = step_fn(x, u)
        return (x_next, idx_next), x_next

    carry0 = (x0, jnp.array(params.current_wp_idx))
    _, trajectory = lax.scan(retrace_step, carry0, None, length=horizon)
    return jnp.vstack([x0[None, :], trajectory])


def _compute_value_pure_quad3d(
    x0: jnp.ndarray,
    dynamics_params: Quad3DDynamicsParams,
    policy_params: object,
    dynamic_obstacles_array: jnp.ndarray,  # (n, 5) [x, y, r, vx, vy]
    static_obstacles_array: jnp.ndarray,   # (m, 3) [x, y, r]
    policy_type: str,
    horizon: int,
    robot_radius: float,
    robot_radius_base: float,
    dt: float
) -> Tuple[float, jnp.ndarray]:
    """Compute Value function V(x) = min h(x(t)) for Quad3D using XY distance."""
    trajectory = _rollout_trajectory_quad3d(dynamics_params, policy_params, policy_type, x0, horizon, dt)
    times = jnp.arange(horizon + 1) * dt

    def compute_h(state, t):
        x, y = state[0], state[1]

        def h_single_dyn(obs):
            obs_x = obs[0] + obs[3] * t
            obs_y = obs[1] + obs[4] * t
            obs_x = jnp.where(obs_x < 2.0, 4.0 - obs_x, obs_x)
            obs_x = jnp.where(obs_x > 98.0, 196.0 - obs_x, obs_x)
            obs_y = jnp.where(obs_y < 2.0, 4.0 - obs_y, obs_y)
            obs_y = jnp.where(obs_y > 98.0, 196.0 - obs_y, obs_y)
            dist = jnp.sqrt((x - obs_x) ** 2 + (y - obs_y) ** 2 + 1e-8)
            return dist - (obs[2] + robot_radius)

        def h_single_stat(obs):
            dist = jnp.sqrt((x - obs[0]) ** 2 + (y - obs[1]) ** 2 + 1e-8)
            return dist - (obs[2] + robot_radius)

        h_dyn = 100.0
        if dynamic_obstacles_array.shape[0] > 0:
            h_dyn = smooth_min(jax.vmap(h_single_dyn)(dynamic_obstacles_array), temperature=100.0)

        h_stat = 100.0
        if static_obstacles_array.shape[0] > 0:
            h_stat = smooth_min(jax.vmap(h_single_stat)(static_obstacles_array), temperature=100.0)

        return smooth_min(jnp.array([h_dyn, h_stat]), temperature=10.0)

    h_all = jax.vmap(compute_h)(trajectory, times)
    V = smooth_min(h_all, temperature=80.0)
    return V, trajectory


_compute_value_jit_quad3d = jax.jit(
    _compute_value_pure_quad3d,
    static_argnums=(5, 6)
)


# =============================================================================
# PCBF_Quad3D Class
# =============================================================================

class PCBF_Quad3D(PCBFBase):
    """PCBF for Quad3D dynamics."""

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

        self._jit_val_grad = None
        self._jit_traj = None

    def set_policy(self, type_str: str, params):
        if type_str != getattr(self, 'policy_type', None):
            self._jit_val_grad = None
            self.policy_type = type_str
        self.backup_policy_params = params

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

    def _compute_value_and_grad(self, state):
        robot_radius = self.robot_spec.get('radius', 1.0) + self.safety_margin
        robot_radius_base = self.robot_spec.get('radius', 1.0)

        if self.dynamic_obstacles:
            dyn_obs_array = jnp.array([
                (o['x'], o['y'], o['radius'], o.get('vx', 0.0), o.get('vy', 0.0))
                for o in self.dynamic_obstacles
            ])
        else:
            dyn_obs_array = jnp.zeros((0, 5))

        if self.static_obstacles:
            stat_obs_array = jnp.array([
                (o['x'], o['y'], o['radius'])
                for o in self.static_obstacles
            ])
        else:
            stat_obs_array = jnp.zeros((0, 3))

        val_grad_fn, traj_fn = self._get_jit_val_grad()

        V_jax, grad_jax = val_grad_fn(
            jnp.array(state),
            self.dynamics_params,
            self.backup_policy_params,
            dyn_obs_array,
            stat_obs_array,
            self.policy_type,
            self.backup_horizon_steps,
            robot_radius,
            robot_radius_base,
            self.dt
        )

        trajectory = traj_fn(
            jnp.array(state),
            self.dynamics_params,
            self.backup_policy_params,
            dyn_obs_array,
            stat_obs_array,
            self.policy_type,
            self.backup_horizon_steps,
            robot_radius,
            robot_radius_base,
            self.dt
        )

        self.latest_trajectory = np.array(trajectory)
        return float(V_jax), np.array(grad_jax), self.latest_trajectory

    def _add_static_hocbf_constraints(self, u, constraints, state):
        if not self.static_obstacles:
            return
        g = float(self.dynamics_params.g)
        Ix = float(self.dynamics_params.Ix)
        Iy = float(self.dynamics_params.Iy)
        L = float(self.dynamics_params.L)

        # 4th-order exponential CBF gains: (s + lambda)^4
        lam = 2.5
        k3 = 4.0 * lam
        k2 = 6.0 * lam * lam
        k1 = 4.0 * lam**3
        k0 = lam**4

        robot_radius = self.robot_spec.get('radius', 1.0) + self.safety_margin
        x, y = float(state[0]), float(state[1])
        theta, phi = float(state[3]), float(state[4])
        vx, vy = float(state[6]), float(state[7])
        q, p = float(state[9]), float(state[10])

        ax = g * theta
        ay = -g * phi
        ax_dot = g * q
        ay_dot = -g * p

        for obs in self.static_obstacles:
            ox, oy = obs['x'], obs['y']
            r = obs['radius'] + robot_radius

            px = x - ox
            py = y - oy
            h = px**2 + py**2 - r**2
            h_dot = 2.0 * (px * vx + py * vy)
            h_ddot = 2.0 * (vx**2 + vy**2) + 2.0 * (px * ax + py * ay)
            h_dddot = 6.0 * (vx * ax + vy * ay) + 2.0 * (px * ax_dot + py * ay_dot)

            base = 6.0 * (ax**2 + ay**2) + 8.0 * (vx * ax_dot + vy * ay_dot)

            coeff_tau_y = 2.0 * px * g / Iy
            coeff_tau_x = -2.0 * py * g / Ix

            c1 = coeff_tau_x * L
            c2 = coeff_tau_y * L
            c3 = -coeff_tau_x * L
            c4 = -coeff_tau_y * L

            rhs = -(base + k3 * h_dddot + k2 * h_ddot + k1 * h_dot + k0 * h)
            constraints.append(c1 * u[0] + c2 * u[1] + c3 * u[2] + c4 * u[3] >= rhs)

    def _add_cbf_constraints(self, u, constraints, state, V, grad_V):
        if np.isfinite(V) and np.all(np.isfinite(grad_V)):
            f, g = self._get_system_matrices(state)
            Lf_V = np.dot(grad_V, f)
            Lg_V = grad_V @ g
            constraints.append(Lg_V @ u >= -self.cbf_alpha * V - Lf_V)

        self._add_static_hocbf_constraints(u, constraints, state)

    def _get_jit_val_grad(self):
        if self._jit_val_grad is None:
            def val_fn(x0, dyn_p, pol_p, dyn_obs, stat_obs, p_type, hor, r_rad, rr_base, dt_val):
                V, _ = _compute_value_pure_quad3d(x0, dyn_p, pol_p, dyn_obs, stat_obs, p_type, hor, r_rad, rr_base, dt_val)
                return V

            self._jit_val_grad = jax.jit(
                jax.value_and_grad(val_fn),
                static_argnums=(5, 6)
            )

            self._jit_traj = jax.jit(
                lambda x0, dp, pp, do, so, pt, h, rr, rb, dt: _compute_value_pure_quad3d(x0, dp, pp, do, so, pt, h, rr, rb, dt)[1],
                static_argnums=(5, 6)
            )

        return self._jit_val_grad, self._jit_traj
