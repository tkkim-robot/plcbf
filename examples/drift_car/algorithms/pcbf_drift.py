"""
Created on January 7th, 2026
@author: Taekyung Kim

@description:
Policy Control Barrier Function (PCBF) implementation for drift car.
Inherits from PCBFBase and implements drift car-specific methods.
"""

from functools import partial
from typing import Tuple, Optional, Any

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import cvxpy as cp

# Import from extracted modules
from examples.drift_car.dynamics.drift_dynamics_jax import (
    DriftingCarDynamicsJAX,
    DynamicsParams,
    angle_normalize_jax,
)
from examples.drift_car.controllers.drift_policies_jax import (
    LaneChangeControllerJAX,
    StoppingControllerJAX,
    BackupPolicyJAX,
    LaneChangePolicyParams,
    StopPolicyParams,
)

# Import base class
from mpcbf.pcbf import PCBFBase

# =============================================================================
# Value Function Computation
# =============================================================================

def smooth_min(x: jnp.ndarray, temperature: float = 10.0) -> float:
    """Smooth approximation of min using negative log-sum-exp."""
    neg_x = -x
    x_max = jnp.max(neg_x)
    smooth_max_neg = x_max + jnp.log(jnp.mean(jnp.exp(temperature * (neg_x - x_max)))) / temperature
    return -smooth_max_neg


def _rollout_trajectory(
    dynamics_params: DynamicsParams,
    policy_params,
    policy_type: str,
    x0: jnp.ndarray,
    horizon: int,
    mu: float,
    dt: float,
) -> jnp.ndarray:
    """Rollout trajectory using backup policy. JIT-compiled."""
    
    def step_fn(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        a = dynamics_params.a
        b = dynamics_params.b
        m = dynamics_params.m
        Iz = dynamics_params.Iz
        Cc_f = dynamics_params.Cc_f
        Cc_r = dynamics_params.Cc_r
        r_w = dynamics_params.r_w
        gamma = dynamics_params.gamma
        gravity = 9.81
        
        delta_max = dynamics_params.delta_max
        tau_max = dynamics_params.tau_max
        v_max = dynamics_params.v_max
        v_min = dynamics_params.v_min
        r_max = dynamics_params.r_max
        beta_max = dynamics_params.beta_max
        
        L = a + b
        Fz_f = m * gravity * b / L
        Fz_r = m * gravity * a / L
        
        x_pos, y_pos, theta = x[0], x[1], x[2]
        r_state, beta, V_vel, delta, tau = x[3], x[4], x[5], x[6], x[7]
        V_safe = jnp.maximum(V_vel, 0.1)
        
        alpha_f = jnp.arctan2(V_vel * jnp.sin(beta) + a * r_state, V_safe * jnp.cos(beta)) - delta
        alpha_r = jnp.arctan2(V_vel * jnp.sin(beta) - b * r_state, V_safe * jnp.cos(beta))
        
        F_lim_r = mu * Fz_r
        Fx_r = F_lim_r * jnp.tanh(tau / (r_w * jnp.maximum(F_lim_r, 1.0)))
        Fx_f = 0.0
        
        def compute_Fy(alpha, Cc, Fz, Fx):
            Fy_max = jnp.sqrt(jnp.maximum((mu * Fz)**2 - gamma * Fx**2, 1.0))
            alpha_sl = jnp.arctan(3 * Fy_max / Cc)
            tan_alpha = jnp.tan(alpha)
            Fy_linear = (-Cc * tan_alpha 
                        + (Cc**2 / (3 * Fy_max)) * jnp.abs(tan_alpha) * tan_alpha
                        - (Cc**3 / (27 * Fy_max**2)) * tan_alpha**3)
            Fy_saturated = -Fy_max * jnp.sign(alpha)
            return jnp.where(jnp.abs(alpha) < alpha_sl, Fy_linear, Fy_saturated)
        
        Fy_f = compute_Fy(alpha_f, Cc_f, Fz_f, Fx_f)
        Fy_r = compute_Fy(alpha_r, Cc_r, Fz_r, Fx_r)
        
        r_dot = (a * (Fx_f * jnp.sin(delta) + Fy_f * jnp.cos(delta)) - b * Fy_r) / Iz
        beta_dot = ((Fx_f * jnp.sin(delta - beta) + Fy_f * jnp.cos(delta - beta)
                    - Fx_r * jnp.sin(beta) + Fy_r * jnp.cos(beta)) / (m * V_safe) - r_state)
        V_dot = ((Fx_f * jnp.cos(delta - beta) - Fy_f * jnp.sin(delta - beta)
                 + Fx_r * jnp.cos(beta) + Fy_r * jnp.sin(beta)) / m)
        
        r_next = jnp.clip(r_state + r_dot * dt, -r_max, r_max)
        beta_next = jnp.clip(beta + beta_dot * dt, -beta_max, beta_max)
        V_next = jnp.clip(V_vel + V_dot * dt, v_min, v_max)
        delta_next = jnp.clip(delta + u[0] * dt, -delta_max, delta_max)
        tau_next = jnp.clip(tau + u[1] * dt, -tau_max, tau_max)
        
        vx_global = V_vel * jnp.cos(theta + beta)
        vy_global = V_vel * jnp.sin(theta + beta)
        x_next = x_pos + vx_global * dt
        y_next = y_pos + vy_global * dt
        theta_next = jnp.mod(theta + r_state * dt + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        return jnp.array([x_next, y_next, theta_next, r_next, beta_next, V_next, delta_next, tau_next])
    
    def policy_fn(state: jnp.ndarray) -> jnp.ndarray:
        if policy_type == 'lane_change':
            return LaneChangeControllerJAX.compute(state, policy_params)
        else:
            return StoppingControllerJAX.compute(state, policy_params)
    
    def body_fn(carry, _):
        x = carry
        u = policy_fn(x)
        x_next = step_fn(x, u)
        return x_next, x_next
    
    _, trajectory = lax.scan(body_fn, x0, None, length=horizon)
    return jnp.vstack([x0[None, :], trajectory])


def _compute_value_pure(
    x0: jnp.ndarray,
    dynamics_params: DynamicsParams,
    policy_params,
    obstacles_array: jnp.ndarray,
    policy_type: str,
    horizon: int,
    robot_radius: float,
    mu: float,
    dt: float,
    track_width: float = 1000.0,
) -> Tuple[float, jnp.ndarray]:
    """Pure JAX value function computation."""
    trajectory = _rollout_trajectory(dynamics_params, policy_params, policy_type, 
                                      x0, horizon, mu, dt)
    
    def compute_h_at_state(state: jnp.ndarray) -> float:
        x, y = state[0], state[1]
        
        def h_single_obs(obs):
            obs_x, obs_y, obs_r = obs[0], obs[1], obs[2]
            combined_r = obs_r + robot_radius
            dist = jnp.sqrt((x - obs_x)**2 + (y - obs_y)**2 + 1e-8)
            return dist - combined_r
        
        h_obs = jnp.where(
            obstacles_array.shape[0] > 0,
            smooth_min(jax.vmap(h_single_obs)(obstacles_array), temperature=5.0),
            100.0
        )
        
        y_max = track_width / 2.0 - robot_radius
        y_min = -track_width / 2.0 + robot_radius
        h_bound_upper = y_max - y
        h_bound_lower = y - y_min
        h_bound = smooth_min(jnp.array([h_bound_upper, h_bound_lower]), temperature=5.0)
        
        return smooth_min(jnp.array([h_obs, h_bound]), temperature=5.0)
    
    h_all = jax.vmap(compute_h_at_state)(trajectory)
    V = smooth_min(h_all, temperature=20.0)
    
    return V, trajectory


# JIT-compiled version for value only
_compute_value_jit = jax.jit(_compute_value_pure, static_argnums=(4, 5))

# JIT-compiled version for value_and_grad (module level to avoid recompilation)
def _value_only_for_grad(x0, dynamics_params, policy_params, obstacles_arr,
                         policy_type, horizon, robot_r, mu, dt, track_width):
    """Wrapper for value_and_grad - returns V as main output, trajectory as aux."""
    V, trajectory = _compute_value_pure(
        x0, dynamics_params, policy_params, obstacles_arr,
        policy_type, horizon, robot_r, mu, dt, track_width
    )
    return V, trajectory

_compute_value_and_grad_jit = jax.jit(
    jax.value_and_grad(_value_only_for_grad, has_aux=True),
    static_argnums=(4, 5)
)


# =============================================================================
# PCBF Controller (Drift Car)
# =============================================================================

class PCBF(PCBFBase):
    """
    Policy Control Barrier Function controller for drift car.
    Inherits from PCBFBase and implements drift car-specific methods.
    """
    
    def __init__(
        self,
        robot,
        robot_spec: dict,
        dt: float = 0.05,
        backup_horizon: float = 10.0,
        cbf_alpha: float = 1.0,
        safety_margin: float = 0.0,
        ax=None,
    ):
        # Store robot instance (drift car specific)
        self.robot = robot
        self.stop_backup_horizon = 3.0
        self.track_width = None
        
        # Call parent init
        super().__init__(robot_spec, dt, backup_horizon, cbf_alpha, safety_margin, ax)
        
        self.stop_backup_horizon_steps = int(self.stop_backup_horizon / dt)
        
        # Control limits
        self.u_min = np.array([-robot_spec.get('delta_dot_max', np.deg2rad(15)),
                               -robot_spec.get('tau_dot_max', 8000.0)])
        self.u_max = np.array([robot_spec.get('delta_dot_max', np.deg2rad(15)),
                               robot_spec.get('tau_dot_max', 8000.0)])
        
        # Visualization
        self.backup_trajs = []
        self.save_every_N = 1
        self.curr_step = 0
        
        # Status
        self.status = 'optimal'
        
        # JIT cache
        self._jit_value_and_grad = None
        
        # Setup dynamics params
        self._setup_dynamics_params()
    
    def _setup_dynamics(self):
        """Initialize drift car dynamics."""
        self.dynamics = DriftingCarDynamicsJAX(self.robot_spec, self.dt)
        self.dynamics_jax = self.dynamics  # Alias for compatibility
    
    def _setup_dynamics_params(self):
        """Setup dynamics parameters as NamedTuple for JIT compilation."""
        self.dynamics_params = DynamicsParams(
            a=float(self.robot_spec.get('a', 1.4)),
            b=float(self.robot_spec.get('b', 1.4)),
            m=float(self.robot_spec.get('m', 2500.0)),
            Iz=float(self.robot_spec.get('Iz', 5000.0)),
            Cc_f=float(self.robot_spec.get('Cc_f', 80000.0)),
            Cc_r=float(self.robot_spec.get('Cc_r', 100000.0)),
            r_w=float(self.robot_spec.get('r_w', 0.35)),
            gamma=float(self.robot_spec.get('gamma', 0.95)),
            delta_max=float(self.robot_spec.get('delta_max', np.deg2rad(20))),
            tau_max=float(self.robot_spec.get('tau_max', 4000.0)),
            v_max=float(self.robot_spec.get('v_max', 20.0)),
            v_min=float(self.robot_spec.get('v_min', 0.0)),
            r_max=float(self.robot_spec.get('r_max', 2.0)),
            beta_max=float(self.robot_spec.get('beta_max', np.deg2rad(45))),
        )
    
    def _get_system_matrices(self, state):
        """Get system dynamics matrices f(x) and g(x)."""
        x_jax = jnp.array(state)
        f = np.array(self.dynamics.f_full(x_jax, self.current_friction))
        G = np.array(self.dynamics.g_full(x_jax))
        return f, G
    
    def _add_input_constraints(self, u, constraints):
        """Add input limit constraints to QP."""
        constraints.append(u >= self.u_min)
        constraints.append(u <= self.u_max)
    
    def _add_cbf_constraints(self, u, constraints, state, V, grad_V):
        """Add CBF constraints to QP - handled in _solve_cbf_qp."""
        pass  # CBF constraint is added in _solve_cbf_qp
    
    def _get_control_dim(self):
        """Return control dimension."""
        return 2
    
    def set_policy(self, policy_type: str, params: Any):
        """Set backup policy (required by base class)."""
        self.policy_type = policy_type
        self.policy_params = params
    
    def set_backup_controller(self, backup_controller, target=None):
        """Set the backup controller (drift car specific interface)."""
        self.backup_controller = backup_controller
        self.backup_target = target
        
        controller_name = backup_controller.get_behavior_name() if hasattr(backup_controller, 'get_behavior_name') else ''
        
        if 'LaneChange' in controller_name:
            self.backup_policy_jax = LaneChangeControllerJAX(self.robot_spec, target_y=target)
            self.policy_type = 'lane_change'
            self.policy_params = LaneChangePolicyParams(
                target_y=float(target), Kp_y=0.15, Kp_theta=1.5, Kd_theta=0.3,
                Kp_delta=3.0, Kp_v=500.0, Kp_tau_dot=2.0,
                target_velocity=float(self.robot_spec.get('v_ref', 8.0)),
                delta_max=float(self.robot_spec.get('delta_max', np.deg2rad(20))),
                delta_dot_max=float(self.robot_spec.get('delta_dot_max', np.deg2rad(15))),
                tau_max=float(self.robot_spec.get('tau_max', 4000.0)),
                tau_dot_max=float(self.robot_spec.get('tau_dot_max', 8000.0)),
                theta_des_max=float(np.deg2rad(15)),
            )
        elif 'Stop' in controller_name:
            self.backup_policy_jax = StoppingControllerJAX(self.robot_spec)
            self.policy_type = 'stop'
            self.policy_params = StopPolicyParams(
                Kd_theta=0.5, Kp_delta=3.0, Kp_v=1000.0, Kp_tau_dot=2.0,
                delta_max=float(self.robot_spec.get('delta_max', np.deg2rad(20))),
                delta_dot_max=float(self.robot_spec.get('delta_dot_max', np.deg2rad(15))),
                tau_max=float(self.robot_spec.get('tau_max', 4000.0)),
                tau_dot_max=float(self.robot_spec.get('tau_dot_max', 8000.0)),
                stop_threshold=0.05, holding_torque=-100.0,
            )
        else:
            self.backup_policy_jax = BackupPolicyJAX(backup_controller, target)
            self.policy_type = 'custom'
            self.policy_params = None
        
        self._jit_value_and_grad = None
    
    def set_environment(self, env):
        """Set the environment for obstacle information."""
        self.env = env
        if hasattr(env, 'track_width'):
            self.track_width = float(env.track_width)
        self._update_obstacles()
    
    def _update_obstacles(self):
        """Update obstacle list from environment."""
        if self.env is None:
            return
        
        new_obstacles = []
        if hasattr(self.env, 'obstacles'):
            for obs in self.env.obstacles:
                obs_x = obs['x']
                obs_y = obs['y']
                obs_radius = obs['spec'].get('radius', 2.5)
                new_obstacles.append((obs_x, obs_y, obs_radius))
        self.obstacles = new_obstacles
    
    def set_friction(self, mu: float):
        """Set current friction coefficient."""
        self.current_friction = mu
        self.dynamics.mu = mu
    
    def _compute_value_and_grad(self, x0_jax: jnp.ndarray) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
        """Compute value function V and its gradient using module-level JIT functions."""
        robot_radius = self.robot_spec.get('radius', 1.5) + self.safety_margin
        
        if len(self.obstacles) == 0:
            return 100.0, jnp.zeros(8), x0_jax[None, :]
        
        obstacles_array = jnp.array(self.obstacles)
        
        # Use shorter horizon for stopping backup
        horizon_steps = self.stop_backup_horizon_steps if self.policy_type == 'stop' else self.backup_horizon_steps
        track_width = self.track_width if self.track_width is not None else 1000.0
        
        # Use module-level JIT function (no re-compilation)
        (V, trajectory), grad_V = _compute_value_and_grad_jit(
            x0_jax, self.dynamics_params, self.policy_params, obstacles_array,
            self.policy_type, horizon_steps, robot_radius, self.current_friction,
            self.dt, track_width,
        )
        
        return V, grad_V, trajectory
    
    def _solve_cbf_qp(self, u_nom: np.ndarray, V: float, grad_V: np.ndarray,
                       f: np.ndarray, G: np.ndarray) -> np.ndarray:
        """Solve the CBF-QP using CVXPY."""
        nu = 2
        u_scale = self.u_max
        u_nom_scaled = u_nom / u_scale
        u_scaled = cp.Variable(nu)
        
        weights = np.array([1.0, 10.0])
        weighted_diff = cp.multiply(weights, u_scaled - u_nom_scaled)
        cost = cp.sum_squares(weighted_diff)
        
        grad_V_G = grad_V @ G
        grad_V_f = grad_V @ f
        cbf_rhs = grad_V_f + self.cbf_alpha * V
        A_cbf = -grad_V_G * u_scale
        
        constraints = [
            A_cbf @ u_scaled <= cbf_rhs + 1e-4,
            u_scaled >= -1.0,
            u_scaled <= 1.0,
        ]
        
        try:
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(solver=cp.SCS, verbose=False, max_iters=2000, eps=1e-4)
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                self.status = 'optimal'
                return u_scaled.value * u_scale
            elif problem.status in ['infeasible', 'infeasible_inaccurate']:
                self.status = 'infeasible'
                raise ValueError("CBF-QP infeasible")
            else:
                self.status = problem.status
                raise ValueError(f"CBF-QP failed: {problem.status}")
        except cp.error.SolverError as e:
            self.status = 'error'
            raise ValueError(f"CBF-QP solver error: {e}")
    
    def solve_control_problem(
        self,
        robot_state: np.ndarray,
        control_ref: Optional[dict] = None,
        friction: Optional[float] = None,
    ) -> np.ndarray:
        """Main PCBF control loop - compute safe control input."""
        robot_state = np.array(robot_state).flatten()
        
        if friction is not None:
            self.set_friction(friction)
        
        if control_ref is not None and 'u_ref' in control_ref:
            u_nom = np.array(control_ref['u_ref']).flatten()
        else:
            u_nom = np.zeros(2)
        
        self._update_obstacles()
        
        if len(self.obstacles) == 0 or self.backup_policy_jax is None:
            self.status = 'optimal'
            return u_nom.reshape(-1, 1)
        
        x0_jax = jnp.array(robot_state)
        
        try:
            V, grad_V, trajectory = self._compute_value_and_grad(x0_jax)
            V = float(V)
            grad_V = np.array(grad_V)
            trajectory = np.array(trajectory)
        except Exception as e:
            print(f"Value computation failed: {e}")
            self.status = 'error'
            return u_nom.reshape(-1, 1)
        
        grad_norm = np.linalg.norm(grad_V)
        max_grad_norm = 50.0
        if grad_norm > max_grad_norm:
            grad_V = grad_V * (max_grad_norm / grad_norm)
        
        f = np.array(self.dynamics.f_full(x0_jax, self.current_friction))
        G = np.array(self.dynamics.g_full(x0_jax))
        
        u_safe = self._solve_cbf_qp(u_nom, V, grad_V, f, G)
        
        self._update_visualization(trajectory)
        
        return u_safe.reshape(-1, 1)
    
    def _setup_visualization(self):
        """Setup visualization handles."""
        if self.ax is None:
            return
        self.backup_traj_line, = self.ax.plot(
            [], [], '-', color='cyan', linewidth=2, alpha=0.8,
            label='PCBF backup rollout', zorder=18
        )
    
    def _update_visualization(self, trajectory: np.ndarray):
        """Update visualization with backup trajectory."""
        if self.ax is None:
            return
        if self.curr_step % self.save_every_N == 0:
            self.backup_trajs.append(trajectory.copy())
        self.curr_step += 1
        if self.backup_traj_line is not None and trajectory is not None:
            self.backup_traj_line.set_data(trajectory[:, 0], trajectory[:, 1])
    
    def get_backup_trajectories(self):
        return self.backup_trajs.copy()
    
    def clear_trajectories(self):
        self.backup_trajs.clear()
    
    def is_using_backup(self):
        return False
    
    def get_status(self):
        return {
            'status': self.status, 'using_backup': False,
            'cbf_alpha': self.cbf_alpha, 'backup_horizon': self.backup_horizon,
            'num_obstacles': len(self.obstacles),
        }
