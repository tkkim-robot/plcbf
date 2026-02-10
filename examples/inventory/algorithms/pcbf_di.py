"""
Created on February 4th, 2026
@author: Taekyung Kim

@description:
PCBF implementation for Double Integrator (PCBF_DI).
Adapts PCBF to 4D state space and handles static obstacles explicitly in QP.
"""

from typing import Callable, Optional, Tuple, List, Dict
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import cvxpy as cp

from examples.inventory.dynamics.dynamics_di_jax import DoubleIntegratorDynamicsJAX, DIDynamicsParams
from examples.inventory.controllers.policies_di_jax import (
    AnglePolicyJAX, StopPolicyJAX, WaypointPolicyJAX,
    AnglePolicyParams, StopPolicyParams, WaypointPolicyParams
)
from examples.drift_car.algorithms.pcbf_drift import smooth_min
from mpcbf.pcbf import PCBFBase

# =============================================================================
# Value Function Computation (JAX)
# =============================================================================

def _rollout_trajectory_di(
    dynamics_params: DIDynamicsParams,
    policy_params,
    policy_type: str,
    x0: jnp.ndarray,
    horizon: int,
    dt: float
) -> jnp.ndarray:
    """Rollout trajectory for DI."""
    
    # Dynamics step function closure
    def step_fn(x, u):
        # Re-implement step logic here or helper to avoid circular imports if strictly needed,
        # but better to use the class method if possible. 
        # For JIT, we need pure functions.
        # Let's inline simple DI dynamics for max speed/simplicity in JIT
        x_pos, y_pos, vx, vy = x[0], x[1], x[2], x[3]
        ax, ay = u[0], u[1]
        
        # Euler integration to match environment truth
        x_next = x_pos + vx * dt
        y_next = y_pos + vy * dt
        vx_next = vx + ax * dt
        vy_next = vy + ay * dt
        
        # Velocity clamp
        v_sq = vx_next**2 + vy_next**2
        v_norm = jnp.sqrt(v_sq + 1e-8)
        scale = jnp.where(v_norm > dynamics_params.v_max, dynamics_params.v_max / v_norm, 1.0)
        
        return jnp.array([x_next, y_next, vx_next*scale, vy_next*scale])

    def policy_fn(state):
        if policy_type == 'angle':
            return AnglePolicyJAX.compute(state, policy_params)
        elif policy_type == 'stop':
            return StopPolicyJAX.compute(state, policy_params)
        elif policy_type == 'waypoint':
            return WaypointPolicyJAX.compute(state, policy_params)
        else:
            return jnp.zeros(2) # Fallback

    def body_fn(carry, _):
        x = carry
        u = policy_fn(x)
        x_next = step_fn(x, u)
        return x_next, x_next

    _, trajectory = lax.scan(body_fn, x0, None, length=horizon)
    return jnp.vstack([x0[None, :], trajectory])

def _compute_value_pure_di(
    x0: jnp.ndarray,
    dynamics_params: DIDynamicsParams,
    policy_params: object, # NamedTuple
    dynamic_obstacles_array: jnp.ndarray, # (n, 5) [x, y, r, vx, vy]
    static_obstacles_array: jnp.ndarray, # (m, 3) [x, y, r]
    policy_type: str,
    horizon: int,
    robot_radius: float, # robot radius + safety margin
    robot_radius_base: float, # robot radius
    dt: float
) -> Tuple[float, jnp.ndarray]:
    """
    Compute Value function V(x) = min h(x(t)) for DI.
    Includes moving obstacles (with margin).
    """
    trajectory = _rollout_trajectory_di(dynamics_params, policy_params, policy_type, x0, horizon, dt)
    times = jnp.arange(horizon + 1) * dt
    
    def compute_h(state, t):
        # Distance to dynamic obstacles
        x, y = state[0], state[1]
        
        def h_single_dyn(obs):
            # obs: [x, y, r, vx, vy]
            # Predict obs pos (with bouncing)
            obs_x = obs[0] + obs[3] * t
            obs_y = obs[1] + obs[4] * t
            
            # obs_x = jnp.where(obs_x < 2.0, 4.0 - obs_x, obs_x)
            # obs_x = jnp.where(obs_x > 98.0, 196.0 - obs_x, obs_x)
            # obs_y = jnp.where(obs_y < 2.0, 4.0 - obs_y, obs_y)
            # obs_y = jnp.where(obs_y > 98.0, 196.0 - obs_y, obs_y)
            
            #dist = jnp.sqrt((x - obs_x)**2 + (y - obs_y)**2 + 1e-8)
            dist = (x - obs_x)**2 + (y - obs_y)**2 
            return dist - (obs[2] + robot_radius)**2

        def h_single_stat(obs):
            # obs: [x, y, r]
            #dist = jnp.sqrt((x - obs[0])**2 + (y - obs[1])**2 + 1e-8)
            dist = (x - obs[0])**2 + (y - obs[1])**2 
            return dist - (obs[2] + robot_radius_base)**2
            
        h_dyn = 100.0
        if dynamic_obstacles_array.shape[0] > 0:
            h_dyn = smooth_min(jax.vmap(h_single_dyn)(dynamic_obstacles_array), temperature=100.0)

        h_stat = 100.0
        if static_obstacles_array.shape[0] > 0:
            h_stat = smooth_min(jax.vmap(h_single_stat)(static_obstacles_array), temperature=100.0)
            
        return jnp.minimum(h_dyn, h_stat)

    h_all = jax.vmap(compute_h)(trajectory, times)
    V = smooth_min(h_all, temperature=40.0)
    
    return V, trajectory

# JIT compiled version
_compute_value_jit_di = jax.jit(
    _compute_value_pure_di,
    static_argnums=(5, 6) # policy_type, horizon
)



# =============================================================================
# PCBF_DI Class - Double Integrator Implementation
# =============================================================================

class PCBF_DI(PCBFBase):
    """
    PCBF for Double Integrator dynamics.
    
    Inherits common QP solving logic from PCBFBase.
    Overrides DI-specific:
    - 4-state dynamics (x, y, vx, vy)
    - Angle and stop policies
    - HO-CBF for static obstacles (in QP) AND Value Function check
    """
    
    def _setup_dynamics(self):
        """Initialize DI dynamics."""
        self.dynamics_jax = DoubleIntegratorDynamicsJAX(self.robot_spec, self.dt)
        self.dynamics_params = DIDynamicsParams(
            v_max=float(self.robot_spec.get('v_max', 10.0)),
            a_max=float(self.robot_spec.get('a_max', 5.0)),
            ax_max=float(self.robot_spec.get('ax_max', 5.0)),
            ay_max=float(self.robot_spec.get('ay_max', 5.0))
        )
        
        # JIT cache
        self._jit_val_grad = None
        self._jit_traj = None
        
    def set_policy(self, type_str: str, params):
        """Set backup policy for DI."""
        if type_str != getattr(self, 'policy_type', None):
             # Only reset cache if policy type changes (static argument)
             self._jit_val_grad = None 
             self.policy_type = type_str
             
        self.backup_policy_params = params
    
    def _get_control_dim(self) -> int:
        """DI control dimension: [ax, ay]."""
        return 2
    
    def _add_input_constraints(self, u, constraints):
        """Add DI input limits: acceleration bounds."""
        constraints.append(u[0] <= self.dynamics_params.ax_max)
        constraints.append(u[0] >= -self.dynamics_params.ax_max)
        constraints.append(u[1] <= self.dynamics_params.ay_max)
        constraints.append(u[1] >= -self.dynamics_params.ay_max)
    
    def _get_system_matrices(self, state):
        """
        Get DI system matrices.
        
        DI dynamics: ẋ = f(x) + g(x)u
        f(x) = [vx, vy, 0, 0]
        g(x) = [[0, 0], [0, 0], [1, 0], [0, 1]]
        """
        f = np.array([state[2], state[3], 0.0, 0.0])
        g = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        return f, g
    
    def _compute_value_and_grad(self, state):
        """Compute value function for DI via JAX rollout."""
        # Prepare values
        robot_radius = self.robot_spec.get('radius', 1.0) + self.safety_margin
        robot_radius_base = self.robot_spec.get('radius', 1.0)
        
        # Prepare Obstacle Arrays (Dynamic + Static)
        
        # Dynamic
        if self.dynamic_obstacles:
            dyn_obs_array = jnp.array([
                (o['x'], o['y'], o['radius'], o.get('vx', 0.0), o.get('vy', 0.0)) 
                for o in self.dynamic_obstacles
            ])
        else:
            dyn_obs_array = jnp.zeros((0, 5))
            
        # Static
        if self.static_obstacles:
            stat_obs_array = jnp.array([
                (o['x'], o['y'], o['radius']) 
                for o in self.static_obstacles
            ])
        else:
            stat_obs_array = jnp.zeros((0, 3))
        
        # Get or create JIT function
        val_grad_fn, traj_fn = self._get_jit_val_grad()
        
        # Compute value and gradient
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
        
        # Get trajectory
        trajectory = traj_fn(
            jnp.array(state), self.dynamics_params, self.backup_policy_params,
            dyn_obs_array, stat_obs_array, self.policy_type, self.backup_horizon_steps, 
            robot_radius, robot_radius_base, self.dt
        )
        
        # Store for visualization
        self.latest_trajectory = np.array(trajectory)
        
        return float(V_jax), np.array(grad_jax), np.array(trajectory)
    
    def get_backup_trajectory(self) -> Optional[np.ndarray]:
        """Return the latest backup trajectory (e.g. for visualization)."""
        return self.latest_trajectory
    
    def _add_cbf_constraints(self, u, constraints, state, V, grad_V, slack=0.0):
        """
        Add CBF constraints for DI.
        """

        if V < 10.0:
            f, g = self._get_system_matrices(state)
            L_f_V = np.dot(grad_V, f)
            L_g_V = grad_V[2:4]  # Only velocity components affect control
            
            # Use slack variable (if passed as Variable or float)
            constraints.append(L_g_V @ u >= -self.cbf_alpha * V - L_f_V - slack)
        
        # 2. Static obstacles: HO-CBF (second-order)
        # (Slack also applied here?)
        gamma1, gamma2 = 2.0, 2.0
        
        # Static hurdles: additive safety margin
        robot_radius = self.robot_spec.get('radius', 1.0) + self.safety_margin
        
        for obs in self.static_obstacles:
            ox, oy = obs['x'], obs['y']
            r = obs['radius'] + robot_radius
            
            px, py, vx, vy = state
            dx, dy = px - ox, py - oy
            dist_sq = dx**2 + dy**2
            
            h = dist_sq - r**2
            h_dot = 2 * (dx*vx + dy*vy)
            term_v = 2 * (vx**2 + vy**2)
            
            lhs = np.array([2*dx, 2*dy])
            rhs = -(term_v + (gamma1 + gamma2)*h_dot + gamma1*gamma2*h)
            
            constraints.append(lhs @ u >= rhs - slack)
    
    def _get_jit_val_grad(self):
        """Get or create JIT-compiled value_and_grad function."""
        if self._jit_val_grad is None:
            def val_fn(x0, dyn_p, pol_p, dyn_obs, stat_obs, p_type, hor, r_rad, rr_base, dt_val):
                V, _ = _compute_value_pure_di(x0, dyn_p, pol_p, dyn_obs, stat_obs, p_type, hor, r_rad, rr_base, dt_val)
                return V
            
            self._jit_val_grad = jax.jit(
                jax.value_and_grad(val_fn),
                static_argnums=(5, 6)  # policy_type, horizon
            )
            
            self._jit_traj = jax.jit(
                lambda x0, dp, pp, do, so, pt, h, rr, rb, dt: _compute_value_pure_di(x0, dp, pp, do, so, pt, h, rr, rb, dt)[1],
                static_argnums=(5, 6)
            )
        
        return self._jit_val_grad, self._jit_traj


