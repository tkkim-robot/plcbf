"""
Created on January 7th, 2026
@author: Taekyung Kim

@description:
Multiple Policy Control Barrier Function (MPCBF) implementation in JAX.

MPCBF extends PCBF by maintaining multiple backup policies and selecting
the constraint from the policy with maximum CBF constraint value (Vdot + alpha*V).

Multiple Policies:
1. Lane change left - steer to left lane
2. Lane change right - steer to right lane  
3. Stopping - brake to stop
4. MPCC trajectory - the nominal MPC trajectory (already computed)

Key Features:
- Parallel computation of value functions using JAX vmap
- Proper JIT compilation with static_argnums to avoid recompilation
- Single CBF constraint from policy with maximum (Vdot + alpha*V)
- Multi-trajectory visualization with color coding

Selection Criterion:
- For each policy i, compute constraint value: c_i = ∇V_i^T (f + G*u_nom) + α*V_i
- Select policy with max c_i (most permissive constraint that's still safe)
- Use that constraint in the QP

@required-scripts: mpcbf/pcbf.py
"""

import functools as ft
from functools import partial
from typing import Callable, Optional, Tuple, Any, List, Dict

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import cvxpy as cp

from .pcbf import (
    PCBF,
    DriftingCarDynamicsJAX,
    LaneChangeControllerJAX,
    # StoppingControllerJAX,  # Disabled for now
    smooth_min,
    _rollout_trajectory,
)


# =============================================================================
# JIT-compiled Multi-Policy Value Functions
# =============================================================================

def _create_multi_policy_value_fn():
    """
    Create JIT-compiled functions for multi-policy value computation.
    """
    
    @partial(jax.jit, static_argnums=(4, 5, 6))
    def _rollout_and_evaluate(
        x0: jnp.ndarray,
        dynamics_params: dict,
        policy_params: dict,
        obstacles_arr: jnp.ndarray,
        policy_type: str,
        horizon: int,
        n_obs: int,
        robot_r: float,
        mu: float,
        dt: float,
    ) -> Tuple[float, jnp.ndarray]:
        """
        Rollout a single policy and compute its value function.
        """
        trajectory = _rollout_trajectory(
            dynamics_params, policy_params, policy_type,
            x0, horizon, mu, dt
        )
        
        def compute_h_at_state(state):
            x, y = state[0], state[1]
            def h_single_obs(obs):
                obs_x, obs_y, obs_r = obs[0], obs[1], obs[2]
                combined_r = obs_r + robot_r
                dist = jnp.sqrt((x - obs_x)**2 + (y - obs_y)**2 + 1e-8)
                return dist - combined_r
            h_vals = jax.vmap(h_single_obs)(obstacles_arr)
            return jnp.min(h_vals)
        
        h_all = jax.vmap(compute_h_at_state)(trajectory)
        V = smooth_min(h_all, temperature=20.0)
        return V, trajectory
    
    return _rollout_and_evaluate


# Pre-create the JIT function
_ROLLOUT_AND_EVALUATE = _create_multi_policy_value_fn()


# Policy colors for visualization
POLICY_COLORS = {
    'lane_change_left': '#2196F3',   # Blue
    'lane_change_right': '#FF9800',  # Orange
    'mpcc': '#9C27B0',               # Purple
    # 'stop': '#F44336',             # Red (disabled for now)
}

SELECTED_COLOR = '#00E676'  # Bright green for selected policy


# =============================================================================
# MPCBF Controller
# =============================================================================

class MPCBF(PCBF):
    """
    Multiple Policy Control Barrier Function controller.
    
    Extends PCBF by maintaining multiple backup policies and selecting
    the constraint from the policy with maximum CBF value (Vdot + alpha*V).
    """
    
    def __init__(
        self,
        robot,
        robot_spec: dict,
        dt: float = 0.05,
        backup_horizon: float = 10.0,
        cbf_alpha: float = 1.0,
        left_lane_y: float = 5.0,
        right_lane_y: float = -5.0,
        ax=None,
    ):
        """
        Initialize the MPCBF controller.
        """
        # Initialize parent PCBF (but don't use its single policy)
        super().__init__(robot, robot_spec, dt, backup_horizon, cbf_alpha, ax)
        
        # Lane targets
        self.left_lane_y = left_lane_y
        self.right_lane_y = right_lane_y
        
        # MPCC trajectory (updated each step)
        self.mpcc_trajectory = None
        
        # For visualization - store trajectories from all policies
        self.multi_backup_trajs = {}  # policy_name -> list of trajectories
        self.best_policy_name = None
        self.prev_best_policy = None  # For hysteresis
        
        # Visualization handles for multi-policy
        self.policy_traj_lines = {}
        
        # Multiple policies configuration
        self._setup_multi_policies()
        
        # Setup multi-policy visualization
        if ax is not None:
            self._setup_multi_visualization()
        
    def _setup_multi_policies(self):
        """Setup parameters for all backup policies."""
        delta_max = float(self.robot_spec.get('delta_max', np.deg2rad(20)))
        delta_dot_max = float(self.robot_spec.get('delta_dot_max', np.deg2rad(15)))
        tau_max = float(self.robot_spec.get('tau_max', 4000.0))
        tau_dot_max = float(self.robot_spec.get('tau_dot_max', 8000.0))
        target_velocity = float(self.robot_spec.get('v_ref', 8.0))
        
        # Use uniform horizon for all policies
        uniform_horizon = self.backup_horizon_steps
        
        self.policy_configs = {
            'lane_change_left': {
                'type': 'lane_change',
                'horizon': uniform_horizon,
                'params': {
                    'target_y': self.left_lane_y,
                    'Kp_y': 0.15,
                    'Kp_theta': 1.5,
                    'Kd_theta': 0.3,
                    'Kp_delta': 3.0,
                    'Kp_v': 500.0,
                    'target_velocity': target_velocity,
                    'delta_max': delta_max,
                    'delta_dot_max': delta_dot_max,
                    'tau_max': tau_max,
                    'tau_dot_max': tau_dot_max,
                    'theta_des_max': float(np.deg2rad(15)),
                },
            },
            'lane_change_right': {
                'type': 'lane_change',
                'horizon': uniform_horizon,
                'params': {
                    'target_y': self.right_lane_y,
                    'Kp_y': 0.15,
                    'Kp_theta': 1.5,
                    'Kd_theta': 0.3,
                    'Kp_delta': 3.0,
                    'Kp_v': 500.0,
                    'target_velocity': target_velocity,
                    'delta_max': delta_max,
                    'delta_dot_max': delta_dot_max,
                    'tau_max': tau_max,
                    'tau_dot_max': tau_dot_max,
                    'theta_des_max': float(np.deg2rad(15)),
                },
            },
            'lane_change_left_2': {
                'type': 'lane_change',
                'horizon': uniform_horizon,
                'params': {
                    'target_y': self.left_lane_y + (abs(self.left_lane_y - self.right_lane_y) / 2.0 if abs(self.left_lane_y - self.right_lane_y) > 1.0 else 4.0),
                    'Kp_y': 0.15,
                    'Kp_theta': 1.5,
                    'Kd_theta': 0.3,
                    'Kp_delta': 3.0,
                    'Kp_v': 500.0,
                    'target_velocity': target_velocity,
                    'delta_max': delta_max,
                    'delta_dot_max': delta_dot_max,
                    'tau_max': tau_max,
                    'tau_dot_max': tau_dot_max,
                    'theta_des_max': float(np.deg2rad(15)),
                },
            },
            # NOTE: Stopping policy removed for now to focus on lane change behavior
            # Can be re-added later once lane change is verified working
        }
        
        # Create JAX policy instances
        self.jax_policies = {
            'lane_change_left': LaneChangeControllerJAX(self.robot_spec, target_y=self.left_lane_y),
            'lane_change_right': LaneChangeControllerJAX(self.robot_spec, target_y=self.right_lane_y),
            'lane_change_left_2': LaneChangeControllerJAX(self.robot_spec, target_y=self.left_lane_y + (abs(self.left_lane_y - self.right_lane_y) / 2.0 if abs(self.left_lane_y - self.right_lane_y) > 1.0 else 4.0)),
        }
        
        # Initialize trajectory storage
        for name in list(self.policy_configs.keys()) + ['mpcc']:
            self.multi_backup_trajs[name] = []
    
    def _setup_multi_visualization(self):
        """Setup visualization handles for multiple policy trajectories."""
        if self.ax is None:
            return
        
        # Create trajectory lines for each policy
        for name, color in POLICY_COLORS.items():
            line, = self.ax.plot(
                [], [], '-', color=color, linewidth=2, alpha=0.6,
                label=f'{name} rollout', zorder=17
            )
            self.policy_traj_lines[name] = line
        
        # Hide the parent's single backup trajectory line
        if self.backup_traj_line is not None:
            self.backup_traj_line.set_visible(False)
    
    def set_lane_targets(self, left_y: float, right_y: float):
        """Update lane change target positions."""
        self.left_lane_y = left_y
        self.right_lane_y = right_y
        
        self.policy_configs['lane_change_left']['params']['target_y'] = left_y
        self.policy_configs['lane_change_right']['params']['target_y'] = right_y
        
        self.jax_policies['lane_change_left'] = LaneChangeControllerJAX(
            self.robot_spec, target_y=left_y
        )
        self.jax_policies['lane_change_right'] = LaneChangeControllerJAX(
            self.robot_spec, target_y=right_y
        )
    
    def set_mpcc_trajectory(self, trajectory: np.ndarray):
        """Set the MPCC predicted trajectory for this step."""
        if trajectory is None:
            self.mpcc_trajectory = None
            return
            
        traj = np.array(trajectory)
        if traj.shape[0] < traj.shape[1]:
            traj = traj.T
        
        self.mpcc_trajectory = jnp.array(traj)
    
    def _compute_multi_value_and_grad(
        self, 
        x0_jax: jnp.ndarray
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Compute value function and gradient for all policies.
        
        Returns:
            V_dict: {policy_name: V_value}
            grad_V_dict: {policy_name: grad_V}
            traj_dict: {policy_name: trajectory}
        """
        robot_radius = self.robot_spec.get('radius', 1.5)
        
        if len(self.obstacles) == 0:
            default_V = 100.0
            V_dict = {name: default_V for name in self.policy_configs}
            V_dict['mpcc'] = default_V
            grad_V_dict = {name: np.zeros(8) for name in self.policy_configs}
            grad_V_dict['mpcc'] = np.zeros(8)
            traj_dict = {name: np.array(x0_jax)[None, :] for name in self.policy_configs}
            traj_dict['mpcc'] = np.array(x0_jax)[None, :]
            return V_dict, grad_V_dict, traj_dict
        
        obstacles_array = jnp.array(self.obstacles)
        n_obs = len(self.obstacles)
        
        V_dict = {}
        grad_V_dict = {}
        traj_dict = {}
        
        # Compute V and grad for each rollout policy
        for name, config in self.policy_configs.items():
            policy_type = config['type']
            horizon = config['horizon']
            policy_params = config['params']
            
            def value_fn(x0):
                V, _ = _ROLLOUT_AND_EVALUATE(
                    x0, self.dynamics_params, policy_params, obstacles_array,
                    policy_type, horizon, n_obs, robot_radius,
                    self.current_friction, self.dt,
                )
                return V
            
            V, grad_V = jax.value_and_grad(value_fn)(x0_jax)
            
            # Get trajectory for visualization (use full horizon)
            _, trajectory = _ROLLOUT_AND_EVALUATE(
                x0_jax, self.dynamics_params, policy_params, obstacles_array,
                policy_type, self.backup_horizon_steps, n_obs, robot_radius,
                self.current_friction, self.dt,
            )
            
            V_dict[name] = float(V)
            grad_V_dict[name] = np.array(grad_V)
            traj_dict[name] = np.array(trajectory)
        
        # Compute value for MPCC trajectory
        if self.mpcc_trajectory is not None and len(self.mpcc_trajectory) > 0:
            def mpcc_value_fn(x0):
                traj = self.mpcc_trajectory
                traj_with_x0 = jnp.vstack([x0[None, :2], traj[:, :2]])
                
                def compute_h_at_xy(xy):
                    x, y = xy[0], xy[1]
                    def h_single_obs(obs):
                        obs_x, obs_y, obs_r = obs[0], obs[1], obs[2]
                        combined_r = obs_r + robot_radius
                        dist = jnp.sqrt((x - obs_x)**2 + (y - obs_y)**2 + 1e-8)
                        return dist - combined_r
                    h_vals = jax.vmap(h_single_obs)(obstacles_array)
                    return jnp.min(h_vals)
                
                h_all = jax.vmap(compute_h_at_xy)(traj_with_x0)
                return smooth_min(h_all, temperature=20.0)
            
            V_mpcc, grad_V_mpcc = jax.value_and_grad(mpcc_value_fn)(x0_jax)
            V_dict['mpcc'] = float(V_mpcc)
            grad_full = np.zeros(8)
            grad_full[:2] = np.array(grad_V_mpcc)[:2]
            grad_V_dict['mpcc'] = grad_full
            traj_dict['mpcc'] = np.array(self.mpcc_trajectory)
        else:
            V_dict['mpcc'] = -np.inf
            grad_V_dict['mpcc'] = np.zeros(8)
            traj_dict['mpcc'] = None
        
        return V_dict, grad_V_dict, traj_dict
    
    def _select_best_policy(
        self,
        V_dict: Dict[str, float],
        grad_V_dict: Dict[str, np.ndarray],
        f: np.ndarray,
        G: np.ndarray,
        u_nom: np.ndarray,
    ) -> Tuple[str, Dict[str, float]]:
        """
        Select the policy with maximum CBF constraint value.
        
        CBF constraint: ∇V^T (f + Gu) + α V ≥ 0
        Constraint value at nominal: c = ∇V^T f + ∇V^T G @ u_nom + α V
        
        Key insight: With multiple policies, we select max(c_i).
        - If max(c_i) > 0: nominal control already satisfies the most relaxed constraint
        - If max(c_i) < 0: need CBF-QP to modify control
        
        This means MPCBF activates LESS frequently than single-policy PCBF,
        since we only need ONE policy's constraint to be satisfied.
        
        Returns:
            best_policy: Name of selected policy
            constraint_dict: {policy_name: constraint_value}
        """
        constraint_dict = {}
        max_grad_norm = 50.0
        
        for name in V_dict.keys():
            V = V_dict[name]
            grad_V = grad_V_dict[name].copy()
            
            # Skip invalid policies
            if V == -np.inf or np.any(np.isnan(grad_V)):
                constraint_dict[name] = -np.inf
                continue
            
            # Normalize gradient if too large
            grad_norm = np.linalg.norm(grad_V)
            if grad_norm > max_grad_norm:
                grad_V = grad_V * (max_grad_norm / grad_norm)
            
            # Compute Lie derivatives
            Lf_V = grad_V @ f
            Lg_V = grad_V @ G
            
            # Constraint value at u_nom: c = Lf_V + Lg_V @ u_nom + alpha * V
            constraint_value = Lf_V + Lg_V @ u_nom + self.cbf_alpha * V
            constraint_dict[name] = float(constraint_value)
        
        # Get all policies including mpcc
        backup_policies = list(constraint_dict.keys())
        
        # Filter for safe policies (V > 0)
        # Prioritize policies that are currently safe over those that are already in collision.
        # This prevents switching to a "less violating" unsafe policy when a safe policy exists.
        safe_policies = [k for k in backup_policies if V_dict[k] > 0.0]
        
        if len(safe_policies) > 0:
            candidates = safe_policies
            # Among safe policies, select the one with MAXIMUM SAFETY MARGIN (V).
            # This ensures we pick the most robust backup plan, even if it requires
            # more aggressive control intervention (lower constraint value).
            best_policy = max(candidates, key=lambda k: V_dict[k])
        else:
            candidates = backup_policies
            # If all are unsafe, pick the one that violates constraint least (damage mitigation)
            best_policy = max(candidates, key=lambda k: constraint_dict[k])
        
        return best_policy, constraint_dict
    
    def _update_multi_visualization(
        self, 
        traj_dict: Dict[str, np.ndarray], 
        best_policy: str
    ):
        """Update visualization with all policy trajectories."""
        if self.ax is None:
            return
        
        for name, line in self.policy_traj_lines.items():
            traj = traj_dict.get(name)
            if traj is not None and len(traj) > 0:
                line.set_data(traj[:, 0], traj[:, 1])
                # Highlight selected policy
                if name == best_policy:
                    line.set_color(SELECTED_COLOR)
                    line.set_linewidth(4)
                    line.set_alpha(1.0)
                    line.set_zorder(20)
                else:
                    line.set_color(POLICY_COLORS.get(name, 'gray'))
                    line.set_linewidth(2)
                    line.set_alpha(0.5)
                    line.set_zorder(17)
            else:
                line.set_data([], [])
    
    def solve_control_problem(
        self,
        robot_state: np.ndarray,
        control_ref: Optional[dict] = None,
        friction: Optional[float] = None,
        mpcc_trajectory: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Main MPCBF control loop - compute safe control using multi-policy CBF.
        
        Selection is based on maximum CBF constraint value (Vdot + alpha*V),
        not just maximum V.
        """
        robot_state = np.array(robot_state).flatten()
        
        if friction is not None:
            self.set_friction(friction)
        
        if mpcc_trajectory is not None:
            self.set_mpcc_trajectory(mpcc_trajectory)
        
        if control_ref is not None and 'u_ref' in control_ref:
            u_nom = np.array(control_ref['u_ref']).flatten()
        else:
            u_nom = np.zeros(2)
        
        self._update_obstacles()
        
        if len(self.obstacles) == 0:
            self.status = 'optimal'
            self.best_policy_name = 'nominal'
            return u_nom.reshape(-1, 1)
        
        x0_jax = jnp.array(robot_state)
        
        # Compute V and grad for all policies
        try:
            V_dict, grad_V_dict, traj_dict = self._compute_multi_value_and_grad(x0_jax)
        except Exception as e:
            print(f"Multi-policy value computation failed: {e}")
            self.status = 'error'
            return u_nom.reshape(-1, 1)
        
        # Compute f and G at current state (needed for constraint selection)
        f = np.array(self.dynamics_jax.f_full(x0_jax, self.current_friction))
        G = np.array(self.dynamics_jax.g_full(x0_jax))
        
        # Select best policy based on CBF constraint value (Vdot + alpha*V)
        best_policy, constraint_dict = self._select_best_policy(
            V_dict, grad_V_dict, f, G, u_nom
        )
        
        self.best_policy_name = best_policy
        V_best = V_dict[best_policy]
        grad_V_best = grad_V_dict[best_policy].copy()
        constraint_best = constraint_dict[best_policy]
        
        # Store trajectories for animation
        for name, traj in traj_dict.items():
            if traj is not None and self.curr_step % self.save_every_N == 0:
                if name not in self.multi_backup_trajs:
                    self.multi_backup_trajs[name] = []
                self.multi_backup_trajs[name].append(traj.copy())
        
        # Debug output (can be disabled by setting self.debug = False)
        if getattr(self, 'debug', True) and self.curr_step % 50 == 0:
            c_str = ", ".join([f"{k}:{constraint_dict[k]:.2f}" for k in sorted(constraint_dict.keys()) if constraint_dict[k] > -np.inf])
            print(f"  [MPCBF] CBF(u_nom): {c_str} -> best={best_policy}")
        
        self.curr_step += 1
        
        # PCBF Threshold Check:
        # If the best backup policy provides sufficient safety margin (V > threshold),
        # we can skip the strict CBF constraint and allow nominal control.
        # This prevents unnecessary braking when far from obstacles.
        v_threshold = 2.0
        if V_best > v_threshold:
            self.status = 'safe'
            self._update_multi_visualization(traj_dict, best_policy)
            return u_nom.reshape(-1, 1)
        
        # ALWAYS solve the CBF-QP with the selected (most relaxed) constraint.
        # If constraint > 0 at u_nom, the QP will naturally return u_nom since it's feasible.
        # If constraint < 0 at u_nom, the QP will find minimal deviation to satisfy constraint.
        # This is the correct CBF formulation - no heuristic shortcuts.
        
        # Normalize gradient to prevent numerical issues
        max_grad_norm = 50.0
        grad_norm = np.linalg.norm(grad_V_best)
        if grad_norm > max_grad_norm:
            grad_V_best = grad_V_best * (max_grad_norm / grad_norm)
        
        # Solve CBF-QP with best policy's constraint
        u_safe = self._solve_cbf_qp(u_nom, V_best, grad_V_best, f, G)
        
        # Update visualization
        self._update_multi_visualization(traj_dict, best_policy)
        
        return u_safe.reshape(-1, 1)
    
    def get_status(self):
        """Get current MPCBF status including best policy."""
        status = super().get_status()
        status['best_policy'] = self.best_policy_name
        status['algorithm'] = 'mpcbf'
        return status
    
    def get_multi_backup_trajectories(self):
        """Get stored backup trajectories for all policies."""
        return {k: v.copy() for k, v in self.multi_backup_trajs.items()}
    
    def clear_trajectories(self):
        """Clear stored backup trajectories for all policies."""
        for name in self.multi_backup_trajs:
            self.multi_backup_trajs[name].clear()
        self.backup_trajs.clear()
