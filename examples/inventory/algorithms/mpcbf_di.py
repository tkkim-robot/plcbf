"""
Created on February 4th, 2026
@author: Taekyung Kim

@description:
MPCBF for Double Integrator (MPCBF_DI).
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import jax
import jax.numpy as jnp
import cvxpy as cp

from .pcbf_di import PCBF_DI, _compute_value_pure_di
from examples.inventory.controllers.policies_di_jax import (
    AnglePolicyJAX, StopPolicyJAX, WaypointPolicyJAX,
    AnglePolicyParams, StopPolicyParams, WaypointPolicyParams
)
from examples.inventory.dynamics.dynamics_di_jax import DIDynamicsParams

class MPCBF_DI(PCBF_DI):
    """
    MPCBF for Double Integrator.
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
        ax=None
    ):
        self.num_angle_policies = num_angle_policies
        super().__init__(robot_spec, dt, backup_horizon, cbf_alpha, safety_margin, ax=None) # Handle ax manually
        self.ax = ax
        
        # Policy Helpers
        self.policy_configs = {} # name -> (type, params)
        self._setup_policies()
        
        # Cache for nominal trajectory
        self.nominal_trajectory = None
        
        # Visualization handles
        self.policy_lines = {}
        if self.ax is not None:
            self._setup_visualization()
            
        # JIT function for multi-evaluation
        # We can reuse the single evaluation function in a loop or vmap if parameters allow.
        # Since params differ, loop is safer/easier.
        self._jit_val_grad_fn = None
            
    def _setup_policies(self):
        # 1. Stop Policy
        self.policy_configs['stop'] = ('stop', StopPolicyParams(
            Kp_v=4.0, a_max=self.dynamics_params.a_max, stop_threshold=0.05
        ))
        
        # 2. Angle Policies
        for i in range(self.num_angle_policies):
            angle = i * (2 * np.pi / self.num_angle_policies)
            name = f'angle_{i}'
            # Use v_max as target speed? Or v_ref?
            v_ref = float(self.robot_spec.get('v_ref', 5.0))
            self.policy_configs[name] = ('angle', AnglePolicyParams(
                target_angle=angle, target_speed=v_ref, Kp_v=4.0, a_max=self.dynamics_params.a_max
            ))
            
        # 3. Nominal Policy (Default Params, will be updated per step)
        self.policy_configs['nominal'] = ('waypoint', WaypointPolicyParams(
            waypoints=jnp.zeros((1, 2)), v_max=v_ref, Kp=4.0, dist_threshold=1.0, 
            a_max=self.dynamics_params.a_max, current_wp_idx=0
        ))
        
    def _setup_visualization(self):
        if self.ax is None:
            return
            
        # Create lines for all policies
        colors = ['#FF0000'] # Stop is Red
        # Angles get rainbow
        import matplotlib.cm as cm
        cmap = cm.get_cmap('hsv', self.num_angle_policies + 1)
        for i in range(self.num_angle_policies):
            colors.append(cmap(i))
        colors.append('#000000') # Nominal is Black
        
        self.policy_lines['stop'], = self.ax.plot([], [], color='red', alpha=0.3, linewidth=1)
        
        for i in range(self.num_angle_policies):
            name = f'angle_{i}'
            self.policy_lines[name], = self.ax.plot([], [], color=cmap(i), alpha=0.3, linewidth=1)
            
        self.policy_lines['nominal'], = self.ax.plot([], [], color='k', linestyle='--', alpha=0.5, linewidth=1)
        
    def set_nominal_traj(self, traj):
        self.nominal_trajectory = traj
        
    def solve_control_problem(self, state, control_ref=None):
        # 1. Update Nominal
        if control_ref and 'u_ref' in control_ref:
            u_nom = np.array(control_ref['u_ref']).flatten()
        else:
            u_nom = np.zeros(2)
            
        # 2. Evaluate All Policies
        results = {} # name -> (V, grad_V, traj)
        
        # Dynamic obstacles array
        if self.dynamic_obstacles:
            obs_array = jnp.array([(o['x'], o['y'], o['radius'], o.get('vx', 0.0), o.get('vy', 0.0)) for o in self.dynamic_obstacles])
        else:
            obs_array = jnp.zeros((0, 5))
            
        if self.static_obstacles:
            stat_obs_array = jnp.array([(o['x'], o['y'], o['radius']) for o in self.static_obstacles])
        else:
            stat_obs_array = jnp.zeros((0, 3))
            
        robot_radius = self.robot_spec.get('radius', 1.0) + self.safety_margin
        robot_radius_base = self.robot_spec.get('radius', 1.0)
        
        # JIT function getter
        val_grad_fn, traj_fn = self._get_jit_val_grad()
        
        state_jax = jnp.array(state)
        
        # Evaluate Defined Policies
        for name, (ptype, params) in self.policy_configs.items():
            # Update Nominal Params dynamically
            if name == 'nominal' and control_ref is not None and 'waypoints' in control_ref:
                params = WaypointPolicyParams(
                    waypoints=jnp.array(control_ref['waypoints']),
                    v_max=float(self.robot_spec.get('v_max', 8.0)),
                    Kp=4.0,
                    dist_threshold=1.0,
                    a_max=self.dynamics_params.a_max,
                    current_wp_idx=control_ref.get('wp_idx', 0)
                )
            
            V_jax, grad_jax = val_grad_fn(
                state_jax, self.dynamics_params, params, obs_array, stat_obs_array, ptype,
                self.backup_horizon_steps, robot_radius, robot_radius_base, self.dt
            )
            traj = traj_fn(
                state_jax, self.dynamics_params, params, obs_array, stat_obs_array, ptype,
                self.backup_horizon_steps, robot_radius, robot_radius_base, self.dt
            )
            results[name] = (float(V_jax), np.array(grad_jax), np.array(traj))
            
        # Nominal policy is now evaluated inside the loop above as 'nominal'
             
        # 3. Select Best Policy
        # Metric: Score = V_dot + alpha * V (Constraint Value)
        # We want the policy that gives us the most flexible constraint.
        # Constraint: grad_V @ (f + g u) + alpha V >= 0
        # Check value at u_nom: score = grad_V @ (f + g u_nom) + alpha V
        
        best_name = None
        best_score = -np.inf
        best_V = -np.inf
        best_alignment = -1.0
        
        scores = {}
        
        f = np.array([state[2], state[3], 0, 0])
        
        for name, (V, grad_V, traj) in results.items():
             # Basic score: V (Safety Margin)
             # Sophisticated: Feasibility at u_nom
             
             # Calculate Lie Derivatives
             # f_dyn = [vx, vy, 0, 0]
             # g_dyn u = [0, 0, ax, ay]
             
             L_f_V = np.dot(grad_V, f)
             # L_g_V = [grad_vx, grad_vy]
             L_g_V = grad_V[2:4]
             
             # Constraint value at u_nom
             # c(u) = L_g_V @ u + L_f_V + alpha * V
             val_at_unom = np.dot(L_g_V, u_nom) + L_f_V + self.cbf_alpha * V
             
             # calculate alignment for tie-breaking
             alignment = -1.0
             if "angle" in name and np.linalg.norm(u_nom) > 0.1:
                 try:
                     idx = int(name.split('_')[1])
                     policy_angle = idx * (2 * np.pi / self.num_angle_policies)
                     nom_angle = np.arctan2(u_nom[1], u_nom[0])
                     alignment = np.cos(policy_angle - nom_angle)
                 except:
                     pass
             
             # Selection Logic
             # 1. Initialization
             if best_name is None:
                 best_name = name
                 best_score = val_at_unom
                 best_V = V
                 best_alignment = alignment
                 continue
                 
             # 2. Comparison
             # Case A: Both Unsafe (V < 0) -> Maximize V (Safety)
             if V < 0 and best_V < 0:
                 if V > best_V:
                     best_name = name
                     best_score = val_at_unom
                     best_V = V
                     best_alignment = alignment
             
             # Case B: Current Safe, Best Unsafe -> Pick Current
             elif V > 0 and best_V < 0:
                 best_name = name
                 best_score = val_at_unom
                 best_V = V
                 best_alignment = alignment
                 
             # Case C: Both Safe -> Maximize Slack, Tie-break with Alignment
             elif V > 0 and best_V > 0:
                 # Significant improvement in slack?
                 if val_at_unom > best_score + 1e-3:
                     best_name = name
                     best_score = val_at_unom
                     best_V = V
                     best_alignment = alignment
                 # Tie (slack is similar) -> Use Alignment
                 elif abs(val_at_unom - best_score) < 1e-3:
                     if alignment > best_alignment:
                         best_name = name
                         best_score = val_at_unom
                         best_V = V
                         best_alignment = alignment
             
        # Tie-breaking for Safe Scenario (V > 50)
        # If best_V is high (safe), we might have picked arbitrary policy due to 0 gradients.
        # DEBUG
        if np.linalg.norm(u_nom) > 1.0 and best_V > 50.0:
             # Check if we are stuck
             pass 
             
        # 4. Visualization Update
        if self.ax:
            for name, (V, g, traj) in results.items():
                if name in self.policy_lines:
                    self.policy_lines[name].set_data(traj[:,0], traj[:,1])
                    if name == best_name:
                        self.policy_lines[name].set_linewidth(3)
                        self.policy_lines[name].set_alpha(1.0)
                    else:
                        self.policy_lines[name].set_linewidth(1)
                        self.policy_lines[name].set_alpha(0.3)
                        
        # 5. Solve QP with Selected Constraint
        V_best, grad_best, _ = results[best_name]
        
        # QP Formulation
        u = cp.Variable(2)
        slack = cp.Variable(1, nonneg=True)
        cost = cp.sum_squares(u - u_nom) + 1e5 * cp.square(slack)
        constraints = []
        
        # Input Limits
        constraints += [
            u[0] <= self.dynamics_params.ax_max,
            u[0] >= -self.dynamics_params.ax_max,
            u[1] <= self.dynamics_params.ay_max,
            u[1] >= -self.dynamics_params.ay_max
        ]
        
        # Safety Constraints (standardized via PCBF_DI)
        self._add_cbf_constraints(u, constraints, state, V_best, grad_best, slack=slack[0])
            
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        # Fallback Logic
        use_fallback = False
        res = u_nom
        
        try:
            prob.solve(verbose=False)
            if prob.status in ['optimal', 'optimal_inaccurate']:
                res = u.value
            else:
                use_fallback = True
        except Exception:
            use_fallback = True
             
        if use_fallback:
            # Fallback to Best Policy Control
            if best_name in self.policy_configs:
                ptype, pparams = self.policy_configs[best_name]
                if ptype == 'angle':
                    res = np.array(AnglePolicyJAX.compute(jnp.array(state), pparams))
                elif ptype == 'stop':
                    res = np.array(StopPolicyJAX.compute(jnp.array(state), pparams))
                elif ptype == 'waypoint':
                    res = np.array(WaypointPolicyJAX.compute(jnp.array(state), pparams))
                else:
                    res = u_nom
            else:
                res = u_nom
                
        u_safe = res

        return u_safe
