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

# =============================================================================
# JIT-compiled Feasible Control Area Computation (for input_space operator)
# =============================================================================

@jax.jit
def _compute_feasible_area_jit(
    grad_V_G: jnp.ndarray,      # (2,) - gradient @ G
    cbf_rhs: float,             # scalar - ∇V·f + α·V
    u_min: jnp.ndarray,         # (2,) - control lower bounds
    u_max: jnp.ndarray,         # (2,) - control upper bounds
) -> float:
    """
    Compute area of feasible control polygon (box ∩ half-space).
    """
    # Box vertices (counter-clockwise)
    box_vertices = jnp.array([
        [u_min[0], u_min[1]],
        [u_max[0], u_min[1]],
        [u_max[0], u_max[1]],
        [u_min[0], u_max[1]],
    ])  # (4, 2)
    
    a = grad_V_G  # (2,)
    b = -cbf_rhs  # scalar
    
    grad_norm = jnp.linalg.norm(a)
    
    def full_box_area():
        return (u_max[0] - u_min[0]) * (u_max[1] - u_min[1])
    
    def clip_polygon():
        inside = jax.vmap(lambda v: jnp.dot(a, v) >= b)(box_vertices)  # (4,)
        all_inside = jnp.all(inside)
        all_outside = jnp.all(~inside)
        
        n_verts = 4
        def edge_intersect(i):
            p1 = box_vertices[i]
            p2 = box_vertices[(i + 1) % n_verts]
            d = p2 - p1
            denom = jnp.dot(a, d)
            t = jnp.where(
                jnp.abs(denom) > 1e-10,
                (b - jnp.dot(a, p1)) / denom,
                0.5
            )
            t = jnp.clip(t, 0.0, 1.0)
            return p1 + t * d
        
        intersections = jax.vmap(edge_intersect)(jnp.arange(n_verts))  # (4, 2)
        
        def process_edge(i):
            p1_in = inside[i]
            p2_in = inside[(i + 1) % n_verts]
            p2 = box_vertices[(i + 1) % n_verts]
            inter = intersections[i]
            
            v1 = jnp.where(p1_in & ~p2_in, inter, p2)
            v2 = jnp.where(~p1_in & p2_in, p2, jnp.zeros(2))
            
            count = jnp.where(
                p1_in & p2_in, 1,
                jnp.where(
                    p1_in & ~p2_in, 1,
                    jnp.where(~p1_in & p2_in, 2, 0)
                )
            )
            return v1, v2, count
        
        v1s, v2s, counts = jax.vmap(process_edge)(jnp.arange(n_verts))
        
        all_verts = jnp.concatenate([v1s, v2s], axis=0)  # (8, 2)
        valid_mask = jnp.concatenate([counts >= 1, counts >= 2])  # (8,)
        
        centroid = jnp.sum(all_verts * valid_mask[:, None], axis=0) / jnp.maximum(jnp.sum(valid_mask), 1.0)
        angles = jnp.arctan2(all_verts[:, 1] - centroid[1], all_verts[:, 0] - centroid[0])
        angles = jnp.where(valid_mask, angles, 100.0)
        sorted_indices = jnp.argsort(angles)
        sorted_verts = all_verts[sorted_indices]
        sorted_valid = valid_mask[sorted_indices]
        
        def shoelace_term(i):
            j = (i + 1) % 8
            term = sorted_verts[i, 0] * sorted_verts[j, 1] - sorted_verts[j, 0] * sorted_verts[i, 1]
            return jnp.where(sorted_valid[i] & sorted_valid[j], term, 0.0)
        
        area = 0.5 * jnp.abs(jnp.sum(jax.vmap(shoelace_term)(jnp.arange(8))))
        area = jnp.where(all_inside, full_box_area(), area)
        area = jnp.where(all_outside, 0.0, area)
        return area
    
    return jax.lax.cond(grad_norm < 1e-8, full_box_area, clip_polygon)

MAX_OPERATOR_TYPES = ['v', 'input_space']

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
        max_operator: str = 'input_space',
        ax=None
    ):
        self.num_angle_policies = num_angle_policies
        self.max_operator = max_operator
        if self.max_operator not in MAX_OPERATOR_TYPES:
            raise ValueError(f"max_operator must be one of {MAX_OPERATOR_TYPES}")
            
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
        # 1. Angle Policies
        for i in range(self.num_angle_policies):
            angle = i * (2 * np.pi / self.num_angle_policies)
            name = f'angle_{i}'
            # Use v_max as target speed? Or v_ref?
            v_ref = float(self.robot_spec.get('v_ref', 5.0))
            self.policy_configs[name] = ('angle', AnglePolicyParams(
                target_angle=angle, target_speed=v_ref, Kp_v=15.0, a_max=self.dynamics_params.a_max
            ))
            
        # 3. Nominal Policy (Default Params, will be updated per step)
        self.policy_configs['nominal'] = ('waypoint', WaypointPolicyParams(
            waypoints=jnp.zeros((1, 2)), v_max=v_ref, Kp=15.0, dist_threshold=1.0, 
            a_max=self.dynamics_params.a_max, current_wp_idx=0
        ))
        
    def _setup_visualization(self):
        if self.ax is None:
            return
            
        # Create lines for all policies
        import matplotlib.cm as cm
        cmap = cm.get_cmap('hsv', self.num_angle_policies + 1)
        
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
                    Kp=15.0, # Sync with user tuning
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
             
        # 3. Select Best Policy
        best_name = None
        best_score = -np.inf
        best_V = -np.inf
        best_alignment = -1.0
        
        # Control bounds for area calculation
        u_min = jnp.array([-self.dynamics_params.ax_max, -self.dynamics_params.ay_max])
        u_max = jnp.array([self.dynamics_params.ax_max, self.dynamics_params.ay_max])
        
        f = np.array([state[2], state[3], 0, 0])
        # G_flat = np.array([0, 0, 1, 0, 0, 0, 0, 1]).reshape(4, 2) # Double Integrator G
        
        for name, (V, grad_V, traj) in results.items():
             # Lie Derivatives
             Lf_V = np.dot(grad_V, f)
             Lg_V = grad_V[2:4] # grad_V @ G
             
             # score based on operator
             if self.max_operator == 'v':
                 score = V
             elif self.max_operator == 'input_space':
                 # Compute area of feasible control set
                 cbf_rhs = Lf_V + self.cbf_alpha * V
                 score = float(_compute_feasible_area_jit(
                     jnp.array(Lg_V), cbf_rhs, u_min, u_max
                 ))
             
             # Alignment for tie-breaking
             alignment = -1.0
             if "angle" in name and np.linalg.norm(u_nom) > 0.1:
                 try:
                     idx = int(name.split('_')[1])
                     policy_angle = idx * (2 * np.pi / self.num_angle_policies)
                     nom_angle = np.arctan2(u_nom[1], u_nom[0])
                     alignment = np.cos(policy_angle - nom_angle)
                 except:
                     pass
             
             if best_name is None:
                 best_name, best_score, best_V, best_alignment = name, score, V, alignment
                 continue
                 
             # Prioritize safety: Among safe policies (V > 0), pick max score.
             # If both safe or both unsafe, pick max score.
             # Actually, if any are safe, we ONLY look at safe ones.
             if V > 0 and best_V > 0:
                 if score > best_score + 1e-3:
                     best_name, best_score, best_V, best_alignment = name, score, V, alignment
                 elif abs(score - best_score) <= 1e-3 and alignment > best_alignment:
                     best_name, best_score, best_V, best_alignment = name, score, V, alignment
             elif V > 0 and best_V <= 0:
                  best_name, best_score, best_V, best_alignment = name, score, V, alignment
             elif V <= 0 and best_V <= 0:
                 if score > best_score + 1e-3:
                     best_name, best_score, best_V, best_alignment = name, score, V, alignment
             
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
        # Heavy penalty on slack to enforce safety
        cost = cp.sum_squares(u - u_nom) + 1e6 * cp.square(slack)
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
                # elif ptype == 'stop':
                #     res = np.array(StopPolicyJAX.compute(jnp.array(state), pparams))
                elif ptype == 'waypoint':
                    res = np.array(WaypointPolicyJAX.compute(jnp.array(state), pparams))
                else:
                    res = u_nom
            else:
                res = u_nom
                
        u_safe = res

        return u_safe
