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
        # Use fixed 2.0s horizon for MPCBF evaluation of dynamic obstacles
        self.eval_horizon_steps = int(2.0 / self.dt)
        
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
        self._jit_val_grad_obs = None
        self.curr_step = 0
        self.debug = False
            
    def _setup_policies(self):
        # 1. Angle Policies
        for i in range(self.num_angle_policies):
            angle = i * (2 * np.pi / self.num_angle_policies)
            name = f'angle_{i}'
            # Use v_max as target speed? Or v_ref?
            v_ref = float(self.robot_spec.get('v_ref', 5.0))
            v_ref = min(v_ref, 3.8)
            self.policy_configs[name] = ('angle', AnglePolicyParams(
                target_angle=angle, target_speed=v_ref, Kp_v=15.0, a_max=self.dynamics_params.a_max
            ))
        
        # 2. Stop Policy (standard backup)
        self.policy_configs['stop'] = ('stop', StopPolicyParams(
            Kp_v=4.0, a_max=self.dynamics_params.a_max, stop_threshold=0.05
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
            obs_array = jnp.array([
                (o['x'], o['y'], o['radius'], o.get('vx', 0.0), o.get('vy', 0.0))
                for o in self.dynamic_obstacles
            ])
        else:
            obs_array = jnp.zeros((0, 5))

        # For MPCBF, use V only for dynamic obstacles; static handled by HO-CBF
        stat_obs_array_eval = jnp.zeros((0, 3))
        
            
        robot_radius = self.robot_spec.get('radius', 1.0) + self.safety_margin
        robot_radius_base = self.robot_spec.get('radius', 1.0)
        
        # JIT function getter
        val_grad_fn, traj_fn = self._get_jit_val_grad()
        grad_obs_fn = self._get_jit_val_grad_obs()
        
        state_jax = jnp.array(state)
        
        # Evaluate Defined Policies
        policy_params_used = {}
        time_derivatives = {}
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
            policy_params_used[name] = (ptype, params)
            
            V_jax, grad_jax = val_grad_fn(
                state_jax, self.dynamics_params, params, obs_array, stat_obs_array_eval, ptype,
                self.eval_horizon_steps, robot_radius, robot_radius_base, self.dt
            )
            traj = traj_fn(
                state_jax, self.dynamics_params, params, obs_array, stat_obs_array_eval, ptype,
                self.eval_horizon_steps, robot_radius, robot_radius_base, self.dt
            )
            results[name] = (float(V_jax), np.array(grad_jax), np.array(traj))
            
            # Time-derivative of V from obstacle motion: dV/dt = dV/dobs · v_obs
            # Use obstacle velocities in the value derivative for proper dynamic CBF.
            if obs_array.shape[0] > 0:
                grad_obs = grad_obs_fn(
                    state_jax, self.dynamics_params, params, obs_array, stat_obs_array_eval, ptype,
                    self.eval_horizon_steps, robot_radius, robot_radius_base, self.dt
                )
                obs_vel = obs_array[:, 3:5]
                time_derivatives[name] = float(jnp.sum(grad_obs[:, 0:2] * obs_vel))
            else:
                time_derivatives[name] = 0.0
             
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
        
        max_grad_norm = 150.0
        min_lg_norm = 1e-3
        
        for name, (V, grad_V, traj) in results.items():
             # Normalize gradient to avoid extreme constraints/selection
             grad_norm = np.linalg.norm(grad_V)
             if grad_norm > max_grad_norm:
                 grad_V = grad_V * (max_grad_norm / grad_norm)
                 results[name] = (V, grad_V, traj)
             
             # Lie Derivatives
             Lf_V = np.dot(grad_V, f)
             Lg_V = grad_V[2:4] # grad_V @ G
             
             # score based on operator
             if self.max_operator == 'v':
                 score = V
             elif self.max_operator == 'input_space':
                 # Compute area of feasible control set
                 # If control influence is near-zero, treat as non-informative
                 if np.linalg.norm(Lg_V) < min_lg_norm:
                     score = V
                 else:
                     cbf_rhs = Lf_V + time_derivatives.get(name, 0.0) + self.cbf_alpha * V
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
                 elif abs(score - best_score) <= 1e-3:
                     if V > best_V + 1e-3:
                         best_name, best_score, best_V, best_alignment = name, score, V, alignment
                     elif abs(V - best_V) <= 1e-3 and alignment > best_alignment:
                         best_name, best_score, best_V, best_alignment = name, score, V, alignment
             elif V > 0 and best_V <= 0:
                  best_name, best_score, best_V, best_alignment = name, score, V, alignment
             elif V <= 0 and best_V <= 0:
                 if score > best_score + 1e-3:
                     best_name, best_score, best_V, best_alignment = name, score, V, alignment
             
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

        # 5. QP objective stays centered at nominal control
        u_qp_nom = u_nom
        
        # 6. Solve QP with Selected Constraint
        V_best, grad_best, _ = results[best_name]
        self._last_time_derivative = time_derivatives.get(best_name, 0.0)
        if self.debug and self.curr_step % 50 == 0:
            lg_norm = float(np.linalg.norm(grad_best[2:4]))
            g_norm = float(np.linalg.norm(grad_best))
            print(f"[MPCBF_DI] step={self.curr_step} best={best_name} V={V_best:.3f} time_dV={self._last_time_derivative:.3f} |g|={g_norm:.3f} |Lg|={lg_norm:.3f}")

        # (Optional) Nominal CBF satisfaction can be computed here for analysis if needed
        f = np.array([state[2], state[3], 0, 0])
        Lf_V = np.dot(grad_best, f)
        Lg_V = grad_best[2:4]
        time_term = self._last_time_derivative
        
        # QP Formulation
        # Scaled QP for numerical stability
        u_scale = np.array([self.dynamics_params.ax_max, self.dynamics_params.ay_max])
        u_nom_scaled = u_qp_nom / u_scale
        u_scaled = cp.Variable(2)
        u = cp.multiply(u_scaled, u_scale)
        cost = cp.sum_squares(u_scaled - u_nom_scaled)
        constraints = [
            u_scaled[0] <= 1.0,
            u_scaled[0] >= -1.0,
            u_scaled[1] <= 1.0,
            u_scaled[1] >= -1.0
        ]
        
        # Safety Constraints (standardized via MPCBF_DI override)
        self._add_cbf_constraints(u, constraints, state, V_best, grad_best)
            
        prob = cp.Problem(cp.Minimize(cost), constraints)
        
        # Fallback Logic
        use_fallback = False
        res = u_qp_nom
        
        try:
            prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=20000)
            if prob.status in ['optimal', 'optimal_inaccurate']:
                res = u_scaled.value * u_scale
                # Enforce CBF constraint numerically; fallback if violated
                if res is None:
                    use_fallback = True
                else:
                    cbf_val = Lg_V @ res + Lf_V + time_term + self.cbf_alpha * V_best
                    if cbf_val < -1e-4:
                        use_fallback = True
            else:
                use_fallback = True
        except Exception:
            use_fallback = True
             
        if use_fallback:
            # Try static-only safety QP before policy fallback
            try:
                u_scaled2 = cp.Variable(2)
                u2 = cp.multiply(u_scaled2, u_scale)
                cost2 = cp.sum_squares(u_scaled2 - u_nom_scaled)
                constraints2 = [
                    u_scaled2[0] <= 1.0,
                    u_scaled2[0] >= -1.0,
                    u_scaled2[1] <= 1.0,
                    u_scaled2[1] >= -1.0
                ]
                self._add_cbf_constraints(u2, constraints2, state, V_best, grad_best, include_dynamic=False)
                prob2 = cp.Problem(cp.Minimize(cost2), constraints2)
                prob2.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=20000)
                if prob2.status in ['optimal', 'optimal_inaccurate'] and u_scaled2.value is not None:
                    res = u_scaled2.value * u_scale
                    use_fallback = False
            except Exception:
                pass

        if use_fallback:
            # Fallback to Best Policy Control
            if best_name in policy_params_used:
                ptype, pparams = policy_params_used[best_name]
                if ptype == 'angle':
                    res = np.array(AnglePolicyJAX.compute(jnp.array(state), pparams))
                elif ptype == 'stop':
                    res = np.array(StopPolicyJAX.compute(jnp.array(state), pparams))
                elif ptype == 'waypoint':
                    res = np.array(WaypointPolicyJAX.compute(jnp.array(state), pparams))
                else:
                    res = u_qp_nom
            else:
                res = u_qp_nom
                
        u_safe = res

        self.curr_step += 1
        return u_safe

    def _add_cbf_constraints(self, u, constraints, state, V, grad_V, slack=0.0, include_dynamic=True, include_static=True):
        """
        Add CBF constraints for MPCBF_DI.
        
        Add HOCBF constarint for static obstacles (assumeing the robot knows the static obs)
        Add MPCBF constraint for dynamic obstacles (robot doesn't know the dynamic obs in advanced)
        """
        f, _ = self._get_system_matrices(state)
        L_f_V = np.dot(grad_V, f)
        L_g_V = grad_V[2:4]  # Only velocity components affect control
        
        # Value-function CBF (dynamic obstacles + rollout)
        if include_dynamic:
            time_term = getattr(self, "_last_time_derivative", 0.0)
            constraints.append(L_g_V @ u >= -self.cbf_alpha * V - L_f_V - time_term - slack)
        
        # Static obstacles: HO-CBF (second-order)
        if include_static:
            gamma1, gamma2 = 3.0, 3.0
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

    def _get_jit_val_grad_obs(self):
        """Get or create JIT-compiled gradient function w.r.t. dynamic obstacles."""
        if self._jit_val_grad_obs is None:
            def val_fn(x0, dyn_p, pol_p, dyn_obs, stat_obs, p_type, hor, r_rad, rr_base, dt_val):
                V, _ = _compute_value_pure_di(x0, dyn_p, pol_p, dyn_obs, stat_obs, p_type, hor, r_rad, rr_base, dt_val)
                return V
            
            self._jit_val_grad_obs = jax.jit(
                jax.grad(val_fn, argnums=3),
                static_argnums=(5, 6)  # policy_type, horizon
            )
        
        return self._jit_val_grad_obs
