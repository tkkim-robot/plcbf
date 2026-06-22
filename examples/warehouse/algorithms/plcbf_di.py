"""
Created on February 4th, 2026
@author: Taekyung Kim

@description:
PLCBF for Double Integrator (PLCBF_DI).
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import jax
import jax.numpy as jnp
import cvxpy as cp

from .pcbf_di import PCBF_DI, _compute_value_pure_di
from examples.warehouse.controllers.policies_di_jax import (
    AnglePolicyJAX, StopPolicyJAX, WaypointPolicyJAX,
    AnglePolicyParams, StopPolicyParams, WaypointPolicyParams
)
from examples.warehouse.dynamics.dynamics_di_jax import DIDynamicsParams

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

class PLCBF_DI(PCBF_DI):
    """
    PLCBF for Double Integrator.
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
        line_width_scale: float = 1.0,
        ax=None
    ):
        self.num_angle_policies = num_angle_policies
        self.max_operator = max_operator
        if self.max_operator not in MAX_OPERATOR_TYPES:
            raise ValueError(f"max_operator must be one of {MAX_OPERATOR_TYPES}")
            
        super().__init__(robot_spec, dt, backup_horizon, cbf_alpha, safety_margin, ax=None) # Handle ax manually
        self.ax = ax
        self.eval_horizon_steps = int(self.backup_horizon / self.dt)
        self.line_width_scale = max(1e-6, float(line_width_scale))
        
        # Policy Helpers
        self.policy_configs = {} # name -> (type, params)
        self.angle_names = []
        self.angle_params_batch = None
        self._setup_policies()
        
        # Cache for nominal trajectory
        self.nominal_trajectory = None
        
        # Visualization handles
        self.policy_lines = {}
        self._last_results = None
        self._last_best_name = None
        if self.ax is not None:
            self._setup_visualization()
            
        # JIT function for multi-evaluation
        # We can reuse the single evaluation function in a loop or vmap if parameters allow.
        # Since params differ, loop is safer/easier.
        self._jit_val_grad_fn = None
        self._jit_val_grad_obs = None
        self._jit_angle_val_grad = None
        self._jit_angle_grad_obs = None
        self.curr_step = 0
        self.debug = False
            
    def _setup_policies(self):
        v_ref = float(self.robot_spec.get('v_ref', self.robot_spec.get('v_max', 5.0)))
        v_ref = min(v_ref, float(self.robot_spec.get('v_max', v_ref)))
        Kp_v_nom = float(self.robot_spec.get('nominal_Kp_v', 6.0))
        K_lat_nom = float(self.robot_spec.get('nominal_K_lat', 1.0))
        v_lat_max_nom = float(self.robot_spec.get('nominal_v_lat_max', self.robot_spec.get('v_ref', 4.0)))
        dist_threshold_nom = float(self.robot_spec.get('nominal_dist_threshold', 1.0))
        Kp_v_angle = float(self.robot_spec.get('angle_Kp_v', Kp_v_nom))
        Kp_v_stop = float(self.robot_spec.get('stop_Kp_v', 3.0))

        # 1. Angle Policies
        angle_params = []
        for i in range(self.num_angle_policies):
            angle = i * (2 * np.pi / self.num_angle_policies)
            name = f'angle_{i}'
            params = AnglePolicyParams(
                target_angle=angle,
                target_speed=v_ref,
                Kp_v=Kp_v_angle,
                a_max=self.dynamics_params.a_max
            )
            self.policy_configs[name] = ('angle', params)
            self.angle_names.append(name)
            angle_params.append(params)

        if angle_params:
            self.angle_params_batch = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs), *angle_params
            )
        
        # 2. Stop Policy (standard backup)
        self.policy_configs['stop'] = ('stop', StopPolicyParams(
            Kp_v=Kp_v_stop, a_max=self.dynamics_params.a_max, stop_threshold=0.05
        ))
            
        # 3. Nominal Policy (Default Params, will be updated per step)
        self.policy_configs['nominal'] = ('waypoint', WaypointPolicyParams(
            waypoints=jnp.zeros((1, 2)),
            v_max=float(self.robot_spec.get('v_max', v_ref)),
            Kp=Kp_v_nom,
            K_lat=K_lat_nom,
            v_lat_max=v_lat_max_nom,
            dist_threshold=dist_threshold_nom,
            a_max=self.dynamics_params.a_max, current_wp_idx=0
        ))
        
    def _setup_visualization(self):
        if self.ax is None:
            return
        if self.policy_lines:
            return

        import matplotlib.cm as cm
        cmap = cm.get_cmap('hsv', self.num_angle_policies + 1)
        lw_base = 1.0 * self.line_width_scale
        
        for i in range(self.num_angle_policies):
            name = f'angle_{i}'
            self.policy_lines[name], = self.ax.plot([], [], color=cmap(i), alpha=0.3, linewidth=lw_base)
            
        self.policy_lines['nominal'], = self.ax.plot([], [], color='k', linestyle='--', alpha=0.5, linewidth=lw_base)

    def _setup_multi_visualization(self):
        self._setup_visualization()

    def update_visualization(self):
        if self.ax is None or self._last_results is None:
            return
        for name, (V, g, traj) in self._last_results.items():
            if name in self.policy_lines:
                self.policy_lines[name].set_data(traj[:, 0], traj[:, 1])
                if name == self._last_best_name:
                    self.policy_lines[name].set_linewidth(3.0 * self.line_width_scale)
                    self.policy_lines[name].set_alpha(1.0)
                else:
                    self.policy_lines[name].set_linewidth(1.0 * self.line_width_scale)
                    self.policy_lines[name].set_alpha(0.3)
        
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

        if self.static_obstacles:
            stat_obs_array_eval = jnp.array([
                (o['x'], o['y'], o['radius']) for o in self.static_obstacles
            ])
        else:
            stat_obs_array_eval = jnp.zeros((0, 3))

        robot_radius = self.robot_spec.get('radius', 1.0) + self.safety_margin
        robot_radius_base = self.robot_spec.get('radius', 1.0)
        
        # JIT function getter
        val_grad_fn, traj_fn = self._get_jit_val_grad()
        grad_obs_fn = self._get_jit_val_grad_obs()
        
        state_jax = jnp.array(state)
        
        policy_params_used = {}
        time_derivatives = {}

        # Batch evaluate angle policies first; this is the expensive PLCBF part.
        if self.angle_params_batch is not None and len(self.angle_names) > 0:
            batch_val_grad_fn, batch_grad_obs_fn = self._get_jit_angle_batch()
            (V_batch, traj_batch), grad_batch = batch_val_grad_fn(
                state_jax, self.dynamics_params, self.angle_params_batch,
                obs_array, stat_obs_array_eval, robot_radius, robot_radius_base
            )
            if obs_array.shape[0] > 0:
                grad_obs_batch = batch_grad_obs_fn(
                    state_jax, self.dynamics_params, self.angle_params_batch,
                    obs_array, stat_obs_array_eval, robot_radius, robot_radius_base
                )
                obs_vel = obs_array[:, 3:5]
                time_deriv_batch = jnp.sum(grad_obs_batch[:, :, 0:2] * obs_vel[None, :, :], axis=(1, 2))
            else:
                time_deriv_batch = jnp.zeros((len(self.angle_names),))

            for i, name in enumerate(self.angle_names):
                params = self.policy_configs[name][1]
                policy_params_used[name] = ('angle', params)
                results[name] = (
                    float(V_batch[i]),
                    np.array(grad_batch[i]),
                    np.array(traj_batch[i])
                )
                time_derivatives[name] = float(time_deriv_batch[i])

        # Evaluate stop and nominal policies individually.
        for name in ['stop', 'nominal']:
            if name not in self.policy_configs:
                continue
            ptype, params = self.policy_configs[name]
            if name == 'nominal' and control_ref is not None and 'waypoints' in control_ref:
                Kp_v_nom = float(self.robot_spec.get('nominal_Kp_v', 6.0))
                K_lat_nom = float(self.robot_spec.get('nominal_K_lat', 1.0))
                v_lat_max_nom = float(self.robot_spec.get('nominal_v_lat_max', self.robot_spec.get('v_ref', 4.0)))
                dist_threshold_nom = float(self.robot_spec.get('nominal_dist_threshold', 1.0))
                params = WaypointPolicyParams(
                    waypoints=jnp.array(control_ref['waypoints']),
                    v_max=float(self.robot_spec.get('v_max', 5.0)),
                    Kp=Kp_v_nom,
                    K_lat=K_lat_nom,
                    v_lat_max=v_lat_max_nom,
                    dist_threshold=dist_threshold_nom,
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
             
             if best_name is None:
                 best_name, best_score, best_V = name, score, V
                 continue
                 
             if V > 0 and best_V > 0:
                 if score > best_score + 1e-6:
                     best_name, best_score, best_V = name, score, V
             elif V > 0 and best_V <= 0:
                  best_name, best_score, best_V = name, score, V
             elif V <= 0 and best_V <= 0:
                 if V > best_V + 1e-6:
                     best_name, best_score, best_V = name, score, V
                 elif abs(V - best_V) <= 1e-6 and score > best_score + 1e-6:
                     best_name, best_score, best_V = name, score, V
             
        if best_name is None:
            best_name = 'nominal'

        self._last_results = results
        self._last_best_name = best_name

        # 5. QP objective stays centered at nominal control
        u_qp_nom = u_nom
        
        # 6. Solve QP with Selected Constraint
        V_best, grad_best, _ = results[best_name]
        self._last_time_derivative = time_derivatives.get(best_name, 0.0)
        if self.debug and self.curr_step % 50 == 0:
            lg_norm = float(np.linalg.norm(grad_best[2:4]))
            g_norm = float(np.linalg.norm(grad_best))
            print(f"[PLCBF_DI] step={self.curr_step} best={best_name} V={V_best:.3f} time_dV={self._last_time_derivative:.3f} |g|={g_norm:.3f} |Lg|={lg_norm:.3f}")

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
        
        # Safety Constraints (standardized via PLCBF_DI override)
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
            # Match Quad3D fallback: execute the selected safest rollout policy.
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
                
        u_safe = np.clip(np.array(res, dtype=float).flatten(), -u_scale, u_scale)

        self.curr_step += 1
        return u_safe

    def _add_cbf_constraints(
        self,
        u,
        constraints,
        state,
        V,
        grad_V,
        slack=0.0,
        include_dynamic=True,
        include_static=True,
    ):
        """
        Add CBF constraints for PLCBF_DI.
        
        Add HOCBF constarint for static obstacles (assumeing the robot knows the static obs)
        Add PLCBF constraint for dynamic obstacles (robot doesn't know the dynamic obs in advanced)
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

    def _get_jit_angle_batch(self):
        """Get batched JIT functions for angle-policy values and obstacle gradients."""
        if self._jit_angle_val_grad is None:
            h = self.eval_horizon_steps
            dt_val = self.dt

            def val_fn(x0, dyn_p, pol_p, dyn_obs, stat_obs, r_rad, rr_base):
                V, traj = _compute_value_pure_di(
                    x0, dyn_p, pol_p, dyn_obs, stat_obs, 'angle', h, r_rad, rr_base, dt_val
                )
                return V, traj

            val_grad = jax.value_and_grad(val_fn, has_aux=True)
            self._jit_angle_val_grad = jax.jit(
                jax.vmap(val_grad, in_axes=(None, None, 0, None, None, None, None))
            )

            def val_only(x0, dyn_p, pol_p, dyn_obs, stat_obs, r_rad, rr_base):
                V, _ = _compute_value_pure_di(
                    x0, dyn_p, pol_p, dyn_obs, stat_obs, 'angle', h, r_rad, rr_base, dt_val
                )
                return V

            grad_obs = jax.grad(val_only, argnums=3)
            self._jit_angle_grad_obs = jax.jit(
                jax.vmap(grad_obs, in_axes=(None, None, 0, None, None, None, None))
            )

        return self._jit_angle_val_grad, self._jit_angle_grad_obs

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
