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
- Select policy with max feasible space (by default) (most permissive constraint that's still safe)
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

from .pcbf_drift import (
    PCBF,
    smooth_min,
    _rollout_trajectory,
    _compute_value_pure,
    _compute_value_and_grad_jit,
    _compute_value_jit,
)
from examples.drift_car.dynamics.drift_dynamics_jax import (
    DriftingCarDynamicsJAX,
    DynamicsParams,
    step_full_state_pure,
)
from examples.drift_car.controllers.drift_policies_jax import (
    LaneChangeControllerJAX,
    StoppingControllerJAX,
    LaneChangePolicyParams,
    StopPolicyParams,
)


# =============================================================================
# Selection Operator Types
# =============================================================================

MAX_OPERATOR_TYPES = ['c', 'v', 'input_space']


# =============================================================================
# JIT-compiled Feasible Control Area Computation (for input_space operator)
# =============================================================================

@jax.jit
def _compute_feasible_area_jit(
    grad_V_G: jnp.ndarray,      # (2,) - gradient @ G
    cbf_rhs: float,              # scalar - ∇V·f + α·V
    u_min: jnp.ndarray,          # (2,) - control lower bounds
    u_max: jnp.ndarray,          # (2,) - control upper bounds
) -> float:
    """
    Compute area of feasible control polygon (box ∩ half-space).
    
    The feasible region is defined by:
    - Box: u_min ≤ u ≤ u_max
    - Half-space: grad_V_G @ u ≥ -cbf_rhs  (CBF constraint)
    
    Uses Sutherland-Hodgman clipping algorithm for polygon intersection.
    
    Args:
        grad_V_G: Gradient of V times G matrix (2,)
        cbf_rhs: Right-hand side of CBF constraint (∇V·f + α·V)
        u_min: Control lower bounds (2,)
        u_max: Control upper bounds (2,)
    
    Returns:
        area: Area of feasible polygon (0 if infeasible)
    """
    # Box vertices (counter-clockwise)
    # v0 = (u_min[0], u_min[1]), v1 = (u_max[0], u_min[1])
    # v2 = (u_max[0], u_max[1]), v3 = (u_min[0], u_max[1])
    box_vertices = jnp.array([
        [u_min[0], u_min[1]],
        [u_max[0], u_min[1]],
        [u_max[0], u_max[1]],
        [u_min[0], u_max[1]],
    ])  # (4, 2)
    
    # Half-space: a @ u >= b, where a = grad_V_G, b = -cbf_rhs
    a = grad_V_G  # (2,)
    b = -cbf_rhs  # scalar
    
    # Check if gradient is essentially zero (no constraint - full box is feasible)
    grad_norm = jnp.linalg.norm(a)
    
    def full_box_area():
        return (u_max[0] - u_min[0]) * (u_max[1] - u_min[1])
    
    def clip_polygon():
        # Sutherland-Hodgman clipping: clip box against half-space a @ u >= b
        # For each edge (p1 -> p2), compute intersection and keep inside points
        
        # Evaluate all vertices: inside if a @ v >= b
        inside = jax.vmap(lambda v: jnp.dot(a, v) >= b)(box_vertices)  # (4,)
        
        # If all inside, return full box area
        all_inside = jnp.all(inside)
        # If all outside, return 0
        all_outside = jnp.all(~inside)
        
        # Compute intersection points for each edge
        n_verts = 4
        
        def edge_intersect(i):
            """Compute intersection of edge i -> (i+1)%4 with half-space boundary."""
            p1 = box_vertices[i]
            p2 = box_vertices[(i + 1) % n_verts]
            
            # Line: p = p1 + t*(p2 - p1), find t where a @ p = b
            # a @ p1 + t * a @ (p2 - p1) = b
            # t = (b - a @ p1) / (a @ (p2 - p1))
            d = p2 - p1
            denom = jnp.dot(a, d)
            t = jnp.where(
                jnp.abs(denom) > 1e-10,
                (b - jnp.dot(a, p1)) / denom,
                0.5  # Default if parallel
            )
            t = jnp.clip(t, 0.0, 1.0)
            return p1 + t * d
        
        # Get all edge intersections
        intersections = jax.vmap(edge_intersect)(jnp.arange(n_verts))  # (4, 2)
        
        # Build clipped polygon vertices:
        # For each edge, output:
        # - If p1 inside and p2 inside: output p2
        # - If p1 inside and p2 outside: output intersection
        # - If p1 outside and p2 inside: output intersection, then p2
        # - If p1 outside and p2 outside: output nothing
        
        # Simplified: collect up to 6 vertices (box can have at most 6 after one clip)
        # We'll use a fixed-size array and track valid count
        
        def process_edge(i):
            """Process edge i, return (vertex1, vertex2, count)."""
            p1_in = inside[i]
            p2_in = inside[(i + 1) % n_verts]
            p2 = box_vertices[(i + 1) % n_verts]
            inter = intersections[i]
            
            # Case 1: both inside - output p2
            # Case 2: p1 in, p2 out - output intersection
            # Case 3: p1 out, p2 in - output intersection, then p2
            # Case 4: both out - output nothing
            
            v1 = jnp.where(p1_in & ~p2_in, inter, p2)  # if exiting, use intersection
            v2 = jnp.where(~p1_in & p2_in, p2, jnp.zeros(2))  # if entering, also output p2
            
            count = jnp.where(
                p1_in & p2_in, 1,  # both in: 1 vertex
                jnp.where(
                    p1_in & ~p2_in, 1,  # exiting: 1 vertex (intersection)
                    jnp.where(
                        ~p1_in & p2_in, 2,  # entering: 2 vertices
                        0  # both out: 0 vertices
                    )
                )
            )
            
            return v1, v2, count
        
        # Process all edges
        results = jax.vmap(process_edge)(jnp.arange(n_verts))
        v1s, v2s, counts = results  # (4, 2), (4, 2), (4,)
        
        # Flatten into polygon vertices (max 8, but typically 3-6)
        # We'll compute area using shoelace formula on collected vertices
        
        # Collect valid vertices
        all_verts = jnp.concatenate([v1s, v2s], axis=0)  # (8, 2)
        valid_mask = jnp.concatenate([
            counts >= 1,  # v1 is valid if count >= 1
            counts >= 2,  # v2 is valid if count >= 2
        ])  # (8,)
        
        # Use shoelace formula with masking
        # Area = 0.5 * |sum_i (x_i * y_{i+1} - x_{i+1} * y_i)|
        # We need to handle variable-length polygon, so use weighted sum
        
        # Sort vertices by angle from centroid for correct polygon ordering
        centroid = jnp.sum(all_verts * valid_mask[:, None], axis=0) / jnp.maximum(jnp.sum(valid_mask), 1.0)
        angles = jnp.arctan2(all_verts[:, 1] - centroid[1], all_verts[:, 0] - centroid[0])
        # Set invalid vertices to large angle to push them to end
        angles = jnp.where(valid_mask, angles, 100.0)
        sorted_indices = jnp.argsort(angles)
        sorted_verts = all_verts[sorted_indices]
        sorted_valid = valid_mask[sorted_indices]
        
        # Shoelace formula
        def shoelace_term(i):
            j = (i + 1) % 8
            term = sorted_verts[i, 0] * sorted_verts[j, 1] - sorted_verts[j, 0] * sorted_verts[i, 1]
            # Only count if both vertices are valid
            return jnp.where(sorted_valid[i] & sorted_valid[j], term, 0.0)
        
        area = 0.5 * jnp.abs(jnp.sum(jax.vmap(shoelace_term)(jnp.arange(8))))
        
        # Handle edge cases
        area = jnp.where(all_inside, full_box_area(), area)
        area = jnp.where(all_outside, 0.0, area)
        
        return area
    
    # If gradient is zero, return full box
    return jax.lax.cond(
        grad_norm < 1e-8,
        full_box_area,
        clip_polygon
    )


# =============================================================================
# JIT-compiled Multi-Policy Value Functions
# =============================================================================

# Pre-create the JIT function NOT NEEDED (we use _compute_value_pure)


# Policy colors for visualization
POLICY_COLORS = {
    'lane_change_left': '#2196F3',   # Blue
    'lane_change_right': '#FF9800',  # Orange
    'nominal': '#9C27B0',            # Purple (nominal controller trajectory)
    'stop': '#F44336',               # Red
}

SELECTED_COLOR = '#00E676'  # Bright green for selected policy

# Per-policy alpha values (smaller alpha = more conservative = less likely to be selected)
# Use smaller alpha for stop to prefer lane change when both are safe
POLICY_ALPHA = {
    'lane_change_left': 5.0,
    'lane_change_right': 5.0,
    'nominal': 5.0,
    'stop': 0.5,  # Smaller alpha makes stop less attractive
}


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
        cbf_alpha: float = 5.0,  # Global permissive alpha for QP execution
        left_lane_y: float = 5.0,
        right_lane_y: float = -5.0,
        safety_margin: float = 0.0,
        max_operator: str = 'input_space',  # Selection operator: 'c', 'v', or 'input_space'
        debug: bool = False,
        ax=None
    ):
        """
        Initialize the MPCBF controller.
        
        Args:
            max_operator: Selection operator for choosing best policy:
                - 'c': Constraint value (Vdot + alpha*V) - most permissive
                - 'v': Value function (V) - maximum safety margin
                - 'input_space': Feasible control area - maximum control authority
        """
        # Initialize parent PCBF (but don't use its single policy)
        super().__init__(robot, robot_spec, dt, backup_horizon, cbf_alpha, safety_margin, ax)
        
        # Validate and store selection operator
        if max_operator not in MAX_OPERATOR_TYPES:
            raise ValueError(f"max_operator must be one of {MAX_OPERATOR_TYPES}, got '{max_operator}'")
        self.max_operator = max_operator
                
        # Lane targets
        self.left_lane_y = left_lane_y
        self.right_lane_y = right_lane_y
        
        # Nominal controller trajectory (updated each step)
        self.nominal_trajectory = None
        self.nominal_controls = None
        
        # For visualization - store trajectories from all policies
        self.multi_backup_trajs = {}  # policy_name -> list of trajectories
        self.best_policy_name = None
        self.prev_best_policy = None  # For hysteresis
        
        # Visualization handles for multi-policy
        self.policy_traj_lines = {}

        self.debug = debug
        
        # Multiple policies configuration
        self._setup_multi_policies()
        
        # Setup multi-policy visualization
        if ax is not None:
            self._setup_multi_visualization()

        # For MPCBF visualization naming, keep terminology explicit.
        if self.backup_traj_line is not None:
            self.backup_traj_line.set_label('PL-CBF fallback rollout')
        
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
                'params': LaneChangePolicyParams(
                    target_y=self.left_lane_y,
                    Kp_y=0.15,  # Reduced gain for precise tracking in narrow corridor
                    Kp_theta=1.5,
                    Kd_theta=0.3,
                    Kp_delta=3.0,
                    Kp_v=500.0,
                    Kp_tau_dot=2.0,
                    target_velocity=target_velocity,
                    delta_max=delta_max,
                    delta_dot_max=delta_dot_max,
                    tau_max=tau_max,
                    tau_dot_max=tau_dot_max,
                    theta_des_max=float(np.deg2rad(15)),
                ),
            },
            'lane_change_right': {
                'type': 'lane_change',
                'horizon': uniform_horizon,
                'params': LaneChangePolicyParams(
                    target_y=self.right_lane_y,
                    Kp_y=0.15,  # Reduced gain for precise tracking in narrow corridor
                    Kp_theta=1.5,
                    Kd_theta=0.3,
                    Kp_delta=3.0,
                    Kp_v=500.0,
                    Kp_tau_dot=2.0,
                    target_velocity=target_velocity,
                    delta_max=delta_max,
                    delta_dot_max=delta_dot_max,
                    tau_max=tau_max,
                    tau_dot_max=tau_dot_max,
                    theta_des_max=float(np.deg2rad(15)),
                ),
            },
            'stop': {
                'type': 'stop',
                'horizon': uniform_horizon,
                'params': StopPolicyParams(
                    Kd_theta=0.5,
                    Kp_delta=3.0,
                    Kp_v=1000.0,
                    Kp_tau_dot=2.0,
                    delta_max=delta_max,
                    delta_dot_max=delta_dot_max,
                    tau_max=tau_max,
                    tau_dot_max=tau_dot_max,
                    stop_threshold=0.05,
                    holding_torque=-100.0,
                ),
            },
        }
        
        # Create JAX policy instances (for reference, not used in pure function)
        self.jax_policies = {
            'lane_change_left': LaneChangeControllerJAX(self.robot_spec, target_y=self.left_lane_y),
            'lane_change_right': LaneChangeControllerJAX(self.robot_spec, target_y=self.right_lane_y),
            'stop': StoppingControllerJAX(self.robot_spec),
        }
        
        # JIT cache for value functions (policy_type -> jit_fn)
        self._jit_policy_fns = {}
        
        # Initialize trajectory storage
        for name in list(self.policy_configs.keys()) + ['nominal']:
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
        
        self.policy_configs['lane_change_left']['params'] = self.policy_configs['lane_change_left']['params']._replace(target_y=left_y)
        self.policy_configs['lane_change_right']['params'] = self.policy_configs['lane_change_right']['params']._replace(target_y=right_y)
        
        self.jax_policies['lane_change_left'] = LaneChangeControllerJAX(
            self.robot_spec, target_y=left_y
        )
        self.jax_policies['lane_change_right'] = LaneChangeControllerJAX(
            self.robot_spec, target_y=right_y
        )
    
    def set_nominal_trajectory(self, trajectory: np.ndarray, controls: Optional[np.ndarray] = None):
        """Set the nominal controller's predicted trajectory and controls for this step."""
        if trajectory is None:
            self.nominal_trajectory = None
            self.nominal_controls = None
            return
            
        traj = np.array(trajectory)
        if traj.shape[0] < traj.shape[1]:
            traj = traj.T
        
        self.nominal_trajectory = jnp.array(traj)
        
        if controls is not None:
            ctrl = np.array(controls)
            if ctrl.shape[0] < ctrl.shape[1]:
                ctrl = ctrl.T
            self.nominal_controls = jnp.array(ctrl)
        else:
            self.nominal_controls = None
    
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
        if self.safety_margin > 0:
            robot_radius = self.robot_spec.get('radius', 1.5) + self.safety_margin
        else:
            robot_radius = self.robot_spec.get('radius', 1.5)
        
        if len(self.obstacles) == 0:
            default_V = 100.0
            V_dict = {name: default_V for name in self.policy_configs}
            grad_V_dict = {name: np.zeros(8) for name in self.policy_configs}
            traj_dict = {name: np.array(x0_jax)[None, :] for name in self.policy_configs}
            if self.nominal_trajectory is not None:
                traj_dict['nominal'] = np.array(self.nominal_trajectory)
                V_dict['nominal'] = default_V # Considered safe if no obstacles
            else:
                traj_dict['nominal'] = None
                V_dict['nominal'] = default_V 
            return V_dict, grad_V_dict, traj_dict
        
        obstacles_array = jnp.array(self.obstacles)
        n_obs = len(self.obstacles)
        
        V_dict = {}
        grad_V_dict = {}
        traj_dict = {}
        
        track_width = self.track_width if self.track_width is not None else 1000.0
        
        # Compute V and grad for each rollout policy
        for name, config in self.policy_configs.items():
            policy_type = config['type']
            horizon = config['horizon']
            policy_params = config['params']
            
            # Use module-level JIT function (no re-compilation)
            # The static_argnums in _compute_value_and_grad_jit are (4, 5) corresponding to
            # policy_type and horizon. Since these change per policy, JAX will re-compile
            # ONLY when they change (which is fine, finite set of policies).
            (V, trajectory), grad_V = _compute_value_and_grad_jit(
                x0_jax, self.dynamics_params, policy_params, obstacles_array,
                policy_type, horizon,
                self.robot_spec.get('radius', 1.5) + self.safety_margin, 
                self.current_friction, self.dt,
                track_width
            )
            
            V_dict[name] = float(V)
            grad_V_dict[name] = np.array(grad_V)
            traj_dict[name] = np.array(trajectory)
        
        # Compute value for nominal controller trajectory (with padding if needed)
        if self.nominal_controls is not None and len(self.nominal_controls) > 0:
            # Create a JIT-able pure function for nominal trajectory
            params = self.dynamics_params
            robot_radius = self.robot_spec.get('radius', 1.5) + self.safety_margin
            mu = self.current_friction
            dt = self.dt
            target_len = self.backup_horizon_steps
            
            # Define the pure function locally but it should be JIT-safe as it uses pure components
            def nominal_value_pure(x0, controls_input, dyn_params, obs_arr, r_rad, fric, t_step, t_width, t_len):
                curr_len = controls_input.shape[0]
                last_u = controls_input[-1]
                
                def get_u_at_t(i, _):
                    # Clamp index: for i >= curr_len, use last control (curr_len-1)
                    idx = jnp.minimum(i, curr_len - 1)
                    # Use safe dynamic indexing for tracers
                    return jax.lax.dynamic_index_in_dim(controls_input, idx, axis=0, keepdims=False)
                
                def body_fn(carry, i):
                    x = carry
                    u = get_u_at_t(i, None)
                    # Use pure step function
                    x_next = step_full_state_pure(x, u, dyn_params, t_step, fric)
                    
                    # Compute h
                    x_curr, y_curr = x_next[0], x_next[1]
                    def h_single_obs(obs):
                        obs_x, obs_y, obs_r = obs[0], obs[1], obs[2]
                        combined_r = obs_r + r_rad
                        dist = jnp.sqrt((x_curr - obs_x)**2 + (y_curr - obs_y)**2 + 1e-8)
                        return dist - combined_r
                    
                    h_obs = jax.lax.select(
                        obs_arr.shape[0] > 0,
                        smooth_min(jax.vmap(h_single_obs)(obs_arr), temperature=5.0),
                        100.0
                    )
                    
                    # Boundary check
                    y_limit = t_width / 2.0 - r_rad
                    h_bound_upper = y_limit - y_curr
                    h_bound_lower = y_curr - (-y_limit)
                    h_bound = smooth_min(jnp.array([h_bound_upper, h_bound_lower]), temperature=5.0)
                    h_val = smooth_min(jnp.array([h_obs, h_bound]), temperature=5.0)
                        
                    return x_next, (x_next, h_val)
                
                _, (traj, h_values) = lax.scan(body_fn, x0, jnp.arange(t_len))
                
                trajectory_full = jnp.vstack([x0[None, :], traj])
                
                # Initial h
                h0_vals = jax.vmap(lambda obs: jnp.sqrt((x0[0]-obs[0])**2 + (x0[1]-obs[1])**2 + 1e-8) - (obs[2]+r_rad))(obs_arr)
                h0 = smooth_min(h0_vals, temperature=5.0) if obs_arr.shape[0] > 0 else 100.0
                
                h_all = jnp.concatenate([h0[None], h_values])
                V = smooth_min(h_all, temperature=20.0)
                
                return V, trajectory_full

            # We need a JIT wrapper that handles static args: t_len
            # We cache this wrapper based on t_len
            nominal_key = f"nominal_{target_len}"
            if nominal_key not in self._jit_policy_fns:
                # Create JIT function
                # Args: x0, controls, dyn_params, obs_arr, r_rad, fric, dt, t_width, t_len
                # Static: t_len (arg 8)
                self._jit_policy_fns[nominal_key] = jax.jit(
                    jax.value_and_grad(nominal_value_pure, has_aux=True),
                    static_argnums=(8,) 
                )
            
            val_grad_fn_nominal = self._jit_policy_fns[nominal_key]
            
            (V_nominal, traj_nominal), grad_V_nominal = val_grad_fn_nominal(
                x0_jax, self.nominal_controls, params, obstacles_array, 
                robot_radius, mu, dt, track_width, target_len
            )
            
            V_dict['nominal'] = float(V_nominal)
            grad_V_dict['nominal'] = np.array(grad_V_nominal)
            traj_dict['nominal'] = np.array(traj_nominal)
            
        elif self.nominal_trajectory is not None and len(self.nominal_trajectory) > 0:
            # Fallback if no controls provided: exclude from safety but keep for vis
            traj_dict['nominal'] = np.array(self.nominal_trajectory)
            V_dict['nominal'] = -np.inf # Mark as unsafe/unknown
            grad_V_dict['nominal'] = np.zeros(8)
        else:
            V_dict['nominal'] = -np.inf
            grad_V_dict['nominal'] = np.zeros(8)
            traj_dict['nominal'] = None
        
        return V_dict, grad_V_dict, traj_dict
    
    def _select_best_policy(
        self,
        V_dict: Dict[str, float],
        grad_V_dict: Dict[str, np.ndarray],
        f: np.ndarray,
        G: np.ndarray,
        u_nom: np.ndarray,
    ) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        """
        Select the best policy based on the configured max_operator.
        
        Selection operators:
        - 'c': Constraint value (Vdot + alpha*V) - most permissive constraint
        - 'v': Value function (V) - maximum safety margin  
        - 'input_space': Feasible control area - maximum control authority
        
        For selection, per-policy alphas are used (stop=0.5, others=5.0).
        For QP execution, global permissive alpha (self.cbf_alpha) is used.
        
        Returns:
            best_policy: Name of selected policy
            constraint_dict: {policy_name: constraint_value} using per-policy alphas
            score_dict: {policy_name: score} based on max_operator
        """
        constraint_dict = {}
        score_dict = {}
        area_dict = {}  # Track feasible area for each policy
        max_grad_norm = 50.0
        
        for name in V_dict.keys():
            V = V_dict[name]
            grad_V = grad_V_dict[name].copy()
            
            # Skip invalid policies
            if V == -np.inf or np.any(np.isnan(grad_V)):
                constraint_dict[name] = -np.inf
                score_dict[name] = -np.inf
                continue
            
            # Normalize gradient if too large
            grad_norm = np.linalg.norm(grad_V)
            if grad_norm > max_grad_norm:
                grad_V = grad_V * (max_grad_norm / grad_norm)
            
            # Compute Lie derivatives
            Lf_V = grad_V @ f
            Lg_V = grad_V @ G
            
            # Use per-policy alpha for SELECTION (smaller alpha for stop makes it less attractive)
            # Check self.alphas first (instance dict), then GLOBAL POLICY_ALPHA, then self.cbf_alpha
            instance_alphas = getattr(self, 'alphas', {})
            if name in instance_alphas:
                 policy_alpha = instance_alphas[name]
            else:
                 policy_alpha = POLICY_ALPHA.get(name, self.cbf_alpha)
            
            # Constraint value at u_nom: c = Lf_V + Lg_V @ u_nom + alpha * V
            constraint_value = Lf_V + Lg_V @ u_nom + policy_alpha * V
            constraint_dict[name] = float(constraint_value)
            
            # ALWAYS compute feasible area for feasibility check
            # CBF constraint: Lg_V @ u >= -(Lf_V + alpha * V)
            # Use GLOBAL alpha for area computation (execution phase perspective)
            cbf_rhs = Lf_V + policy_alpha * V
            area = float(_compute_feasible_area_jit(
                jnp.array(Lg_V),
                cbf_rhs,
                jnp.array(self.u_min),
                jnp.array(self.u_max),
            ))
            area_dict[name] = area
            
            # Compute score based on operator
            if self.max_operator == 'c':
                # Constraint value (permissiveness)
                score_dict[name] = constraint_value
                
            elif self.max_operator == 'v':
                # Value function (safety margin)
                score_dict[name] = V
                
            elif self.max_operator == 'input_space':
                # Feasible control area (control authority)
                score_dict[name] = area
            
            if self.debug:
                 print(f"Policy {name}: V={V:.4f}, c={constraint_value:.4f}, area={area:.4f}, rhs={cbf_rhs:.4f}")
        
        # Get all policies including nominal
        backup_policies = list(constraint_dict.keys())
        
        # Filter for safe policies (V > 0)
        safe_policies = [k for k in backup_policies if V_dict[k] > 0.0]
        
        if len(safe_policies) == 0:
             print("WARNING: All policies unsafe (V <= 0)!")
             for k in backup_policies:
                 print(f"  {k}: V={V_dict[k]:.4f}")
        
        if len(safe_policies) > 0:
            candidates = safe_policies
            # Among safe policies, select based on max_operator score
            best_policy = max(candidates, key=lambda k: score_dict[k])
        else:
            # If all are unsafe, pick the one with highest score (damage mitigation)
            candidates = backup_policies
            best_policy = max(candidates, key=lambda k: score_dict[k])
        
        return best_policy, constraint_dict, score_dict, area_dict
    
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

    def _compute_policy_control(
        self,
        policy_name: str,
        robot_state: np.ndarray,
        u_nom: np.ndarray,
    ) -> np.ndarray:
        """Compute one-step control from the selected policy for QP fallback."""
        if policy_name == 'nominal':
            return np.array(u_nom, dtype=float).reshape(-1)

        config = self.policy_configs.get(policy_name)
        if config is None:
            return np.array(u_nom, dtype=float).reshape(-1)

        policy_type = config['type']
        policy_params = config['params']
        x_jax = jnp.array(robot_state)

        if policy_type == 'lane_change':
            u = np.array(LaneChangeControllerJAX.compute(x_jax, policy_params), dtype=float)
            return u.reshape(-1)

        if policy_type == 'stop':
            u = np.array(StoppingControllerJAX.compute(x_jax, policy_params), dtype=float)
            return u.reshape(-1)

        return np.array(u_nom, dtype=float).reshape(-1)
    
    def solve_control_problem(
        self,
        robot_state: np.ndarray,
        control_ref: Optional[dict] = None,
        friction: Optional[float] = None,
        nominal_trajectory: Optional[np.ndarray] = None,
        nominal_controls: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Main MPCBF control loop - compute safe control using multi-policy CBF.
        
        Selection is based on maximum CBF constraint value (Vdot + alpha*V),
        not just maximum V.
        """
        robot_state = np.array(robot_state).flatten()
        
        if friction is not None:
            self.set_friction(friction)
        
        if nominal_trajectory is not None:
            self.set_nominal_trajectory(nominal_trajectory, nominal_controls)
        
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
        
        # Select best policy based on max_operator (c, v, or input_space)
        best_policy, constraint_dict, score_dict, area_dict = self._select_best_policy(
            V_dict, grad_V_dict, f, G, u_nom
        )
        
        self.best_policy_name = best_policy
        V_best = V_dict[best_policy]
        grad_V_best = grad_V_dict[best_policy].copy()
        constraint_best = constraint_dict[best_policy]
        
        # CRITICAL: Normalize gradient to match what was used in _select_best_policy
        # This ensures consistency between input_space feasibility computation and QP solving
        max_grad_norm = 50.0
        grad_norm = np.linalg.norm(grad_V_best)
        if grad_norm > max_grad_norm:
            grad_V_best = grad_V_best * (max_grad_norm / grad_norm)
        
        # Store trajectories for animation
        for name, traj in traj_dict.items():
            if traj is not None and self.curr_step % self.save_every_N == 0:
                if name not in self.multi_backup_trajs:
                    self.multi_backup_trajs[name] = []
                self.multi_backup_trajs[name].append(traj.copy())
        
        # Debug output (can be disabled by setting self.debug = False)
        if getattr(self, 'debug', True) and self.curr_step % 50 == 0:
            v_str = ", ".join([f"{k}:{V_dict[k]:.2f}" for k in sorted(V_dict.keys()) if V_dict[k] > -np.inf])
            score_str = ", ".join([f"{k}:{score_dict[k]:.2f}" for k in sorted(score_dict.keys()) if score_dict[k] > -np.inf])
            area_str = ", ".join([f"{k}:{area_dict.get(k, 0):.0f}" for k in sorted(V_dict.keys()) if k in area_dict])
            print(f"  [MPCBF] V: {v_str}")
            print(f"  [MPCBF] area: {area_str}")
            print(f"  [MPCBF] {self.max_operator}: {score_str} -> best={best_policy}")
        
        self.curr_step += 1
        

        # UPDATE QP ALPHA to match selected policy
        instance_alphas = getattr(self, 'alphas', {})
        if best_policy in instance_alphas:
             self.cbf_alpha = instance_alphas[best_policy]
        elif best_policy in POLICY_ALPHA:
             self.cbf_alpha = POLICY_ALPHA[best_policy]
        # else keep existing self.cbf_alpha
        
        # Solve CBF-QP with best policy's constraint.
        try:
            u_safe = self._solve_cbf_qp(u_nom, V_best, grad_V_best, f, G)
        except ValueError as err:
            if V_best > 0.0:
                self.status = 'policy_fallback'
                u_safe = self._compute_policy_control(best_policy, robot_state, u_nom)
            else:
                raise err
        
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
        
