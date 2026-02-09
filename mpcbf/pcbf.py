"""
Created on February 9th, 2026
@author: Taekyung Kim

@description:
Abstract base class for PCBF (Policy Control Barrier Function) implementations.

Provides common QP solving structure and control flow. Subclasses override
scenario-specific methods for dynamics, value functions, and CBF constraints.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import numpy as np
import cvxpy as cp


class PCBFBase(ABC):
    """
    Abstract base class for PCBF implementations.
    
    Common pattern:
    1. Compute value function V(x) and gradient ∇V(x) via backup rollout
    2. Formulate CBF-QP: min ||u - u_nom||^2
       s.t. ∇V^T (f + Gu) + α·V >= 0 (CBF constraint)
            u_min <= u <= u_max (input limits)
    3. Solve QP and return safe control
    
    Subclasses must override:
    - _setup_dynamics(): Initialize dynamics model
    - _setup_policy(): Initialize backup policy
    - _compute_value_and_grad(): Compute V, ∇V via rollout
    - _get_system_matrices(): Get f(x), g(x)
    - _add_input_constraints(): Add input limit constraints
    - _add_cbf_constraints(): Add CBF constraints
    - _get_control_dim(): Return control dimension
    """
    
    def __init__(
        self,
        robot_spec: dict,
        dt: float = 0.05,
        backup_horizon: float = 2.0,
        cbf_alpha: float = 5.0,
        safety_margin: float = 0.0,
        ax=None
    ):
        """
        Initialize base PCBF controller.
        
        Args:
            robot_spec: Robot specification dictionary
            dt: Time step
            backup_horizon: Backup trajectory horizon (seconds)
            cbf_alpha: CBF class-K function parameter
            safety_margin: Additional safety margin for obstacles
            ax: Matplotlib axis for visualization
        """
        self.robot_spec = robot_spec
        self.dt = dt
        self.backup_horizon = backup_horizon
        self.backup_horizon_steps = int(backup_horizon / dt)
        self.cbf_alpha = cbf_alpha
        self.safety_margin = safety_margin
        self.ax = ax
        
        # To be initialized by subclasses
        self.dynamics = None
        self.backup_policy = None
        self.policy_type = None
        self.policy_params = None
        
        # Common state
        self.dynamic_obstacles = []
        self.static_obstacles = []
        self.env = None
        self.current_friction = robot_spec.get('mu', 1.0)
        
        # Visualization
        self.backup_traj_line = None
        
        # Subclass initialization
        self._setup_dynamics()
        self._setup_visualization()
    
    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def _setup_dynamics(self):
        """Initialize dynamics model. Must set self.dynamics."""
        pass
    
    @abstractmethod
    def set_policy(self, policy_type: str, params: Any):
        """
        Set backup policy.
        
        Args:
            policy_type: Type of policy (e.g., 'lane_change', 'stop', 'angle')
            params: Policy parameters (implementation-specific)
        """
        pass
    
    @abstractmethod
    def _compute_value_and_grad(self, state: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute value function and gradient via backup rollout.
        
        Args:
            state: Current state
            
        Returns:
            V: Value function (scalar, positive = safe)
            grad_V: Gradient of V w.r.t. state
            trajectory: Backup rollout trajectory
        """
        pass
    
    @abstractmethod
    def _get_system_matrices(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get system dynamics matrices f(x) and g(x).
        
        For system: ẋ = f(x) + g(x)u
        
        Args:
            state: Current state
            
        Returns:
            f: Drift vector f(x)
            g: Input matrix g(x)
        """
        pass
    
    @abstractmethod
    def _add_input_constraints(self, u: cp.Variable, constraints: list):
        """
        Add input limit constraints to QP.
        
        Args:
            u: Control variable
            constraints: List to append constraints to
        """
        pass
    
    @abstractmethod
    def _add_cbf_constraints(
        self, 
        u: cp.Variable, 
        constraints: list, 
        state: np.ndarray, 
        V: float, 
        grad_V: np.ndarray
    ):
        """
        Add CBF constraints to QP.
        
        Args:
            u: Control variable
            constraints: List to append constraints to
            state: Current state
            V: Value function
            grad_V: Gradient of V
        """
        pass
    
    @abstractmethod
    def _get_control_dim(self) -> int:
        """Return control dimension."""
        pass
    
    # =========================================================================
    # Common implementation - shared across all subclasses
    # =========================================================================
    
    def solve_control_problem(
        self, 
        state: np.ndarray, 
        control_ref: Optional[dict] = None,
        friction: Optional[float] = None
    ) -> np.ndarray:
        """
        Solve PCBF-QP for safe control.
        
        Args:
            state: Current state
            control_ref: Reference control dict with 'u_ref' key
            friction: Current friction coefficient (optional)
            
        Returns:
            u_safe: Safe control satisfying CBF constraint
        """
        # Update friction if provided
        if friction is not None:
            self.current_friction = friction
        
        # 1. Get nominal control
        u_nom = self._get_nominal_control(control_ref)
        
        # 2. Compute value function and gradient
        V, grad_V, trajectory = self._compute_value_and_grad(state)
        
        # 3. Update visualization
        self._update_visualization(trajectory)
        
        # 4. Formulate QP
        u = cp.Variable(self._get_control_dim())
        cost = cp.sum_squares(u - u_nom)
        constraints = []
        
        # Add input limits
        self._add_input_constraints(u, constraints)
        
        # Add CBF constraints
        self._add_cbf_constraints(u, constraints, state, V, grad_V)
        
        # 5. Solve QP
        return self._solve_qp(u, cost, constraints, u_nom)
    
    def update_obstacles(self, dynamic_obs: list, static_obs: list):
        """
        Update obstacle information.
        
        Args:
            dynamic_obs: List of dynamic obstacle dicts
            static_obs: List of static obstacle dicts
        """
        self.dynamic_obstacles = dynamic_obs
        self.static_obstacles = static_obs
    
    def set_environment(self, env):
        """Set environment for obstacle information."""
        self.env = env
    
    def set_friction(self, mu: float):
        """Set current friction coefficient."""
        self.current_friction = mu
    
    # =========================================================================
    # Helper methods - common utilities
    # =========================================================================
    
    def _get_nominal_control(self, control_ref: Optional[dict]) -> np.ndarray:
        """Extract nominal control from reference."""
        if control_ref and 'u_ref' in control_ref:
            return np.array(control_ref['u_ref']).flatten()
        return np.zeros(self._get_control_dim())
    
    def _solve_qp(
        self, 
        u: cp.Variable, 
        cost, 
        constraints: list, 
        u_nom: np.ndarray
    ) -> np.ndarray:
        """
        Solve QP and handle failures gracefully.
        
        Args:
            u: Control variable
            cost: Cost function
            constraints: List of constraints
            u_nom: Nominal control (fallback)
            
        Returns:
            Optimal control or nominal control if infeasible
        """
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(verbose=False)
            if prob.status in ['optimal', 'optimal_inaccurate']:
                return u.value
            else:
                return u_nom
        except Exception:
            return u_nom
    
    def _setup_visualization(self):
        """Setup visualization elements."""
        if self.ax is not None:
            self.backup_traj_line, = self.ax.plot(
                [], [], 'c-', alpha=0.8, linewidth=2, label='PCBF Backup'
            )
    
    def _update_visualization(self, trajectory: Optional[np.ndarray]):
        """Update backup trajectory visualization."""
        if self.backup_traj_line is not None:
            if trajectory is not None and len(trajectory) > 0:
                self.backup_traj_line.set_data(trajectory[:, 0], trajectory[:, 1])
            else:
                self.backup_traj_line.set_data([], [])
