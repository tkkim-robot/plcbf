"""
Created on January 7th, 2026
@author: Taekyung Kim

@description:
Policy Control Barrier Function (PCBF) implementation in JAX.

PCBF is an algorithm that:
1. Rolls out a backup policy (e.g., lane change, stop) for a finite horizon
2. Computes the value function V(x) = min_{t ∈ [0, T]} h(x(t)) over the rollout
   where h(x) > 0 means safe, h(x) < 0 means unsafe
3. Uses autograd to compute ∇V(x)
4. Formulates a CBF constraint: ∇V^T (f + Gu) + α V ≥ 0
5. Solves a QP to find control minimally deviating from nominal that satisfies the CBF constraint

This is a simplified (deterministic) version of the Robust Policy CBF (RPCBF) from:
"Safety on the Fly: Real-time Safety Assurance for Autonomous Systems"

Sign Convention (consistent with safe_control repo):
- h(x) > 0 means SAFE (no collision)
- h(x) < 0 means UNSAFE (collision)
- V = min_t h(x(t)) is the minimum safety margin along backup trajectory
- V > 0 means entire backup trajectory is safe
- V < 0 means backup trajectory has collision

@required-scripts: safe_control/robots/dynamic_bicycle2D.py
"""

import functools as ft
from functools import partial
from typing import Callable, Optional, Tuple, Any, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
import jaxopt
import cvxpy as cp


# =============================================================================
# NamedTuple Definitions for JAX PyTree Compatibility
# =============================================================================

class DynamicsParams(NamedTuple):
    """Dynamics parameters as a NamedTuple for JAX PyTree compatibility.
    
    Using NamedTuple avoids JIT recompilation when parameter values change,
    since JAX treats NamedTuples as data (PyTree nodes) rather than static config.
    """
    a: float           # Front axle to CG
    b: float           # Rear axle to CG
    m: float           # Vehicle mass
    Iz: float          # Yaw moment of inertia
    Cc_f: float        # Front cornering stiffness
    Cc_r: float        # Rear cornering stiffness
    r_w: float         # Wheel radius
    gamma: float       # Numeric stability parameter
    delta_max: float   # Max steering angle
    tau_max: float     # Max torque
    v_max: float       # Max velocity
    v_min: float       # Min velocity
    r_max: float       # Max yaw rate
    beta_max: float    # Max slip angle


class LaneChangePolicyParams(NamedTuple):
    """Lane change policy parameters for JAX PyTree compatibility."""
    target_y: float
    Kp_y: float
    Kp_theta: float
    Kd_theta: float
    Kp_delta: float
    Kp_v: float
    Kp_tau_dot: float      # Torque rate gain (smooth control, gradient-safe)
    target_velocity: float
    delta_max: float
    delta_dot_max: float
    tau_max: float
    tau_dot_max: float
    theta_des_max: float


class StopPolicyParams(NamedTuple):
    """Stopping policy parameters for JAX PyTree compatibility."""
    Kd_theta: float
    Kp_delta: float
    Kp_v: float
    Kp_tau_dot: float      # Torque rate gain (smooth control, gradient-safe)
    delta_max: float
    delta_dot_max: float
    tau_max: float
    tau_dot_max: float
    stop_threshold: float
    holding_torque: float


def angle_normalize_jax(x):
    """Normalize angle to [-pi, pi] using JAX."""
    return jnp.mod(x + jnp.pi, 2 * jnp.pi) - jnp.pi


# =============================================================================
# JAX-compatible Drifting Car Dynamics
# =============================================================================

class DriftingCarDynamicsJAX:
    """
    JAX-compatible drifting car dynamics for PCBF.
    
    State: [x, y, theta, r, beta, V, delta, tau] (8 states)
    Control: [delta_dot, tau_dot] (2 controls)
    
    This wraps the dynamics from DynamicBicycle2D in JAX for autograd compatibility.
    """
    
    def __init__(self, robot_spec: dict, dt: float):
        """
        Initialize JAX dynamics.
        
        Args:
            robot_spec: Robot specification dictionary
            dt: Time step
        """
        self.robot_spec = robot_spec
        self.dt = dt
        self.gravity = 9.81
        
        # Extract parameters
        self.a = robot_spec.get('a', 1.4)  # Front axle to CG
        self.b = robot_spec.get('b', 1.4)  # Rear axle to CG
        self.m = robot_spec.get('m', 2500.0)  # Vehicle mass
        self.Iz = robot_spec.get('Iz', 5000.0)  # Yaw moment of inertia
        self.Cc_f = robot_spec.get('Cc_f', 80000.0)  # Front cornering stiffness
        self.Cc_r = robot_spec.get('Cc_r', 100000.0)  # Rear cornering stiffness
        self.r_w = robot_spec.get('r_w', 0.35)  # Wheel radius
        self.gamma = robot_spec.get('gamma', 0.95)  # Numeric stability
        
        # Limits
        self.delta_max = robot_spec.get('delta_max', np.deg2rad(20))
        self.delta_dot_max = robot_spec.get('delta_dot_max', np.deg2rad(15))
        self.tau_max = robot_spec.get('tau_max', 4000.0)
        self.tau_dot_max = robot_spec.get('tau_dot_max', 8000.0)
        self.v_max = robot_spec.get('v_max', 20.0)
        self.v_min = robot_spec.get('v_min', 0.0)
        self.r_max = robot_spec.get('r_max', 2.0)
        self.beta_max = robot_spec.get('beta_max', np.deg2rad(45))
        
        # Normal forces (static)
        L = self.a + self.b
        self.Fz_f = self.m * self.gravity * self.b / L
        self.Fz_r = self.m * self.gravity * self.a / L
        
        # Default friction (can be overridden per call)
        self.mu = robot_spec.get('mu', 1.0)
    
    def _compute_slip_angles(self, r, beta, V, delta):
        """Compute front and rear slip angles."""
        V_safe = jnp.maximum(V, 0.1)
        
        # Front slip angle
        alpha_f = jnp.arctan2(V * jnp.sin(beta) + self.a * r, V_safe * jnp.cos(beta)) - delta
        
        # Rear slip angle
        alpha_r = jnp.arctan2(V * jnp.sin(beta) - self.b * r, V_safe * jnp.cos(beta))
        
        return alpha_f, alpha_r
    
    def _compute_lateral_force(self, alpha, Cc, Fz, Fx, mu):
        """Compute lateral tire force using Fiala brush model."""
        # Maximum lateral force
        Fy_max = jnp.sqrt(jnp.maximum((mu * Fz)**2 - self.gamma * Fx**2, 1.0))
        
        # Slip angle at which tire starts sliding
        alpha_sl = jnp.arctan(3 * Fy_max / Cc)
        
        tan_alpha = jnp.tan(alpha)
        
        # Linear region force
        Fy_linear = (-Cc * tan_alpha 
                    + (Cc**2 / (3 * Fy_max)) * jnp.abs(tan_alpha) * tan_alpha
                    - (Cc**3 / (27 * Fy_max**2)) * tan_alpha**3)
        
        # Saturated force
        Fy_saturated = -Fy_max * jnp.sign(alpha)
        
        # Switch between linear and saturated regions
        Fy = jnp.where(jnp.abs(alpha) < alpha_sl, Fy_linear, Fy_saturated)
        
        return Fy
    
    def _compute_longitudinal_force(self, tau, Fz, mu):
        """Compute longitudinal tire force with soft saturation."""
        F_lim = mu * Fz
        eps = 1.0
        F_lim_safe = jnp.maximum(F_lim, eps)
        Fx = F_lim * jnp.tanh(tau / (self.r_w * F_lim_safe))
        return Fx
    
    def _compute_tire_forces(self, r, beta, V, delta, tau, mu):
        """Compute all tire forces."""
        alpha_f, alpha_r = self._compute_slip_angles(r, beta, V, delta)
        
        # Longitudinal forces (rear wheel drive)
        Fx_f = 0.0
        Fx_r = self._compute_longitudinal_force(tau, self.Fz_r, mu)
        
        # Lateral forces
        Fy_f = self._compute_lateral_force(alpha_f, self.Cc_f, self.Fz_f, Fx_f, mu)
        Fy_r = self._compute_lateral_force(alpha_r, self.Cc_r, self.Fz_r, Fx_r, mu)
        
        return Fx_f, Fy_f, Fx_r, Fy_r
    
    def f_dyn(self, x_dyn, mu=None):
        """
        Compute drift dynamics f(x) for dynamics state.
        
        Args:
            x_dyn: Dynamics state [r, beta, V, delta, tau] (5,)
            mu: Friction coefficient (optional, uses default if None)
            
        Returns:
            f(x): Drift term (5,)
        """
        if mu is None:
            mu = self.mu
            
        r, beta, V, delta, tau = x_dyn[0], x_dyn[1], x_dyn[2], x_dyn[3], x_dyn[4]
        
        Fx_f, Fy_f, Fx_r, Fy_r = self._compute_tire_forces(r, beta, V, delta, tau, mu)
        
        V_safe = jnp.maximum(V, 0.1)
        
        # Yaw acceleration
        r_dot = (self.a * (Fx_f * jnp.sin(delta) + Fy_f * jnp.cos(delta)) - self.b * Fy_r) / self.Iz
        
        # Side slip rate
        beta_dot = ((Fx_f * jnp.sin(delta - beta) + Fy_f * jnp.cos(delta - beta)
                    - Fx_r * jnp.sin(beta) + Fy_r * jnp.cos(beta)) / (self.m * V_safe) - r)
        
        # Velocity rate
        V_dot = ((Fx_f * jnp.cos(delta - beta) - Fy_f * jnp.sin(delta - beta)
                 + Fx_r * jnp.cos(beta) + Fy_r * jnp.sin(beta)) / self.m)
        
        return jnp.array([r_dot, beta_dot, V_dot, 0.0, 0.0])
    
    def g_dyn(self, x_dyn):
        """
        Compute input matrix g(x) for dynamics state.
        
        Args:
            x_dyn: Dynamics state [r, beta, V, delta, tau] (5,)
            
        Returns:
            g(x): Input matrix (5, 2)
        """
        g = jnp.zeros((5, 2))
        g = g.at[3, 0].set(1.0)  # delta_dot
        g = g.at[4, 1].set(1.0)  # tau_dot
        return g
    
    def step_full_state(self, x_full, u, mu=None):
        """
        Step the full state forward by dt using Euler integration.
        
        Args:
            x_full: Full state [x, y, theta, r, beta, V, delta, tau] (8,)
            u: Control [delta_dot, tau_dot] (2,)
            mu: Friction coefficient (optional)
            
        Returns:
            x_next: Next full state (8,)
        """
        if mu is None:
            mu = self.mu
            
        # Extract states
        x_pos, y_pos, theta = x_full[0], x_full[1], x_full[2]
        x_dyn = x_full[3:8]  # [r, beta, V, delta, tau]
        
        # Dynamics step
        f = self.f_dyn(x_dyn, mu)
        g = self.g_dyn(x_dyn)
        x_dyn_next = x_dyn + (f + g @ u) * self.dt
        
        # Clamp dynamics states
        r_next = jnp.clip(x_dyn_next[0], -self.r_max, self.r_max)
        beta_next = jnp.clip(x_dyn_next[1], -self.beta_max, self.beta_max)
        V_next = jnp.clip(x_dyn_next[2], self.v_min, self.v_max)
        delta_next = jnp.clip(x_dyn_next[3], -self.delta_max, self.delta_max)
        tau_next = jnp.clip(x_dyn_next[4], -self.tau_max, self.tau_max)
        
        # Global position update
        V = x_dyn[2]
        beta = x_dyn[1]
        r = x_dyn[0]
        
        vx_global = V * jnp.cos(theta + beta)
        vy_global = V * jnp.sin(theta + beta)
        
        x_next = x_pos + vx_global * self.dt
        y_next = y_pos + vy_global * self.dt
        theta_next = angle_normalize_jax(theta + r * self.dt)
        
        return jnp.array([x_next, y_next, theta_next, r_next, beta_next, V_next, delta_next, tau_next])
    
    def f_full(self, x_full, mu=None):
        """
        Compute drift dynamics f(x) for full state (for CBF constraint).
        
        Args:
            x_full: Full state [x, y, theta, r, beta, V, delta, tau] (8,)
            mu: Friction coefficient
            
        Returns:
            f(x): Drift term (8,)
        """
        if mu is None:
            mu = self.mu
            
        theta = x_full[2]
        x_dyn = x_full[3:8]
        
        # Dynamics drift
        f_dyn = self.f_dyn(x_dyn, mu)
        
        # Global position drift
        V, beta, r = x_dyn[2], x_dyn[1], x_dyn[0]
        vx_global = V * jnp.cos(theta + beta)
        vy_global = V * jnp.sin(theta + beta)
        
        return jnp.array([vx_global, vy_global, r, f_dyn[0], f_dyn[1], f_dyn[2], 0.0, 0.0])
    
    def g_full(self, x_full):
        """
        Compute input matrix g(x) for full state (for CBF constraint).
        
        Args:
            x_full: Full state [x, y, theta, r, beta, V, delta, tau] (8,)
            
        Returns:
            g(x): Input matrix (8, 2)
        """
        g = jnp.zeros((8, 2))
        g = g.at[6, 0].set(1.0)  # delta_dot
        g = g.at[7, 1].set(1.0)  # tau_dot
        return g


# =============================================================================
# Pure JAX Backup Policies (for tracing/autograd compatibility)
# =============================================================================

class LaneChangeControllerJAX:
    """
    Pure JAX implementation of lane change controller for PCBF.
    
    This controller steers the vehicle to change to a target lane Y position
    and stabilize there using cascaded PD control.
    
    All control laws are smooth and differentiable for gradient computation.
    """
    
    def __init__(self, robot_spec: dict, target_y: float = 4.0):
        """
        Args:
            robot_spec: Robot specification dictionary
            target_y: Target lateral position (lane center Y coordinate)
        """
        self.target_y = target_y
        
        # Cascaded control gains (tuned for smooth lane change)
        self.Kp_y = 0.15          # Proportional gain: lateral error -> heading
        self.Kp_theta = 1.5       # Proportional gain: heading error -> steering
        self.Kd_theta = 0.3       # Derivative gain (using yaw rate)
        self.Kp_delta = 3.0       # Proportional gain: steering error -> steering rate
        self.Kp_v = 500.0         # Proportional gain: velocity error -> torque
        self.Kp_tau_dot = 2.0     # Proportional gain: torque error -> torque rate
                                  # (smooth control, gradient-safe)
        
        self.target_velocity = robot_spec.get('v_ref', 8.0)
        
        # Limits
        self.delta_max = robot_spec.get('delta_max', np.deg2rad(20))
        self.delta_dot_max = robot_spec.get('delta_dot_max', np.deg2rad(15))
        self.tau_max = robot_spec.get('tau_max', 4000.0)
        self.tau_dot_max = robot_spec.get('tau_dot_max', 8000.0)
        self.theta_des_max = np.deg2rad(15)
    
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """Compute control (instance method wrapper)."""
        # Create params tuple from instance attributes
        params = LaneChangePolicyParams(
            target_y=self.target_y,
            Kp_y=self.Kp_y,
            Kp_theta=self.Kp_theta,
            Kd_theta=self.Kd_theta,
            Kp_delta=self.Kp_delta,
            Kp_v=self.Kp_v,
            Kp_tau_dot=self.Kp_tau_dot,
            target_velocity=self.target_velocity,
            delta_max=self.delta_max,
            delta_dot_max=self.delta_dot_max,
            tau_max=self.tau_max,
            tau_dot_max=self.tau_dot_max,
            theta_des_max=self.theta_des_max,
        )
        return LaneChangeControllerJAX.compute(state, params)
    
    @staticmethod
    def compute(state: jnp.ndarray, params: LaneChangePolicyParams) -> jnp.ndarray:
        """
        Compute lane change control input (static method for JIT compatibility).
        
        Args:
            state: JAX array [x, y, theta, r, beta, V, delta, tau] (8,)
            params: LaneChangePolicyParams NamedTuple
            
        Returns:
            u: JAX array [delta_dot, tau_dot] (2,)
        """
        # Extract state
        y = state[1]
        theta = state[2]
        r = state[3]
        V = jnp.maximum(state[5], 0.1)  # Minimum velocity for stability
        delta = state[6]
        tau = state[7]
        
        # ===== Outer loop: Lateral position control =====
        y_error = params.target_y - y
        theta_des = jnp.arctan(params.Kp_y * y_error)
        theta_des = jnp.clip(theta_des, -params.theta_des_max, params.theta_des_max)
        
        # ===== Inner loop: Heading control =====
        theta_error = jnp.mod(theta_des - theta + jnp.pi, 2 * jnp.pi) - jnp.pi
        delta_des = params.Kp_theta * theta_error - params.Kd_theta * r
        delta_des = jnp.clip(delta_des, -params.delta_max, params.delta_max)
        
        # ===== Actuator loop: Steering rate control =====
        delta_error = delta_des - delta
        delta_dot = params.Kp_delta * delta_error
        delta_dot = jnp.clip(delta_dot, -params.delta_dot_max, params.delta_dot_max)
        
        # ===== Velocity control =====
        V_error = params.target_velocity - V
        tau_des = params.Kp_v * V_error
        tau_des = jnp.clip(tau_des, -params.tau_max, params.tau_max)
        
        # ===== Torque rate control (smooth proportional, gradient-safe) =====
        tau_error = tau_des - tau
        tau_dot = params.Kp_tau_dot * tau_error
        tau_dot = jnp.clip(tau_dot, -params.tau_dot_max, params.tau_dot_max)
        
        return jnp.array([delta_dot, tau_dot])


class StoppingControllerJAX:
    """
    Pure JAX implementation of stopping controller for PCBF.
    
    This controller brings the vehicle to a complete stop.
    
    All control laws are smooth and differentiable for gradient computation.
    """
    
    def __init__(self, robot_spec: dict):
        """
        Args:
            robot_spec: Robot specification dictionary
        """
        # Braking control gains
        self.Kp_v = 1000.0         # Proportional gain: velocity -> torque
        self.Kp_tau_dot = 2.0      # Proportional gain: torque error -> torque rate
                                   # (smooth control, gradient-safe)
        
        # Steering control gains
        self.Kd_theta = 0.5        # Derivative gain (yaw rate damping)
        self.Kp_delta = 3.0        # Proportional gain: steering error -> steering rate
        
        # Limits
        self.delta_max = robot_spec.get('delta_max', np.deg2rad(20))
        self.delta_dot_max = robot_spec.get('delta_dot_max', np.deg2rad(15))
        self.tau_max = robot_spec.get('tau_max', 4000.0)
        self.tau_dot_max = robot_spec.get('tau_dot_max', 8000.0)
        
        self.stop_velocity_threshold = 0.05
        self.holding_torque = -100.0
    
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """Compute control (instance method wrapper)."""
        params = StopPolicyParams(
            Kd_theta=self.Kd_theta,
            Kp_delta=self.Kp_delta,
            Kp_v=self.Kp_v,
            Kp_tau_dot=self.Kp_tau_dot,
            delta_max=self.delta_max,
            delta_dot_max=self.delta_dot_max,
            tau_max=self.tau_max,
            tau_dot_max=self.tau_dot_max,
            stop_threshold=self.stop_velocity_threshold,
            holding_torque=self.holding_torque,
        )
        return StoppingControllerJAX.compute(state, params)
    
    @staticmethod
    def compute(state: jnp.ndarray, params: StopPolicyParams) -> jnp.ndarray:
        """
        Compute stopping control input (static method for JIT compatibility).
        
        Args:
            state: JAX array [x, y, theta, r, beta, V, delta, tau] (8,)
            params: StopPolicyParams NamedTuple
            
        Returns:
            u: JAX array [delta_dot, tau_dot] (2,)
        """
        # Extract state
        r = state[3]
        V = state[5]
        delta = state[6]
        tau = state[7]
        
        # ===== Braking control (smooth proportional, gradient-safe) =====
        tau_braking = -params.Kp_v * V + params.holding_torque
        tau_des = jnp.clip(tau_braking, -params.tau_max, params.tau_max)
        
        # ===== Torque rate control =====
        tau_error = tau_des - tau
        tau_dot = params.Kp_tau_dot * tau_error
        tau_dot = jnp.clip(tau_dot, -params.tau_dot_max, params.tau_dot_max)
        
        # ===== Steering control =====
        delta_des = -params.Kd_theta * r
        delta_des = jnp.clip(delta_des, -params.delta_max, params.delta_max)
        
        delta_error = delta_des - delta
        delta_dot = params.Kp_delta * delta_error
        delta_dot = jnp.clip(delta_dot, -params.delta_dot_max, params.delta_dot_max)
        
        return jnp.array([delta_dot, tau_dot])


class BackupPolicyJAX:
    """
    Wrapper to make backup controllers compatible with JAX.
    
    NOTE: This wrapper should NOT be used for gradient computation.
    Use LaneChangeControllerJAX or StoppingControllerJAX instead.
    This is kept for backward compatibility with non-JAX code paths.
    """
    
    def __init__(self, backup_controller, target=None):
        """
        Args:
            backup_controller: BackupController instance from safe_control
            target: Target for the controller (e.g., target_y for lane change)
        """
        self.backup_controller = backup_controller
        self.target = target
    
    def compute_control_numpy(self, state_np):
        """Compute control using numpy arrays (non-JAX path)."""
        u_np = self.backup_controller.compute_control(state_np, self.target)
        return u_np


# =============================================================================
# Value Function Computation (with JIT compilation)
# =============================================================================

def smooth_min(x: jnp.ndarray, temperature: float = 10.0) -> float:
    """
    Smooth approximation of min using negative log-sum-exp.
    
    smooth_min(x) = -smooth_max(-x)
    As temperature -> inf, this approaches the true min.
    """
    # smooth_min = -log_sum_exp(-temperature * x) / temperature
    neg_x = -x
    x_max = jnp.max(neg_x)
    smooth_max_neg = x_max + jnp.log(jnp.mean(jnp.exp(temperature * (neg_x - x_max)))) / temperature
    return -smooth_max_neg


def _create_single_obs_h_func(obstacle_x: float, obstacle_y: float, combined_radius: float):
    """Create a JIT-compatible barrier function for single obstacle."""
    def h_func(state: jnp.ndarray) -> float:
        """Barrier function: POSITIVE means SAFE (no collision)."""
        x, y = state[0], state[1]
        dist_sq = (x - obstacle_x)**2 + (y - obstacle_y)**2
        dist = jnp.sqrt(dist_sq + 1e-8)
        return dist - combined_radius  # Positive when safe (far from obstacle)
    return h_func

def _rollout_trajectory(
    dynamics_params: DynamicsParams,
    policy_params,  # LaneChangePolicyParams or StopPolicyParams
    policy_type: str,
    x0: jnp.ndarray,
    horizon: int,
    mu: float,
    dt: float,
) -> jnp.ndarray:
    """
    Rollout trajectory using backup policy. JIT-compiled.
    
    Args:
        dynamics_params: DynamicsParams NamedTuple
        policy_params: LaneChangePolicyParams or StopPolicyParams NamedTuple
        policy_type: 'lane_change' or 'stop'
        x0: Initial state (8,)
        horizon: Number of steps
        mu: Friction coefficient
        dt: Time step
        
    Returns:
        trajectory: (horizon+1, 8) array of states
    """
    # Create dynamics step function with parameters (using NamedTuple attribute access)
    def step_fn(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Single dynamics step."""
        # Extract parameters via NamedTuple attributes
        a = dynamics_params.a
        b = dynamics_params.b
        m = dynamics_params.m
        Iz = dynamics_params.Iz
        Cc_f = dynamics_params.Cc_f
        Cc_r = dynamics_params.Cc_r
        r_w = dynamics_params.r_w
        gamma = dynamics_params.gamma
        gravity = 9.81
        
        # Limits
        delta_max = dynamics_params.delta_max
        tau_max = dynamics_params.tau_max
        v_max = dynamics_params.v_max
        v_min = dynamics_params.v_min
        r_max = dynamics_params.r_max
        beta_max = dynamics_params.beta_max
        
        # Normal forces
        L = a + b
        Fz_f = m * gravity * b / L
        Fz_r = m * gravity * a / L
        
        # Extract states
        x_pos, y_pos, theta = x[0], x[1], x[2]
        r_state, beta, V_vel, delta, tau = x[3], x[4], x[5], x[6], x[7]
        
        V_safe = jnp.maximum(V_vel, 0.1)
        
        # Slip angles
        alpha_f = jnp.arctan2(V_vel * jnp.sin(beta) + a * r_state, V_safe * jnp.cos(beta)) - delta
        alpha_r = jnp.arctan2(V_vel * jnp.sin(beta) - b * r_state, V_safe * jnp.cos(beta))
        
        # Longitudinal force (rear)
        F_lim_r = mu * Fz_r
        Fx_r = F_lim_r * jnp.tanh(tau / (r_w * jnp.maximum(F_lim_r, 1.0)))
        Fx_f = 0.0
        
        # Lateral forces (Fiala brush model)
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
        
        # Dynamics
        r_dot = (a * (Fx_f * jnp.sin(delta) + Fy_f * jnp.cos(delta)) - b * Fy_r) / Iz
        beta_dot = ((Fx_f * jnp.sin(delta - beta) + Fy_f * jnp.cos(delta - beta)
                    - Fx_r * jnp.sin(beta) + Fy_r * jnp.cos(beta)) / (m * V_safe) - r_state)
        V_dot = ((Fx_f * jnp.cos(delta - beta) - Fy_f * jnp.sin(delta - beta)
                 + Fx_r * jnp.cos(beta) + Fy_r * jnp.sin(beta)) / m)
        
        # Update dynamics states
        r_next = jnp.clip(r_state + r_dot * dt, -r_max, r_max)
        beta_next = jnp.clip(beta + beta_dot * dt, -beta_max, beta_max)
        V_next = jnp.clip(V_vel + V_dot * dt, v_min, v_max)
        delta_next = jnp.clip(delta + u[0] * dt, -delta_max, delta_max)
        tau_next = jnp.clip(tau + u[1] * dt, -tau_max, tau_max)
        
        # Global position update
        vx_global = V_vel * jnp.cos(theta + beta)
        vy_global = V_vel * jnp.sin(theta + beta)
        x_next = x_pos + vx_global * dt
        y_next = y_pos + vy_global * dt
        theta_next = theta + r_state * dt
        theta_next = jnp.mod(theta_next + jnp.pi, 2 * jnp.pi) - jnp.pi
        
        return jnp.array([x_next, y_next, theta_next, r_next, beta_next, V_next, delta_next, tau_next])
    
    # Policy function: call the appropriate static controller method
    def policy_fn(state: jnp.ndarray) -> jnp.ndarray:
        if policy_type == 'lane_change':
            return LaneChangeControllerJAX.compute(state, policy_params)
        else:  # stop
            return StoppingControllerJAX.compute(state, policy_params)
    
    def body_fn(carry, _):
        x = carry
        u = policy_fn(x)
        x_next = step_fn(x, u)
        return x_next, x_next
    
    _, trajectory = lax.scan(body_fn, x0, None, length=horizon)
    return jnp.vstack([x0[None, :], trajectory])


# JIT-compiled value function computation
@partial(jax.jit, static_argnums=(4, 5))
def _compute_value_jit(
    x0: jnp.ndarray,
    dynamics_params: dict,
    policy_params: dict,
    obstacles_array: jnp.ndarray,  # (n_obs, 3) array of [x, y, radius]
    policy_type: str,  # static: 'lane_change' or 'stop'
    horizon: int,      # static
    robot_radius: float,
    mu: float,
    dt: float,
) -> Tuple[float, jnp.ndarray]:
    """
    JIT-compiled value function computation.
    
    V(x0) = min_{t ∈ [0, T]} h(x(t))
    where h(x) = min_{obs} (dist_to_obs - combined_radius)
    h > 0 means SAFE
    """
    # Rollout trajectory
    trajectory = _rollout_trajectory(dynamics_params, policy_params, policy_type, 
                                      x0, horizon, mu, dt)
    
    # Compute h values along trajectory
    def compute_h_at_state(state: jnp.ndarray) -> float:
        """Compute barrier h(x) = min over obstacles of (dist - combined_radius)."""
        x, y = state[0], state[1]
        
        def h_single_obs(obs):
            obs_x, obs_y, obs_r = obs[0], obs[1], obs[2]
            combined_r = obs_r + robot_radius
            dist = jnp.sqrt((x - obs_x)**2 + (y - obs_y)**2 + 1e-8)
            return dist - combined_r  # Positive when safe
        
        # Compute h for all obstacles
        h_vals = jax.vmap(h_single_obs)(obstacles_array)
        # Take min over obstacles (most dangerous obstacle)
        return jnp.min(h_vals)
    
    # Compute h for all states in trajectory
    h_all = jax.vmap(compute_h_at_state)(trajectory)
    
    # V = min over time (smooth min for gradients)
    V = smooth_min(h_all, temperature=20.0)
    
    return V, trajectory


def compute_value_function(
    dynamics: DriftingCarDynamicsJAX,
    policy: Callable,
    x0: jnp.ndarray,
    obstacle_x: float,
    obstacle_y: float,
    obstacle_radius: float,
    robot_radius: float,
    horizon: int,
    mu: float = 1.0,
) -> Tuple[float, jnp.ndarray]:
    """
    Compute the value function V(x0) = min_{t ∈ [0, T]} h(x(t)).
    
    The barrier function h is the signed distance to obstacle.
    h(x) = ||[x, y] - [obs_x, obs_y]|| - (obstacle_radius + robot_radius)
    h(x) > 0 means SAFE (no collision)
    h(x) < 0 means UNSAFE (collision)
    
    Args:
        dynamics: DriftingCarDynamicsJAX instance
        policy: Backup policy function: state -> control
        x0: Initial state (8,)
        obstacle_x, obstacle_y: Obstacle center position
        obstacle_radius: Obstacle collision radius
        robot_radius: Robot collision radius
        horizon: Number of steps to rollout
        mu: Friction coefficient
        
    Returns:
        V: Value function (scalar) - min h over trajectory (positive = safe)
        trajectory: State trajectory (horizon+1, 8)
    """
    combined_radius = obstacle_radius + robot_radius
    
    def h_func(state):
        """Barrier function: POSITIVE means SAFE (no collision)."""
        x, y = state[0], state[1]
        dist_sq = (x - obstacle_x)**2 + (y - obstacle_y)**2
        dist = jnp.sqrt(dist_sq + 1e-8)
        return dist - combined_radius  # Positive when safe
    
    def body_fn(carry, _):
        x = carry
        u = policy(x)
        x_next = dynamics.step_full_state(x, u, mu)
        h_val = h_func(x_next)
        return x_next, (x_next, h_val)
    
    h0 = h_func(x0)
    _, (trajectory, h_values) = lax.scan(body_fn, x0, None, length=horizon)
    
    trajectory_full = jnp.vstack([x0[None, :], trajectory])
    h_all = jnp.concatenate([h0[None], h_values])
    
    # Value function is the MINIMUM h over the trajectory (most dangerous point)
    # Use smooth min for better gradients
    V = smooth_min(h_all, temperature=20.0)
    
    return V, trajectory_full


def compute_value_function_multi_obs(
    dynamics: DriftingCarDynamicsJAX,
    policy: Callable,
    x0: jnp.ndarray,
    obstacles: list,  # List of (x, y, radius)
    robot_radius: float,
    horizon: int,
    mu: float = 1.0,
) -> Tuple[float, jnp.ndarray]:
    """
    Compute value function with multiple obstacles.
    
    V(x0) = min_{t} min_{obs} h_obs(x(t))
    where h > 0 means SAFE
    """
    def h_func(state):
        """Barrier function for all obstacles. POSITIVE means SAFE."""
        x, y = state[0], state[1]
        h_vals = []
        for obs_x, obs_y, obs_radius in obstacles:
            combined_radius = obs_radius + robot_radius
            dist_sq = (x - obs_x)**2 + (y - obs_y)**2
            dist = jnp.sqrt(dist_sq + 1e-8)
            h_vals.append(dist - combined_radius)  # Positive when safe
        # Take min over obstacles (most dangerous)
        return jnp.min(jnp.array(h_vals))
    
    def body_fn(carry, _):
        x = carry
        u = policy(x)
        x_next = dynamics.step_full_state(x, u, mu)
        h_val = h_func(x_next)
        return x_next, (x_next, h_val)
    
    h0 = h_func(x0)
    _, (trajectory, h_values) = lax.scan(body_fn, x0, None, length=horizon)
    
    trajectory_full = jnp.vstack([x0[None, :], trajectory])
    h_all = jnp.concatenate([h0[None], h_values])
    # Take min over time
    V = smooth_min(h_all, temperature=20.0)
    
    return V, trajectory_full


# =============================================================================
# PCBF Controller
# =============================================================================

class PCBF:
    """
    Policy Control Barrier Function controller.
    
    Uses a backup policy to compute a value function, then enforces a CBF
    constraint via QP to ensure the system can always safely execute the
    backup policy.
    """
    
    def __init__(
        self,
        robot,
        robot_spec: dict,
        dt: float = 0.05,
        backup_horizon: float = 10.0,
        cbf_alpha: float = 1.0,
        ax=None,
    ):
        """
        Initialize the PCBF controller.
        
        Args:
            robot: Robot instance (DriftingCar)
            robot_spec: Robot specification dictionary
            dt: Time step
            backup_horizon: Horizon for backup policy rollout (seconds)
            cbf_alpha: CBF class-K function parameter
            ax: Matplotlib axis for visualization
        """
        self.robot = robot
        self.robot_spec = robot_spec
        self.dt = dt
        self.backup_horizon = backup_horizon
        self.backup_horizon_steps = int(backup_horizon / dt)
        # Shorter horizon for stopping (3 seconds) - keeps gradients well-conditioned
        self.stop_backup_horizon = 3.0
        self.stop_backup_horizon_steps = int(self.stop_backup_horizon / dt)
        self.cbf_alpha = cbf_alpha
        self.ax = ax
        
        # JAX dynamics
        self.dynamics_jax = DriftingCarDynamicsJAX(robot_spec, dt)
        
        # Backup controller (will be set externally)
        self.backup_controller = None
        self.backup_target = None
        self.backup_policy_jax = None
        
        # Environment (for obstacle info)
        self.env = None
        self.obstacles = []  # List of (x, y, radius)
        
        # Control limits
        self.u_min = np.array([-robot_spec.get('delta_dot_max', np.deg2rad(15)),
                               -robot_spec.get('tau_dot_max', 8000.0)])
        self.u_max = np.array([robot_spec.get('delta_dot_max', np.deg2rad(15)),
                               robot_spec.get('tau_dot_max', 8000.0)])
        
        # Visualization
        self.backup_traj_line = None
        self.backup_trajs = []
        self.save_every_N = 1
        self.curr_step = 0
        
        # Current friction (updated per step)
        self.current_friction = robot_spec.get('mu', 1.0)
        
        # Status
        self.status = 'optimal'
        
        # Policy type (for JIT)
        self.policy_type = None  # Set when backup controller is set
        self.policy_params = None  # Parameters for JIT-compiled policy
        self.dynamics_params = None  # Parameters for JIT-compiled dynamics
        
        # Setup dynamics parameters for JIT
        self._setup_dynamics_params()
        
        # JIT-compiled functions (created on first use)
        self._jit_value_and_grad = None
        
        if ax is not None:
            self._setup_visualization()
    
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
    
    def _setup_visualization(self):
        """Setup visualization handles."""
        if self.ax is None:
            return
        
        # Backup trajectory line
        self.backup_traj_line, = self.ax.plot(
            [], [], '-', color='cyan', linewidth=2, alpha=0.8,
            label='PCBF backup rollout', zorder=18
        )
    
    def set_backup_controller(self, backup_controller, target=None):
        """
        Set the backup controller.
        
        Args:
            backup_controller: BackupController instance
            target: Target for the controller (e.g., target_y for lane change)
        """
        self.backup_controller = backup_controller
        self.backup_target = target
        
        # Create JAX-compatible policy (pure JAX implementation)
        # Detect controller type and create the appropriate JAX version
        controller_name = backup_controller.get_behavior_name() if hasattr(backup_controller, 'get_behavior_name') else ''
        
        if 'LaneChange' in controller_name:
            self.backup_policy_jax = LaneChangeControllerJAX(self.robot_spec, target_y=target)
            self.policy_type = 'lane_change'
            self.policy_params = LaneChangePolicyParams(
                target_y=float(target),
                Kp_y=0.15,
                Kp_theta=1.5,
                Kd_theta=0.3,
                Kp_delta=3.0,
                Kp_v=500.0,
                Kp_tau_dot=2.0,  # Torque rate gain (smooth control, gradient-safe)
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
                Kd_theta=0.5,
                Kp_delta=3.0,
                Kp_v=1000.0,
                Kp_tau_dot=2.0,  # Torque rate gain (smooth control, gradient-safe)
                delta_max=float(self.robot_spec.get('delta_max', np.deg2rad(20))),
                delta_dot_max=float(self.robot_spec.get('delta_dot_max', np.deg2rad(15))),
                tau_max=float(self.robot_spec.get('tau_max', 4000.0)),
                tau_dot_max=float(self.robot_spec.get('tau_dot_max', 8000.0)),
                stop_threshold=0.05,
                holding_torque=-100.0,
            )
        else:
            # Fallback to wrapper (won't work with gradients, but can still be used for simulation)
            self.backup_policy_jax = BackupPolicyJAX(backup_controller, target)
            self.policy_type = 'custom'
            self.policy_params = None
            print(f"Warning: Using BackupPolicyJAX wrapper for {controller_name}. Gradients may not work.")
        
        # Reset JIT cache when controller changes
        self._jit_value_and_grad = None
    
    def set_environment(self, env):
        """
        Set the environment for obstacle information.
        
        Args:
            env: DriftingEnv instance
        """
        self.env = env
        self._update_obstacles()
    
    def _update_obstacles(self):
        """Update obstacle list from environment."""
        if self.env is None:
            # Don't clear obstacles if manually set and no env
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
        self.dynamics_jax.mu = mu
    
    def _compute_value_and_grad(self, x0_jax: jnp.ndarray) -> Tuple[float, jnp.ndarray, jnp.ndarray]:
        """
        Compute value function V and its gradient ∇V using JIT-compiled functions.
        
        Args:
            x0_jax: Current state as JAX array (8,)
            
        Returns:
            V: Value function (scalar, positive = safe)
            grad_V: Gradient of V w.r.t. x (8,)
            trajectory: Backup rollout trajectory (horizon+1, 8)
        """
        robot_radius = self.robot_spec.get('radius', 1.5)
        
        if len(self.obstacles) == 0:
            # No obstacles - return a large POSITIVE value (safe)
            return 100.0, jnp.zeros(8), x0_jax[None, :]
        
        # Convert obstacles to JAX array
        obstacles_array = jnp.array(self.obstacles)  # (n_obs, 3)
        
        # Create JIT-compiled value_and_grad function if not exists
        if self._jit_value_and_grad is None:
            # Create the JIT-compiled function with static arguments
            @partial(jax.jit, static_argnums=(4, 5))
            def _value_fn_jit(x0, dynamics_params, policy_params, obstacles_arr, 
                             policy_type, horizon, robot_r, mu, dt):
                """JIT-compiled value function."""
                trajectory = _rollout_trajectory(
                    dynamics_params, policy_params, policy_type,
                    x0, horizon, mu, dt
                )
                
                # Compute h values along trajectory
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
            
            # Create value_and_grad wrapper
            def value_only(x0, dynamics_params, policy_params, obstacles_arr,
                          policy_type, horizon, robot_r, mu, dt):
                V, _ = _value_fn_jit(x0, dynamics_params, policy_params, obstacles_arr,
                                     policy_type, horizon, robot_r, mu, dt)
                return V
            
            self._jit_value_fn = _value_fn_jit
            self._jit_value_and_grad = jax.jit(
                jax.value_and_grad(value_only),
                static_argnums=(4, 5)
            )
        
        # Use shorter horizon for stopping backup to get well-conditioned gradients
        # The stopping dynamics are very sensitive to velocity, so a shorter horizon
        # prevents gradient explosion while still capturing the braking behavior
        if self.policy_type == 'stop':
            horizon_steps = self.stop_backup_horizon_steps
        else:
            horizon_steps = self.backup_horizon_steps
        
        # Call JIT-compiled functions
        V, grad_V = self._jit_value_and_grad(
            x0_jax,
            self.dynamics_params,
            self.policy_params,
            obstacles_array,
            self.policy_type,
            horizon_steps,
            robot_radius,
            self.current_friction,
            self.dt
        )
        
        # Get trajectory for visualization (use full horizon for better viz)
        _, trajectory = self._jit_value_fn(
            x0_jax,
            self.dynamics_params,
            self.policy_params,
            obstacles_array,
            self.policy_type,
            self.backup_horizon_steps,  # Full horizon for visualization
            robot_radius,
            self.current_friction,
            self.dt
        )
        
        return V, grad_V, trajectory
    
    def _solve_cbf_qp(
        self,
        u_nom: np.ndarray,
        V: float,
        grad_V: np.ndarray,
        f: np.ndarray,
        G: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the CBF-QP using CVXPY.
        
        With sign convention h > 0 = SAFE:
        
        min_u ||u - u_nom||^2
        s.t.  ∇V^T (f + Gu) + α V ≥ 0  (CBF constraint: V_dot ≥ -α V)
              u_min ≤ u ≤ u_max
        
        Args:
            u_nom: Nominal control (2,)
            V: Value function (scalar, positive = safe)
            grad_V: Gradient of V (8,)
            f: Drift dynamics (8,)
            G: Input matrix (8, 2)
            
        Returns:
            u_safe: Safe control (2,)
        """
        nu = 2
        
        # Decision variable
        u = cp.Variable(nu)
        
        # QP cost: ||u - u_nom||^2
        cost = cp.sum_squares(u - u_nom)
        
        # CBF constraint: ∇V^T (f + Gu) + α V ≥ 0
        # Rearranged: -∇V^T G u ≤ ∇V^T f + α V
        grad_V_G = grad_V @ G  # (2,)
        grad_V_f = grad_V @ f  # scalar
        
        cbf_rhs = grad_V_f + self.cbf_alpha * V
        
        constraints = [
            -grad_V_G @ u <= cbf_rhs + 1e-4,  # Small margin for numerical stability
            u >= self.u_min,
            u <= self.u_max,
        ]
        
        # Solve QP with OSQP
        try:
            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(solver=cp.OSQP, verbose=False, max_iter=10000,
                         eps_abs=1e-5, eps_rel=1e-5, polish=True)
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                self.status = 'optimal'
                return u.value
            elif problem.status == 'infeasible':
                self.status = 'infeasible'
                return np.clip(u_nom, self.u_min, self.u_max)
            else:
                self.status = problem.status
                return np.clip(u_nom, self.u_min, self.u_max)
                
        except Exception as e:
            self.status = 'error'
            return np.clip(u_nom, self.u_min, self.u_max)
    
    def solve_control_problem(
        self,
        robot_state: np.ndarray,
        control_ref: Optional[dict] = None,
        friction: Optional[float] = None,
    ) -> np.ndarray:
        """
        Main PCBF control loop - compute safe control input.
        
        Args:
            robot_state: Current robot state [x, y, theta, r, beta, V, delta, tau] (8x1 or 8,)
            control_ref: Dictionary with 'u_ref' key for nominal control (optional)
            friction: Current friction coefficient (optional)
            
        Returns:
            u_safe: Safe control input [delta_dot, tau_dot] (2x1)
        """
        # Handle state shape
        robot_state = np.array(robot_state).flatten()
        
        # Update friction if provided
        if friction is not None:
            self.set_friction(friction)
        
        # Get nominal control
        if control_ref is not None and 'u_ref' in control_ref:
            u_nom = np.array(control_ref['u_ref']).flatten()
        else:
            u_nom = np.zeros(2)
        
        # Update obstacles
        self._update_obstacles()
        
        # If no obstacles or no backup controller, return nominal
        if len(self.obstacles) == 0 or self.backup_policy_jax is None:
            self.status = 'optimal'
            return u_nom.reshape(-1, 1)
        
        # Convert to JAX array
        x0_jax = jnp.array(robot_state)
        
        # Compute value function and gradient
        try:
            V, grad_V, trajectory = self._compute_value_and_grad(x0_jax)
            V = float(V)
            grad_V = np.array(grad_V)
            trajectory = np.array(trajectory)
        except Exception as e:
            print(f"Value computation failed: {e}")
            self.status = 'error'
            return u_nom.reshape(-1, 1)
        
        # Check gradient magnitude and value function
        grad_norm = np.linalg.norm(grad_V)
        
        # Determine thresholds based on backup policy type
        if self.policy_type == 'stop':
            # For stopping backup: use normal CBF-QP but with appropriate thresholds
            v_threshold = 5.0  # Activate when backup trajectory gets close
            max_grad_norm = 50.0
        else:
            # For lane change: normal CBF-QP approach
            v_threshold = 2.0
            max_grad_norm = 50.0
        
        # If V is above threshold, backup trajectory is safe - use nominal
        if V > v_threshold:
            self.status = 'safe'
            self._update_visualization(trajectory)
            return u_nom.reshape(-1, 1)
        
        # Normalize gradient to max_grad_norm
        # This keeps the gradient direction but controls magnitude
        if grad_norm > max_grad_norm:
            grad_V = grad_V * (max_grad_norm / grad_norm)
        
        # Compute f and G at current state
        f = np.array(self.dynamics_jax.f_full(x0_jax, self.current_friction))
        G = np.array(self.dynamics_jax.g_full(x0_jax))
        
        # Solve CBF-QP
        u_safe = self._solve_cbf_qp(u_nom, V, grad_V, f, G)
        
        # Update visualization
        self._update_visualization(trajectory)
        
        return u_safe.reshape(-1, 1)
    
    def _update_visualization(self, trajectory: np.ndarray):
        """Update visualization with backup trajectory."""
        if self.ax is None:
            return
        
        # Store trajectory
        if self.curr_step % self.save_every_N == 0:
            self.backup_trajs.append(trajectory.copy())
        self.curr_step += 1
        
        # Update backup trajectory line
        if self.backup_traj_line is not None and trajectory is not None:
            self.backup_traj_line.set_data(trajectory[:, 0], trajectory[:, 1])
    
    def get_backup_trajectories(self):
        """Get stored backup trajectories for plotting."""
        return self.backup_trajs.copy()
    
    def clear_trajectories(self):
        """Clear stored backup trajectories."""
        self.backup_trajs.clear()
    
    def is_using_backup(self):
        """Check if PCBF constraint is active (always returns False for PCBF)."""
        # PCBF doesn't have an explicit backup mode like Gatekeeper
        return False
    
    def get_status(self):
        """Get current PCBF status."""
        return {
            'status': self.status,
            'using_backup': False,
            'cbf_alpha': self.cbf_alpha,
            'backup_horizon': self.backup_horizon,
            'num_obstacles': len(self.obstacles),
        }

