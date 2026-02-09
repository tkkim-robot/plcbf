"""
Drifting car dynamics (JAX implementation).

State: [x, y, theta, r, beta, V, delta, tau] (8 states)
Control: [delta_dot, tau_dot] (2 controls)
"""

from typing import NamedTuple
import numpy as np
import jax.numpy as jnp


class DynamicsParams(NamedTuple):
    """Dynamics parameters as a NamedTuple for JAX PyTree compatibility."""
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


def angle_normalize_jax(x):
    """Normalize angle to [-pi, pi] using JAX."""
    return jnp.mod(x + jnp.pi, 2 * jnp.pi) - jnp.pi


class DriftingCarDynamicsJAX:
    """
    JAX-compatible drifting car dynamics for PCBF.
    
    State: [x, y, theta, r, beta, V, delta, tau] (8 states)
    Control: [delta_dot, tau_dot] (2 controls)
    """
    
    def __init__(self, robot_spec: dict, dt: float):
        self.robot_spec = robot_spec
        self.dt = dt
        self.gravity = 9.81
        
        # Extract parameters
        self.a = robot_spec.get('a', 1.4)
        self.b = robot_spec.get('b', 1.4)
        self.m = robot_spec.get('m', 2500.0)
        self.Iz = robot_spec.get('Iz', 5000.0)
        self.Cc_f = robot_spec.get('Cc_f', 80000.0)
        self.Cc_r = robot_spec.get('Cc_r', 100000.0)
        self.r_w = robot_spec.get('r_w', 0.35)
        self.gamma = robot_spec.get('gamma', 0.95)
        
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
        
        # Default friction
        self.mu = robot_spec.get('mu', 1.0)
    
    def _compute_slip_angles(self, r, beta, V, delta):
        """Compute front and rear slip angles."""
        V_safe = jnp.maximum(V, 0.1)
        alpha_f = jnp.arctan2(V * jnp.sin(beta) + self.a * r, V_safe * jnp.cos(beta)) - delta
        alpha_r = jnp.arctan2(V * jnp.sin(beta) - self.b * r, V_safe * jnp.cos(beta))
        return alpha_f, alpha_r
    
    def _compute_lateral_force(self, alpha, Cc, Fz, Fx, mu):
        """Compute lateral tire force using Fiala brush model."""
        Fy_max = jnp.sqrt(jnp.maximum((mu * Fz)**2 - self.gamma * Fx**2, 1.0))
        alpha_sl = jnp.arctan(3 * Fy_max / Cc)
        tan_alpha = jnp.tan(alpha)
        Fy_linear = (-Cc * tan_alpha 
                    + (Cc**2 / (3 * Fy_max)) * jnp.abs(tan_alpha) * tan_alpha
                    - (Cc**3 / (27 * Fy_max**2)) * tan_alpha**3)
        Fy_saturated = -Fy_max * jnp.sign(alpha)
        return jnp.where(jnp.abs(alpha) < alpha_sl, Fy_linear, Fy_saturated)
    
    def _compute_longitudinal_force(self, tau, Fz, mu):
        """Compute longitudinal tire force with soft saturation."""
        F_lim = mu * Fz
        F_lim_safe = jnp.maximum(F_lim, 1.0)
        return F_lim * jnp.tanh(tau / (self.r_w * F_lim_safe))
    
    def _compute_tire_forces(self, r, beta, V, delta, tau, mu):
        """Compute all tire forces."""
        alpha_f, alpha_r = self._compute_slip_angles(r, beta, V, delta)
        Fx_f = 0.0
        Fx_r = self._compute_longitudinal_force(tau, self.Fz_r, mu)
        Fy_f = self._compute_lateral_force(alpha_f, self.Cc_f, self.Fz_f, Fx_f, mu)
        Fy_r = self._compute_lateral_force(alpha_r, self.Cc_r, self.Fz_r, Fx_r, mu)
        return Fx_f, Fy_f, Fx_r, Fy_r
    
    def f_dyn(self, x_dyn, mu=None):
        """Compute drift dynamics f(x) for dynamics state."""
        if mu is None:
            mu = self.mu
        r, beta, V, delta, tau = x_dyn[0], x_dyn[1], x_dyn[2], x_dyn[3], x_dyn[4]
        Fx_f, Fy_f, Fx_r, Fy_r = self._compute_tire_forces(r, beta, V, delta, tau, mu)
        V_safe = jnp.maximum(V, 0.1)
        r_dot = (self.a * (Fx_f * jnp.sin(delta) + Fy_f * jnp.cos(delta)) - self.b * Fy_r) / self.Iz
        beta_dot = ((Fx_f * jnp.sin(delta - beta) + Fy_f * jnp.cos(delta - beta)
                    - Fx_r * jnp.sin(beta) + Fy_r * jnp.cos(beta)) / (self.m * V_safe) - r)
        V_dot = ((Fx_f * jnp.cos(delta - beta) - Fy_f * jnp.sin(delta - beta)
                 + Fx_r * jnp.cos(beta) + Fy_r * jnp.sin(beta)) / self.m)
        return jnp.array([r_dot, beta_dot, V_dot, 0.0, 0.0])
    
    def g_dyn(self, x_dyn):
        """Compute input matrix g(x) for dynamics state."""
        g = jnp.zeros((5, 2))
        g = g.at[3, 0].set(1.0)  # delta_dot
        g = g.at[4, 1].set(1.0)  # tau_dot
        return g
    
    def step_full_state(self, x_full, u, mu=None):
        """Step the full state forward by dt using Euler integration."""
        if mu is None:
            mu = self.mu
        x_pos, y_pos, theta = x_full[0], x_full[1], x_full[2]
        x_dyn = x_full[3:8]
        f = self.f_dyn(x_dyn, mu)
        g = self.g_dyn(x_dyn)
        x_dyn_next = x_dyn + (f + g @ u) * self.dt
        r_next = jnp.clip(x_dyn_next[0], -self.r_max, self.r_max)
        beta_next = jnp.clip(x_dyn_next[1], -self.beta_max, self.beta_max)
        V_next = jnp.clip(x_dyn_next[2], self.v_min, self.v_max)
        delta_next = jnp.clip(x_dyn_next[3], -self.delta_max, self.delta_max)
        tau_next = jnp.clip(x_dyn_next[4], -self.tau_max, self.tau_max)
        V, beta, r = x_dyn[2], x_dyn[1], x_dyn[0]
        vx_global = V * jnp.cos(theta + beta)
        vy_global = V * jnp.sin(theta + beta)
        x_next = x_pos + vx_global * self.dt
        y_next = y_pos + vy_global * self.dt
        theta_next = angle_normalize_jax(theta + r * self.dt)
        return jnp.array([x_next, y_next, theta_next, r_next, beta_next, V_next, delta_next, tau_next])
    
    def f_full(self, x_full, mu=None):
        """Compute drift dynamics f(x) for full state (for CBF constraint)."""
        if mu is None:
            mu = self.mu
        theta = x_full[2]
        x_dyn = x_full[3:8]
        f_dyn = self.f_dyn(x_dyn, mu)
        V, beta, r = x_dyn[2], x_dyn[1], x_dyn[0]
        vx_global = V * jnp.cos(theta + beta)
        vy_global = V * jnp.sin(theta + beta)
        return jnp.array([vx_global, vy_global, r, f_dyn[0], f_dyn[1], f_dyn[2], 0.0, 0.0])
    
    def g_full(self, x_full):
        """Compute input matrix g(x) for full state (for CBF constraint)."""
        g = jnp.zeros((8, 2))
        g = g.at[6, 0].set(1.0)  # delta_dot
        g = g.at[7, 1].set(1.0)  # tau_dot
        return g


def step_full_state_pure(x_full, u, params: DynamicsParams, dt: float, mu: float):
    """
    Pure function version of step_full_state for JIT compilation.
    
    Args:
        x_full: State [x, y, theta, r, beta, V, delta, tau]
        u: Control [delta_dot, tau_dot]
        params: DynamicsParams NamedTuple
        dt: Time step
        mu: Friction coefficient
    """
    # Extract parameters
    a = params.a
    b = params.b
    m = params.m
    Iz = params.Iz
    Cc_f = params.Cc_f
    Cc_r = params.Cc_r
    r_w = params.r_w
    gamma = params.gamma
    
    # Limits
    delta_max = params.delta_max
    tau_max = params.tau_max
    v_max = params.v_max
    v_min = params.v_min
    r_max = params.r_max
    beta_max = params.beta_max
    
    gravity = 9.81
    
    # Normal forces
    L = a + b
    Fz_f = m * gravity * b / L
    Fz_r = m * gravity * a / L
    
    # State extraction
    x_pos, y_pos, theta = x_full[0], x_full[1], x_full[2]
    r_state, beta, V_vel, delta, tau = x_full[3], x_full[4], x_full[5], x_full[6], x_full[7]
    
    # Tire forces
    V_safe = jnp.maximum(V_vel, 0.1)
    
    # Slip angles
    alpha_f = jnp.arctan2(V_vel * jnp.sin(beta) + a * r_state, V_safe * jnp.cos(beta)) - delta
    alpha_r = jnp.arctan2(V_vel * jnp.sin(beta) - b * r_state, V_safe * jnp.cos(beta))
    
    # Longitudinal force (rear)
    F_lim_r = mu * Fz_r
    # Soft saturation
    F_lim_safe = jnp.maximum(F_lim_r, 1.0)
    Fx_r = F_lim_r * jnp.tanh(tau / (r_w * F_lim_safe))
    Fx_f = 0.0
    
    # Lateral forces (Fiala)
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
    
    # Update
    # x_dyn = [r, beta, V, delta, tau]
    # x_dyn_next = x_dyn + (f + g*u) * dt
    
    r_next = jnp.clip(r_state + r_dot * dt, -r_max, r_max)
    beta_next = jnp.clip(beta + beta_dot * dt, -beta_max, beta_max)
    V_next = jnp.clip(V_vel + V_dot * dt, v_min, v_max)
    delta_next = jnp.clip(delta + u[0] * dt, -delta_max, delta_max)
    tau_next = jnp.clip(tau + u[1] * dt, -tau_max, tau_max)
    
    vx_global = V_vel * jnp.cos(theta + beta)
    vy_global = V_vel * jnp.sin(theta + beta)
    
    x_next = x_pos + vx_global * dt
    y_next = y_pos + vy_global * dt
    theta_next = angle_normalize_jax(theta + r_state * dt)
    
    return jnp.array([x_next, y_next, theta_next, r_next, beta_next, V_next, delta_next, tau_next])
