"""
Created on February 4th, 2026
@author: Taekyung Kim

@description:
JAX-compatible Double Integrator dynamics for PCBF/MPCBF.

State: [x, y, vx, vy] (4 states)
Control: [ax, ay] (2 controls)

Matches interface of DriftingCarDynamicsJAX in pcbf.py.
"""

from typing import NamedTuple, Optional
import jax.numpy as jnp
import numpy as np

class DIDynamicsParams(NamedTuple):
    """Dynamics parameters for Double Integrator (JAX PyTree compatibility)."""
    # Limits
    v_max: float       # Max velocity
    a_max: float       # Max acceleration (per axis or norm?) - typically per axis in box constraints
    ax_max: float      # Max x-acceleration
    ay_max: float      # Max y-acceleration

class DoubleIntegratorDynamicsJAX:
    """
    JAX-compatible Double Integrator dynamics.
    
    State: [x, y, vx, vy]
    Control: [ax, ay]
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
        
        # Limits
        self.v_max = float(robot_spec.get('v_max', 10.0))
        self.a_max = float(robot_spec.get('a_max', 5.0))
        self.ax_max = float(robot_spec.get('ax_max', self.a_max))
        self.ay_max = float(robot_spec.get('ay_max', self.a_max))
        
    def f_full(self, x_full, mu=None):
        """
        Compute dynamics f(x) for full state (for CBF constraint).
        x_dot = f(x) + g(x)u
        
        Args:
            x_full: Full state [x, y, vx, vy] (4,)
            mu: Friction coefficient (unused for DI)
            
        Returns:
            f(x): Drift term (4,)
        """
        # x_dot = vx
        # y_dot = vy
        # vx_dot = 0
        # vy_dot = 0
        return jnp.array([x_full[2], x_full[3], 0.0, 0.0])
    
    def g_full(self, x_full):
        """
        Compute input matrix g(x) for full state.
        
        Args:
            x_full: Full state [x, y, vx, vy] (4,)
            
        Returns:
            g(x): Input matrix (4, 2)
        """
        # x_dot terms are 0
        # vx_dot = ax
        # vy_dot = ay
        return jnp.array([
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
    def step_full_state(self, x_full, u, mu=None):
        """
        Step the state forward by dt.
        
        Args:
            x_full: Full state [x, y, vx, vy] (4,)
            u: Control [ax, ay] (2,)
            mu: Friction (unused)
            
        Returns:
            x_next: Next state (4,)
        """
        x, y, vx, vy = x_full[0], x_full[1], x_full[2], x_full[3]
        ax, ay = u[0], u[1]
        
        # Euler integration
        x_next = x + vx * self.dt
        y_next = y + vy * self.dt
        
        vx_next = vx + ax * self.dt
        vy_next = vy + ay * self.dt
        
        # Velocity clamping (optional, but good for safety)
        # Assuming simple box clamping or norm clamping? 
        # DoubleIntegrator2D.py does norm clamping in `step` if v_max exists.
        
        v_sq = vx_next**2 + vy_next**2
        v_norm = jnp.sqrt(v_sq + 1e-8)
        
        scale = jnp.where(v_norm > self.v_max, self.v_max / v_norm, 1.0)
        vx_next = vx_next * scale
        vy_next = vy_next * scale
            
        return jnp.array([x_next, y_next, vx_next, vy_next])

