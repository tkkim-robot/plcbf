"""
Created on February 4th, 2026
@author: Taekyung Kim

@description:
JAX-compatible backup policies for Double Integrator PCBF/MPCBF.
"""

from typing import NamedTuple
import jax.numpy as jnp
import jax

class AnglePolicyParams(NamedTuple):
    """Parameters for AnglePolicy."""
    target_angle: float    # Direction to move towards
    target_speed: float    # Desired speed in that direction
    Kp_v: float            # Proportional gain for velocity error
    a_max: float           # Maximum acceleration (norm)

class StopPolicyParams(NamedTuple):
    """Parameters for StopPolicy."""
    Kp_v: float            # Proportional gain for stopping
    a_max: float           # Maximum acceleration (norm)
    stop_threshold: float  # Velocity threshold to consider stopped

class AnglePolicyJAX:
    """
    Accelerates towards a target velocity vector defined by (speed, angle).
    u = Kp * (v_des - v)
    """
    
    def __init__(self, robot_spec: dict, target_angle: float, target_speed: float = 1.0):
        self.target_angle = target_angle
        self.target_speed = target_speed
        self.Kp_v = 4.0  # Tunable gain
        self.a_max = float(robot_spec.get('a_max', 5.0))
        
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        params = AnglePolicyParams(
            target_angle=self.target_angle,
            target_speed=self.target_speed,
            Kp_v=self.Kp_v,
            a_max=self.a_max
        )
        return AnglePolicyJAX.compute(state, params)
        
    @staticmethod
    def compute(state: jnp.ndarray, params: AnglePolicyParams) -> jnp.ndarray:
        # State: [x, y, vx, vy]
        vx, vy = state[2], state[3]
        
        # Desired velocity vector
        vx_des = params.target_speed * jnp.cos(params.target_angle)
        vy_des = params.target_speed * jnp.sin(params.target_angle)
        
        # Error
        ex = vx_des - vx
        ey = vy_des - vy
        
        # P-Control
        ax = params.Kp_v * ex
        ay = params.Kp_v * ey
        
        # Clip to a_max (ball constraint)
        a_norm = jnp.sqrt(ax**2 + ay**2 + 1e-8)
        scale = jnp.where(a_norm > params.a_max, params.a_max / a_norm, 1.0)
        
        ax = ax * scale
        ay = ay * scale
        
        return jnp.array([ax, ay])

class StopPolicyJAX:
    """
    Brakes to zero velocity.
    """
    
    def __init__(self, robot_spec: dict):
        self.Kp_v = 4.0
        self.a_max = float(robot_spec.get('a_max', 5.0))
        self.stop_threshold = 0.05
        
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        params = StopPolicyParams(
            Kp_v=self.Kp_v,
            a_max=self.a_max,
            stop_threshold=self.stop_threshold
        )
        return StopPolicyJAX.compute(state, params)
        
    @staticmethod
    def compute(state: jnp.ndarray, params: StopPolicyParams) -> jnp.ndarray:
        vx, vy = state[2], state[3]
        
        # Desired = 0
        ax = -params.Kp_v * vx
        ay = -params.Kp_v * vy
        
        # Clip
        a_norm = jnp.sqrt(ax**2 + ay**2 + 1e-8)
        scale = jnp.where(a_norm > params.a_max, params.a_max / a_norm, 1.0)
        
        return jnp.array([ax * scale, ay * scale])
