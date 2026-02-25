"""
Created on February 4th, 2026
@author: Taekyung Kim

@description:
JAX-compatible backup policies for Double Integrator PCBF/PLCBF.
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

class WaypointPolicyParams(NamedTuple):
    """Parameters for WaypointPolicy."""
    waypoints: jnp.ndarray  # Array of [x, y] waypoints
    v_max: float           # Maximum speed
    Kp: float              # Velocity error gain
    dist_threshold: float  # Distance to switch waypoints
    a_max: float           # Max acceleration
    current_wp_idx: int    # Starting waypoint index

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

class WaypointPolicyJAX:
    """
    Follows a series of waypoints. 
    """
    
    @staticmethod
    def compute(state: jnp.ndarray, params: WaypointPolicyParams) -> jnp.ndarray:
        pos = state[0:2]
        vel = state[2:4]
        
        # Simplified: Just target the current waypoint index.
        # Rolling out waypoint switching in JAX usually requires putting wp_idx into the state,
        # which would require changing DIDynamicsParams.
        # For a short backup rollout, targeting the "nominal" target waypoint is often sufficient.
        target = params.waypoints[params.current_wp_idx]
        
        dist = jnp.sqrt(jnp.sum((target - pos)**2) + 1e-8)
        
        # Direction
        v_des_dir = (target - pos) / (dist + 1e-6)
        
        # Braking distance logic (similar to nominal.py)
        # Use params.a_max for braking capability
        braking_speed = jnp.sqrt(2 * params.a_max * dist)
        speed = jnp.minimum(params.v_max, braking_speed)
        v_des = v_des_dir * speed
        
        acc = params.Kp * (v_des - vel)
        
        # Clip
        a_norm = jnp.sqrt(jnp.sum(acc**2) + 1e-8)
        scale = jnp.where(a_norm > params.a_max, params.a_max / a_norm, 1.0)
        
        return acc * scale
