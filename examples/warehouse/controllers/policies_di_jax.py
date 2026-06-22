"""
Created on February 4th, 2026
@author: Taekyung Kim

@description:
JAX-compatible backup policies for Double Integrator PCBF/PLCBF.
"""

from typing import NamedTuple
import jax.numpy as jnp

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


class RetracePolicyParams(NamedTuple):
    """Parameters for retracing the nominal waypoint path backwards."""
    waypoints: jnp.ndarray  # Array of [x, y] waypoints
    v_max: float            # Maximum speed
    Kp: float               # Velocity error gain
    dist_threshold: float   # Distance to switch retrace waypoints
    a_max: float            # Max acceleration
    current_wp_idx: int     # Starting retrace waypoint index


def _clip_accel(acc: jnp.ndarray, a_max: float) -> jnp.ndarray:
    """Clip 2D acceleration by Euclidean norm."""
    a_norm = jnp.sqrt(jnp.sum(acc**2) + 1e-8)
    scale = jnp.where(a_norm > a_max, a_max / a_norm, 1.0)
    return acc * scale

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
        
        return _clip_accel(jnp.array([ax, ay]), params.a_max)

class WaypointPolicyParams(NamedTuple):
    """Parameters for WaypointPolicy."""
    waypoints: jnp.ndarray  # Array of [x, y] waypoints
    v_max: float           # Maximum speed
    Kp: float              # Velocity error gain
    K_lat: float           # Lateral path correction gain
    v_lat_max: float       # Lateral correction velocity limit
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
        
        return _clip_accel(jnp.array([ax, ay]), params.a_max)

class WaypointPolicyJAX:
    """
    Follows a series of waypoints. 
    """
    
    @staticmethod
    def compute(state: jnp.ndarray, params: WaypointPolicyParams) -> jnp.ndarray:
        pos = state[0:2]
        vel = state[2:4]
        
        target = params.waypoints[params.current_wp_idx]
        prev_idx = jnp.maximum(params.current_wp_idx - 1, 0)
        prev = params.waypoints[prev_idx]
        seg = target - prev
        seg_norm = jnp.sqrt(jnp.sum(seg**2) + 1e-8)
        target_dist = jnp.sqrt(jnp.sum((target - pos)**2) + 1e-8)
        seg_dir = jnp.where(seg_norm > 1e-6, seg / seg_norm, (target - pos) / (target_dist + 1e-6))
        perp_dir = jnp.array([-seg_dir[1], seg_dir[0]])

        dist_along = jnp.dot(target - pos, seg_dir)
        braking_speed = jnp.sqrt(2.0 * params.a_max * jnp.abs(dist_along))
        v_long = jnp.minimum(params.v_max, braking_speed)
        v_long_dir = jnp.where(dist_along >= 0.0, 1.0, -1.0)

        lat_err = jnp.dot(pos - prev, perp_dir)
        v_lat = -params.K_lat * lat_err
        v_lat = jnp.clip(v_lat, -params.v_lat_max, params.v_lat_max)

        v_des = v_long_dir * v_long * seg_dir + v_lat * perp_dir
        v_des_norm = jnp.sqrt(jnp.sum(v_des**2) + 1e-8)
        v_des = jnp.where(v_des_norm > params.v_max, v_des * (params.v_max / v_des_norm), v_des)
        
        return _clip_accel(params.Kp * (v_des - vel), params.a_max)


class RetracePolicyJAX:
    """
    Follows a retrace waypoint target. Rollout-time waypoint switching is handled
    in the DI PCBF rollout so the index can be carried through JAX scan.
    """

    @staticmethod
    def compute(state: jnp.ndarray, params: RetracePolicyParams) -> jnp.ndarray:
        pos = state[0:2]
        vel = state[2:4]
        idx = jnp.clip(params.current_wp_idx, 0, params.waypoints.shape[0] - 1)
        target = params.waypoints[idx]

        dist = jnp.sqrt(jnp.sum((target - pos)**2) + 1e-8)
        v_des_dir = (target - pos) / (dist + 1e-6)
        braking_speed = jnp.sqrt(2.0 * params.a_max * jnp.maximum(dist, 0.0))
        speed = jnp.minimum(params.v_max, braking_speed)
        v_des = v_des_dir * speed

        return _clip_accel(params.Kp * (v_des - vel), params.a_max)
