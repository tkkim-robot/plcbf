"""
Drifting car controllers/policies (JAX implementation).
"""

from typing import NamedTuple
import numpy as np
import jax.numpy as jnp


class LaneChangePolicyParams(NamedTuple):
    """Lane change policy parameters for JAX PyTree compatibility."""
    target_y: float
    Kp_y: float
    Kp_theta: float
    Kd_theta: float
    Kp_delta: float
    Kp_v: float
    Kp_tau_dot: float
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
    Kp_tau_dot: float
    delta_max: float
    delta_dot_max: float
    tau_max: float
    tau_dot_max: float
    stop_threshold: float
    holding_torque: float


class LaneChangeControllerJAX:
    """Pure JAX implementation of lane change controller for PCBF."""
    
    def __init__(self, robot_spec: dict, target_y: float = 4.0):
        self.target_y = target_y
        self.Kp_y = 0.15
        self.Kp_theta = 1.5
        self.Kd_theta = 0.3
        self.Kp_delta = 3.0
        self.Kp_v = 500.0
        self.Kp_tau_dot = 2.0
        self.target_velocity = robot_spec.get('v_ref', 8.0)
        self.delta_max = robot_spec.get('delta_max', np.deg2rad(20))
        self.delta_dot_max = robot_spec.get('delta_dot_max', np.deg2rad(15))
        self.tau_max = robot_spec.get('tau_max', 4000.0)
        self.tau_dot_max = robot_spec.get('tau_dot_max', 8000.0)
        self.theta_des_max = np.deg2rad(15)
    
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        params = LaneChangePolicyParams(
            target_y=self.target_y, Kp_y=self.Kp_y, Kp_theta=self.Kp_theta,
            Kd_theta=self.Kd_theta, Kp_delta=self.Kp_delta, Kp_v=self.Kp_v,
            Kp_tau_dot=self.Kp_tau_dot, target_velocity=self.target_velocity,
            delta_max=self.delta_max, delta_dot_max=self.delta_dot_max,
            tau_max=self.tau_max, tau_dot_max=self.tau_dot_max,
            theta_des_max=self.theta_des_max,
        )
        return LaneChangeControllerJAX.compute(state, params)
    
    @staticmethod
    def compute(state: jnp.ndarray, params: LaneChangePolicyParams) -> jnp.ndarray:
        y, theta, r = state[1], state[2], state[3]
        V = jnp.maximum(state[5], 0.1)
        delta, tau = state[6], state[7]
        
        # Outer loop: Lateral position control
        y_error = params.target_y - y
        theta_des = jnp.arctan(params.Kp_y * y_error)
        theta_des = jnp.clip(theta_des, -params.theta_des_max, params.theta_des_max)
        
        # Inner loop: Heading control
        theta_error = jnp.mod(theta_des - theta + jnp.pi, 2 * jnp.pi) - jnp.pi
        delta_des = params.Kp_theta * theta_error - params.Kd_theta * r
        delta_des = jnp.clip(delta_des, -params.delta_max, params.delta_max)
        
        # Steering rate control
        delta_error = delta_des - delta
        delta_dot = params.Kp_delta * delta_error
        delta_dot = jnp.clip(delta_dot, -params.delta_dot_max, params.delta_dot_max)
        
        # Velocity control
        V_error = params.target_velocity - V
        tau_des = params.Kp_v * V_error
        tau_des = jnp.clip(tau_des, -params.tau_max, params.tau_max)
        
        # Torque rate control
        tau_error = tau_des - tau
        tau_dot = params.Kp_tau_dot * tau_error
        tau_dot = jnp.clip(tau_dot, -params.tau_dot_max, params.tau_dot_max)
        
        return jnp.array([delta_dot, tau_dot])


class StoppingControllerJAX:
    """Pure JAX implementation of stopping controller for PCBF."""
    
    def __init__(self, robot_spec: dict):
        self.Kp_v = 1000.0
        self.Kp_tau_dot = 2.0
        self.Kd_theta = 0.5
        self.Kp_delta = 3.0
        self.delta_max = robot_spec.get('delta_max', np.deg2rad(20))
        self.delta_dot_max = robot_spec.get('delta_dot_max', np.deg2rad(15))
        self.tau_max = robot_spec.get('tau_max', 4000.0)
        self.tau_dot_max = robot_spec.get('tau_dot_max', 8000.0)
        self.stop_velocity_threshold = 0.05
        self.holding_torque = -100.0
    
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        params = StopPolicyParams(
            Kd_theta=self.Kd_theta, Kp_delta=self.Kp_delta,
            Kp_v=self.Kp_v, Kp_tau_dot=self.Kp_tau_dot,
            delta_max=self.delta_max, delta_dot_max=self.delta_dot_max,
            tau_max=self.tau_max, tau_dot_max=self.tau_dot_max,
            stop_threshold=self.stop_velocity_threshold,
            holding_torque=self.holding_torque,
        )
        return StoppingControllerJAX.compute(state, params)
    
    @staticmethod
    def compute(state: jnp.ndarray, params: StopPolicyParams) -> jnp.ndarray:
        r, V, delta, tau = state[3], state[5], state[6], state[7]
        
        # Braking control
        tau_braking = -params.Kp_v * V + params.holding_torque
        tau_des = jnp.clip(tau_braking, -params.tau_max, params.tau_max)
        tau_error = tau_des - tau
        tau_dot = params.Kp_tau_dot * tau_error
        tau_dot = jnp.clip(tau_dot, -params.tau_dot_max, params.tau_dot_max)
        
        # Steering control
        delta_des = -params.Kd_theta * r
        delta_des = jnp.clip(delta_des, -params.delta_max, params.delta_max)
        delta_error = delta_des - delta
        delta_dot = params.Kp_delta * delta_error
        delta_dot = jnp.clip(delta_dot, -params.delta_dot_max, params.delta_dot_max)
        
        return jnp.array([delta_dot, tau_dot])


class BackupPolicyJAX:
    """Wrapper for backward compatibility with non-JAX controllers."""
    
    def __init__(self, backup_controller, target=None):
        self.backup_controller = backup_controller
        self.target = target
    
    def compute_control_numpy(self, state_np):
        return self.backup_controller.compute_control(state_np, self.target)
