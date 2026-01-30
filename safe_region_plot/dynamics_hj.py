import jax.numpy as jnp
from .hj_minimal import ControlAndDisturbanceAffineDynamics, Box

class DoubleIntegratorHJ(ControlAndDisturbanceAffineDynamics):
    def __init__(self, a_max=1.0, 
                 control_mode="max", 
                 disturbance_mode="min", 
                 control_space=None, 
                 disturbance_space=None):
        if control_space is None:
            control_space = Box(jnp.array([-a_max, -a_max]), jnp.array([a_max, a_max]))
        if disturbance_space is None:
            disturbance_space = Box(jnp.array([0., 0.]), jnp.array([0., 0.]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        return jnp.array([state[2], state[3], 0., 0.])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0., 0.],
            [0., 0.],
            [1., 0.],
            [0., 1.]
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]
        ])
