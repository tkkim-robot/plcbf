import jax.numpy as jnp
from .hj_minimal import ControlAndDisturbanceAffineDynamics, Box

class DoubleIntegratorHJ(ControlAndDisturbanceAffineDynamics):
    def __init__(self,
                 x=[0, 0, 0, 0],
                 u_min=[-1, -1],
                 u_max=[1, 1],
                 d_min=[0, 0],
                 d_max=[0, 0],
                 u_mode="min",
                 d_mode="max",
                 control_space="box", # Legacy, we override
                 mu=1.0,
                 sidewind=0.0):
        # Reconstruct Boxes for parent init
        if control_space == "box":
             control_space = Box(jnp.array(u_min), jnp.array(u_max))
        
        disturbance_space = Box(jnp.array(d_min), jnp.array(d_max))
        
        super().__init__(u_mode, d_mode, control_space, disturbance_space)
        self.mu = mu
        self.sidewind = sidewind
        # Effective a_max for Circle Limit (assuming square box input so u_max[0] == a_max)
        self.a_max_val = u_max[0] 

    def open_loop_dynamics(self, state, time):
        # state: [x, y, vx, vy]
        # x_dot = [vx, vy, 0, sidewind] (Open loop means u=0?)
        # Affine form: x_dot = f(x) + g(x)u
        # open_loop_dynamics usually returns f(x).
        return jnp.array([state[2], state[3], 0., self.sidewind])

    def control_jacobian(self, state, time):
        # g(x)
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

    def hamiltonian(self, state, t, value, grad_value):
        """
        Hamiltonian H = max_u min_d p^T (f(x) + g(x)u + d)
                      = p^T f(x) + max_u (p^T g(x) u) + min_d (...)
        """
        # grad_value is p. Shape (4, ...)
        
        # 1. f(x) term
        # f = [vx, vy, 0, sidewind]
        # p^T f = p[0]*vx + p[1]*vy + p[3]*sidewind
        
        vx = state[2]
        vy = state[3]
        p0 = grad_value[0]
        p1 = grad_value[1]
        p2 = grad_value[2]
        p3 = grad_value[3]
        
        ham_f = p0 * vx + p1 * vy + p3 * self.sidewind
        
        # 2. Control term (Box Saturation per Axis)
        # ham_u = p^T u
        # Box: |ax| <= mu * a_max, |ay| <= mu * a_max
        # Maximize: |p2|*limit + |p3|*limit
        
        limit = self.mu * self.a_max_val
        ham_u = (jnp.abs(p2) + jnp.abs(p3)) * limit
        
        if self.control_mode == "min":
            ham_u = -ham_u
            
        return ham_f + ham_u

    def dynamics(self, state, control, disturbance):
        # Fallback/Debug mainly, solve computation uses Hamiltonian directly usually
        # But for trajectory simulation using HJ dynamics class?
        # Typically simulation uses Python class.
        # Implemening for completeness.
        
        u = control
        # Saturate u?
        # If this function is called, u should already be optimized.
        # But we assume u respects constraints.
        
        dx0 = state[2]
        dx1 = state[3]
        dx2 = u[0]
        dx3 = u[1] + self.sidewind
        return jnp.stack([dx0, dx1, dx2, dx3], axis=0)

