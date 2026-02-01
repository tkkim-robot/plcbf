from safe_control.robots.double_integrator2D import DoubleIntegrator2D
import numpy as np
import casadi as ca

class DoubleIntegratorSim(DoubleIntegrator2D):
    def __init__(self, dt, robot_spec):
        super().__init__(dt, robot_spec)
        self.mu = robot_spec.get('mu', 1.0)
        self.sidewind = robot_spec.get('sidewind', 0.0)
        # Ensure a_max is available (redundant if parent works)
        if not hasattr(self, 'a_max'):
             self.a_max = robot_spec.get('a_max', 1.0)
             print("DEBUG SIM: a_max was missing, manually set.")
        
    def step(self, X, U):
        # 1. Input Saturation (Friction)
        limit = self.mu * self.a_max
        # U is (2, 1) numpy array
        # Per-axis clipping (Box Constraint)
        U = np.clip(U, -limit, limit)
        
        return super().step(X, U)

    def f(self, X, casadi=False):
        res = super().f(X, casadi)
        
        if casadi:
            start_row = 3 # 4th element
            # Add sidewind
            # Since res is likely (4,1) dense matrix or SX
            # We construct vector to add
            wind_vec = ca.vertcat(0, 0, 0, self.sidewind)
            return res + wind_vec
        else:
            # Numpy (4, 1) or (4,)
            if res.ndim > 1:
                res[3, 0] += self.sidewind
            else:
                res[3] += self.sidewind
            return res
