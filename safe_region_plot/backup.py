import numpy as np

class StopBackupController:
    def __init__(self, a_max=1.0, k_v=5.0):
        self.a_max = a_max
        self.k_v = k_v

    def compute_control(self, state, target=None):
        # state is [x, y, vx, vy]
        state = state.flatten()
        v = state[2:4]
        u = -self.k_v * v
        # Clamp to max acceleration
        u_mag = np.linalg.norm(u)
        if u_mag > self.a_max:
            u = u * self.a_max / u_mag
        return u.reshape(-1, 1)

class TurnBackupController:
    def __init__(self, a_max=1.0, k_v=5.0, decision_y=0.0):
        self.a_max = a_max
        self.k_v = k_v
        self.decision_y = decision_y

    def compute_control(self, state, target=None):
        # state is [x, y, vx, vy]
        state = state.flatten()
        y = state[1]
        vx = state[2]
        
        # Decide direction based on decision boundary
        # If target (obstacle y) is provided, it can override decision_y if needed
        dy = target if target is not None else self.decision_y
        
        if y >= dy:
            ay = self.a_max
        else:
            ay = -self.a_max
            
        # Stop x velocity
        ax = -self.k_v * vx
        # Ensure we don't exceed total a_max if possible, or just clamp components
        # Here we follow "maximum acceleration of the y, while a P controller try to decelerate the x"
        # Since u = [ax, ay], and typically |ax| <= a_max, |ay| <= a_max in 2D DI
        # But if total is a_max:
        if abs(ax) > self.a_max:
             ax = np.sign(ax) * self.a_max
             
        # Re-check total magnitude if robot spec is norm based
        # For simplicity, we use box constraints if not specified.
        return np.array([ax, ay]).reshape(-1, 1)
