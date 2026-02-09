"""Controllers for the Inventory scenario."""

import numpy as np


class WaypointFollower:
    """Simple P-Controller for nominal path following."""
    def __init__(self, waypoints, v_max=5.0, Kp=4.0, debug=False):
        self.waypoints = waypoints
        self.v_max = v_max
        self.Kp = Kp
        self.wp_idx = 1  # Start at waypoint 1 (skip starting position at waypoint 0)
        self.dist_threshold = 1.0  # Larger threshold for corners
        self.debug = debug
        
    def get_control(self, state, update_state=True):
        # state: [x, y, vx, vy]
        pos = state[:2]
        vel = state[2:4]
        
        # Determine target
        if self.wp_idx >= len(self.waypoints):
            target = self.waypoints[-1]
        else:
            target = self.waypoints[self.wp_idx]
            
        dist = np.linalg.norm(target - pos)
        
        if update_state:
            if dist < self.dist_threshold and self.wp_idx < len(self.waypoints) - 1:
                if self.debug:
                    print(f"Switching from WP {self.wp_idx} to {self.wp_idx+1} at dist {dist:.2f}")
                self.wp_idx += 1
                target = self.waypoints[self.wp_idx]
                dist = np.linalg.norm(target - pos)

        # P-Control on Velocity with SQRT Braking Profile
        err_pos = target - pos
        v_des_dir = err_pos / (dist + 1e-6)
        
        # Braking distance: v^2 = 2*a*d. v = sqrt(2*a*d)
        braking_speed = np.sqrt(2 * 4.0 * dist)
        
        # Ramp down speed
        speed = min(self.v_max, braking_speed)
            
        v_des = v_des_dir * speed
        
        # Acceleration control
        acc = self.Kp * (v_des - vel)
        return acc


class GhostPredictor:
    """Predicts ghost positions for Baselines and Gatekeeper/MPS."""
    def __init__(self, env):
        self.env = env
        
    def __call__(self, t):
        """
        Predict ghost positions at relative time t.
        
        Args:
            t: Relative time from current moment (in seconds)
            
        Returns:
            List of ghost dicts with predicted positions, or single dict for backward compatibility
        """
        ghosts_now = self.env.ghosts
        predicted_ghosts = []
        
        rel_t = t  # t is relative time from now
        
        # Predict all ghosts
        for g in ghosts_now:
            px = g['x'] + g['vx'] * rel_t
            py = g['y'] + g['vy'] * rel_t
            predicted_ghosts.append({
                'x': px, 'y': py, 
                'radius': g['radius'], 
                'vx': g['vx'], 'vy': g['vy']
            })
        
        # For backward compatibility with BackupCBF (expects single obstacle or None)
        # Return first ghost if exists, otherwise None
        if len(predicted_ghosts) == 0:
            return None
        elif len(predicted_ghosts) == 1:
            return predicted_ghosts[0]
        else:
            # Return list for gatekeeper/mps (they can handle multiple)
            return predicted_ghosts


class StopBackupController:
    """Stop backup controller."""
    def compute_control(self, state, target=None):
        return np.zeros(2)


class MoveAwayBackupController:
    """Move away from ghosts backup controller."""
    def __init__(self, env):
        self.env = env
        
    def compute_control(self, state, target=None):
        pos = state[:2, 0] if state.shape[0] > 2 else state[:2]
        
        # Move away from closest ghost
        if len(self.env.ghosts) > 0:
            closest_ghost = min(self.env.ghosts, 
                              key=lambda g: np.linalg.norm(pos - np.array([g['x'], g['y']])))
            ghost_pos = np.array([closest_ghost['x'], closest_ghost['y']])
            away_dir = pos - ghost_pos
            away_dir = away_dir / (np.linalg.norm(away_dir) + 1e-6)
            return away_dir * 2.0  # Move away with acceleration 2.0
        return np.zeros(2)
