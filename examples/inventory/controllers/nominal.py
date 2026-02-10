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
    """Stop backup controller (Brakes to zero velocity)."""
    def __init__(self, Kp_braking=4.0):
        self.Kp = Kp_braking

    def compute_control(self, state, target=None):
        # state: [x, y, vx, vy]
        if state.ndim == 2:
            vel = state[2:4, 0]
        else:
            vel = state[2:4]
        
        # Desired velocity is zero
        acc = -self.Kp * vel
        return acc

    def get_sensitivity(self, state):
        # u = -Kp * [vx, vy]
        # du/dx = [0, 0, -Kp, 0]
        #         [0, 0, 0, -Kp]
        K = np.zeros((2, 4))
        K[0, 2] = -self.Kp
        K[1, 3] = -self.Kp
        return K


class MovingBackBackupController:
    """
    Backup controller that moves in the opposite direction of current velocity.
    Determines a target direction at the start of the rollout and sticks to it.
    """
    def __init__(self, Kp=4.0, target_speed=1.0, a_max=5.0, env=None):
        self.Kp = Kp
        self.target_speed = target_speed
        self.a_max = a_max
        self.env = env
        self.fixed_target_v = None
        
    def prepare_rollout(self, state):
        """Prepare for a new rollout by fixing the target velocity."""
        if state.ndim == 2:
            vel = state[2:4, 0]
            pos = state[:2, 0]
        else:
            vel = state[2:4]
    def prepare_rollout(self, state):
        """Prepare for backup trajectory rollout by choosing escape direction."""
        if state.ndim == 2:
            pos = state[:2, 0]
            vel = state[2:4, 0]
        else:
            pos = state[:2]
            vel = state[2:4]
            
        # 1. Identify closest active ghost
        away_dir = None
        if self.env is not None and hasattr(self.env, 'ghosts') and len(self.env.ghosts) > 0:
            active_ghosts = [g for g in self.env.ghosts if g.get('active', True)]
            if active_ghosts:
                closest_ghost = min(active_ghosts, 
                                  key=lambda g: np.linalg.norm(pos - np.array([g['x'], g['y']])))
                ghost_pos = np.array([closest_ghost['x'], closest_ghost['y']])
                away_dir = pos - ghost_pos
                dist = np.linalg.norm(away_dir)
                if dist < 0.01:
                    away_dir = np.array([1.0, 0.0]) # Default fallback
                else:
                    away_dir = away_dir / dist
        
        # 2. Finalize retreat direction (aligned with hallway axis)
        if away_dir is not None:
            # Align with hallway axis: prefer moving along the dominant retreat component
            # to stay within hallway boundaries.
            if abs(away_dir[0]) > abs(away_dir[1]):
                retreat_dir = np.array([np.sign(away_dir[0]), 0.0])
            else:
                retreat_dir = np.array([0.0, np.sign(away_dir[1])])
                
            self.fixed_target_v = retreat_dir * self.target_speed
        else:
            # Fallback if no ghosts: move opposite of current velocity or default to -X
            speed = np.linalg.norm(vel)
            if speed > 0.1:
                self.fixed_target_v = -(vel / speed) * self.target_speed
            else:
                self.fixed_target_v = np.array([-self.target_speed, 0.0])
            
    def compute_control(self, state, target=None):
        if state.ndim == 2:
            pos = state[:2, 0]
            vel = state[2:4, 0]
        else:
            pos = state[:2]
            vel = state[2:4]
            
        if self.fixed_target_v is None:
            self.prepare_rollout(state)
            
        target_v = self.fixed_target_v.copy()
        
        # Safety Check: If moving towards a static obstacle/boundary, BRAKE!
        # Simple lookahead
        lookahead_dist = 3.0
        v_norm = np.linalg.norm(vel)
        if v_norm > 0.1:
            # Check if trajectory intersects any static obstacle
            # Prediction: pos + vel * t
            # Only checking static obstacles for backup policy safety
            if self.env is not None:
                # 1. Boundaries
                valid = True
                pred_pos = pos + (vel / v_norm) * lookahead_dist
                
                robot_radius = 0.5 # Approximate
                if hasattr(self.env, 'width'):
                    if not (robot_radius < pred_pos[0] < self.env.width - robot_radius and
                            robot_radius < pred_pos[1] < self.env.height - robot_radius):
                        valid = False
                
                # 2. Static Obstacles
                if valid and hasattr(self.env, 'obstacles'):
                    # Segment: pos -> pred_pos
                    p1 = pos
                    p2 = pred_pos
                    segment_vec = p2 - p1
                    segment_len_sq = np.dot(segment_vec, segment_vec)
                    
                    for obs in self.env.obstacles:
                        obs_pos = np.array([obs.get('x', 0), obs.get('y', 0)])
                        obs_r = obs.get('radius', 1.0)
                        
                        # Distance from point to segment
                        if segment_len_sq == 0:
                            dist = np.linalg.norm(obs_pos - p1)
                        else:
                            t = np.dot(obs_pos - p1, segment_vec) / segment_len_sq
                            t = np.clip(t, 0, 1)
                            projection = p1 + t * segment_vec
                            dist = np.linalg.norm(obs_pos - projection)
                            
                        if dist < (obs_r + robot_radius + 0.5):
                            valid = False
                            break
                            
                if not valid:
                    # Obstacle ahead! Stop!
                    target_v = np.zeros(2)
            
        acc = self.Kp * (target_v - vel)
        
        # Clip
        a_norm = np.linalg.norm(acc)
        if a_norm > self.a_max:
            acc = acc * (self.a_max / a_norm)
        return acc

    def get_sensitivity(self, state):
        # u = Kp * (v_fixed - v)
        # du/dv = -Kp
        K = np.zeros((2, 4))
        K[0, 2] = -self.Kp
        K[1, 3] = -self.Kp
        return K



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


class RetraceBackupController:
    """
    Backup controller that retraces the nominal path backwards.
    Maintains a reference to the nominal WaypointFollower to know the current target index.
    """
    def __init__(self, nominal_controller, Kp=8.0, target_speed=6.0, a_max=10.0):
        self.nom = nominal_controller
        self.Kp = Kp
        self.target_speed = target_speed
        self.a_max = a_max
        
    def compute_control(self, state, target=None):
        if state.ndim == 2:
            pos = state[:2, 0]
            vel = state[2:4, 0]
        else:
            pos = state[:2]
            vel = state[2:4]
            
        # Determine target waypoint index based on current nominal progress
        # We want to go to wp_idx - 1 (start of current segment)
        # If we are close to it, go to wp_idx - 2
        
        current_nominal_idx = self.nom.wp_idx
        target_idx = max(0, current_nominal_idx - 1)
        
        target_pos = self.nom.waypoints[target_idx]
        
        # Check if we reached this backup target during rollout
        dist = np.linalg.norm(target_pos - pos)
        if dist < 2.0 and target_idx > 0:
             target_idx = max(0, target_idx - 1)
             target_pos = self.nom.waypoints[target_idx]
             dist = np.linalg.norm(target_pos - pos)
             
        # P-Control to target
        err_pos = target_pos - pos
        
        # Smooth velocity profile to avoid singularity at dist=0 (infinite stiffness)
        if dist < 0.1:
             # Scale speed linearly with distance
             # Max usage of stiffness: speed / dist = target_speed / 0.1
             current_speed_target = (self.target_speed / 0.1) * dist
        else:
             current_speed_target = self.target_speed
             
        v_des_dir = err_pos / (dist + 1e-6)
        v_des = v_des_dir * current_speed_target
        
        acc = self.Kp * (v_des - vel)
        
        # Clip
        a_norm = np.linalg.norm(acc)
        if a_norm > self.a_max:
            acc = acc * (self.a_max / a_norm)
        return acc

    def get_sensitivity(self, state):
        pos = state[:2]
        # Recalculate target to get correct linearization point
        current_nominal_idx = self.nom.wp_idx
        target_idx = max(0, current_nominal_idx - 1)
        target_pos = self.nom.waypoints[target_idx]
        
        # Check if we reached this backup target during rollout check
        # (Sensitivity is local, so we check distance at current state)
        dist = np.linalg.norm(target_pos - pos)
        if dist < 2.0 and target_idx > 0:
             target_idx = max(0, target_idx - 1)
             target_pos = self.nom.waypoints[target_idx]
             dist = np.linalg.norm(target_pos - pos)

        K = np.zeros((2, 4))
        
        # du/dv = -Kp
        K[0, 2] = -self.Kp
        K[1, 3] = -self.Kp
        
        # du/dx = Kp * d(v_des)/dx
        # v_des = speed * (target - pos) / dist
        # let d = target - pos. v_des = speed * d / ||d||
        # d(v_des)/dx = d(v_des)/dd * dd/dx
        # dd/dx = -I
        # d(v_des)/dd = speed * (I/||d|| - d*d.T/||d||^3)
        #             = (speed / dist) * (I - d_hat * d_hat.T)
        
        if dist > 0.1:
            d = target_pos - pos
            d_hat = d / dist
            
            # Jacobian of normalized vector scaling
            # J_dir = (I - d_hat*d_hat.T) / dist
            # dv_des/dx = speed * J_dir * (-I)
            # dv_dx = -self.target_speed * (I - np.outer(d_hat, d_hat)) / dist
            
            I = np.eye(2)
            J_dir = (I - np.outer(d_hat, d_hat)) / dist
            dv_dx = -self.target_speed * J_dir
            
        else:
             # Linear ramp region: v_des = (v_max/0.1) * (target - pos)
             # dv_des/dx = -(v_max/0.1) * I
             ramp_slope = self.target_speed / 0.1
             dv_dx = -ramp_slope * np.eye(2)
            
        # du/dx = Kp * dv_dx
        K[:, :2] = self.Kp * dv_dx
            
        return K
    
    def prepare_rollout(self, state):
        pass
