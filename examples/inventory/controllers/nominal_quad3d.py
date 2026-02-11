"""Controllers for the Inventory scenario with Quad3D dynamics."""

import numpy as np

from examples.inventory.controllers.nominal_di import GhostPredictor


def _build_b2_inv(robot_spec):
    L = float(robot_spec.get('L', 0.3))
    nu = float(robot_spec.get('nu', 0.1))
    B2 = np.array([
        [1.0,  1.0,  1.0,  1.0],
        [0.0,  L,    0.0, -L],
        [L,    0.0, -L,    0.0],
        [nu,  -nu,   nu,  -nu],
    ])
    return np.linalg.pinv(B2)


def _accel_to_u(state, ax_des, ay_des, az_des, params):
    m, Ix, Iy, Iz, g, K_ang, Kd_ang, u_min, u_max, B2_inv = params

    theta = state[3]
    phi = state[4]
    psi = state[5]
    q = state[9]
    p = state[10]
    r = state[11]

    theta_des = ax_des / g
    phi_des = -ay_des / g
    F_des = m * az_des

    tau_y = Iy * (K_ang * (theta_des - theta) + Kd_ang * (0.0 - q))
    tau_x = Ix * (K_ang * (phi_des - phi) + Kd_ang * (0.0 - p))
    tau_z = Iz * (K_ang * (0.0 - psi) + Kd_ang * (0.0 - r))

    wrench = np.array([F_des, tau_y, tau_x, tau_z])
    u = B2_inv @ wrench
    return np.clip(u, u_min, u_max)


class WaypointFollowerQuad3D:
    """Waypoint follower for Quad3D (tracks XY, holds Z)."""
    def __init__(self, waypoints, robot_spec, v_max=5.0, Kp_v=6.0, K_lat=1.0, v_lat_max=None, debug=False):
        self.waypoints = waypoints
        self.v_max = v_max
        self.Kp_v = Kp_v
        self.K_lat = K_lat
        self.v_lat_max = v_max if v_lat_max is None else v_lat_max
        self.wp_idx = 1
        self.dist_threshold = 0.8
        self.debug = debug

        self.z_ref = float(robot_spec.get('z_ref', 0.0))
        self.Kp_z = float(robot_spec.get('Kp_z', 4.0))
        self.Kd_z = float(robot_spec.get('Kd_z', 3.0))
        self.a_max_xy = float(robot_spec.get('a_max_xy', 3.0))

        self.m = float(robot_spec.get('mass', 3.0))
        self.Ix = float(robot_spec.get('Ix', 0.5))
        self.Iy = float(robot_spec.get('Iy', 0.5))
        self.Iz = float(robot_spec.get('Iz', 0.5))
        self.g = float(robot_spec.get('g', 9.8))
        self.K_ang = float(robot_spec.get('K_ang', 5.0))
        self.Kd_ang = float(robot_spec.get('Kd_ang', 2.0))
        self.u_min = float(robot_spec.get('u_min', -10.0))
        self.u_max = float(robot_spec.get('u_max', 10.0))
        self.B2_inv = _build_b2_inv(robot_spec)

    def _clip_xy_accel(self, ax, ay):
        a_norm = np.linalg.norm([ax, ay])
        if a_norm > self.a_max_xy:
            scale = self.a_max_xy / (a_norm + 1e-8)
            ax *= scale
            ay *= scale
        return ax, ay

    def get_control(self, state, update_state=True):
        # state: [x, y, z, theta, phi, psi, vx, vy, vz, q, p, r]
        pos = state[0:3]
        vel = state[6:9]

        if self.wp_idx >= len(self.waypoints):
            target_xy = self.waypoints[-1]
        else:
            target_xy = self.waypoints[self.wp_idx]

        dist = np.linalg.norm(target_xy - pos[:2])
        if update_state and dist < self.dist_threshold and self.wp_idx < len(self.waypoints) - 1:
            if self.debug:
                print(f"Switching from WP {self.wp_idx} to {self.wp_idx+1} at dist {dist:.2f}")
            self.wp_idx += 1
            target_xy = self.waypoints[self.wp_idx]
            dist = np.linalg.norm(target_xy - pos[:2])

        prev_idx = max(self.wp_idx - 1, 0)
        prev_xy = self.waypoints[prev_idx]
        seg = target_xy - prev_xy
        seg_norm = np.linalg.norm(seg)
        if seg_norm > 1e-6:
            seg_dir = seg / seg_norm
        else:
            seg_dir = (target_xy - pos[:2]) / (dist + 1e-6)
        perp_dir = np.array([-seg_dir[1], seg_dir[0]])

        dist_along = np.dot(target_xy - pos[:2], seg_dir)
        if update_state and dist_along < 0.0 and self.wp_idx < len(self.waypoints) - 1:
            self.wp_idx += 1
            target_xy = self.waypoints[self.wp_idx]
            dist = np.linalg.norm(target_xy - pos[:2])
            prev_idx = max(self.wp_idx - 1, 0)
            prev_xy = self.waypoints[prev_idx]
            seg = target_xy - prev_xy
            seg_norm = np.linalg.norm(seg)
            if seg_norm > 1e-6:
                seg_dir = seg / seg_norm
            else:
                seg_dir = (target_xy - pos[:2]) / (dist + 1e-6)
            perp_dir = np.array([-seg_dir[1], seg_dir[0]])
            dist_along = np.dot(target_xy - pos[:2], seg_dir)

        braking_speed = np.sqrt(2.0 * self.a_max_xy * abs(dist_along))
        v_long = min(self.v_max, braking_speed)
        v_long_dir = 1.0 if dist_along >= 0.0 else -1.0

        lat_err = np.dot(pos[:2] - prev_xy, perp_dir)
        v_lat = -self.K_lat * lat_err
        v_lat = np.clip(v_lat, -self.v_lat_max, self.v_lat_max)

        v_des_xy = v_long_dir * v_long * seg_dir + v_lat * perp_dir
        v_norm = np.linalg.norm(v_des_xy)
        if v_norm > self.v_max:
            v_des_xy = v_des_xy * (self.v_max / (v_norm + 1e-8))

        ax = self.Kp_v * (v_des_xy[0] - vel[0])
        ay = self.Kp_v * (v_des_xy[1] - vel[1])
        ax, ay = self._clip_xy_accel(ax, ay)

        az = self.Kp_z * (self.z_ref - pos[2]) - self.Kd_z * vel[2]

        params = (self.m, self.Ix, self.Iy, self.Iz, self.g, self.K_ang, self.Kd_ang, self.u_min, self.u_max, self.B2_inv)
        return _accel_to_u(state, ax, ay, az, params)


class StopBackupControllerQuad3D:
    """Stop backup controller (brakes to zero velocity, holds Z)."""
    def __init__(self, robot_spec, Kp_braking=3.0):
        self.Kp = Kp_braking
        self.z_ref = float(robot_spec.get('z_ref', 0.0))
        self.Kp_z = float(robot_spec.get('Kp_z', 4.0))
        self.Kd_z = float(robot_spec.get('Kd_z', 3.0))
        self.a_max_xy = float(robot_spec.get('a_max_xy', 3.0))

        self.m = float(robot_spec.get('mass', 3.0))
        self.Ix = float(robot_spec.get('Ix', 0.5))
        self.Iy = float(robot_spec.get('Iy', 0.5))
        self.Iz = float(robot_spec.get('Iz', 0.5))
        self.g = float(robot_spec.get('g', 9.8))
        self.K_ang = float(robot_spec.get('K_ang', 5.0))
        self.Kd_ang = float(robot_spec.get('Kd_ang', 2.0))
        self.u_min = float(robot_spec.get('u_min', -10.0))
        self.u_max = float(robot_spec.get('u_max', 10.0))
        self.B2_inv = _build_b2_inv(robot_spec)

    def _clip_xy_accel(self, ax, ay):
        a_norm = np.linalg.norm([ax, ay])
        if a_norm > self.a_max_xy:
            scale = self.a_max_xy / (a_norm + 1e-8)
            ax *= scale
            ay *= scale
        return ax, ay

    def compute_control(self, state, target=None):
        if state.ndim == 2:
            state = state[:, 0]

        vel = state[6:9]
        ax = -self.Kp * vel[0]
        ay = -self.Kp * vel[1]
        ax, ay = self._clip_xy_accel(ax, ay)

        az = self.Kp_z * (self.z_ref - state[2]) - self.Kd_z * vel[2]

        params = (self.m, self.Ix, self.Iy, self.Iz, self.g, self.K_ang, self.Kd_ang, self.u_min, self.u_max, self.B2_inv)
        return _accel_to_u(state, ax, ay, az, params)


class MovingBackBackupControllerQuad3D:
    """Backup controller that moves opposite to current velocity in XY."""
    def __init__(self, robot_spec, Kp=3.0, target_speed=1.0, env=None):
        self.Kp = Kp
        self.target_speed = target_speed
        self.env = env
        self.fixed_target_v = None

        self.z_ref = float(robot_spec.get('z_ref', 0.0))
        self.Kp_z = float(robot_spec.get('Kp_z', 4.0))
        self.Kd_z = float(robot_spec.get('Kd_z', 3.0))
        self.a_max_xy = float(robot_spec.get('a_max_xy', 3.0))

        self.m = float(robot_spec.get('mass', 3.0))
        self.Ix = float(robot_spec.get('Ix', 0.5))
        self.Iy = float(robot_spec.get('Iy', 0.5))
        self.Iz = float(robot_spec.get('Iz', 0.5))
        self.g = float(robot_spec.get('g', 9.8))
        self.K_ang = float(robot_spec.get('K_ang', 5.0))
        self.Kd_ang = float(robot_spec.get('Kd_ang', 2.0))
        self.u_min = float(robot_spec.get('u_min', -10.0))
        self.u_max = float(robot_spec.get('u_max', 10.0))
        self.B2_inv = _build_b2_inv(robot_spec)

    def _clip_xy_accel(self, ax, ay):
        a_norm = np.linalg.norm([ax, ay])
        if a_norm > self.a_max_xy:
            scale = self.a_max_xy / (a_norm + 1e-8)
            ax *= scale
            ay *= scale
        return ax, ay

    def prepare_rollout(self, state):
        if state.ndim == 2:
            pos = state[:2, 0]
            vel = state[6:8, 0]
        else:
            pos = state[:2]
            vel = state[6:8]

        away_dir = None
        if self.env is not None and hasattr(self.env, 'ghosts') and len(self.env.ghosts) > 0:
            closest_ghost = min(self.env.ghosts, key=lambda g: np.linalg.norm(pos - np.array([g['x'], g['y']])))
            ghost_pos = np.array([closest_ghost['x'], closest_ghost['y']])
            away_dir = pos - ghost_pos
            dist = np.linalg.norm(away_dir)
            if dist < 1e-3:
                away_dir = np.array([1.0, 0.0])
            else:
                away_dir = away_dir / dist

        if away_dir is not None:
            if abs(away_dir[0]) > abs(away_dir[1]):
                retreat_dir = np.array([np.sign(away_dir[0]), 0.0])
            else:
                retreat_dir = np.array([0.0, np.sign(away_dir[1])])
            self.fixed_target_v = retreat_dir * self.target_speed
        else:
            speed = np.linalg.norm(vel)
            if speed > 0.1:
                self.fixed_target_v = -(vel / speed) * self.target_speed
            else:
                self.fixed_target_v = np.array([-self.target_speed, 0.0])

    def compute_control(self, state, target=None):
        if state.ndim == 2:
            state = state[:, 0]

        if self.fixed_target_v is None:
            self.prepare_rollout(state)

        vel = state[6:8]
        ax = self.Kp * (self.fixed_target_v[0] - vel[0])
        ay = self.Kp * (self.fixed_target_v[1] - vel[1])
        ax, ay = self._clip_xy_accel(ax, ay)

        az = self.Kp_z * (self.z_ref - state[2]) - self.Kd_z * state[8]

        params = (self.m, self.Ix, self.Iy, self.Iz, self.g, self.K_ang, self.Kd_ang, self.u_min, self.u_max, self.B2_inv)
        return _accel_to_u(state, ax, ay, az, params)


class MoveAwayBackupControllerQuad3D:
    """Move away from closest ghost in XY."""
    def __init__(self, robot_spec, env):
        self.env = env
        self.controller = MovingBackBackupControllerQuad3D(robot_spec, env=env)

    def compute_control(self, state, target=None):
        return self.controller.compute_control(state, target)


class RetraceBackupControllerQuad3D:
    """Retrace nominal waypoints backwards (XY), hold Z."""
    def __init__(self, nominal_controller, robot_spec, Kp=6.0, target_speed=2.8):
        self.nom = nominal_controller
        self.Kp = Kp
        self.target_speed = target_speed

        self.z_ref = float(robot_spec.get('z_ref', 0.0))
        self.Kp_z = float(robot_spec.get('Kp_z', 4.0))
        self.Kd_z = float(robot_spec.get('Kd_z', 3.0))
        self.a_max_xy = float(robot_spec.get('a_max_xy', 3.0))

        self.m = float(robot_spec.get('mass', 3.0))
        self.Ix = float(robot_spec.get('Ix', 0.5))
        self.Iy = float(robot_spec.get('Iy', 0.5))
        self.Iz = float(robot_spec.get('Iz', 0.5))
        self.g = float(robot_spec.get('g', 9.8))
        self.K_ang = float(robot_spec.get('K_ang', 5.0))
        self.Kd_ang = float(robot_spec.get('Kd_ang', 2.0))
        self.u_min = float(robot_spec.get('u_min', -10.0))
        self.u_max = float(robot_spec.get('u_max', 10.0))
        self.B2_inv = _build_b2_inv(robot_spec)

        self.active_retrace_idx = 0
        self.last_nominal_idx = -1
        self.retrace_idx = 0

    def _clip_xy_accel(self, ax, ay):
        a_norm = np.linalg.norm([ax, ay])
        if a_norm > self.a_max_xy:
            scale = self.a_max_xy / (a_norm + 1e-8)
            ax *= scale
            ay *= scale
        return ax, ay

    def prepare_rollout(self, state):
        pos = state.flatten()[:2]

        if self.nom.wp_idx > self.last_nominal_idx:
            self.last_nominal_idx = self.nom.wp_idx
            self.active_retrace_idx = max(0, self.nom.wp_idx - 1)

        self.retrace_idx = self.active_retrace_idx

    def get_current_target(self):
        if self.nom.waypoints is None:
            return np.zeros(2)
        idx = min(len(self.nom.waypoints) - 1, self.active_retrace_idx)
        return self.nom.waypoints[idx]

    def compute_control(self, state, target=None):
        if state.ndim == 2:
            state = state[:, 0]

        pos = state[:2]
        vel = state[6:8]

        target_pos = self.nom.waypoints[self.retrace_idx]
        dist = np.linalg.norm(target_pos - pos)
        if dist < 0.8 and self.retrace_idx > 0:
            self.retrace_idx = max(0, self.retrace_idx - 1)
            target_pos = self.nom.waypoints[self.retrace_idx]
            dist = np.linalg.norm(target_pos - pos)

        err_pos = target_pos - pos
        v_des_dir = err_pos / (dist + 1e-6)
        braking_speed = np.sqrt(2.0 * self.a_max_xy * max(dist, 0.0))
        v_des_speed = min(self.target_speed, braking_speed)
        v_des = v_des_dir * v_des_speed

        ax = self.Kp * (v_des[0] - vel[0])
        ay = self.Kp * (v_des[1] - vel[1])
        ax, ay = self._clip_xy_accel(ax, ay)

        az = self.Kp_z * (self.z_ref - state[2]) - self.Kd_z * state[8]

        params = (self.m, self.Ix, self.Iy, self.Iz, self.g, self.K_ang, self.Kd_ang, self.u_min, self.u_max, self.B2_inv)
        return _accel_to_u(state, ax, ay, az, params)
