"""
Created on February 11th, 2026
@author: Taekyung Kim

@description:
JAX-compatible backup policies for Quad3D PCBF/MPCBF.
"""

from typing import NamedTuple
import jax.numpy as jnp


class Quad3DControlParams(NamedTuple):
    m: float
    Ix: float
    Iy: float
    Iz: float
    g: float
    B2_inv: jnp.ndarray
    u_min: float
    u_max: float
    K_ang: float
    Kd_ang: float
    z_ref: float
    Kp_z: float
    Kd_z: float
    a_max_xy: float


class AnglePolicyParams(NamedTuple):
    target_angle: float
    target_speed: float
    Kp_v: float
    ctrl: Quad3DControlParams


class StopPolicyParams(NamedTuple):
    Kp_v: float
    ctrl: Quad3DControlParams


class WaypointPolicyParams(NamedTuple):
    waypoints: jnp.ndarray
    v_max: float
    Kp: float
    K_lat: float
    v_lat_max: float
    dist_threshold: float
    current_wp_idx: int
    ctrl: Quad3DControlParams


def _clip_xy_accel(ax, ay, a_max_xy):
    norm = jnp.sqrt(ax * ax + ay * ay + 1e-8)
    scale = jnp.where(norm > a_max_xy, a_max_xy / norm, 1.0)
    return ax * scale, ay * scale


def _accel_to_u(state, ax_des, ay_des, az_des, ctrl: Quad3DControlParams):
    theta = state[3]
    phi = state[4]
    psi = state[5]
    q = state[9]
    p = state[10]
    r = state[11]

    theta_des = ax_des / ctrl.g
    phi_des = -ay_des / ctrl.g
    F_des = ctrl.m * az_des

    tau_y = ctrl.Iy * (ctrl.K_ang * (theta_des - theta) + ctrl.Kd_ang * (0.0 - q))
    tau_x = ctrl.Ix * (ctrl.K_ang * (phi_des - phi) + ctrl.Kd_ang * (0.0 - p))
    tau_z = ctrl.Iz * (ctrl.K_ang * (0.0 - psi) + ctrl.Kd_ang * (0.0 - r))

    wrench = jnp.array([F_des, tau_y, tau_x, tau_z])
    u = ctrl.B2_inv @ wrench
    u = jnp.clip(u, ctrl.u_min, ctrl.u_max)
    return u


class AnglePolicyJAX:
    @staticmethod
    def compute(state: jnp.ndarray, params: AnglePolicyParams) -> jnp.ndarray:
        vx, vy, vz = state[6], state[7], state[8]
        ctrl = params.ctrl

        vx_des = params.target_speed * jnp.cos(params.target_angle)
        vy_des = params.target_speed * jnp.sin(params.target_angle)

        ax = params.Kp_v * (vx_des - vx)
        ay = params.Kp_v * (vy_des - vy)
        ax, ay = _clip_xy_accel(ax, ay, ctrl.a_max_xy)

        az = ctrl.Kp_z * (ctrl.z_ref - state[2]) - ctrl.Kd_z * vz

        return _accel_to_u(state, ax, ay, az, ctrl)


class StopPolicyJAX:
    @staticmethod
    def compute(state: jnp.ndarray, params: StopPolicyParams) -> jnp.ndarray:
        vx, vy, vz = state[6], state[7], state[8]
        ctrl = params.ctrl

        ax = -params.Kp_v * vx
        ay = -params.Kp_v * vy
        ax, ay = _clip_xy_accel(ax, ay, ctrl.a_max_xy)

        az = ctrl.Kp_z * (ctrl.z_ref - state[2]) - ctrl.Kd_z * vz

        return _accel_to_u(state, ax, ay, az, ctrl)


class WaypointPolicyJAX:
    @staticmethod
    def compute(state: jnp.ndarray, params: WaypointPolicyParams) -> jnp.ndarray:
        pos = state[0:2]
        vel = state[6:8]
        ctrl = params.ctrl

        target = params.waypoints[params.current_wp_idx]
        prev_idx = jnp.maximum(params.current_wp_idx - 1, 0)
        prev = params.waypoints[prev_idx]
        seg = target - prev
        seg_norm = jnp.sqrt(jnp.sum(seg ** 2) + 1e-8)
        seg_dir = jnp.where(seg_norm > 1e-6, seg / seg_norm, (target - pos) / (jnp.sqrt(jnp.sum((target - pos) ** 2)) + 1e-6))
        perp_dir = jnp.array([-seg_dir[1], seg_dir[0]])

        dist_along = jnp.dot(target - pos, seg_dir)
        braking_speed = jnp.sqrt(2.0 * ctrl.a_max_xy * jnp.abs(dist_along))
        v_long = jnp.minimum(params.v_max, braking_speed)
        v_long_dir = jnp.where(dist_along >= 0.0, 1.0, -1.0)

        lat_err = jnp.dot(pos - prev, perp_dir)
        v_lat = -params.K_lat * lat_err
        v_lat = jnp.clip(v_lat, -params.v_lat_max, params.v_lat_max)

        v_des = v_long_dir * v_long * seg_dir + v_lat * perp_dir
        v_des_norm = jnp.sqrt(jnp.sum(v_des ** 2) + 1e-8)
        v_des = jnp.where(v_des_norm > params.v_max, v_des * (params.v_max / v_des_norm), v_des)

        ax = params.Kp * (v_des[0] - vel[0])
        ay = params.Kp * (v_des[1] - vel[1])
        ax, ay = _clip_xy_accel(ax, ay, ctrl.a_max_xy)

        az = ctrl.Kp_z * (ctrl.z_ref - state[2]) - ctrl.Kd_z * state[8]

        return _accel_to_u(state, ax, ay, az, ctrl)
