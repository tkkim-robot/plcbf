"""Setup for the additional Warehouse Quad3D baselines."""

from __future__ import annotations

import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "safe_control"))

from safe_control.envs.warehouse_env import WarehouseEnv

from examples.warehouse.algorithms.library_pcbf_mi_quad3d import (
    LibraryPCBFMinInterventionQuad3D,
)
from examples.warehouse.algorithms.multi_backup_cbf_mi_quad3d import (
    MultiBackupCBFMinInterventionQuad3D,
)
from examples.warehouse.controllers.nominal_quad3d import WaypointFollowerQuad3D
from examples.warehouse.controllers.policies_quad3d_jax import Quad3DControlParams
from examples.warehouse.dynamics.dynamics_quad3d_jax import _build_quad3d_matrices
from examples.warehouse.dynamics.quad3d import Quad3D


BASELINE_KEYS = ("multi_backup_cbf_mi", "library_pcbf_mi")
DEFAULT_SENSING_RANGE_M = 13.0


def setup_test(
    algo: str,
    level: int,
    safety_margin: float = 1.3,
    num_angle_policies: int = 64,
    alpha: float | None = None,
):
    if algo not in BASELINE_KEYS:
        raise ValueError(
            f"Unsupported additional baseline {algo!r}; valid keys are {BASELINE_KEYS}"
        )

    env = WarehouseEnv(level=level)
    alpha_val = 6.0 if alpha is None else float(alpha)
    robot_spec = {
        "model": "Quad3D",
        "radius": 1.0,
        "mass": 3.0,
        "Ix": 0.5,
        "Iy": 0.5,
        "Iz": 0.5,
        "L": 0.3,
        "nu": 0.1,
        "g": 9.8,
        "u_max": 10.0,
        "u_min": -10.0,
        "v_max": 3.5,
        "v_ref": 3.0,
        "a_max_xy": 8.0,
        "z_ref": 0.0,
        "Kp_z": 4.0,
        "Kd_z": 3.0,
        "K_ang": 10.0,
        "Kd_ang": 4.0,
        "nominal_Kp_v": 7.0,
        "nominal_K_lat": 1.2,
        "nominal_v_lat_max": 2.5,
        "nominal_dist_threshold": 1.0,
        "angle_Kp_v": 7.0,
        "stop_Kp_v": 3.0,
        "backup_Kp": 6.0,
        "backup_speed": 2.8,
    }

    robot = Quad3D(env.dt, robot_spec)
    nominal_ctrl = WaypointFollowerQuad3D(
        env.get_nominal_waypoints(),
        robot_spec=robot_spec,
        v_max=robot_spec["v_max"],
        Kp_v=robot_spec["nominal_Kp_v"],
        K_lat=robot_spec["nominal_K_lat"],
        v_lat_max=robot_spec["nominal_v_lat_max"],
        debug=False,
    )
    nominal_ctrl.dist_threshold = robot_spec["nominal_dist_threshold"]

    _, _, _, b2_inv = _build_quad3d_matrices(
        robot_spec["mass"],
        robot_spec["Ix"],
        robot_spec["Iy"],
        robot_spec["Iz"],
        robot_spec["L"],
        robot_spec["nu"],
        robot_spec["g"],
    )
    ctrl_params = Quad3DControlParams(
        m=robot_spec["mass"],
        Ix=robot_spec["Ix"],
        Iy=robot_spec["Iy"],
        Iz=robot_spec["Iz"],
        g=robot_spec["g"],
        B2_inv=b2_inv,
        u_min=robot_spec["u_min"],
        u_max=robot_spec["u_max"],
        K_ang=robot_spec["K_ang"],
        Kd_ang=robot_spec["Kd_ang"],
        z_ref=robot_spec["z_ref"],
        Kp_z=robot_spec["Kp_z"],
        Kd_z=robot_spec["Kd_z"],
        a_max_xy=robot_spec["a_max_xy"],
    )

    backup_horizon = 4.0
    if algo == "library_pcbf_mi":
        shielding = LibraryPCBFMinInterventionQuad3D(
            robot_spec,
            dt=env.dt,
            backup_horizon=backup_horizon,
            cbf_alpha=alpha_val,
            safety_margin=safety_margin,
            num_angle_policies=num_angle_policies,
            line_width_scale=1.0,
        )
    else:
        shielding = MultiBackupCBFMinInterventionQuad3D(
            robot=robot,
            robot_spec=robot_spec,
            dt=env.dt,
            backup_horizon=backup_horizon,
            cbf_alpha=2.0,
            terminal_alpha=2.0,
            safety_margin=safety_margin,
            num_angle_policies=num_angle_policies,
            ax=None,
        )
    shielding.set_environment(env)
    return env, robot, nominal_ctrl, shielding, robot_spec, ctrl_params
