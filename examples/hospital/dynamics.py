"""Double-integrator dynamics and nominal waypoint tracking."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import RobotConfig


def clip_acceleration(control: Sequence[float], robot: RobotConfig) -> np.ndarray:
    """Apply the playground's component-wise acceleration bounds."""

    return np.clip(np.asarray(control, dtype=float), -robot.a_max, robot.a_max)


def step_double_integrator(
    state: Sequence[float],
    control: Sequence[float],
    dt: float,
    robot: RobotConfig,
) -> np.ndarray:
    """Semi-implicit DI step for ``[x, y, vx, vy]``."""

    value = np.asarray(state, dtype=float)
    acceleration = clip_acceleration(control, robot)
    velocity = value[2:4] + acceleration * float(dt)
    speed = float(np.linalg.norm(velocity))
    if speed > robot.v_max:
        velocity *= robot.v_max / speed
    position = value[:2] + velocity * float(dt)
    return np.concatenate((position, velocity))


def waypoint_control(
    state: Sequence[float],
    target: Sequence[float],
    robot: RobotConfig,
    target_speed: float | None = None,
) -> np.ndarray:
    value = np.asarray(state, dtype=float)
    delta = np.asarray(target, dtype=float) - value[:2]
    distance = float(np.linalg.norm(delta))
    if distance < 1e-9:
        desired_velocity = np.zeros(2)
    else:
        speed = min(
            robot.v_max if target_speed is None else float(target_speed),
            robot.k_position * distance,
        )
        desired_velocity = delta * (speed / distance)
    return clip_acceleration(
        robot.k_velocity * (desired_velocity - value[2:4]), robot
    )


