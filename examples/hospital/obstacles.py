"""Dynamic hospital obstacles with geometry-faithful prediction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, hypot, sin
from typing import Protocol, Sequence

import numpy as np

from .environment import HospitalEnvironment, Rect


def _reflected_motion(
    position: float,
    speed: float,
    elapsed: float,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Exact one-dimensional reflection, including multiple endpoint hits."""

    if upper <= lower or abs(speed) < 1e-12 or elapsed <= 0.0:
        return min(upper, max(lower, position)), speed
    span = upper - lower
    raw = position - lower + speed * elapsed
    period = 2.0 * span
    phase = raw % period
    if phase <= span:
        coordinate = lower + phase
        direction = 1.0
    else:
        coordinate = lower + (period - phase)
        direction = -1.0
    return coordinate, abs(speed) * direction * (1.0 if speed >= 0 else -1.0)


class DynamicObstacle(Protocol):
    identifier: str

    @property
    def center(self) -> np.ndarray: ...

    @property
    def velocity(self) -> np.ndarray: ...

    def advance(self, dt: float, environment: HospitalEnvironment) -> None: ...

    def predicted(
        self, elapsed: float, environment: HospitalEnvironment
    ) -> "DynamicObstacle": ...

    def signed_clearance(
        self,
        point: Sequence[float],
        robot_radius: float,
        extra_margin: float = 0.0,
    ) -> float: ...


@dataclass
class Human:
    identifier: str
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 0.52

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=float)

    def advance(self, dt: float, environment: HospitalEnvironment) -> None:
        next_position = self.center + self.velocity * dt
        if not environment.is_collision(next_position, self.radius):
            self.x, self.y = (float(value) for value in next_position)
            return
        next_x = np.array([self.x + self.vx * dt, self.y])
        next_y = np.array([self.x, self.y + self.vy * dt])
        moved = False
        if not environment.is_collision(next_x, self.radius):
            self.x = float(next_x[0])
            self.vy *= -1.0
            moved = True
        if not environment.is_collision(next_y, self.radius):
            self.y = float(next_y[1])
            self.vx *= -1.0
            moved = True
        if not moved:
            self.vx *= -1.0
            self.vy *= -1.0

    def predicted(
        self, elapsed: float, environment: HospitalEnvironment
    ) -> "Human":
        copy = replace(self)
        remaining = max(0.0, float(elapsed))
        while remaining > 1e-10:
            step = min(0.05, remaining)
            copy.advance(step, environment)
            remaining -= step
        return copy

    def signed_clearance(
        self,
        point: Sequence[float],
        robot_radius: float,
        extra_margin: float = 0.0,
    ) -> float:
        return (
            float(np.linalg.norm(np.asarray(point, dtype=float) - self.center))
            - self.radius
            - robot_radius
            - extra_margin
        )


@dataclass
class Stretcher:
    identifier: str
    coordinate: float
    lateral: float
    speed: float
    axis: str
    route_min: float
    route_max: float
    length: float = 4.1
    width: float = 1.45
    reflect_at_route_bounds: bool = True

    @property
    def theta(self) -> float:
        return 0.0 if self.axis == "x" else np.pi / 2.0

    @property
    def center(self) -> np.ndarray:
        if self.axis == "x":
            return np.array([self.coordinate, self.lateral], dtype=float)
        return np.array([self.lateral, self.coordinate], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        if self.axis == "x":
            return np.array([self.speed, 0.0], dtype=float)
        return np.array([0.0, self.speed], dtype=float)

    @property
    def cross_section_width(self) -> float:
        return self.width

    def advance(self, dt: float, environment: HospitalEnvironment) -> None:
        del environment
        if not self.reflect_at_route_bounds:
            self.coordinate += self.speed * float(dt)
            return
        self.coordinate, self.speed = _reflected_motion(
            self.coordinate, self.speed, dt, self.route_min, self.route_max
        )

    def predicted(
        self, elapsed: float, environment: HospitalEnvironment
    ) -> "Stretcher":
        del environment
        if not self.reflect_at_route_bounds:
            return replace(
                self,
                coordinate=self.coordinate
                + self.speed * max(0.0, float(elapsed)),
            )
        coordinate, speed = _reflected_motion(
            self.coordinate,
            self.speed,
            max(0.0, float(elapsed)),
            self.route_min,
            self.route_max,
        )
        return replace(self, coordinate=coordinate, speed=speed)

    def signed_clearance(
        self,
        point: Sequence[float],
        robot_radius: float,
        extra_margin: float = 0.0,
    ) -> float:
        px, py = np.asarray(point, dtype=float) - self.center
        c, s = cos(self.theta), sin(self.theta)
        local_x = c * px + s * py
        local_y = -s * px + c * py
        qx = abs(local_x) - 0.5 * self.length
        qy = abs(local_y) - 0.5 * self.width
        outside = hypot(max(qx, 0.0), max(qy, 0.0))
        inside = min(max(qx, qy), 0.0)
        # Exact Euclidean signed distance from a circle to the oriented
        # rectangle: box SDF minus the circle radius (and requested margin).
        return outside + inside - robot_radius - extra_margin

    def proxy_discs(
        self, count: int = 5, elapsed: float = 0.0
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        center = self.center + self.velocity * 0.0
        if elapsed:
            if self.reflect_at_route_bounds:
                coordinate, speed = _reflected_motion(
                    self.coordinate,
                    self.speed,
                    elapsed,
                    self.route_min,
                    self.route_max,
                )
            else:
                coordinate = self.coordinate + self.speed * elapsed
                speed = self.speed
            center = (
                np.array([coordinate, self.lateral])
                if self.axis == "x"
                else np.array([self.lateral, coordinate])
            )
            velocity = (
                np.array([speed, 0.0])
                if self.axis == "x"
                else np.array([0.0, speed])
            )
        else:
            velocity = self.velocity
        direction = np.array([cos(self.theta), sin(self.theta)])
        offsets = (
            np.array([0.0])
            if count <= 1
            else np.linspace(-0.5 * self.length, 0.5 * self.length, count)
        )
        return [
            (center + direction * offset, velocity.copy(), 0.5 * self.width)
            for offset in offsets
        ]


def stretcher_route(
    corridor: Rect,
    coordinate: float,
    lateral: float | None = None,
    length: float = 4.1,
    width: float = 1.45,
) -> tuple[str, float, float, float, float]:
    """Construct an axis/lane route that keeps a rectangle in a corridor."""

    axis = "x" if corridor.width >= corridor.height else "y"
    half_length = 0.5 * length
    half_width = 0.5 * width
    if axis == "x":
        route_min = corridor.x + half_length + 0.25
        route_max = corridor.x1 - half_length - 0.25
        lane_min = corridor.y + half_width + 0.24
        lane_max = corridor.y1 - half_width - 0.24
        lane = min(lane_max, max(lane_min, corridor.center[1] if lateral is None else lateral))
    else:
        route_min = corridor.y + half_length + 0.25
        route_max = corridor.y1 - half_length - 0.25
        lane_min = corridor.x + half_width + 0.24
        lane_max = corridor.x1 - half_width - 0.24
        lane = min(lane_max, max(lane_min, corridor.center[0] if lateral is None else lateral))
    coordinate = min(route_max, max(route_min, coordinate))
    return axis, coordinate, lane, route_min, route_max


def obstacle_clearance(
    obstacle: DynamicObstacle,
    point: Sequence[float],
    robot_radius: float,
    human_margin: float,
    stretcher_margin: float,
) -> float:
    margin = human_margin if isinstance(obstacle, Human) else stretcher_margin
    return obstacle.signed_clearance(point, robot_radius, margin)
