"""Clearance-aware grid planner used by the hospital example."""

from __future__ import annotations

import heapq
from math import hypot, sqrt
from typing import Sequence

import numpy as np

from .environment import HospitalEnvironment, unique_points


GridIndex = tuple[int, int]


class HospitalGridPlanner:
    """Eight-connected A* with a soft preference for wider passages."""

    _NEIGHBORS = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, sqrt(2.0)),
        (-1, 1, sqrt(2.0)),
        (1, -1, sqrt(2.0)),
        (1, 1, sqrt(2.0)),
    )

    def __init__(
        self,
        environment: HospitalEnvironment,
        resolution: float = 1.5,
        clearance: float = 0.73,
        preferred_clearance: float = 3.6,
        clearance_weight: float = 4.5,
    ) -> None:
        self.environment = environment
        self.resolution = float(resolution)
        self.clearance = float(clearance)
        self.preferred_clearance = float(preferred_clearance)
        self.clearance_weight = float(clearance_weight)
        self.xs = np.arange(
            0.5 * self.resolution, environment.width, self.resolution
        )
        self.ys = np.arange(
            0.5 * self.resolution, environment.height, self.resolution
        )
        self.free = np.zeros((self.xs.size, self.ys.size), dtype=bool)
        for ix, x in enumerate(self.xs):
            for iy, y in enumerate(self.ys):
                self.free[ix, iy] = not environment.is_collision(
                    (float(x), float(y)), self.clearance
                )
        self.free_indices = np.argwhere(self.free)
        if self.free_indices.size == 0:
            raise ValueError("hospital planner grid has no free cells")
        self.clearance_map = self._compute_clearance_map()

    def point(self, index: GridIndex) -> np.ndarray:
        return np.array([self.xs[index[0]], self.ys[index[1]]], dtype=float)

    def nearest_free_index(self, point: Sequence[float]) -> GridIndex:
        query = np.asarray(point, dtype=float)
        points = np.column_stack(
            (
                self.xs[self.free_indices[:, 0]],
                self.ys[self.free_indices[:, 1]],
            )
        )
        best = int(np.argmin(np.linalg.norm(points - query, axis=1)))
        return tuple(int(value) for value in self.free_indices[best])

    def nearest_free_point(self, point: Sequence[float]) -> np.ndarray:
        return self.point(self.nearest_free_index(point))

    def plan(
        self, start_point: Sequence[float], goal_point: Sequence[float]
    ) -> list[np.ndarray]:
        start = self.nearest_free_index(start_point)
        goal = self.nearest_free_index(goal_point)
        indices = self._astar(start, goal)
        if not indices:
            raise ValueError("no hospital path found")
        path = [self.point(index) for index in indices]
        path[0] = np.asarray(start_point, dtype=float).copy()
        path[-1] = np.asarray(goal_point, dtype=float).copy()
        return self._compress_to_corners(path)

    def _astar(self, start: GridIndex, goal: GridIndex) -> list[GridIndex]:
        frontier: list[tuple[float, int, GridIndex]] = []
        serial = 0
        heapq.heappush(frontier, (0.0, serial, start))
        came_from: dict[GridIndex, GridIndex | None] = {start: None}
        cost_so_far = {start: 0.0}

        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for neighbor, step_cost in self._neighbors(current):
                candidate = (
                    cost_so_far[current]
                    + step_cost * self.resolution * self._cell_cost(neighbor)
                )
                if neighbor not in cost_so_far or candidate < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = candidate
                    serial += 1
                    heuristic = hypot(
                        neighbor[0] - goal[0], neighbor[1] - goal[1]
                    ) * self.resolution
                    heapq.heappush(
                        frontier, (candidate + heuristic, serial, neighbor)
                    )
                    came_from[neighbor] = current

        if goal not in came_from:
            return []
        path: list[GridIndex] = []
        current: GridIndex | None = goal
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def _neighbors(self, index: GridIndex) -> list[tuple[GridIndex, float]]:
        ix, iy = index
        output: list[tuple[GridIndex, float]] = []
        for dx, dy, cost in self._NEIGHBORS:
            nx, ny = ix + dx, iy + dy
            if (
                nx < 0
                or ny < 0
                or nx >= self.free.shape[0]
                or ny >= self.free.shape[1]
                or not self.free[nx, ny]
            ):
                continue
            if dx and dy and (
                not self.free[ix + dx, iy] or not self.free[ix, iy + dy]
            ):
                continue
            output.append(((nx, ny), cost))
        return output

    def _cell_cost(self, index: GridIndex) -> float:
        clearance = float(self.clearance_map[index])
        if not np.isfinite(clearance) or clearance >= self.preferred_clearance:
            return 1.0
        fraction = (self.preferred_clearance - clearance) / max(
            self.preferred_clearance, 1e-9
        )
        return 1.0 + self.clearance_weight * fraction * fraction

    def _compute_clearance_map(self) -> np.ndarray:
        distances = np.full(self.free.shape, np.inf, dtype=float)
        frontier: list[tuple[float, int, GridIndex]] = []
        serial = 0
        for ix in range(self.free.shape[0]):
            for iy in range(self.free.shape[1]):
                if not self.free[ix, iy]:
                    distances[ix, iy] = 0.0
                    heapq.heappush(frontier, (0.0, serial, (ix, iy)))
                    serial += 1
        while frontier:
            distance, _, current = heapq.heappop(frontier)
            if distance > distances[current] + 1e-12:
                continue
            for (neighbor, step) in self._neighbors_for_distance(current):
                candidate = distance + step * self.resolution
                if candidate < distances[neighbor]:
                    distances[neighbor] = candidate
                    heapq.heappush(frontier, (candidate, serial, neighbor))
                    serial += 1
        return distances

    def _neighbors_for_distance(
        self, index: GridIndex
    ) -> list[tuple[GridIndex, float]]:
        ix, iy = index
        output = []
        for dx, dy, cost in self._NEIGHBORS:
            nx, ny = ix + dx, iy + dy
            if (
                0 <= nx < self.free.shape[0]
                and 0 <= ny < self.free.shape[1]
            ):
                output.append(((nx, ny), cost))
        return output

    def _compress_to_corners(
        self, path: list[np.ndarray], max_spacing: float = 26.0
    ) -> list[np.ndarray]:
        if len(path) <= 2:
            return [point.copy() for point in path]
        kept = [path[0].copy()]
        distance_since_keep = 0.0
        for index in range(1, len(path) - 1):
            previous_step = path[index] - path[index - 1]
            next_step = path[index + 1] - path[index]
            previous_norm = float(np.linalg.norm(previous_step))
            if previous_norm < 1e-9:
                continue
            previous_direction = tuple(np.sign(np.round(previous_step, 6)))
            next_direction = tuple(np.sign(np.round(next_step, 6)))
            distance_since_keep += previous_norm
            if (
                previous_direction != next_direction
                or distance_since_keep >= max_spacing
            ):
                kept.append(path[index].copy())
                distance_since_keep = 0.0
        kept.append(path[-1].copy())
        return unique_points(kept)

