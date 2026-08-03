"""Hospital policy controller and dynamic HOCBF constraints.

Room refuge behavior is represented by ordinary closed-loop backup policies.
There is deliberately no latched enter/hold/exit controller: the complete
policy library is rebuilt and optimized at every certificate update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from plcbf.policy_library import (
    CBFHalfspace,
    DecisionDiagnostics,
    PolicyCertificate,
    PolicyDecision,
    SelectionMode,
    select_policy,
)

from .config import HospitalConfig
from .dynamics import step_double_integrator, waypoint_control
from .environment import HospitalEnvironment, Rect, Room
from .jax_rollout import (
    DEFAULT_OBSTACLE_BUCKETS,
    HospitalJaxCapacities,
    compiled_evaluator_cache_info,
    compiled_grouped_evaluator_cache_info,
    evaluate_policy_batch,
    evaluate_policy_groups,
    grouped_capacities_for_obstacle_count,
    pack_obstacle_batch,
    pack_parameters,
    pack_policy_batch,
    pack_policy_groups,
    pack_static_geometry,
    select_obstacle_bucket,
    warmup_policy_batch,
    warmup_policy_groups,
)
from .obstacles import DynamicObstacle, Human, Stretcher, obstacle_clearance
from .planner import HospitalGridPlanner
from .policies import (
    HospitalPolicy,
    PolicyEvaluation,
    build_policy_library,
    rollout_value,
)


@dataclass(frozen=True)
class HocbfConstraint:
    a: np.ndarray
    b: float
    label: str
    obstacle_id: str
    proxy_index: int
    h: float
    h_dot: float
    psi1: float
    safe_distance: float

    def margin(self, control: Sequence[float]) -> float:
        return float(np.dot(self.a, np.asarray(control, dtype=float)) - self.b)


@dataclass(frozen=True)
class ControllerResult:
    control: np.ndarray
    nominal_control: np.ndarray
    selected_policy: str
    inside_refuge: bool
    hocbf_constraints: tuple[HocbfConstraint, ...]
    feasible: bool
    min_hocbf_margin: float
    decision: PolicyDecision
    certificates: tuple[PolicyCertificate, ...]
    policy_evaluations: tuple["PolicyCbfEvaluation", ...]
    candidate_policy_count: int


@dataclass(frozen=True)
class PolicyCbfEvaluation:
    """Rollout and differential terms behind one policy certificate."""

    policy: HospitalPolicy
    value: float
    trajectory: np.ndarray
    gradient: np.ndarray
    value_time_derivative: float
    certificate: PolicyCertificate


def _obstacle_proxies(
    obstacle: DynamicObstacle, proxy_count: int
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    if isinstance(obstacle, Human):
        return [(obstacle.center, obstacle.velocity, obstacle.radius)]
    if isinstance(obstacle, Stretcher):
        return obstacle.proxy_discs(proxy_count)
    return []


def _segment_intersects_rect(
    start: np.ndarray,
    end: np.ndarray,
    rect: Rect,
    padding: float = 0.0,
) -> bool:
    lower = np.array([rect.x - padding, rect.y - padding], dtype=float)
    upper = np.array([rect.x1 + padding, rect.y1 + padding], dtype=float)
    direction = end - start
    minimum, maximum = 0.0, 1.0
    for axis in range(2):
        if abs(direction[axis]) < 1e-12:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return False
            continue
        inverse = 1.0 / direction[axis]
        near = (lower[axis] - start[axis]) * inverse
        far = (upper[axis] - start[axis]) * inverse
        if near > far:
            near, far = far, near
        minimum = max(minimum, near)
        maximum = min(maximum, far)
        if minimum > maximum:
            return False
    return True


def _segment_intersects_circle(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> bool:
    segment = end - start
    denominator = float(segment @ segment)
    if denominator < 1e-12:
        return bool(np.linalg.norm(start - center) <= radius)
    fraction = np.clip(float((center - start) @ segment) / denominator, 0.0, 1.0)
    closest = start + fraction * segment
    return bool(np.linalg.norm(closest - center) <= radius)


def _segment_intersects_stretcher(
    start: np.ndarray,
    end: np.ndarray,
    obstacle: Stretcher,
    padding: float,
) -> bool:
    cosine = float(np.cos(obstacle.theta))
    sine = float(np.sin(obstacle.theta))

    def local(point: np.ndarray) -> np.ndarray:
        relative = point - obstacle.center
        return np.array(
            [
                cosine * relative[0] + sine * relative[1],
                -sine * relative[0] + cosine * relative[1],
            ]
        )

    rectangle = Rect(
        -0.5 * obstacle.length,
        -0.5 * obstacle.width,
        obstacle.length,
        obstacle.width,
    )
    return _segment_intersects_rect(local(start), local(end), rectangle, padding)


def _line_of_sight_blocked(
    start: np.ndarray,
    end: np.ndarray,
    environment: HospitalEnvironment,
    obstacles: Sequence[DynamicObstacle],
    target_id: str,
    config: HospitalConfig,
) -> bool:
    distance = float(np.linalg.norm(end - start))
    samples = max(
        2,
        int(np.ceil(distance / max(config.safety.occlusion_floor_step, 1e-6))),
    )
    for index in range(samples + 1):
        point = start + (end - start) * (index / samples)
        if not environment.is_on_floor(point):
            return True
    if any(
        _segment_intersects_rect(
            start,
            end,
            wall,
            config.safety.occlusion_clearance,
        )
        for wall in environment.wall_rects
    ):
        return True

    target_distance = float(np.linalg.norm(end - start))
    for obstacle in obstacles:
        if obstacle.identifier == target_id:
            continue
        blocker_distance = float(np.linalg.norm(obstacle.center - start))
        if blocker_distance > target_distance - 0.15:
            continue
        if isinstance(obstacle, Human):
            blocked = _segment_intersects_circle(
                start,
                end,
                obstacle.center,
                obstacle.radius + config.safety.occlusion_clearance,
            )
        elif isinstance(obstacle, Stretcher):
            blocked = _segment_intersects_stretcher(
                start,
                end,
                obstacle,
                config.safety.occlusion_clearance,
            )
        else:
            blocked = False
        if blocked:
            return True
    return False


def dynamic_hocbf_constraints(
    state: Sequence[float],
    obstacles: Sequence[DynamicObstacle],
    config: HospitalConfig,
) -> list[HocbfConstraint]:
    """Relative-degree-two HOCBF halfspaces ``a @ u >= b`` for DI control."""

    if not config.safety.enable_hocbf:
        return []
    value = np.asarray(state, dtype=float)
    position, velocity = value[:2], value[2:4]
    constraints: list[HocbfConstraint] = []
    for obstacle in obstacles:
        proxies = _obstacle_proxies(
            obstacle, config.safety.stretcher_proxy_count
        )
        for proxy_index, (proxy_position, proxy_velocity, proxy_radius) in enumerate(
            proxies
        ):
            if isinstance(obstacle, Human):
                object_margin = config.safety.human_margin
            else:
                object_margin = config.safety.stretcher_margin
            safe_distance = (
                config.robot.radius
                + proxy_radius
                + config.safety.hocbf_margin
                + object_margin
            )
            relative_position = position - proxy_position
            distance = float(np.linalg.norm(relative_position))
            activation_margin = config.safety.hocbf_activation_margin
            if (
                isinstance(obstacle, Stretcher)
                and obstacle.width
                >= config.safety.hocbf_wide_stretcher_width
            ):
                activation_margin = (
                    config.safety.hocbf_wide_stretcher_activation_margin
                )
            if (
                distance
                > safe_distance + activation_margin
            ):
                continue
            relative_velocity = velocity - proxy_velocity
            h = float(np.dot(relative_position, relative_position) - safe_distance**2)
            h_dot = float(2.0 * np.dot(relative_position, relative_velocity))
            psi1 = h_dot + config.safety.hocbf_lambda1 * h
            constant = (
                2.0 * float(np.dot(relative_velocity, relative_velocity))
                + config.safety.hocbf_lambda1 * h_dot
                + config.safety.hocbf_lambda2 * psi1
            )
            constraints.append(
                HocbfConstraint(
                    a=2.0 * relative_position,
                    b=-constant,
                    label=f"hocbf:{obstacle.identifier}:{proxy_index}",
                    obstacle_id=obstacle.identifier,
                    proxy_index=proxy_index,
                    h=h,
                    h_dot=h_dot,
                    psi1=psi1,
                    safe_distance=safe_distance,
                )
            )
    return constraints


def _closest_point_on_rect(point: np.ndarray, rect: Rect) -> np.ndarray:
    return np.array(
        [
            np.clip(point[0], rect.x, rect.x1),
            np.clip(point[1], rect.y, rect.y1),
        ],
        dtype=float,
    )


def _closest_rect_boundary(
    point: np.ndarray,
    rect: Rect,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not rect.contains(point):
        closest = _closest_point_on_rect(point, rect)
        distance = float(np.linalg.norm(point - closest))
        outward = (
            (point - closest) / distance
            if distance > 1e-12
            else np.zeros(2)
        )
        return closest, outward, distance
    candidates = (
        (np.array([rect.x, point[1]]), np.array([-1.0, 0.0]), point[0] - rect.x),
        (np.array([rect.x1, point[1]]), np.array([1.0, 0.0]), rect.x1 - point[0]),
        (np.array([point[0], rect.y]), np.array([0.0, -1.0]), point[1] - rect.y),
        (np.array([point[0], rect.y1]), np.array([0.0, 1.0]), rect.y1 - point[1]),
    )
    closest, outward, distance = min(candidates, key=lambda item: item[2])
    return closest, outward, float(distance)


def static_hocbf_constraints(
    state: Sequence[float],
    environment: HospitalEnvironment,
    config: HospitalConfig,
) -> list[HocbfConstraint]:
    """Playground-equivalent HOCBFs for nearby walls and floor boundaries."""

    if not config.safety.enable_static_hocbf:
        return []
    value = np.asarray(state, dtype=float)
    position, velocity = value[:2], value[2:4]
    safe_distance = config.robot.radius + config.safety.static_hocbf_margin
    activation = safe_distance + config.safety.static_hocbf_activation_margin
    candidates: list[tuple[float, str, np.ndarray]] = []
    for index, wall in enumerate(environment.wall_rects):
        closest = _closest_point_on_rect(position, wall)
        distance = float(np.linalg.norm(position - closest))
        if distance <= activation:
            candidates.append((distance, f"wall-{index}", closest))
    for index, floor in enumerate(environment.floor_rects):
        closest, outward, distance = _closest_rect_boundary(position, floor)
        probe = closest + outward * (safe_distance + 0.15)
        if not environment.is_on_floor(probe) and distance <= activation:
            candidates.append((distance, f"floor-{index}", closest))

    output: list[HocbfConstraint] = []
    for _, identifier, closest in sorted(candidates, key=lambda item: item[0])[
        : config.safety.max_static_hocbf_constraints
    ]:
        relative_position = position - closest
        h = float(relative_position @ relative_position - safe_distance**2)
        h_dot = float(2.0 * relative_position @ velocity)
        psi1 = h_dot + config.safety.static_hocbf_lambda1 * h
        constant = (
            2.0 * float(velocity @ velocity)
            + config.safety.static_hocbf_lambda1 * h_dot
            + config.safety.static_hocbf_lambda2 * psi1
        )
        output.append(
            HocbfConstraint(
                a=2.0 * relative_position,
                b=-constant,
                label=f"hocbf:{identifier}:0",
                obstacle_id=identifier,
                proxy_index=0,
                h=h,
                h_dot=h_dot,
                psi1=psi1,
                safe_distance=safe_distance,
            )
        )
    return output


def current_hocbf_constraints(
    state: Sequence[float],
    obstacles: Sequence[DynamicObstacle],
    environment: HospitalEnvironment,
    config: HospitalConfig,
) -> list[HocbfConstraint]:
    return [
        *dynamic_hocbf_constraints(state, obstacles, config),
        *static_hocbf_constraints(state, environment, config),
    ]


def sensed_obstacles(
    state: Sequence[float],
    obstacles: Sequence[DynamicObstacle],
    config: HospitalConfig,
    *,
    environment: HospitalEnvironment | None = None,
) -> tuple[DynamicObstacle, ...]:
    """Return the nearest sensed objects used by certificates and HOCBFs.

    Physical collision checking continues to use the complete scene.  This
    mirrors the playground's local controller and keeps dense 50-human,
    15-stretcher scenes computationally meaningful.
    """

    position = np.asarray(state, dtype=float)[:2]
    nearby: list[tuple[float, DynamicObstacle]] = []
    for obstacle in obstacles:
        proxy_radius = (
            obstacle.radius
            if isinstance(obstacle, Human)
            else 0.5 * np.hypot(obstacle.length, obstacle.width)
        )
        range_distance = float(
            np.linalg.norm(obstacle.center - position) - proxy_radius
        )
        if range_distance > config.robot.sensing_range:
            continue
        if environment is not None and _line_of_sight_blocked(
                position,
                obstacle.center,
                environment,
                obstacles,
                obstacle.identifier,
                config,
            ):
            continue
        nearby.append((range_distance, obstacle))
    nearby.sort(key=lambda item: item[0])
    return tuple(
        obstacle
        for _, obstacle in nearby[: config.safety.max_obstacles]
    )


def _clip_polygon(
    polygon: list[np.ndarray], a: np.ndarray, b: float
) -> list[np.ndarray]:
    if not polygon:
        return []
    norm = float(np.linalg.norm(a))
    if norm < 1e-12:
        return polygon if b <= 0.0 else []

    def inside(point: np.ndarray) -> bool:
        return float(np.dot(a, point)) >= b - 1e-10

    def intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        direction = second - first
        denominator = float(np.dot(a, direction))
        if abs(denominator) < 1e-12:
            return first.copy()
        alpha = np.clip((b - float(np.dot(a, first))) / denominator, 0.0, 1.0)
        return first + alpha * direction

    output: list[np.ndarray] = []
    for index, current in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        current_inside, following_inside = inside(current), inside(following)
        if current_inside and following_inside:
            output.append(following.copy())
        elif current_inside and not following_inside:
            output.append(intersection(current, following))
        elif not current_inside and following_inside:
            output.extend(
                (intersection(current, following), following.copy())
            )
    return output


def _closest_on_segment(
    point: np.ndarray, first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    direction = second - first
    denominator = float(np.dot(direction, direction))
    if denominator < 1e-12:
        return first.copy()
    alpha = np.clip(
        float(np.dot(point - first, direction)) / denominator, 0.0, 1.0
    )
    return first + alpha * direction


def solve_control_halfspaces(
    nominal_control: Sequence[float],
    constraints: Sequence[HocbfConstraint],
    acceleration_limit: float,
) -> tuple[np.ndarray, bool]:
    """Exact 2-D projection onto a box intersected with HOCBF halfspaces."""

    limit = float(acceleration_limit)
    nominal = np.clip(np.asarray(nominal_control, dtype=float), -limit, limit)
    polygon = [
        np.array([-limit, -limit]),
        np.array([limit, -limit]),
        np.array([limit, limit]),
        np.array([-limit, limit]),
    ]
    for constraint in constraints:
        polygon = _clip_polygon(polygon, constraint.a, constraint.b)
        if not polygon:
            return nominal, False
    if all(constraint.margin(nominal) >= -1e-8 for constraint in constraints):
        return nominal, True

    candidates = [point.copy() for point in polygon]
    for index, current in enumerate(polygon):
        following = polygon[(index + 1) % len(polygon)]
        candidates.append(_closest_on_segment(nominal, current, following))
    best = min(candidates, key=lambda point: float(np.dot(point - nominal, point - nominal)))
    return np.clip(best, -limit, limit), True


class RoomPolicyProvider:
    """Stateless planner for room backup-policy paths."""

    def __init__(
        self,
        environment: HospitalEnvironment,
        planner: HospitalGridPlanner,
        config: HospitalConfig,
    ) -> None:
        self.environment = environment
        self.planner = planner
        self.config = config
        door_paths: list[
            tuple[Room, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        for room in environment.rooms:
            try:
                outside, door, inside, _terminal_center = (
                    environment.room_door_path(
                        room,
                        config.robot.radius,
                        config.refuge.inside_door_offset,
                        config.refuge.outside_door_offset,
                    )
                )
            except ValueError:
                continue
            door_paths.append((room, outside, door, inside))
        # Door geometry is immutable.  Caching it avoids rediscovering every
        # room's centered doorway on every 60 ms controller decision.
        self._door_paths = tuple(door_paths)

    @staticmethod
    def _distinct_path(
        points: Sequence[Sequence[float]],
        threshold: float = 0.35,
    ) -> list[np.ndarray]:
        output: list[np.ndarray] = []
        for point in points:
            candidate = np.asarray(point, dtype=float)
            if any(
                np.linalg.norm(candidate - existing) <= threshold
                for existing in output
            ):
                continue
            output.append(candidate.copy())
        return output

    def room_paths(
        self, position: Sequence[float]
    ) -> list[tuple[Room, list[np.ndarray]]]:
        start = np.asarray(position, dtype=float)
        # Match the playground: once any refuge room is occupied, no room
        # branch is offered.  The ordinary stop/directional/nominal policies
        # decide whether to remain or leave through their QP certificates.
        if self.environment.room_containing(start) is not None:
            return []
        sensing_range = self.config.robot.sensing_range
        geometries = [
            item
            for item in self._door_paths
            if (
                np.linalg.norm(item[2] - start) <= sensing_range
                and np.linalg.norm(item[3] - start) <= sensing_range
            )
        ]
        if not geometries:
            return []

        path_radius = self.config.robot.radius + 0.06
        direct_targets = np.asarray(
            [
                point
                for _room, outside, door, inside in geometries
                for point in (outside, door, inside)
            ],
            dtype=float,
        )
        direct_free = self.environment.segments_are_free(
            np.repeat(start[None, :], len(direct_targets), axis=0),
            direct_targets,
            path_radius,
            step=0.65,
        ).reshape(len(geometries), 3)

        raw_candidates: list[tuple[Room, list[np.ndarray]]] = []
        for (room, outside, door, inside), reachable in zip(
            geometries,
            direct_free,
            strict=True,
        ):
            outside_reachable, door_reachable, entry_reachable = (
                bool(value) for value in reachable
            )
            try:
                if door_reachable:
                    raw_path = [door, inside]
                elif outside_reachable:
                    raw_path = [outside, door, inside]
                elif entry_reachable:
                    raw_path = [inside]
                else:
                    planned = self.planner.plan(start, outside)
                    raw_path = [*planned[1:], door, inside]
            except ValueError:
                continue
            path = self._distinct_path(raw_path)
            if any(np.linalg.norm(point - start) > sensing_range for point in path):
                continue
            raw_candidates.append((room, path))

        segment_starts: list[np.ndarray] = []
        segment_ends: list[np.ndarray] = []
        segment_owners: list[int] = []
        for owner, (_room, path) in enumerate(raw_candidates):
            for first, second in zip(path, path[1:], strict=False):
                segment_starts.append(first)
                segment_ends.append(second)
                segment_owners.append(owner)
        valid = np.ones(len(raw_candidates), dtype=bool)
        if segment_starts:
            segment_free = self.environment.segments_are_free(
                np.asarray(segment_starts),
                np.asarray(segment_ends),
                path_radius,
                step=0.65,
            )
            for owner, free in zip(
                segment_owners,
                segment_free,
                strict=True,
            ):
                valid[owner] &= bool(free)

        candidates = []
        for (room, path), path_is_valid in zip(
            raw_candidates,
            valid,
            strict=True,
        ):
            if not path_is_valid:
                continue
            cost = sum(
                float(np.linalg.norm(path[index] - path[index - 1]))
                for index in range(1, len(path))
            )
            if path:
                cost += float(np.linalg.norm(path[0] - start))
            if cost > min(
                self.config.policies.room_search_radius,
                sensing_range,
            ):
                continue
            candidates.append((cost, room, path))
        candidates.sort(key=lambda item: item[0])
        return [(room, path) for _, room, path in candidates]


class HospitalController:
    """Localized controller interface suitable for later benchmark adapters."""

    def __init__(
        self,
        environment: HospitalEnvironment,
        planner: HospitalGridPlanner,
        config: HospitalConfig,
        initial_state: Sequence[float],
        goal: Sequence[float],
    ) -> None:
        self.environment = environment
        self.planner = planner
        self.config = config
        self.room_policy_provider = RoomPolicyProvider(
            environment, planner, config
        )
        self.goal = np.asarray(goal, dtype=float)
        self.navigation_path = self._plan_navigation_path(
            np.asarray(initial_state, dtype=float)[:2], self.goal
        )
        self.navigation_index = min(1, len(self.navigation_path) - 1)
        self._jax_geometry = pack_static_geometry(environment)
        self._jax_parameters = pack_parameters(config)
        maximum_obstacles = max(1, int(config.safety.max_obstacles))
        self._jax_obstacle_buckets = tuple(
            sorted(
                {
                    *(
                        bucket
                        for bucket in DEFAULT_OBSTACLE_BUCKETS
                        if bucket < maximum_obstacles
                    ),
                    maximum_obstacles,
                }
            )
        )

    def set_goal(
        self, state: Sequence[float], goal: Sequence[float]
    ) -> None:
        self.goal = np.asarray(goal, dtype=float)
        self.navigation_path = self._plan_navigation_path(
            np.asarray(state, dtype=float)[:2], self.goal
        )
        self.navigation_index = min(1, len(self.navigation_path) - 1)

    @staticmethod
    def _append_distinct(
        points: list[np.ndarray],
        point: Sequence[float],
        *,
        threshold: float = 0.35,
    ) -> None:
        candidate = np.asarray(point, dtype=float)
        if not points or np.linalg.norm(candidate - points[-1]) > threshold:
            points.append(candidate.copy())

    def _room_exit_route(self, room: Room) -> list[np.ndarray]:
        """Return the centered room -> inside -> door -> outside route.

        The room center is an ordinary nominal waypoint, not a refuge mode or
        hold target.  Including it makes an unscheduled room entry geometrically
        well posed from either side of the doorway before the route is aligned
        with the centered door and resumed outside.
        """

        try:
            outside, door, inside, _terminal_center = (
                self.environment.room_door_path(
                    room,
                    self.config.robot.radius,
                    self.config.refuge.inside_door_offset,
                    self.config.refuge.outside_door_offset,
                    # The playground nominal route uses radius + 0.08 for
                    # both its point and segment checks.
                    segment_clearance_buffer=0.08,
                )
            )
        except ValueError:
            return []
        route: list[np.ndarray] = []
        for point in (room.center, inside, door, outside):
            self._append_distinct(route, point)
        return route

    @staticmethod
    def _remaining_route_has_exit_suffix(
        remaining: Sequence[Sequence[float]],
        exit_route: Sequence[Sequence[float]],
        *,
        tolerance: float = 0.45,
    ) -> bool:
        """Return whether the active route starts with an exact exit suffix.

        A consumed center/inside/door waypoint is allowed, because the
        navigation index advances normally.  Unlike the old loose proximity
        test, unrelated corridor waypoints near a doorway cannot masquerade as
        an ordered room-exit route.
        """

        active = [np.asarray(point, dtype=float) for point in remaining]
        expected = [np.asarray(point, dtype=float) for point in exit_route]
        for consumed in range(len(expected)):
            suffix = expected[consumed:]
            if len(active) < len(suffix):
                continue
            if all(
                np.linalg.norm(active[index] - point) <= tolerance
                for index, point in enumerate(suffix)
            ):
                return True
        return False

    def _preserved_route_after_exit(
        self,
        room: Room,
        outside: np.ndarray,
        remaining: Sequence[Sequence[float]],
    ) -> list[np.ndarray]:
        """Bridge back to the first usable waypoint in the previous suffix."""

        old_suffix = [np.asarray(point, dtype=float).copy() for point in remaining]
        route_radius = self.config.robot.radius + 0.08
        for resume_index, candidate in enumerate(old_suffix):
            if room.contains(candidate, margin=-0.05):
                continue
            try:
                connector = self.planner.plan(outside, candidate)
            except ValueError:
                continue
            if not all(
                self.environment.segment_is_free(
                    start,
                    end,
                    route_radius,
                    step=0.55,
                )
                for start, end in zip(
                    connector,
                    connector[1:],
                    strict=False,
                )
            ):
                continue
            preserved: list[np.ndarray] = []
            for point in connector[1:]:
                self._append_distinct(preserved, point)
            for point in old_suffix[resume_index:]:
                self._append_distinct(preserved, point)
            return preserved

        try:
            fallback = self.planner.plan(outside, self.goal)
        except ValueError:
            return [self.goal.copy()]
        return [np.asarray(point, dtype=float).copy() for point in fallback[1:]]

    def _plan_navigation_path(
        self,
        start: Sequence[float],
        goal: Sequence[float],
    ) -> list[np.ndarray]:
        """Plan a nominal path with the playground's room-door alignment."""

        start_point = np.asarray(start, dtype=float)
        goal_point = np.asarray(goal, dtype=float)
        current_room = self.environment.room_containing(start_point)
        goal_room = self.environment.room_containing(goal_point)
        if current_room is not None and current_room is not goal_room:
            exit_route = self._room_exit_route(current_room)
            if exit_route:
                exit_plan = self.planner.plan(exit_route[-1], goal_point)
                path = [start_point.copy()]
                for point in exit_route:
                    self._append_distinct(path, point)
                for point in exit_plan[1:]:
                    if current_room.contains(point, margin=-0.05):
                        continue
                    self._append_distinct(path, point)
                return path
        return self.planner.plan(start_point, goal_point)

    def _ensure_room_exit_waypoints(self, state: Sequence[float]) -> bool:
        """Repair a stale nominal route after an unscheduled room entry.

        This is a geometry-only planner consistency operation, equivalent to
        ``_ensureRoomExitWaypoint`` in the playground.  It is not a refuge
        state, controller switch, or obstacle-dependent guard.
        """

        point = np.asarray(state, dtype=float)[:2]
        room = self.environment.room_containing(point)
        goal_room = self.environment.room_containing(self.goal)
        if room is None or room is goal_room:
            return False
        exit_route = self._room_exit_route(room)
        if not exit_route:
            return False
        start_index = min(
            self.navigation_index,
            max(0, len(self.navigation_path) - 1),
        )
        remaining = self.navigation_path[start_index:]
        if self._remaining_route_has_exit_suffix(remaining, exit_route):
            return False

        repaired: list[np.ndarray] = [point.copy()]
        for waypoint in exit_route:
            self._append_distinct(repaired, waypoint)
        for waypoint in self._preserved_route_after_exit(
            room,
            np.asarray(exit_route[-1], dtype=float),
            remaining,
        ):
            self._append_distinct(repaired, waypoint)
        self.navigation_path = repaired
        self.navigation_index = min(1, len(self.navigation_path) - 1)
        return True

    def _navigation_target(self, state: np.ndarray) -> np.ndarray:
        self._ensure_room_exit_waypoints(state)
        while (
            self.navigation_index < len(self.navigation_path) - 1
            and np.linalg.norm(
                state[:2] - self.navigation_path[self.navigation_index]
            )
            <= 1.05
        ):
            self.navigation_index += 1
        return self.navigation_path[self.navigation_index]

    def candidate_policies(
        self, state: Sequence[float]
    ) -> list[HospitalPolicy]:
        value = np.asarray(state, dtype=float)
        self._ensure_room_exit_waypoints(value)
        room_paths = self.room_policy_provider.room_paths(value[:2])
        return build_policy_library(
            value,
            self.navigation_path[self.navigation_index :],
            room_paths,
            self.environment,
            self.config,
        )

    def evaluate_candidates(
        self,
        state: Sequence[float],
        obstacles: Sequence[DynamicObstacle],
    ) -> list[PolicyEvaluation]:
        prediction_cache: dict[tuple[int, float], DynamicObstacle] = {}
        return [
            rollout_value(
                policy,
                state,
                obstacles,
                self.environment,
                self.config,
                prediction_cache,
            )
            for policy in self.candidate_policies(state)
        ]

    def _active_policy_candidates(
        self,
        state: np.ndarray,
        obstacles: Sequence[DynamicObstacle] | None = None,
    ) -> list[HospitalPolicy]:
        """Return the complete current library.

        ``obstacles`` is accepted for API compatibility but never gates policy
        availability.  In particular, a previous room choice cannot latch or
        collapse the library.
        """

        del obstacles
        return self.candidate_policies(state)

    def _trajectory_min_clearance(
        self,
        trajectory: np.ndarray,
        policy: HospitalPolicy,
        obstacles: Sequence[DynamicObstacle],
        stop_index: int | None = None,
        start_index: int = 0,
    ) -> float:
        count = len(trajectory) if stop_index is None else min(len(trajectory), stop_index)
        values: list[float] = []
        for index, rollout_state in enumerate(trajectory[:count]):
            elapsed = (start_index + index) * policy.rollout_dt
            values.append(
                self.environment.static_clearance(
                    rollout_state[:2],
                    self.config.robot.radius
                    + self.config.safety.static_margin,
                )
            )
            values.extend(
                obstacle_clearance(
                    obstacle.predicted(elapsed, self.environment),
                    rollout_state[:2],
                    self.config.robot.radius
                    + self.config.safety.safety_margin,
                    self.config.safety.human_margin,
                    self.config.safety.stretcher_margin,
                )
                for obstacle in obstacles
            )
        return min(values, default=float("inf"))

    @staticmethod
    def jax_cache_info() -> dict[str, dict[str, int | None]]:
        """Return process-wide compile-cache counters for benchmark audits."""

        def normalize(info: object) -> dict[str, int | None]:
            return {
                name: getattr(info, name, None)
                for name in ("hits", "misses", "maxsize", "currsize")
            }

        return {
            "batch": normalize(compiled_evaluator_cache_info()),
            "grouped": normalize(compiled_grouped_evaluator_cache_info()),
        }

    def _uses_retrace_policy(
        self,
        policies: Sequence[HospitalPolicy],
    ) -> bool:
        return any(policy.kind == "retrace" for policy in policies)

    def _jax_batch_capacity(
        self,
        policies: Sequence[HospitalPolicy],
        obstacle_bucket: int,
    ) -> HospitalJaxCapacities:
        """Return the exact fixed shape for a retrace-containing batch."""

        step_counts = tuple(
            max(1, int(policy.horizon / policy.rollout_dt))
            for policy in policies
        )
        prediction_time = max(
            steps * policy.rollout_dt
            for steps, policy in zip(step_counts, policies, strict=True)
        ) + max(
            self.config.dt,
            self.config.policies.time_derivative_step,
        )
        return HospitalJaxCapacities(
            max_policies=max(1, len(policies)),
            max_obstacles=obstacle_bucket,
            max_horizon_steps=max(step_counts),
            max_swept_samples=(
                3 if any(policy.kind == "room" for policy in policies) else 2
            ),
            human_prediction_steps=max(1, int(np.ceil(prediction_time / 0.05))),
        )

    def _jax_policy_evaluation(
        self,
        state: np.ndarray,
        nominal_control: np.ndarray,
        policies: Sequence[HospitalPolicy],
        obstacles: Sequence[DynamicObstacle],
        *,
        include_diagnostics: bool,
    ):
        """Run the smallest precompiled fixed-shape numerical backend."""

        obstacle_bucket = select_obstacle_bucket(
            len(obstacles), self._jax_obstacle_buckets
        )
        if self._uses_retrace_policy(policies):
            capacity = self._jax_batch_capacity(
                policies,
                obstacle_bucket,
            )
            packed_policies = pack_policy_batch(
                policies, self.config, capacity
            )
            packed_obstacles = pack_obstacle_batch(obstacles, capacity)
            return evaluate_policy_batch(
                state,
                nominal_control,
                packed_policies,
                packed_obstacles,
                self._jax_geometry,
                self._jax_parameters,
                include_diagnostics=include_diagnostics,
            )

        capacities = grouped_capacities_for_obstacle_count(
            self.config,
            len(obstacles),
            buckets=self._jax_obstacle_buckets,
        )
        packed_policies = pack_policy_groups(
            policies, self.config, capacities
        )
        packed_obstacles = pack_obstacle_batch(obstacles, capacities)
        return evaluate_policy_groups(
            state,
            nominal_control,
            packed_policies,
            packed_obstacles,
            self._jax_geometry,
            self._jax_parameters,
            include_diagnostics=include_diagnostics,
        )

    def warmup_certificate_oracle(
        self,
        state: Sequence[float],
        *,
        policies: Sequence[HospitalPolicy] | None = None,
        nominal_control: Sequence[float] | None = None,
        include_diagnostics: bool = False,
    ) -> dict[str, dict[str, int | None]]:
        """Compile every obstacle bucket without changing navigation state."""

        saved_path = [point.copy() for point in self.navigation_path]
        saved_index = int(self.navigation_index)
        saved_goal = self.goal.copy()
        try:
            value = np.asarray(state, dtype=float)
            candidates = (
                self.candidate_policies(value)
                if policies is None
                else list(policies)
            )
            if not candidates:
                return self.jax_cache_info()
            if nominal_control is None:
                nominal = candidates[0].control(value, self.config)
            else:
                nominal = np.asarray(nominal_control, dtype=float)
            retrace = self._uses_retrace_policy(candidates)
            for obstacle_bucket in self._jax_obstacle_buckets:
                if retrace:
                    capacity = self._jax_batch_capacity(
                        candidates,
                        obstacle_bucket,
                    )
                    packed_policies = pack_policy_batch(
                        candidates, self.config, capacity
                    )
                    packed_obstacles = pack_obstacle_batch((), capacity)
                    warmup_policy_batch(
                        value,
                        nominal,
                        packed_policies,
                        packed_obstacles,
                        self._jax_geometry,
                        self._jax_parameters,
                        include_diagnostics=include_diagnostics,
                    )
                else:
                    capacities = grouped_capacities_for_obstacle_count(
                        self.config,
                        obstacle_bucket,
                        buckets=(obstacle_bucket,),
                    )
                    packed_policies = pack_policy_groups(
                        candidates, self.config, capacities
                    )
                    packed_obstacles = pack_obstacle_batch((), capacities)
                    warmup_policy_groups(
                        value,
                        nominal,
                        packed_policies,
                        packed_obstacles,
                        self._jax_geometry,
                        self._jax_parameters,
                        include_diagnostics=include_diagnostics,
                    )
            return self.jax_cache_info()
        finally:
            self.navigation_path = saved_path
            self.navigation_index = saved_index
            self.goal = saved_goal

    def _certificate_from_jax(
        self,
        state: np.ndarray,
        policy: HospitalPolicy,
        value: float,
        trajectory: np.ndarray,
        gradient: np.ndarray,
        value_time_derivative: float,
        nominal_prefix_value: float,
        terminal_clearance: float,
        diagnostics_available: bool,
        hocbf_constraints: Sequence[HocbfConstraint],
    ) -> PolicyCbfEvaluation:
        """Build the public certificate object from one batched JAX row."""

        clipped_gradient = np.asarray(gradient, dtype=float).copy()
        gradient_norm = float(np.linalg.norm(clipped_gradient))
        if gradient_norm > self.config.policies.max_gradient_norm:
            clipped_gradient *= (
                self.config.policies.max_gradient_norm / gradient_norm
            )
        drift = np.array([state[2], state[3], 0.0, 0.0])
        control_matrix = np.array(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        )
        backup_control = policy.control(state, self.config)
        terminal_safe: bool | None
        prefix_safe: bool | None
        prefix_value: float | None
        if diagnostics_available:
            terminal_safe = bool(terminal_clearance >= 0.0)
            if policy.kind == "room" and policy.target_room is not None:
                terminal = trajectory[-1]
                terminal_safe = bool(
                    terminal_safe
                    and policy.target_room.contains(terminal[:2])
                    and policy.target_room.interior_margin(terminal[:2])
                    >= self.config.refuge.terminal_interior_margin
                    and np.linalg.norm(terminal[2:4])
                    <= self.config.refuge.terminal_speed_max
                )
            prefix_safe = bool(nominal_prefix_value >= 0.0)
            prefix_value = float(nominal_prefix_value)
        else:
            terminal_safe = None
            prefix_safe = None
            prefix_value = None
        metadata = {
            "rollout_safe": bool(value >= 0.0),
            "terminal_safe": terminal_safe,
            "nominal_prefix_safe": prefix_safe,
            "nominal_prefix_value": prefix_value,
            "rollout_diagnostics_available": diagnostics_available,
            "terminal_cost": float(
                np.linalg.norm(trajectory[-1, :2] - self.goal)
                + 0.1 * np.linalg.norm(trajectory[-1, 2:4])
            ),
        }
        base_certificate = PolicyCertificate.from_cbf(
            policy.name,
            value=value,
            gradient=clipped_gradient,
            drift=drift,
            control_matrix=control_matrix,
            value_time_derivative=value_time_derivative,
            alpha=self.config.policies.cbf_alpha,
            buffer=self.config.policies.cbf_value_buffer,
            backup_control=backup_control,
            metadata=metadata,
        )
        certificate = PolicyCertificate(
            policy_id=base_certificate.policy_id,
            value=base_certificate.value,
            halfspaces=(
                *base_certificate.halfspaces,
                *(
                    CBFHalfspace(
                        constraint.a,
                        constraint.b,
                        constraint.label,
                    )
                    for constraint in hocbf_constraints
                ),
            ),
            backup_control=base_certificate.backup_control,
            metadata=metadata,
        )
        return PolicyCbfEvaluation(
            policy=policy,
            value=float(value),
            trajectory=np.asarray(trajectory, dtype=float),
            gradient=clipped_gradient,
            value_time_derivative=float(value_time_derivative),
            certificate=certificate,
        )

    def _certificate_for_policy(
        self,
        state: np.ndarray,
        policy: HospitalPolicy,
        obstacles: Sequence[DynamicObstacle],
        hocbf_constraints: Sequence[HocbfConstraint],
        nominal_control: np.ndarray,
        prediction_cache: dict[tuple[int, float], DynamicObstacle],
    ) -> PolicyCbfEvaluation:
        base = rollout_value(
            policy,
            state,
            obstacles,
            self.environment,
            self.config,
            prediction_cache,
        )

        def value_at(
            candidate_state: np.ndarray,
            candidate_obstacles: Sequence[DynamicObstacle] = obstacles,
        ) -> float:
            return rollout_value(
                policy,
                candidate_state,
                candidate_obstacles,
                self.environment,
                self.config,
                (
                    prediction_cache
                    if candidate_obstacles is obstacles
                    else None
                ),
            ).value

        gradient = np.zeros(4, dtype=float)
        for index, step in enumerate(self.config.policies.gradient_steps):
            plus = state.copy()
            minus = state.copy()
            plus[index] += step
            minus[index] -= step
            gradient[index] = (
                value_at(plus) - value_at(minus)
            ) / (2.0 * step)
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm > self.config.policies.max_gradient_norm:
            gradient *= (
                self.config.policies.max_gradient_norm / gradient_norm
            )
        time_step = self.config.policies.time_derivative_step
        shifted_value = rollout_value(
            policy,
            state,
            obstacles,
            self.environment,
            self.config,
            prediction_cache,
            time_offset=time_step,
        ).value
        value_time_derivative = (shifted_value - base.value) / time_step
        drift = np.array([state[2], state[3], 0.0, 0.0])
        control_matrix = np.array(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        )
        backup_control = policy.control(state, self.config)
        terminal_clearance = self._trajectory_min_clearance(
            base.trajectory[-1:],
            policy,
            obstacles,
            start_index=len(base.trajectory) - 1,
        )
        terminal_safe = terminal_clearance >= 0.0
        if policy.kind == "room" and policy.target_room is not None:
            terminal = base.trajectory[-1]
            terminal_safe = bool(
                terminal_safe
                and policy.target_room.contains(terminal[:2])
                and policy.target_room.interior_margin(terminal[:2])
                >= self.config.refuge.terminal_interior_margin
                and np.linalg.norm(terminal[2:4])
                <= self.config.refuge.terminal_speed_max
            )
        # This diagnostic has the PL-CBF interpretation: execute the nominal
        # input for one control interval, then ask whether this backup still
        # has a safe rollout from the resulting state and obstacle time.
        prefix_state = step_double_integrator(
            state,
            nominal_control,
            self.config.dt,
            self.config.robot,
        )
        nominal_prefix_value = rollout_value(
            policy,
            prefix_state,
            obstacles,
            self.environment,
            self.config,
            prediction_cache,
            time_offset=self.config.dt,
        ).value
        metadata = {
            "rollout_safe": bool(base.value >= 0.0),
            "terminal_safe": terminal_safe,
            "nominal_prefix_safe": bool(nominal_prefix_value >= 0.0),
            "nominal_prefix_value": float(nominal_prefix_value),
            "terminal_cost": float(
                np.linalg.norm(base.trajectory[-1, :2] - self.goal)
                + 0.1 * np.linalg.norm(base.trajectory[-1, 2:4])
            ),
        }
        base_certificate = PolicyCertificate.from_cbf(
            policy.name,
            value=base.value,
            gradient=gradient,
            drift=drift,
            control_matrix=control_matrix,
            value_time_derivative=value_time_derivative,
            alpha=self.config.policies.cbf_alpha,
            buffer=self.config.policies.cbf_value_buffer,
            backup_control=backup_control,
            metadata=metadata,
        )
        joint_halfspaces = (
            *base_certificate.halfspaces,
            *(
                CBFHalfspace(
                    constraint.a,
                    constraint.b,
                    constraint.label,
                )
                for constraint in hocbf_constraints
            ),
        )
        certificate = PolicyCertificate(
            policy_id=base_certificate.policy_id,
            value=base_certificate.value,
            halfspaces=joint_halfspaces,
            backup_control=base_certificate.backup_control,
            metadata=metadata,
        )
        return PolicyCbfEvaluation(
            policy=policy,
            value=base.value,
            trajectory=base.trajectory,
            gradient=gradient,
            value_time_derivative=value_time_derivative,
            certificate=certificate,
        )

    def build_policy_certificates(
        self,
        state: Sequence[float],
        obstacles: Sequence[DynamicObstacle],
        policies: Sequence[HospitalPolicy] | None = None,
        nominal_control: Sequence[float] | None = None,
        *,
        obstacles_are_sensed: bool = False,
        include_diagnostics: bool = True,
        hocbf_constraints: Sequence[HocbfConstraint] | None = None,
    ) -> tuple[tuple[PolicyCertificate, ...], tuple[PolicyCbfEvaluation, ...]]:
        """Build rollout-derived PL-CBF certificates for benchmark adapters."""

        value = np.asarray(state, dtype=float)
        active_obstacles = (
            tuple(obstacles)
            if obstacles_are_sensed
            else sensed_obstacles(
                value,
                obstacles,
                self.config,
                environment=self.environment,
            )
        )
        hocbf = list(
            current_hocbf_constraints(
                value,
                active_obstacles,
                self.environment,
                self.config,
            )
            if hocbf_constraints is None
            else hocbf_constraints
        )
        candidates = (
            self._active_policy_candidates(value, obstacles)
            if policies is None
            else list(policies)
        )
        if not candidates:
            return (), ()
        if nominal_control is None:
            nominal = candidates[0].control(value, self.config)
        else:
            nominal = np.asarray(nominal_control, dtype=float)
        if all(
            isinstance(obstacle, (Human, Stretcher))
            for obstacle in active_obstacles
        ):
            batched = self._jax_policy_evaluation(
                value,
                nominal,
                candidates,
                active_obstacles,
                include_diagnostics=include_diagnostics,
            )
            if batched.names != tuple(policy.name for policy in candidates):
                raise RuntimeError("JAX policy order does not match the library")
            evaluations = tuple(
                self._certificate_from_jax(
                    value,
                    policy,
                    float(batched.values[index]),
                    batched.trajectories[index][
                        batched.trajectory_mask[index]
                    ],
                    batched.gradients[index],
                    float(batched.time_derivatives[index]),
                    float(batched.nominal_prefix_values[index]),
                    float(batched.terminal_clearances[index]),
                    batched.diagnostics_available,
                    hocbf,
                )
                for index, policy in enumerate(candidates)
            )
        else:
            # Preserve the public DynamicObstacle protocol for custom user
            # objects; benchmark Humans and Stretchers always take the JIT
            # path above.
            prediction_cache: dict[tuple[int, float], DynamicObstacle] = {}
            evaluations = tuple(
                self._certificate_for_policy(
                    value,
                    policy,
                    active_obstacles,
                    hocbf,
                    nominal,
                    prediction_cache,
                )
                for policy in candidates
            )
        return (
            tuple(evaluation.certificate for evaluation in evaluations),
            evaluations,
        )

    def certificate_oracle(
        self,
        state: Sequence[float],
        obstacles: Sequence[DynamicObstacle],
        policies: Sequence[HospitalPolicy] | None = None,
    ) -> list[PolicyCertificate]:
        """Public policy-certificate oracle used by common benchmarks."""

        certificates, _ = self.build_policy_certificates(
            state, obstacles, policies
        )
        return list(certificates)

    def compute(
        self,
        state: Sequence[float],
        obstacles: Sequence[DynamicObstacle],
        time: float,
        *,
        obstacles_are_sensed: bool = False,
        include_rollout_diagnostics: bool = True,
    ) -> ControllerResult:
        del time
        value = np.asarray(state, dtype=float)
        target = self._navigation_target(value)
        nominal = waypoint_control(
            value,
            target,
            self.config.robot,
            self.config.policies.nominal_target_speed,
        )
        selected = "nominal"

        active_obstacles = (
            tuple(obstacles)
            if obstacles_are_sensed
            else sensed_obstacles(
                value,
                obstacles,
                self.config,
                environment=self.environment,
            )
        )
        constraints = current_hocbf_constraints(
            value,
            active_obstacles,
            self.environment,
            self.config,
        )
        policies = self._active_policy_candidates(value, obstacles)
        if active_obstacles or constraints or include_rollout_diagnostics:
            certificates, evaluations = self.build_policy_certificates(
                value,
                active_obstacles,
                policies,
                nominal_control=nominal,
                obstacles_are_sensed=True,
                include_diagnostics=include_rollout_diagnostics,
                hocbf_constraints=constraints,
            )
        else:
            # Headless benchmarks do not need rollout drawings at a step where
            # the playground applies the nominal tracker without any active
            # safety row.  Keep one finite diagnostic certificate so the
            # public decision object remains well formed.
            nominal_certificate = PolicyCertificate(
                policy_id="nominal",
                value=0.0,
                backup_control=np.clip(
                    nominal,
                    -self.config.robot.a_max,
                    self.config.robot.a_max,
                ),
                metadata={
                    "rollout_diagnostics_available": False,
                    "inactive_safety_filter": True,
                },
            )
            certificates = (nominal_certificate,)
            evaluations = ()
        limit = self.config.robot.a_max
        emergency_hocbf_feasible: bool | None = None
        if active_obstacles or constraints:
            raw_decision = select_policy(
                certificates,
                nominal_control=nominal,
                lower=np.array([-limit, -limit]),
                upper=np.array([limit, limit]),
                mode="input_volume",
                safe_value_threshold=self.config.policies.safe_value_threshold,
                fallback_control=np.clip(
                    -self.config.policies.stop_gain * value[2:4],
                    -limit,
                    limit,
                ),
                tolerance=self.config.policies.constraint_tolerance,
            )
            diagnostics = raw_decision.diagnostics
            by_id = {
                certificate.policy_id: certificate
                for certificate in certificates
            }
            strict_eligible = [
                item
                for item in diagnostics.evaluations
                if (
                    item.safe_value
                    and item.feasible
                    and item.input_volume
                    > self.config.policies.constraint_tolerance
                )
            ]
            if strict_eligible:
                # Playground tie order: input volume, then V, then original
                # policy-library order.  No intervention-cost or policy-ID
                # tie-break is introduced.
                best = strict_eligible[0]
                for item in strict_eligible[1:]:
                    if item.input_volume > best.input_volume + 1e-8:
                        best = item
                    elif (
                        abs(item.input_volume - best.input_volume) <= 1e-8
                        and item.value > best.value + 1e-8
                    ):
                        best = item
                certificate = by_id[best.policy_id]
                assert best.control is not None
                decision = PolicyDecision(
                    certificate=certificate,
                    control=best.control,
                    diagnostics=DecisionDiagnostics(
                        mode=SelectionMode.INPUT_VOLUME,
                        selected_policy_id=certificate.policy_id,
                        used_fallback=False,
                        fallback_reason=None,
                        fallback_source=None,
                        safe_policy_count=diagnostics.safe_policy_count,
                        feasible_policy_count=(
                            diagnostics.feasible_policy_count
                        ),
                        eligible_policy_count=len(strict_eligible),
                        evaluations=diagnostics.evaluations,
                    ),
                )
                selected = certificate.policy_id
            else:
                # Match the playground's no-feasible-policy path: choose the
                # first maximum-V thresholded policy (or first maximum-V
                # policy if none meets the threshold) and project only its
                # direct backup action through the current HOCBFs.
                thresholded = [
                    item
                    for item in diagnostics.evaluations
                    if item.safe_value and item.policy_id in by_id
                ]
                emergency_pool = thresholded or [
                    item
                    for item in diagnostics.evaluations
                    if item.policy_id in by_id
                    and by_id[item.policy_id].valid
                    and np.isfinite(item.value)
                ]
                emergency_evaluation = emergency_pool[0]
                for item in emergency_pool[1:]:
                    if item.value > emergency_evaluation.value + 1e-8:
                        emergency_evaluation = item
                emergency_certificate = by_id[
                    emergency_evaluation.policy_id
                ]
                emergency_reference = (
                    emergency_certificate.backup_control
                    if emergency_certificate.backup_control is not None
                    else np.clip(
                        -self.config.policies.stop_gain * value[2:4],
                        -limit,
                        limit,
                    )
                )
                emergency_control, emergency_hocbf_feasible = (
                    solve_control_halfspaces(
                        emergency_reference,
                        constraints,
                        limit,
                    )
                )
                decision = PolicyDecision(
                    certificate=emergency_certificate,
                    control=emergency_control,
                    diagnostics=DecisionDiagnostics(
                        mode=SelectionMode.INPUT_VOLUME,
                        selected_policy_id=(
                            emergency_certificate.policy_id
                        ),
                        used_fallback=True,
                        fallback_reason=(
                            "no_positive_input_volume_policy"
                        ),
                        fallback_source="selected_policy_backup",
                        safe_policy_count=diagnostics.safe_policy_count,
                        feasible_policy_count=(
                            diagnostics.feasible_policy_count
                        ),
                        eligible_policy_count=0,
                        evaluations=diagnostics.evaluations,
                    ),
                )
                selected = emergency_certificate.policy_id
        else:
            # The playground leaves the route tracker unconstrained when no
            # sensed obstacle or active HOCBF exists.  Rollouts above are kept
            # solely for visualization/diagnostics in this branch.
            nominal_certificate = next(
                certificate
                for certificate in certificates
                if certificate.policy_id == "nominal"
            )
            decision = PolicyDecision(
                certificate=nominal_certificate,
                control=np.clip(nominal, -limit, limit),
                diagnostics=DecisionDiagnostics(
                    mode=SelectionMode.INPUT_VOLUME,
                    selected_policy_id="nominal",
                    used_fallback=False,
                    fallback_reason=None,
                    fallback_source=None,
                    safe_policy_count=0,
                    feasible_policy_count=0,
                    eligible_policy_count=0,
                    evaluations=(),
                ),
            )
        control = np.asarray(decision.control, dtype=float)
        if decision.diagnostics.used_fallback:
            if emergency_hocbf_feasible is None:
                # Generic no-safe/no-feasible-policy path: execute that
                # policy's direct backup through current HOCBFs only.
                control, feasible = solve_control_halfspaces(
                    control,
                    constraints,
                    self.config.robot.a_max,
                )
            else:
                feasible = emergency_hocbf_feasible
        else:
            # Normal policy QPs already contain all current HOCBF rows.
            feasible = True
        if not feasible:
            selected = f"{selected}:hocbf_infeasible"
        margins = [constraint.margin(control) for constraint in constraints]
        return ControllerResult(
            control=control,
            nominal_control=nominal,
            selected_policy=selected,
            inside_refuge=bool(
                self.environment.room_containing(value[:2]) is not None
            ),
            hocbf_constraints=tuple(constraints),
            feasible=feasible,
            min_hocbf_margin=min(margins, default=float("inf")),
            decision=decision,
            certificates=certificates,
            policy_evaluations=evaluations,
            candidate_policy_count=len(policies),
        )
