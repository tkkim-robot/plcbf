"""Backup-policy definitions and rollout safety values."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, pi, sin
from typing import Sequence

import numpy as np

from .config import HospitalConfig
from .dynamics import step_double_integrator, waypoint_control
from .environment import HospitalEnvironment, Room
from .obstacles import DynamicObstacle, Human, obstacle_clearance


MAX_POLICY_WAYPOINTS = 18
ROOM_APPROACH_RADIUS = 1.8
ROOM_WAYPOINT_RADIUS = 1.5
NOMINAL_WAYPOINT_RADIUS = 2.8
RETRACE_WAYPOINT_RADIUS = 1.0


def _angle_normalize(value: float) -> float:
    return float((value + pi) % (2.0 * pi) - pi)


def _limit_room_path(
    points: Sequence[Sequence[float]],
) -> list[np.ndarray]:
    """Limit a planned room path without ever discarding its terminal tail."""

    path = [np.asarray(point, dtype=float).copy() for point in points]
    if len(path) <= MAX_POLICY_WAYPOINTS:
        return path
    tail_count = min(4, len(path))
    head_count = MAX_POLICY_WAYPOINTS - tail_count
    output: list[np.ndarray] = []
    for point in [*path[:head_count], *path[-tail_count:]]:
        if any(np.linalg.norm(point - existing) <= 0.35 for existing in output):
            continue
        output.append(point)
    return output[:MAX_POLICY_WAYPOINTS]


def _smooth_min(
    values: Sequence[float], temperature: float, empty_value: float = 100.0
) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if finite.size == 0:
        return empty_value
    minimum = float(np.min(finite))
    return minimum - float(
        np.log(np.maximum(np.exp(-temperature * (finite - minimum)).sum(), 1e-12))
        / temperature
    )


@dataclass
class HospitalPolicy:
    name: str
    kind: str
    horizon: float
    rollout_dt: float
    target_speed: float = 0.0
    angle: float | None = None
    waypoints: list[np.ndarray] = field(default_factory=list)
    target_room: Room | None = None
    max_rollout_distance: float | None = None
    feedback_gain: float | None = None

    def control_with_cursor(
        self,
        state: Sequence[float],
        config: HospitalConfig,
        waypoint_index: int,
    ) -> tuple[np.ndarray, int]:
        """Evaluate a warehouse-style retrace policy with a local cursor.

        The cursor belongs to one independently simulated backup rollout.  It
        is returned to the caller rather than stored on the policy, so finite
        differences and competing MPS/Gatekeeper candidates cannot leak
        waypoint progress into one another.
        """

        if self.kind != "retrace" or not self.waypoints:
            return self.control(state, config), int(waypoint_index)
        value = np.asarray(state, dtype=float)
        index = int(np.clip(waypoint_index, 0, len(self.waypoints) - 1))
        target = self.waypoints[index]
        distance = float(np.linalg.norm(target - value[:2]))
        if (
            distance < RETRACE_WAYPOINT_RADIUS
            and index + 1 < len(self.waypoints)
        ):
            index += 1
            target = self.waypoints[index]
            distance = float(np.linalg.norm(target - value[:2]))

        direction = (target - value[:2]) / (distance + 1e-6)
        braking_speed = np.sqrt(
            2.0 * config.robot.a_max * max(distance, 0.0)
        )
        desired_speed = min(
            self.target_speed,
            braking_speed,
            config.robot.v_max,
        )
        desired_velocity = direction * desired_speed
        gain = (
            config.robot.k_velocity
            if self.feedback_gain is None
            else float(self.feedback_gain)
        )
        control = np.clip(
            gain * (desired_velocity - value[2:4]),
            -config.robot.a_max,
            config.robot.a_max,
        )
        return control, index

    def control(
        self,
        state: Sequence[float],
        config: HospitalConfig,
    ) -> np.ndarray:
        value = np.asarray(state, dtype=float)
        if self.kind == "retrace" and self.waypoints:
            control, _ = self.control_with_cursor(value, config, 0)
            return control
        if self.kind in {"nominal", "room"} and self.waypoints:
            target = self.waypoints[-1]
            for index, waypoint in enumerate(self.waypoints):
                if self.kind == "room":
                    radius = (
                        ROOM_APPROACH_RADIUS
                        if index == 0
                        else ROOM_WAYPOINT_RADIUS
                    )
                else:
                    radius = NOMINAL_WAYPOINT_RADIUS
                if np.linalg.norm(waypoint - value[:2]) > radius:
                    target = waypoint
                    break
            return waypoint_control(
                value, target, config.robot, self.target_speed
            )
        if self.kind == "stop":
            return np.clip(
                -config.policies.stop_gain * value[2:4],
                -config.robot.a_max,
                config.robot.a_max,
            )
        if self.kind in {"angle", "reverse"}:
            assert self.angle is not None
            desired_velocity = self.target_speed * np.array(
                [cos(self.angle), sin(self.angle)]
            )
            return np.clip(
                config.robot.k_velocity * (desired_velocity - value[2:4]),
                -config.robot.a_max,
                config.robot.a_max,
            )
        return np.zeros(2)


@dataclass(frozen=True)
class PolicyEvaluation:
    policy: HospitalPolicy
    value: float
    trajectory: np.ndarray


def rollout_policy(
    policy: HospitalPolicy,
    state: Sequence[float],
    config: HospitalConfig,
) -> np.ndarray:
    steps = max(1, int(policy.horizon / policy.rollout_dt))
    origin = np.asarray(state, dtype=float)
    trajectory = [origin.copy()]
    retrace_waypoint_index = 0
    for _ in range(steps):
        current = trajectory[-1]
        if policy.kind == "retrace":
            control, retrace_waypoint_index = policy.control_with_cursor(
                current,
                config,
                retrace_waypoint_index,
            )
        else:
            control = policy.control(current, config)
        following = step_double_integrator(
            current, control, policy.rollout_dt, config.robot
        )
        if (
            policy.max_rollout_distance is not None
            and np.linalg.norm(following[:2] - origin[:2])
            > policy.max_rollout_distance
        ):
            break
        trajectory.append(following)
        if (
            policy.kind == "room"
            and policy.target_room is not None
            and policy.target_room.contains(following[:2])
            and policy.target_room.interior_margin(following[:2])
            >= config.refuge.terminal_interior_margin
        ):
            break
    return np.asarray(trajectory)


def rollout_value(
    policy: HospitalPolicy,
    state: Sequence[float],
    obstacles: Sequence[DynamicObstacle],
    environment: HospitalEnvironment,
    config: HospitalConfig,
    prediction_cache: dict[tuple[int, float], DynamicObstacle] | None = None,
    *,
    time_offset: float = 0.0,
) -> PolicyEvaluation:
    trajectory = rollout_policy(policy, state, config)
    values: list[float] = []

    def predicted_obstacle(
        obstacle: DynamicObstacle,
        elapsed: float,
    ) -> DynamicObstacle:
        if prediction_cache is None:
            return obstacle.predicted(elapsed, environment)
        key = (id(obstacle), round(float(elapsed), 10))
        if key not in prediction_cache:
            prior = [
                (cached_time, cached_obstacle)
                for (obstacle_id, cached_time), cached_obstacle
                in prediction_cache.items()
                if obstacle_id == id(obstacle) and cached_time <= key[1]
            ]
            if prior:
                cached_time, cached_obstacle = max(
                    prior, key=lambda item: item[0]
                )
                prediction_cache[key] = cached_obstacle.predicted(
                    key[1] - cached_time,
                    environment,
                )
            else:
                prediction_cache[key] = obstacle.predicted(
                    key[1], environment
                )
        return prediction_cache[key]

    for index, rollout_state in enumerate(trajectory):
        elapsed = time_offset + index * policy.rollout_dt
        point = rollout_state[:2]
        static = environment.static_clearance(
            point, config.robot.radius + config.safety.static_margin
        )
        dynamic_values = []
        for obstacle in obstacles:
            predicted = predicted_obstacle(obstacle, elapsed)
            dynamic_values.append(
                obstacle_clearance(
                    predicted,
                    point,
                    config.robot.radius + config.safety.safety_margin,
                    config.safety.human_margin,
                    config.safety.stretcher_margin,
                )
            )
        dynamic = _smooth_min(dynamic_values, 24.0)
        values.append(
            _smooth_min(
                (static, dynamic), config.policies.component_temperature
            )
        )
        if index:
            previous = trajectory[index - 1, :2]
            sample_count = 3 if policy.kind == "room" else 2
            for sample in range(1, sample_count + 1):
                alpha = sample / (sample_count + 1)
                swept_point = previous + alpha * (point - previous)
                swept_time = (
                    time_offset
                    + (index - 1 + alpha) * policy.rollout_dt
                )
                swept = [
                    obstacle_clearance(
                        predicted_obstacle(obstacle, swept_time),
                        swept_point,
                        config.robot.radius + config.safety.safety_margin,
                        config.safety.human_margin,
                        config.safety.stretcher_margin,
                    )
                    for obstacle in obstacles
                ]
                values.append(_smooth_min(swept, 24.0))
    if policy.kind == "room" and policy.target_room is not None:
        terminal = trajectory[-1]
        values.append(
            policy.target_room.interior_margin(terminal[:2])
            - config.refuge.terminal_interior_margin
        )
        values.append(
            config.refuge.terminal_speed_max
            - np.linalg.norm(terminal[2:4])
        )
    return PolicyEvaluation(
        policy=policy,
        value=_smooth_min(values, config.policies.time_temperature),
        trajectory=trajectory,
    )


def build_policy_library(
    state: Sequence[float],
    nominal_waypoints: Sequence[Sequence[float]],
    room_paths: Sequence[tuple[Room, Sequence[Sequence[float]]]],
    environment: HospitalEnvironment,
    config: HospitalConfig,
) -> list[HospitalPolicy]:
    """Create the nominal, directional, reverse, stop, and room inventory."""

    value = np.asarray(state, dtype=float)
    nominal_points = [
        np.asarray(point, dtype=float)
        for point in nominal_waypoints[:MAX_POLICY_WAYPOINTS]
    ]
    if not nominal_points:
        nominal_points = [value[:2].copy()]
    route_target = nominal_points[min(1, len(nominal_points) - 1)]
    direction = route_target - value[:2]
    route_angle = (
        atan2(direction[1], direction[0])
        if np.linalg.norm(direction) > 1e-9
        else atan2(value[3], value[2])
    )
    policies = [
        HospitalPolicy(
            "nominal",
            "nominal",
            config.policies.nominal_horizon,
            config.policies.rollout_dt,
            config.policies.nominal_target_speed,
            waypoints=nominal_points,
            max_rollout_distance=config.robot.sensing_range,
        )
    ]
    count = config.policies.num_angle_policies
    angle_candidates: list[tuple[int, float, np.ndarray]] = []
    for index in range(count):
        offset = (
            0.0
            if count <= 1
            else -0.5 * config.policies.angle_arc
            + index * config.policies.angle_arc / (count - 1)
        )
        angle = route_angle + offset
        preview = value[:2] + config.policies.angle_preview_distance * np.array(
            [cos(angle), sin(angle)]
        )
        if any(room.contains(preview) for room in environment.rooms):
            continue
        angle_candidates.append((index, angle, preview))
    angle_free = (
        environment.segments_are_free(
            np.repeat(value[None, :2], len(angle_candidates), axis=0),
            np.asarray([item[2] for item in angle_candidates]),
            config.robot.radius + config.safety.static_margin,
            step=0.65,
        )
        if angle_candidates
        else np.zeros(0, dtype=bool)
    )
    for (index, angle, _preview), is_free in zip(
        angle_candidates,
        angle_free,
        strict=True,
    ):
        if not is_free:
            continue
        policies.append(
            HospitalPolicy(
                f"angle_{index}",
                "angle",
                config.policies.angle_horizon,
                config.policies.rollout_dt,
                config.policies.angle_target_speed,
                angle=angle,
                max_rollout_distance=config.robot.sensing_range,
            )
        )

    reverse_count = max(0, config.policies.num_reverse_policies)
    reverse_preview = config.policies.reverse_preview_distance
    reverse_candidates: list[tuple[str, float, np.ndarray]] = []

    def queue_reverse(name: str, angle: float) -> None:
        preview = value[:2] - reverse_preview * np.array(
            [cos(angle), sin(angle)]
        )
        reverse_candidates.append((name, angle, preview))

    for index in range(reverse_count):
        offset = (
            0.0
            if reverse_count <= 1
            else -0.5 * config.policies.reverse_policy_arc
            + index
            * config.policies.reverse_policy_arc
            / (reverse_count - 1)
        )
        queue_reverse(f"reverse_{index}", route_angle + offset)

    speed = float(np.linalg.norm(value[2:4]))
    body_reverse_angle = atan2(value[3], value[2]) if speed > 1e-6 else 0.0
    if (
        config.policies.body_reverse_policy
        and abs(_angle_normalize(body_reverse_angle - route_angle))
        >= config.policies.body_reverse_min_angle
    ):
        queue_reverse("reverse_body", body_reverse_angle)

    reverse_free = (
        environment.segments_are_free(
            np.repeat(value[None, :2], len(reverse_candidates), axis=0),
            np.asarray([item[2] for item in reverse_candidates]),
            config.robot.radius + config.safety.static_margin,
            step=0.65,
        )
        if reverse_candidates
        else np.zeros(0, dtype=bool)
    )
    for (name, angle, _preview), is_free in zip(
        reverse_candidates,
        reverse_free,
        strict=True,
    ):
        if not is_free:
            continue
        policies.append(
            HospitalPolicy(
                name,
                "reverse",
                config.policies.reverse_horizon,
                config.policies.rollout_dt,
                -config.policies.reverse_target_speed,
                angle=angle,
                max_rollout_distance=config.robot.sensing_range,
            )
        )

    policies.append(
        HospitalPolicy(
            "stop",
            "stop",
            config.policies.stop_horizon,
            config.policies.rollout_dt,
            max_rollout_distance=config.robot.sensing_range,
        )
    )
    for index, (room, raw_path) in enumerate(
        room_paths[: config.policies.room_policy_count]
    ):
        policies.append(
            HospitalPolicy(
                f"room_{index}",
                "room",
                config.policies.room_horizon,
                config.policies.room_rollout_dt,
                config.policies.room_target_speed,
                waypoints=_limit_room_path(raw_path),
                target_room=room,
                max_rollout_distance=config.robot.sensing_range,
            )
        )
    return policies


def policy_names(policies: Sequence[HospitalPolicy]) -> set[str]:
    return {policy.name for policy in policies}
