"""Fixed-shape, batched JAX rollouts for the Hospital policy library.

This module contains no policy-selection or scenario logic.  It is a numerical
backend for evaluating the same closed-loop policy inventory used by
``examples.hospital.policies``.  Host objects are packed into fixed-capacity
arrays so that policy availability, obstacle count, and waypoint count can
change without changing an XLA input shape.

The public API deliberately separates packing from evaluation:

``pack_static_geometry``
    Packs the immutable floor union and wall rectangles.
``pack_policy_batch``
    Packs nominal, angle, reverse, stop, room, and retrace policies.
``pack_obstacle_batch``
    Packs circular Humans and oriented rectangular Stretchers.
``evaluate_policy_batch``
    Returns rollout values, exact state gradients, shifted-time values,
    nominal-prefix values, terminal clearances, trajectories, and masks.
``evaluate_policy_groups``
    Evaluates horizon-bucketed policy batches while sharing one obstacle
    prediction checkpoint table across the short, room, and long rollouts.

The compiled evaluator is cached process-wide.  Controllers recreated for
different benchmark worlds therefore reuse the same executable whenever their
capacity and static-geometry shapes match.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, cos, pi, sin
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .config import HospitalConfig
from .environment import HospitalEnvironment
from .obstacles import DynamicObstacle, Human, Stretcher
from .policies import (
    MAX_POLICY_WAYPOINTS,
    NOMINAL_WAYPOINT_RADIUS,
    RETRACE_WAYPOINT_RADIUS,
    ROOM_APPROACH_RADIUS,
    ROOM_WAYPOINT_RADIUS,
    HospitalPolicy,
)


POLICY_INACTIVE = 0
POLICY_NOMINAL = 1
POLICY_ANGLE = 2
POLICY_REVERSE = 3
POLICY_STOP = 4
POLICY_ROOM = 5
POLICY_RETRACE = 6

OBSTACLE_INACTIVE = 0
OBSTACLE_CIRCLE = 1
OBSTACLE_RECTANGLE = 2

_POLICY_KIND = {
    "nominal": POLICY_NOMINAL,
    "angle": POLICY_ANGLE,
    "reverse": POLICY_REVERSE,
    "stop": POLICY_STOP,
    "room": POLICY_ROOM,
    "retrace": POLICY_RETRACE,
}

_HUMAN_PREDICTION_DT = 0.05
_VALUE_EMPTY = 100.0
_MASKED_VALUE = 1.0e6
DEFAULT_OBSTACLE_BUCKETS = (4, 8, 16, 32, 53)


@dataclass(frozen=True)
class HospitalJaxCapacities:
    """Static array capacities used as an XLA compilation key."""

    max_policies: int = 24
    max_waypoints: int = MAX_POLICY_WAYPOINTS
    max_obstacles: int = 53
    max_horizon_steps: int = 50
    max_swept_samples: int = 3
    human_prediction_steps: int = 243

    def __post_init__(self) -> None:
        for name, value in (
            ("max_policies", self.max_policies),
            ("max_waypoints", self.max_waypoints),
            ("max_obstacles", self.max_obstacles),
            ("max_horizon_steps", self.max_horizon_steps),
            ("max_swept_samples", self.max_swept_samples),
            ("human_prediction_steps", self.human_prediction_steps),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def for_config(
        cls,
        config: HospitalConfig,
        *,
        max_policies: int = 24,
        max_obstacles: int | None = None,
        include_retrace: bool = False,
    ) -> "HospitalJaxCapacities":
        """Build one stable capacity for a controller configuration.

        PL-CBF itself needs at most the longest configured policy horizon.
        ``include_retrace`` additionally covers a plant-rate, 12-second
        Warehouse-style retrace certificate.
        """

        policy = config.policies
        step_counts = (
            max(1, int(policy.nominal_horizon / policy.rollout_dt)),
            max(1, int(policy.angle_horizon / policy.rollout_dt)),
            max(1, int(policy.reverse_horizon / policy.rollout_dt)),
            max(1, int(policy.stop_horizon / policy.rollout_dt)),
            max(1, int(policy.room_horizon / policy.room_rollout_dt)),
        )
        horizon_steps = max(step_counts)
        maximum_time = max(
            policy.nominal_horizon,
            policy.angle_horizon,
            policy.reverse_horizon,
            policy.stop_horizon,
            policy.room_horizon,
        )
        if include_retrace:
            horizon_steps = max(horizon_steps, int(ceil(12.0 / config.dt)))
            maximum_time = max(maximum_time, 12.0)
        prediction_time = maximum_time + max(
            config.dt, policy.time_derivative_step
        )
        return cls(
            max_policies=max_policies,
            max_obstacles=(
                config.safety.max_obstacles
                if max_obstacles is None
                else int(max_obstacles)
            ),
            max_horizon_steps=horizon_steps,
            human_prediction_steps=int(
                ceil(prediction_time / _HUMAN_PREDICTION_DT)
            ),
        )


@dataclass(frozen=True)
class HospitalJaxPolicyGroupSpec:
    """Static policy count and rollout length for one compiled group."""

    max_policies: int
    max_horizon_steps: int
    max_swept_samples: int = 3

    def __post_init__(self) -> None:
        if int(self.max_policies) < 1:
            raise ValueError("max_policies must be positive")
        if int(self.max_horizon_steps) < 1:
            raise ValueError("max_horizon_steps must be positive")
        if int(self.max_swept_samples) < 1:
            raise ValueError("max_swept_samples must be positive")


@dataclass(frozen=True)
class HospitalJaxGroupedCapacities:
    """Fixed shapes for horizon-bucketed policy evaluation.

    Groups must be ordered by strictly increasing horizon.  Packing places a
    policy in the first group that has enough horizon and remaining policy
    capacity.  The default Hospital inventory therefore evaluates nominal and
    angle policies for 12 steps, room policies for 30, and stop/reverse
    policies for 50 instead of scanning all policies for 50 steps.
    """

    groups: tuple[HospitalJaxPolicyGroupSpec, ...]
    max_waypoints: int = MAX_POLICY_WAYPOINTS
    max_obstacles: int = 53
    human_prediction_steps: int = 243

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("at least one policy group is required")
        horizons = tuple(group.max_horizon_steps for group in self.groups)
        if any(right <= left for left, right in zip(horizons, horizons[1:])):
            raise ValueError("policy-group horizons must be strictly increasing")
        for name, value in (
            ("max_waypoints", self.max_waypoints),
            ("max_obstacles", self.max_obstacles),
            ("human_prediction_steps", self.human_prediction_steps),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def for_config(
        cls,
        config: HospitalConfig,
        *,
        max_obstacles: int | None = None,
        include_retrace: bool = False,
    ) -> "HospitalJaxGroupedCapacities":
        """Build exact horizon groups for the configured policy inventory."""

        policy = config.policies
        counts_by_steps: dict[int, int] = {}
        swept_by_steps: dict[int, int] = {}

        def add(
            count: int,
            horizon: float,
            rollout_dt: float,
            swept_samples: int,
        ) -> None:
            if count <= 0:
                return
            steps = max(1, int(horizon / rollout_dt))
            counts_by_steps[steps] = counts_by_steps.get(steps, 0) + int(count)
            swept_by_steps[steps] = max(
                swept_by_steps.get(steps, 0), swept_samples
            )

        add(1, policy.nominal_horizon, policy.rollout_dt, 2)
        add(
            policy.num_angle_policies,
            policy.angle_horizon,
            policy.rollout_dt,
            2,
        )
        reverse_count = policy.num_reverse_policies + int(policy.body_reverse_policy)
        add(reverse_count, policy.reverse_horizon, policy.rollout_dt, 2)
        add(1, policy.stop_horizon, policy.rollout_dt, 2)
        add(
            policy.room_policy_count,
            policy.room_horizon,
            policy.room_rollout_dt,
            3,
        )
        maximum_time = max(
            policy.nominal_horizon,
            policy.angle_horizon,
            policy.reverse_horizon,
            policy.stop_horizon,
            policy.room_horizon,
        )
        if include_retrace:
            retrace_steps = max(1, int(ceil(12.0 / config.dt)))
            counts_by_steps[retrace_steps] = counts_by_steps.get(retrace_steps, 0) + 1
            swept_by_steps[retrace_steps] = max(
                swept_by_steps.get(retrace_steps, 0), 2
            )
            maximum_time = max(maximum_time, 12.0)
        prediction_time = maximum_time + max(
            config.dt, policy.time_derivative_step
        )
        return cls(
            groups=tuple(
                HospitalJaxPolicyGroupSpec(count, steps, swept_by_steps[steps])
                for steps, count in sorted(counts_by_steps.items())
            ),
            max_obstacles=(
                config.safety.max_obstacles
                if max_obstacles is None
                else int(max_obstacles)
            ),
            human_prediction_steps=int(
                ceil(prediction_time / _HUMAN_PREDICTION_DT)
            ),
        )

    def capacity_for_group(
        self, group: HospitalJaxPolicyGroupSpec
    ) -> HospitalJaxCapacities:
        """Return the ordinary capacity represented by one horizon group."""

        return HospitalJaxCapacities(
            max_policies=group.max_policies,
            max_waypoints=self.max_waypoints,
            max_obstacles=self.max_obstacles,
            max_horizon_steps=group.max_horizon_steps,
            max_swept_samples=group.max_swept_samples,
            human_prediction_steps=self.human_prediction_steps,
        )


class HospitalJaxPolicyBatch(NamedTuple):
    active: np.ndarray
    kinds: np.ndarray
    horizon_steps: np.ndarray
    rollout_dt: np.ndarray
    target_speeds: np.ndarray
    angles: np.ndarray
    waypoints: np.ndarray
    waypoint_mask: np.ndarray
    room_bounds: np.ndarray
    has_room: np.ndarray
    max_rollout_distance: np.ndarray
    feedback_gains: np.ndarray
    swept_samples: np.ndarray


class HospitalJaxObstacleBatch(NamedTuple):
    active: np.ndarray
    kinds: np.ndarray
    centers: np.ndarray
    velocities: np.ndarray
    radii: np.ndarray
    half_lengths: np.ndarray
    half_widths: np.ndarray
    cosines: np.ndarray
    sines: np.ndarray
    bounce: np.ndarray
    reflect: np.ndarray
    axes: np.ndarray
    route_min: np.ndarray
    route_max: np.ndarray
    route_speed: np.ndarray


class HospitalJaxStaticGeometry(NamedTuple):
    floor_bounds: np.ndarray
    floor_centers: np.ndarray
    floor_half: np.ndarray
    floor_boundary_starts: np.ndarray
    floor_boundary_ends: np.ndarray
    wall_bounds: np.ndarray
    wall_centers: np.ndarray
    wall_half: np.ndarray
    width: np.ndarray
    height: np.ndarray


class HospitalJaxParameters(NamedTuple):
    robot_radius: np.ndarray
    v_max: np.ndarray
    a_max: np.ndarray
    k_position: np.ndarray
    k_velocity: np.ndarray
    stop_gain: np.ndarray
    safety_margin: np.ndarray
    human_margin: np.ndarray
    stretcher_margin: np.ndarray
    static_margin: np.ndarray
    component_temperature: np.ndarray
    time_temperature: np.ndarray
    time_derivative_step: np.ndarray
    plant_dt: np.ndarray
    terminal_interior_margin: np.ndarray
    terminal_speed_max: np.ndarray


@dataclass(frozen=True)
class PackedHospitalPolicies:
    names: tuple[str, ...]
    batch: HospitalJaxPolicyBatch
    active_count: int
    capacities: HospitalJaxCapacities


@dataclass(frozen=True)
class PackedHospitalPolicyGroups:
    """Policy batches plus indices restoring the caller's original order."""

    names: tuple[str, ...]
    groups: tuple[PackedHospitalPolicies, ...]
    original_indices: tuple[tuple[int, ...], ...]
    active_count: int
    capacities: HospitalJaxGroupedCapacities


@dataclass(frozen=True)
class HospitalJaxEvaluation:
    """Host arrays returned by :func:`evaluate_policy_batch`."""

    names: tuple[str, ...]
    values: np.ndarray
    gradients: np.ndarray
    shifted_values: np.ndarray
    time_derivatives: np.ndarray
    nominal_prefix_values: np.ndarray
    terminal_clearances: np.ndarray
    trajectories: np.ndarray
    trajectory_mask: np.ndarray
    diagnostics_available: bool


@dataclass(frozen=True)
class _EvaluatorStructure:
    capacities: HospitalJaxCapacities
    floor_count: int
    floor_boundary_count: int
    wall_count: int
    include_diagnostics: bool


@dataclass(frozen=True)
class _GroupedEvaluatorStructure:
    capacities: HospitalJaxGroupedCapacities
    floor_count: int
    floor_boundary_count: int
    wall_count: int
    include_diagnostics: bool
    shared_time_grids: tuple[bool, ...]


def select_obstacle_bucket(
    active_count: int,
    buckets: Sequence[int] = DEFAULT_OBSTACLE_BUCKETS,
) -> int:
    """Return the smallest fixed obstacle bucket that preserves every input."""

    count = int(active_count)
    if count < 0:
        raise ValueError("active_count must be nonnegative")
    normalized = tuple(sorted({int(item) for item in buckets}))
    if not normalized or normalized[0] < 1:
        raise ValueError("obstacle buckets must contain positive integers")
    for bucket in normalized:
        if count <= bucket:
            return bucket
    raise ValueError(
        f"{count} active obstacles exceed the largest bucket {normalized[-1]}"
    )


def capacities_for_obstacle_count(
    config: HospitalConfig,
    active_count: int,
    *,
    max_policies: int = 24,
    buckets: Sequence[int] = DEFAULT_OBSTACLE_BUCKETS,
    include_retrace: bool = False,
) -> HospitalJaxCapacities:
    """Build a controller-stable capacity using a prewarmable size bucket."""

    return HospitalJaxCapacities.for_config(
        config,
        max_policies=max_policies,
        max_obstacles=select_obstacle_bucket(active_count, buckets),
        include_retrace=include_retrace,
    )


def grouped_capacities_for_obstacle_count(
    config: HospitalConfig,
    active_count: int,
    *,
    buckets: Sequence[int] = DEFAULT_OBSTACLE_BUCKETS,
    include_retrace: bool = False,
) -> HospitalJaxGroupedCapacities:
    """Build horizon groups using the smallest prewarmable obstacle bucket."""

    return HospitalJaxGroupedCapacities.for_config(
        config,
        max_obstacles=select_obstacle_bucket(active_count, buckets),
        include_retrace=include_retrace,
    )


def pack_static_geometry(
    environment: HospitalEnvironment,
) -> HospitalJaxStaticGeometry:
    """Pack the immutable floor and wall rectangles without approximation."""

    return HospitalJaxStaticGeometry(
        floor_bounds=np.asarray(environment._floor_bounds, dtype=np.float32),
        floor_centers=np.asarray(environment._floor_centers, dtype=np.float32),
        floor_half=np.asarray(environment._floor_half, dtype=np.float32),
        floor_boundary_starts=np.asarray(
            environment._floor_boundary_starts, dtype=np.float32
        ),
        floor_boundary_ends=np.asarray(
            environment._floor_boundary_ends, dtype=np.float32
        ),
        wall_bounds=np.asarray(environment._wall_bounds, dtype=np.float32),
        wall_centers=np.asarray(environment._wall_centers, dtype=np.float32),
        wall_half=np.asarray(environment._wall_half, dtype=np.float32),
        width=np.asarray(environment.width, dtype=np.float32),
        height=np.asarray(environment.height, dtype=np.float32),
    )


def pack_parameters(config: HospitalConfig) -> HospitalJaxParameters:
    """Pack numerical controller parameters as dynamic JIT arguments."""

    robot = config.robot
    safety = config.safety
    policy = config.policies
    refuge = config.refuge
    scalar = lambda value: np.asarray(value, dtype=np.float32)
    return HospitalJaxParameters(
        robot_radius=scalar(robot.radius),
        v_max=scalar(robot.v_max),
        a_max=scalar(robot.a_max),
        k_position=scalar(robot.k_position),
        k_velocity=scalar(robot.k_velocity),
        stop_gain=scalar(policy.stop_gain),
        safety_margin=scalar(safety.safety_margin),
        human_margin=scalar(safety.human_margin),
        stretcher_margin=scalar(safety.stretcher_margin),
        static_margin=scalar(safety.static_margin),
        component_temperature=scalar(policy.component_temperature),
        time_temperature=scalar(policy.time_temperature),
        time_derivative_step=scalar(policy.time_derivative_step),
        plant_dt=scalar(config.dt),
        terminal_interior_margin=scalar(refuge.terminal_interior_margin),
        terminal_speed_max=scalar(refuge.terminal_speed_max),
    )


def pack_policy_batch(
    policies: Sequence[HospitalPolicy],
    config: HospitalConfig,
    capacities: HospitalJaxCapacities,
) -> PackedHospitalPolicies:
    """Pack a variable policy library into fixed-capacity arrays."""

    policy_tuple = tuple(policies)
    if len(policy_tuple) > capacities.max_policies:
        raise ValueError(
            f"{len(policy_tuple)} policies exceed capacity "
            f"{capacities.max_policies}"
        )
    count = capacities.max_policies
    waypoint_count = capacities.max_waypoints
    active = np.zeros(count, dtype=bool)
    kinds = np.zeros(count, dtype=np.int32)
    horizon_steps = np.ones(count, dtype=np.int32)
    rollout_dt = np.full(count, config.policies.rollout_dt, dtype=np.float32)
    target_speeds = np.zeros(count, dtype=np.float32)
    angles = np.zeros(count, dtype=np.float32)
    waypoints = np.zeros((count, waypoint_count, 2), dtype=np.float32)
    waypoint_mask = np.zeros((count, waypoint_count), dtype=bool)
    room_bounds = np.zeros((count, 4), dtype=np.float32)
    has_room = np.zeros(count, dtype=bool)
    max_rollout_distance = np.full(count, 1.0e6, dtype=np.float32)
    feedback_gains = np.full(count, config.robot.k_velocity, dtype=np.float32)
    swept_samples = np.full(count, 2, dtype=np.int32)
    names: list[str] = []

    for index, policy in enumerate(policy_tuple):
        try:
            kind = _POLICY_KIND[policy.kind]
        except KeyError as error:
            raise ValueError(
                f"unsupported Hospital policy kind {policy.kind!r}"
            ) from error
        steps = max(1, int(policy.horizon / policy.rollout_dt))
        if steps > capacities.max_horizon_steps:
            raise ValueError(
                f"policy {policy.name!r} needs {steps} rollout steps, exceeding "
                f"capacity {capacities.max_horizon_steps}"
            )
        required_prediction_time = (
            steps * policy.rollout_dt
            + max(config.dt, config.policies.time_derivative_step)
        )
        available_prediction_time = (
            capacities.human_prediction_steps * _HUMAN_PREDICTION_DT
        )
        if required_prediction_time > available_prediction_time + 1.0e-9:
            raise ValueError(
                f"policy {policy.name!r} needs obstacle predictions through "
                f"{required_prediction_time:.3f} s, exceeding capacity "
                f"{available_prediction_time:.3f} s"
            )
        points = tuple(np.asarray(point, dtype=float) for point in policy.waypoints)
        if len(points) > waypoint_count:
            raise ValueError(
                f"policy {policy.name!r} has {len(points)} waypoints, exceeding "
                f"capacity {waypoint_count}"
            )
        if kind in {POLICY_NOMINAL, POLICY_ROOM, POLICY_RETRACE} and not points:
            raise ValueError(f"policy {policy.name!r} requires at least one waypoint")

        active[index] = True
        kinds[index] = kind
        horizon_steps[index] = steps
        rollout_dt[index] = policy.rollout_dt
        target_speeds[index] = policy.target_speed
        angles[index] = 0.0 if policy.angle is None else policy.angle
        if points:
            waypoints[index, : len(points)] = np.asarray(points, dtype=np.float32)
            waypoint_mask[index, : len(points)] = True
        if policy.target_room is not None:
            rectangle = policy.target_room.rect
            room_bounds[index] = (
                rectangle.x,
                rectangle.y,
                rectangle.x1,
                rectangle.y1,
            )
            has_room[index] = True
        if policy.max_rollout_distance is not None:
            max_rollout_distance[index] = policy.max_rollout_distance
        if policy.feedback_gain is not None:
            feedback_gains[index] = policy.feedback_gain
        swept_samples[index] = 3 if kind == POLICY_ROOM else 2
        names.append(policy.name)

    return PackedHospitalPolicies(
        names=tuple(names),
        batch=HospitalJaxPolicyBatch(
            active=active,
            kinds=kinds,
            horizon_steps=horizon_steps,
            rollout_dt=rollout_dt,
            target_speeds=target_speeds,
            angles=angles,
            waypoints=waypoints,
            waypoint_mask=waypoint_mask,
            room_bounds=room_bounds,
            has_room=has_room,
            max_rollout_distance=max_rollout_distance,
            feedback_gains=feedback_gains,
            swept_samples=swept_samples,
        ),
        active_count=len(policy_tuple),
        capacities=capacities,
    )


def pack_policy_groups(
    policies: Sequence[HospitalPolicy],
    config: HospitalConfig,
    capacities: HospitalJaxGroupedCapacities,
) -> PackedHospitalPolicyGroups:
    """Pack policies into the shortest horizon group that can contain them."""

    policy_tuple = tuple(policies)
    assignments: list[list[HospitalPolicy]] = [
        [] for _unused in capacities.groups
    ]
    original_indices: list[list[int]] = [
        [] for _unused in capacities.groups
    ]
    for original_index, policy in enumerate(policy_tuple):
        required_steps = max(1, int(policy.horizon / policy.rollout_dt))
        required_swept = 3 if policy.kind == "room" else 2
        selected_group = None
        for group_index, group in enumerate(capacities.groups):
            if (
                required_steps <= group.max_horizon_steps
                and required_swept <= group.max_swept_samples
                and len(assignments[group_index]) < group.max_policies
            ):
                selected_group = group_index
                break
        if selected_group is None:
            raise ValueError(
                f"policy {policy.name!r} needs {required_steps} rollout steps, "
                "but no policy group has sufficient remaining capacity"
            )
        assignments[selected_group].append(policy)
        original_indices[selected_group].append(original_index)

    packed_groups = tuple(
        pack_policy_batch(
            group_policies,
            config,
            capacities.capacity_for_group(group),
        )
        for group_policies, group in zip(
            assignments, capacities.groups, strict=True
        )
    )
    return PackedHospitalPolicyGroups(
        names=tuple(policy.name for policy in policy_tuple),
        groups=packed_groups,
        original_indices=tuple(tuple(indices) for indices in original_indices),
        active_count=len(policy_tuple),
        capacities=capacities,
    )


def pack_obstacle_batch(
    obstacles: Sequence[DynamicObstacle],
    capacities: HospitalJaxCapacities | HospitalJaxGroupedCapacities,
) -> HospitalJaxObstacleBatch:
    """Pack sensed Humans and Stretchers into fixed-capacity arrays."""

    obstacle_tuple = tuple(obstacles)
    if len(obstacle_tuple) > capacities.max_obstacles:
        raise ValueError(
            f"{len(obstacle_tuple)} obstacles exceed capacity "
            f"{capacities.max_obstacles}"
        )
    count = capacities.max_obstacles
    active = np.zeros(count, dtype=bool)
    kinds = np.zeros(count, dtype=np.int32)
    centers = np.zeros((count, 2), dtype=np.float32)
    velocities = np.zeros((count, 2), dtype=np.float32)
    radii = np.zeros(count, dtype=np.float32)
    half_lengths = np.zeros(count, dtype=np.float32)
    half_widths = np.zeros(count, dtype=np.float32)
    cosines = np.ones(count, dtype=np.float32)
    sines = np.zeros(count, dtype=np.float32)
    bounce = np.zeros(count, dtype=bool)
    reflect = np.zeros(count, dtype=bool)
    axes = np.zeros(count, dtype=np.int32)
    route_min = np.zeros(count, dtype=np.float32)
    route_max = np.ones(count, dtype=np.float32)
    route_speed = np.zeros(count, dtype=np.float32)

    for index, obstacle in enumerate(obstacle_tuple):
        active[index] = True
        centers[index] = obstacle.center
        velocities[index] = obstacle.velocity
        if isinstance(obstacle, Human):
            kinds[index] = OBSTACLE_CIRCLE
            radii[index] = obstacle.radius
            bounce[index] = True
        elif isinstance(obstacle, Stretcher):
            kinds[index] = OBSTACLE_RECTANGLE
            half_lengths[index] = 0.5 * obstacle.length
            half_widths[index] = 0.5 * obstacle.width
            cosines[index] = cos(obstacle.theta)
            sines[index] = sin(obstacle.theta)
            reflect[index] = obstacle.reflect_at_route_bounds
            axes[index] = 0 if obstacle.axis == "x" else 1
            route_min[index] = obstacle.route_min
            route_max[index] = obstacle.route_max
            route_speed[index] = obstacle.speed
        else:
            raise TypeError(
                "the Hospital JAX backend supports Human circles and "
                f"Stretcher rectangles, not {type(obstacle).__name__}"
            )

    return HospitalJaxObstacleBatch(
        active=active,
        kinds=kinds,
        centers=centers,
        velocities=velocities,
        radii=radii,
        half_lengths=half_lengths,
        half_widths=half_widths,
        cosines=cosines,
        sines=sines,
        bounce=bounce,
        reflect=reflect,
        axes=axes,
        route_min=route_min,
        route_max=route_max,
        route_speed=route_speed,
    )


def _safe_norm(value: jax.Array, axis: int = -1) -> jax.Array:
    squared = jnp.sum(jnp.square(value), axis=axis)
    return jnp.where(
        squared > 1.0e-18,
        jnp.sqrt(jnp.maximum(squared, 1.0e-18)),
        jnp.zeros_like(squared),
    )


def _smooth_min_masked(
    values: jax.Array,
    mask: jax.Array,
    temperature: jax.Array,
    *,
    axis: int | tuple[int, ...] = -1,
) -> jax.Array:
    finite_mask = mask & jnp.isfinite(values)
    safe_values = jnp.where(finite_mask, values, 0.0)
    masked = jnp.where(finite_mask, safe_values, _MASKED_VALUE)
    minimum = jnp.min(masked, axis=axis, keepdims=True)
    # Substitute before exponentiation.  ``jnp.where(mask, exp(...), 0)``
    # still evaluates the inactive exponential and can inject ``inf * 0``
    # into reverse-mode derivatives for padded entries.
    delta = jnp.where(finite_mask, safe_values - minimum, 0.0)
    weights = (
        jnp.exp(-temperature * delta) * finite_mask.astype(values.dtype)
    )
    total = jnp.sum(weights, axis=axis)
    squeezed = jnp.squeeze(minimum, axis=axis)
    result = squeezed - jnp.log(jnp.maximum(total, 1.0e-12)) / temperature
    count = jnp.sum(finite_mask.astype(jnp.int32), axis=axis)
    return jnp.where(count > 0, result, _VALUE_EMPTY)


def _rect_signed_distance(
    points: jax.Array,
    centers: jax.Array,
    half: jax.Array,
) -> jax.Array:
    q = jnp.abs(points - centers) - half
    return _safe_norm(jnp.maximum(q, 0.0)) + jnp.minimum(
        jnp.maximum(q[..., 0], q[..., 1]), 0.0
    )


def _static_clearance(
    point: jax.Array,
    radius: jax.Array,
    geometry: HospitalJaxStaticGeometry,
) -> jax.Array:
    segments = geometry.floor_boundary_ends - geometry.floor_boundary_starts
    lengths_squared = jnp.sum(jnp.square(segments), axis=1)
    relative = point[None, :] - geometry.floor_boundary_starts
    fractions = jnp.clip(
        jnp.sum(relative * segments, axis=1)
        / jnp.maximum(lengths_squared, 1.0e-18),
        0.0,
        1.0,
    )
    closest = geometry.floor_boundary_starts + fractions[:, None] * segments
    boundary_distance = jnp.min(_safe_norm(point[None, :] - closest, axis=1))
    floor_lower = geometry.floor_bounds[:, :2]
    floor_upper = floor_lower + geometry.floor_bounds[:, 2:]
    on_floor = jnp.any(
        jnp.all(
            (point[None, :] >= floor_lower)
            & (point[None, :] <= floor_upper),
            axis=1,
        )
    )
    floor_margin = jnp.where(on_floor, boundary_distance, -boundary_distance) - radius
    wall_signed = _rect_signed_distance(
        point[None, :], geometry.wall_centers, geometry.wall_half
    )
    wall_margin = jnp.min(wall_signed) - radius
    return jnp.minimum(floor_margin, wall_margin)


def _environment_collision(
    points: jax.Array,
    radii: jax.Array,
    geometry: HospitalJaxStaticGeometry,
) -> jax.Array:
    """JAX equivalent of ``HospitalEnvironment.collisions``."""

    original_shape = points.shape[:-1]
    flattened = points.reshape((-1, 2))
    flat_radii = jnp.broadcast_to(radii, original_shape).reshape((-1,))
    clearances = jax.vmap(
        lambda point, radius: _static_clearance(point, radius, geometry)
    )(flattened, flat_radii)
    return (clearances <= 0.0).reshape(original_shape)


def _advance_bouncing_circles(
    positions: jax.Array,
    velocities: jax.Array,
    elapsed: jax.Array,
    obstacles: HospitalJaxObstacleBatch,
    geometry: HospitalJaxStaticGeometry,
) -> tuple[jax.Array, jax.Array]:
    """Apply ``Human.advance`` to every active bouncing circle."""

    dt = jnp.asarray(elapsed)
    while dt.ndim < positions.ndim - 1:
        dt = dt[..., None]
    next_positions = positions + velocities * dt[..., None]
    radii = jnp.broadcast_to(obstacles.radii, positions.shape[:-1])
    bounce = jnp.broadcast_to(
        obstacles.active
        & obstacles.bounce
        & (obstacles.kinds == OBSTACLE_CIRCLE),
        positions.shape[:-1],
    )
    full_collision = _environment_collision(next_positions, radii, geometry)
    collision = bounce & full_collision

    next_x = jnp.stack(
        (positions[..., 0] + velocities[..., 0] * dt, positions[..., 1]),
        axis=-1,
    )
    next_y = jnp.stack(
        (positions[..., 0], positions[..., 1] + velocities[..., 1] * dt),
        axis=-1,
    )
    free_x = ~_environment_collision(next_x, radii, geometry)
    free_y = ~_environment_collision(next_y, radii, geometry)
    neither = ~(free_x | free_y)
    collided_position = jnp.stack(
        (
            jnp.where(free_x, next_x[..., 0], positions[..., 0]),
            jnp.where(free_y, next_y[..., 1], positions[..., 1]),
        ),
        axis=-1,
    )
    collided_velocity = jnp.stack(
        (
            jnp.where(free_y | neither, -velocities[..., 0], velocities[..., 0]),
            jnp.where(free_x | neither, -velocities[..., 1], velocities[..., 1]),
        ),
        axis=-1,
    )
    result_position = jnp.where(
        collision[..., None], collided_position, next_positions
    )
    result_velocity = jnp.where(
        collision[..., None], collided_velocity, velocities
    )
    # Non-bouncing and inactive entries are predicted analytically elsewhere;
    # keep their checkpoint state unchanged here.
    result_position = jnp.where(
        bounce[..., None], result_position, positions
    )
    result_velocity = jnp.where(
        bounce[..., None], result_velocity, velocities
    )
    return result_position, result_velocity


def _circle_checkpoints(
    obstacles: HospitalJaxObstacleBatch,
    geometry: HospitalJaxStaticGeometry,
    prediction_steps: int,
) -> tuple[jax.Array, jax.Array]:
    initial = (obstacles.centers, obstacles.velocities)

    def advance(
        carry: tuple[jax.Array, jax.Array], _unused: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        following = _advance_bouncing_circles(
            carry[0], carry[1], _HUMAN_PREDICTION_DT, obstacles, geometry
        )
        return following, following

    _, (position_tail, velocity_tail) = jax.lax.scan(
        advance, initial, xs=jnp.arange(prediction_steps)
    )
    return (
        jnp.concatenate((obstacles.centers[None, :, :], position_tail), axis=0),
        jnp.concatenate((obstacles.velocities[None, :, :], velocity_tail), axis=0),
    )


def _reflected_coordinate(
    coordinate: jax.Array,
    speed: jax.Array,
    elapsed: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    span = jnp.maximum(upper - lower, 1.0e-9)
    phase = jnp.mod(coordinate - lower + speed * elapsed, 2.0 * span)
    return jnp.where(
        phase <= span,
        lower + phase,
        lower + (2.0 * span - phase),
    )


def _obstacle_centers_at(
    times: jax.Array,
    obstacles: HospitalJaxObstacleBatch,
    geometry: HospitalJaxStaticGeometry,
    checkpoint_positions: jax.Array,
    checkpoint_velocities: jax.Array,
) -> jax.Array:
    """Return obstacle centers for arbitrary query times."""

    original_shape = times.shape
    flattened = jnp.maximum(times.reshape((-1,)), 0.0)
    max_index = checkpoint_positions.shape[0] - 1
    indices = jnp.clip(
        jnp.floor(flattened / _HUMAN_PREDICTION_DT).astype(jnp.int32),
        0,
        max_index,
    )
    checkpoint_time = indices.astype(flattened.dtype) * _HUMAN_PREDICTION_DT
    remainder = flattened - checkpoint_time
    remainder = jnp.where(remainder <= 1.0e-10, 0.0, remainder)
    circle_positions = checkpoint_positions[indices]
    circle_velocities = checkpoint_velocities[indices]
    circle_positions, _ = _advance_bouncing_circles(
        circle_positions,
        circle_velocities,
        remainder,
        obstacles,
        geometry,
    )
    linear = (
        obstacles.centers[None, :, :]
        + flattened[:, None, None] * obstacles.velocities[None, :, :]
    )
    bouncing = obstacles.bounce & (obstacles.kinds == OBSTACLE_CIRCLE)
    circle_centers = jnp.where(bouncing[None, :, None], circle_positions, linear)

    axes = obstacles.axes
    initial_coordinate = jnp.where(
        axes == 0, obstacles.centers[:, 0], obstacles.centers[:, 1]
    )
    reflected = _reflected_coordinate(
        initial_coordinate[None, :],
        obstacles.route_speed[None, :],
        flattened[:, None],
        obstacles.route_min[None, :],
        obstacles.route_max[None, :],
    )
    rectangle_centers = linear
    reflected_x = jnp.stack((reflected, linear[..., 1]), axis=-1)
    reflected_y = jnp.stack((linear[..., 0], reflected), axis=-1)
    reflected_centers = jnp.where(
        (axes == 0)[None, :, None], reflected_x, reflected_y
    )
    rectangle_centers = jnp.where(
        obstacles.reflect[None, :, None], reflected_centers, rectangle_centers
    )
    centers = jnp.where(
        (obstacles.kinds == OBSTACLE_CIRCLE)[None, :, None],
        circle_centers,
        rectangle_centers,
    )
    return centers.reshape(original_shape + obstacles.centers.shape)


def _policy_obstacle_tables(
    policy: HospitalJaxPolicyBatch,
    time_offset: jax.Array,
    obstacles: HospitalJaxObstacleBatch,
    geometry: HospitalJaxStaticGeometry,
    checkpoint_positions: jax.Array,
    checkpoint_velocities: jax.Array,
    horizon_steps: int,
    swept_capacity: int,
) -> tuple[jax.Array, jax.Array]:
    return _obstacle_tables_for_time_grid(
        policy.rollout_dt,
        policy.swept_samples,
        time_offset,
        obstacles,
        geometry,
        checkpoint_positions,
        checkpoint_velocities,
        horizon_steps,
        swept_capacity,
    )


def _obstacle_tables_for_time_grid(
    rollout_dt: jax.Array,
    swept_samples: jax.Array,
    time_offset: jax.Array,
    obstacles: HospitalJaxObstacleBatch,
    geometry: HospitalJaxStaticGeometry,
    checkpoint_positions: jax.Array,
    checkpoint_velocities: jax.Array,
    horizon_steps: int,
    swept_capacity: int,
) -> tuple[jax.Array, jax.Array]:
    """Predict obstacles once for one shared policy integration grid."""

    vertex_indices = jnp.arange(horizon_steps + 1, dtype=jnp.float32)
    vertex_times = time_offset + vertex_indices * rollout_dt
    segment_indices = jnp.arange(horizon_steps, dtype=jnp.float32)[:, None]
    sample_indices = jnp.arange(1, swept_capacity + 1, dtype=jnp.float32)[None, :]
    denominators = swept_samples.astype(jnp.float32) + 1.0
    alphas = sample_indices / denominators
    swept_times = time_offset + (
        segment_indices + alphas
    ) * rollout_dt
    return (
        _obstacle_centers_at(
            vertex_times,
            obstacles,
            geometry,
            checkpoint_positions,
            checkpoint_velocities,
        ),
        _obstacle_centers_at(
            swept_times,
            obstacles,
            geometry,
            checkpoint_positions,
            checkpoint_velocities,
        ),
    )


def _clip_control(control: jax.Array, parameters: HospitalJaxParameters) -> jax.Array:
    return jnp.clip(control, -parameters.a_max, parameters.a_max)


def _waypoint_control(
    state: jax.Array,
    target: jax.Array,
    target_speed: jax.Array,
    parameters: HospitalJaxParameters,
) -> jax.Array:
    delta = target - state[:2]
    distance = _safe_norm(delta)
    speed = jnp.minimum(target_speed, parameters.k_position * distance)
    desired = jnp.where(
        distance > 1.0e-9,
        delta * speed / jnp.maximum(distance, 1.0e-9),
        jnp.zeros(2, dtype=state.dtype),
    )
    return _clip_control(
        parameters.k_velocity * (desired - state[2:4]), parameters
    )


def _policy_control(
    state: jax.Array,
    cursor: jax.Array,
    policy: HospitalJaxPolicyBatch,
    parameters: HospitalJaxParameters,
) -> tuple[jax.Array, jax.Array]:
    waypoint_count = jnp.sum(policy.waypoint_mask.astype(jnp.int32))
    last_waypoint = jnp.maximum(waypoint_count - 1, 0)
    waypoint_cursor = jnp.clip(cursor, 0, last_waypoint)
    waypoint_target = policy.waypoints[waypoint_cursor]
    waypoint_distance = _safe_norm(waypoint_target - state[:2])
    waypoint_radius = jnp.where(
        policy.kinds == POLICY_ROOM,
        jnp.where(
            waypoint_cursor == 0,
            ROOM_APPROACH_RADIUS,
            ROOM_WAYPOINT_RADIUS,
        ),
        NOMINAL_WAYPOINT_RADIUS,
    )
    waypoint_kind = (policy.kinds == POLICY_NOMINAL) | (
        policy.kinds == POLICY_ROOM
    )
    waypoint_should_advance = (
        waypoint_kind
        & (waypoint_distance < waypoint_radius)
        & (waypoint_cursor + 1 < waypoint_count)
    )
    waypoint_cursor = jnp.where(
        waypoint_should_advance,
        waypoint_cursor + 1,
        waypoint_cursor,
    )
    waypoint_target = policy.waypoints[waypoint_cursor]
    waypoint_u = _waypoint_control(
        state, waypoint_target, policy.target_speeds, parameters
    )

    retrace_cursor = jnp.clip(cursor, 0, last_waypoint)
    retrace_target = policy.waypoints[retrace_cursor]
    retrace_distance = _safe_norm(retrace_target - state[:2])
    should_advance = (
        (retrace_distance < RETRACE_WAYPOINT_RADIUS)
        & (retrace_cursor + 1 < waypoint_count)
    )
    retrace_cursor = jnp.where(
        should_advance, retrace_cursor + 1, retrace_cursor
    )
    retrace_target = policy.waypoints[retrace_cursor]
    retrace_delta = retrace_target - state[:2]
    retrace_distance = _safe_norm(retrace_delta)
    retrace_direction = retrace_delta / jnp.maximum(retrace_distance, 1.0e-6)
    braking_speed = jnp.sqrt(
        jnp.maximum(2.0 * parameters.a_max * retrace_distance, 0.0)
    )
    desired_speed = jnp.minimum(
        jnp.minimum(policy.target_speeds, braking_speed), parameters.v_max
    )
    retrace_desired = retrace_direction * desired_speed
    retrace_u = _clip_control(
        policy.feedback_gains * (retrace_desired - state[2:4]), parameters
    )

    directional_desired = policy.target_speeds * jnp.array(
        [jnp.cos(policy.angles), jnp.sin(policy.angles)]
    )
    directional_u = _clip_control(
        parameters.k_velocity * (directional_desired - state[2:4]),
        parameters,
    )
    stop_u = _clip_control(-parameters.stop_gain * state[2:4], parameters)
    control = jnp.zeros(2, dtype=state.dtype)
    control = jnp.where(
        (policy.kinds == POLICY_NOMINAL) | (policy.kinds == POLICY_ROOM),
        waypoint_u,
        control,
    )
    control = jnp.where(
        (policy.kinds == POLICY_ANGLE) | (policy.kinds == POLICY_REVERSE),
        directional_u,
        control,
    )
    control = jnp.where(policy.kinds == POLICY_STOP, stop_u, control)
    control = jnp.where(policy.kinds == POLICY_RETRACE, retrace_u, control)
    next_cursor = jnp.where(waypoint_kind, waypoint_cursor, cursor)
    next_cursor = jnp.where(
        policy.kinds == POLICY_RETRACE, retrace_cursor, next_cursor
    )
    return control, next_cursor


def _step_double_integrator(
    state: jax.Array,
    control: jax.Array,
    dt: jax.Array,
    parameters: HospitalJaxParameters,
) -> jax.Array:
    acceleration = _clip_control(control, parameters)
    velocity = state[2:4] + acceleration * dt
    speed = _safe_norm(velocity)
    velocity = jnp.where(
        speed > parameters.v_max,
        velocity * parameters.v_max / jnp.maximum(speed, 1.0e-9),
        velocity,
    )
    position = state[:2] + velocity * dt
    return jnp.concatenate((position, velocity))


def _rollout_one(
    initial_state: jax.Array,
    policy: HospitalJaxPolicyBatch,
    parameters: HospitalJaxParameters,
    horizon_steps: int,
) -> tuple[jax.Array, jax.Array]:
    origin = initial_state[:2]

    def advance(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        index: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        state, cursor, done = carry
        running = policy.active & ~done & (index < policy.horizon_steps)
        control, following_cursor = _policy_control(
            state, cursor, policy, parameters
        )
        candidate = _step_double_integrator(
            state, control, policy.rollout_dt, parameters
        )
        within_distance = (
            _safe_norm(candidate[:2] - origin) <= policy.max_rollout_distance
        )
        accepted = running & within_distance
        following = jnp.where(accepted, candidate, state)
        following_cursor = jnp.where(accepted, following_cursor, cursor)
        horizon_done = index + 1 >= policy.horizon_steps
        following_done = done | ~accepted | horizon_done
        return (
            (following, following_cursor, following_done),
            (following, accepted),
        )

    _, (tail, tail_mask) = jax.lax.scan(
        advance,
        (initial_state, jnp.asarray(0, dtype=jnp.int32), ~policy.active),
        jnp.arange(horizon_steps, dtype=jnp.int32),
    )
    states = jnp.concatenate((initial_state[None, :], tail), axis=0)
    mask = jnp.concatenate((policy.active[None], tail_mask), axis=0)
    return states, mask


def _dynamic_clearance(
    point: jax.Array,
    centers: jax.Array,
    obstacles: HospitalJaxObstacleBatch,
    parameters: HospitalJaxParameters,
) -> tuple[jax.Array, jax.Array]:
    circle = (
        _safe_norm(point[None, :] - centers)
        - obstacles.radii
        - parameters.robot_radius
        - parameters.safety_margin
        - parameters.human_margin
    )
    delta = point[None, :] - centers
    local_x = obstacles.cosines * delta[:, 0] + obstacles.sines * delta[:, 1]
    local_y = -obstacles.sines * delta[:, 0] + obstacles.cosines * delta[:, 1]
    qx = jnp.abs(local_x) - obstacles.half_lengths
    qy = jnp.abs(local_y) - obstacles.half_widths
    rectangle = (
        _safe_norm(jnp.stack((jnp.maximum(qx, 0.0), jnp.maximum(qy, 0.0)), axis=1))
        + jnp.minimum(jnp.maximum(qx, qy), 0.0)
        - parameters.robot_radius
        - parameters.safety_margin
        - parameters.stretcher_margin
    )
    values = jnp.where(
        obstacles.kinds == OBSTACLE_CIRCLE, circle, rectangle
    )
    mask = obstacles.active & (obstacles.kinds != OBSTACLE_INACTIVE)
    return values, mask


def _trajectory_value(
    trajectory: jax.Array,
    trajectory_mask: jax.Array,
    policy: HospitalJaxPolicyBatch,
    vertex_centers: jax.Array,
    swept_centers: jax.Array,
    obstacles: HospitalJaxObstacleBatch,
    geometry: HospitalJaxStaticGeometry,
    parameters: HospitalJaxParameters,
    swept_capacity: int,
) -> jax.Array:
    static_radius = parameters.robot_radius + parameters.static_margin

    def vertex_value(
        state: jax.Array, centers: jax.Array
    ) -> jax.Array:
        dynamic_values, dynamic_mask = _dynamic_clearance(
            state[:2], centers, obstacles, parameters
        )
        dynamic = _smooth_min_masked(
            dynamic_values, dynamic_mask, jnp.asarray(24.0)
        )
        static = _static_clearance(state[:2], static_radius, geometry)
        return _smooth_min_masked(
            jnp.stack((static, dynamic)),
            jnp.ones(2, dtype=bool),
            parameters.component_temperature,
        )

    vertex_values = jax.vmap(vertex_value)(trajectory, vertex_centers)
    segment_valid = trajectory_mask[:-1] & trajectory_mask[1:]
    sample_ids = jnp.arange(1, swept_capacity + 1)
    sample_mask = sample_ids <= policy.swept_samples
    alphas = sample_ids.astype(trajectory.dtype) / (
        policy.swept_samples.astype(trajectory.dtype) + 1.0
    )
    swept_points = (
        trajectory[:-1, None, :2]
        + alphas[None, :, None]
        * (trajectory[1:, None, :2] - trajectory[:-1, None, :2])
    )

    def swept_value(point: jax.Array, centers: jax.Array) -> jax.Array:
        values, mask = _dynamic_clearance(point, centers, obstacles, parameters)
        return _smooth_min_masked(values, mask, jnp.asarray(24.0))

    swept_values = jax.vmap(jax.vmap(swept_value))(swept_points, swept_centers)
    swept_mask = segment_valid[:, None] & sample_mask[None, :]
    values = jnp.concatenate((vertex_values, swept_values.reshape((-1,))))
    masks = jnp.concatenate((trajectory_mask, swept_mask.reshape((-1,))))

    terminal_index = jnp.maximum(jnp.sum(trajectory_mask.astype(jnp.int32)) - 1, 0)
    terminal = trajectory[terminal_index]
    x0, y0, x1, y1 = policy.room_bounds
    interior = jnp.min(
        jnp.array(
            [
                terminal[0] - x0,
                x1 - terminal[0],
                terminal[1] - y0,
                y1 - terminal[1],
            ]
        )
    ) - parameters.terminal_interior_margin
    terminal_speed = parameters.terminal_speed_max - _safe_norm(terminal[2:4])
    values = jnp.concatenate((values, jnp.array([interior, terminal_speed])))
    room_terminal_mask = policy.active & (policy.kinds == POLICY_ROOM) & policy.has_room
    masks = jnp.concatenate(
        (masks, jnp.array([room_terminal_mask, room_terminal_mask]))
    )
    return _smooth_min_masked(values, masks, parameters.time_temperature)


def _terminal_clearance(
    trajectory: jax.Array,
    trajectory_mask: jax.Array,
    vertex_centers: jax.Array,
    obstacles: HospitalJaxObstacleBatch,
    geometry: HospitalJaxStaticGeometry,
    parameters: HospitalJaxParameters,
) -> jax.Array:
    terminal_index = jnp.maximum(jnp.sum(trajectory_mask.astype(jnp.int32)) - 1, 0)
    terminal = trajectory[terminal_index]
    centers = vertex_centers[terminal_index]
    dynamic_values, dynamic_mask = _dynamic_clearance(
        terminal[:2], centers, obstacles, parameters
    )
    dynamic = jnp.min(jnp.where(dynamic_mask, dynamic_values, jnp.inf))
    static = _static_clearance(
        terminal[:2], parameters.robot_radius + parameters.static_margin, geometry
    )
    return jnp.minimum(static, dynamic)


def _evaluate_policy_arrays_with_checkpoints(
    state: jax.Array,
    nominal_control: jax.Array,
    policies: HospitalJaxPolicyBatch,
    obstacles: HospitalJaxObstacleBatch,
    geometry: HospitalJaxStaticGeometry,
    parameters: HospitalJaxParameters,
    checkpoint_positions: jax.Array,
    checkpoint_velocities: jax.Array,
    *,
    horizon_steps: int,
    swept_capacity: int,
    include_diagnostics: bool,
    share_policy_time_grids: bool = False,
) -> tuple[jax.Array, ...]:
    def tables(
        policy: HospitalJaxPolicyBatch, offset: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return _policy_obstacle_tables(
            policy,
            offset,
            obstacles,
            geometry,
            checkpoint_positions,
            checkpoint_velocities,
            horizon_steps,
            swept_capacity,
        )

    zero_offsets = jnp.zeros(policies.active.shape, dtype=state.dtype)
    shifted_offsets = jnp.full(
        policies.active.shape, parameters.time_derivative_step, dtype=state.dtype
    )
    if share_policy_time_grids:
        # Every ordinary Hospital policy uses the standard rollout grid and
        # every room policy uses the room grid.  Computing obstacle motion for
        # each candidate separately repeats the same bouncing-human scan up to
        # 23 times.  Build the two exact grids once and broadcast them; the
        # rollout/value/gradient calculation remains independently vmapped.
        room_mask = policies.active & (policies.kinds == POLICY_ROOM)
        standard_mask = policies.active & ~room_mask
        standard_dt = jnp.max(
            jnp.where(standard_mask, policies.rollout_dt, 0.0)
        )
        room_dt = jnp.max(jnp.where(room_mask, policies.rollout_dt, 0.0))
        standard_dt = jnp.where(standard_dt > 0.0, standard_dt, room_dt)
        room_dt = jnp.where(room_dt > 0.0, room_dt, standard_dt)
        standard_samples = jnp.max(
            jnp.where(standard_mask, policies.swept_samples, 1)
        )
        room_samples = jnp.max(
            jnp.where(room_mask, policies.swept_samples, standard_samples)
        )

        def shared_tables(offset: jax.Array) -> tuple[jax.Array, jax.Array]:
            standard_vertices, standard_swept = (
                _obstacle_tables_for_time_grid(
                    standard_dt,
                    standard_samples,
                    offset,
                    obstacles,
                    geometry,
                    checkpoint_positions,
                    checkpoint_velocities,
                    horizon_steps,
                    swept_capacity,
                )
            )
            room_vertices, room_swept = _obstacle_tables_for_time_grid(
                room_dt,
                room_samples,
                offset,
                obstacles,
                geometry,
                checkpoint_positions,
                checkpoint_velocities,
                horizon_steps,
                swept_capacity,
            )
            vertex_selector = room_mask.reshape(
                (room_mask.shape[0],) + (1,) * standard_vertices.ndim
            )
            swept_selector = room_mask.reshape(
                (room_mask.shape[0],) + (1,) * standard_swept.ndim
            )
            return (
                jnp.where(
                    vertex_selector,
                    room_vertices[None, ...],
                    standard_vertices[None, ...],
                ),
                jnp.where(
                    swept_selector,
                    room_swept[None, ...],
                    standard_swept[None, ...],
                ),
            )

        base_vertices, base_swept = shared_tables(jnp.asarray(0.0, state.dtype))
        shifted_vertices, shifted_swept = shared_tables(
            parameters.time_derivative_step
        )
    else:
        base_vertices, base_swept = jax.vmap(tables)(policies, zero_offsets)
        shifted_vertices, shifted_swept = jax.vmap(tables)(
            policies, shifted_offsets
        )
    if include_diagnostics:
        prefix_offsets = jnp.full(
            policies.active.shape, parameters.plant_dt, dtype=state.dtype
        )
        if share_policy_time_grids:
            prefix_vertices, prefix_swept = shared_tables(
                parameters.plant_dt
            )
        else:
            prefix_vertices, prefix_swept = jax.vmap(tables)(
                policies, prefix_offsets
            )
        prefix_state = _step_double_integrator(
            state, nominal_control, parameters.plant_dt, parameters
        )
    else:
        # These aliases add no computation; the mapped function's static
        # decision-only branch does not consume them.
        prefix_vertices, prefix_swept = base_vertices, base_swept
        prefix_state = state

    def evaluate_one(
        policy: HospitalJaxPolicyBatch,
        base_vertex_centers: jax.Array,
        base_swept_centers: jax.Array,
        shifted_vertex_centers: jax.Array,
        shifted_swept_centers: jax.Array,
        prefix_vertex_centers: jax.Array,
        prefix_swept_centers: jax.Array,
    ) -> tuple[jax.Array, ...]:
        def active_evaluation(_: None) -> tuple[jax.Array, ...]:
            def value_with_aux(candidate_state: jax.Array):
                trajectory, mask = _rollout_one(
                    candidate_state, policy, parameters, horizon_steps
                )
                value = _trajectory_value(
                    trajectory,
                    mask,
                    policy,
                    base_vertex_centers,
                    base_swept_centers,
                    obstacles,
                    geometry,
                    parameters,
                    swept_capacity,
                )
                return value, (trajectory, mask)

            (value, (trajectory, mask)), gradient = jax.value_and_grad(
                value_with_aux, has_aux=True
            )(state)
            shifted_value = _trajectory_value(
                trajectory,
                mask,
                policy,
                shifted_vertex_centers,
                shifted_swept_centers,
                obstacles,
                geometry,
                parameters,
                swept_capacity,
            )
            if include_diagnostics:
                prefix_trajectory, prefix_mask = _rollout_one(
                    prefix_state, policy, parameters, horizon_steps
                )
                prefix_value = _trajectory_value(
                    prefix_trajectory,
                    prefix_mask,
                    policy,
                    prefix_vertex_centers,
                    prefix_swept_centers,
                    obstacles,
                    geometry,
                    parameters,
                    swept_capacity,
                )
                terminal = _terminal_clearance(
                    trajectory,
                    mask,
                    base_vertex_centers,
                    obstacles,
                    geometry,
                    parameters,
                )
            else:
                prefix_value = jnp.asarray(jnp.nan, dtype=state.dtype)
                terminal = jnp.asarray(jnp.nan, dtype=state.dtype)
            return (
                value,
                gradient,
                shifted_value,
                prefix_value,
                terminal,
                trajectory,
                mask,
            )

        def inactive_evaluation(_: None) -> tuple[jax.Array, ...]:
            trajectory = jnp.broadcast_to(
                state[None, :], (horizon_steps + 1, state.shape[0])
            )
            return (
                jnp.asarray(-_MASKED_VALUE, dtype=state.dtype),
                jnp.zeros_like(state),
                jnp.asarray(-_MASKED_VALUE, dtype=state.dtype),
                jnp.asarray(-_MASKED_VALUE, dtype=state.dtype),
                jnp.asarray(-_MASKED_VALUE, dtype=state.dtype),
                trajectory,
                jnp.zeros((horizon_steps + 1,), dtype=bool),
            )

        return jax.lax.cond(
            policy.active, active_evaluation, inactive_evaluation, operand=None
        )

    outputs = jax.vmap(evaluate_one)(
        policies,
        base_vertices,
        base_swept,
        shifted_vertices,
        shifted_swept,
        prefix_vertices,
        prefix_swept,
    )
    values, gradients, shifted, prefix, terminal, trajectories, masks = outputs
    derivatives = (shifted - values) / parameters.time_derivative_step
    if not include_diagnostics:
        # Reverse-mode AD still uses each trajectory inside the executable, but
        # a headless decision does not need to transfer or retain those rollout
        # arrays on the host.  The diagnostics executable below remains the
        # visualization/debugging path with identical numerical calculations.
        return values, gradients, shifted, derivatives
    return (
        values,
        gradients,
        shifted,
        derivatives,
        prefix,
        terminal,
        trajectories,
        masks,
    )


def _evaluate_arrays(
    state: jax.Array,
    nominal_control: jax.Array,
    policies: HospitalJaxPolicyBatch,
    obstacles: HospitalJaxObstacleBatch,
    geometry: HospitalJaxStaticGeometry,
    parameters: HospitalJaxParameters,
    *,
    horizon_steps: int,
    swept_capacity: int,
    prediction_steps: int,
    include_diagnostics: bool,
) -> tuple[jax.Array, ...]:
    checkpoint_positions, checkpoint_velocities = _circle_checkpoints(
        obstacles, geometry, prediction_steps
    )
    return _evaluate_policy_arrays_with_checkpoints(
        state,
        nominal_control,
        policies,
        obstacles,
        geometry,
        parameters,
        checkpoint_positions,
        checkpoint_velocities,
        horizon_steps=horizon_steps,
        swept_capacity=swept_capacity,
        include_diagnostics=include_diagnostics,
    )


@lru_cache(maxsize=16)
def _shared_compiled_evaluator(structure: _EvaluatorStructure):
    """Return one process-wide executable for a fixed array structure."""

    capacity = structure.capacities

    def evaluate(
        state: jax.Array,
        nominal_control: jax.Array,
        policies: HospitalJaxPolicyBatch,
        obstacles: HospitalJaxObstacleBatch,
        geometry: HospitalJaxStaticGeometry,
        parameters: HospitalJaxParameters,
    ) -> tuple[jax.Array, ...]:
        return _evaluate_arrays(
            state,
            nominal_control,
            policies,
            obstacles,
            geometry,
            parameters,
            horizon_steps=capacity.max_horizon_steps,
            swept_capacity=capacity.max_swept_samples,
            prediction_steps=capacity.human_prediction_steps,
            include_diagnostics=structure.include_diagnostics,
        )

    return jax.jit(evaluate)


@lru_cache(maxsize=16)
def _shared_compiled_grouped_evaluator(
    structure: _GroupedEvaluatorStructure,
):
    """Return an executable sharing obstacle checkpoints across all groups."""

    capacities = structure.capacities

    def evaluate(
        state: jax.Array,
        nominal_control: jax.Array,
        policy_groups: tuple[HospitalJaxPolicyBatch, ...],
        obstacles: HospitalJaxObstacleBatch,
        geometry: HospitalJaxStaticGeometry,
        parameters: HospitalJaxParameters,
    ) -> tuple[tuple[jax.Array, ...], ...]:
        checkpoint_positions, checkpoint_velocities = _circle_checkpoints(
            obstacles, geometry, capacities.human_prediction_steps
        )
        return tuple(
            _evaluate_policy_arrays_with_checkpoints(
                state,
                nominal_control,
                policies,
                obstacles,
                geometry,
                parameters,
                checkpoint_positions,
                checkpoint_velocities,
                horizon_steps=group.max_horizon_steps,
                swept_capacity=group.max_swept_samples,
                include_diagnostics=structure.include_diagnostics,
                share_policy_time_grids=share_time_grid,
            )
            for group, policies, share_time_grid in zip(
                capacities.groups,
                policy_groups,
                structure.shared_time_grids,
                strict=True,
            )
        )

    return jax.jit(evaluate)


def _device_tree(value):
    return jax.tree_util.tree_map(jnp.asarray, value)


def _can_share_policy_time_grids(policies: PackedHospitalPolicies) -> bool:
    """Return whether ordinary and room candidates each share one time grid."""

    active = np.asarray(policies.batch.active[: policies.active_count], dtype=bool)
    kinds = np.asarray(policies.batch.kinds[: policies.active_count])
    rollout_dt = np.asarray(
        policies.batch.rollout_dt[: policies.active_count], dtype=float
    )
    swept_samples = np.asarray(
        policies.batch.swept_samples[: policies.active_count], dtype=int
    )
    for room_group in (False, True):
        selected = active & ((kinds == POLICY_ROOM) == room_group)
        if not np.any(selected):
            continue
        if (
            np.unique(rollout_dt[selected]).size != 1
            or np.unique(swept_samples[selected]).size != 1
        ):
            return False
    return True


def evaluate_policy_batch(
    state: Sequence[float],
    nominal_control: Sequence[float],
    policies: PackedHospitalPolicies,
    obstacles: HospitalJaxObstacleBatch,
    geometry: HospitalJaxStaticGeometry,
    parameters: HospitalJaxParameters,
    *,
    include_diagnostics: bool = True,
) -> HospitalJaxEvaluation:
    """Evaluate all active policies with one fixed-shape compiled call."""

    state_array = np.asarray(state, dtype=np.float32).reshape(4)
    nominal_array = np.asarray(nominal_control, dtype=np.float32).reshape(2)
    capacity = policies.capacities
    if obstacles.active.shape != (capacity.max_obstacles,):
        raise ValueError("obstacle batch does not match policy capacities")
    structure = _EvaluatorStructure(
        capacities=capacity,
        floor_count=int(geometry.floor_bounds.shape[0]),
        floor_boundary_count=int(geometry.floor_boundary_starts.shape[0]),
        wall_count=int(geometry.wall_bounds.shape[0]),
        include_diagnostics=bool(include_diagnostics),
    )
    compiled = _shared_compiled_evaluator(structure)
    outputs = compiled(
        jnp.asarray(state_array),
        jnp.asarray(nominal_array),
        _device_tree(policies.batch),
        _device_tree(obstacles),
        _device_tree(geometry),
        _device_tree(parameters),
    )
    host = tuple(np.asarray(item) for item in outputs)
    active = policies.active_count
    if include_diagnostics:
        prefix = host[4][:active].astype(float)
        terminal = host[5][:active].astype(float)
        trajectories = host[6][:active].astype(float)
        trajectory_mask = host[7][:active].astype(bool)
    else:
        prefix = np.full(active, np.nan, dtype=float)
        terminal = np.full(active, np.nan, dtype=float)
        trajectories = np.empty((active, 0, state_array.size), dtype=float)
        trajectory_mask = np.empty((active, 0), dtype=bool)
    return HospitalJaxEvaluation(
        names=policies.names,
        values=host[0][:active].astype(float),
        gradients=host[1][:active].astype(float),
        shifted_values=host[2][:active].astype(float),
        time_derivatives=host[3][:active].astype(float),
        nominal_prefix_values=prefix,
        terminal_clearances=terminal,
        trajectories=trajectories,
        trajectory_mask=trajectory_mask,
        diagnostics_available=bool(include_diagnostics),
    )


def evaluate_policy_groups(
    state: Sequence[float],
    nominal_control: Sequence[float],
    policies: PackedHospitalPolicyGroups,
    obstacles: HospitalJaxObstacleBatch,
    geometry: HospitalJaxStaticGeometry,
    parameters: HospitalJaxParameters,
    *,
    include_diagnostics: bool = True,
) -> HospitalJaxEvaluation:
    """Evaluate tight horizon groups with one shared obstacle prediction scan."""

    state_array = np.asarray(state, dtype=np.float32).reshape(4)
    nominal_array = np.asarray(nominal_control, dtype=np.float32).reshape(2)
    capacities = policies.capacities
    if obstacles.active.shape != (capacities.max_obstacles,):
        raise ValueError("obstacle batch does not match grouped capacities")
    if len(policies.groups) != len(capacities.groups):
        raise ValueError("packed policy groups do not match grouped capacities")
    structure = _GroupedEvaluatorStructure(
        capacities=capacities,
        floor_count=int(geometry.floor_bounds.shape[0]),
        floor_boundary_count=int(geometry.floor_boundary_starts.shape[0]),
        wall_count=int(geometry.wall_bounds.shape[0]),
        include_diagnostics=bool(include_diagnostics),
        shared_time_grids=tuple(
            _can_share_policy_time_grids(group) for group in policies.groups
        ),
    )
    compiled = _shared_compiled_grouped_evaluator(structure)
    outputs = compiled(
        jnp.asarray(state_array),
        jnp.asarray(nominal_array),
        tuple(_device_tree(group.batch) for group in policies.groups),
        _device_tree(obstacles),
        _device_tree(geometry),
        _device_tree(parameters),
    )
    host_groups = tuple(
        tuple(np.asarray(item) for item in group_outputs)
        for group_outputs in outputs
    )

    count = policies.active_count
    values = np.empty(count, dtype=float)
    gradients = np.empty((count, state_array.size), dtype=float)
    shifted = np.empty(count, dtype=float)
    derivatives = np.empty(count, dtype=float)
    prefix = np.empty(count, dtype=float)
    terminal = np.empty(count, dtype=float)
    if include_diagnostics:
        maximum_horizon = max(
            group.max_horizon_steps for group in capacities.groups
        )
        trajectories = np.empty(
            (count, maximum_horizon + 1, state_array.size), dtype=float
        )
        masks = np.zeros((count, maximum_horizon + 1), dtype=bool)
    else:
        prefix.fill(np.nan)
        terminal.fill(np.nan)
        trajectories = np.empty((count, 0, state_array.size), dtype=float)
        masks = np.empty((count, 0), dtype=bool)
    for packed_group, original_indices, host in zip(
        policies.groups,
        policies.original_indices,
        host_groups,
        strict=True,
    ):
        active = packed_group.active_count
        if active == 0:
            continue
        indices = np.asarray(original_indices, dtype=np.int32)
        values[indices] = host[0][:active]
        gradients[indices] = host[1][:active]
        shifted[indices] = host[2][:active]
        derivatives[indices] = host[3][:active]
        if include_diagnostics:
            prefix[indices] = host[4][:active]
            terminal[indices] = host[5][:active]
            group_trajectories = host[6][:active]
            group_masks = host[7][:active]
            group_length = group_trajectories.shape[1]
            trajectories[indices] = group_trajectories[:, -1:, :]
            trajectories[indices, :group_length] = group_trajectories
            masks[indices, :group_length] = group_masks
    return HospitalJaxEvaluation(
        names=policies.names,
        values=values,
        gradients=gradients,
        shifted_values=shifted,
        time_derivatives=derivatives,
        nominal_prefix_values=prefix,
        terminal_clearances=terminal,
        trajectories=trajectories,
        trajectory_mask=masks,
        diagnostics_available=bool(include_diagnostics),
    )


def warmup_policy_batch(
    state: Sequence[float],
    nominal_control: Sequence[float],
    policies: PackedHospitalPolicies,
    obstacles: HospitalJaxObstacleBatch,
    geometry: HospitalJaxStaticGeometry,
    parameters: HospitalJaxParameters,
    *,
    include_diagnostics: bool = False,
) -> None:
    """Compile and execute the shared evaluator outside benchmark timing."""

    evaluate_policy_batch(
        state,
        nominal_control,
        policies,
        obstacles,
        geometry,
        parameters,
        include_diagnostics=include_diagnostics,
    )


def warmup_policy_groups(
    state: Sequence[float],
    nominal_control: Sequence[float],
    policies: PackedHospitalPolicyGroups,
    obstacles: HospitalJaxObstacleBatch,
    geometry: HospitalJaxStaticGeometry,
    parameters: HospitalJaxParameters,
    *,
    include_diagnostics: bool = False,
) -> None:
    """Compile and execute the grouped evaluator outside benchmark timing."""

    evaluate_policy_groups(
        state,
        nominal_control,
        policies,
        obstacles,
        geometry,
        parameters,
        include_diagnostics=include_diagnostics,
    )


def warmup_obstacle_buckets(
    state: Sequence[float],
    nominal_control: Sequence[float],
    policy_library: Sequence[HospitalPolicy],
    environment: HospitalEnvironment,
    config: HospitalConfig,
    *,
    max_policies: int = 24,
    buckets: Sequence[int] = DEFAULT_OBSTACLE_BUCKETS,
    include_retrace: bool = False,
    include_diagnostics: bool = False,
) -> tuple[HospitalJaxCapacities, ...]:
    """Precompile every supported local-obstacle shape outside timing.

    Zero-valued obstacle slots are dynamic inputs, not closed-over constants,
    so this compiles the same executable later used by a populated bucket.
    A benchmark can call this once per process and subsequently select the
    smallest fitting bucket with :func:`capacities_for_obstacle_count` without
    incurring a runtime compilation.
    """

    normalized = tuple(sorted({int(item) for item in buckets}))
    if not normalized or normalized[0] < 1:
        raise ValueError("obstacle buckets must contain positive integers")
    geometry = pack_static_geometry(environment)
    parameters = pack_parameters(config)
    output: list[HospitalJaxCapacities] = []
    for bucket in normalized:
        capacity = HospitalJaxCapacities.for_config(
            config,
            max_policies=max_policies,
            max_obstacles=bucket,
            include_retrace=include_retrace,
        )
        packed_policies = pack_policy_batch(policy_library, config, capacity)
        packed_obstacles = pack_obstacle_batch((), capacity)
        warmup_policy_batch(
            state,
            nominal_control,
            packed_policies,
            packed_obstacles,
            geometry,
            parameters,
            include_diagnostics=include_diagnostics,
        )
        output.append(capacity)
    return tuple(output)


def compiled_evaluator_cache_info():
    """Expose cache statistics for regression tests and benchmark metadata."""

    return _shared_compiled_evaluator.cache_info()


def compiled_grouped_evaluator_cache_info():
    """Expose grouped executable cache statistics for regression tests."""

    return _shared_compiled_grouped_evaluator.cache_info()


def clear_compiled_evaluator_cache() -> None:
    """Clear the process-wide executable cache (tests only)."""

    _shared_compiled_evaluator.cache_clear()
    _shared_compiled_grouped_evaluator.cache_clear()


__all__ = [
    "DEFAULT_OBSTACLE_BUCKETS",
    "HospitalJaxCapacities",
    "HospitalJaxEvaluation",
    "HospitalJaxGroupedCapacities",
    "HospitalJaxObstacleBatch",
    "HospitalJaxParameters",
    "HospitalJaxPolicyBatch",
    "HospitalJaxPolicyGroupSpec",
    "HospitalJaxStaticGeometry",
    "OBSTACLE_CIRCLE",
    "OBSTACLE_INACTIVE",
    "OBSTACLE_RECTANGLE",
    "POLICY_ANGLE",
    "POLICY_INACTIVE",
    "POLICY_NOMINAL",
    "POLICY_RETRACE",
    "POLICY_REVERSE",
    "POLICY_ROOM",
    "POLICY_STOP",
    "PackedHospitalPolicies",
    "PackedHospitalPolicyGroups",
    "capacities_for_obstacle_count",
    "clear_compiled_evaluator_cache",
    "compiled_evaluator_cache_info",
    "compiled_grouped_evaluator_cache_info",
    "evaluate_policy_groups",
    "evaluate_policy_batch",
    "grouped_capacities_for_obstacle_count",
    "pack_obstacle_batch",
    "pack_parameters",
    "pack_policy_batch",
    "pack_policy_groups",
    "pack_static_geometry",
    "select_obstacle_bucket",
    "warmup_obstacle_buckets",
    "warmup_policy_batch",
    "warmup_policy_groups",
]
