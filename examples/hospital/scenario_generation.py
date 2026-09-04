"""Deterministic playground-scale traffic generation for hospital scenarios.

The generated traffic is intentionally independent of the guaranteed
full-width convoy used by the refuge benchmark.  A caller can therefore layer
two or three blocking stretchers over the playground's full background
density, yielding 17 or 18 total stretchers respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import DEFAULT_CONFIG, HospitalConfig
from .environment import HospitalEnvironment, Rect
from .obstacles import DynamicObstacle, Human, Stretcher, stretcher_route


# These values mirror the browser playground's default crowd density.  The
# guaranteed full-width event is added on top of that background traffic.
DEFAULT_HUMAN_COUNT = 50
DEFAULT_ORDINARY_STRETCHER_COUNT = 15
GUARANTEED_BLOCKER_COUNTS = (2, 3)
TOTAL_STRETCHER_COUNTS_WITH_BLOCKERS = (17, 18)

TRAFFIC_SPEED_CAP = 1.45
HUMAN_SPEED_RANGE = (0.42 * TRAFFIC_SPEED_CAP, TRAFFIC_SPEED_CAP)
STRETCHER_SPEED_RANGE = (0.38 * TRAFFIC_SPEED_CAP, TRAFFIC_SPEED_CAP)
HUMAN_RADIUS = 0.52
STRETCHER_LENGTH = 4.1
STRETCHER_WIDTH = 1.45

DEFAULT_PROTECTED_CLEARANCE = 5.0
INITIAL_PAIRWISE_CLEARANCE = 0.16
STATIC_PLACEMENT_MARGIN = 0.06
MAX_PLACEMENT_ATTEMPTS_PER_OBSTACLE = 400


@dataclass(frozen=True)
class HospitalCrowdMetadata:
    """Reproducibility and validation information for a generated crowd."""

    seed: int
    human_count: int
    ordinary_stretcher_count: int
    protected_point_count: int
    protected_clearance: float
    minimum_pairwise_clearance: float | None
    minimum_protected_clearance: float | None
    placement_attempts: int

    @property
    def obstacle_count(self) -> int:
        return self.human_count + self.ordinary_stretcher_count


@dataclass(frozen=True)
class GeneratedHospitalCrowd:
    """Background dynamic traffic ready to pass to ``HospitalSimulation``."""

    humans: tuple[Human, ...]
    stretchers: tuple[Stretcher, ...]
    metadata: HospitalCrowdMetadata

    @property
    def obstacles(self) -> tuple[DynamicObstacle, ...]:
        return (*self.humans, *self.stretchers)


def obstacle_pair_clearance(
    first: DynamicObstacle,
    second: DynamicObstacle,
) -> float:
    """Return physical signed separation between two dynamic obstacles."""

    if isinstance(first, Human) and isinstance(second, Human):
        return (
            float(np.linalg.norm(first.center - second.center))
            - first.radius
            - second.radius
        )
    if isinstance(first, Human) and isinstance(second, Stretcher):
        return second.signed_clearance(first.center, first.radius)
    if isinstance(first, Stretcher) and isinstance(second, Human):
        return first.signed_clearance(second.center, second.radius)
    if not isinstance(first, Stretcher) or not isinstance(second, Stretcher):
        raise TypeError("unsupported dynamic obstacle type")

    first_half = _stretcher_half_extents(first)
    second_half = _stretcher_half_extents(second)
    q = np.abs(first.center - second.center) - first_half - second_half
    outside = float(np.linalg.norm(np.maximum(q, 0.0)))
    inside = min(max(float(q[0]), float(q[1])), 0.0)
    return outside + inside


def generate_hospital_crowd(
    environment: HospitalEnvironment,
    config: HospitalConfig = DEFAULT_CONFIG,
    *,
    seed: int,
    ego_position: Sequence[float],
    goal_position: Sequence[float],
    human_count: int = DEFAULT_HUMAN_COUNT,
    ordinary_stretcher_count: int = DEFAULT_ORDINARY_STRETCHER_COUNT,
    protected_points: Sequence[Sequence[float]] = (),
    protected_clearance: float = DEFAULT_PROTECTED_CLEARANCE,
    existing_obstacles: Sequence[DynamicObstacle] = (),
    human_corridors: Sequence[Rect] | None = None,
) -> GeneratedHospitalCrowd:
    """Generate non-overlapping dynamic hospital traffic from ``seed``.

    Humans are sampled throughout the traversable corridor union by default.
    ``human_corridors`` can restrict that sampling to a fixed,
    method-independent subset (for example, the corridor and junctions
    involved in a publication story). Ordinary stretchers are placed on
    reflecting, corridor-aligned routes outside the main corridor so the
    strict benchmark can layer its guaranteed convoy there. ``ego_position``,
    ``goal_position``, and any additional protected points retain the requested
    obstacle-to-robot clearance.
    ``existing_obstacles`` lets callers place the guaranteed convoy first;
    generated traffic will then also be separated from those blockers.
    """

    _validate_generation_arguments(
        human_count,
        ordinary_stretcher_count,
        protected_clearance,
    )
    protected = np.asarray(
        [ego_position, goal_position, *protected_points],
        dtype=float,
    )
    if protected.ndim != 2 or protected.shape[1] != 2:
        raise ValueError("protected points must be two-dimensional")
    if not np.all(np.isfinite(protected)):
        raise ValueError("protected points must be finite")

    rng = np.random.default_rng(int(seed))
    attempts = 0
    stretchers: list[Stretcher] = []
    obstacles: list[DynamicObstacle] = list(existing_obstacles)

    stretcher_corridors = [
        corridor
        for corridor in environment.corridor_rects
        if corridor.name != "Main corridor"
        and _corridor_supports_stretcher(corridor)
    ]
    if ordinary_stretcher_count and not stretcher_corridors:
        raise RuntimeError("hospital has no corridor that can contain a stretcher")

    for index in range(ordinary_stretcher_count):
        accepted: Stretcher | None = None
        for _ in range(MAX_PLACEMENT_ATTEMPTS_PER_OBSTACLE):
            attempts += 1
            corridor = stretcher_corridors[
                int(rng.integers(0, len(stretcher_corridors)))
            ]
            candidate = _sample_stretcher(
                rng,
                corridor,
                index,
            )
            if not _rectangle_is_static_safe(
                candidate,
                environment,
                STATIC_PLACEMENT_MARGIN,
            ):
                continue
            if not _is_clear_of_protected_points(
                candidate,
                protected,
                config.robot.radius,
                protected_clearance,
            ):
                continue
            if not _is_clear_of_obstacles(candidate, obstacles):
                continue
            accepted = candidate
            break
        if accepted is None:
            raise RuntimeError(
                "could not place the requested ordinary stretchers without "
                f"overlap (placed {len(stretchers)} of "
                f"{ordinary_stretcher_count})"
            )
        stretchers.append(accepted)
        obstacles.append(accepted)

    humans: list[Human] = []
    requested_human_corridors = (
        environment.corridor_rects
        if human_corridors is None
        else tuple(human_corridors)
    )
    sampled_human_corridors, human_probabilities = _weighted_human_corridors(
        requested_human_corridors
    )
    if human_count and not sampled_human_corridors:
        raise RuntimeError("hospital has no corridor that can contain a human")
    for index in range(human_count):
        accepted_human: Human | None = None
        for _ in range(MAX_PLACEMENT_ATTEMPTS_PER_OBSTACLE):
            attempts += 1
            corridor_index = int(
                rng.choice(
                    len(sampled_human_corridors),
                    p=human_probabilities,
                )
            )
            candidate = _sample_human(
                rng,
                sampled_human_corridors[corridor_index],
                index,
            )
            if environment.is_collision(
                candidate.center,
                candidate.radius + STATIC_PLACEMENT_MARGIN,
            ):
                continue
            if not _is_clear_of_protected_points(
                candidate,
                protected,
                config.robot.radius,
                protected_clearance,
            ):
                continue
            if not _is_clear_of_obstacles(candidate, obstacles):
                continue
            accepted_human = candidate
            break
        if accepted_human is None:
            raise RuntimeError(
                "could not place the requested humans without overlap "
                f"(placed {len(humans)} of {human_count})"
            )
        humans.append(accepted_human)
        obstacles.append(accepted_human)

    all_generated: list[DynamicObstacle] = [*stretchers, *humans]
    pairwise = [
        obstacle_pair_clearance(first, second)
        for index, first in enumerate(all_generated)
        for second in all_generated[index + 1 :]
    ]
    pairwise.extend(
        obstacle_pair_clearance(generated, existing)
        for generated in all_generated
        for existing in existing_obstacles
    )
    protected_separations = [
        _obstacle_point_clearance(obstacle, point, config.robot.radius)
        for obstacle in all_generated
        for point in protected
    ]
    metadata = HospitalCrowdMetadata(
        seed=int(seed),
        human_count=len(humans),
        ordinary_stretcher_count=len(stretchers),
        protected_point_count=len(protected),
        protected_clearance=float(protected_clearance),
        minimum_pairwise_clearance=min(pairwise) if pairwise else None,
        minimum_protected_clearance=(
            min(protected_separations) if protected_separations else None
        ),
        placement_attempts=attempts,
    )
    return GeneratedHospitalCrowd(
        humans=tuple(humans),
        stretchers=tuple(stretchers),
        metadata=metadata,
    )


def _validate_generation_arguments(
    human_count: int,
    ordinary_stretcher_count: int,
    protected_clearance: float,
) -> None:
    if isinstance(human_count, bool) or human_count < 0:
        raise ValueError("human_count must be a nonnegative integer")
    if int(human_count) != human_count:
        raise ValueError("human_count must be a nonnegative integer")
    if (
        isinstance(ordinary_stretcher_count, bool)
        or ordinary_stretcher_count < 0
    ):
        raise ValueError(
            "ordinary_stretcher_count must be a nonnegative integer"
        )
    if int(ordinary_stretcher_count) != ordinary_stretcher_count:
        raise ValueError(
            "ordinary_stretcher_count must be a nonnegative integer"
        )
    if not np.isfinite(protected_clearance) or protected_clearance < 0.0:
        raise ValueError("protected_clearance must be finite and nonnegative")


def _corridor_supports_stretcher(corridor: Rect) -> bool:
    route_span = (
        corridor.width if corridor.width >= corridor.height else corridor.height
    )
    lane_span = (
        corridor.height if corridor.width >= corridor.height else corridor.width
    )
    return (
        route_span > STRETCHER_LENGTH + 0.7
        and lane_span > STRETCHER_WIDTH + 0.6
    )


def _sample_stretcher(
    rng: np.random.Generator,
    corridor: Rect,
    index: int,
) -> Stretcher:
    horizontal = corridor.width >= corridor.height
    half_length = 0.5 * STRETCHER_LENGTH
    half_width = 0.5 * STRETCHER_WIDTH
    if horizontal:
        coordinate = rng.uniform(
            corridor.x + half_length + 0.35,
            corridor.x1 - half_length - 0.35,
        )
        lateral = rng.uniform(
            corridor.y + half_width + 0.3,
            corridor.y1 - half_width - 0.3,
        )
    else:
        coordinate = rng.uniform(
            corridor.y + half_length + 0.35,
            corridor.y1 - half_length - 0.35,
        )
        lateral = rng.uniform(
            corridor.x + half_width + 0.3,
            corridor.x1 - half_width - 0.3,
        )
    axis, coordinate, lateral, route_min, route_max = stretcher_route(
        corridor,
        coordinate=coordinate,
        lateral=lateral,
        length=STRETCHER_LENGTH,
        width=STRETCHER_WIDTH,
    )
    magnitude = float(rng.uniform(*STRETCHER_SPEED_RANGE))
    # Alternating signs guarantee bidirectional background traffic whenever
    # at least two stretchers are requested; speed magnitudes remain seeded.
    direction = 1.0 if index % 2 == 0 else -1.0
    return Stretcher(
        identifier=f"random-stretcher-{index:03d}",
        coordinate=float(coordinate),
        lateral=float(lateral),
        speed=direction * magnitude,
        axis=axis,
        route_min=float(route_min),
        route_max=float(route_max),
        length=STRETCHER_LENGTH,
        width=STRETCHER_WIDTH,
        reflect_at_route_bounds=True,
    )


def _weighted_human_corridors(
    corridors: Sequence[Rect],
) -> tuple[list[Rect], np.ndarray]:
    margin = HUMAN_RADIUS + STATIC_PLACEMENT_MARGIN
    valid: list[Rect] = []
    weights: list[float] = []
    for corridor in corridors:
        inner_width = corridor.width - 2.0 * margin
        inner_height = corridor.height - 2.0 * margin
        if inner_width <= 0.0 or inner_height <= 0.0:
            continue
        valid.append(corridor)
        weights.append(inner_width * inner_height)
    if not valid:
        return [], np.asarray([], dtype=float)
    probabilities = np.asarray(weights, dtype=float)
    probabilities /= np.sum(probabilities)
    return valid, probabilities


def _sample_human(
    rng: np.random.Generator,
    corridor: Rect,
    index: int,
) -> Human:
    margin = HUMAN_RADIUS + STATIC_PLACEMENT_MARGIN
    point = np.array(
        [
            rng.uniform(corridor.x + margin, corridor.x1 - margin),
            rng.uniform(corridor.y + margin, corridor.y1 - margin),
        ],
        dtype=float,
    )
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    speed = float(rng.uniform(*HUMAN_SPEED_RANGE))
    return Human(
        identifier=f"random-human-{index:03d}",
        x=float(point[0]),
        y=float(point[1]),
        vx=float(np.cos(angle) * speed),
        vy=float(np.sin(angle) * speed),
        radius=HUMAN_RADIUS,
    )


def _stretcher_half_extents(stretcher: Stretcher) -> np.ndarray:
    if stretcher.axis == "x":
        return np.array(
            [0.5 * stretcher.length, 0.5 * stretcher.width],
            dtype=float,
        )
    return np.array(
        [0.5 * stretcher.width, 0.5 * stretcher.length],
        dtype=float,
    )


def _rectangle_is_static_safe(
    stretcher: Stretcher,
    environment: HospitalEnvironment,
    margin: float,
) -> bool:
    half = _stretcher_half_extents(stretcher) + margin
    # Sampling each padded edge at <= 0.2 m is conservative for this
    # axis-aligned floor plan and catches both wall crossings and floor gaps.
    x_samples = np.linspace(
        stretcher.center[0] - half[0],
        stretcher.center[0] + half[0],
        max(2, int(np.ceil(2.0 * half[0] / 0.2)) + 1),
    )
    y_samples = np.linspace(
        stretcher.center[1] - half[1],
        stretcher.center[1] + half[1],
        max(2, int(np.ceil(2.0 * half[1] / 0.2)) + 1),
    )
    perimeter = [
        *((x, y_samples[0]) for x in x_samples),
        *((x, y_samples[-1]) for x in x_samples),
        *((x_samples[0], y) for y in y_samples),
        *((x_samples[-1], y) for y in y_samples),
    ]
    return all(
        not environment.is_collision(point)
        for point in perimeter
    )


def _obstacle_point_clearance(
    obstacle: DynamicObstacle,
    point: Sequence[float],
    robot_radius: float,
) -> float:
    return obstacle.signed_clearance(point, robot_radius)


def _is_clear_of_protected_points(
    obstacle: DynamicObstacle,
    protected_points: np.ndarray,
    robot_radius: float,
    protected_clearance: float,
) -> bool:
    return all(
        _obstacle_point_clearance(obstacle, point, robot_radius)
        >= protected_clearance
        for point in protected_points
    )


def _is_clear_of_obstacles(
    candidate: DynamicObstacle,
    obstacles: Sequence[DynamicObstacle],
) -> bool:
    return all(
        obstacle_pair_clearance(candidate, obstacle)
        >= INITIAL_PAIRWISE_CLEARANCE
        for obstacle in obstacles
    )


__all__ = [
    "DEFAULT_HUMAN_COUNT",
    "DEFAULT_ORDINARY_STRETCHER_COUNT",
    "DEFAULT_PROTECTED_CLEARANCE",
    "GUARANTEED_BLOCKER_COUNTS",
    "GeneratedHospitalCrowd",
    "HUMAN_RADIUS",
    "HUMAN_SPEED_RANGE",
    "HospitalCrowdMetadata",
    "INITIAL_PAIRWISE_CLEARANCE",
    "STRETCHER_LENGTH",
    "STRETCHER_SPEED_RANGE",
    "STRETCHER_WIDTH",
    "TOTAL_STRETCHER_COUNTS_WITH_BLOCKERS",
    "TRAFFIC_SPEED_CAP",
    "generate_hospital_crowd",
    "obstacle_pair_clearance",
]
