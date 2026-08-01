"""Deterministic nonlinear-quadrotor scenarios and spherical obstacles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .dynamics import make_state


PLAYGROUND_CROWDED_SCENARIO = "playground_crowded"
PLAYGROUND_STRESS_SCENARIO = "playground_stress"
PLAYGROUND_OBSTACLE_COUNT = 32
PLAYGROUND_REFERENCE_OBSTACLE_COUNT = 5
PLAYGROUND_OBSTACLE_RADIUS = 0.5
PLAYGROUND_OBSTACLE_SPEED_MIN = 0.45
PLAYGROUND_OBSTACLE_SPEED_MAX = 1.5
PLAYGROUND_START_GOAL_PROTECTION = 3.5

# The stress protocol remains recognizably playground-derived (same plant,
# world, endpoints, sphere radius, and reflecting obstacle dynamics), but
# concentrates traffic around the route instead of spending most samples in
# the large unused corners of the 20 x 20 x 10 m box.  The first twenty-four
# obstacles form four time-coordinated, balanced +/-x, +/-y, and +/-z stream
# events.  Their ordering is part of the protocol and is useful for auditing,
# never for control logic.
PLAYGROUND_STRESS_PROTOCOL_VERSION = "balanced_six_axis_streams_v2"
PLAYGROUND_STRESS_OBSTACLE_COUNT = 48
PLAYGROUND_STRESS_STRUCTURED_STREAM_COUNT = 24
# Backward-compatible name for downstream callers written against the first
# stress draft.  It now counts every structured stream, including +/-x.
PLAYGROUND_STRESS_CROSS_FLOW_COUNT = PLAYGROUND_STRESS_STRUCTURED_STREAM_COUNT
PLAYGROUND_STRESS_STREAM_DIRECTIONS = (
    "+x",
    "-x",
    "+y",
    "-y",
    "+z",
    "-z",
)
PLAYGROUND_STRESS_OBSTACLE_SPEED_MIN = 0.75
PLAYGROUND_STRESS_OBSTACLE_SPEED_MAX = 2.25
PLAYGROUND_STRESS_PAIR_CLEARANCE = 0.15
PLAYGROUND_STRESS_CORRIDOR_LOWER = (2.5, 3.5, 1.25)
PLAYGROUND_STRESS_CORRIDOR_UPPER = (17.5, 16.5, 8.75)


@dataclass(frozen=True)
class WorldBounds:
    """Axis-aligned world used only for spherical obstacle reflection."""

    lower: tuple[float, float, float] = (-12.0, -12.0, -2.0)
    upper: tuple[float, float, float] = (32.0, 15.0, 16.0)

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.upper) <= np.asarray(self.lower)):
            raise ValueError("every upper world bound must exceed its lower bound")


@dataclass(frozen=True)
class NLQuad3DScenario:
    """A named waypoint problem with ``[position, radius, velocity]`` obstacles."""

    name: str
    waypoints: np.ndarray
    obstacles: np.ndarray
    bounds: WorldBounds | None = WorldBounds()
    description: str = ""
    reach_threshold: float = 0.8
    default_steps: int = 800
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        waypoints = np.asarray(self.waypoints, dtype=float)
        obstacles = np.asarray(self.obstacles, dtype=float)
        initial_velocity = np.asarray(self.initial_velocity, dtype=float)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3:
            raise ValueError("waypoints must have shape (N, 3)")
        if obstacles.size == 0:
            obstacles = np.zeros((0, 7), dtype=float)
        if obstacles.ndim != 2 or obstacles.shape[1] != 7:
            raise ValueError("obstacles must have shape (M, 7)")
        if np.any(obstacles[:, 3] <= 0.0):
            raise ValueError("obstacle radii must be positive")
        if initial_velocity.shape != (3,) or not np.all(
            np.isfinite(initial_velocity)
        ):
            raise ValueError("initial_velocity must contain three finite values")
        object.__setattr__(self, "waypoints", waypoints.copy())
        object.__setattr__(self, "obstacles", obstacles.copy())
        object.__setattr__(
            self,
            "initial_velocity",
            tuple(float(value) for value in initial_velocity),
        )

    @property
    def initial_state(self) -> np.ndarray:
        return make_state(self.waypoints[0], self.initial_velocity)

    @property
    def goal(self) -> np.ndarray:
        return self.waypoints[-1].copy()


def _reflect_axis(
    position: np.ndarray,
    velocity: np.ndarray,
    radius: np.ndarray,
    dt: float,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    low = lower + radius
    high = upper - radius
    span = high - low
    if np.any(span <= 0.0):
        raise ValueError("an obstacle diameter exceeds a world dimension")
    raw = position + velocity * dt
    phase = np.mod(raw - low, 2.0 * span)
    forward = phase <= span
    reflected = low + np.where(forward, phase, 2.0 * span - phase)
    reflected_velocity = np.where(forward, velocity, -velocity)
    return reflected, reflected_velocity


def advance_obstacles(
    obstacles: np.ndarray,
    dt: float,
    bounds: WorldBounds | None,
) -> np.ndarray:
    """Advance moving spheres, reflecting elastically at optional world bounds."""

    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    result = np.asarray(obstacles, dtype=float).reshape(-1, 7).copy()
    if result.shape[0] == 0:
        return result
    if bounds is None:
        result[:, :3] += result[:, 4:7] * dt
        return result
    for axis in range(3):
        result[:, axis], result[:, axis + 4] = _reflect_axis(
            result[:, axis],
            result[:, axis + 4],
            result[:, 3],
            dt,
            bounds.lower[axis],
            bounds.upper[axis],
        )
    return result


def predict_obstacles(
    obstacles: np.ndarray,
    time: float,
    bounds: WorldBounds | None,
) -> np.ndarray:
    """Predict obstacle states without mutating the supplied snapshot."""

    return advance_obstacles(obstacles, time, bounds)


def minimum_clearance(
    position: np.ndarray,
    obstacles: np.ndarray,
    robot_radius: float,
) -> float:
    obstacle_array = np.asarray(obstacles, dtype=float).reshape(-1, 7)
    if obstacle_array.shape[0] == 0:
        return float("inf")
    distances = np.linalg.norm(obstacle_array[:, :3] - position, axis=1)
    return float(np.min(distances - obstacle_array[:, 3] - robot_radius))


def _head_on() -> NLQuad3DScenario:
    return NLQuad3DScenario(
        "head_on",
        np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        np.array([[18.0, 0.0, 0.0, 0.6, -2.5, 0.0, 0.0]]),
        description="High relative velocity directly on the approach axis.",
    )


def _cross_traffic() -> NLQuad3DScenario:
    return NLQuad3DScenario(
        "cross_traffic",
        np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        np.array(
            [
                [6.0, -8.0, 0.0, 0.5, 0.0, 2.0, 0.0],
                [12.0, 8.0, 0.0, 0.5, 0.0, -2.0, 0.0],
                [16.0, -6.0, 0.0, 0.5, 0.0, 1.5, 0.0],
            ]
        ),
        description="Three lateral crossings at different arrival times.",
    )


def _vertical_drop() -> NLQuad3DScenario:
    return NLQuad3DScenario(
        "vertical_drop",
        np.array([[0.0, 0.0, 2.0], [15.0, 0.0, 2.0]]),
        np.array(
            [
                [5.0, 0.0, 10.0, 0.6, 0.0, 0.0, -2.5],
                [10.0, 0.0, 12.0, 0.6, 0.0, 0.0, -3.0],
            ]
        ),
        description="Obstacles drop vertically through a horizontal route.",
    )


def _moving_wall() -> NLQuad3DScenario:
    obstacles: list[list[float]] = []
    for y in (-2.5, -1.0, 0.5, 2.0):
        for z in (0.5, 2.0, 3.5):
            if y == 0.5 and z == 2.0:
                continue
            obstacles.append([15.0, y, z, 0.6, -1.0, 0.0, 0.0])
    return NLQuad3DScenario(
        "moving_wall",
        np.array([[0.0, 0.0, 2.0], [20.0, 0.0, 2.0]]),
        np.asarray(obstacles),
        description="An approaching wall of spheres with one opening.",
    )


def _asteroid_field() -> NLQuad3DScenario:
    obstacles = []
    for index in range(12):
        obstacles.append(
            [
                5.0 + index * 1.2,
                np.sin(index) * 5.0,
                1.0 + index % 4,
                0.3 + (index % 3) * 0.15,
                -0.5 + np.cos(index) * 0.5,
                np.sin(index * 2) * 0.8,
                np.cos(index) * 0.4,
            ]
        )
    return NLQuad3DScenario(
        "asteroid_field",
        np.array([[0.0, 0.0, 1.0], [10.0, 8.0, 4.0], [20.0, 0.0, 1.0]]),
        np.asarray(obstacles),
        description="Dense deterministic multi-directional traffic.",
    )


def _collapsing_sphere() -> NLQuad3DScenario:
    center = np.array([0.0, 0.0, 5.0])
    obstacles = []
    count = 75
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for index in range(count):
        y = 1.0 - index / float(count - 1) * 2.0
        radial = np.sqrt(1.0 - y * y)
        theta = golden * index
        direction = np.array(
            [np.cos(theta) * radial, y, np.sin(theta) * radial]
        )
        position = center + direction * 5.0
        velocity = -direction * 0.4
        obstacles.append([*position, 0.15, *velocity])
    return NLQuad3DScenario(
        "collapsing_sphere",
        np.array([center, [15.0, 0.0, 5.0]]),
        np.asarray(obstacles),
        bounds=None,
        description="A 75-sphere cage closes around the launch point.",
    )


def _climb_and_dodge() -> NLQuad3DScenario:
    return NLQuad3DScenario(
        "climb_and_dodge",
        np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 4.0]]),
        np.array(
            [
                [5.0, 0.0, 0.5, 0.5, -0.5, 0.0, 0.3],
                [10.0, -3.0, 3.5, 0.5, 0.0, 0.5, 0.0],
                [20.0, 0.0, 6.5, 0.5, -0.5, 0.0, -0.3],
            ]
        ),
        description="Diagonal climb through scattered moving obstacles.",
    )


def _descend_through() -> NLQuad3DScenario:
    return NLQuad3DScenario(
        "descend_through",
        np.array([[0.0, 0.0, 5.0], [20.0, 0.0, 0.5]]),
        np.array(
            [
                [4.0, 0.0, 4.0, 0.5, 0.0, 0.0, 0.5],
                [9.0, 2.0, 3.0, 0.5, 0.0, -0.8, 0.0],
                [14.0, 0.0, 1.5, 0.5, 0.0, 0.0, -0.4],
            ]
        ),
        description="Diagonal descent through scattered moving obstacles.",
    )


def _vertical_climb() -> NLQuad3DScenario:
    obstacles = np.array(
        [
            [-1.0, -1.0, 2.5, 0.25, 0.5, 0.0, 0.0],
            [-1.0, 0.0, 2.5, 0.25, 0.5, 0.0, 0.0],
            [-1.0, 1.0, 2.5, 0.25, 0.5, 0.0, 0.0],
            [-1.0, 2.0, 5.0, 0.25, 0.0, -0.8, 0.0],
            [0.0, 2.0, 5.0, 0.25, 0.0, -0.8, 0.0],
            [1.0, 2.0, 5.0, 0.25, 0.0, -0.8, 0.0],
            [-4.0, -3.0, 7.5, 0.30, 0.7, 0.7, 0.0],
            [-3.5, -3.5, 7.5, 0.30, 0.7, 0.7, 0.0],
            [-3.0, -4.0, 7.5, 0.30, 0.7, 0.7, 0.0],
            [7.0, -1.5, 9.5, 0.20, -1.5, 0.0, 0.0],
            [7.0, 0.0, 9.5, 0.20, -1.5, 0.0, 0.0],
            [7.0, 1.5, 9.5, 0.20, -1.5, 0.0, 0.0],
        ]
    )
    return NLQuad3DScenario(
        "vertical_climb",
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 11.0]]),
        obstacles,
        description="Pure vertical climb through four staggered threats.",
    )


def _vertical_descent() -> NLQuad3DScenario:
    obstacles = np.array(
        [
            [1.0, -1.0, 8.5, 0.25, -1.2, 0.0, 0.0],
            [1.0, 0.0, 8.5, 0.25, -1.2, 0.0, 0.0],
            [1.0, 1.0, 8.5, 0.25, -1.2, 0.0, 0.0],
            [-1.5, 2.0, 6.0, 0.25, 0.0, -0.9, 0.0],
            [0.0, 2.0, 6.0, 0.25, 0.0, -0.9, 0.0],
            [1.5, 2.0, 6.0, 0.25, 0.0, -0.9, 0.0],
            [-4.0, -3.0, 3.5, 0.30, 0.6, 0.6, 0.0],
            [-3.5, -3.5, 3.5, 0.30, 0.6, 0.6, 0.0],
            [-3.0, -4.0, 3.5, 0.30, 0.6, 0.6, 0.0],
            [-5.0, -1.5, 1.5, 0.20, 0.8, 0.0, 0.0],
            [-5.0, 0.0, 1.5, 0.20, 0.8, 0.0, 0.0],
            [-5.0, 1.5, 1.5, 0.20, 0.8, 0.0, 0.0],
        ]
    )
    return NLQuad3DScenario(
        "vertical_descent",
        np.array([[0.0, 0.0, 12.0], [0.0, 0.0, 0.5]]),
        obstacles,
        description="Pure vertical descent through four staggered threats.",
    )


def _attitude_diamond() -> NLQuad3DScenario:
    return NLQuad3DScenario(
        "attitude_diamond",
        np.array(
            [
                [0.0, 0.0, 2.0],
                [8.0, 0.0, 2.0],
                [0.0, 8.0, 2.0],
                [-8.0, 0.0, 2.0],
                [0.0, 0.0, 2.0],
            ]
        ),
        np.zeros((0, 7)),
        description="Obstacle-free inner-loop attitude regression tour.",
    )


def _velocity_tour() -> NLQuad3DScenario:
    return NLQuad3DScenario(
        "velocity_tour",
        np.array(
            [
                [0.0, 0.0, 1.5],
                [15.0, 0.0, 3.0],
                [15.0, 10.0, 5.0],
                [0.0, 10.0, 3.0],
                [0.0, 0.0, 1.5],
            ]
        ),
        np.zeros((0, 7)),
        description="Obstacle-free position/velocity tracking tour.",
    )


def _playground_corridor() -> NLQuad3DScenario:
    """Sparse five-threat scene retained for deterministic regression tests."""

    return NLQuad3DScenario(
        "playground_corridor",
        np.array([[1.0, 10.0, 5.0], [19.0, 10.0, 5.0]]),
        np.array(
            [
                [4.7, 10.0, 5.0, 0.5, -0.15, 0.45, 0.25],
                [5.8, 8.9, 5.7, 0.5, -0.35, 0.35, -0.2],
                [6.4, 11.1, 4.4, 0.5, -0.25, -0.45, 0.2],
                [7.6, 10.4, 6.0, 0.5, -0.45, 0.1, -0.25],
                [8.4, 9.6, 4.0, 0.5, -0.3, 0.35, 0.25],
            ]
        ),
        bounds=WorldBounds((0.0, 0.0, 0.0), (20.0, 20.0, 10.0)),
        description=(
            "Sparse five-threat playground reference retained as a regression "
            "case; use playground_stress for benchmarking."
        ),
        initial_velocity=(1.0, 0.0, 0.0),
    )


def _playground_random(seed: int) -> Callable[[], float]:
    """Return the playground's 32-bit LCG as a deterministic Python closure."""

    state = int(seed) & 0xFFFFFFFF

    def sample() -> float:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state / 0xFFFFFFFF

    return sample


def make_playground_crowded_scenario(
    seed: int = 0,
    *,
    obstacle_count: int = PLAYGROUND_OBSTACLE_COUNT,
) -> NLQuad3DScenario:
    """Build one replayable crowded scene using the playground protocol.

    The five authored threats are always retained.  Remaining spheres are
    sampled with the same LCG, world bounds, radius, speed distribution, and
    start/goal exclusion radius as the interactive playground.
    """

    obstacle_count = int(obstacle_count)
    if obstacle_count < PLAYGROUND_REFERENCE_OBSTACLE_COUNT:
        raise ValueError(
            "playground obstacle_count must retain all five reference threats"
        )
    bounds = WorldBounds((0.0, 0.0, 0.0), (20.0, 20.0, 10.0))
    start = np.array([1.0, 10.0, 5.0])
    goal = np.array([19.0, 10.0, 5.0])
    reference = _playground_corridor().obstacles
    obstacles = [row.tolist() for row in reference]
    random_sample = _playground_random(seed)
    lower = np.asarray(bounds.lower) + PLAYGROUND_OBSTACLE_RADIUS
    upper = np.asarray(bounds.upper) - PLAYGROUND_OBSTACLE_RADIUS
    attempts = 0
    max_attempts = obstacle_count * 20
    while len(obstacles) < obstacle_count and attempts < max_attempts:
        attempts += 1
        position = np.array(
            [
                lower[axis]
                + random_sample() * (upper[axis] - lower[axis])
                for axis in range(3)
            ]
        )
        if (
            np.linalg.norm(position - start)
            < PLAYGROUND_START_GOAL_PROTECTION
            or np.linalg.norm(position - goal)
            < PLAYGROUND_START_GOAL_PROTECTION
        ):
            continue
        angle_1 = random_sample() * 2.0 * np.pi
        angle_2 = random_sample() * 2.0 * np.pi
        speed = (
            PLAYGROUND_OBSTACLE_SPEED_MIN
            + random_sample()
            * (
                PLAYGROUND_OBSTACLE_SPEED_MAX
                - PLAYGROUND_OBSTACLE_SPEED_MIN
            )
        )
        velocity = speed * np.array(
            [
                np.cos(angle_1) * np.cos(angle_2),
                np.sin(angle_2),
                np.sin(angle_1) * np.cos(angle_2),
            ]
        )
        obstacles.append(
            [
                *position,
                PLAYGROUND_OBSTACLE_RADIUS,
                *velocity,
            ]
        )
    if len(obstacles) != obstacle_count:
        raise RuntimeError(
            f"could only place {len(obstacles)} of {obstacle_count} obstacles"
        )
    return NLQuad3DScenario(
        PLAYGROUND_CROWDED_SCENARIO,
        np.array([start, goal]),
        np.asarray(obstacles, dtype=float),
        bounds=bounds,
        description=(
            "Seeded crowded playground scene: five fixed reference threats "
            f"plus {obstacle_count - PLAYGROUND_REFERENCE_OBSTACLE_COUNT} "
            "random moving spheres."
        ),
        initial_velocity=(1.0, 0.0, 0.0),
    )


def make_playground_stress_scenario(
    seed: int = 0,
    *,
    obstacle_count: int = PLAYGROUND_STRESS_OBSTACLE_COUNT,
) -> NLQuad3DScenario:
    """Build the dense, multi-directional benchmark stress protocol.

    Traffic is generated once from ``seed`` before an episode and is identical
    for every method.  Up to twenty-four leading spheres are grouped into four
    six-axis events (+/-x, +/-y, and +/-z).  Every event has a predeclared
    arrival-time and route-position window; small seeded offsets prevent the
    event from becoming a solid geometric wall.  In particular, the +x stream
    approaches from the route-retrace side while the paired -x stream
    approaches from ahead, so simply reversing is not universally safe while
    lateral and vertical escape space remains.  Remaining spheres are sampled
    from a corridor-centered prism with the playground's two-angle velocity
    sampler.  This creates threats from genuinely different escape directions
    without observing a controller, trajectory, policy value, or method
    outcome.

    Obstacles are independent prescribed hazards, as in the playground: they
    reflect at the world box but do not collide with one another.  Consequently
    initially separated spheres may later overlap or pass through each other.

    Unlike the interactive playground sampler, all initial sphere pairs have
    at least ``PLAYGROUND_STRESS_PAIR_CLEARANCE`` metres of surface clearance.
    The full 3.5 m start/goal protection regions are also retained.
    """

    obstacle_count = int(obstacle_count)
    if obstacle_count < PLAYGROUND_REFERENCE_OBSTACLE_COUNT:
        raise ValueError(
            "playground stress obstacle_count must be at least five"
        )

    bounds = WorldBounds((0.0, 0.0, 0.0), (20.0, 20.0, 10.0))
    start = np.array([1.0, 10.0, 5.0])
    goal = np.array([19.0, 10.0, 5.0])
    radius = PLAYGROUND_OBSTACLE_RADIUS
    minimum_center_distance = (
        2.0 * radius + PLAYGROUND_STRESS_PAIR_CLEARANCE
    )
    # A fixed xor separates this generator's sequence from the exact
    # playground_crowded sequence while retaining the playground's auditable
    # 32-bit LCG and full signed/unsigned seed replay behavior.
    random_sample = _playground_random(int(seed) ^ 0x9E3779B9)
    obstacles: list[list[float]] = []

    def uniform(lower: float, upper: float) -> float:
        return lower + random_sample() * (upper - lower)

    def position_is_valid(position: np.ndarray) -> bool:
        if (
            np.linalg.norm(position - start)
            < PLAYGROUND_START_GOAL_PROTECTION
            or np.linalg.norm(position - goal)
            < PLAYGROUND_START_GOAL_PROTECTION
        ):
            return False
        return all(
            np.linalg.norm(position - np.asarray(other[:3]))
            >= minimum_center_distance
            for other in obstacles
        )

    structured_stream_count = min(
        PLAYGROUND_STRESS_STRUCTURED_STREAM_COUNT,
        obstacle_count,
    )
    # Four event windows cover the early/middle traverse.  Each complete event
    # contains one sphere from every signed world axis.  A partial count keeps
    # the same stable lane order for small smoke-test overrides.
    event_time_centers = (2.00, 2.55, 3.10, 3.65)
    event_x_centers = (6.50, 8.10, 9.80, 11.60)
    # Orthogonal offsets make the six prescribed trajectories a compact 3-D
    # cluster rather than six coincident sphere centers.  They are fixed
    # protocol geometry, not generated from any controller behavior.
    lane_offsets = np.asarray(
        (
            (0.00, -0.60, 0.00),
            (0.00, 0.60, 0.00),
            (0.60, 0.00, 0.00),
            (-0.60, 0.00, 0.00),
            (0.35, -0.50, 0.00),
            (-0.35, 0.50, 0.00),
        ),
        dtype=float,
    )
    lower_centers = np.asarray(bounds.lower, dtype=float) + radius
    upper_centers = np.asarray(bounds.upper, dtype=float) - radius
    # Axis coordinates at t=0 are deliberately staggered across events.  This
    # avoids relying on rejection sampling to separate successive streams and
    # makes the arrival-time construction directly auditable: velocity is the
    # signed distance from this coordinate to the event ring divided by the
    # sampled event time.
    initial_axis_centers = np.asarray(
        (
            (4.50, 5.90, 7.30, 8.70),  # +x: route-retrace side
            (8.60, 10.50, 12.50, 14.50),  # -x: route-forward side
            (8.00, 5.80, 3.60, 2.00),  # +y
            (12.50, 14.00, 15.50, 17.00),  # -y
            (2.50, 2.00, 1.50, 1.00),  # +z
            (8.00, 8.50, 9.00, 9.35),  # -z
        ),
        dtype=float,
    )
    placed_streams = 0
    event_index = 0
    while placed_streams < structured_stream_count:
        lane_count = min(
            len(PLAYGROUND_STRESS_STREAM_DIRECTIONS),
            structured_stream_count - placed_streams,
        )
        placed_event = False
        for _ in range(200):
            crossing_time = uniform(
                event_time_centers[event_index] - 0.10,
                event_time_centers[event_index] + 0.10,
            )
            center = np.array(
                [
                    uniform(
                        event_x_centers[event_index] - 0.20,
                        event_x_centers[event_index] + 0.20,
                    ),
                    uniform(9.85, 10.15),
                    uniform(4.85, 5.15),
                ]
            )
            proposed: list[list[float]] = []
            valid_event = True
            for lane in range(lane_count):
                axis = lane // 2
                direction_sign = 1.0 if lane % 2 == 0 else -1.0
                crossing = center + lane_offsets[lane]
                position = crossing.copy()
                position[axis] = uniform(
                    initial_axis_centers[lane, event_index] - 0.04,
                    initial_axis_centers[lane, event_index] + 0.04,
                )
                signed_travel = direction_sign * (
                    crossing[axis] - position[axis]
                )
                speed = signed_travel / crossing_time
                velocity = np.zeros(3)
                velocity[axis] = direction_sign * speed
                if (
                    speed < PLAYGROUND_STRESS_OBSTACLE_SPEED_MIN
                    or speed > PLAYGROUND_STRESS_OBSTACLE_SPEED_MAX
                    or np.any(position < lower_centers)
                    or np.any(position > upper_centers)
                    or not position_is_valid(position)
                    or any(
                        np.linalg.norm(position - np.asarray(other[:3]))
                        < minimum_center_distance
                        for other in proposed
                    )
                ):
                    valid_event = False
                    break
                proposed.append([*position, radius, *velocity])
            if not valid_event:
                continue
            obstacles.extend(proposed)
            placed_streams += len(proposed)
            placed_event = True
            break
        if not placed_event:
            raise RuntimeError(
                f"could not place stress six-axis event {event_index}"
            )
        event_index += 1

    corridor_lower = np.asarray(PLAYGROUND_STRESS_CORRIDOR_LOWER)
    corridor_upper = np.asarray(PLAYGROUND_STRESS_CORRIDOR_UPPER)
    attempts = 0
    max_attempts = max(1, obstacle_count) * 200
    while len(obstacles) < obstacle_count and attempts < max_attempts:
        attempts += 1
        position = np.array(
            [
                uniform(corridor_lower[axis], corridor_upper[axis])
                for axis in range(3)
            ]
        )
        if not position_is_valid(position):
            continue
        angle_1 = uniform(0.0, 2.0 * np.pi)
        angle_2 = uniform(0.0, 2.0 * np.pi)
        speed = uniform(
            PLAYGROUND_STRESS_OBSTACLE_SPEED_MIN,
            PLAYGROUND_STRESS_OBSTACLE_SPEED_MAX,
        )
        velocity = speed * np.array(
            [
                np.cos(angle_1) * np.cos(angle_2),
                np.sin(angle_2),
                np.sin(angle_1) * np.cos(angle_2),
            ]
        )
        obstacles.append([*position, radius, *velocity])

    if len(obstacles) != obstacle_count:
        raise RuntimeError(
            f"could only place {len(obstacles)} of {obstacle_count} stress "
            "obstacles"
        )
    return NLQuad3DScenario(
        PLAYGROUND_STRESS_SCENARIO,
        np.array([start, goal]),
        np.asarray(obstacles, dtype=float),
        bounds=bounds,
        description=(
            "Seeded playground stress scene: "
            f"{structured_stream_count} balanced six-axis stream threats "
            f"plus {obstacle_count - structured_stream_count} corridor-biased moving "
            "spheres with pairwise-safe initial placement."
        ),
        initial_velocity=(1.0, 0.0, 0.0),
    )


def _playground_crowded() -> NLQuad3DScenario:
    return make_playground_crowded_scenario(0)


def _playground_stress() -> NLQuad3DScenario:
    return make_playground_stress_scenario(0)


_SCENARIOS: dict[str, Callable[[], NLQuad3DScenario]] = {
    "head_on": _head_on,
    "cross_traffic": _cross_traffic,
    "vertical_drop": _vertical_drop,
    "moving_wall": _moving_wall,
    "asteroid_field": _asteroid_field,
    "collapsing_sphere": _collapsing_sphere,
    "climb_and_dodge": _climb_and_dodge,
    "descend_through": _descend_through,
    "vertical_climb": _vertical_climb,
    "vertical_descent": _vertical_descent,
    "attitude_diamond": _attitude_diamond,
    "velocity_tour": _velocity_tour,
    "t1": _attitude_diamond,
    "t2": _velocity_tour,
    "playground_corridor": _playground_corridor,
    PLAYGROUND_CROWDED_SCENARIO: _playground_crowded,
    PLAYGROUND_STRESS_SCENARIO: _playground_stress,
}


def scenario_names() -> tuple[str, ...]:
    return tuple(_SCENARIOS)


def get_scenario(name: str) -> NLQuad3DScenario:
    """Return a fresh copy of a registered deterministic scenario."""

    try:
        source = _SCENARIOS[name]()
    except KeyError as exc:
        available = ", ".join(scenario_names())
        raise KeyError(f"unknown nl_quad3d scenario {name!r}; choose from {available}") from exc
    return NLQuad3DScenario(
        name=name,
        waypoints=source.waypoints,
        obstacles=source.obstacles,
        bounds=source.bounds,
        description=source.description,
        reach_threshold=source.reach_threshold,
        default_steps=source.default_steps,
        initial_velocity=source.initial_velocity,
    )
