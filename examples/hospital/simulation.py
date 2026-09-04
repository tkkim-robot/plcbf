"""Headless hospital simulation and deterministic blocked-hall scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .config import DEFAULT_CONFIG, HospitalConfig
from .controller import ControllerResult, HospitalController
from .dynamics import step_double_integrator
from .environment import HospitalEnvironment, build_hospital_environment
from .obstacles import (
    DynamicObstacle,
    Human,
    Stretcher,
    obstacle_clearance,
    stretcher_route,
)
from .planner import HospitalGridPlanner


STRICT_REFUGE_INITIAL_STATE = np.array([57.5, 47.5, 0.0, 0.0])
STRICT_REFUGE_CONVOY_SPEED = -4.2
STRICT_REFUGE_POSITIONS = {
    2: (76.5, 82.75),
    3: (76.5, 82.75, 89.0),
}
STRICT_WEST_JUNCTION_X = 28.0


def validate_strict_refuge_protocol(
    config: HospitalConfig,
) -> HospitalConfig:
    """Compatibility validator for the geometry-only blocked-hall protocol.

    The benchmark intentionally has no minimum hold timer or guarded-release
    state machine.  Safety is assessed from collision geometry and clearance.
    """

    return config


def strict_refuge_scenario_metadata(
    config: HospitalConfig = DEFAULT_CONFIG,
) -> dict[str, bool | float | str]:
    """Describe the geometry-grounded blocked-hall benchmark contract."""

    validate_strict_refuge_protocol(config)
    return {
        "convoy_motion": "one_way_left_exit",
        "convoy_reflects": False,
        "convoy_speed_mps": abs(STRICT_REFUGE_CONVOY_SPEED),
        "minimum_seeded_convoy_speed_mps": abs(
            STRICT_REFUGE_CONVOY_SPEED
        ),
        "robot_max_retreat_speed_mps": config.robot.v_max,
        "robot_initial_x": float(STRICT_REFUGE_INITIAL_STATE[0]),
        "west_junction_boundary_x": STRICT_WEST_JUNCTION_X,
        "designated_refuge_room": "Nurse",
        "external_refuge_state_machine": False,
        "necessity_rationale": (
            "the full-width one-way convoy overtakes a maximum-speed corridor "
            "retreat before the west vertical junction; a nearby room is the "
            "only collision-free lateral refuge"
        ),
    }


@dataclass(frozen=True)
class TraceRecord:
    time: float
    state: np.ndarray
    selected_policy: str
    inside_refuge: bool
    min_clearance: float
    collision: bool
    reached_goal: bool


@dataclass(frozen=True)
class ClearanceWitness:
    """Geometry and synchronized sample that attained a clearance minimum.

    ``elapsed_s`` and ``sample_fraction`` are relative to the swept transition.
    Benchmark code combines them with the plant-step start time and index.  The
    witness is diagnostic only and is never consumed by a controller.
    """

    value: float
    source_kind: Literal["static", "human", "stretcher"]
    obstacle_identifier: str | None
    elapsed_s: float
    sample_index: int
    sample_fraction: float
    robot_position: tuple[float, float]


@dataclass(frozen=True)
class SweptTransitionSafety:
    """Geometry metrics over one synchronized robot/obstacle transition."""

    collision: bool
    minimum_clearance: float
    minimum_safety_clearance: float
    minimum_clearance_witness: ClearanceWitness | None = None
    minimum_safety_clearance_witness: ClearanceWitness | None = None


def evaluate_swept_transition(
    environment: HospitalEnvironment,
    obstacles: Sequence[DynamicObstacle],
    start_state: Sequence[float],
    end_state: Sequence[float],
    dt: float,
    config: HospitalConfig,
    *,
    dynamic_substeps: int = 8,
) -> SweptTransitionSafety:
    """Check a plant step without missing between-sample collisions.

    Static geometry is checked along the complete robot segment. Dynamic
    geometry is checked at synchronized robot and predicted-obstacle samples,
    including both endpoints. Obstacles must still be at their start-of-step
    state when this function is called.
    """

    if dynamic_substeps < 2:
        raise ValueError("dynamic_substeps must be at least two")
    elapsed = float(dt)
    if elapsed < 0.0:
        raise ValueError("dt must be nonnegative")
    start = np.asarray(start_state, dtype=float)
    end = np.asarray(end_state, dtype=float)
    if start.shape != (4,) or end.shape != (4,):
        raise ValueError("start_state and end_state must have shape (4,)")

    start_position = start[:2]
    end_position = end[:2]
    static_free = environment.segment_is_free(
        start_position,
        end_position,
        config.robot.radius,
    )
    minimum_clearance_witness: ClearanceWitness | None = None
    minimum_safety_clearance_witness: ClearanceWitness | None = None
    sampled_positions: list[tuple[float, np.ndarray]] = []

    def witness(
        value: float,
        *,
        source_kind: Literal["static", "human", "stretcher"],
        obstacle_identifier: str | None,
        sample_index: int,
        sample_fraction: float,
        position: np.ndarray,
    ) -> ClearanceWitness:
        return ClearanceWitness(
            value=float(value),
            source_kind=source_kind,
            obstacle_identifier=obstacle_identifier,
            elapsed_s=float(sample_fraction * elapsed),
            sample_index=int(sample_index),
            sample_fraction=float(sample_fraction),
            robot_position=(float(position[0]), float(position[1])),
        )

    def lower(
        current: ClearanceWitness | None,
        value: float,
        *,
        source_kind: Literal["static", "human", "stretcher"],
        obstacle_identifier: str | None,
        sample_index: int,
        sample_fraction: float,
        position: np.ndarray,
    ) -> ClearanceWitness:
        if current is None or value < current.value:
            return witness(
                value,
                source_kind=source_kind,
                obstacle_identifier=obstacle_identifier,
                sample_index=sample_index,
                sample_fraction=sample_fraction,
                position=position,
            )
        return current

    for index in range(dynamic_substeps + 1):
        alpha = index / dynamic_substeps
        position = start_position + alpha * (end_position - start_position)
        sampled_positions.append((alpha, position))
        minimum_clearance_witness = lower(
            minimum_clearance_witness,
            environment.static_clearance(position, config.robot.radius),
            source_kind="static",
            obstacle_identifier=None,
            sample_index=index,
            sample_fraction=alpha,
            position=position,
        )
        minimum_safety_clearance_witness = lower(
            minimum_safety_clearance_witness,
            environment.static_clearance(
                position,
                config.robot.radius + config.safety.static_margin,
            ),
            source_kind="static",
            obstacle_identifier=None,
            sample_index=index,
            sample_fraction=alpha,
            position=position,
        )
        for obstacle in obstacles:
            predicted = obstacle.predicted(alpha * elapsed, environment)
            if isinstance(predicted, Human):
                source_kind: Literal["human", "stretcher"] = "human"
            else:
                # Match ``obstacle_clearance``'s existing protocol behavior:
                # every non-Human dynamic obstacle uses stretcher geometry and
                # margin semantics. This keeps structural test doubles and
                # future stretcher-compatible implementations supported.
                source_kind = "stretcher"
            minimum_clearance_witness = lower(
                minimum_clearance_witness,
                obstacle_clearance(
                    predicted,
                    position,
                    config.robot.radius,
                    0.0,
                    0.0,
                ),
                source_kind=source_kind,
                obstacle_identifier=predicted.identifier,
                sample_index=index,
                sample_fraction=alpha,
                position=position,
            )
            minimum_safety_clearance_witness = lower(
                minimum_safety_clearance_witness,
                obstacle_clearance(
                    predicted,
                    position,
                    config.robot.radius + config.safety.safety_margin,
                    config.safety.human_margin,
                    config.safety.stretcher_margin,
                ),
                source_kind=source_kind,
                obstacle_identifier=predicted.identifier,
                sample_index=index,
                sample_fraction=alpha,
                position=position,
            )
    if (
        minimum_clearance_witness is None
        or minimum_safety_clearance_witness is None
    ):
        raise RuntimeError("swept transition produced no clearance samples")
    if not static_free and minimum_clearance_witness.value > 0.0:
        collision_sample = next(
            (
                (index, alpha, position)
                for index, (alpha, position) in enumerate(sampled_positions)
                if environment.is_collision(position, config.robot.radius)
            ),
            None,
        )
        if collision_sample is None:
            # ``segment_is_free`` and the synchronized sweep both include the
            # endpoints and midpoint.  Retain a deterministic location even if
            # their conservative perimeter samplers disagree at roundoff.
            index = minimum_clearance_witness.sample_index
            alpha, position = sampled_positions[index]
        else:
            index, alpha, position = collision_sample
        minimum_clearance_witness = witness(
            0.0,
            source_kind="static",
            obstacle_identifier=None,
            sample_index=index,
            sample_fraction=alpha,
            position=position,
        )
    if not static_free and minimum_safety_clearance_witness.value > 0.0:
        minimum_safety_clearance_witness = witness(
            0.0,
            source_kind="static",
            obstacle_identifier=None,
            sample_index=minimum_clearance_witness.sample_index,
            sample_fraction=minimum_clearance_witness.sample_fraction,
            position=np.asarray(minimum_clearance_witness.robot_position),
        )
    return SweptTransitionSafety(
        collision=bool(not static_free or minimum_clearance_witness.value <= 0.0),
        minimum_clearance=minimum_clearance_witness.value,
        minimum_safety_clearance=minimum_safety_clearance_witness.value,
        minimum_clearance_witness=minimum_clearance_witness,
        minimum_safety_clearance_witness=minimum_safety_clearance_witness,
    )


class HospitalSimulation:
    """Deterministic simulator with an interface usable by benchmark adapters."""

    def __init__(
        self,
        initial_state: Sequence[float],
        goal: Sequence[float],
        obstacles: Sequence[DynamicObstacle] = (),
        config: HospitalConfig = DEFAULT_CONFIG,
        environment: HospitalEnvironment | None = None,
        planner: HospitalGridPlanner | None = None,
    ) -> None:
        self.config = config
        self.environment = environment or build_hospital_environment()
        self.planner = planner or HospitalGridPlanner(
            self.environment,
            resolution=config.planner.resolution,
            clearance=config.robot.radius + config.planner.clearance_buffer,
            preferred_clearance=config.planner.preferred_clearance,
            clearance_weight=config.planner.clearance_weight,
        )
        self.state = np.asarray(initial_state, dtype=float).copy()
        self.goal = np.asarray(goal, dtype=float).copy()
        self.obstacles = list(obstacles)
        self.controller = HospitalController(
            self.environment,
            self.planner,
            config,
            self.state,
            self.goal,
        )
        self.time = 0.0
        self.collision = False
        self.reached_goal = False
        self.last_controller: ControllerResult | None = None
        self.trace: list[TraceRecord] = []
        self.benchmark_scenario_metrics: dict[
            str, bool | int | float | str
        ] = {}

    def minimum_clearance(
        self, state: Sequence[float] | None = None, safety: bool = False
    ) -> float:
        value = self.state if state is None else np.asarray(state, dtype=float)
        robot_radius = self.config.robot.radius
        if safety:
            robot_radius += self.config.safety.safety_margin
        dynamic = [
            obstacle_clearance(
                obstacle,
                value[:2],
                robot_radius,
                self.config.safety.human_margin if safety else 0.0,
                self.config.safety.stretcher_margin if safety else 0.0,
            )
            for obstacle in self.obstacles
        ]
        static = self.environment.static_clearance(
            value[:2],
            self.config.robot.radius
            + (self.config.safety.static_margin if safety else 0.0),
        )
        return min([static, *dynamic])

    def step(self) -> TraceRecord:
        if self.collision or self.reached_goal:
            return self.trace[-1]

        dt = self.config.dt
        result = self.controller.compute(self.state, self.obstacles, self.time)
        previous = self.state.copy()
        following = step_double_integrator(
            self.state, result.control, dt, self.config.robot
        )
        transition = evaluate_swept_transition(
            self.environment,
            self.obstacles,
            previous,
            following,
            dt,
            self.config,
        )
        for obstacle in self.obstacles:
            obstacle.advance(dt, self.environment)
        self.state = following
        self.time += dt
        self.last_controller = result
        self.collision = transition.collision
        self.reached_goal = (
            np.linalg.norm(self.state[:2] - self.goal) <= 1.35
        )
        record = TraceRecord(
            time=self.time,
            state=self.state.copy(),
            selected_policy=result.selected_policy,
            inside_refuge=bool(
                self.environment.room_containing(self.state[:2]) is not None
            ),
            min_clearance=transition.minimum_clearance,
            collision=self.collision,
            reached_goal=self.reached_goal,
        )
        self.trace.append(record)
        return record

    def run(self, steps: int) -> list[TraceRecord]:
        for _ in range(int(steps)):
            self.step()
            if self.collision or self.reached_goal:
                break
        return self.trace

    def snapshot(self) -> dict:
        return {
            "time": self.time,
            "state": self.state.copy(),
            "goal": self.goal.copy(),
            "collision": self.collision,
            "reached_goal": self.reached_goal,
            "inside_refuge": bool(
                self.environment.room_containing(self.state[:2]) is not None
            ),
            "selected_policy": (
                self.last_controller.selected_policy
                if self.last_controller is not None
                else "nominal"
            ),
            "obstacles": tuple(self.obstacles),
            "active_path": (),
        }


def build_blocked_main_hall_scenario(
    stretcher_count: int = 3,
    config: HospitalConfig = DEFAULT_CONFIG,
) -> HospitalSimulation:
    """Create the strict refuge benchmark with two or three full-width stretchers.

    The one-way emergency convoy approaches from the goal side, completely
    occupies the main-hall cross-section, and moves faster than the robot's
    bounded retreat speed. From the initial position west of the center
    junction, even maximum reverse is swept before reaching the west vertical
    hall. The nearby Nurse room is therefore the intended lateral refuge.
    After the convoy exits the modeled hall to the left, the nominal policy
    becomes safe again and continuous QP selection can return to it.
    """

    validate_strict_refuge_protocol(config)
    if stretcher_count not in (2, 3):
        raise ValueError("stretcher_count must be two or three")
    environment = build_hospital_environment()
    main = next(
        corridor
        for corridor in environment.corridor_rects
        if corridor.name == "Main corridor"
    )
    positions = STRICT_REFUGE_POSITIONS[stretcher_count]
    obstacles: list[DynamicObstacle] = []
    for index, position in enumerate(positions):
        axis, coordinate, lateral, route_min, route_max = stretcher_route(
            main,
            coordinate=position,
            lateral=main.center[1],
            length=5.4,
            width=7.1,
        )
        obstacles.append(
            Stretcher(
                identifier=f"blocking-stretcher-{index}",
                coordinate=coordinate,
                lateral=lateral,
                speed=STRICT_REFUGE_CONVOY_SPEED,
                axis=axis,
                route_min=route_min,
                route_max=route_max,
                length=5.4,
                width=7.1,
                reflect_at_route_bounds=False,
            )
        )
    return HospitalSimulation(
        initial_state=STRICT_REFUGE_INITIAL_STATE,
        goal=np.array([130.0, 47.5]),
        obstacles=obstacles,
        config=config,
        environment=environment,
    )
