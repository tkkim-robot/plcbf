"""Headless eight-method benchmark for crowded hospital refuge cases.

All methods integrate the same double-integrator dynamics, exact obstacle
geometry, and policy-certificate oracle.  The full room/directional/stop
library is continuously re-evaluated.  There is no latched room executor,
timed hold, release guard, or phase-specific nominal controller.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
import time
from typing import Iterable, Sequence

import numpy as np

from plcbf.baselines import (
    BENCHMARK_METHODS,
    BaselineDecision,
    BenchmarkMethod,
)
from plcbf.benchmarking import (
    BenchmarkOutcome,
    BenchmarkResult,
    aggregate_results,
    write_benchmark_reports,
)

from .config import DEFAULT_CONFIG, HospitalConfig, load_hospital_config
from .baselines import HospitalBaselineSuite
from .controller import HospitalController, sensed_obstacles
from .dynamics import step_double_integrator, waypoint_control
from .environment import Room
from .obstacles import Human, Stretcher
from .policies import HospitalPolicy
from .scenario_generation import (
    DEFAULT_HUMAN_COUNT,
    DEFAULT_ORDINARY_STRETCHER_COUNT,
    generate_hospital_crowd,
)
from .reporting import (
    hospital_story_id,
    write_hospital_benchmark_markdown,
)
from .provenance import hospital_benchmark_source_manifest
from .scenarios import (
    DEFAULT_HOSPITAL_TRAFFIC_SEEDS,
    HOSPITAL_STORIES,
    HOSPITAL_STORY_IDS,
    PUBLICATION_HUMAN_COUNT,
    PUBLICATION_SENSING_RANGE_M,
    HospitalStoryScenario,
    build_hospital_story_scenario,
    get_hospital_story,
    hospital_story_protocol_metadata,
)
from .simulation import (
    ClearanceWitness,
    HospitalSimulation,
    STRICT_WEST_JUNCTION_X,
    build_blocked_main_hall_scenario,
    evaluate_swept_transition,
    strict_refuge_scenario_metadata,
    validate_strict_refuge_protocol,
)


STRICT_HOSPITAL_CASES: dict[str, int] = {
    "blocked_2_stretchers": 2,
    "blocked_3_stretchers": 3,
}
HOSPITAL_BENCHMARK_STORIES = HOSPITAL_STORY_IDS
PUBLICATION_MAX_SENSED_OBSTACLES = PUBLICATION_HUMAN_COUNT + max(
    story.blocker_count for story in HOSPITAL_STORIES
)
RANDOMIZED_EGO_X_RANGE_M = (57.5, 58.0)
RANDOMIZED_EGO_Y_RANGE_M = (46.9, 48.1)
RANDOMIZED_EGO_VX_RANGE_MPS = (0.0, 0.2)
RANDOMIZED_EGO_VY_RANGE_MPS = (-0.1, 0.1)
RANDOMIZED_STRETCHER_SHIFT_RANGE_M = (-0.7, 0.0)
RANDOMIZED_STRETCHER_SPEED_FACTOR_RANGE = (1.0, 1.08)

# A 66 s horizon was only barely longer than an unobstructed story traversal
# and converted slow-but-moving trials into artificial timeouts.  The
# publication protocol now gives every method three minutes of simulated time.
DEFAULT_HOSPITAL_SIMULATION_TIME_S = 180.0

# Deadlock is an observation-only post-convoy stagnation diagnostic.  It is
# deliberately conservative and, unlike a controller state machine, never
# changes a policy, control, obstacle, or terminal decision.
DEADLOCK_WINDOW_S = 30.0
DEADLOCK_POST_CONVOY_GRACE_S = 10.0
DEADLOCK_LEGACY_ELIGIBLE_AFTER_S = 45.0
DEADLOCK_MAX_PATH_LENGTH_M = 0.50
DEADLOCK_MAX_GOAL_PROGRESS_M = 0.25
DEADLOCK_MAX_SPEED_MPS = 0.10


def default_hospital_benchmark_steps(
    config: HospitalConfig = DEFAULT_CONFIG,
) -> int:
    """Return the plant-step count for the publication time horizon."""

    return int(math.ceil(DEFAULT_HOSPITAL_SIMULATION_TIME_S / config.dt))


def hospital_randomization_protocol_metadata() -> dict[str, object]:
    """Return the frozen five-story, humans-only sampling protocol."""

    return hospital_story_protocol_metadata()


def publication_benchmark_config(
    config: HospitalConfig = DEFAULT_CONFIG,
) -> HospitalConfig:
    """Apply the fixed, method-independent publication perception envelope.

    The 24 m range gives the 3 m/s full-width convoy enough physical lead for
    at least one room backup to remain viable when first observed.  Line-of-
    sight filtering remains identical to the playground.  Capacity is raised
    to the complete fixed world size, and there is no obstacle-ID priority
    rule.  Every method receives the same resulting snapshot.
    """

    if (
        config.safety.max_obstacles >= PUBLICATION_MAX_SENSED_OBSTACLES
        and config.robot.sensing_range >= PUBLICATION_SENSING_RANGE_M
    ):
        return config
    return replace(
        config,
        robot=replace(
            config.robot,
            sensing_range=max(
                config.robot.sensing_range,
                PUBLICATION_SENSING_RANGE_M,
            ),
        ),
        safety=replace(
            config.safety,
            max_obstacles=max(
                config.safety.max_obstacles,
                PUBLICATION_MAX_SENSED_OBSTACLES,
            ),
        ),
    )


def compact_benchmark_config(
    config: HospitalConfig = DEFAULT_CONFIG,
) -> HospitalConfig:
    """Retain all policy types while reducing redundant directional branches."""

    return replace(
        config,
        policies=replace(
            config.policies,
            num_angle_policies=4,
            room_policy_count=1,
        ),
    )


def _case_stretcher_count(case_id: str) -> int:
    if str(case_id) in HOSPITAL_BENCHMARK_STORIES:
        return get_hospital_story(str(case_id)).blocker_count
    try:
        return STRICT_HOSPITAL_CASES[str(case_id)]
    except KeyError as exc:
        choices = ", ".join(STRICT_HOSPITAL_CASES)
        raise ValueError(
            f"unknown hospital benchmark case {case_id!r}; choose {choices}"
        ) from exc


def build_benchmark_scenario(
    case_id: str,
    *,
    seed: int = 0,
    config: HospitalConfig = DEFAULT_CONFIG,
) -> HospitalSimulation:
    """Build one publication story or an explicit legacy strict case."""

    if str(case_id) in HOSPITAL_BENCHMARK_STORIES:
        scenario = build_hospital_story_scenario(
            str(case_id),
            traffic_seed=int(seed),
            # World generation is intentionally independent of controller
            # tuning.  Runtime parameters are installed only on the clone.
            config=DEFAULT_CONFIG,
        )
        return scenario.to_simulation(publication_benchmark_config(config))

    validate_strict_refuge_protocol(config)
    count = _case_stretcher_count(case_id)
    simulation = build_blocked_main_hall_scenario(count, config)
    if int(seed) != 0:
        generator = np.random.default_rng(int(seed))
        simulation.state = np.array(
            [
                generator.uniform(*RANDOMIZED_EGO_X_RANGE_M),
                generator.uniform(*RANDOMIZED_EGO_Y_RANGE_M),
                generator.uniform(*RANDOMIZED_EGO_VX_RANGE_MPS),
                generator.uniform(*RANDOMIZED_EGO_VY_RANGE_MPS),
            ],
            dtype=float,
        )
        for obstacle in simulation.obstacles:
            if not isinstance(obstacle, Stretcher):
                continue
            obstacle.coordinate = float(
                np.clip(
                    # Seeded cases may make the sweep stricter, never weaken
                    # the room-necessity contract by moving it away.
                    obstacle.coordinate
                    + generator.uniform(*RANDOMIZED_STRETCHER_SHIFT_RANGE_M),
                    obstacle.route_min,
                    obstacle.route_max,
                )
            )
            # The nominal velocity is negative, so factors >= 1 increase the
            # closing-speed magnitude.
            obstacle.speed *= float(
                generator.uniform(*RANDOMIZED_STRETCHER_SPEED_FACTOR_RANGE)
            )
        # The controller plans its navigation route during construction, so a
        # sampled ego state must be installed through a fresh controller rather
        # than mutating only the simulator's plant state.
        simulation.controller = HospitalController(
            simulation.environment,
            simulation.planner,
            config,
            simulation.state,
            simulation.goal,
        )

    crowd = generate_hospital_crowd(
        simulation.environment,
        config,
        seed=int(seed),
        ego_position=simulation.state[:2],
        goal_position=simulation.goal,
        existing_obstacles=simulation.obstacles,
    )
    simulation.obstacles.extend(crowd.obstacles)

    main = next(
        corridor
        for corridor in simulation.environment.corridor_rects
        if corridor.name == "Main corridor"
    )
    corridor_cross_section = min(main.width, main.height)
    blockers = [
        obstacle
        for obstacle in simulation.obstacles
        if isinstance(obstacle, Stretcher)
        and obstacle.identifier.startswith("blocking-stretcher-")
    ]
    stretchers = [
        obstacle
        for obstacle in simulation.obstacles
        if isinstance(obstacle, Stretcher)
    ]
    humans = [
        obstacle
        for obstacle in simulation.obstacles
        if isinstance(obstacle, Human)
    ]
    if len(blockers) != count:
        raise RuntimeError("strict hospital case has the wrong stretcher count")
    for stretcher in blockers:
        effective_cross_section = stretcher.cross_section_width + 2.0 * (
            config.robot.radius + config.safety.stretcher_margin
        )
        if effective_cross_section < corridor_cross_section:
            raise RuntimeError(
                "strict hospital benchmark requires full-width stretchers"
            )
        if stretcher.reflect_at_route_bounds:
            raise RuntimeError(
                "strict hospital convoy must exit instead of reflecting"
            )
        if stretcher.speed >= -config.robot.v_max:
            raise RuntimeError(
                "strict hospital convoy must overtake bounded retreat"
            )

    initial_safety_clearance = simulation.minimum_clearance(safety=True)
    if initial_safety_clearance <= 0.0:
        raise RuntimeError(
            "strict hospital benchmark must begin inside the operational safe set"
        )
    reverse_swept, reverse_violation_x = _maximum_reverse_is_swept(
        simulation
    )
    if not reverse_swept:
        raise RuntimeError(
            "strict hospital benchmark requires maximum retreat to be swept "
            "before the west junction"
        )

    speeds = [abs(stretcher.speed) for stretcher in blockers]
    coordinates = [stretcher.coordinate for stretcher in blockers]
    sampled_metrics: dict[str, bool | int | float | str] = {
        "scenario_seed": int(seed),
        "scenario_randomized": int(seed) != 0,
        "robot_initial_x": float(simulation.state[0]),
        "robot_initial_y": float(simulation.state[1]),
        "robot_initial_vx": float(simulation.state[2]),
        "robot_initial_vy": float(simulation.state[3]),
        "initial_safety_clearance": float(initial_safety_clearance),
        "convoy_initial_coordinate_min": float(min(coordinates)),
        "convoy_initial_coordinate_max": float(max(coordinates)),
        "convoy_speed_min_mps": float(min(speeds)),
        "convoy_speed_max_mps": float(max(speeds)),
        "full_width_blockade": True,
        "maximum_reverse_swept_before_west_junction": True,
        "maximum_reverse_violation_x": float(reverse_violation_x),
        "human_count": len(humans),
        "ordinary_stretcher_count": len(stretchers) - len(blockers),
        "blocking_stretcher_count": len(blockers),
        "total_stretcher_count": len(stretchers),
        "dynamic_obstacle_count": len(simulation.obstacles),
        "controller_max_sensed_obstacles": config.safety.max_obstacles,
        "crowd_generation_attempts": crowd.metadata.placement_attempts,
        "crowd_minimum_pairwise_clearance": (
            crowd.metadata.minimum_pairwise_clearance
        ),
        "crowd_minimum_protected_clearance": (
            crowd.metadata.minimum_protected_clearance
        ),
    }
    for index, stretcher in enumerate(blockers):
        sampled_metrics[f"stretcher_{index}_initial_coordinate"] = float(
            stretcher.coordinate
        )
        sampled_metrics[f"stretcher_{index}_speed_mps"] = float(
            abs(stretcher.speed)
        )
    simulation.benchmark_scenario_metrics = sampled_metrics
    return simulation


def _maximum_reverse_is_swept(
    simulation: HospitalSimulation,
    *,
    steps: int = 300,
) -> tuple[bool, float]:
    """Check the strict contract on copies without changing the trial."""

    state = simulation.state.copy()
    obstacles = [
        replace(obstacle)
        for obstacle in simulation.obstacles
        if isinstance(obstacle, Stretcher)
        and obstacle.identifier.startswith("blocking-stretcher-")
    ]
    for _ in range(int(steps)):
        following = step_double_integrator(
            state,
            [-simulation.config.robot.a_max, 0.0],
            simulation.config.dt,
            simulation.config.robot,
        )
        transition = evaluate_swept_transition(
            simulation.environment,
            obstacles,
            state,
            following,
            simulation.config.dt,
            simulation.config,
        )
        for obstacle in obstacles:
            obstacle.advance(
                simulation.config.dt,
                simulation.environment,
            )
        state = following
        if transition.minimum_safety_clearance < 0.0:
            return bool(state[0] > STRICT_WEST_JUNCTION_X), float(state[0])
        if state[0] <= STRICT_WEST_JUNCTION_X:
            break
    return False, float(state[0])


@dataclass
class _DeadlockMonitor:
    """Observe persistent post-convoy stagnation without controlling the plant.

    A detected episode is diagnostic only.  The benchmark deliberately keeps
    integrating after detection so later recovery, goal arrival, or physical
    collision remains observable.
    """

    eligible_after_s: float
    window_s: float = DEADLOCK_WINDOW_S
    max_path_length_m: float = DEADLOCK_MAX_PATH_LENGTH_M
    max_goal_progress_m: float = DEADLOCK_MAX_GOAL_PROGRESS_M
    max_speed_mps: float = DEADLOCK_MAX_SPEED_MPS
    samples: deque[tuple[float, np.ndarray, float, float, float]] = field(
        default_factory=deque
    )
    cumulative_path_length_m: float = 0.0
    previous_position: np.ndarray | None = None
    currently_deadlocked: bool = False
    detected: bool = False
    first_detected_at_s: float | None = None
    episode_count: int = 0
    resolved_episode_count: int = 0
    last_window_path_length_m: float = 0.0
    last_window_goal_progress_m: float = 0.0
    last_window_max_speed_mps: float = 0.0

    def update(
        self,
        time_s: float,
        state: Sequence[float],
        goal: Sequence[float],
    ) -> bool:
        """Update the rolling observation and return current stagnation."""

        value = np.asarray(state, dtype=float).reshape(4)
        position = value[:2].copy()
        if self.previous_position is not None:
            self.cumulative_path_length_m += float(
                np.linalg.norm(position - self.previous_position)
            )
        self.previous_position = position

        time_value = float(time_s)
        if time_value < self.eligible_after_s:
            self.samples.clear()
            self.currently_deadlocked = False
            return False

        goal_distance = float(
            np.linalg.norm(position - np.asarray(goal, dtype=float)[:2])
        )
        speed = float(np.linalg.norm(value[2:4]))
        self.samples.append(
            (
                time_value,
                position,
                goal_distance,
                speed,
                self.cumulative_path_length_m,
            )
        )
        cutoff = time_value - self.window_s
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()

        first = self.samples[0]
        duration = time_value - first[0]
        path_length = self.cumulative_path_length_m - first[4]
        goal_progress = first[2] - goal_distance
        maximum_speed = max(sample[3] for sample in self.samples)
        self.last_window_path_length_m = float(path_length)
        self.last_window_goal_progress_m = float(goal_progress)
        self.last_window_max_speed_mps = float(maximum_speed)

        deadlocked = bool(
            duration >= self.window_s - 1e-9
            and path_length <= self.max_path_length_m
            and goal_progress <= self.max_goal_progress_m
            and maximum_speed <= self.max_speed_mps
            and goal_distance > 1.35
        )
        if deadlocked and not self.currently_deadlocked:
            self.detected = True
            self.episode_count += 1
            if self.first_detected_at_s is None:
                self.first_detected_at_s = time_value
        elif not deadlocked and self.currently_deadlocked:
            self.resolved_episode_count += 1
        self.currently_deadlocked = deadlocked
        return deadlocked


def _deadlock_eligible_after_s(
    scenario_metrics: dict[str, bool | int | float | str],
) -> float:
    """Start monitoring only after the immutable convoy has fully cleared."""

    clear_time = scenario_metrics.get("convoy_clear_time_s")
    if isinstance(clear_time, (int, float)) and np.isfinite(clear_time):
        return float(clear_time) + DEADLOCK_POST_CONVOY_GRACE_S
    return DEADLOCK_LEGACY_ELIGIBLE_AFTER_S


@dataclass
class _HospitalNarrativeTrace:
    """Observation-only room/blockage story trace.

    Publication stories begin inside a start room, so that initial occupancy
    is deliberately not counted as refuge entry.  No field in this monitor is
    ever read by a controller or solver.
    """

    initial_room: Room | None
    blockage_started_at_s: float | None
    blockage_cleared_at_s: float | None
    previous_room: Room | None
    start_room_left: bool = False
    start_room_left_at_s: float | None = None
    post_departure_room: Room | None = None
    post_departure_room_entered_at_s: float | None = None
    post_departure_room_left_at_s: float | None = None
    room_occupied_during_blockage: bool = False
    minimum_blockage_safety_clearance: float = float("inf")
    blockage_window_observed: bool = False
    room_exited_after_clear: bool = False
    goal_reached_after_clear: bool = False

    @classmethod
    def from_simulation(
        cls,
        simulation: HospitalSimulation,
    ) -> "_HospitalNarrativeTrace":
        metadata = simulation.benchmark_scenario_metrics

        def optional_time(name: str) -> float | None:
            value = metadata.get(name)
            if value is None or isinstance(value, bool):
                return None
            parsed = float(value)
            return parsed if np.isfinite(parsed) else None

        initial_room = simulation.environment.room_containing(
            simulation.state[:2]
        )
        return cls(
            initial_room=initial_room,
            blockage_started_at_s=optional_time("blockage_started_at_s"),
            blockage_cleared_at_s=optional_time("blockage_cleared_at_s"),
            previous_room=initial_room,
            start_room_left=initial_room is None,
        )

    def update(
        self,
        simulation: HospitalSimulation,
        *,
        transition_started_at_s: float,
        transition_safety_clearance: float,
    ) -> None:
        current_room = simulation.environment.room_containing(
            simulation.state[:2]
        )
        if (
            not self.start_room_left
            and current_room is not self.initial_room
        ):
            self.start_room_left = True
            self.start_room_left_at_s = simulation.time
        if (
            self.start_room_left
            and self.post_departure_room is None
            and current_room is not None
        ):
            # Re-entering the start room is a valid room refuge too; only its
            # initial occupancy is excluded from this event.
            self.post_departure_room = current_room
            self.post_departure_room_entered_at_s = simulation.time
        if (
            self.post_departure_room is not None
            and self.previous_room is self.post_departure_room
            and current_room is not self.post_departure_room
            and self.post_departure_room_left_at_s is None
        ):
            self.post_departure_room_left_at_s = simulation.time
            if (
                self.blockage_cleared_at_s is not None
                and simulation.time >= self.blockage_cleared_at_s
            ):
                self.room_exited_after_clear = True

        start = self.blockage_started_at_s
        clear = self.blockage_cleared_at_s
        if (
            start is not None
            and clear is not None
            and transition_started_at_s < clear
            and simulation.time >= start
        ):
            self.minimum_blockage_safety_clearance = min(
                self.minimum_blockage_safety_clearance,
                float(transition_safety_clearance),
            )
            if self.start_room_left and current_room is not None:
                self.room_occupied_during_blockage = True
        if clear is not None and simulation.time >= clear:
            self.blockage_window_observed = True
            if simulation.reached_goal:
                self.goal_reached_after_clear = True
        self.previous_room = current_room

    @property
    def safe_through_blockage(self) -> bool:
        return bool(
            self.blockage_window_observed
            and np.isfinite(self.minimum_blockage_safety_clearance)
            and self.minimum_blockage_safety_clearance >= 0.0
        )


def _nominal_control(
    controller: HospitalController,
    state: np.ndarray,
) -> np.ndarray:
    target = controller._navigation_target(state)
    return waypoint_control(
        state,
        target,
        controller.config.robot,
        controller.config.policies.nominal_target_speed,
    )


def _candidate_policies(
    controller: HospitalController,
    state: np.ndarray,
    obstacles: Sequence,
) -> list[HospitalPolicy]:
    del obstacles
    return controller.candidate_policies(state)


def _raw_local_obstacle_ranks(
    state: Sequence[float],
    obstacles: Sequence,
    config: HospitalConfig,
) -> tuple[int, tuple[int, ...]]:
    """Return range-only population and blocker ranks without ID priority."""

    position = np.asarray(state, dtype=float)[:2]
    ranked: list[tuple[float, str]] = []
    for obstacle in obstacles:
        proxy_radius = (
            obstacle.radius
            if isinstance(obstacle, Human)
            else 0.5 * np.hypot(obstacle.length, obstacle.width)
        )
        range_distance = float(
            np.linalg.norm(obstacle.center - position) - proxy_radius
        )
        if range_distance <= config.robot.sensing_range:
            ranked.append((range_distance, obstacle.identifier))
    ranked.sort(key=lambda item: item[0])
    blocker_ranks = tuple(
        index
        for index, (_distance, identifier) in enumerate(ranked, start=1)
        if identifier.startswith("blocking-stretcher-")
    )
    return len(ranked), blocker_ranks


def _classify_outcome(
    *,
    physical_collision: bool,
    goal_reached: bool,
) -> BenchmarkOutcome:
    """Return the Hospital task outcome; safety remains a separate metric."""

    if physical_collision:
        return BenchmarkOutcome.COLLISION
    if goal_reached:
        return BenchmarkOutcome.SUCCESS
    return BenchmarkOutcome.TIMEOUT


def _operational_safety_violation(clearance: float) -> bool:
    """The closed certified safe set includes its zero-clearance boundary."""

    return float(clearance) < 0.0


def _clearance_witness_metrics(
    prefix: str,
    witness: ClearanceWitness,
    *,
    step_index: int,
    absolute_time_s: float,
) -> dict[str, bool | int | float | str | None]:
    """Flatten one structured witness into report-compatible scalar fields."""

    return {
        f"{prefix}_source_kind": witness.source_kind,
        f"{prefix}_obstacle_identifier": witness.obstacle_identifier,
        f"{prefix}_step_index": int(step_index),
        f"{prefix}_time_s": float(absolute_time_s),
        f"{prefix}_transition_elapsed_s": float(witness.elapsed_s),
        f"{prefix}_substep_index": int(witness.sample_index),
        f"{prefix}_substep_fraction": float(witness.sample_fraction),
        f"{prefix}_robot_x_m": float(witness.robot_position[0]),
        f"{prefix}_robot_y_m": float(witness.robot_position[1]),
    }


_ROOM_POLICY_METHODS = frozenset(
    {
        BenchmarkMethod.PLCBF,
        BenchmarkMethod.MULTI_BACKUP_CBF_MI,
        BenchmarkMethod.LIBRARY_PCBF_MI,
    }
)


def _room_policy_available_to_method(method: BenchmarkMethod) -> bool:
    """Whether this native method receives Hospital room-policy candidates."""

    return method in _ROOM_POLICY_METHODS


def _decision_event_flags(
    method: BenchmarkMethod,
    decision: BaselineDecision,
    *,
    feasible: bool | None = None,
) -> tuple[bool, bool, bool, bool]:
    """Split exceptional solver fallback from normal backup execution."""

    status = decision.status
    policy_decision = decision.policy_decision
    selector_fallback = bool(
        (
            policy_decision is not None
            and policy_decision.diagnostics.used_fallback
        )
        # Preserve the predicate for lightweight test/custom adapters that do
        # not attach the underlying PolicyDecision.
        or (
            policy_decision is None
            and status.startswith("fallback:")
        )
    )
    final_feasible = decision.feasible if feasible is None else bool(feasible)
    solver_fallback = bool(not final_feasible or selector_fallback)
    mps_backup = method is BenchmarkMethod.MPS and decision.used_fallback
    gatekeeper_backup = (
        method is BenchmarkMethod.GATEKEEPER and decision.used_fallback
    )
    selected_policy_backup = bool(
        selector_fallback
        and method
        in (
            BenchmarkMethod.PLCBF,
            BenchmarkMethod.LIBRARY_PCBF_MI,
        )
    )
    fixed_backup_emergency = bool(
        method is BenchmarkMethod.BACKUP_CBF
        and decision.used_fallback
        and status.startswith("fallback_after:")
    )
    multi_backup_emergency = bool(
        method is BenchmarkMethod.MULTI_BACKUP_CBF_MI
        and decision.used_fallback
    )
    mi_mpc_emergency = bool(
        method is BenchmarkMethod.MI_MPC and decision.used_fallback
    )
    backup_executed = bool(
        selected_policy_backup
        or mps_backup
        or gatekeeper_backup
        or fixed_backup_emergency
        or multi_backup_emergency
        or mi_mpc_emergency
    )
    shield_active = bool(
        final_feasible
        and (
            mps_backup or gatekeeper_backup
        )
    )
    return (
        selector_fallback,
        solver_fallback,
        backup_executed,
        shield_active,
    )


def _mi_mpc_safety_flags(
    method: BenchmarkMethod,
    baseline_suite: object,
) -> tuple[bool, bool] | None:
    """Return requested-safety and relaxed-admission diagnostics.

    The Big-M MPC's warehouse-compatible emergency admission can produce an
    optimization-feasible trajectory whose selected branch does not meet the
    originally requested safety threshold.  Keep those two facts separate.
    This read-only helper is called only after the control has been chosen.
    """

    if method is not BenchmarkMethod.MI_MPC:
        return None
    result = getattr(baseline_suite, "last_mi_mpc_result", None)
    if result is None:
        return None
    return (
        bool(result.safety_feasible),
        bool(result.safety_threshold_relaxed),
    )


def run_hospital_trial(
    method: BenchmarkMethod | str,
    case_id: str,
    *,
    seed: int = 0,
    steps: int | None = None,
    config: HospitalConfig = DEFAULT_CONFIG,
    oracle_period_s: float | None = None,
    warmup: bool = True,
    raise_errors: bool = False,
    _scenario: HospitalStoryScenario | None = None,
) -> BenchmarkResult:
    """Execute one method/case/seed trial and return a common result record."""

    publication_case = str(case_id) in HOSPITAL_BENCHMARK_STORIES
    if publication_case:
        config = publication_benchmark_config(config)
    validate_strict_refuge_protocol(config)
    parsed_method = (
        method if isinstance(method, BenchmarkMethod) else BenchmarkMethod(method)
    )
    if steps is None:
        steps = default_hospital_benchmark_steps(config)
    if steps <= 0:
        raise ValueError("steps must be positive")
    if oracle_period_s is not None and not np.isclose(
        oracle_period_s, config.dt
    ):
        raise ValueError(
            "Hospital policy certificates must be recomputed every plant "
            "step; oracle_period_s, when supplied, must equal config.dt"
        )
    plcbf_selector = parsed_method is BenchmarkMethod.PLCBF
    result_case_id = (
        f"{case_id}/seed-{int(seed)}" if publication_case else str(case_id)
    )
    simulation: HospitalSimulation | None = None
    sampled_scenario_metrics: dict[str, bool | int | float | str] = {}
    try:
        if _scenario is not None:
            if (
                not publication_case
                or _scenario.trial.story_id != str(case_id)
                or _scenario.trial.traffic_seed != int(seed)
            ):
                raise ValueError("prebuilt Hospital story does not match trial")
            simulation = _scenario.to_simulation(config)
        else:
            simulation = build_benchmark_scenario(
                case_id,
                seed=seed,
                config=config,
            )
        sampled_scenario_metrics = dict(
            simulation.benchmark_scenario_metrics
        )
        scenario_contract = (
            {
                "publication_story": True,
                "paired_world_shared_across_methods": True,
                "success_uses_room_diagnostics": False,
            }
            if publication_case
            else strict_refuge_scenario_metadata(config)
        )
        controller = simulation.controller
        baseline_suite = HospitalBaselineSuite(
            controller, simulation.environment, config
        )
        jax_certificate_method = parsed_method in {
            BenchmarkMethod.PLCBF,
            BenchmarkMethod.POLICY_PCBF,
            BenchmarkMethod.LIBRARY_PCBF_MI,
        }
        jit_cache_before_warmup = controller.jax_cache_info()
        jit_warmup_elapsed_s = 0.0
        if warmup and jax_certificate_method:
            warmup_started = time.perf_counter()
            if parsed_method is BenchmarkMethod.POLICY_PCBF:
                warmup_policies = (
                    baseline_suite.fixed_backup_policy(simulation.state),
                )
            else:
                warmup_policies = None
            controller.warmup_certificate_oracle(
                simulation.state,
                policies=warmup_policies,
                include_diagnostics=False,
            )
            jit_warmup_elapsed_s = time.perf_counter() - warmup_started
        jit_cache_after_warmup = controller.jax_cache_info()
        narrative_trace = _HospitalNarrativeTrace.from_simulation(simulation)
        deadlock_monitor = _DeadlockMonitor(
            eligible_after_s=_deadlock_eligible_after_s(
                sampled_scenario_metrics
            )
        )
        deadlock_monitor.update(
            simulation.time,
            simulation.state,
            simulation.goal,
        )
        limit = config.robot.a_max
        lower = np.array([-limit, -limit])
        upper = np.array([limit, limit])
        initial_transition = evaluate_swept_transition(
            simulation.environment,
            simulation.obstacles,
            simulation.state,
            simulation.state,
            0.0,
            config,
        )
        minimum_clearance = initial_transition.minimum_clearance
        minimum_safety_clearance = (
            initial_transition.minimum_safety_clearance
        )
        minimum_clearance_witness = (
            initial_transition.minimum_clearance_witness
        )
        minimum_safety_clearance_witness = (
            initial_transition.minimum_safety_clearance_witness
        )
        if (
            minimum_clearance_witness is None
            or minimum_safety_clearance_witness is None
        ):
            raise RuntimeError(
                "Hospital swept-clearance diagnostics are unavailable"
            )
        minimum_clearance_step_index = 0
        minimum_clearance_time_s = float(simulation.time)
        minimum_safety_clearance_step_index = 0
        minimum_safety_clearance_time_s = float(simulation.time)
        initial_distance = float(
            np.linalg.norm(simulation.goal - simulation.state[:2])
        )
        intervention_sum = 0.0
        intervention_energy = 0.0
        solver_times: list[float] = []
        oracle_time_total = 0.0
        solver_time_total = 0.0
        oracle_calls = 0
        selector_fallback_count = 0
        solver_fallback_count = 0
        backup_executed_count = 0
        shield_active_count = 0
        mi_mpc_result_count = 0
        mi_mpc_result_missing_count = 0
        mi_mpc_requested_safety_feasible_count = 0
        mi_mpc_safety_threshold_relaxed_count = 0
        infeasible_count = 0
        policy_count_max = 0
        policy_count_min = 1_000_000
        room_selection_count = 0
        normal_qp_room_selection_count = 0
        numerical_fallback_room_selection_count = 0
        max_raw_sensed_obstacle_count = 0
        max_blocker_range_rank = 0
        blocker_capacity_miss_count = 0
        executed_steps = 0
        last_decision: BaselineDecision | None = None
        selected_policy = "nominal"
        refresh_period = float(config.dt)

        for step_index in range(int(steps)):
            # Freeze one method-independent perception snapshot per physical
            # step.  All solvers receive this same local observation, while
            # collision scoring and obstacle motion below retain the complete
            # world state.
            perceived_obstacles = sensed_obstacles(
                simulation.state,
                simulation.obstacles,
                config,
                environment=simulation.environment,
            )
            raw_count, blocker_ranks = _raw_local_obstacle_ranks(
                simulation.state,
                simulation.obstacles,
                config,
            )
            max_raw_sensed_obstacle_count = max(
                max_raw_sensed_obstacle_count,
                raw_count,
            )
            max_blocker_range_rank = max(
                max_blocker_range_rank,
                max(blocker_ranks, default=0),
            )
            blocker_capacity_miss_count += sum(
                rank > config.safety.max_obstacles for rank in blocker_ranks
            )
            nominal = _nominal_control(
                controller,
                simulation.state,
            )
            oracle_started = time.perf_counter()
            if parsed_method is BenchmarkMethod.POLICY_PCBF:
                policies = (
                    baseline_suite.fixed_backup_policy(simulation.state),
                )
                certificates, _ = controller.build_policy_certificates(
                    simulation.state,
                    perceived_obstacles,
                    policies,
                    nominal_control=nominal,
                    obstacles_are_sensed=True,
                    include_diagnostics=False,
                )
                policy_count = 1
                refreshed = True
            elif parsed_method is BenchmarkMethod.LIBRARY_PCBF_MI:
                policies = _candidate_policies(
                    controller,
                    simulation.state,
                    perceived_obstacles,
                )
                certificates, _ = controller.build_policy_certificates(
                    simulation.state,
                    perceived_obstacles,
                    policies,
                    nominal_control=nominal,
                    obstacles_are_sensed=True,
                    include_diagnostics=False,
                )
                policy_count = len(policies)
                refreshed = True
            elif parsed_method is BenchmarkMethod.PLCBF:
                policies = ()
                certificates = ()
                policy_count = 0
                refreshed = True
            elif parsed_method is BenchmarkMethod.MULTI_BACKUP_CBF_MI:
                policies = _candidate_policies(
                    controller,
                    simulation.state,
                    perceived_obstacles,
                )
                certificates = ()
                policy_count = len(policies)
                refreshed = False
            elif parsed_method is BenchmarkMethod.MI_MPC:
                policies = baseline_suite.mi_mpc_policies(
                    simulation.state
                )
                certificates = ()
                policy_count = len(policies)
                refreshed = False
            else:
                policies = ()
                certificates = ()
                policy_count = 1
                refreshed = False
            oracle_elapsed = time.perf_counter() - oracle_started
            oracle_time_total += oracle_elapsed
            oracle_calls += int(refreshed)
            solver_started = time.perf_counter()
            decision = baseline_suite.solve(
                parsed_method,
                simulation.state,
                perceived_obstacles,
                nominal,
                certificates=certificates,
                policies=policies or None,
                time_seconds=simulation.time,
            )
            mi_mpc_safety = _mi_mpc_safety_flags(
                parsed_method,
                baseline_suite,
            )
            if parsed_method is BenchmarkMethod.MI_MPC:
                if mi_mpc_safety is None:
                    mi_mpc_result_missing_count += 1
                else:
                    requested_safe, threshold_relaxed = mi_mpc_safety
                    mi_mpc_result_count += 1
                    mi_mpc_requested_safety_feasible_count += int(
                        requested_safe
                    )
                    mi_mpc_safety_threshold_relaxed_count += int(
                        threshold_relaxed
                    )
            selected_room_policy = bool(
                decision.policy_id is not None
                and decision.policy_id.startswith("room")
            )
            room_selection_count += int(selected_room_policy)
            control = np.asarray(decision.control, dtype=float)
            feasible = decision.feasible
            solver_elapsed = time.perf_counter() - solver_started
            solver_time_total += solver_elapsed
            solver_times.append(oracle_elapsed + solver_elapsed)
            if parsed_method is BenchmarkMethod.PLCBF:
                plcbf_result = baseline_suite.last_plcbf_result
                policy_count = (
                    1
                    if plcbf_result is None
                    else plcbf_result.candidate_policy_count
                )
            policy_count_max = max(policy_count_max, policy_count)
            policy_count_min = min(policy_count_min, policy_count)
            (
                selector_fallback,
                solver_fallback,
                backup_executed,
                shield_active,
            ) = _decision_event_flags(
                parsed_method,
                decision,
                feasible=feasible,
            )
            selector_fallback_count += int(selector_fallback)
            solver_fallback_count += int(solver_fallback)
            backup_executed_count += int(backup_executed)
            shield_active_count += int(shield_active)
            infeasible_count += int(not feasible)
            if parsed_method is BenchmarkMethod.PLCBF and selected_room_policy:
                normal_qp_room_selection_count += int(
                    feasible and not selector_fallback
                )
                numerical_fallback_room_selection_count += int(
                    selector_fallback
                )
            last_decision = decision
            selected_policy = decision.policy_id or "fallback"

            squared_deviation = float(np.sum((control - nominal) ** 2))
            intervention_sum += squared_deviation
            intervention_energy += squared_deviation * config.dt
            previous_time = simulation.time
            previous = simulation.state.copy()
            following = step_double_integrator(
                simulation.state,
                control,
                config.dt,
                config.robot,
            )
            transition = evaluate_swept_transition(
                simulation.environment,
                simulation.obstacles,
                previous,
                following,
                config.dt,
                config,
            )
            for obstacle in simulation.obstacles:
                obstacle.advance(config.dt, simulation.environment)
            simulation.state = following
            simulation.time += config.dt
            simulation.collision = transition.collision
            simulation.reached_goal = bool(
                np.linalg.norm(simulation.state[:2] - simulation.goal) <= 1.35
            )
            executed_steps = step_index + 1
            if transition.minimum_clearance < minimum_clearance:
                if transition.minimum_clearance_witness is None:
                    raise RuntimeError(
                        "physical-clearance witness is unavailable"
                    )
                minimum_clearance = transition.minimum_clearance
                minimum_clearance_witness = (
                    transition.minimum_clearance_witness
                )
                minimum_clearance_step_index = step_index + 1
                minimum_clearance_time_s = (
                    previous_time + minimum_clearance_witness.elapsed_s
                )
            if (
                transition.minimum_safety_clearance
                < minimum_safety_clearance
            ):
                if transition.minimum_safety_clearance_witness is None:
                    raise RuntimeError(
                        "operational-clearance witness is unavailable"
                    )
                minimum_safety_clearance = (
                    transition.minimum_safety_clearance
                )
                minimum_safety_clearance_witness = (
                    transition.minimum_safety_clearance_witness
                )
                minimum_safety_clearance_step_index = step_index + 1
                minimum_safety_clearance_time_s = (
                    previous_time
                    + minimum_safety_clearance_witness.elapsed_s
                )
            narrative_trace.update(
                simulation,
                transition_started_at_s=previous_time,
                transition_safety_clearance=(
                    transition.minimum_safety_clearance
                ),
            )
            deadlock_monitor.update(
                simulation.time,
                simulation.state,
                simulation.goal,
            )
            if simulation.collision or simulation.reached_goal:
                break

        final_distance = float(
            np.linalg.norm(simulation.goal - simulation.state[:2])
        )
        progress = (initial_distance - final_distance) / max(
            initial_distance,
            1e-12,
        )
        room_entered = narrative_trace.post_departure_room is not None
        room_left = (
            narrative_trace.post_departure_room_left_at_s is not None
        )
        operational_safety_violation = _operational_safety_violation(
            minimum_safety_clearance
        )
        benchmark_success = bool(
            simulation.reached_goal and not simulation.collision
        )
        unsafe = bool(
            simulation.collision
            or operational_safety_violation
        )
        outcome = _classify_outcome(
            physical_collision=simulation.collision,
            goal_reached=simulation.reached_goal,
        )
        termination_reason = (
            "physical_collision"
            if outcome is BenchmarkOutcome.COLLISION
            else (
                "goal_reached"
                if outcome is BenchmarkOutcome.SUCCESS
                else (
                    "deadlock_at_horizon"
                    if deadlock_monitor.currently_deadlocked
                    else "max_horizon"
                )
            )
        )
        total_decision_time = oracle_time_total + solver_time_total
        jit_cache_after_trial = controller.jax_cache_info()

        def cache_misses(snapshot: dict[str, dict[str, int | None]]) -> int:
            return sum(
                int(family.get("misses") or 0)
                for family in snapshot.values()
            )

        runtime_jit_miss_delta = (
            cache_misses(jit_cache_after_trial)
            - cache_misses(jit_cache_after_warmup)
        )
        return BenchmarkResult(
            algorithm=parsed_method.value,
            case_id=result_case_id,
            seed=int(seed),
            outcome=outcome,
            min_clearance=float(minimum_clearance),
            intervention=intervention_sum / max(1, executed_steps),
            solve_times_s=tuple(solver_times),
            case_metrics={
                **scenario_contract,
                **sampled_scenario_metrics,
                "collision": simulation.collision,
                "unsafe": unsafe,
                "unsafe_reason": (
                    "physical_collision"
                    if simulation.collision
                    else (
                        "operational_safety_clearance"
                        if operational_safety_violation
                        else "none"
                    )
                ),
                "completion": benchmark_success,
                "goal_reached": simulation.reached_goal,
                "operationally_safe_goal_completion": bool(
                    benchmark_success and not operational_safety_violation
                ),
                "termination_reason": termination_reason,
                "hospital_task_outcome_schema": (
                    "goal_collision_or_timeout_v1"
                ),
                "operational_safety_affects_task_outcome": False,
                "solver_infeasibility_affects_task_outcome": False,
                "progress": float(progress),
                "final_distance": final_distance,
                "clearance_diagnostic_schema_version": (
                    "hospital_clearance_witness_v1"
                ),
                "minimum_physical_clearance": minimum_clearance,
                **_clearance_witness_metrics(
                    "minimum_physical_clearance",
                    minimum_clearance_witness,
                    step_index=minimum_clearance_step_index,
                    absolute_time_s=minimum_clearance_time_s,
                ),
                "minimum_safety_clearance": minimum_safety_clearance,
                **_clearance_witness_metrics(
                    "minimum_safety_clearance",
                    minimum_safety_clearance_witness,
                    step_index=minimum_safety_clearance_step_index,
                    absolute_time_s=minimum_safety_clearance_time_s,
                ),
                "operational_safety_violation": (
                    operational_safety_violation
                ),
                "steps": executed_steps,
                "sim_time_s": executed_steps * config.dt,
                "maximum_steps": int(steps),
                "maximum_sim_time_s": float(steps * config.dt),
                "deadlock_checker_enabled": True,
                "deadlock_checker_is_controller_input": False,
                "deadlock_checker_stops_simulation": False,
                "deadlock_eligible_after_s": (
                    deadlock_monitor.eligible_after_s
                ),
                "deadlock_window_s": deadlock_monitor.window_s,
                "deadlock_max_path_length_m": (
                    deadlock_monitor.max_path_length_m
                ),
                "deadlock_max_goal_progress_m": (
                    deadlock_monitor.max_goal_progress_m
                ),
                "deadlock_max_speed_mps": (
                    deadlock_monitor.max_speed_mps
                ),
                "deadlock_detected": deadlock_monitor.detected,
                "deadlock_first_detected_at_s": (
                    deadlock_monitor.first_detected_at_s
                ),
                "deadlock_episode_count": deadlock_monitor.episode_count,
                "deadlock_resolved_episode_count": (
                    deadlock_monitor.resolved_episode_count
                ),
                "deadlocked_at_end": (
                    deadlock_monitor.currently_deadlocked
                ),
                "deadlock_last_window_path_length_m": (
                    deadlock_monitor.last_window_path_length_m
                ),
                "deadlock_last_window_goal_progress_m": (
                    deadlock_monitor.last_window_goal_progress_m
                ),
                "deadlock_last_window_max_speed_mps": (
                    deadlock_monitor.last_window_max_speed_mps
                ),
                "stretcher_count": _case_stretcher_count(case_id),
                "room_entered": room_entered,
                "post_departure_room_entered": room_entered,
                "post_departure_room_entered_at_s": (
                    narrative_trace.post_departure_room_entered_at_s
                ),
                "room_left": room_left,
                "post_departure_room_left_at_s": (
                    narrative_trace.post_departure_room_left_at_s
                ),
                "post_departure_room_label": (
                    None
                    if narrative_trace.post_departure_room is None
                    else narrative_trace.post_departure_room.label
                ),
                "start_room_left": narrative_trace.start_room_left,
                "start_room_left_at_s": (
                    narrative_trace.start_room_left_at_s
                ),
                "room_occupied_during_blockage": (
                    narrative_trace.room_occupied_during_blockage
                ),
                "blockage_window_observed": (
                    narrative_trace.blockage_window_observed
                ),
                "minimum_blockage_safety_clearance": (
                    None
                    if not np.isfinite(
                        narrative_trace.minimum_blockage_safety_clearance
                    )
                    else narrative_trace.minimum_blockage_safety_clearance
                ),
                "safe_through_blockage": narrative_trace.safe_through_blockage,
                "room_exited_after_clear": (
                    narrative_trace.room_exited_after_clear
                ),
                "goal_reached_after_clear": (
                    narrative_trace.goal_reached_after_clear
                ),
                "room_occupancy_is_diagnostic_only": True,
                "blockage_diagnostics_are_success_requirements": False,
                "external_room_policy_executor": False,
                "external_room_selector": False,
                "external_refuge_state_machine": False,
                "room_policy_available_to_method": (
                    _room_policy_available_to_method(parsed_method)
                ),
                "plcbf_selector": plcbf_selector,
                "gatekeeper_stateful": (
                    parsed_method is BenchmarkMethod.GATEKEEPER
                ),
                "selector_fallback_count": selector_fallback_count,
                "selector_fallback_rate": (
                    selector_fallback_count / max(1, executed_steps)
                ),
                "solver_fallback_count": solver_fallback_count,
                "solver_fallback_rate": (
                    solver_fallback_count / max(1, executed_steps)
                ),
                "backup_executed_count": backup_executed_count,
                "backup_executed_rate": (
                    backup_executed_count / max(1, executed_steps)
                ),
                "shield_active_count": shield_active_count,
                "shield_active_rate": (
                    shield_active_count / max(1, executed_steps)
                ),
                "mi_mpc_result_count": (
                    mi_mpc_result_count
                    if parsed_method is BenchmarkMethod.MI_MPC
                    else None
                ),
                "mi_mpc_result_missing_count": (
                    mi_mpc_result_missing_count
                    if parsed_method is BenchmarkMethod.MI_MPC
                    else None
                ),
                "mi_mpc_requested_safety_feasible_count": (
                    mi_mpc_requested_safety_feasible_count
                    if parsed_method is BenchmarkMethod.MI_MPC
                    else None
                ),
                "mi_mpc_requested_safety_feasible_rate": (
                    mi_mpc_requested_safety_feasible_count
                    / max(1, mi_mpc_result_count)
                    if parsed_method is BenchmarkMethod.MI_MPC
                    else None
                ),
                "mi_mpc_safety_threshold_relaxed_count": (
                    mi_mpc_safety_threshold_relaxed_count
                    if parsed_method is BenchmarkMethod.MI_MPC
                    else None
                ),
                "mi_mpc_safety_threshold_relaxed_rate": (
                    mi_mpc_safety_threshold_relaxed_count
                    / max(1, mi_mpc_result_count)
                    if parsed_method is BenchmarkMethod.MI_MPC
                    else None
                ),
                "infeasible_count": infeasible_count,
                "infeasible_rate": infeasible_count / max(1, executed_steps),
                "intervention_energy": intervention_energy,
                "intervention_squared_deviation_total": intervention_sum,
                "oracle_calls": oracle_calls,
                "oracle_period_s": refresh_period,
                "plant_dt_s": config.dt,
                "policy_count_max": policy_count_max,
                "policy_count_min": (
                    0 if policy_count_min == 1_000_000 else policy_count_min
                ),
                "room_policy_selection_count": room_selection_count,
                "room_policy_selection_rate": (
                    room_selection_count / max(1, executed_steps)
                ),
                "normal_qp_room_selection_count": (
                    normal_qp_room_selection_count
                ),
                "normal_qp_room_selection_rate": (
                    normal_qp_room_selection_count / max(1, executed_steps)
                ),
                "normal_qp_room_selected": (
                    normal_qp_room_selection_count > 0
                ),
                "numerical_fallback_room_selection_count": (
                    numerical_fallback_room_selection_count
                ),
                "numerical_fallback_room_selection_rate": (
                    numerical_fallback_room_selection_count
                    / max(1, executed_steps)
                ),
                "numerical_fallback_room_selected": (
                    numerical_fallback_room_selection_count > 0
                ),
                "controller_max_sensed_obstacles": (
                    config.safety.max_obstacles
                ),
                "maximum_raw_in_range_obstacle_count": (
                    max_raw_sensed_obstacle_count
                ),
                "maximum_blocker_range_rank": max_blocker_range_rank,
                "blocker_capacity_miss_count": blocker_capacity_miss_count,
                "blocker_id_priority_used": False,
                "oracle_time_total_s": oracle_time_total,
                "solver_time_total_s": solver_time_total,
                "oracle_and_solver_time_total_s": total_decision_time,
                "jit_warmup_enabled": bool(
                    warmup and jax_certificate_method
                ),
                "jit_warmup_elapsed_s": float(jit_warmup_elapsed_s),
                "jit_warmup_excluded_from_step_timing": bool(
                    warmup and jax_certificate_method
                ),
                "jit_cache_misses_before_warmup": cache_misses(
                    jit_cache_before_warmup
                ),
                "jit_cache_misses_after_warmup": cache_misses(
                    jit_cache_after_warmup
                ),
                "jit_cache_misses_after_trial": cache_misses(
                    jit_cache_after_trial
                ),
                "jit_grouped_cache_size_after_warmup": int(
                    jit_cache_after_warmup["grouped"].get("currsize") or 0
                ),
                "jit_batch_cache_size_after_warmup": int(
                    jit_cache_after_warmup["batch"].get("currsize") or 0
                ),
                "runtime_jit_cache_miss_delta": int(
                    runtime_jit_miss_delta
                ),
                "runtime_jit_compilation_detected": bool(
                    jax_certificate_method and runtime_jit_miss_delta > 0
                ),
                "last_solver_status": (
                    "none" if last_decision is None else last_decision.status
                ),
                "last_selected_policy": selected_policy,
            },
        )
    except Exception as exc:
        if raise_errors:
            raise
        minimum = (
            None
            if simulation is None
            else float(simulation.minimum_clearance())
        )
        return BenchmarkResult(
            algorithm=parsed_method.value,
            case_id=result_case_id,
            seed=int(seed),
            outcome=BenchmarkOutcome.ERROR,
            min_clearance=minimum,
            intervention=0.0,
            case_metrics={
                **(
                    {
                        "publication_story": True,
                        "paired_world_shared_across_methods": True,
                    }
                    if publication_case
                    else strict_refuge_scenario_metadata(config)
                ),
                **sampled_scenario_metrics,
                "stretcher_count": _case_stretcher_count(case_id),
                "external_room_policy_executor": False,
                "external_room_selector": False,
                "external_refuge_state_machine": False,
                "room_policy_available_to_method": (
                    _room_policy_available_to_method(parsed_method)
                ),
                "plcbf_selector": plcbf_selector,
            },
            error=f"{type(exc).__name__}: {exc}",
        )


def run_hospital_benchmark(
    *,
    methods: Iterable[BenchmarkMethod | str] = BENCHMARK_METHODS,
    cases: Iterable[str] = HOSPITAL_BENCHMARK_STORIES,
    seeds: Iterable[int] = DEFAULT_HOSPITAL_TRAFFIC_SEEDS,
    steps: int | None = None,
    config: HospitalConfig = DEFAULT_CONFIG,
    oracle_period_s: float | None = None,
    compact_policy_library: bool = False,
    progress: bool = False,
) -> tuple[BenchmarkResult, ...]:
    """Materialize the deterministic Cartesian product in stable order."""

    validate_strict_refuge_protocol(config)
    if steps is None:
        steps = default_hospital_benchmark_steps(config)
    if steps <= 0:
        raise ValueError("steps must be positive")
    parsed_methods = tuple(
        item if isinstance(item, BenchmarkMethod) else BenchmarkMethod(item)
        for item in methods
    )
    case_ids = tuple(str(item) for item in cases)
    seed_values = tuple(int(item) for item in seeds)
    if not parsed_methods or not case_ids or not seed_values:
        raise ValueError("methods, cases, and seeds must all be non-empty")
    for case_id in case_ids:
        _case_stretcher_count(case_id)
    benchmark_config = (
        compact_benchmark_config(config)
        if compact_policy_library
        else config
    )
    results = []
    total = len(parsed_methods) * len(case_ids) * len(seed_values)
    index = 0
    for case_id in case_ids:
        for seed in seed_values:
            scenario = (
                build_hospital_story_scenario(
                    case_id,
                    traffic_seed=seed,
                    config=DEFAULT_CONFIG,
                )
                if case_id in HOSPITAL_BENCHMARK_STORIES
                else None
            )
            for method in parsed_methods:
                index += 1
                if progress:
                    print(
                        f"[{index}/{total}] {method.value} "
                        f"{case_id} seed={seed}",
                        flush=True,
                    )
                results.append(
                    run_hospital_trial(
                        method,
                        case_id,
                        seed=seed,
                        steps=steps,
                        config=benchmark_config,
                        oracle_period_s=oracle_period_s,
                        _scenario=scenario,
                    )
                )
    return tuple(results)


def _result_summary(results: Sequence[BenchmarkResult]) -> dict[str, object]:
    def summarize(rows: Sequence[BenchmarkResult]) -> dict[str, object]:
        return {
            aggregate.algorithm: {
                "trials": aggregate.trial_count,
                "success_rate": aggregate.success_rate,
                "collision_rate": aggregate.collision_rate,
                "infeasible_rate": aggregate.infeasible_rate,
                "decision_time_mean_s": aggregate.solve_time_mean_s,
                "decision_time_p95_s": aggregate.solve_time_p95_s,
                "clearance_mean": (
                    None
                    if aggregate.clearance is None
                    else aggregate.clearance.mean
                ),
            }
            for aggregate in aggregate_results(rows)
        }

    stories = sorted({hospital_story_id(row.case_id) for row in results})
    return {
        "pooled": summarize(results),
        "by_story": {
            story: summarize(
                [
                    row
                    for row in results
                    if hospital_story_id(row.case_id) == story
                ]
            )
            for story in stories
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=BENCHMARK_METHODS,
        default=list(BENCHMARK_METHODS),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=(*HOSPITAL_BENCHMARK_STORIES, *STRICT_HOSPITAL_CASES),
        default=list(HOSPITAL_BENCHMARK_STORIES),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_HOSPITAL_TRAFFIC_SEEDS),
    )
    parser.add_argument(
        "--steps",
        type=int,
        help=(
            "maximum plant steps per trial (default: 180 simulated seconds)"
        ),
    )
    parser.add_argument(
        "--oracle-period",
        type=float,
        help=(
            "deprecated audit option; hospital certificates are always "
            "recomputed at plant dt, so any supplied value must equal dt"
        ),
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        help=(
            "direct HospitalConfig JSON or a tuning summary containing "
            "best_config"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hospital_benchmark"),
        help="report prefix; CSV, JSON, and Markdown are written",
    )
    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument(
        "--full-policy-library",
        dest="policy_library_mode",
        action="store_const",
        const="full",
        help="preserve every direction/room candidate in the selected config",
    )
    policy_group.add_argument(
        "--compact-policy-library",
        dest="policy_library_mode",
        action="store_const",
        const="compact",
        help="force the practical four-direction, nearest-room library",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="two-step all-method smoke run on one story and one seed",
    )
    return parser


def _resolve_cli_protocol(
    arguments: argparse.Namespace,
) -> tuple[HospitalConfig, float, bool]:
    """Resolve config, oracle cadence, and library mode without hidden overrides."""

    config_path = arguments.config_json
    config = (
        DEFAULT_CONFIG
        if config_path is None
        else load_hospital_config(config_path)
    )
    if any(case in HOSPITAL_BENCHMARK_STORIES for case in arguments.cases):
        config = publication_benchmark_config(config)
    mode = arguments.policy_library_mode
    compact = mode == "compact"
    if arguments.oracle_period is not None:
        oracle_period = float(arguments.oracle_period)
        if not np.isclose(oracle_period, config.dt):
            raise ValueError(
                "--oracle-period must equal the hospital plant dt; stale "
                "policy certificates are not permitted"
            )
    else:
        oracle_period = float(config.dt)
    validate_strict_refuge_protocol(config)
    return config, oracle_period, compact


def main(argv: Sequence[str] | None = None) -> int:
    # Capture before any trials run: if a dirty working tree changes while a
    # long shard is executing, the report still identifies the source loaded
    # at shard startup rather than whatever happens to be on disk at the end.
    implementation_source_manifest = hospital_benchmark_source_manifest()
    arguments = build_parser().parse_args(argv)
    runtime_config, oracle_period, compact_policy_library = (
        _resolve_cli_protocol(arguments)
    )
    methods = arguments.methods
    cases = arguments.cases
    seeds = arguments.seeds
    steps = (
        default_hospital_benchmark_steps(runtime_config)
        if arguments.steps is None
        else arguments.steps
    )
    if arguments.quick:
        cases = cases[:1]
        seeds = seeds[:1]
        steps = min(steps, 2)
    results = run_hospital_benchmark(
        methods=methods,
        cases=cases,
        seeds=seeds,
        steps=steps,
        config=runtime_config,
        oracle_period_s=oracle_period,
        compact_policy_library=compact_policy_library,
        progress=True,
    )
    publication_only = all(
        case in HOSPITAL_BENCHMARK_STORIES for case in cases
    )
    metadata = {
        "case": (
            "hospital_fixed_story_refuge"
            if publication_only
            else "hospital_refuge_legacy"
        ),
        **({"scenarios": list(cases)} if publication_only else {}),
        "publication_protocol": (
            hospital_story_protocol_metadata()
            if publication_only
            else None
        ),
        "legacy_strict_cases": (
            None if publication_only else STRICT_HOSPITAL_CASES
        ),
        "randomization_protocol": (
            hospital_randomization_protocol_metadata()
        ),
        "methods": list(methods),
        "seeds": list(seeds),
        "steps": steps,
        "oracle_period_s": oracle_period,
        "plant_dt_s": runtime_config.dt,
        "compact_policy_library": compact_policy_library,
        "configuration_source": (
            "defaults"
            if arguments.config_json is None
            else str(arguments.config_json)
        ),
        "implementation_source_manifest": (
            implementation_source_manifest
        ),
        "external_room_policy_executor": False,
        "external_room_selector": False,
        "external_refuge_state_machine": False,
        "hospital_library_contains_room_policies": True,
        "paired_worlds_prebuilt_once_then_cloned_per_method": (
            publication_only
        ),
        "blocker_id_priority_used": False,
        "publication_sensor_capacity": (
            PUBLICATION_MAX_SENSED_OBSTACLES
            if publication_only
            else None
        ),
        "publication_sensing_range_m": (
            PUBLICATION_SENSING_RANGE_M if publication_only else None
        ),
        "success_requires": [
            "goal_reached",
            "no_physical_collision",
        ],
        "exclusive_task_outcomes": ["success", "collision", "timeout"],
        "operational_safety_clearance_is_diagnostic_only": True,
        "solver_infeasibility_is_diagnostic_only": True,
        "maximum_sim_time_s": steps * runtime_config.dt,
        "deadlock_diagnostic": {
            "controller_input": False,
            "stops_simulation": False,
            "eligible_after": (
                "convoy_clear_time_s_plus_"
                f"{DEADLOCK_POST_CONVOY_GRACE_S:g}_seconds"
            ),
            "window_s": DEADLOCK_WINDOW_S,
            "max_path_length_m": DEADLOCK_MAX_PATH_LENGTH_M,
            "max_goal_progress_m": DEADLOCK_MAX_GOAL_PROGRESS_M,
            "max_speed_mps": DEADLOCK_MAX_SPEED_MPS,
        },
        "decision_metrics": {
            "selector_fallback": (
                "The shared policy selector explicitly used its fallback "
                "control."
            ),
            "solver_fallback": (
                "The decision was infeasible or a policy selector explicitly "
                "used its fallback path."
            ),
            "backup_executed": (
                "The method entered an executable backup/emergency path. "
                "Normal MI-MPC continuous controls and ordinary filtered-QP "
                "policy selections are excluded."
            ),
            "shield_active": (
                "A feasible normal MPS or Gatekeeper committed-backup action "
                "was executed."
            ),
            "mi_mpc_requested_safety_feasible": (
                "The selected MI-MPC branch met the original requested safety "
                "threshold, separately from MILP trajectory feasibility."
            ),
            "mi_mpc_safety_threshold_relaxed": (
                "The warehouse max-safety emergency admission lowered the "
                "effective threshold because no branch met the requested one."
            ),
        },
        "room_occupancy_is_diagnostic_only": True,
        "blockage_narrative_diagnostics_are_success_requirements": False,
        "narrative_diagnostics": {
            "post_departure_room_entered": (
                "Physical entry into any room after leaving the initial room."
            ),
            "room_occupied_during_blockage": (
                "Any physical room occupancy during the fixed open-loop "
                "blockage window."
            ),
            "normal_qp_room_selected": (
                "PL-CBF selected a room policy with used_fallback=False."
            ),
            "numerical_fallback_room_selected": (
                "PL-CBF selected a room policy through the permitted "
                "numerical emergency path."
            ),
        },
        "current_hocbf_enforcement": (
            "inside_policy_qps_or_playground_emergency_projection"
        ),
        "certificate_rollout_backend": (
            "fixed_shape_horizon_grouped_jax"
        ),
        "jit_warmup_excluded_from_decision_timing": True,
        "runtime_jit_compilation_audited_per_trial": True,
        "collision_check": "static_segment_and_9_synchronized_samples",
        "config": asdict(
            compact_benchmark_config(runtime_config)
            if compact_policy_library
            else runtime_config
        ),
    }
    paths = write_benchmark_reports(
        arguments.output,
        results,
        metadata=metadata,
        title="Hospital refuge benchmark",
    )
    if publication_only:
        write_hospital_benchmark_markdown(
            paths.markdown,
            results,
            title="Hospital fixed-story refuge benchmark",
        )
    print(
        json.dumps(
            {
                "summary": _result_summary(results),
                "reports": {
                    "csv": str(paths.csv),
                    "json": str(paths.json),
                    "markdown": str(paths.markdown),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return int(any(result.outcome is BenchmarkOutcome.ERROR for result in results))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEADLOCK_MAX_GOAL_PROGRESS_M",
    "DEADLOCK_MAX_PATH_LENGTH_M",
    "DEADLOCK_MAX_SPEED_MPS",
    "DEADLOCK_POST_CONVOY_GRACE_S",
    "DEADLOCK_WINDOW_S",
    "DEFAULT_HOSPITAL_SIMULATION_TIME_S",
    "RANDOMIZED_EGO_VX_RANGE_MPS",
    "RANDOMIZED_EGO_VY_RANGE_MPS",
    "RANDOMIZED_EGO_X_RANGE_M",
    "RANDOMIZED_EGO_Y_RANGE_M",
    "RANDOMIZED_STRETCHER_SHIFT_RANGE_M",
    "RANDOMIZED_STRETCHER_SPEED_FACTOR_RANGE",
    "HOSPITAL_BENCHMARK_STORIES",
    "PUBLICATION_MAX_SENSED_OBSTACLES",
    "PUBLICATION_SENSING_RANGE_M",
    "STRICT_HOSPITAL_CASES",
    "build_benchmark_scenario",
    "compact_benchmark_config",
    "default_hospital_benchmark_steps",
    "hospital_benchmark_source_manifest",
    "hospital_randomization_protocol_metadata",
    "publication_benchmark_config",
    "run_hospital_benchmark",
    "run_hospital_trial",
]
