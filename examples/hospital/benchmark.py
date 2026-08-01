"""Headless eight-method benchmark for crowded hospital refuge cases.

All methods integrate the same double-integrator dynamics, exact obstacle
geometry, and policy-certificate oracle.  The full room/directional/stop
library is continuously re-evaluated.  There is no latched room executor,
timed hold, release guard, or phase-specific nominal controller.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
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
from .controller import HospitalController
from .dynamics import step_double_integrator, waypoint_control
from .environment import Room
from .obstacles import Human, Stretcher
from .policies import HospitalPolicy
from .scenario_generation import (
    DEFAULT_HUMAN_COUNT,
    DEFAULT_ORDINARY_STRETCHER_COUNT,
    generate_hospital_crowd,
)
from .simulation import (
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

RANDOMIZED_EGO_X_RANGE_M = (57.5, 58.0)
RANDOMIZED_EGO_Y_RANGE_M = (46.9, 48.1)
RANDOMIZED_EGO_VX_RANGE_MPS = (0.0, 0.2)
RANDOMIZED_EGO_VY_RANGE_MPS = (-0.1, 0.1)
RANDOMIZED_STRETCHER_SHIFT_RANGE_M = (-0.7, 0.0)
RANDOMIZED_STRETCHER_SPEED_FACTOR_RANGE = (1.0, 1.08)


def hospital_randomization_protocol_metadata() -> dict[str, object]:
    """Return the deterministic publication-trial sampling protocol."""

    return {
        "reference_seed": 0,
        "reference_seed_is_unmodified": False,
        "every_seed_generates_dynamic_background_traffic": True,
        "paired_seed_shared_across_methods": True,
        "human_count": DEFAULT_HUMAN_COUNT,
        "ordinary_stretcher_count": DEFAULT_ORDINARY_STRETCHER_COUNT,
        "guaranteed_full_width_blockers": [2, 3],
        "total_stretcher_counts": [17, 18],
        "ego_initial_x_m": {
            "distribution": "uniform",
            "minimum": RANDOMIZED_EGO_X_RANGE_M[0],
            "maximum": RANDOMIZED_EGO_X_RANGE_M[1],
        },
        "ego_initial_y_m": {
            "distribution": "uniform",
            "minimum": RANDOMIZED_EGO_Y_RANGE_M[0],
            "maximum": RANDOMIZED_EGO_Y_RANGE_M[1],
        },
        "ego_initial_vx_mps": {
            "distribution": "uniform",
            "minimum": RANDOMIZED_EGO_VX_RANGE_MPS[0],
            "maximum": RANDOMIZED_EGO_VX_RANGE_MPS[1],
        },
        "ego_initial_vy_mps": {
            "distribution": "uniform",
            "minimum": RANDOMIZED_EGO_VY_RANGE_MPS[0],
            "maximum": RANDOMIZED_EGO_VY_RANGE_MPS[1],
        },
        "per_stretcher_coordinate_shift_m": {
            "distribution": "uniform",
            "minimum": RANDOMIZED_STRETCHER_SHIFT_RANGE_M[0],
            "maximum": RANDOMIZED_STRETCHER_SHIFT_RANGE_M[1],
        },
        "per_stretcher_speed_magnitude_factor": {
            "distribution": "uniform",
            "minimum": RANDOMIZED_STRETCHER_SPEED_FACTOR_RANGE[0],
            "maximum": RANDOMIZED_STRETCHER_SPEED_FACTOR_RANGE[1],
        },
        "contract_preserving": (
            "every seed generates dense background traffic; nonzero seeds "
            "also move the ego and guaranteed blockers only in directions "
            "that preserve the full-width encounter"
        ),
    }


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
    """Build one seeded strict case without weakening its full-width blockade."""

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
class _RoomOccupancyTrace:
    """Geometry-only room-occupancy monitor; it never influences control."""

    entered: bool = False
    left_room: bool = False
    room: Room | None = None
    was_inside: bool = False
    inside_steps: int = 0
    maximum_inside_run_steps: int = 0
    entered_at_s: float | None = None
    left_at_s: float | None = None

    def update(self, simulation: HospitalSimulation) -> None:
        position = simulation.state[:2]
        environment = simulation.environment
        containing = environment.room_containing(position)
        if self.room is None and containing is not None:
            self.room = containing
            self.entered = True
            self.was_inside = True
            self.entered_at_s = simulation.time
        if self.room is None:
            return

        inside = self.room.contains(position)
        if inside:
            self.entered = True
            self.inside_steps += 1
            self.maximum_inside_run_steps = max(
                self.maximum_inside_run_steps,
                self.inside_steps,
            )
        elif self.was_inside:
            self.left_room = True
            if self.left_at_s is None:
                self.left_at_s = simulation.time
            self.inside_steps = 0
        else:
            self.inside_steps = 0
        self.was_inside = inside


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


def _classify_outcome(
    *,
    physical_collision: bool,
    operational_safety_violation: bool,
    goal_reached: bool,
    infeasible_count: int,
) -> BenchmarkOutcome:
    """Keep physical collision distinct from protocol violations."""

    if physical_collision:
        return BenchmarkOutcome.COLLISION
    if operational_safety_violation:
        return BenchmarkOutcome.INFEASIBLE
    if goal_reached:
        return BenchmarkOutcome.SUCCESS
    if infeasible_count > 0:
        return BenchmarkOutcome.INFEASIBLE
    return BenchmarkOutcome.TIMEOUT


def _operational_safety_violation(clearance: float) -> bool:
    """The closed certified safe set includes its zero-clearance boundary."""

    return float(clearance) < 0.0


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
    steps: int = 1100,
    config: HospitalConfig = DEFAULT_CONFIG,
    oracle_period_s: float | None = None,
    raise_errors: bool = False,
) -> BenchmarkResult:
    """Execute one method/case/seed trial and return a common result record."""

    validate_strict_refuge_protocol(config)
    parsed_method = (
        method if isinstance(method, BenchmarkMethod) else BenchmarkMethod(method)
    )
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
    simulation: HospitalSimulation | None = None
    sampled_scenario_metrics: dict[str, bool | int | float | str] = {}
    try:
        simulation = build_benchmark_scenario(
            case_id,
            seed=seed,
            config=config,
        )
        sampled_scenario_metrics = dict(
            simulation.benchmark_scenario_metrics
        )
        scenario_contract = strict_refuge_scenario_metadata(config)
        controller = simulation.controller
        baseline_suite = HospitalBaselineSuite(
            controller, simulation.environment, config
        )
        room_trace = _RoomOccupancyTrace()
        limit = config.robot.a_max
        lower = np.array([-limit, -limit])
        upper = np.array([limit, limit])
        minimum_clearance = simulation.minimum_clearance()
        minimum_safety_clearance = simulation.minimum_clearance(safety=True)
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
        executed_steps = 0
        last_decision: BaselineDecision | None = None
        selected_policy = "nominal"
        refresh_period = float(config.dt)

        for step_index in range(int(steps)):
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
                    simulation.obstacles,
                    policies,
                    nominal_control=nominal,
                )
                policy_count = 1
                refreshed = True
            elif parsed_method is BenchmarkMethod.LIBRARY_PCBF_MI:
                policies = _candidate_policies(
                    controller,
                    simulation.state,
                    simulation.obstacles,
                )
                certificates, _ = controller.build_policy_certificates(
                    simulation.state,
                    simulation.obstacles,
                    policies,
                    nominal_control=nominal,
                )
                policy_count = len(policies)
                refreshed = True
            elif parsed_method is BenchmarkMethod.PLCBF:
                policies = ()
                certificates = ()
                policy_count = len(
                    controller.candidate_policies(simulation.state)
                )
                refreshed = True
            elif parsed_method is BenchmarkMethod.MULTI_BACKUP_CBF_MI:
                policies = _candidate_policies(
                    controller,
                    simulation.state,
                    simulation.obstacles,
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
            policy_count_max = max(policy_count_max, policy_count)
            policy_count_min = min(policy_count_min, policy_count)

            solver_started = time.perf_counter()
            decision = baseline_suite.solve(
                parsed_method,
                simulation.state,
                simulation.obstacles,
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
            room_selection_count += int(
                bool(
                    decision.policy_id is not None
                    and decision.policy_id.startswith("room")
                )
            )
            control = np.asarray(decision.control, dtype=float)
            feasible = decision.feasible
            solver_elapsed = time.perf_counter() - solver_started
            solver_time_total += solver_elapsed
            solver_times.append(oracle_elapsed + solver_elapsed)
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
            last_decision = decision
            selected_policy = decision.policy_id or "fallback"

            squared_deviation = float(np.sum((control - nominal) ** 2))
            intervention_sum += squared_deviation
            intervention_energy += squared_deviation * config.dt
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
            minimum_clearance = min(
                minimum_clearance,
                transition.minimum_clearance,
            )
            minimum_safety_clearance = min(
                minimum_safety_clearance,
                transition.minimum_safety_clearance,
            )
            room_trace.update(simulation)
            if (
                simulation.collision
                or _operational_safety_violation(
                    minimum_safety_clearance
                )
                or simulation.reached_goal
            ):
                break

        final_distance = float(
            np.linalg.norm(simulation.goal - simulation.state[:2])
        )
        progress = (initial_distance - final_distance) / max(
            initial_distance,
            1e-12,
        )
        room_entered = room_trace.entered
        room_left = room_trace.left_room
        operational_safety_violation = _operational_safety_violation(
            minimum_safety_clearance
        )
        benchmark_success = bool(
            simulation.reached_goal
            and not simulation.collision
            and not operational_safety_violation
        )
        unsafe = bool(
            simulation.collision
            or operational_safety_violation
        )
        outcome = _classify_outcome(
            physical_collision=simulation.collision,
            operational_safety_violation=operational_safety_violation,
            goal_reached=simulation.reached_goal,
            infeasible_count=infeasible_count,
        )
        total_decision_time = oracle_time_total + solver_time_total
        return BenchmarkResult(
            algorithm=parsed_method.value,
            case_id=case_id,
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
                "progress": float(progress),
                "final_distance": final_distance,
                "minimum_safety_clearance": minimum_safety_clearance,
                "operational_safety_violation": (
                    operational_safety_violation
                ),
                "steps": executed_steps,
                "sim_time_s": executed_steps * config.dt,
                "stretcher_count": _case_stretcher_count(case_id),
                "room_entered": room_entered,
                "room_entered_at_s": room_trace.entered_at_s,
                "room_left": room_left,
                "room_left_at_s": room_trace.left_at_s,
                "entered_room_label": (
                    None
                    if room_trace.room is None
                    else room_trace.room.label
                ),
                "maximum_room_occupancy_s": (
                    room_trace.maximum_inside_run_steps * config.dt
                ),
                "room_occupancy_is_diagnostic_only": True,
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
                "oracle_time_total_s": oracle_time_total,
                "solver_time_total_s": solver_time_total,
                "oracle_and_solver_time_total_s": total_decision_time,
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
            case_id=case_id,
            seed=int(seed),
            outcome=BenchmarkOutcome.ERROR,
            min_clearance=minimum,
            intervention=0.0,
            case_metrics={
                **strict_refuge_scenario_metadata(config),
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
    cases: Iterable[str] = tuple(STRICT_HOSPITAL_CASES),
    seeds: Iterable[int] = (0,),
    steps: int = 1100,
    config: HospitalConfig = DEFAULT_CONFIG,
    oracle_period_s: float | None = None,
    compact_policy_library: bool = False,
    progress: bool = False,
) -> tuple[BenchmarkResult, ...]:
    """Materialize the deterministic Cartesian product in stable order."""

    validate_strict_refuge_protocol(config)
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
                    )
                )
    return tuple(results)


def _result_summary(results: Sequence[BenchmarkResult]) -> dict[str, object]:
    return {
        aggregate.algorithm: {
            "trials": aggregate.trial_count,
            "success_rate": aggregate.success_rate,
            "collision_rate": aggregate.collision_rate,
            "infeasible_rate": aggregate.infeasible_rate,
            "clearance_mean": (
                None
                if aggregate.clearance is None
                else aggregate.clearance.mean
            ),
        }
        for aggregate in aggregate_results(results)
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
        choices=tuple(STRICT_HOSPITAL_CASES),
        default=list(STRICT_HOSPITAL_CASES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=1100)
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
        help="two-step all-method smoke run on both strict cases",
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
    arguments = build_parser().parse_args(argv)
    runtime_config, oracle_period, compact_policy_library = (
        _resolve_cli_protocol(arguments)
    )
    methods = arguments.methods
    cases = arguments.cases
    seeds = arguments.seeds
    steps = arguments.steps
    if arguments.quick:
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
    metadata = {
        "case": "hospital_refuge",
        "strict_cases": STRICT_HOSPITAL_CASES,
        "strict_scenario_contract": strict_refuge_scenario_metadata(
            runtime_config
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
        "external_room_policy_executor": False,
        "external_room_selector": False,
        "external_refuge_state_machine": False,
        "hospital_library_contains_room_policies": True,
        "success_requires": [
            "goal_reached",
            "no_physical_collision",
            "nonnegative_operational_safety_clearance",
        ],
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
        "current_hocbf_enforcement": (
            "inside_policy_qps_or_playground_emergency_projection"
        ),
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
    "RANDOMIZED_EGO_VX_RANGE_MPS",
    "RANDOMIZED_EGO_VY_RANGE_MPS",
    "RANDOMIZED_EGO_X_RANGE_M",
    "RANDOMIZED_EGO_Y_RANGE_M",
    "RANDOMIZED_STRETCHER_SHIFT_RANGE_M",
    "RANDOMIZED_STRETCHER_SPEED_FACTOR_RANGE",
    "STRICT_HOSPITAL_CASES",
    "build_benchmark_scenario",
    "compact_benchmark_config",
    "hospital_randomization_protocol_metadata",
    "run_hospital_benchmark",
    "run_hospital_trial",
]
