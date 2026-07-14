"""Baseline-only randomized Warehouse Quad3D benchmark.

Only MB-CBF-MI and Lib-PCBF-MI are selectable here. Certificate loss and
candidate-QP failure are diagnostics; the simulator always receives the exact
valid control returned by the selected baseline.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "safe_control"))

from safe_control.envs.warehouse_env import WarehouseEnv
import examples.warehouse.additional_baselines_setup_quad as test_quad


@dataclass(frozen=True)
class AlgoSpec:
    key: str
    label: str


@dataclass(frozen=True)
class TrialScenario:
    run_idx: int
    seed: int
    ghosts: Tuple[Tuple[float, float, float, float, float], ...]


@dataclass
class TrialResult:
    collision: bool
    infeasible: bool
    unrecoverable_infeasible: bool
    historical_failure: bool
    reached_goal: bool
    nominal_tracking_pct: float
    solve_time_sum_sec: float
    timed_steps: int
    total_steps: int
    algorithm: str = ""
    seed: int = 0
    run_idx: int = 0
    obstacle_geometry: List[Tuple[float, float, float, float, float]] = field(
        default_factory=list
    )
    p_or_library_size: int = 0
    certificate_lost: bool = False
    qp_infeasible: bool = False
    runtime_error: bool = False
    survived_horizon: bool = False
    task_completed: bool = False
    completed_or_survived: bool = False
    filter_failure: bool = False
    union_failure: bool = False
    certificate_loss_steps: int = 0
    qp_infeasible_steps: int = 0
    fallback_steps: int = 0
    mean_compute_ms: float = float("nan")
    median_compute_ms: float = float("nan")
    p95_compute_ms: float = float("nan")
    max_compute_ms: float = float("nan")
    mean_intervention_l2: float = float("nan")
    max_intervention_l2: float = float("nan")
    nominal_tracking_fraction: float = 0.0
    policy_switch_count: int = 0
    selected_policy_histogram: Dict[str, int] = field(default_factory=dict)
    num_candidate_qps_solved: int = 0
    num_steps_with_no_safe_policy: int = 0
    mean_feasible_backup_candidates: float = float("nan")
    terminal_failure_count: int = 0
    mean_rollout_safe_candidates: float = float("nan")
    mean_qp_feasible_candidates: float = float("nan")
    num_feasible_backup_candidates_per_step: List[int] = field(default_factory=list)
    num_rollout_safe_candidates_per_step: List[int] = field(default_factory=list)
    num_qp_feasible_candidates_per_step: List[int] = field(default_factory=list)
    projection_occurred: bool = False
    num_post_projection_audits: int = 0
    projection_event_count: int = 0
    post_projection_rejection_count: int = 0
    max_projection_delta_inf: float = 0.0
    max_post_projection_constraint_violation: float = 0.0
    max_post_projection_violation_ratio: float = 0.0
    num_post_projection_audits_per_step: List[int] = field(default_factory=list)
    projection_event_count_per_step: List[int] = field(default_factory=list)
    post_projection_rejection_count_per_step: List[int] = field(default_factory=list)
    max_projection_delta_inf_per_step: List[float] = field(default_factory=list)
    max_post_projection_constraint_violation_per_step: List[float] = field(
        default_factory=list
    )
    max_post_projection_violation_ratio_per_step: List[float] = field(
        default_factory=list
    )


@dataclass
class SummaryRow:
    key: str
    label: str
    n_trials: int
    collisions: int
    infeasibles: int
    fail_count: int
    union_failures: int
    certificate_losses: int
    qp_infeasibles: int
    runtime_errors: int
    goal_reaches: int
    task_completions: int
    horizon_survivals: int
    successful_outcomes: int
    filter_failures: int
    collision_rate_pct: float
    infeasible_rate_pct: float
    fail_rate_pct: float
    union_failure_rate_pct: float
    certificate_loss_rate_pct: float
    qp_infeasible_rate_pct: float
    runtime_error_rate_pct: float
    goal_reach_rate_pct: float
    task_completion_rate_pct: float
    horizon_survival_rate_pct: float
    successful_outcome_rate_pct: float
    filter_failure_rate_pct: float
    avg_nominal_tracking_pct: float
    avg_compute_ms: float
    total_timed_steps: int
    projection_trial_count: int
    num_post_projection_audits: int
    projection_event_count: int
    post_projection_rejection_count: int
    max_projection_delta_inf: float
    max_post_projection_constraint_violation: float
    max_post_projection_violation_ratio: float
    library_size: int = 0


ALGO_SPECS: List[AlgoSpec] = [
    AlgoSpec("multi_backup_cbf_mi", "MB-CBF-MI†"),
    AlgoSpec("library_pcbf_mi", "Lib-PCBF-MI"),
]
BASELINE_KEYS = tuple(spec.key for spec in ALGO_SPECS)


def _fmt_count_rate(count: int, total: int) -> str:
    return f"{count}/{total} ({100.0 * count / max(total, 1):.1f}%)"


def _json_safe(value):
    """Convert NumPy/non-finite values to strict JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _solve_with_timing(fn):
    t0 = time.perf_counter()
    try:
        out = fn()
        err = None
    except Exception as exc:  # pragma: no cover - defensive runtime path
        out = None
        err = exc
    dt = time.perf_counter() - t0
    return out, dt, err


def _step_failure_events(step_metrics: dict, shielding) -> Tuple[bool, bool, bool]:
    """Split certificate loss, selected-QP failure, and fallback use.

    The additive methods expose separate no-certificate/no-feasible-QP
    counters.  Prefer those over their legacy aggregate ``infeasible`` flag so
    an empty certified set is not double-counted as QP infeasibility.
    """

    no_certificate = step_metrics.get("num_steps_with_no_certified_rollout")
    if no_certificate is None:
        no_certificate = step_metrics.get("num_steps_with_no_safe_policy")
    if no_certificate is None:
        certificate_lost = bool(
            step_metrics.get(
                "certificate_lost",
                getattr(shielding, "certificate_lost", False),
            )
        )
    else:
        certificate_lost = int(no_certificate or 0) > 0

    no_feasible_qp = step_metrics.get("num_steps_with_no_feasible_qp")
    if no_feasible_qp is None:
        qp_infeasible = bool(step_metrics.get("qp_infeasible", False))
    else:
        qp_infeasible = int(no_feasible_qp or 0) > 0
    # Under the reported definition, an empty certified set is certificate
    # loss, not QP infeasibility—even if the controller's damage-mitigation QP
    # also happens to fail on that same step.
    if certificate_lost:
        qp_infeasible = False

    fallback_used = bool(
        step_metrics.get("fallback_used", False)
        or step_metrics.get("emergency_action_used", False)
    )
    return certificate_lost, qp_infeasible, fallback_used


def validate_returned_control(
    control,
    robot_spec: Dict[str, float],
    *,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Validate a native controller output without clipping or substitution."""

    vector = np.asarray(control, dtype=float).reshape(-1)
    if vector.shape != (4,):
        raise ValueError(f"controller returned shape {vector.shape}; expected (4,)")
    if not np.all(np.isfinite(vector)):
        raise ValueError("controller returned a non-finite control")
    lower = np.broadcast_to(np.asarray(robot_spec["u_min"], dtype=float), vector.shape)
    upper = np.broadcast_to(np.asarray(robot_spec["u_max"], dtype=float), vector.shape)
    if np.any(vector < lower - tolerance) or np.any(vector > upper + tolerance):
        raise ValueError("controller returned an out-of-bounds control")
    return vector


def _sample_velocity(rng: np.random.Generator, speed_min: float, speed_max: float) -> Tuple[float, float]:
    speed = float(rng.uniform(speed_min, speed_max))
    mode = float(rng.random())

    # Mostly axis-aligned motion to mimic level-7 cross-flow style,
    # with some diagonal/random movers for variability.
    if mode < 0.45:
        vx = speed if rng.random() < 0.5 else -speed
        vy = float(rng.uniform(-0.25, 0.25))
    elif mode < 0.90:
        vx = float(rng.uniform(-0.25, 0.25))
        vy = speed if rng.random() < 0.5 else -speed
    else:
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)

    return float(vx), float(vy)


def generate_random_scenarios(
    *,
    level: int,
    num_trials: int,
    seed: int,
    num_dynamic_obstacles: int,
    ghost_radius: float,
    speed_min: float,
    speed_max: float,
    start_exclusion_max_x: float,
    start_exclusion_max_y: float,
    start_clearance_radius: float,
    inter_ghost_clearance: float,
) -> List[TrialScenario]:
    """
    Generate randomized dynamic-obstacle scenarios.

    Static obstacles and waypoints are not randomized.
    Dynamic obstacles are kept away from the initial robot area to avoid immediate failure.
    """
    env = WarehouseEnv(level=level)
    static_obs = env.get_static_obstacles()
    start_pos = np.array(env.start_pos, dtype=float)

    x_min, x_max = 3.0, 97.0
    y_min, y_max = 3.0, 97.0

    rng = np.random.default_rng(seed)
    scenarios: List[TrialScenario] = []

    for run_idx in range(num_trials):
        ghosts: List[Tuple[float, float, float, float, float]] = []

        for _ in range(num_dynamic_obstacles):
            placed = False
            for _attempt in range(2000):
                x = float(rng.uniform(x_min, x_max))
                y = float(rng.uniform(y_min, y_max))

                # Keep a square near the start free (user requested).
                if x <= start_exclusion_max_x and y <= start_exclusion_max_y:
                    continue

                # Additional radial clearance from the initial state.
                if np.linalg.norm(np.array([x, y]) - start_pos) < start_clearance_radius:
                    continue

                # Avoid spawning inside/too close to static obstacles.
                blocked_by_static = False
                for obs in static_obs:
                    dist = np.hypot(x - float(obs["x"]), y - float(obs["y"]))
                    min_dist = float(obs["radius"]) + ghost_radius + 0.2
                    if dist < min_dist:
                        blocked_by_static = True
                        break
                if blocked_by_static:
                    continue

                # Keep some spacing among dynamic obstacles.
                blocked_by_ghost = False
                for gx, gy, _, _, gr in ghosts:
                    dist = np.hypot(x - gx, y - gy)
                    if dist < (ghost_radius + gr + inter_ghost_clearance):
                        blocked_by_ghost = True
                        break
                if blocked_by_ghost:
                    continue

                vx, vy = _sample_velocity(rng, speed_min=speed_min, speed_max=speed_max)
                ghosts.append((x, y, vx, vy, ghost_radius))
                placed = True
                break

            if not placed:
                raise RuntimeError(
                    "Failed to place dynamic obstacles without violating initial safety. "
                    f"run_idx={run_idx}, placed={len(ghosts)}, target={num_dynamic_obstacles}"
                )

        scenarios.append(
            TrialScenario(
                run_idx=run_idx,
                seed=int(rng.integers(0, 2**31 - 1)),
                ghosts=tuple(ghosts),
            )
        )

    return scenarios


def _apply_scenario_to_env(env: WarehouseEnv, scenario: TrialScenario):
    env.ghosts = [
        {
            "x": float(x),
            "y": float(y),
            "vx": float(vx),
            "vy": float(vy),
            "radius": float(r),
        }
        for x, y, vx, vy, r in scenario.ghosts
    ]


def _build_initial_state(env: WarehouseEnv, robot_spec: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            env.start_pos[0],
            env.start_pos[1],
            robot_spec["z_ref"],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def run_trial(
    *,
    algo: str,
    scenario: TrialScenario,
    level: int,
    safety_margin: float,
    alpha: float,
    max_steps: int,
    jit_warmup_steps: int,
    tracking_tol: float,
    num_angle_policies: int,
) -> TrialResult:
    if algo not in BASELINE_KEYS:
        raise ValueError(f"Unsupported baseline {algo!r}; valid keys are {BASELINE_KEYS}")

    env, robot, nom_ctrl, shielding, robot_spec, _ = test_quad.setup_test(
        algo=algo,
        level=level,
        safety_margin=safety_margin,
        num_angle_policies=num_angle_policies,
        alpha=alpha,
    )
    _apply_scenario_to_env(env, scenario)
    current_state = _build_initial_state(env, robot_spec)

    collision = False
    infeasible = False
    qp_infeasible = False
    runtime_error = False
    reached_goal = False
    certificate_lost = False
    certificate_loss_steps = 0
    qp_infeasible_steps = 0
    fallback_steps = 0
    nominal_track_steps = 0
    total_steps = 0
    solve_time_sum = 0.0
    timed_steps = 0
    solve_times_sec: List[float] = []
    intervention_l2_values: List[float] = []
    selected_policy_histogram: Counter = Counter()
    policy_switch_count = 0
    num_candidate_qps_solved = 0
    num_steps_with_no_safe_policy = 0
    feasible_backup_counts: List[float] = []
    terminal_failure_count = 0
    rollout_safe_counts: List[float] = []
    qp_feasible_counts: List[float] = []
    num_post_projection_audits = 0
    projection_event_count = 0
    post_projection_rejection_count = 0
    max_projection_delta_inf = 0.0
    max_post_projection_constraint_violation = 0.0
    max_post_projection_violation_ratio = 0.0
    num_post_projection_audits_per_step: List[int] = []
    projection_event_count_per_step: List[int] = []
    post_projection_rejection_count_per_step: List[int] = []
    max_projection_delta_inf_per_step: List[float] = []
    max_post_projection_constraint_violation_per_step: List[float] = []
    max_post_projection_violation_ratio_per_step: List[float] = []

    for step in range(max_steps):
        try:
            env.step()
            ghosts = env.get_dynamic_obstacles()
            statics = env.get_static_obstacles()
            shielding.update_obstacles(ghosts, statics)
            u_nom = np.asarray(
                nom_ctrl.get_control(current_state), dtype=float
            ).reshape(-1)
            if u_nom.shape != (4,) or not np.all(np.isfinite(u_nom)):
                raise ValueError("nominal controller returned an invalid control")
            control_ref = {
                "u_ref": u_nom,
                "waypoints": nom_ctrl.waypoints,
                "wp_idx": nom_ctrl.wp_idx,
            }
        except Exception:
            runtime_error = True
            infeasible = True
            break

        u_safe, solve_dt, solve_err = _solve_with_timing(
            lambda: shielding.solve_control_problem(current_state, control_ref)
        )
        if solve_err is not None:
            runtime_error = True
            infeasible = True
            break

        try:
            step_metrics = (
                dict(shielding.get_last_step_metrics())
                if hasattr(shielding, "get_last_step_metrics")
                else {}
            )
        except Exception:
            runtime_error = True
            infeasible = True
            break

        if bool(step_metrics.get("runtime_error", False)) or bool(
            getattr(shielding, "runtime_error", False)
        ):
            runtime_error = True
            infeasible = True
            break

        try:
            u_safe_vec = validate_returned_control(u_safe, robot_spec)
        except (TypeError, ValueError):
            runtime_error = True
            infeasible = True
            break

        (
            step_certificate_lost,
            step_qp_infeasible,
            step_fallback_used,
        ) = _step_failure_events(step_metrics, shielding)

        try:
            next_state = robot.step(
                current_state.reshape(-1, 1), u_safe_vec.reshape(-1, 1)
            )
            current_state = np.asarray(next_state, dtype=float).reshape(-1)
            if current_state.shape != (12,) or not np.all(np.isfinite(current_state)):
                raise ValueError("simulator returned an invalid state")
            env.robot_pos = current_state[:2]
        except Exception:
            runtime_error = True
            infeasible = True
            break

        certificate_lost = certificate_lost or step_certificate_lost
        qp_infeasible = qp_infeasible or step_qp_infeasible
        certificate_loss_steps += int(step_certificate_lost)
        qp_infeasible_steps += int(step_qp_infeasible)
        fallback_steps += int(step_fallback_used)

        selected_policy = step_metrics.get("selected_policy")
        if selected_policy is not None:
            selected_policy_histogram[str(selected_policy)] += 1
        policy_switch_count += int(bool(step_metrics.get("policy_switched", False)))
        num_candidate_qps_solved += int(
            step_metrics.get("num_candidate_qps_solved", 0) or 0
        )
        num_steps_with_no_safe_policy += int(
            step_metrics.get("num_steps_with_no_safe_policy", 0) or 0
        )
        if "num_feasible_backup_candidates" in step_metrics:
            feasible_backup_counts.append(
                float(step_metrics["num_feasible_backup_candidates"])
            )
        terminal_failure_count += int(
            step_metrics.get("terminal_failure_count", 0) or 0
        )
        rollout_count = step_metrics.get(
            "num_rollout_safe_candidates",
            step_metrics.get("num_safe_candidates"),
        )
        if rollout_count is not None:
            rollout_safe_counts.append(float(rollout_count))
        if "num_qp_feasible_candidates" in step_metrics:
            qp_feasible_counts.append(
                float(step_metrics["num_qp_feasible_candidates"])
            )
        step_audits = int(step_metrics.get("num_post_projection_audits", 0) or 0)
        step_projection_events = int(
            step_metrics.get("projection_event_count", 0) or 0
        )
        step_projection_rejections = int(
            step_metrics.get("post_projection_rejection_count", 0) or 0
        )
        step_max_delta = float(
            step_metrics.get("max_projection_delta_inf", 0.0) or 0.0
        )
        step_max_violation = float(
            step_metrics.get(
                "max_post_projection_constraint_violation", 0.0
            )
            or 0.0
        )
        step_max_ratio = float(
            step_metrics.get("max_post_projection_violation_ratio", 0.0) or 0.0
        )
        num_post_projection_audits += step_audits
        projection_event_count += step_projection_events
        post_projection_rejection_count += step_projection_rejections
        max_projection_delta_inf = max(max_projection_delta_inf, step_max_delta)
        max_post_projection_constraint_violation = max(
            max_post_projection_constraint_violation, step_max_violation
        )
        max_post_projection_violation_ratio = max(
            max_post_projection_violation_ratio, step_max_ratio
        )
        num_post_projection_audits_per_step.append(step_audits)
        projection_event_count_per_step.append(step_projection_events)
        post_projection_rejection_count_per_step.append(step_projection_rejections)
        max_projection_delta_inf_per_step.append(step_max_delta)
        max_post_projection_constraint_violation_per_step.append(step_max_violation)
        max_post_projection_violation_ratio_per_step.append(step_max_ratio)

        intervention_l2_values.append(float(np.linalg.norm(u_safe_vec - u_nom)))
        if np.linalg.norm(u_safe_vec - u_nom) < tracking_tol:
            nominal_track_steps += 1
        total_steps += 1
        if step >= jit_warmup_steps:
            solve_times_sec.append(float(solve_dt))
            solve_time_sum += float(solve_dt)
            timed_steps += 1

        try:
            for obs in statics:
                distance = np.linalg.norm(
                    current_state[:2] - np.array([obs["x"], obs["y"]])
                )
                if distance < (obs["radius"] + robot_spec["radius"]):
                    collision = True
                    break
            if not collision:
                for ghost in ghosts:
                    distance = np.linalg.norm(
                        current_state[:2] - np.array([ghost["x"], ghost["y"]])
                    )
                    if distance < (ghost["radius"] + robot_spec["radius"]):
                        collision = True
                        break
            if not collision and (
                np.linalg.norm(current_state[:2] - env.goal_pos) < env.goal_radius
            ):
                reached_goal = True
        except Exception:
            runtime_error = True
            infeasible = True
            break

        if collision or reached_goal:
            break

    survived_horizon = bool(
        total_steps == max_steps and not collision and not runtime_error
    )
    task_completed = bool(reached_goal)
    completed_or_survived = bool(task_completed or survived_horizon)
    filter_failure = bool(collision or certificate_lost or qp_infeasible)
    union_failure = bool(
        filter_failure or runtime_error or not completed_or_survived
    )
    infeasible = bool(infeasible or runtime_error)
    historical_failure = bool(collision or infeasible)
    nominal_tracking_pct = 100.0 * nominal_track_steps / max(total_steps, 1)

    solve_times_array = np.asarray(solve_times_sec, dtype=float)
    if solve_times_array.size:
        mean_compute_ms = 1000.0 * float(np.mean(solve_times_array))
        median_compute_ms = 1000.0 * float(np.median(solve_times_array))
        p95_compute_ms = 1000.0 * float(np.percentile(solve_times_array, 95))
        max_compute_ms = 1000.0 * float(np.max(solve_times_array))
    else:
        mean_compute_ms = median_compute_ms = p95_compute_ms = max_compute_ms = float("nan")

    intervention_array = np.asarray(intervention_l2_values, dtype=float)
    mean_intervention_l2 = (
        float(np.mean(intervention_array)) if intervention_array.size else float("nan")
    )
    max_intervention_l2 = (
        float(np.max(intervention_array)) if intervention_array.size else float("nan")
    )
    library_size = len(getattr(shielding, "policy_configs", {}))

    return TrialResult(
        collision=collision,
        infeasible=infeasible,
        unrecoverable_infeasible=infeasible,
        historical_failure=historical_failure,
        reached_goal=reached_goal,
        nominal_tracking_pct=nominal_tracking_pct,
        solve_time_sum_sec=solve_time_sum,
        timed_steps=timed_steps,
        total_steps=total_steps,
        algorithm=algo,
        seed=scenario.seed,
        run_idx=scenario.run_idx,
        obstacle_geometry=[tuple(item) for item in scenario.ghosts],
        p_or_library_size=library_size,
        certificate_lost=certificate_lost,
        qp_infeasible=qp_infeasible,
        runtime_error=runtime_error,
        survived_horizon=survived_horizon,
        task_completed=task_completed,
        completed_or_survived=completed_or_survived,
        filter_failure=filter_failure,
        union_failure=union_failure,
        certificate_loss_steps=certificate_loss_steps,
        qp_infeasible_steps=qp_infeasible_steps,
        fallback_steps=fallback_steps,
        mean_compute_ms=mean_compute_ms,
        median_compute_ms=median_compute_ms,
        p95_compute_ms=p95_compute_ms,
        max_compute_ms=max_compute_ms,
        mean_intervention_l2=mean_intervention_l2,
        max_intervention_l2=max_intervention_l2,
        nominal_tracking_fraction=nominal_tracking_pct / 100.0,
        policy_switch_count=policy_switch_count,
        selected_policy_histogram=dict(selected_policy_histogram),
        num_candidate_qps_solved=num_candidate_qps_solved,
        num_steps_with_no_safe_policy=num_steps_with_no_safe_policy,
        mean_feasible_backup_candidates=(
            float(np.mean(feasible_backup_counts))
            if feasible_backup_counts
            else float("nan")
        ),
        terminal_failure_count=terminal_failure_count,
        mean_rollout_safe_candidates=(
            float(np.mean(rollout_safe_counts))
            if rollout_safe_counts
            else float("nan")
        ),
        mean_qp_feasible_candidates=(
            float(np.mean(qp_feasible_counts))
            if qp_feasible_counts
            else float("nan")
        ),
        num_feasible_backup_candidates_per_step=[
            int(value) for value in feasible_backup_counts
        ],
        num_rollout_safe_candidates_per_step=[
            int(value) for value in rollout_safe_counts
        ],
        num_qp_feasible_candidates_per_step=[
            int(value) for value in qp_feasible_counts
        ],
        projection_occurred=bool(projection_event_count > 0),
        num_post_projection_audits=num_post_projection_audits,
        projection_event_count=projection_event_count,
        post_projection_rejection_count=post_projection_rejection_count,
        max_projection_delta_inf=max_projection_delta_inf,
        max_post_projection_constraint_violation=(
            max_post_projection_constraint_violation
        ),
        max_post_projection_violation_ratio=(
            max_post_projection_violation_ratio
        ),
        num_post_projection_audits_per_step=num_post_projection_audits_per_step,
        projection_event_count_per_step=projection_event_count_per_step,
        post_projection_rejection_count_per_step=(
            post_projection_rejection_count_per_step
        ),
        max_projection_delta_inf_per_step=max_projection_delta_inf_per_step,
        max_post_projection_constraint_violation_per_step=(
            max_post_projection_constraint_violation_per_step
        ),
        max_post_projection_violation_ratio_per_step=(
            max_post_projection_violation_ratio_per_step
        ),
    )


def summarize_trials(algo_spec: AlgoSpec, trials: List[TrialResult]) -> SummaryRow:
    n = len(trials)
    collisions = sum(int(t.collision) for t in trials)
    infeasibles = sum(int(t.infeasible) for t in trials)
    fail_count = sum(int(t.historical_failure) for t in trials)
    union_failures = sum(int(t.union_failure) for t in trials)
    certificate_losses = sum(int(t.certificate_lost) for t in trials)
    qp_infeasibles = sum(int(t.qp_infeasible) for t in trials)
    runtime_errors = sum(int(t.runtime_error) for t in trials)
    goal_reaches = sum(int(t.reached_goal) for t in trials)
    task_completions = sum(int(t.task_completed) for t in trials)
    horizon_survivals = sum(int(t.survived_horizon) for t in trials)
    successful_outcomes = sum(int(t.completed_or_survived) for t in trials)
    filter_failures = sum(int(t.filter_failure) for t in trials)
    projection_trial_count = sum(int(t.projection_occurred) for t in trials)
    num_post_projection_audits = sum(
        t.num_post_projection_audits for t in trials
    )
    projection_event_count = sum(t.projection_event_count for t in trials)
    post_projection_rejection_count = sum(
        t.post_projection_rejection_count for t in trials
    )
    max_projection_delta_inf = max(
        (t.max_projection_delta_inf for t in trials), default=0.0
    )
    max_post_projection_constraint_violation = max(
        (t.max_post_projection_constraint_violation for t in trials), default=0.0
    )
    max_post_projection_violation_ratio = max(
        (t.max_post_projection_violation_ratio for t in trials), default=0.0
    )

    nominal_vals = [t.nominal_tracking_pct for t in trials]
    avg_nominal = float(np.mean(nominal_vals)) if nominal_vals else 0.0

    total_solve_sec = float(np.sum([t.solve_time_sum_sec for t in trials]))
    total_timed_steps = int(np.sum([t.timed_steps for t in trials]))
    avg_compute_ms = 1000.0 * total_solve_sec / max(total_timed_steps, 1)

    return SummaryRow(
        key=algo_spec.key,
        label=algo_spec.label,
        n_trials=n,
        collisions=collisions,
        infeasibles=infeasibles,
        fail_count=fail_count,
        union_failures=union_failures,
        certificate_losses=certificate_losses,
        qp_infeasibles=qp_infeasibles,
        runtime_errors=runtime_errors,
        goal_reaches=goal_reaches,
        task_completions=task_completions,
        horizon_survivals=horizon_survivals,
        successful_outcomes=successful_outcomes,
        filter_failures=filter_failures,
        collision_rate_pct=100.0 * collisions / max(n, 1),
        infeasible_rate_pct=100.0 * infeasibles / max(n, 1),
        fail_rate_pct=100.0 * fail_count / max(n, 1),
        union_failure_rate_pct=100.0 * union_failures / max(n, 1),
        certificate_loss_rate_pct=100.0 * certificate_losses / max(n, 1),
        qp_infeasible_rate_pct=100.0 * qp_infeasibles / max(n, 1),
        runtime_error_rate_pct=100.0 * runtime_errors / max(n, 1),
        goal_reach_rate_pct=100.0 * goal_reaches / max(n, 1),
        task_completion_rate_pct=100.0 * task_completions / max(n, 1),
        horizon_survival_rate_pct=100.0 * horizon_survivals / max(n, 1),
        successful_outcome_rate_pct=100.0 * successful_outcomes / max(n, 1),
        filter_failure_rate_pct=100.0 * filter_failures / max(n, 1),
        avg_nominal_tracking_pct=avg_nominal,
        avg_compute_ms=avg_compute_ms,
        total_timed_steps=total_timed_steps,
        projection_trial_count=projection_trial_count,
        num_post_projection_audits=num_post_projection_audits,
        projection_event_count=projection_event_count,
        post_projection_rejection_count=post_projection_rejection_count,
        max_projection_delta_inf=max_projection_delta_inf,
        max_post_projection_constraint_violation=(
            max_post_projection_constraint_violation
        ),
        max_post_projection_violation_ratio=(
            max_post_projection_violation_ratio
        ),
        library_size=max((t.p_or_library_size for t in trials), default=0),
    )


def run_algorithm_trials(
    algo_spec: AlgoSpec,
    scenarios: List[TrialScenario],
    args: argparse.Namespace,
    *,
    verbose: bool,
) -> Tuple[List[TrialResult], SummaryRow]:
    if algo_spec.key not in BASELINE_KEYS:
        raise ValueError(f"Unsupported baseline: {algo_spec.key}")

    payloads = [
        (
            algo_spec.key,
            scenario,
            args.level,
            args.safety_margin,
            args.alpha,
            args.max_steps,
            args.jit_warmup_steps,
            args.tracking_tol,
            args.num_angle_policies,
        )
        for scenario in scenarios
    ]
    trials: List[TrialResult] = []
    started = time.perf_counter()
    worker_count = max(1, int(args.num_workers))
    if worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            iterator = executor.map(_run_trial_payload, payloads)
            for index, result in enumerate(iterator, start=1):
                trials.append(result)
                if verbose and index % max(1, args.progress_every) == 0:
                    print(
                        f"  {algo_spec.key:22s} trial {index:3d}/{len(scenarios)} "
                        f"failure={int(result.historical_failure)} "
                        f"certificate={int(result.certificate_lost)} "
                        f"qp={int(result.qp_infeasible)}"
                    )
    else:
        for index, payload in enumerate(payloads, start=1):
            result = _run_trial_payload(payload)
            trials.append(result)
            if verbose and index % max(1, args.progress_every) == 0:
                print(
                    f"  {algo_spec.key:22s} trial {index:3d}/{len(scenarios)} "
                    f"failure={int(result.historical_failure)} "
                    f"certificate={int(result.certificate_lost)} "
                    f"qp={int(result.qp_infeasible)}"
                )

    summary = summarize_trials(algo_spec, trials)
    elapsed = time.perf_counter() - started
    print(
        f"[Done] {algo_spec.label:<28} "
        f"failure={_fmt_count_rate(summary.fail_count, summary.n_trials)} "
        f"collision={_fmt_count_rate(summary.collisions, summary.n_trials)} "
        f"certificate={_fmt_count_rate(summary.certificate_losses, summary.n_trials)} "
        f"qp={_fmt_count_rate(summary.qp_infeasibles, summary.n_trials)} "
        f"avg_compute={summary.avg_compute_ms:.3f} ms "
        f"elapsed={elapsed / 60.0:.1f} min"
    )
    return trials, summary


def _run_trial_payload(payload) -> TrialResult:
    """Pickle-friendly adapter for independent seeded trials."""

    (
        algo,
        scenario,
        level,
        safety_margin,
        alpha,
        max_steps,
        jit_warmup_steps,
        tracking_tol,
        num_angle_policies,
    ) = payload
    return run_trial(
        algo=algo,
        scenario=scenario,
        level=level,
        safety_margin=safety_margin,
        alpha=alpha,
        max_steps=max_steps,
        jit_warmup_steps=jit_warmup_steps,
        tracking_tol=tracking_tol,
        num_angle_policies=num_angle_policies,
    )


def format_markdown(
    *,
    summaries: List[SummaryRow],
    args: argparse.Namespace,
    scenario_seed: int,
) -> str:
    lines = [
        "# Warehouse Quad3D Additional-Baseline Results",
        "",
        f"- Level layout: {args.level} (static obstacles and waypoints fixed)",
        f"- Trials per algorithm: {args.num_trials}",
        f"- Scenario seed: {scenario_seed}",
        f"- Dynamic obstacles per trial: {args.num_dynamic_obstacles}",
        f"- Max steps per trial: {args.max_steps}",
        f"- Safety margin: {args.safety_margin:.2f}",
        (
            f"- Policy library: P={args.num_angle_policies} angle policies + stop + "
            f"nominal = P+2 = {args.num_angle_policies + 2}"
        ),
        "- Main failure: collision OR unrecoverable infeasibility/runtime failure",
        (
            "- Certificate loss and candidate-QP failure are diagnostics only. "
            "The simulator applies the selected baseline's returned control unchanged."
        ),
        (
            f"- Timing excludes the first {args.jit_warmup_steps} filter calls and "
            f"uses {args.num_workers} worker(s). Publication timing uses one worker."
        ),
        (
            "- † MB-CBF-MI uses a sampled terminal proxy, not a proven "
            "control-invariant terminal set, and therefore does not inherit the "
            "formal guarantee of Chen et al."
        ),
        "",
        (
            "| Algorithm | P | Library size | Failure (historical) | Collision | "
            "Unrecoverable infeasible | Certificate loss | Candidate-QP failure | "
            "Goal | Horizon survival | Avg Compute Time (ms) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.label} | {args.num_angle_policies} | "
            f"{summary.library_size or args.num_angle_policies + 2} | "
            f"{_fmt_count_rate(summary.fail_count, summary.n_trials)} | "
            f"{_fmt_count_rate(summary.collisions, summary.n_trials)} | "
            f"{_fmt_count_rate(summary.infeasibles, summary.n_trials)} | "
            f"{_fmt_count_rate(summary.certificate_losses, summary.n_trials)} | "
            f"{_fmt_count_rate(summary.qp_infeasibles, summary.n_trials)} | "
            f"{_fmt_count_rate(summary.task_completions, summary.n_trials)} | "
            f"{_fmt_count_rate(summary.horizon_survivals, summary.n_trials)} | "
            f"{summary.avg_compute_ms:.3f} |"
        )
    lines.append("")
    lines.extend(
        [
            "## Post-projection QP audit",
            "",
            (
                "Every finite, actuator-tolerance-valid successful-status candidate "
                "is checked against its original QP inequalities after actuator-bound "
                "projection. A residual above the declared post-projection audit "
                "tolerance rejects that candidate before minimum-intervention "
                "selection."
            ),
            "",
            (
                "| Algorithm | Audited candidates | Trials with projection | "
                "Projection events (candidates) | Audit rejections | Max "
                "$\\|\\Delta u\\|_\\infty$ (native units) | Max violation | Max "
                "violation/tolerance |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        lines.append(
            f"| {summary.label} | {summary.num_post_projection_audits} | "
            f"{summary.projection_trial_count}/{summary.n_trials} | "
            f"{summary.projection_event_count} | "
            f"{summary.post_projection_rejection_count} | "
            f"{summary.max_projection_delta_inf:.9g} | "
            f"{summary.max_post_projection_constraint_violation:.9g} | "
            f"{summary.max_post_projection_violation_ratio:.9g} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run only the two additional Warehouse Quad3D baselines"
    )
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--level", type=int, default=7)
    parser.add_argument("--num-dynamic-obstacles", type=int, default=45)
    parser.add_argument("--safety-margin", type=float, default=1.3)
    parser.add_argument("--alpha", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=350)
    parser.add_argument("--num-angle-policies", type=int, default=64)
    parser.add_argument("--jit-warmup-steps", type=int, default=10)
    parser.add_argument("--tracking-tol", type=float, default=0.1)
    parser.add_argument("--speed-min", type=float, default=3.0)
    parser.add_argument("--speed-max", type=float, default=4.5)
    parser.add_argument("--ghost-radius", type=float, default=2.4)
    parser.add_argument("--inter-ghost-clearance", type=float, default=0.2)
    parser.add_argument("--start-exclusion-max-x", type=float, default=18.0)
    parser.add_argument("--start-exclusion-max-y", type=float, default=18.0)
    parser.add_argument("--start-clearance-radius", type=float, default=8.0)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=BASELINE_KEYS,
        default=None,
        help="Run a selected subset; by default both additional baselines run.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output-md",
        default="output/additional_baselines/warehouse_additional_baselines.md",
    )
    parser.add_argument(
        "--output-json",
        default="output/additional_baselines/warehouse_additional_baselines.json",
    )
    parser.add_argument(
        "--output-csv",
        default="output/additional_baselines/warehouse_additional_baselines.csv",
    )
    args = parser.parse_args(argv)

    warnings.filterwarnings(
        "ignore", message="Solution may be inaccurate.*", module="cvxpy"
    )
    requested = set(args.algorithms or BASELINE_KEYS)
    algo_specs = [spec for spec in ALGO_SPECS if spec.key in requested]
    if {spec.key for spec in algo_specs} != requested:
        raise ValueError(f"Unsupported baseline selection: {sorted(requested)}")

    scenarios = generate_random_scenarios(
        level=args.level,
        num_trials=args.num_trials,
        seed=args.seed,
        num_dynamic_obstacles=args.num_dynamic_obstacles,
        ghost_radius=args.ghost_radius,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        start_exclusion_max_x=args.start_exclusion_max_x,
        start_exclusion_max_y=args.start_exclusion_max_y,
        start_clearance_radius=args.start_clearance_radius,
        inter_ghost_clearance=args.inter_ghost_clearance,
    )

    all_trial_results: Dict[str, List[TrialResult]] = {}
    summaries: List[SummaryRow] = []
    for algo_spec in algo_specs:
        print(f"Running {algo_spec.label}...")
        trials, summary = run_algorithm_trials(
            algo_spec, scenarios, args, verbose=args.verbose
        )
        all_trial_results[algo_spec.key] = trials
        summaries.append(summary)

    markdown = format_markdown(
        summaries=summaries,
        args=args,
        scenario_seed=args.seed,
    )

    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    if not output_md.is_absolute():
        output_md = Path(PROJECT_ROOT) / output_md
    if not output_json.is_absolute():
        output_json = Path(PROJECT_ROOT) / output_json
    if not output_csv.is_absolute():
        output_csv = Path(PROJECT_ROOT) / output_csv
    for path in (output_md, output_json, output_csv):
        path.parent.mkdir(parents=True, exist_ok=True)

    output_md.write_text(markdown, encoding="utf-8")
    payload = {
        "algorithms": [spec.key for spec in algo_specs],
        "config": vars(args),
        "scenario_seed": args.seed,
        "failure_semantics": {
            "historical_failure": (
                "collision OR unrecoverable infeasibility/runtime failure"
            ),
            "unrecoverable_infeasible": (
                "solve exception, runtime error, non-finite control, invalid "
                "control dimension, out-of-bounds control, or simulator failure"
            ),
            "certificate_loss": "no policy has a positive rollout certificate",
            "qp_infeasible": (
                "a certified policy exists but no candidate QP returns an "
                "accepted bounded input"
            ),
            "filter_failure": (
                "collision OR certificate_loss OR qp_infeasible; diagnostic only"
            ),
            "post_diagnostic_action": (
                "exact control returned by the selected baseline"
            ),
        },
        "policy_library": {
            "angle_policy_count": args.num_angle_policies,
            "extra_entries": ["stop", "nominal"],
            "library_size": args.num_angle_policies + 2,
        },
        "timing": {
            "worker_count": max(1, int(args.num_workers)),
            "warmup_calls_excluded": args.jit_warmup_steps,
            "scope": "solve_control_problem wall-clock time",
        },
        "projection_audit": {
            "scope": (
                "every finite, actuator-tolerance-valid successful-status candidate QP"
            ),
            "action": (
                "project to exact actuator bounds, evaluate every original affine "
                "QP inequality, and reject the candidate if any scale-aware "
                "residual exceeds its declared tolerance"
            ),
            "osqp_absolute_tolerance": 1e-5,
            "osqp_relative_tolerance": 1e-5,
            "multi_backup_scs_fallback_absolute_tolerance": 1e-4,
            "multi_backup_scs_fallback_relative_tolerance": 1e-4,
        },
        "summaries": [asdict(summary) for summary in summaries],
        "trials": {
            key: [asdict(trial) for trial in trials]
            for key, trials in all_trial_results.items()
        },
    }
    output_json.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    rows = []
    for trials in all_trial_results.values():
        for trial in trials:
            row = asdict(trial)
            for key, value in list(row.items()):
                if isinstance(value, (dict, list, tuple)):
                    row[key] = json.dumps(_json_safe(value), sort_keys=True)
                elif isinstance(value, float) and not np.isfinite(value):
                    row[key] = ""
            rows.append(row)
    if rows:
        with output_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(rows[0].keys()),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    print("\n" + markdown)
    print(f"Saved markdown report: {output_md}")
    print(f"Saved json report: {output_json}")
    print(f"Saved per-trial CSV: {output_csv}")


if __name__ == "__main__":
    main()
