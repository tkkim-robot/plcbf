"""
Randomized benchmark for Warehouse Quad3D scenario.

This benchmark keeps the Warehouse level layout (static obstacles + waypoints) fixed,
randomizes only dynamic obstacles, and evaluates six algorithms over many trials.

Metrics:
- Collision rate
- Infeasible rate
- Average nominal tracking percentage
- Average safety-filter compute time

Compute-time workflow:
1) Run full benchmark table.
2) Re-run each algorithm one-by-one to refresh compute-time numbers.
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

import jax.numpy as jnp
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "safe_control"))

from safe_control.envs.warehouse_env import WarehouseEnv
import examples.warehouse.test_warehouse_quad as test_quad
from examples.warehouse.controllers.policies_quad3d_jax import (
    RetracePolicyParams,
    StopPolicyJAX,
)


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
    library_size: int = 0


ALGO_SPECS: List[AlgoSpec] = [
    AlgoSpec("plcbf", "PLCBF"),
    AlgoSpec("pcbf", "PCBF (Retrace Backup)"),
    AlgoSpec("gatekeeper", "Gatekeeper (Retrace Backup)"),
    AlgoSpec("mps", "MPS (Retrace Backup)"),
    AlgoSpec("backup_cbf", "Backup CBF (Retrace Backup)"),
    AlgoSpec("mip_mpc", "MIP MPC"),
]

# Additive comparison rows.  They are registered for explicit selection but
# are not added to the historical default run, so invoking the benchmark with
# no new flags preserves the existing algorithms and output exactly.
ADDITIONAL_ALGO_SPECS: List[AlgoSpec] = [
    AlgoSpec("multi_backup_cbf_mi", "MB-CBF-MI"),
    AlgoSpec("library_pcbf_mi", "Lib-PCBF-MI"),
]
ALL_ALGO_SPECS: List[AlgoSpec] = ALGO_SPECS + ADDITIONAL_ALGO_SPECS


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


ADDITIONAL_COMPARISON_ALGOS = frozenset(
    {"multi_backup_cbf_mi", "library_pcbf_mi"}
)


def _comparison_control_enabled(algo: str, comparison_mode: bool) -> bool:
    """Scope the shared continuation rule to explicit comparison runs."""

    return bool(
        algo in ADDITIONAL_COMPARISON_ALGOS
        or (comparison_mode and algo == "plcbf")
    )


def _common_stop_control(shielding, state: np.ndarray) -> np.ndarray:
    """Return the exact stop entry shared by all three library controllers."""

    policy_configs = getattr(shielding, "policy_configs", {})
    if "stop" not in policy_configs:
        raise RuntimeError("comparison controller has no shared stop policy")
    policy_type, params = policy_configs["stop"]
    if policy_type != "stop":
        raise RuntimeError("comparison controller's stop entry changed type")
    control = np.asarray(
        StopPolicyJAX.compute(jnp.asarray(state), params), dtype=float
    ).reshape(-1)
    lower = float(params.ctrl.u_min)
    upper = float(params.ctrl.u_max)
    if control.shape != (4,) or not np.all(np.isfinite(control)):
        raise RuntimeError("shared stop policy returned an invalid input")
    return np.clip(control, lower, upper)


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
        or certificate_lost
        or qp_infeasible
    )
    return certificate_lost, qp_infeasible, fallback_used


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
    plcbf_num_angle_policies: int,
    mip_num_angle_policies: int,
    comparison_mode: bool = False,
) -> TrialResult:
    env, robot, nom_ctrl, shielding, robot_spec, ctrl_params = test_quad.setup_test(
        algo=algo,
        level=level,
        safety_margin=safety_margin,
        plcbf_num_angle_policies=plcbf_num_angle_policies,
        mip_num_angle_policies=mip_num_angle_policies,
        alpha=alpha,
    )

    _apply_scenario_to_env(env, scenario)

    current_state = _build_initial_state(env, robot_spec)

    collision = False
    infeasible = False
    qp_infeasible = False
    runtime_error = False
    reached_goal = False

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
    certificate_lost = False
    certificate_loss_steps = 0
    qp_infeasible_steps = 0
    fallback_steps = 0

    warmup_steps = (
        jit_warmup_steps
        if algo in {
            "pcbf", "plcbf", "multi_backup_cbf_mi", "library_pcbf_mi"
        }
        else 0
    )
    comparison_control_enabled = _comparison_control_enabled(
        algo, comparison_mode
    )

    for step in range(max_steps):
        env.step()
        ghosts = env.get_dynamic_obstacles()
        statics = env.get_static_obstacles()

        u_nom = None
        u_safe = None

        if algo in {
            "pcbf", "plcbf", "mip_mpc",
            "multi_backup_cbf_mi", "library_pcbf_mi",
        }:
            shielding.update_obstacles(ghosts, statics)
            u_nom = np.array(nom_ctrl.get_control(current_state)).flatten()
            control_ref = {"u_ref": u_nom}

            if algo in {
                "plcbf", "mip_mpc",
                "multi_backup_cbf_mi", "library_pcbf_mi",
            }:
                control_ref["waypoints"] = nom_ctrl.waypoints
                control_ref["wp_idx"] = nom_ctrl.wp_idx
            elif algo == "pcbf":
                if hasattr(shielding, "backup_controller") and shielding.backup_controller is not None:
                    if hasattr(shielding.backup_controller, "prepare_rollout"):
                        shielding.backup_controller.prepare_rollout(current_state)

                    active_idx = int(getattr(shielding.backup_controller, "active_retrace_idx", 0))
                    wps_jax = jnp.array(nom_ctrl.waypoints)
                    new_params = RetracePolicyParams(
                        waypoints=wps_jax,
                        v_max=robot_spec["backup_speed"],
                        Kp=robot_spec["backup_Kp"],
                        dist_threshold=robot_spec["nominal_dist_threshold"],
                        current_wp_idx=active_idx,
                        ctrl=ctrl_params,
                    )
                    shielding.set_policy("retrace_waypoint", new_params)

            u_safe, solve_dt, solve_err = _solve_with_timing(
                lambda: shielding.solve_control_problem(current_state, control_ref)
            )

        elif algo in {"backup_cbf", "gatekeeper", "mps"}:
            u_nom = np.array(nom_ctrl.get_control(current_state, update_state=True)).flatten()

            if hasattr(shielding, "backup_controller") and shielding.backup_controller is not None:
                if hasattr(shielding.backup_controller, "prepare_rollout"):
                    shielding.backup_controller.prepare_rollout(current_state)

            if algo in {"gatekeeper", "mps"}:
                nom_traj_x = [current_state]
                nom_traj_u = []
                temp_x = current_state.copy()
                for _ in range(30):
                    u_pred = np.array(nom_ctrl.get_control(temp_x, update_state=False)).flatten()
                    nom_traj_u.append(u_pred)
                    temp_x = robot.step(temp_x.reshape(-1, 1), u_pred.reshape(-1, 1)).flatten()
                    nom_traj_x.append(temp_x)
                shielding.set_nominal_trajectory(np.array(nom_traj_x), np.array(nom_traj_u))

            u_safe, solve_dt, solve_err = _solve_with_timing(
                lambda: shielding.solve_control_problem(current_state)
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        step_metrics = {}
        if hasattr(shielding, "get_last_step_metrics"):
            try:
                step_metrics = dict(shielding.get_last_step_metrics())
            except Exception:
                step_metrics = {}

        selected_policy = step_metrics.get("selected_policy")
        if selected_policy is not None:
            selected_policy_histogram[str(selected_policy)] += 1
        policy_switch_count += int(bool(step_metrics.get("policy_switched", False)))
        num_candidate_qps_solved += int(
            step_metrics.get("num_candidate_qps_solved", 0) or 0
        )
        no_safe_this_step = int(
            step_metrics.get("num_steps_with_no_safe_policy", 0) or 0
        )
        num_steps_with_no_safe_policy += no_safe_this_step
        if "num_feasible_backup_candidates" in step_metrics:
            feasible_backup_counts.append(
                float(step_metrics["num_feasible_backup_candidates"])
            )
        if "terminal_failure_count" in step_metrics:
            terminal_failure_count += int(step_metrics["terminal_failure_count"] or 0)
        if "num_safe_candidates" in step_metrics:
            rollout_safe_counts.append(float(step_metrics["num_safe_candidates"]))
        if "num_rollout_safe_candidates" in step_metrics:
            rollout_safe_counts.append(
                float(step_metrics["num_rollout_safe_candidates"])
            )
        if "num_qp_feasible_candidates" in step_metrics:
            qp_feasible_counts.append(
                float(step_metrics["num_qp_feasible_candidates"])
            )

        step_certificate_lost = False
        step_qp_infeasible = False
        step_fallback_used = False
        if solve_err is None and comparison_control_enabled:
            (
                step_certificate_lost,
                step_qp_infeasible,
                step_fallback_used,
            ) = _step_failure_events(step_metrics, shielding)
            certificate_lost = certificate_lost or step_certificate_lost
            qp_infeasible = qp_infeasible or step_qp_infeasible
            certificate_loss_steps += int(step_certificate_lost)
            qp_infeasible_steps += int(step_qp_infeasible)
            fallback_steps += int(step_fallback_used)
            if step_certificate_lost or step_qp_infeasible:
                # Apply one identical continuation action for all three rows.
                # PL-CBF's internal least-negative/fallback output is retained
                # as an algorithm diagnostic but is not allowed to change the
                # post-failure physical trajectory relative to the MI rows.
                u_safe = _common_stop_control(shielding, current_state)

        if step >= warmup_steps:
            solve_times_sec.append(float(solve_dt))
            solve_time_sum += float(solve_dt)
            timed_steps += 1

        if solve_err is not None:
            runtime_error = True
            infeasible = True
            if comparison_control_enabled:
                try:
                    u_safe = _common_stop_control(shielding, current_state)
                    fallback_steps += 1
                except Exception:
                    break
            else:
                # Preserve the historical termination behavior for algorithms
                # outside the explicit three-method comparison.
                break

        u_safe = np.array(u_safe).flatten()
        if u_safe.shape[0] != 4 or not np.all(np.isfinite(u_safe)):
            runtime_error = True
            infeasible = True
            if comparison_control_enabled:
                try:
                    u_safe = _common_stop_control(shielding, current_state)
                    fallback_steps += 1
                except Exception:
                    break
            else:
                break

        if u_nom is not None:
            intervention_l2_values.append(
                float(np.linalg.norm(u_safe - np.asarray(u_nom).reshape(-1)))
            )

        current_state = robot.step(current_state.reshape(-1, 1), u_safe.reshape(-1, 1)).flatten()
        env.robot_pos = current_state[:2]

        # Count every applied physical control step, including the step that
        # causes a collision or reaches the goal.
        if u_nom is not None and np.linalg.norm(u_safe - u_nom) < tracking_tol:
            nominal_track_steps += 1
        total_steps += 1

        # Collision check: static
        for obs in statics:
            dist = np.linalg.norm(current_state[:2] - np.array([obs["x"], obs["y"]]))
            if dist < (obs["radius"] + robot_spec["radius"]):
                collision = True
                break

        # Collision check: dynamic
        if not collision:
            for g in ghosts:
                dist_g = np.linalg.norm(current_state[:2] - np.array([g["x"], g["y"]]))
                if dist_g < (g["radius"] + robot_spec["radius"]):
                    collision = True
                    break

        if collision:
            break

        if np.linalg.norm(current_state[:2] - env.goal_pos) < env.goal_radius:
            reached_goal = True
            break

    survived_horizon = bool(
        total_steps == max_steps and not collision and not runtime_error
    )
    # A completed task means reaching the goal.  Horizon survival is reported
    # separately because it is a safety outcome, not task completion.
    task_completed = bool(reached_goal)
    completed_or_survived = bool(task_completed or survived_horizon)
    filter_failure = bool(collision or certificate_lost or qp_infeasible)
    union_failure = bool(filter_failure or runtime_error or not completed_or_survived)
    # Compatibility field retained for older readers.  New analysis should use
    # the explicit certificate/QP/runtime fields above.
    infeasible = bool(infeasible or runtime_error)
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
            if feasible_backup_counts else float("nan")
        ),
        terminal_failure_count=terminal_failure_count,
        mean_rollout_safe_candidates=(
            float(np.mean(rollout_safe_counts))
            if rollout_safe_counts else float("nan")
        ),
        mean_qp_feasible_candidates=(
            float(np.mean(qp_feasible_counts))
            if qp_feasible_counts else float("nan")
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
    )


def summarize_trials(algo_spec: AlgoSpec, trials: List[TrialResult]) -> SummaryRow:
    n = len(trials)
    collisions = sum(int(t.collision) for t in trials)
    infeasibles = sum(int(t.infeasible) for t in trials)
    fail_count = sum(int(t.collision or t.infeasible) for t in trials)
    union_failures = sum(int(t.union_failure) for t in trials)
    certificate_losses = sum(int(t.certificate_lost) for t in trials)
    qp_infeasibles = sum(int(t.qp_infeasible) for t in trials)
    runtime_errors = sum(int(t.runtime_error) for t in trials)
    goal_reaches = sum(int(t.reached_goal) for t in trials)
    task_completions = sum(int(t.task_completed) for t in trials)
    horizon_survivals = sum(int(t.survived_horizon) for t in trials)
    successful_outcomes = sum(int(t.completed_or_survived) for t in trials)
    filter_failures = sum(int(t.filter_failure) for t in trials)

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
        library_size=max((t.p_or_library_size for t in trials), default=0),
    )


def run_algorithm_trials(
    algo_spec: AlgoSpec,
    scenarios: List[TrialScenario],
    args: argparse.Namespace,
    *,
    verbose: bool,
) -> Tuple[List[TrialResult], SummaryRow]:
    trials: List[TrialResult] = []
    comparison_mode = bool(
        set(getattr(args, "algorithms", None) or ())
        & ADDITIONAL_COMPARISON_ALGOS
    )

    t_algo0 = time.perf_counter()
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
            args.plcbf_num_angle_policies,
            args.mip_num_angle_policies,
            comparison_mode,
        )
        for scenario in scenarios
    ]
    worker_count = max(1, int(getattr(args, "num_workers", 1)))
    if worker_count > 1:
        # Trials are independent and executor.map preserves their seeded order.
        # This changes only wall-clock scheduling; each worker runs the exact
        # same run_trial path and produces its own per-step timing samples.
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            result_iterator = executor.map(_run_trial_payload, payloads)
            indexed_results = enumerate(result_iterator)
            for idx, result in indexed_results:
                trials.append(result)
                if verbose and ((idx + 1) % max(1, args.progress_every) == 0):
                    print(
                        f"  {algo_spec.key:22s} trial {idx + 1:3d}/{len(scenarios)} | "
                        f"collision={int(result.collision)} "
                        f"certificate={int(result.certificate_lost)} "
                        f"qp={int(result.qp_infeasible)} "
                        f"track={result.nominal_tracking_pct:.1f}%"
                    )
    else:
        indexed_results = (
            (idx, _run_trial_payload(payload)) for idx, payload in enumerate(payloads)
        )
        for idx, result in indexed_results:
            trials.append(result)
            if verbose and ((idx + 1) % max(1, args.progress_every) == 0):
                print(
                    f"  {algo_spec.key:22s} trial {idx + 1:3d}/{len(scenarios)} | "
                    f"collision={int(result.collision)} "
                    f"certificate={int(result.certificate_lost)} "
                    f"qp={int(result.qp_infeasible)} "
                    f"track={result.nominal_tracking_pct:.1f}%"
                )

    # The loop above intentionally replaces the historical serial loop only
    # when --num-workers is requested.  Summary semantics are unchanged.

    elapsed = time.perf_counter() - t_algo0
    summary = summarize_trials(algo_spec, trials)
    print(
        f"[Done] {algo_spec.label:<28} "
        f"collision={_fmt_count_rate(summary.collisions, summary.n_trials)} "
        f"certificate={_fmt_count_rate(summary.certificate_losses, summary.n_trials)} "
        f"qp={_fmt_count_rate(summary.qp_infeasibles, summary.n_trials)} "
        f"union={_fmt_count_rate(summary.union_failures, summary.n_trials)} "
        f"nominal={summary.avg_nominal_tracking_pct:.1f}% "
        f"avg_compute={summary.avg_compute_ms:.3f} ms "
        f"elapsed={elapsed/60.0:.1f} min"
    )

    return trials, summary


def _run_trial_payload(payload) -> TrialResult:
    """Pickle-friendly adapter for optional process-level trial parallelism."""
    (
        algo,
        scenario,
        level,
        safety_margin,
        alpha,
        max_steps,
        jit_warmup_steps,
        tracking_tol,
        plcbf_num_angle_policies,
        mip_num_angle_policies,
        comparison_mode,
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
        plcbf_num_angle_policies=plcbf_num_angle_policies,
        mip_num_angle_policies=mip_num_angle_policies,
        comparison_mode=comparison_mode,
    )


def refresh_timing_one_by_one(
    algo_specs: List[AlgoSpec],
    scenarios: List[TrialScenario],
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Recompute average solve time per algorithm by running algorithms one-by-one."""
    refreshed: Dict[str, float] = {}
    comparison_mode = bool(
        set(getattr(args, "algorithms", None) or ())
        & ADDITIONAL_COMPARISON_ALGOS
    )

    print("\n=== Timing Refresh (one algorithm at a time) ===")
    timing_scenarios = scenarios[: max(1, min(args.timing_refresh_trials, len(scenarios)))]
    print(f"Timing refresh scenarios: {len(timing_scenarios)}")
    for algo_spec in algo_specs:
        t0 = time.perf_counter()
        total_solve_sec = 0.0
        total_timed_steps = 0

        for scenario in timing_scenarios:
            trial = run_trial(
                algo=algo_spec.key,
                scenario=scenario,
                level=args.level,
                safety_margin=args.safety_margin,
                alpha=args.alpha,
                max_steps=args.max_steps,
                jit_warmup_steps=args.jit_warmup_steps,
                tracking_tol=args.tracking_tol,
                plcbf_num_angle_policies=args.plcbf_num_angle_policies,
                mip_num_angle_policies=args.mip_num_angle_policies,
                comparison_mode=comparison_mode,
            )
            total_solve_sec += trial.solve_time_sum_sec
            total_timed_steps += trial.timed_steps

        avg_ms = 1000.0 * total_solve_sec / max(total_timed_steps, 1)
        refreshed[algo_spec.key] = avg_ms
        print(
            f"[Timing] {algo_spec.label:<28} avg_compute={avg_ms:.3f} ms "
            f"timed_steps={total_timed_steps} elapsed={(time.perf_counter()-t0)/60.0:.1f} min"
        )

    return refreshed


def _select_animation_work_items(
    *,
    algo_specs: List[AlgoSpec],
    scenarios: List[TrialScenario],
    args: argparse.Namespace,
) -> Tuple[List[AlgoSpec], List[int]]:
    selected_algo_specs = algo_specs
    if args.animation_algos:
        requested = set(args.animation_algos)
        selected_algo_specs = [s for s in algo_specs if s.key in requested]
    if not selected_algo_specs:
        return [], []

    if args.animation_indices:
        selected_indices = sorted(set(int(i) for i in args.animation_indices))
        invalid = [i for i in selected_indices if i < 0 or i >= len(scenarios)]
        if invalid:
            raise ValueError(
                f"Invalid --animation-indices {invalid}; valid range is [0, {len(scenarios) - 1}]"
            )
    else:
        n_sets = max(0, min(int(args.animation_sets), len(scenarios)))
        if n_sets <= 0:
            return [], []
        selected_indices = list(range(n_sets))

    return selected_algo_specs, selected_indices


def save_randomized_animations(
    *,
    algo_specs: List[AlgoSpec],
    scenarios: List[TrialScenario],
    args: argparse.Namespace,
):
    selected_algo_specs, selected_indices = _select_animation_work_items(
        algo_specs=algo_specs,
        scenarios=scenarios,
        args=args,
    )
    if not selected_algo_specs:
        print("No animation algorithms selected; skipping animation export.")
        return
    if not selected_indices:
        print("No animation scenarios selected; skipping animation export.")
        return

    output_root = Path(args.animation_output_dir)
    if not output_root.is_absolute():
        output_root = Path(PROJECT_ROOT) / output_root
    output_root = output_root / f"seed_{args.seed}"
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        f"\n=== Saving randomized animations ({len(selected_indices)} scenario sets, "
        f"{len(selected_algo_specs)} algorithms) ==="
    )
    print(f"Animation output root: {output_root}")
    print(
        "Animation defaults: "
        f"safety_margin={args.animation_safety_margin:.2f}, "
        f"max_steps={args.animation_max_steps}"
    )

    for idx in selected_indices:
        scenario = scenarios[idx]
        scenario_dir = output_root / f"idx_{idx:02d}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        for algo_spec in selected_algo_specs:
            filename = f"warehouse_lvl{args.level}_{algo_spec.key}_idx{idx:02d}.mp4"
            print(
                f"[Animation] idx={idx:02d} algo={algo_spec.key:10s} "
                f"-> {scenario_dir / filename}"
            )

            anim_args = argparse.Namespace(
                algo=algo_spec.key,
                level=args.level,
                no_render=False,
                save=True,
                save_dir=str(scenario_dir),
                save_name=filename,
                paper_animation=args.paper_animation,
                paper_no_zoom=args.paper_no_zoom,
                save_svg=args.save_svg,
                paper_zoom_half_window=args.paper_zoom_half_window,
                paper_arrow_length=args.paper_arrow_length,
                paper_linewidth_scale=args.paper_linewidth_scale,
                safety_margin=args.animation_safety_margin,
                animation_safety_margin=args.animation_safety_margin,
                alpha=args.alpha,
                plcbf_num_angle_policies=args.plcbf_num_angle_policies,
                mip_num_angle_policies=args.mip_num_angle_policies,
                timing_warmup_steps=args.jit_warmup_steps,
                sensing_range=args.sensing_range,
                max_steps=args.animation_max_steps,
                animation_max_steps=args.animation_max_steps,
            )
            result = test_quad.run_simulation(anim_args, scenario_ghosts=scenario.ghosts)
            print(
                f"  result: collision={int(result.get('collision', False))} "
                f"infeasible={int(result.get('infeasible', False))} "
                f"reach_goal={int(result.get('reach_goal', False))}"
            )


def render_randomized_animations(
    *,
    algo_specs: List[AlgoSpec],
    scenarios: List[TrialScenario],
    args: argparse.Namespace,
):
    selected_algo_specs, selected_indices = _select_animation_work_items(
        algo_specs=algo_specs,
        scenarios=scenarios,
        args=args,
    )
    if not selected_algo_specs:
        print("No animation algorithms selected; skipping rendering.")
        return
    if not selected_indices:
        print("No animation scenarios selected; skipping rendering.")
        return

    print(
        f"\n=== Rendering randomized animations ({len(selected_indices)} scenario sets, "
        f"{len(selected_algo_specs)} algorithms) ==="
    )
    print(
        "Render defaults: "
        f"safety_margin={args.animation_safety_margin:.2f}, "
        f"max_steps={args.animation_max_steps}"
    )

    for idx in selected_indices:
        scenario = scenarios[idx]

        for algo_spec in selected_algo_specs:
            print(f"[Render] idx={idx:02d} algo={algo_spec.key:10s}")

            render_args = argparse.Namespace(
                algo=algo_spec.key,
                level=args.level,
                no_render=False,
                save=False,
                save_dir=None,
                save_name=None,
                paper_animation=args.paper_animation,
                paper_no_zoom=args.paper_no_zoom,
                save_svg=False,
                paper_zoom_half_window=args.paper_zoom_half_window,
                paper_arrow_length=args.paper_arrow_length,
                paper_linewidth_scale=args.paper_linewidth_scale,
                safety_margin=args.animation_safety_margin,
                animation_safety_margin=args.animation_safety_margin,
                alpha=args.alpha,
                plcbf_num_angle_policies=args.plcbf_num_angle_policies,
                mip_num_angle_policies=args.mip_num_angle_policies,
                timing_warmup_steps=args.jit_warmup_steps,
                sensing_range=args.sensing_range,
                max_steps=args.animation_max_steps,
                animation_max_steps=args.animation_max_steps,
            )

            result = test_quad.run_simulation(render_args, scenario_ghosts=scenario.ghosts)
            print(
                f"  result: collision={int(result.get('collision', False))} "
                f"infeasible={int(result.get('infeasible', False))} "
                f"reach_goal={int(result.get('reach_goal', False))}"
            )


def format_markdown(
    *,
    summaries: List[SummaryRow],
    args: argparse.Namespace,
    scenario_seed: int,
    timing_refreshed: bool,
) -> str:
    lines: List[str] = []
    lines.append("# Warehouse Quad3D Randomized Benchmark Results")
    lines.append("")
    lines.append(f"- Level layout: {args.level} (static obstacles and waypoints fixed)")
    lines.append(f"- Trials per algorithm: {args.num_trials}")
    lines.append(f"- Scenario seed: {scenario_seed}")
    lines.append(f"- Dynamic obstacles per trial: {args.num_dynamic_obstacles} (randomized)")
    lines.append(f"- Max steps per trial: {args.max_steps}")
    lines.append(
        "- Initial safety guard: dynamic obstacles excluded from start-area square "
        f"x<={args.start_exclusion_max_x:.1f}, y<={args.start_exclusion_max_y:.1f}"
    )
    lines.append(f"- Safety margin: {args.safety_margin:.2f}")
    lines.append(f"- PLCBF angle policies: {args.plcbf_num_angle_policies}")
    detailed_comparison = any(
        summary.key in {"multi_backup_cbf_mi", "library_pcbf_mi"}
        for summary in summaries
    )
    if detailed_comparison:
        lines.append(
            "- Runtime PL-CBF library: P angle policies + stop + nominal "
            f"= P+2 = {args.plcbf_num_angle_policies + 2} policies"
        )
        lines.append(
            "- All three comparison controllers apply the exact shared stop action "
            "after certificate loss or QP failure; episodes stop only on "
            "collision, goal, unrecoverable runtime error, or the fixed horizon"
        )
        lines.append(
            "- Certificate loss: no policy has a positive rollout certificate; "
            "QP infeasible: a certified set exists but no required QP returns an "
            "accepted bounded input"
        )
        lines.append(
            "- Task completion: goal reached; horizon survival is reported separately"
        )
        lines.append(
            "- Union failure: collision OR certificate loss OR QP infeasibility "
            "OR runtime error OR neither goal completion nor horizon survival"
        )
    lines.append(f"- MIP angle policies: {args.mip_num_angle_policies}")
    lines.append(
        "- Timing warmup skip (PCBF/PLCBF/MB-CBF-MI/Lib-PCBF-MI): "
        f"{args.jit_warmup_steps} steps"
    )
    lines.append(
        "- Compute-time column uses solve-control time only "
        "(plotting/logging excluded; refreshed one-by-one after full table "
        f"using {args.timing_refresh_trials} scenario(s))"
        if timing_refreshed
        else "- Compute-time column uses solve-control time only (plotting/logging excluded)"
    )
    lines.append("")
    if detailed_comparison:
        lines.append(
            "| Algorithm | P | Library size | Collision | Certificate loss | "
            "QP infeasible | Goal | Horizon survival | Goal or survival | "
            "Union failure | Avg Compute Time (ms) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in summaries:
            lines.append(
                "| "
                f"{s.label} | "
                f"{args.plcbf_num_angle_policies} | "
                f"{s.library_size or args.plcbf_num_angle_policies + 2} | "
                f"{_fmt_count_rate(s.collisions, s.n_trials)} | "
                f"{_fmt_count_rate(s.certificate_losses, s.n_trials)} | "
                f"{_fmt_count_rate(s.qp_infeasibles, s.n_trials)} | "
                f"{_fmt_count_rate(s.task_completions, s.n_trials)} | "
                f"{_fmt_count_rate(s.horizon_survivals, s.n_trials)} | "
                f"{_fmt_count_rate(s.successful_outcomes, s.n_trials)} | "
                f"{_fmt_count_rate(s.union_failures, s.n_trials)} | "
                f"{s.avg_compute_ms:.3f} |"
            )
    else:
        lines.append(
            "| Algorithm | Collision Rate | Infeasible Rate | Collision+Infeasible Rate | "
            "Avg Nominal Tracking (%) | Avg Compute Time (ms) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")

        for s in summaries:
            lines.append(
                "| "
                f"{s.label} | "
                f"{_fmt_count_rate(s.collisions, s.n_trials)} | "
                f"{_fmt_count_rate(s.infeasibles, s.n_trials)} | "
                f"{_fmt_count_rate(s.fail_count, s.n_trials)} | "
                f"{s.avg_nominal_tracking_pct:.1f} | "
                f"{s.avg_compute_ms:.3f} |"
            )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Warehouse Quad3D randomized benchmark")
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--level", type=int, default=7)
    parser.add_argument("--num-dynamic-obstacles", type=int, default=45)

    parser.add_argument("--safety-margin", type=float, default=1.3)
    parser.add_argument("--alpha", type=float, default=6.0)
    parser.add_argument("--max-steps", type=int, default=350)

    parser.add_argument("--plcbf-num-angle-policies", type=int, default=64)
    parser.add_argument("--mip-num-angle-policies", type=int, default=32)

    parser.add_argument("--jit-warmup-steps", type=int, default=10)
    parser.add_argument("--tracking-tol", type=float, default=0.1)

    parser.add_argument("--speed-min", type=float, default=3.0)
    parser.add_argument("--speed-max", type=float, default=4.5)
    parser.add_argument("--ghost-radius", type=float, default=2.4)
    parser.add_argument("--inter-ghost-clearance", type=float, default=0.2)

    parser.add_argument("--start-exclusion-max-x", type=float, default=18.0)
    parser.add_argument("--start-exclusion-max-y", type=float, default=18.0)
    parser.add_argument("--start-clearance-radius", type=float, default=8.0)


    parser.add_argument("--skip-mip", action="store_true")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=[spec.key for spec in ALL_ALGO_SPECS],
        default=None,
        help=(
            "Run only the selected rows. The two additive baselines are "
            "multi_backup_cbf_mi and library_pcbf_mi; omitting this flag "
            "preserves the historical benchmark set."
        ),
    )
    parser.add_argument("--skip-timing-refresh", action="store_true")
    parser.add_argument("--timing-refresh-trials", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Independent seeded trials to run concurrently. Default 1 "
            "preserves historical serial execution; timings under concurrency "
            "are tentative."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sensing-range", type=float, default=test_quad.DEFAULT_SENSING_RANGE_M)
    parser.add_argument("--save-animations", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--animations-only", action="store_true")
    parser.add_argument(
        "--paper-animation",
        action="store_true",
        help="Enable paper-focused animation mode (velocity arrows + robot-centered zoom).",
    )
    parser.add_argument(
        "--save-svg",
        action="store_true",
        help="When saving animations, also write per-frame SVG files (kept on disk).",
    )
    parser.add_argument(
        "--paper-zoom-half-window",
        type=float,
        default=None,
        help="Paper mode half-window in meters (default handled in test script).",
    )
    parser.add_argument(
        "--paper-arrow-length",
        type=float,
        default=test_quad.PAPER_ARROW_SCALE_M,
        help="Arrow length for dynamic-obstacle velocity direction in paper mode.",
    )
    parser.add_argument(
        "--paper-linewidth-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to visualization line widths.",
    )
    parser.add_argument(
        "--paper-no-zoom",
        action="store_true",
        help="Disable paper-mode robot-centered zoom (keep full-map view).",
    )
    parser.add_argument("--animation-sets", type=int, default=5)
    parser.add_argument("--animation-indices", type=int, nargs="+", default=None)
    parser.add_argument(
        "--animation-algos",
        type=str,
        nargs="+",
        default=None,
        choices=[s.key for s in ALL_ALGO_SPECS],
    )
    parser.add_argument(
        "--animation-safety-margin",
        type=float,
        default=test_quad.DEFAULT_ANIMATION_SAFETY_MARGIN,
    )
    parser.add_argument(
        "--animation-max-steps",
        type=int,
        default=test_quad.DEFAULT_ANIMATION_MAX_STEPS,
    )
    parser.add_argument(
        "--animation-output-dir",
        type=str,
        default="output/animations/warehouse_randomized_quad_sets",
    )

    parser.add_argument(
        "--output-md",
        type=str,
        default="examples/warehouse/benchmark_warehouse_randomized_quad_results.md",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="examples/warehouse/benchmark_warehouse_randomized_quad_results.json",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Per-trial metrics CSV (selected-policy histograms are JSON encoded).",
    )

    args = parser.parse_args()

    warnings.filterwarnings("ignore", message="Solution may be inaccurate.*", module="cvxpy")

    selected_specs = ALL_ALGO_SPECS if args.algorithms else ALGO_SPECS
    requested = set(args.algorithms or [])
    algo_specs = [
        spec
        for spec in selected_specs
        if (not requested or spec.key in requested)
        and not (args.skip_mip and spec.key == "mip_mpc")
    ]
    if args.output_csv is None and any(
        spec.key in {item.key for item in ADDITIONAL_ALGO_SPECS}
        for spec in algo_specs
    ):
        args.output_csv = (
            "examples/warehouse/"
            "benchmark_warehouse_randomized_quad_additional_results.csv"
        )

    print("Generating randomized dynamic-obstacle scenarios...")
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

    print(
        f"Generated {len(scenarios)} scenarios with {args.num_dynamic_obstacles} dynamic obstacles each "
        f"(seed={args.seed})."
    )

    if args.animations_only and not args.save_animations:
        raise ValueError("--animations-only requires --save-animations")

    if args.render:
        render_randomized_animations(algo_specs=algo_specs, scenarios=scenarios, args=args)
        print("Render-only run complete.")
        return

    if args.save_animations:
        save_randomized_animations(algo_specs=algo_specs, scenarios=scenarios, args=args)
        if args.animations_only:
            print("Animation-only run complete.")
            return

    all_trial_results: Dict[str, List[TrialResult]] = {}
    summaries: List[SummaryRow] = []

    print("\n=== Phase A: Full Benchmark Table ===")
    for algo_spec in algo_specs:
        print(f"\nRunning {algo_spec.label}...")
        trials, summary = run_algorithm_trials(algo_spec, scenarios, args, verbose=args.verbose)
        all_trial_results[algo_spec.key] = trials
        summaries.append(summary)

    timing_refreshed = False
    if not args.skip_timing_refresh:
        refreshed_times = refresh_timing_one_by_one(algo_specs, scenarios, args)
        key_to_summary = {s.key: s for s in summaries}
        for key, avg_ms in refreshed_times.items():
            key_to_summary[key].avg_compute_ms = avg_ms
        timing_refreshed = True

    markdown = format_markdown(
        summaries=summaries,
        args=args,
        scenario_seed=args.seed,
        timing_refreshed=timing_refreshed,
    )

    output_md = Path(args.output_md)
    if not output_md.is_absolute():
        output_md = Path(PROJECT_ROOT) / output_md
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")

    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = Path(PROJECT_ROOT) / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)

    json_payload = {
        "config": vars(args),
        "scenario_seed": args.seed,
        "failure_semantics": {
            "certificate_loss": "no policy has a positive rollout certificate",
            "qp_infeasible": (
                "a certified policy exists but no required QP returns an accepted "
                "bounded input"
            ),
            "filter_failure": "collision OR certificate_loss OR qp_infeasible",
            "task_completed": "goal reached",
            "survived_horizon": (
                "fixed horizon reached without collision or runtime error"
            ),
            "completed_or_survived": "task_completed OR survived_horizon",
            "union_failure": (
                "filter_failure OR runtime_error OR NOT completed_or_survived"
            ),
            "post_filter_event_action": "exact shared stop library entry",
        },
        "policy_library": {
            "angle_policy_count": args.plcbf_num_angle_policies,
            "extra_entries": ["stop", "nominal"],
            "library_size": args.plcbf_num_angle_policies + 2,
        },
        "summaries": [asdict(s) for s in summaries],
        "timing_refreshed": timing_refreshed,
    }
    additional_keys = {spec.key for spec in ADDITIONAL_ALGO_SPECS}
    if any(spec.key in additional_keys for spec in algo_specs):
        json_payload["trials"] = {
            key: [asdict(trial) for trial in trials]
            for key, trials in all_trial_results.items()
        }
    output_json.write_text(
        json.dumps(_json_safe(json_payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    trial_rows = []
    output_csv = None
    if args.output_csv is not None:
        output_csv = Path(args.output_csv)
        if not output_csv.is_absolute():
            output_csv = Path(PROJECT_ROOT) / output_csv
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        for trials in all_trial_results.values():
            for trial in trials:
                row = asdict(trial)
                for key, value in list(row.items()):
                    if isinstance(value, (dict, list, tuple)):
                        row[key] = json.dumps(_json_safe(value), sort_keys=True)
                    elif isinstance(value, float) and not np.isfinite(value):
                        row[key] = ""
                trial_rows.append(row)
        if trial_rows:
            with output_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(trial_rows[0].keys()))
                writer.writeheader()
                writer.writerows(trial_rows)

    print("\n" + markdown)
    print(f"Saved markdown report: {output_md}")
    print(f"Saved json report: {output_json}")
    if output_csv is not None and trial_rows:
        print(f"Saved per-trial CSV: {output_csv}")


if __name__ == "__main__":
    main()
