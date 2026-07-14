"""Baseline-only drift-car black-ice benchmark.

This entry point intentionally exposes only MB-CBF-MI and Lib-PCBF-MI.  The
historical benchmark remains in ``benchmark_black_ice.py`` byte-for-byte.  A
certificate or candidate-QP event is recorded without changing the action
returned by the selected baseline.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root and submodule path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "safe_control"))

from safe_control.envs.drifting_env import DriftingEnv
from safe_control.position_control.mpcc import MPCC
from safe_control.robots.drifting_car import DriftingCar, DriftingCarSimulator

from examples.drift_car.algorithms.plcbf_drift import PLCBF
from examples.drift_car.algorithms.library_pcbf_mi_drift import (
    LibraryPCBFMinInterventionDrift,
)
from examples.drift_car.algorithms.multi_backup_cbf_mi_drift import (
    MultiBackupCBFMinInterventionDrift,
)

QUIET_SINK = open(os.devnull, "w", encoding="utf-8")
ADDITIONAL_ALGOS = ("multi_backup_cbf_mi", "library_pcbf_mi")
FILTER_FAILURE_DEFINITION = "collision OR certificate_lost OR qp_infeasible"
HISTORICAL_FAILURE_DEFINITION = "collision OR unrecoverable_infeasibility"
UNION_FAILURE_DEFINITION = (
    "filter_failure OR runtime_error OR NOT completed_or_survived; task_completed "
    "means reaching the track goal and horizon survival is reported separately"
)
UNRECOVERABLE_INFEASIBLE_DEFINITION = (
    "unrecoverable solve exception, runtime error, non-finite control, invalid "
    "control dimension, or out-of-bounds control"
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fmt_count_rate(count: int, total: int, rate: float) -> str:
    return f"{count}/{total} ({rate:.1f}%)"


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.05
    tf: float = 14.0
    track_length: float = 300.0
    lane_width: float = 4.0
    num_lanes: int = 5
    nominal_horizon_time: float = 1.5
    backup_horizon_time: float = 3.0
    event_offset: float = 0.1
    safety_margin: float = 1.5
    initial_velocity: float = 10.0
    target_velocity: float = 10.0
    nominal_track_eps: float = 0.05
    # Black-ice puddle (always enabled)
    puddle_x: float = 70.0
    puddle_radius: float = 15.0
    puddle_friction: float = 0.30


@dataclass(frozen=True)
class AlgoVariant:
    key: str
    label: str
    algo: str
    backup_policy: Optional[str] = None


@dataclass(frozen=True)
class Scenario:
    run_idx: int
    seed: int
    num_obstacles: int
    # tuples: (x, lane_name) lane_name in {"middle","left","right"}
    obstacles: Tuple[Tuple[float, str], ...]


@dataclass
class EpisodeResult:
    algorithm: str
    seed: int
    run_idx: int
    obstacle_geometry: List[Tuple[float, str]]
    P_or_library_size: int
    collision: bool
    infeasible: bool
    unrecoverable_infeasible: bool
    historical_failure: bool
    certificate_lost: bool
    qp_infeasible: bool
    runtime_error: bool
    reached_goal: bool
    survived_horizon: bool
    task_completed: bool
    completed_or_survived: bool
    filter_failure: bool
    union_failure: bool
    total_steps: int
    timed_steps: int
    mean_compute_ms: float
    median_compute_ms: float
    p95_compute_ms: float
    max_compute_ms: float
    nominal_tracking_fraction: float
    mean_intervention_l2: float
    max_intervention_l2: float
    policy_switch_count: int
    selected_policy_histogram: Dict[str, int] = field(default_factory=dict)
    num_candidate_qps_solved: int = 0
    num_steps_with_no_safe_policy: int = 0
    num_feasible_backup_candidates_per_step: List[int] = field(default_factory=list)
    num_certified_backup_candidates_per_step: List[int] = field(default_factory=list)
    terminal_failure_count: int = 0
    num_rollout_safe_candidates_per_step: List[int] = field(default_factory=list)
    num_qp_feasible_candidates_per_step: List[int] = field(default_factory=list)
    certificate_loss_steps: int = 0
    qp_infeasible_steps: int = 0
    fallback_steps: int = 0
    candidate_qp_failure_count: int = 0
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

    @property
    def nominal_tracking_pct(self) -> float:
        """Compatibility accessor used by older analysis scripts."""

        return 100.0 * self.nominal_tracking_fraction


def build_vehicle_spec() -> Dict[str, float]:
    return {
        "model": "DriftingCar",
        "a": 1.4,
        "b": 1.4,
        "wheel_base": 2.8,
        "body_length": 4.5,
        "body_width": 2.0,
        "radius": 1.5,
        "m": 2500.0,
        "Iz": 5000.0,
        "Cc_f": 80000.0,
        "Cc_r": 100000.0,
        "mu": 1.0,  # default outside puddle
        "r_w": 0.35,
        "gamma": 0.95,
        "delta_max": np.deg2rad(20.0),
        # Keep agile steering from drift_pcbf tests
        "delta_dot_max": np.deg2rad(50.0),
        "tau_max": 4000.0,
        "tau_dot_max": 8000.0,
        "v_max": 20.0,
        "v_min": 0.0,
        "r_max": 2.0,
        "beta_max": np.deg2rad(45.0),
        "v_psi_max": 15.0,
        "v_ref": 10.0,
    }


def make_variants() -> List[AlgoVariant]:
    """Return the complete and exclusive publication-baseline registry."""

    return [
        AlgoVariant(
            "multi_backup_cbf_mi",
            "MB-CBF-MI†",
            "multi_backup_cbf_mi",
            None,
        ),
        AlgoVariant(
            "library_pcbf_mi",
            "Lib-PCBF-MI",
            "library_pcbf_mi",
            None,
        ),
    ]


def generate_scenarios(num_runs: int, seed: int) -> List[Scenario]:
    rng = np.random.default_rng(seed)
    scenarios: List[Scenario] = []
    lane_names = np.array(["middle", "left", "right"], dtype=object)

    for run_idx in range(num_runs):
        num_obs = int(rng.integers(1, 3))  # 1 or 2
        # Keep one primary obstacle near default center-line location.
        # Single-obstacle cases can still be center/left/right.
        x_first = float(rng.uniform(79.0, 83.0))
        if num_obs == 1:
            lane_first = str(rng.choice(lane_names, p=[0.7, 0.15, 0.15]))
            obstacles = ((x_first, lane_first),)
        else:
            lane_second = str(
                rng.choice(np.array(["left", "right"], dtype=object), p=[0.7, 0.3])
            )
            x_second = float(np.clip(x_first + rng.uniform(-3.5, 2.0), 75.0, 85.0))
            if abs(x_second - x_first) < 0.8:
                x_second = float(np.clip(x_first + 1.0, 75.0, 85.0))
            obstacles = ((x_first, "middle"), (x_second, lane_second))

        # Preserve the historical RNG stream (the old field consumed one draw
        # but was never used to initialize a trial).  Trial records now store
        # the actual master scenario seed plus run index, which is replayable.
        _unused_historical_trial_token = int(rng.integers(0, 2**31 - 1))
        scenarios.append(
            Scenario(
                run_idx=run_idx,
                seed=int(seed),
                num_obstacles=num_obs,
                obstacles=obstacles,
            )
        )
    return scenarios


def setup_env_and_lanes(cfg: SimConfig) -> Tuple[DriftingEnv, Dict[str, float]]:
    total_width = cfg.lane_width * cfg.num_lanes
    env = DriftingEnv(
        track_type="straight",
        track_width=total_width,
        track_length=cfg.track_length,
        num_lanes=cfg.num_lanes,
    )
    middle_idx = env.get_middle_lane_idx()
    lanes = {
        "middle": float(env.get_lane_center(middle_idx)),
        "left": float(env.get_lane_center(middle_idx - 1)),
        "right": float(env.get_lane_center(middle_idx + 1)),
    }
    return env, lanes


def add_black_ice_and_obstacles(env: DriftingEnv, lanes: Dict[str, float], cfg: SimConfig, scenario: Scenario):
    env.add_puddle(
        x=cfg.puddle_x,
        y=lanes["middle"],
        radius=cfg.puddle_radius,
        friction=cfg.puddle_friction,
    )

    obstacle_spec = {
        "body_length": 4.5,
        "body_width": 2.0,
        "a": 1.4,
        "b": 1.4,
        "radius": 2.0,
    }
    for obs_x, lane_name in scenario.obstacles:
        env.add_obstacle_car(
            x=float(obs_x),
            y=float(lanes[lane_name]),
            theta=0.0,
            robot_spec=obstacle_spec,
        )


def make_initial_state(lanes: Dict[str, float], cfg: SimConfig) -> np.ndarray:
    return np.array(
        [
            5.0,
            lanes["middle"],
            0.0,
            0.0,
            0.0,
            cfg.initial_velocity,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def setup_mpcc(car: DriftingCar, env: DriftingEnv, lanes: Dict[str, float], cfg: SimConfig) -> MPCC:
    horizon_steps = int(cfg.nominal_horizon_time / cfg.dt)
    mpcc = MPCC(car, car.robot_spec, horizon=horizon_steps)
    ref_x = env.centerline[:, 0]
    ref_y = np.full_like(ref_x, lanes["middle"])
    mpcc.set_reference_path(ref_x, ref_y)
    mpcc.set_cost_weights(
        Q_c=30.0,
        Q_l=1.0,
        Q_theta=20.0,
        Q_v=50.0,
        Q_r=80.0,
        v_ref=cfg.target_velocity,
        R=np.array([300.0, 0.5, 0.1]),
    )
    mpcc.set_progress_rate(cfg.target_velocity)
    return mpcc


def create_reference_plcbf(
    car: DriftingCar,
    env: DriftingEnv,
    lanes: Dict[str, float],
    cfg: SimConfig,
) -> PLCBF:
    """Construct the one runtime library shared by all multi-policy methods."""

    controller = PLCBF(
        robot=car,
        robot_spec=car.robot_spec,
        dt=cfg.dt,
        backup_horizon=cfg.backup_horizon_time,
        cbf_alpha=6.0,
        left_lane_y=lanes["left"],
        right_lane_y=lanes["right"],
        safety_margin=1.15,
        max_operator="input_space",
        debug=False,
        ax=None,
    )
    controller.set_environment(env)
    return controller


def setup_shielding(
    variant: AlgoVariant,
    car: DriftingCar,
    env: DriftingEnv,
    lanes: Dict[str, float],
    cfg: SimConfig,
):
    if variant.algo not in ADDITIONAL_ALGOS:
        raise ValueError(
            f"Unknown baseline {variant.algo!r}; valid keys are {ADDITIONAL_ALGOS}"
        )

    # This object is only a policy-library equality oracle. Its PL-CBF solver
    # is never called by this baseline-only benchmark.
    reference_plcbf = create_reference_plcbf(car, env, lanes, cfg)
    if variant.algo == "multi_backup_cbf_mi":
        shielding = MultiBackupCBFMinInterventionDrift(
            robot=car,
            robot_spec=car.robot_spec,
            dt=cfg.dt,
            backup_horizon=cfg.backup_horizon_time,
            ax=None,
            reference_plcbf=reference_plcbf,
        )
    else:
        shielding = LibraryPCBFMinInterventionDrift(
            robot=car,
            robot_spec=car.robot_spec,
            dt=cfg.dt,
            backup_horizon=cfg.backup_horizon_time,
            cbf_alpha=6.0,
            left_lane_y=lanes["left"],
            right_lane_y=lanes["right"],
            safety_margin=1.15,
            debug=False,
            ax=None,
            reference_plcbf=reference_plcbf,
        )
    shielding.set_environment(env)
    return shielding


def solve_safe_control(
    variant: AlgoVariant,
    shielding,
    state: np.ndarray,
    u_nom: np.ndarray,
    pred_states: Optional[np.ndarray],
    pred_controls: Optional[np.ndarray],
    friction: float,
):
    if variant.algo not in ADDITIONAL_ALGOS:
        raise ValueError(f"Unsupported baseline: {variant.algo}")
    return shielding.solve_control_problem(
        state,
        control_ref={"u_ref": u_nom},
        friction=friction,
        nominal_trajectory=pred_states.T if pred_states is not None else None,
        nominal_controls=pred_controls.T if pred_controls is not None else None,
    )


def call_quiet(quiet: bool, fn, *args, **kwargs):
    if quiet:
        with contextlib.redirect_stdout(QUIET_SINK), contextlib.redirect_stderr(QUIET_SINK):
            return fn(*args, **kwargs)
    return fn(*args, **kwargs)


def classify_baseline_status(variant: AlgoVariant, shielding, status: dict):
    """Return disjoint certificate/QP/runtime events for one filter step.

    MB-CBF-MI exposes the split directly. Lib-PCBF-MI's already-computed
    candidate list disambiguates an empty certified set from a certified set
    whose QPs all failed. This helper is diagnostic only and never changes the
    controller's returned action.
    """

    if variant.algo not in ADDITIONAL_ALGOS:
        return False, False, False, False

    status_text = str(status.get("status", "")).lower()
    runtime_error = bool(status.get("runtime_error", False)) or status_text.startswith(
        ("error", "value_error", "candidate_evaluation_error")
    )
    if runtime_error:
        return False, False, True, bool(status.get("fallback_applied", False))

    certificate_lost = bool(status.get("certificate_lost", False))
    qp_infeasible = bool(
        status.get("qp_infeasible", status.get("infeasible", False))
    ) and not certificate_lost
    return (
        certificate_lost,
        qp_infeasible,
        False,
        bool(status.get("fallback_applied", False)),
    )


def validate_returned_control(
    control,
    shielding,
    *,
    expected_dimension: int = 2,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Validate, but never clip or replace, a controller return value."""

    vector = np.asarray(control, dtype=float).reshape(-1)
    if vector.shape != (expected_dimension,):
        raise ValueError(
            f"controller returned shape {vector.shape}; expected {(expected_dimension,)}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("controller returned a non-finite control")

    lower = np.asarray(getattr(shielding, "u_min", -np.inf), dtype=float)
    upper = np.asarray(getattr(shielding, "u_max", np.inf), dtype=float)
    lower = np.broadcast_to(lower, vector.shape)
    upper = np.broadcast_to(upper, vector.shape)
    if np.any(vector < lower - tolerance) or np.any(vector > upper + tolerance):
        raise ValueError("controller returned an out-of-bounds control")
    return vector


def run_episode(
    variant: AlgoVariant,
    scenario: Scenario,
    cfg: SimConfig,
    verbose: bool = False,
) -> EpisodeResult:
    env, lanes = setup_env_and_lanes(cfg)
    add_black_ice_and_obstacles(env, lanes, cfg, scenario)

    vehicle_spec = build_vehicle_spec()
    x0 = make_initial_state(lanes, cfg)
    car = DriftingCar(x0, vehicle_spec, cfg.dt, ax=None)
    simulator = DriftingCarSimulator(car, env, show_animation=False)

    mpcc = setup_mpcc(car, env, lanes, cfg)
    shielding = setup_shielding(variant, car, env, lanes, cfg)

    n_steps = int(cfg.tf / cfg.dt)
    total_steps = 0
    nominal_like_steps = 0
    collision = False
    infeasible = False
    certificate_lost = False
    qp_infeasible = False
    runtime_error = False
    certificate_loss_steps = 0
    qp_infeasible_steps = 0
    fallback_steps = 0
    observed_candidate_qp_failure_count = 0
    reached_goal = False
    compute_times_s: List[float] = []
    intervention_samples: List[float] = []
    selected_policy_histogram: Dict[str, int] = {}
    policy_switch_count = 0
    previous_policy: Optional[str] = None
    solve_calls = 0
    timing_warmup_steps = 5

    u_scale = np.array(
        [
            float(vehicle_spec["delta_dot_max"]),
            float(vehicle_spec["tau_dot_max"]),
        ],
        dtype=float,
    )
    quiet = not verbose

    for step in range(n_steps):
        state = car.get_state()
        pos = car.get_position()

        curr_mu = env.get_friction_at_position(pos, default_friction=vehicle_spec["mu"])
        if abs(curr_mu - car.get_friction()) > 1e-8:
            car.set_friction(curr_mu)
            if hasattr(shielding, "set_friction"):
                shielding.set_friction(curr_mu)

        try:
            u_nom = call_quiet(quiet, mpcc.solve_control_problem, state)
            pred_states, pred_controls = call_quiet(quiet, mpcc.get_full_predictions)
        except Exception:
            runtime_error = True
            infeasible = True
            break

        try:
            solve_calls += 1
            solve_started = time.perf_counter()
            u_safe = call_quiet(
                quiet,
                solve_safe_control,
                variant=variant,
                shielding=shielding,
                state=state,
                u_nom=u_nom,
                pred_states=pred_states,
                pred_controls=pred_controls,
                friction=car.get_friction(),
            )
            solve_elapsed = time.perf_counter() - solve_started
        except Exception:
            runtime_error = True
            infeasible = True
            break

        try:
            u_nom_vec = np.asarray(u_nom, dtype=float).reshape(-1)
            if u_nom_vec.shape != (2,) or not np.all(np.isfinite(u_nom_vec)):
                raise ValueError("nominal controller returned an invalid control")
            u_safe_vec = validate_returned_control(u_safe, shielding)
        except (TypeError, ValueError):
            runtime_error = True
            infeasible = True
            break

        try:
            status = shielding.get_status() if hasattr(shielding, "get_status") else {}
        except Exception:
            runtime_error = True
            infeasible = True
            break
        (
            step_certificate_lost,
            step_qp_infeasible,
            step_runtime_error,
            step_fallback,
        ) = classify_baseline_status(variant, shielding, status)
        certificate_lost = certificate_lost or step_certificate_lost
        qp_infeasible = qp_infeasible or step_qp_infeasible
        runtime_error = runtime_error or step_runtime_error
        certificate_loss_steps += int(step_certificate_lost)
        qp_infeasible_steps += int(step_qp_infeasible)
        fallback_steps += int(step_fallback)
        observed_candidate_qp_failure_count += sum(
            bool(candidate.qp_solved and not candidate.feasible)
            for candidate in getattr(shielding, "last_candidate_results", [])
        )

        # Certificate loss and candidate-QP infeasibility are diagnostics, not
        # physical terminal conditions. Apply the exact finite control returned
        # by the baseline and continue. The benchmark never substitutes an
        # action.
        if step_runtime_error:
            infeasible = True
            break

        diff = np.linalg.norm((u_safe_vec - u_nom_vec) / np.maximum(u_scale, 1e-8))
        intervention_samples.append(float(np.linalg.norm(u_safe_vec - u_nom_vec)))
        if diff < cfg.nominal_track_eps:
            nominal_like_steps += 1
        selected_policy = status.get("best_policy")
        if selected_policy is not None:
            selected_policy = str(selected_policy)
            selected_policy_histogram[selected_policy] = (
                selected_policy_histogram.get(selected_policy, 0) + 1
            )
            if previous_policy is not None and previous_policy != selected_policy:
                policy_switch_count += 1
            previous_policy = selected_policy
        try:
            sim_res = simulator.step(u_safe_vec.reshape(-1, 1))
        except Exception:
            runtime_error = True
            infeasible = True
            break
        total_steps += 1
        if solve_calls > timing_warmup_steps:
            compute_times_s.append(solve_elapsed)
        collision = bool(sim_res["collision"])
        if collision:
            break

        # Track finished
        if car.get_position()[0] > env.track_length - 10.0:
            reached_goal = True
            break

    survived_horizon = bool(
        total_steps == n_steps and not collision and not runtime_error
    )
    # Goal completion and fixed-horizon survival are distinct outcomes.
    task_completed = bool(reached_goal)
    completed_or_survived = bool(task_completed or survived_horizon)
    filter_failure = bool(collision or certificate_lost or qp_infeasible)
    union_failure = bool(filter_failure or runtime_error or not completed_or_survived)
    # Preserve the historical meaning: only an unrecoverable solve/runtime
    # failure is infeasible. Candidate-QP failure is a separate diagnostic.
    infeasible = bool(infeasible or runtime_error)
    historical_failure = bool(collision or infeasible)
    nominal_fraction = nominal_like_steps / max(total_steps, 1)
    timing_ms = 1000.0 * np.asarray(compute_times_s, dtype=float)
    try:
        controller_metrics = (
            shielding.get_metrics() if hasattr(shielding, "get_metrics") else {}
        )
    except Exception:
        controller_metrics = {}
        runtime_error = True
        infeasible = True
        historical_failure = True
        survived_horizon = False
        union_failure = True
    if controller_metrics.get("selected_policy_histogram"):
        selected_policy_histogram = dict(controller_metrics["selected_policy_histogram"])
        policy_switch_count = int(controller_metrics.get("policy_switch_count", policy_switch_count))
    mean_intervention = float(
        controller_metrics.get(
            "mean_intervention_l2",
            np.mean(intervention_samples) if intervention_samples else 0.0,
        )
    )
    max_intervention = float(
        controller_metrics.get(
            "max_intervention_l2",
            np.max(intervention_samples) if intervention_samples else 0.0,
        )
    )
    if verbose:
        print(
            f"run={scenario.run_idx:02d} variant={variant.key} "
            f"obs={scenario.num_obstacles} collision={collision} infeasible={infeasible} "
            f"certificate_lost={certificate_lost} qp_infeasible={qp_infeasible} "
            f"task_completed={task_completed} nominal={100.0 * nominal_fraction:.1f}% "
            f"steps={total_steps}"
        )
    return EpisodeResult(
        algorithm=variant.key,
        seed=scenario.seed,
        run_idx=scenario.run_idx,
        obstacle_geometry=list(scenario.obstacles),
        P_or_library_size=4,
        collision=collision,
        infeasible=infeasible,
        unrecoverable_infeasible=infeasible,
        historical_failure=historical_failure,
        certificate_lost=certificate_lost,
        qp_infeasible=qp_infeasible,
        runtime_error=runtime_error,
        reached_goal=reached_goal,
        survived_horizon=survived_horizon,
        task_completed=task_completed,
        completed_or_survived=completed_or_survived,
        filter_failure=filter_failure,
        union_failure=union_failure,
        total_steps=total_steps,
        timed_steps=len(timing_ms),
        mean_compute_ms=float(np.mean(timing_ms)) if len(timing_ms) else float("nan"),
        median_compute_ms=float(np.median(timing_ms)) if len(timing_ms) else float("nan"),
        p95_compute_ms=float(np.percentile(timing_ms, 95)) if len(timing_ms) else float("nan"),
        max_compute_ms=float(np.max(timing_ms)) if len(timing_ms) else float("nan"),
        nominal_tracking_fraction=nominal_fraction,
        mean_intervention_l2=mean_intervention,
        max_intervention_l2=max_intervention,
        policy_switch_count=policy_switch_count,
        selected_policy_histogram=selected_policy_histogram,
        num_candidate_qps_solved=int(controller_metrics.get("num_candidate_qps_solved", 0)),
        num_steps_with_no_safe_policy=int(
            controller_metrics.get("num_steps_with_no_safe_policy", 0)
        ),
        num_feasible_backup_candidates_per_step=list(
            controller_metrics.get("num_feasible_backup_candidates_per_step", [])
        ),
        num_certified_backup_candidates_per_step=list(
            controller_metrics.get("num_certified_backup_candidates_per_step", [])
        ),
        terminal_failure_count=int(controller_metrics.get("terminal_failure_count", 0)),
        num_rollout_safe_candidates_per_step=list(
            controller_metrics.get("num_rollout_safe_candidates_per_step", [])
        ),
        num_qp_feasible_candidates_per_step=list(
            controller_metrics.get("num_qp_feasible_candidates_per_step", [])
        ),
        certificate_loss_steps=certificate_loss_steps,
        qp_infeasible_steps=qp_infeasible_steps,
        fallback_steps=fallback_steps,
        candidate_qp_failure_count=int(
            max(
                observed_candidate_qp_failure_count,
                controller_metrics.get("candidate_qp_failure_count", 0),
            )
        ),
        projection_occurred=bool(
            controller_metrics.get("projection_occurred", False)
        ),
        num_post_projection_audits=int(
            controller_metrics.get("num_post_projection_audits", 0)
        ),
        projection_event_count=int(
            controller_metrics.get("projection_event_count", 0)
        ),
        post_projection_rejection_count=int(
            controller_metrics.get("post_projection_rejection_count", 0)
        ),
        max_projection_delta_inf=float(
            controller_metrics.get("max_projection_delta_inf", 0.0)
        ),
        max_post_projection_constraint_violation=float(
            controller_metrics.get(
                "max_post_projection_constraint_violation", 0.0
            )
        ),
        max_post_projection_violation_ratio=float(
            controller_metrics.get("max_post_projection_violation_ratio", 0.0)
        ),
        num_post_projection_audits_per_step=list(
            controller_metrics.get("num_post_projection_audits_per_step", [])
        ),
        projection_event_count_per_step=list(
            controller_metrics.get("projection_event_count_per_step", [])
        ),
        post_projection_rejection_count_per_step=list(
            controller_metrics.get("post_projection_rejection_count_per_step", [])
        ),
        max_projection_delta_inf_per_step=list(
            controller_metrics.get("max_projection_delta_inf_per_step", [])
        ),
        max_post_projection_constraint_violation_per_step=list(
            controller_metrics.get(
                "max_post_projection_constraint_violation_per_step", []
            )
        ),
        max_post_projection_violation_ratio_per_step=list(
            controller_metrics.get(
                "max_post_projection_violation_ratio_per_step", []
            )
        ),
    )


def aggregate_results(
    variants: List[AlgoVariant],
    scenarios: List[Scenario],
    cfg: SimConfig,
    verbose: bool = False,
    num_workers: int = 1,
):
    keys = [variant.key for variant in variants]
    if len(keys) != len(set(keys)) or not set(keys).issubset(ADDITIONAL_ALGOS):
        raise ValueError(
            f"Baseline registry must contain unique keys drawn from {ADDITIONAL_ALGOS}"
        )
    all_results: Dict[str, List[EpisodeResult]] = {v.key: [] for v in variants}

    for variant in variants:
        if verbose:
            print(f"\n=== Running {variant.label} ===")
        payloads = [
            (variant, scenario, cfg, False)
            for scenario in scenarios
        ]
        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                iterator = executor.map(_run_episode_payload, payloads)
                for index, ep in enumerate(iterator):
                    all_results[variant.key].append(ep)
                    if verbose:
                        print(
                            f"  {variant.key:22s} run {index + 1:2d}/{len(scenarios)} "
                            f"collision={int(ep.collision)} infeasible={int(ep.infeasible)}"
                        )
        else:
            for payload in payloads:
                all_results[variant.key].append(_run_episode_payload(payload))

    summary_rows = []
    for variant in variants:
        rows = all_results[variant.key]
        n = len(rows)
        fail_count = sum(1 for r in rows if r.historical_failure)
        collision_count = sum(1 for r in rows if r.collision)
        infeasible_count = sum(1 for r in rows if r.infeasible)
        certificate_lost_count = sum(1 for r in rows if r.certificate_lost)
        qp_infeasible_count = sum(1 for r in rows if r.qp_infeasible)
        runtime_error_count = sum(1 for r in rows if r.runtime_error)
        task_completed_count = sum(1 for r in rows if r.task_completed)
        survived_horizon_count = sum(1 for r in rows if r.survived_horizon)
        successful_outcome_count = sum(1 for r in rows if r.completed_or_survived)
        filter_failure_count = sum(1 for r in rows if r.filter_failure)
        union_failure_count = sum(1 for r in rows if r.union_failure)
        projection_trial_count = sum(
            1 for r in rows if getattr(r, "projection_occurred", False)
        )
        num_post_projection_audits = sum(
            int(getattr(r, "num_post_projection_audits", 0)) for r in rows
        )
        projection_event_count = sum(
            int(getattr(r, "projection_event_count", 0)) for r in rows
        )
        post_projection_rejection_count = sum(
            int(getattr(r, "post_projection_rejection_count", 0)) for r in rows
        )
        max_projection_delta_inf = max(
            (float(getattr(r, "max_projection_delta_inf", 0.0)) for r in rows),
            default=0.0,
        )
        max_post_projection_constraint_violation = max(
            (
                float(
                    getattr(r, "max_post_projection_constraint_violation", 0.0)
                )
                for r in rows
            ),
            default=0.0,
        )
        max_post_projection_violation_ratio = max(
            (
                float(getattr(r, "max_post_projection_violation_ratio", 0.0))
                for r in rows
            ),
            default=0.0,
        )
        nominal_avg = float(np.mean([r.nominal_tracking_pct for r in rows])) if rows else 0.0
        timed_rows = [
            r for r in rows
            if r.timed_steps > 0 and np.isfinite(r.mean_compute_ms)
        ]
        total_timed_steps = sum(r.timed_steps for r in timed_rows)
        mean_compute_ms = (
            float(
                sum(r.mean_compute_ms * r.timed_steps for r in timed_rows)
                / total_timed_steps
            )
            if total_timed_steps
            else float("nan")
        )
        summary_rows.append(
            {
                "key": variant.key,
                "label": variant.label,
                "n": n,
                "fail_count": fail_count,
                "fail_rate": 100.0 * fail_count / max(n, 1),
                "collision_count": collision_count,
                "collision_rate": 100.0 * collision_count / max(n, 1),
                "infeasible_count": infeasible_count,
                "infeasible_rate": 100.0 * infeasible_count / max(n, 1),
                "certificate_lost_count": certificate_lost_count,
                "certificate_lost_rate": 100.0 * certificate_lost_count / max(n, 1),
                "qp_infeasible_count": qp_infeasible_count,
                "qp_infeasible_rate": 100.0 * qp_infeasible_count / max(n, 1),
                "runtime_error_count": runtime_error_count,
                "task_completed_count": task_completed_count,
                "task_completed_rate": 100.0 * task_completed_count / max(n, 1),
                "survived_horizon_count": survived_horizon_count,
                "survived_horizon_rate": 100.0 * survived_horizon_count / max(n, 1),
                "successful_outcome_count": successful_outcome_count,
                "successful_outcome_rate": 100.0 * successful_outcome_count / max(n, 1),
                "filter_failure_count": filter_failure_count,
                "filter_failure_rate": 100.0 * filter_failure_count / max(n, 1),
                "union_failure_count": union_failure_count,
                "union_failure_rate": 100.0 * union_failure_count / max(n, 1),
                "projection_trial_count": projection_trial_count,
                "num_post_projection_audits": num_post_projection_audits,
                "projection_event_count": projection_event_count,
                "post_projection_rejection_count": (
                    post_projection_rejection_count
                ),
                "max_projection_delta_inf": max_projection_delta_inf,
                "max_post_projection_constraint_violation": (
                    max_post_projection_constraint_violation
                ),
                "max_post_projection_violation_ratio": (
                    max_post_projection_violation_ratio
                ),
                "nominal_avg": nominal_avg,
                "mean_compute_ms": mean_compute_ms,
            }
        )
    return summary_rows, all_results


def _run_episode_payload(payload):
    """Pickle-friendly adapter for optional independent-trial parallelism."""
    variant, scenario, cfg, verbose = payload
    return run_episode(
        variant,
        scenario,
        cfg,
        verbose=verbose,
    )


def format_markdown_table(
    summary_rows: List[dict],
    num_runs: int,
    seed: int,
    cfg: SimConfig,
    detailed: bool = True,
) -> str:
    lines: List[str] = []
    lines.append("# Drift Car Additional-Baseline Results")
    lines.append("")
    lines.append(f"- Runs per algorithm: {num_runs}")
    lines.append(f"- Scenario seed: {seed}")
    lines.append(
        f"- Puddle: x={cfg.puddle_x:.1f}, radius={cfg.puddle_radius:.1f}, friction={cfg.puddle_friction:.2f}"
    )
    lines.append("- Obstacles per run: random 1 or 2, placed near and beyond puddle center (x in ~[72, 85])")
    lines.append(f"- Main failure: `{HISTORICAL_FAILURE_DEFINITION}`")
    lines.append(f"- Filter diagnostic: `{FILTER_FAILURE_DEFINITION}`")
    lines.append(
        "- Certificate loss and candidate-QP failure are diagnostics only. The "
        "simulator applies the selected baseline's returned control unchanged."
    )
    lines.append(
        "- † MB-CBF-MI uses a sampled terminal proxy, not a proven "
        "control-invariant terminal set, and therefore does not inherit the formal "
        "guarantee of Chen et al."
    )
    lines.append("")
    lines.append(
        "| Algorithm | Failure (historical) | Collision | Unrecoverable infeasible | "
        "Certificate loss | Candidate-QP failure | Goal | Horizon survival | Mean Time [ms] |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['label']} | "
            f"{_fmt_count_rate(row['fail_count'], row['n'], row['fail_rate'])} | "
            f"{_fmt_count_rate(row['collision_count'], row['n'], row['collision_rate'])} | "
            f"{_fmt_count_rate(row['infeasible_count'], row['n'], row['infeasible_rate'])} | "
            f"{_fmt_count_rate(row['certificate_lost_count'], row['n'], row['certificate_lost_rate'])} | "
            f"{_fmt_count_rate(row['qp_infeasible_count'], row['n'], row['qp_infeasible_rate'])} | "
            f"{_fmt_count_rate(row['task_completed_count'], row['n'], row['task_completed_rate'])} | "
            f"{_fmt_count_rate(row['survived_horizon_count'], row['n'], row['survived_horizon_rate'])} | "
            f"{row['mean_compute_ms']:.2f} |"
        )
    lines.append("")
    lines.append("## Post-projection QP audit")
    lines.append("")
    lines.append(
        "Every finite, actuator-tolerance-valid successful-status candidate is "
        "checked against its original QP inequalities after actuator-bound "
        "projection. A residual above the declared post-projection audit tolerance "
        "rejects that candidate before minimum-intervention selection."
    )
    lines.append("")
    lines.append(
        "| Algorithm | Audited candidates | Trials with projection | Projection "
        "events (candidates) | Audit rejections | Max "
        "$\\|\\Delta u\\|_\\infty$ (native units) | Max violation | Max "
        "violation/tolerance |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['label']} | {row['num_post_projection_audits']} | "
            f"{row['projection_trial_count']}/{row['n']} | "
            f"{row['projection_event_count']} | "
            f"{row['post_projection_rejection_count']} | "
            f"{row['max_projection_delta_inf']:.9g} | "
            f"{row['max_post_projection_constraint_violation']:.9g} | "
            f"{row['max_post_projection_violation_ratio']:.9g} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run only the two additional drift-car baselines"
    )
    parser.add_argument("--num-runs", type=int, default=10, help="Runs per algorithm")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for scenario generation")
    parser.add_argument(
        "--output-md",
        type=str,
        default="output/additional_baselines/drift_additional_baselines.md",
        help="Path to output markdown report",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="output/additional_baselines/drift_additional_baselines.json",
        help="Path to detailed per-trial JSON metrics.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="output/additional_baselines/drift_additional_baselines.csv",
        help="Path to per-trial CSV metrics.",
    )
    parser.add_argument(
        "--variant-key",
        action="append",
        default=None,
        help=(
            "Run only this baseline key; repeat to select multiple keys. By "
            "default both additional baselines run."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose per-run logging")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Independent seeded trials to run concurrently. Publication timing "
            "must use one worker."
        ),
    )
    args = parser.parse_args(argv)

    warnings.filterwarnings(
        "ignore",
        message="Solution may be inaccurate.*",
        module="cvxpy",
    )

    cfg = SimConfig()
    variants = make_variants()
    if args.variant_key:
        requested = set(args.variant_key)
        known = {variant.key for variant in variants}
        unknown = requested - known
        if unknown:
            raise ValueError(
                f"Unknown --variant-key values {sorted(unknown)}; valid keys: {sorted(known)}"
            )
        variants = [variant for variant in variants if variant.key in requested]
    scenarios = generate_scenarios(args.num_runs, args.seed)

    summary_rows, all_results = aggregate_results(
        variants,
        scenarios,
        cfg,
        verbose=args.verbose,
        num_workers=max(1, int(args.num_workers)),
    )
    markdown = format_markdown_table(
        summary_rows,
        args.num_runs,
        args.seed,
        cfg,
        detailed=True,
    )

    output_path = Path(args.output_md)
    if not output_path.is_absolute():
        output_path = Path(PROJECT_ROOT) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    json_path_arg = args.output_json
    json_path = None
    if json_path_arg is not None:
        json_path = Path(json_path_arg)
        if not json_path.is_absolute():
            json_path = Path(PROJECT_ROOT) / json_path
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_runs": args.num_runs,
            "scenario_seed": args.seed,
            "algorithms": [variant.key for variant in variants],
            "timing_note": (
                "Sequential candidates within each control step; end-to-end wall-clock "
                "filter latency. Each baseline excludes its first five warm-up calls. "
                "Independent "
                f"trials used {max(1, int(args.num_workers))} worker(s)."
            ),
            "historical_failure_definition": HISTORICAL_FAILURE_DEFINITION,
            "filter_failure_definition": FILTER_FAILURE_DEFINITION,
            "union_failure_definition": UNION_FAILURE_DEFINITION,
            "unrecoverable_infeasible_definition": UNRECOVERABLE_INFEASIBLE_DEFINITION,
            "post_diagnostic_action": "exact control returned by selected baseline",
            "projection_audit": {
                "scope": (
                    "every finite, actuator-tolerance-valid successful-status "
                    "candidate QP"
                ),
                "action": (
                    "project to exact actuator bounds, evaluate every original "
                    "affine QP inequality, and reject the candidate if any "
                    "scale-aware residual exceeds its declared tolerance"
                ),
                "osqp_absolute_tolerance": 1e-5,
                "osqp_relative_tolerance": 1e-5,
                "multi_backup_scs_fallback_absolute_tolerance": 1e-4,
                "multi_backup_scs_fallback_relative_tolerance": 1e-4,
                "drift_library_scs_absolute_tolerance": 1e-4,
                "drift_library_scs_relative_tolerance": 1e-4,
            },
            "config": asdict(cfg),
            "summary": summary_rows,
            "trials": {
                key: [asdict(result) for result in results]
                for key, results in all_results.items()
            },
        }
        json_path.write_text(
            json.dumps(_json_safe(payload), indent=2, allow_nan=False),
            encoding="utf-8",
        )

    csv_path_arg = args.output_csv
    csv_path = None
    if csv_path_arg is not None:
        csv_path = Path(csv_path_arg)
        if not csv_path.is_absolute():
            csv_path = Path(PROJECT_ROOT) / csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for results in all_results.values():
            for result in results:
                row = asdict(result)
                for key, value in list(row.items()):
                    if isinstance(value, (dict, list, tuple)):
                        row[key] = json.dumps(_json_safe(value), sort_keys=True)
                    elif isinstance(value, float) and not np.isfinite(value):
                        row[key] = ""
                rows.append(row)
        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=list(rows[0].keys()),
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)

    print(markdown)
    print(f"\nSaved markdown report to: {output_path}")
    if json_path is not None:
        print(f"Saved detailed JSON metrics to: {json_path}")
    if csv_path is not None:
        print(f"Saved per-trial CSV metrics to: {csv_path}")


if __name__ == "__main__":
    main()
