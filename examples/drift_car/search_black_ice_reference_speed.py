"""
Reference-speed search for black-ice drift benchmark.

Goal:
- Keep the existing black-ice benchmark setup unchanged (same algorithms,
  same fallback policies, same scenario randomization seed/rules).
- Search for the largest reference speed v_ref (shared by vehicle + MPCC)
  such that each algorithm variant has zero collision/infeasible outcomes
  over benchmark trials.

Termination rules for this script only:
- Success if x-position passes threshold (default: x >= 100 m).
- Success if the robot stays inside the ice area for a long duration
  without collision/infeasible.
- Success if the robot stays near the ice with near-zero velocity for
  a sustained duration (to avoid uninformative long timeouts).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
import sys
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add project root and submodule path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "safe_control"))

from safe_control.position_control.mpcc import MPCC
from safe_control.robots.drifting_car import DriftingCar, DriftingCarSimulator

import examples.drift_car.benchmark_black_ice as bb


@dataclass
class EpisodeOutcome:
    collision: bool
    infeasible: bool
    passed_x_threshold: bool
    stuck_success: bool
    stationary_success: bool
    timeout_success: bool
    nominal_tracking_pct: float
    total_steps: int

    @property
    def failed(self) -> bool:
        return bool(self.collision or self.infeasible or self.timeout_success)


@dataclass
class SpeedEvalSummary:
    v_ref: float
    n_trials: int
    fail_count: int
    collision_count: int
    infeasible_count: int
    avg_nominal_tracking_pct: float
    avg_steps: float

    @property
    def pass_all(self) -> bool:
        return self.fail_count == 0


@dataclass
class VariantSpeedResult:
    key: str
    label: str
    largest_passing_v_ref: Optional[float]
    start_v_ref: float
    num_speed_evals: int
    pass_summary: Optional[SpeedEvalSummary]
    searched_speeds_desc: List[float]


@dataclass
class AnalyticEstimate:
    obstacle_x_min: float
    obstacle_x_max: float
    ice_entry_x: float
    worst_center_distance_m: float
    worst_clearance_distance_m: float
    mu_analysis: float
    mu_actual: float
    v_est_analysis_mu: float
    v_est_actual_mu: float


def _build_vehicle_spec_with_ref_speed(v_ref: float) -> Dict[str, float]:
    spec = bb.build_vehicle_spec()
    spec["v_ref"] = float(v_ref)
    return spec


def _setup_mpcc_with_ref_speed(
    car: DriftingCar,
    env,
    lanes: Dict[str, float],
    cfg: bb.SimConfig,
    v_ref: float,
) -> MPCC:
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
        v_ref=float(v_ref),
        R=np.array([300.0, 0.5, 0.1]),
    )
    mpcc.set_progress_rate(float(v_ref))
    return mpcc


def _is_inside_ice(x_pos: float, cfg: bb.SimConfig) -> bool:
    return (cfg.puddle_x - cfg.puddle_radius) <= x_pos <= (cfg.puddle_x + cfg.puddle_radius)


def run_episode_for_speed(
    *,
    variant: bb.AlgoVariant,
    scenario: bb.Scenario,
    cfg: bb.SimConfig,
    v_ref: float,
    max_steps: int,
    success_x_threshold: float,
    ice_stuck_steps: int,
    stationary_stuck_steps: int,
    stationary_speed_threshold: float,
    stationary_x_buffer: float,
    allow_stationary_success: bool,
    initial_x_default: float,
    slow_start_speed_threshold: float,
    slow_start_x: float,
    verbose: bool = False,
) -> EpisodeOutcome:
    env, lanes = bb.setup_env_and_lanes(cfg)
    bb.add_black_ice_and_obstacles(env, lanes, cfg, scenario)

    vehicle_spec = _build_vehicle_spec_with_ref_speed(v_ref)
    x0 = bb.make_initial_state(lanes, replace(cfg, initial_velocity=float(v_ref)))
    start_x = float(initial_x_default)
    if float(v_ref) <= float(slow_start_speed_threshold):
        start_x = float(slow_start_x)
    x0[0] = start_x
    car = DriftingCar(x0, vehicle_spec, cfg.dt, ax=None)
    simulator = DriftingCarSimulator(car, env, show_animation=False)

    mpcc = _setup_mpcc_with_ref_speed(car, env, lanes, cfg, v_ref=v_ref)
    shielding = bb.setup_shielding(variant, car, env, lanes, cfg)

    collision = False
    infeasible = False
    passed_x_threshold = False
    stuck_success = False
    stationary_success = False
    timeout_success = False

    total_steps = 0
    nominal_like_steps = 0
    ice_dwell_steps = 0
    stationary_dwell_steps = 0

    u_scale = np.array(
        [
            float(vehicle_spec["delta_dot_max"]),
            float(vehicle_spec["tau_dot_max"]),
        ],
        dtype=float,
    )

    quiet = not verbose
    for _step in range(max_steps):
        state = car.get_state()
        pos = car.get_position()

        curr_mu = env.get_friction_at_position(pos, default_friction=vehicle_spec["mu"])
        if abs(curr_mu - car.get_friction()) > 1e-8:
            car.set_friction(curr_mu)
            if hasattr(shielding, "set_friction"):
                shielding.set_friction(curr_mu)

        try:
            u_nom = bb.call_quiet(quiet, mpcc.solve_control_problem, state)
            pred_states, pred_controls = bb.call_quiet(quiet, mpcc.get_full_predictions)
        except Exception:
            infeasible = True
            break

        if variant.algo in ("backup_cbf", "mps", "gatekeeper"):
            if pred_states is not None and pred_controls is not None:
                shielding.set_nominal_trajectory(pred_states, pred_controls)

        try:
            u_safe = bb.call_quiet(
                quiet,
                bb.solve_safe_control,
                variant=variant,
                shielding=shielding,
                state=state,
                u_nom=u_nom,
                pred_states=pred_states,
                pred_controls=pred_controls,
                friction=car.get_friction(),
            )
        except Exception:
            infeasible = True
            break

        u_nom_vec = np.array(u_nom).flatten()
        u_safe_vec = np.array(u_safe).flatten()
        if (not np.all(np.isfinite(u_safe_vec))) or (u_safe_vec.shape[0] != 2):
            infeasible = True
            break

        diff = np.linalg.norm((u_safe_vec - u_nom_vec) / np.maximum(u_scale, 1e-8))
        if diff < cfg.nominal_track_eps:
            nominal_like_steps += 1
        total_steps += 1

        sim_res = simulator.step(np.array(u_safe).reshape(-1, 1))
        collision = bool(sim_res["collision"])
        if collision:
            break

        x_pos = float(car.get_position()[0])
        if x_pos >= success_x_threshold:
            passed_x_threshold = True
            break

        if ice_stuck_steps > 0 and _is_inside_ice(x_pos, cfg):
            ice_dwell_steps += 1
            if ice_dwell_steps >= ice_stuck_steps:
                stuck_success = True
                break
        else:
            ice_dwell_steps = 0

        near_ice = (
            (cfg.puddle_x - cfg.puddle_radius - stationary_x_buffer)
            <= x_pos
            <= (cfg.puddle_x + cfg.puddle_radius + stationary_x_buffer)
        )
        if (
            allow_stationary_success
            and near_ice
            and abs(float(car.get_velocity())) <= stationary_speed_threshold
        ):
            stationary_dwell_steps += 1
            if stationary_dwell_steps >= stationary_stuck_steps:
                stationary_success = True
                break
        else:
            stationary_dwell_steps = 0

    if (
        not collision
        and not infeasible
        and not passed_x_threshold
        and not stuck_success
        and not stationary_success
    ):
        timeout_success = True

    nominal_pct = 100.0 * nominal_like_steps / max(total_steps, 1)
    return EpisodeOutcome(
        collision=collision,
        infeasible=infeasible,
        passed_x_threshold=passed_x_threshold,
        stuck_success=stuck_success,
        stationary_success=stationary_success,
        timeout_success=timeout_success,
        nominal_tracking_pct=nominal_pct,
        total_steps=total_steps,
    )


def run_nominal_tracking_sanity_for_speed(
    *,
    cfg: bb.SimConfig,
    v_ref: float,
    nominal_sanity_seconds: float,
    initial_x_default: float,
    slow_start_speed_threshold: float,
    slow_start_x: float,
) -> tuple[bool, float, float]:
    """Check that obstacle-free nominal control can track the requested speed."""
    env, lanes = bb.setup_env_and_lanes(cfg)
    env.add_puddle(
        x=cfg.puddle_x,
        y=lanes["middle"],
        radius=cfg.puddle_radius,
        friction=cfg.puddle_friction,
    )

    vehicle_spec = _build_vehicle_spec_with_ref_speed(v_ref)
    x0 = bb.make_initial_state(lanes, replace(cfg, initial_velocity=float(v_ref)))
    x0[0] = float(slow_start_x if v_ref <= slow_start_speed_threshold else initial_x_default)
    car = DriftingCar(x0, vehicle_spec, cfg.dt, ax=None)
    simulator = DriftingCarSimulator(car, env, show_animation=False)
    mpcc = _setup_mpcc_with_ref_speed(car, env, lanes, cfg, v_ref=v_ref)

    n_steps = max(1, int(round(float(nominal_sanity_seconds) / cfg.dt)))
    speeds_abs: List[float] = []
    for _ in range(n_steps):
        state = car.get_state()
        pos = car.get_position()
        curr_mu = env.get_friction_at_position(pos, default_friction=vehicle_spec["mu"])
        if abs(curr_mu - car.get_friction()) > 1e-8:
            car.set_friction(curr_mu)

        try:
            u_nom = bb.call_quiet(True, mpcc.solve_control_problem, state)
        except Exception:
            return False, 0.0, 0.0

        sim_res = simulator.step(np.array(u_nom).reshape(-1, 1))
        if bool(sim_res["collision"]):
            return False, 0.0, 0.0
        speeds_abs.append(abs(float(car.get_velocity())))

    if not speeds_abs:
        return False, 0.0, 0.0

    tail = speeds_abs[len(speeds_abs) // 2 :]
    mean_speed_abs = float(np.mean(tail if tail else speeds_abs))
    std_speed_abs = float(np.std(tail if tail else speeds_abs))
    target = float(v_ref)
    min_expected_speed = max(0.05, 0.35 * target)
    err_allowed = max(0.08, 0.35 * target)
    std_allowed = max(0.06, 0.40 * target)
    tracking_ok = (
        mean_speed_abs >= min_expected_speed
        and abs(mean_speed_abs - target) <= err_allowed
        and std_speed_abs <= std_allowed
    )
    return tracking_ok, mean_speed_abs, std_speed_abs


def evaluate_variant_at_speed(
    *,
    variant: bb.AlgoVariant,
    scenarios: List[bb.Scenario],
    cfg: bb.SimConfig,
    v_ref: float,
    max_steps: int,
    success_x_threshold: float,
    ice_stuck_steps: int,
    stationary_stuck_steps: int,
    stationary_speed_threshold: float,
    stationary_x_buffer: float,
    nominal_sanity_seconds: float,
    initial_x_default: float,
    slow_start_speed_threshold: float,
    slow_start_x: float,
    max_trials: Optional[int] = None,
    fail_fast: bool = True,
    num_workers: int = 1,
    verbose: bool = False,
) -> SpeedEvalSummary:
    outcomes: List[EpisodeOutcome] = []
    run_scenarios = scenarios if max_trials is None else scenarios[:max_trials]
    nominal_ok, nominal_mean_speed, nominal_std_speed = run_nominal_tracking_sanity_for_speed(
        cfg=cfg,
        v_ref=v_ref,
        nominal_sanity_seconds=nominal_sanity_seconds,
        initial_x_default=initial_x_default,
        slow_start_speed_threshold=slow_start_speed_threshold,
        slow_start_x=slow_start_x,
    )
    if verbose:
        state = "PASS" if nominal_ok else "FAIL"
        print(
            f"    nominal sanity ({state}): mean |V|={nominal_mean_speed:.3f} m/s "
            f"(std={nominal_std_speed:.3f}) for v_ref={v_ref:.3f}; "
            f"stationary success enabled={nominal_ok}"
        )
    if num_workers <= 1:
        for sc in run_scenarios:
            ep = run_episode_for_speed(
                variant=variant,
                scenario=sc,
                cfg=cfg,
                v_ref=v_ref,
                max_steps=max_steps,
                success_x_threshold=success_x_threshold,
                ice_stuck_steps=ice_stuck_steps,
                stationary_stuck_steps=stationary_stuck_steps,
                stationary_speed_threshold=stationary_speed_threshold,
                stationary_x_buffer=stationary_x_buffer,
                allow_stationary_success=nominal_ok,
                initial_x_default=initial_x_default,
                slow_start_speed_threshold=slow_start_speed_threshold,
                slow_start_x=slow_start_x,
                verbose=verbose,
            )
            outcomes.append(ep)
            if fail_fast and ep.failed:
                break
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            batch_size = max(1, int(num_workers))
            for i in range(0, len(run_scenarios), batch_size):
                batch = run_scenarios[i:i + batch_size]
                futures = [
                    ex.submit(
                        run_episode_for_speed,
                        variant=variant,
                        scenario=sc,
                        cfg=cfg,
                        v_ref=v_ref,
                        max_steps=max_steps,
                        success_x_threshold=success_x_threshold,
                        ice_stuck_steps=ice_stuck_steps,
                        stationary_stuck_steps=stationary_stuck_steps,
                        stationary_speed_threshold=stationary_speed_threshold,
                        stationary_x_buffer=stationary_x_buffer,
                        allow_stationary_success=nominal_ok,
                        initial_x_default=initial_x_default,
                        slow_start_speed_threshold=slow_start_speed_threshold,
                        slow_start_x=slow_start_x,
                        verbose=False,
                    )
                    for sc in batch
                ]
                batch_failed = False
                for fut in futures:
                    try:
                        ep = fut.result()
                    except Exception:
                        ep = EpisodeOutcome(
                            collision=False,
                            infeasible=True,
                            passed_x_threshold=False,
                            stuck_success=False,
                            stationary_success=False,
                            timeout_success=False,
                            nominal_tracking_pct=0.0,
                            total_steps=0,
                        )
                    outcomes.append(ep)
                    if fail_fast and ep.failed:
                        batch_failed = True
                if fail_fast and batch_failed:
                    break

    n = len(outcomes)
    collision_count = sum(int(o.collision) for o in outcomes)
    infeasible_count = sum(int(o.infeasible) for o in outcomes)
    fail_count = sum(int(o.failed) for o in outcomes)
    avg_nom = float(np.mean([o.nominal_tracking_pct for o in outcomes])) if outcomes else 0.0
    avg_steps = float(np.mean([o.total_steps for o in outcomes])) if outcomes else 0.0
    return SpeedEvalSummary(
        v_ref=float(v_ref),
        n_trials=n,
        fail_count=fail_count,
        collision_count=collision_count,
        infeasible_count=infeasible_count,
        avg_nominal_tracking_pct=avg_nom,
        avg_steps=avg_steps,
    )


def _build_speed_grid(v_min: float, v_max: float, step: float) -> List[float]:
    n = int(round((v_max - v_min) / step))
    vals = [v_min + i * step for i in range(n + 1)]
    return [float(np.round(v, 6)) for v in vals]


def _nearest_grid_speed(v: float, grid: List[float]) -> float:
    arr = np.array(grid, dtype=float)
    return float(arr[np.argmin(np.abs(arr - v))])


def generate_fixed_six_scenarios() -> List[bb.Scenario]:
    """Six representative obstacle layouts requested by user."""
    cases = [
        ((80.0, "middle"),),
        ((80.0, "left"),),
        ((80.0, "right"),),
        ((80.0, "middle"), (84.0, "left")),
        ((80.0, "left"), (84.0, "right")),
        ((80.0, "middle"), (84.0, "right")),
    ]
    scenarios: List[bb.Scenario] = []
    for idx, obs in enumerate(cases):
        scenarios.append(
            bb.Scenario(
                run_idx=idx,
                seed=idx,
                num_obstacles=len(obs),
                obstacles=tuple((float(x), str(lane)) for x, lane in obs),
            )
        )
    return scenarios


def compute_analytic_estimate(
    scenarios: List[bb.Scenario],
    cfg: bb.SimConfig,
    mu_analysis: float = 0.25,
) -> AnalyticEstimate:
    obstacle_x_all: List[float] = []
    for sc in scenarios:
        for x_obs, _lane in sc.obstacles:
            obstacle_x_all.append(float(x_obs))

    obstacle_x_min = float(np.min(obstacle_x_all))
    obstacle_x_max = float(np.max(obstacle_x_all))

    ice_entry_x = float(cfg.puddle_x - cfg.puddle_radius)
    worst_center_distance = max(0.0, obstacle_x_min - ice_entry_x)

    robot_radius = float(bb.build_vehicle_spec()["radius"])
    obstacle_radius = 2.0  # matches add_black_ice_and_obstacles
    worst_clearance = max(0.0, worst_center_distance - (robot_radius + obstacle_radius))

    g = 9.81
    v_est_analysis_mu = math.sqrt(max(0.0, 2.0 * mu_analysis * g * worst_clearance))
    v_est_actual_mu = math.sqrt(max(0.0, 2.0 * cfg.puddle_friction * g * worst_clearance))

    return AnalyticEstimate(
        obstacle_x_min=obstacle_x_min,
        obstacle_x_max=obstacle_x_max,
        ice_entry_x=ice_entry_x,
        worst_center_distance_m=worst_center_distance,
        worst_clearance_distance_m=worst_clearance,
        mu_analysis=float(mu_analysis),
        mu_actual=float(cfg.puddle_friction),
        v_est_analysis_mu=float(v_est_analysis_mu),
        v_est_actual_mu=float(v_est_actual_mu),
    )


def search_largest_passing_speed_reverse(
    *,
    variant: bb.AlgoVariant,
    scenarios: List[bb.Scenario],
    cfg: bb.SimConfig,
    speed_grid: List[float],
    start_speed: float,
    max_steps: int,
    success_x_threshold: float,
    ice_stuck_steps: int,
    stationary_stuck_steps: int,
    stationary_speed_threshold: float,
    stationary_x_buffer: float,
    nominal_sanity_seconds: float,
    initial_x_default: float,
    slow_start_speed_threshold: float,
    slow_start_x: float,
    num_workers: int,
    verbose: bool = False,
) -> VariantSpeedResult:
    cache: Dict[float, SpeedEvalSummary] = {}
    searched_order: List[float] = []

    def _eval(v: float) -> SpeedEvalSummary:
        if v not in cache:
            if verbose:
                print(f"  eval v_ref={v:.3f} ...")
            cache[v] = evaluate_variant_at_speed(
                variant=variant,
                scenarios=scenarios,
                cfg=cfg,
                v_ref=v,
                max_steps=max_steps,
                success_x_threshold=success_x_threshold,
                ice_stuck_steps=ice_stuck_steps,
                stationary_stuck_steps=stationary_stuck_steps,
                stationary_speed_threshold=stationary_speed_threshold,
                stationary_x_buffer=stationary_x_buffer,
                nominal_sanity_seconds=nominal_sanity_seconds,
                initial_x_default=initial_x_default,
                slow_start_speed_threshold=slow_start_speed_threshold,
                slow_start_x=slow_start_x,
                fail_fast=True,
                num_workers=num_workers,
                verbose=False,
            )
        searched_order.append(v)
        return cache[v]

    start_v = _nearest_grid_speed(start_speed, speed_grid)
    idx_start = speed_grid.index(start_v)

    start_summary = _eval(start_v)
    best_v: Optional[float] = None

    if start_summary.pass_all:
        best_v = start_v
        for idx in range(idx_start + 1, len(speed_grid)):
            v = speed_grid[idx]
            s = _eval(v)
            if s.pass_all:
                best_v = v
            else:
                break
    else:
        for idx in range(idx_start - 1, -1, -1):
            v = speed_grid[idx]
            s = _eval(v)
            if s.pass_all:
                best_v = v
                break

    pass_summary = cache.get(best_v) if best_v is not None else None
    return VariantSpeedResult(
        key=variant.key,
        label=variant.label,
        largest_passing_v_ref=best_v,
        start_v_ref=start_v,
        num_speed_evals=len(searched_order),
        pass_summary=pass_summary,
        searched_speeds_desc=searched_order,
    )


def search_largest_passing_speed_sparse(
    *,
    variant: bb.AlgoVariant,
    scenarios: List[bb.Scenario],
    cfg: bb.SimConfig,
    speed_grid: List[float],
    max_steps: int,
    success_x_threshold: float,
    ice_stuck_steps: int,
    stationary_stuck_steps: int,
    stationary_speed_threshold: float,
    stationary_x_buffer: float,
    nominal_sanity_seconds: float,
    initial_x_default: float,
    slow_start_speed_threshold: float,
    slow_start_x: float,
    coarse_trials: int,
    num_workers: int,
    verbose: bool = False,
) -> VariantSpeedResult:
    """Robust descending search with coarse screening then full verification."""
    cache_coarse: Dict[float, SpeedEvalSummary] = {}
    cache_full: Dict[float, SpeedEvalSummary] = {}
    searched_order: List[float] = []

    v_min = float(min(speed_grid))
    v_max = float(max(speed_grid))
    coarse_n = max(1, min(coarse_trials, len(scenarios)))

    def _snap(v: float) -> float:
        return _nearest_grid_speed(float(np.clip(v, v_min, v_max)), speed_grid)

    def _eval_coarse(v: float) -> SpeedEvalSummary:
        v = _snap(v)
        if v not in cache_coarse:
            if verbose:
                print(f"  eval coarse v_ref={v:.3f} ...")
            cache_coarse[v] = evaluate_variant_at_speed(
                variant=variant,
                scenarios=scenarios,
                cfg=cfg,
                v_ref=v,
                max_steps=max_steps,
                success_x_threshold=success_x_threshold,
                ice_stuck_steps=ice_stuck_steps,
                stationary_stuck_steps=stationary_stuck_steps,
                stationary_speed_threshold=stationary_speed_threshold,
                stationary_x_buffer=stationary_x_buffer,
                nominal_sanity_seconds=nominal_sanity_seconds,
                initial_x_default=initial_x_default,
                slow_start_speed_threshold=slow_start_speed_threshold,
                slow_start_x=slow_start_x,
                max_trials=coarse_n,
                fail_fast=True,
                num_workers=num_workers,
                verbose=False,
            )
        searched_order.append(v)
        return cache_coarse[v]

    def _eval_full(v: float) -> SpeedEvalSummary:
        v = _snap(v)
        if v not in cache_full:
            if verbose:
                print(f"  eval full v_ref={v:.3f} ...")
            cache_full[v] = evaluate_variant_at_speed(
                variant=variant,
                scenarios=scenarios,
                cfg=cfg,
                v_ref=v,
                max_steps=max_steps,
                success_x_threshold=success_x_threshold,
                ice_stuck_steps=ice_stuck_steps,
                stationary_stuck_steps=stationary_stuck_steps,
                stationary_speed_threshold=stationary_speed_threshold,
                stationary_x_buffer=stationary_x_buffer,
                nominal_sanity_seconds=nominal_sanity_seconds,
                initial_x_default=initial_x_default,
                slow_start_speed_threshold=slow_start_speed_threshold,
                slow_start_x=slow_start_x,
                max_trials=None,
                fail_fast=True,
                num_workers=num_workers,
                verbose=False,
            )
        searched_order.append(v)
        return cache_full[v]

    start_v = _snap(5.0)
    best_v: Optional[float] = None
    v_upper = v_max if variant.algo == "plcbf" else min(5.0, v_max)
    speeds_desc = [v for v in sorted(speed_grid, reverse=True) if v <= v_upper + 1e-9]
    for v in speeds_desc:
        c = _eval_coarse(v)
        if not c.pass_all:
            continue
        s = _eval_full(v)
        if s.pass_all:
            best_v = v
            break

    return VariantSpeedResult(
        key=variant.key,
        label=variant.label,
        largest_passing_v_ref=best_v,
        start_v_ref=start_v,
        num_speed_evals=len(searched_order),
        pass_summary=cache_full.get(best_v),
        searched_speeds_desc=searched_order,
    )


def format_markdown_prefix(
    *,
    cfg: bb.SimConfig,
    num_trials: int,
    seed: int,
    scenario_desc: str,
    max_steps: int,
    success_x_threshold: float,
    ice_stuck_steps: int,
    stationary_stuck_steps: int,
    stationary_speed_threshold: float,
    stationary_x_buffer: float,
    initial_x_default: float,
    slow_start_speed_threshold: float,
    slow_start_x: float,
    speed_min: float,
    speed_max: float,
    speed_step: float,
    analytic: AnalyticEstimate,
) -> List[str]:
    lines: List[str] = []
    lines.append("# Drift Car Black-Ice Reference-Speed Search")
    lines.append("")
    lines.append("This report searches the largest passing reference speed `v_ref` per algorithm variant.")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Trials per variant: {num_trials}")
    lines.append(f"- Scenario set: {scenario_desc}")
    lines.append(f"- Scenario seed: {seed}")
    lines.append(
        f"- Puddle: x={cfg.puddle_x:.1f}, radius={cfg.puddle_radius:.1f}, friction={cfg.puddle_friction:.2f}"
    )
    lines.append(
        f"- Speed grid: [{speed_min:.2f}, {speed_max:.2f}] step {speed_step:.2f} m/s"
    )
    lines.append(
        f"- Simulation max steps: {max_steps} (dt={cfg.dt:.2f}s, horizon={max_steps * cfg.dt:.1f}s)"
    )
    lines.append(f"- Success condition A: x >= {success_x_threshold:.1f} m")
    if ice_stuck_steps > 0:
        lines.append(
            f"- Success condition B: continuous dwell in ice >= {ice_stuck_steps} steps ({ice_stuck_steps * cfg.dt:.1f}s)"
        )
    else:
        lines.append("- Success condition B: disabled")
    lines.append(
        f"- Success condition C: near-ice low-speed dwell >= {stationary_stuck_steps} steps "
        f"({stationary_stuck_steps * cfg.dt:.1f}s), |V| <= {stationary_speed_threshold:.2f} m/s, "
        f"x-buffer ±{stationary_x_buffer:.1f} m"
    )
    lines.append(
        f"- Start position policy: x={initial_x_default:.1f} by default; "
        f"for v_ref <= {slow_start_speed_threshold:.2f} m/s, start at x={slow_start_x:.1f}"
    )
    lines.append("")
    lines.append("## Analytic Start Speed")
    lines.append(f"- Obstacle x-range from scenarios: [{analytic.obstacle_x_min:.2f}, {analytic.obstacle_x_max:.2f}]")
    lines.append(f"- Ice entry x: {analytic.ice_entry_x:.2f}")
    lines.append(
        f"- Worst center distance at ice entry: {analytic.worst_center_distance_m:.2f} m "
        "(entry to nearest obstacle center)"
    )
    lines.append(
        f"- Worst clearance distance (subtracting robot+obstacle radii): {analytic.worst_clearance_distance_m:.2f} m"
    )
    lines.append(
        f"- Estimated safe entry speed (mu={analytic.mu_analysis:.2f}): {analytic.v_est_analysis_mu:.2f} m/s"
    )
    lines.append(
        f"- Estimated safe entry speed (actual puddle mu={analytic.mu_actual:.2f}): "
        f"{analytic.v_est_actual_mu:.2f} m/s"
    )
    lines.append("")
    lines.append(
        "| Variant | Largest Passing v_ref (m/s) | Verified Trials | "
        "Collision/Infeasible at Passing Speed | Search Start (m/s) | Speed Evals |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    return lines


def format_variant_row(r: VariantSpeedResult) -> str:
    if r.pass_summary is None or r.largest_passing_v_ref is None:
        return f"| {r.label} | NOT FOUND | - | - | {r.start_v_ref:.2f} | {r.num_speed_evals} |"

    ps = r.pass_summary
    return (
        f"| {r.label} | {r.largest_passing_v_ref:.2f} | {ps.n_trials} | "
        f"{ps.fail_count}/{ps.n_trials} | {r.start_v_ref:.2f} | {r.num_speed_evals} |"
    )


def format_markdown_report(
    *,
    cfg: bb.SimConfig,
    num_trials: int,
    seed: int,
    scenario_desc: str,
    max_steps: int,
    success_x_threshold: float,
    ice_stuck_steps: int,
    stationary_stuck_steps: int,
    stationary_speed_threshold: float,
    stationary_x_buffer: float,
    initial_x_default: float,
    slow_start_speed_threshold: float,
    slow_start_x: float,
    speed_min: float,
    speed_max: float,
    speed_step: float,
    analytic: AnalyticEstimate,
    variant_results: List[VariantSpeedResult],
) -> str:
    lines = format_markdown_prefix(
        cfg=cfg,
        num_trials=num_trials,
        seed=seed,
        scenario_desc=scenario_desc,
        max_steps=max_steps,
        success_x_threshold=success_x_threshold,
        ice_stuck_steps=ice_stuck_steps,
        stationary_stuck_steps=stationary_stuck_steps,
        stationary_speed_threshold=stationary_speed_threshold,
        stationary_x_buffer=stationary_x_buffer,
        initial_x_default=initial_x_default,
        slow_start_speed_threshold=slow_start_speed_threshold,
        slow_start_x=slow_start_x,
        speed_min=speed_min,
        speed_max=speed_max,
        speed_step=speed_step,
        analytic=analytic,
    )
    for r in variant_results:
        lines.append(format_variant_row(r))
    lines.append("")
    return "\n".join(lines)


def append_rows_markdown(
    *,
    output_md: Path,
    cfg: bb.SimConfig,
    num_trials: int,
    seed: int,
    scenario_desc: str,
    max_steps: int,
    success_x_threshold: float,
    ice_stuck_steps: int,
    stationary_stuck_steps: int,
    stationary_speed_threshold: float,
    stationary_x_buffer: float,
    initial_x_default: float,
    slow_start_speed_threshold: float,
    slow_start_x: float,
    speed_min: float,
    speed_max: float,
    speed_step: float,
    analytic: AnalyticEstimate,
    variant_results: List[VariantSpeedResult],
):
    if (not output_md.exists()) or output_md.stat().st_size == 0:
        prefix = format_markdown_prefix(
            cfg=cfg,
            num_trials=num_trials,
            seed=seed,
            scenario_desc=scenario_desc,
            max_steps=max_steps,
            success_x_threshold=success_x_threshold,
            ice_stuck_steps=ice_stuck_steps,
            stationary_stuck_steps=stationary_stuck_steps,
            stationary_speed_threshold=stationary_speed_threshold,
            stationary_x_buffer=stationary_x_buffer,
            initial_x_default=initial_x_default,
            slow_start_speed_threshold=slow_start_speed_threshold,
            slow_start_x=slow_start_x,
            speed_min=speed_min,
            speed_max=speed_max,
            speed_step=speed_step,
            analytic=analytic,
        )
        output_md.write_text("\n".join(prefix) + "\n", encoding="utf-8")

    with output_md.open("a", encoding="utf-8") as f:
        for r in variant_results:
            f.write(format_variant_row(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Search largest passing v_ref for black-ice drift benchmark")
    parser.add_argument("--num-runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--scenario-mode",
        type=str,
        default="fixed6",
        choices=["fixed6", "random"],
        help="Use six representative fixed layouts or random benchmark scenarios",
    )
    parser.add_argument("--speed-min", type=float, default=0.25)
    parser.add_argument("--speed-max", type=float, default=10.0)
    parser.add_argument("--speed-step", type=float, default=0.25)
    parser.add_argument(
        "--mu-analysis",
        type=float,
        default=0.25,
        help="Conservative mu used only for analytic initial speed estimate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60000,
        help="Very long simulation cap for this search script",
    )
    parser.add_argument("--success-x-threshold", type=float, default=100.0)
    parser.add_argument(
        "--ice-stuck-seconds",
        type=float,
        default=0.0,
        help="Continuous seconds inside ice area to count as success (<=0 disables)",
    )
    parser.add_argument(
        "--stationary-seconds",
        type=float,
        default=2.0,
        help="Continuous seconds of low speed in/near ice to count as success",
    )
    parser.add_argument(
        "--stationary-speed-threshold",
        type=float,
        default=0.08,
        help="Low-speed threshold for near-ice stationary success condition",
    )
    parser.add_argument(
        "--stationary-x-buffer",
        type=float,
        default=0.0,
        help="Extra x-buffer around puddle bounds for stationary success condition",
    )
    parser.add_argument(
        "--nominal-sanity-seconds",
        type=float,
        default=8.0,
        help="Obstacle-free nominal tracking check duration before enabling stationary success",
    )
    parser.add_argument(
        "--initial-x-default",
        type=float,
        default=5.0,
        help="Default robot start x-position for normal-speed runs",
    )
    parser.add_argument(
        "--slow-start-speed-threshold",
        type=float,
        default=2.0,
        help="For v_ref <= threshold, start closer to ice to speed up low-speed sanity checks",
    )
    parser.add_argument(
        "--slow-start-x",
        type=float,
        default=45.0,
        help="Start x-position used for low-speed runs (<= slow-start-speed-threshold)",
    )
    parser.add_argument(
        "--search-mode",
        type=str,
        default="reverse",
        choices=["reverse", "sparse"],
        help="Speed search strategy",
    )
    parser.add_argument(
        "--coarse-trials",
        type=int,
        default=5,
        help="Sparse mode only: trial count for coarse screening before full 50-trial validation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel workers for full-trial validation (algorithms still run one-by-one)",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="examples/drift_car/benchmark_black_ice_speed_search_results.md",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="examples/drift_car/benchmark_black_ice_speed_search_results.json",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--variant-key",
        type=str,
        default=None,
        help="Run only one variant key (e.g., plcbf, backup_cbf_stop, mps_lane_change_left)",
    )
    parser.add_argument(
        "--append-md",
        action="store_true",
        help="Append result rows to existing markdown instead of overwriting full report",
    )
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore",
        message="Solution may be inaccurate.*",
        module="cvxpy",
    )

    cfg = bb.SimConfig()
    variants = bb.make_variants()
    if args.variant_key is not None:
        matches = [v for v in variants if v.key == args.variant_key]
        if not matches:
            valid = ", ".join(v.key for v in variants)
            raise ValueError(f"Unknown --variant-key '{args.variant_key}'. Valid keys: {valid}")
        variants = matches

    if args.scenario_mode == "fixed6":
        scenarios = generate_fixed_six_scenarios()
        scenario_desc = "fixed representative 6 cases (3 single-obstacle + 3 two-obstacle)"
    else:
        scenarios = bb.generate_scenarios(args.num_runs, args.seed)
        scenario_desc = "randomized benchmark scenarios"
    num_trials = len(scenarios)
    ice_stuck_steps = int(round(args.ice_stuck_seconds / cfg.dt)) if args.ice_stuck_seconds > 0.0 else 0
    stationary_stuck_steps = max(1, int(round(args.stationary_seconds / cfg.dt)))
    speed_grid = _build_speed_grid(args.speed_min, args.speed_max, args.speed_step)

    analytic = compute_analytic_estimate(
        scenarios=scenarios,
        cfg=cfg,
        mu_analysis=args.mu_analysis,
    )
    start_speed = analytic.v_est_analysis_mu

    variant_results: List[VariantSpeedResult] = []
    for variant in variants:
        print(f"\n=== {variant.label} ===")
        if args.search_mode == "sparse":
            result = search_largest_passing_speed_sparse(
                variant=variant,
                scenarios=scenarios,
                cfg=cfg,
                speed_grid=speed_grid,
                max_steps=args.max_steps,
                success_x_threshold=args.success_x_threshold,
                ice_stuck_steps=ice_stuck_steps,
                stationary_stuck_steps=stationary_stuck_steps,
                stationary_speed_threshold=args.stationary_speed_threshold,
                stationary_x_buffer=args.stationary_x_buffer,
                nominal_sanity_seconds=args.nominal_sanity_seconds,
                initial_x_default=args.initial_x_default,
                slow_start_speed_threshold=args.slow_start_speed_threshold,
                slow_start_x=args.slow_start_x,
                coarse_trials=args.coarse_trials,
                num_workers=max(1, int(args.num_workers)),
                verbose=args.verbose,
            )
        else:
            result = search_largest_passing_speed_reverse(
                variant=variant,
                scenarios=scenarios,
                cfg=cfg,
                speed_grid=speed_grid,
                start_speed=start_speed,
                max_steps=args.max_steps,
                success_x_threshold=args.success_x_threshold,
                ice_stuck_steps=ice_stuck_steps,
                stationary_stuck_steps=stationary_stuck_steps,
                stationary_speed_threshold=args.stationary_speed_threshold,
                stationary_x_buffer=args.stationary_x_buffer,
                nominal_sanity_seconds=args.nominal_sanity_seconds,
                initial_x_default=args.initial_x_default,
                slow_start_speed_threshold=args.slow_start_speed_threshold,
                slow_start_x=args.slow_start_x,
                num_workers=max(1, int(args.num_workers)),
                verbose=args.verbose,
            )

        if result.largest_passing_v_ref is None:
            print("  largest passing v_ref: NOT FOUND in grid")
        else:
            print(f"  largest passing v_ref: {result.largest_passing_v_ref:.2f} m/s")
        variant_results.append(result)

    output_md = Path(args.output_md)
    if not output_md.is_absolute():
        output_md = Path(PROJECT_ROOT) / output_md
    output_md.parent.mkdir(parents=True, exist_ok=True)
    if args.append_md:
        append_rows_markdown(
            output_md=output_md,
            cfg=cfg,
            num_trials=num_trials,
            seed=args.seed,
            scenario_desc=scenario_desc,
            max_steps=args.max_steps,
            success_x_threshold=args.success_x_threshold,
            ice_stuck_steps=ice_stuck_steps,
            stationary_stuck_steps=stationary_stuck_steps,
            stationary_speed_threshold=args.stationary_speed_threshold,
            stationary_x_buffer=args.stationary_x_buffer,
            initial_x_default=args.initial_x_default,
            slow_start_speed_threshold=args.slow_start_speed_threshold,
            slow_start_x=args.slow_start_x,
            speed_min=args.speed_min,
            speed_max=args.speed_max,
            speed_step=args.speed_step,
            analytic=analytic,
            variant_results=variant_results,
        )
        markdown = output_md.read_text(encoding="utf-8")
    else:
        markdown = format_markdown_report(
            cfg=cfg,
            num_trials=num_trials,
            seed=args.seed,
            scenario_desc=scenario_desc,
            max_steps=args.max_steps,
            success_x_threshold=args.success_x_threshold,
            ice_stuck_steps=ice_stuck_steps,
            stationary_stuck_steps=stationary_stuck_steps,
            stationary_speed_threshold=args.stationary_speed_threshold,
            stationary_x_buffer=args.stationary_x_buffer,
            initial_x_default=args.initial_x_default,
            slow_start_speed_threshold=args.slow_start_speed_threshold,
            slow_start_x=args.slow_start_x,
            speed_min=args.speed_min,
            speed_max=args.speed_max,
            speed_step=args.speed_step,
            analytic=analytic,
            variant_results=variant_results,
        )
        output_md.write_text(markdown, encoding="utf-8")

    payload = {
        "config": vars(args),
        "sim_config": asdict(cfg),
        "analytic": asdict(analytic),
        "results": [asdict(r) for r in variant_results],
    }
    output_json = Path(args.output_json)
    if not output_json.is_absolute():
        output_json = Path(PROJECT_ROOT) / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + markdown)
    print(f"\nSaved markdown report to: {output_md}")
    print(f"Saved JSON report to: {output_json}")


if __name__ == "__main__":
    main()
