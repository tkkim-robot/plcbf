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
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "safe_control"))

from safe_control.envs.warehouse_env import WarehouseEnv
import examples.warehouse.test_warehouse_quad as test_quad
from examples.warehouse.controllers.policies_quad3d_jax import RetracePolicyParams


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


@dataclass
class SummaryRow:
    key: str
    label: str
    n_trials: int
    collisions: int
    infeasibles: int
    fail_count: int
    collision_rate_pct: float
    infeasible_rate_pct: float
    fail_rate_pct: float
    avg_nominal_tracking_pct: float
    avg_compute_ms: float
    total_timed_steps: int


ALGO_SPECS: List[AlgoSpec] = [
    AlgoSpec("plcbf", "PLCBF"),
    AlgoSpec("pcbf", "PCBF (Retrace Backup)"),
    AlgoSpec("gatekeeper", "Gatekeeper (Retrace Backup)"),
    AlgoSpec("mps", "MPS (Retrace Backup)"),
    AlgoSpec("backup_cbf", "Backup CBF (Retrace Backup)"),
    AlgoSpec("mip_mpc", "MIP MPC"),
]


def _fmt_count_rate(count: int, total: int) -> str:
    return f"{count}/{total} ({100.0 * count / max(total, 1):.1f}%)"


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
    reached_goal = False

    nominal_track_steps = 0
    total_steps = 0

    solve_time_sum = 0.0
    timed_steps = 0

    warmup_steps = jit_warmup_steps if algo in {"pcbf", "plcbf"} else 0

    for step in range(max_steps):
        env.step()
        ghosts = env.get_dynamic_obstacles()
        statics = env.get_static_obstacles()

        u_nom = None
        u_safe = None

        if algo in {"pcbf", "plcbf", "mip_mpc"}:
            shielding.update_obstacles(ghosts, statics)
            u_nom = np.array(nom_ctrl.get_control(current_state)).flatten()
            control_ref = {"u_ref": u_nom}

            if algo in {"plcbf", "mip_mpc"}:
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

        if solve_err is not None:
            infeasible = True
            break

        if step >= warmup_steps:
            solve_time_sum += float(solve_dt)
            timed_steps += 1

        u_safe = np.array(u_safe).flatten()
        if u_safe.shape[0] != 4 or not np.all(np.isfinite(u_safe)):
            infeasible = True
            break

        current_state = robot.step(current_state.reshape(-1, 1), u_safe.reshape(-1, 1)).flatten()
        env.robot_pos = current_state[:2]

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

        if u_nom is not None:
            if np.linalg.norm(u_safe - u_nom) < tracking_tol:
                nominal_track_steps += 1
            total_steps += 1

    nominal_tracking_pct = 100.0 * nominal_track_steps / max(total_steps, 1)

    return TrialResult(
        collision=collision,
        infeasible=infeasible,
        reached_goal=reached_goal,
        nominal_tracking_pct=nominal_tracking_pct,
        solve_time_sum_sec=solve_time_sum,
        timed_steps=timed_steps,
        total_steps=total_steps,
    )


def summarize_trials(algo_spec: AlgoSpec, trials: List[TrialResult]) -> SummaryRow:
    n = len(trials)
    collisions = sum(int(t.collision) for t in trials)
    infeasibles = sum(int(t.infeasible) for t in trials)
    fail_count = sum(int(t.collision or t.infeasible) for t in trials)

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
        collision_rate_pct=100.0 * collisions / max(n, 1),
        infeasible_rate_pct=100.0 * infeasibles / max(n, 1),
        fail_rate_pct=100.0 * fail_count / max(n, 1),
        avg_nominal_tracking_pct=avg_nominal,
        avg_compute_ms=avg_compute_ms,
        total_timed_steps=total_timed_steps,
    )


def run_algorithm_trials(
    algo_spec: AlgoSpec,
    scenarios: List[TrialScenario],
    args: argparse.Namespace,
    *,
    verbose: bool,
) -> Tuple[List[TrialResult], SummaryRow]:
    trials: List[TrialResult] = []

    t_algo0 = time.perf_counter()
    for idx, scenario in enumerate(scenarios):
        result = run_trial(
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
        )
        trials.append(result)

        if verbose and ((idx + 1) % max(1, args.progress_every) == 0):
            print(
                f"  {algo_spec.key:10s} trial {idx + 1:3d}/{len(scenarios)} | "
                f"collision={int(result.collision)} infeasible={int(result.infeasible)} "
                f"track={result.nominal_tracking_pct:.1f}%"
            )

    elapsed = time.perf_counter() - t_algo0
    summary = summarize_trials(algo_spec, trials)
    print(
        f"[Done] {algo_spec.label:<28} "
        f"collision={_fmt_count_rate(summary.collisions, summary.n_trials)} "
        f"infeasible={_fmt_count_rate(summary.infeasibles, summary.n_trials)} "
        f"nominal={summary.avg_nominal_tracking_pct:.1f}% "
        f"avg_compute={summary.avg_compute_ms:.3f} ms "
        f"elapsed={elapsed/60.0:.1f} min"
    )

    return trials, summary


def refresh_timing_one_by_one(
    algo_specs: List[AlgoSpec],
    scenarios: List[TrialScenario],
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Recompute average solve time per algorithm by running algorithms one-by-one."""
    refreshed: Dict[str, float] = {}

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


def save_randomized_animations(
    *,
    algo_specs: List[AlgoSpec],
    scenarios: List[TrialScenario],
    args: argparse.Namespace,
):
    n_sets = max(0, min(int(args.animation_sets), len(scenarios)))
    if n_sets <= 0:
        print("No animation sets requested; skipping animation export.")
        return

    output_root = Path(args.animation_output_dir)
    if not output_root.is_absolute():
        output_root = Path(PROJECT_ROOT) / output_root
    output_root = output_root / f"seed_{args.seed}"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Saving randomized animations ({n_sets} scenario sets) ===")
    print(f"Animation output root: {output_root}")

    for idx in range(n_sets):
        scenario = scenarios[idx]
        scenario_dir = output_root / f"idx_{idx:02d}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        for algo_spec in algo_specs:
            filename = f"warehouse_lvl{args.level}_{algo_spec.key}_idx{idx:02d}.mp4"
            print(
                f"[Animation] idx={idx:02d}/{n_sets - 1:02d} algo={algo_spec.key:10s} "
                f"-> {scenario_dir / filename}"
            )

            anim_args = argparse.Namespace(
                algo=algo_spec.key,
                level=args.level,
                no_render=False,
                save=True,
                save_dir=str(scenario_dir),
                save_name=filename,
                safety_margin=args.safety_margin,
                alpha=args.alpha,
                plcbf_num_angle_policies=args.plcbf_num_angle_policies,
                mip_num_angle_policies=args.mip_num_angle_policies,
                timing_warmup_steps=args.jit_warmup_steps,
                sensing_range=args.sensing_range,
                max_steps=args.max_steps,
            )
            result = test_quad.run_simulation(anim_args, scenario_ghosts=scenario.ghosts)
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
    lines.append(f"- MIP angle policies: {args.mip_num_angle_policies}")
    lines.append(f"- JIT warmup skip (PCBF/PLCBF): {args.jit_warmup_steps} steps")
    lines.append(
        "- Compute-time column uses solve-control time only "
        "(plotting/logging excluded; refreshed one-by-one after full table "
        f"using {args.timing_refresh_trials} scenario(s))"
        if timing_refreshed
        else "- Compute-time column uses solve-control time only (plotting/logging excluded)"
    )
    lines.append("")
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
    parser.add_argument("--skip-timing-refresh", action="store_true")
    parser.add_argument("--timing-refresh-trials", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sensing-range", type=float, default=test_quad.DEFAULT_SENSING_RANGE_M)
    parser.add_argument("--save-animations", action="store_true")
    parser.add_argument("--animations-only", action="store_true")
    parser.add_argument("--animation-sets", type=int, default=5)
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

    args = parser.parse_args()

    warnings.filterwarnings("ignore", message="Solution may be inaccurate.*", module="cvxpy")

    algo_specs = [s for s in ALGO_SPECS if not (args.skip_mip and s.key == "mip_mpc")]

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
        "summaries": [asdict(s) for s in summaries],
        "timing_refreshed": timing_refreshed,
    }
    output_json.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    print("\n" + markdown)
    print(f"Saved markdown report: {output_md}")
    print(f"Saved json report: {output_json}")


if __name__ == "__main__":
    main()
