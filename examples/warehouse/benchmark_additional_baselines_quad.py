"""Baseline-only randomized Warehouse Quad3D benchmark.

Only MB-CBF-MI and Lib-PCBF-MI are selectable.  The driver records the seeded
scenario, historical failure outcome, and solve timing needed for the paper
table.  The simulator always receives the exact valid control returned by the
selected baseline.
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
from dataclasses import asdict, dataclass
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
    algorithm: str
    seed: int
    run_idx: int
    obstacle_geometry: List[Tuple[float, float, float, float, float]]
    library_size: int
    collision: bool
    unrecoverable_infeasible: bool
    historical_failure: bool
    solve_time_sum_sec: float
    timed_steps: int
    total_steps: int


@dataclass
class SummaryRow:
    key: str
    label: str
    n_trials: int
    library_size: int
    collisions: int
    unrecoverable_infeasibles: int
    fail_count: int
    collision_rate_pct: float
    unrecoverable_infeasible_rate_pct: float
    fail_rate_pct: float
    avg_compute_ms: float
    total_timed_steps: int


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


def _sample_velocity(
    rng: np.random.Generator,
    speed_min: float,
    speed_max: float,
) -> Tuple[float, float]:
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
    unrecoverable_infeasible = False
    reached_goal = False
    total_steps = 0
    solve_time_sum = 0.0
    timed_steps = 0

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
            unrecoverable_infeasible = True
            break

        u_safe, solve_dt, solve_err = _solve_with_timing(
            lambda: shielding.solve_control_problem(current_state, control_ref)
        )
        if solve_err is not None:
            unrecoverable_infeasible = True
            break

        try:
            controller_runtime_error = bool(
                getattr(shielding, "runtime_error", False)
            )
        except Exception:
            controller_runtime_error = True
        if controller_runtime_error:
            unrecoverable_infeasible = True
            break

        try:
            u_safe_vec = validate_returned_control(u_safe, robot_spec)
        except (TypeError, ValueError):
            unrecoverable_infeasible = True
            break

        try:
            next_state = robot.step(
                current_state.reshape(-1, 1), u_safe_vec.reshape(-1, 1)
            )
            current_state = np.asarray(next_state, dtype=float).reshape(-1)
            if current_state.shape != (12,) or not np.all(np.isfinite(current_state)):
                raise ValueError("simulator returned an invalid state")
            env.robot_pos = current_state[:2]
        except Exception:
            unrecoverable_infeasible = True
            break

        total_steps += 1
        if step >= jit_warmup_steps:
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
            unrecoverable_infeasible = True
            break

        if collision or reached_goal:
            break

    historical_failure = bool(collision or unrecoverable_infeasible)
    library_size = len(getattr(shielding, "policy_configs", {}))

    return TrialResult(
        algorithm=algo,
        seed=scenario.seed,
        run_idx=scenario.run_idx,
        obstacle_geometry=[tuple(item) for item in scenario.ghosts],
        library_size=library_size,
        collision=collision,
        unrecoverable_infeasible=unrecoverable_infeasible,
        historical_failure=historical_failure,
        solve_time_sum_sec=solve_time_sum,
        timed_steps=timed_steps,
        total_steps=total_steps,
    )


def summarize_trials(algo_spec: AlgoSpec, trials: List[TrialResult]) -> SummaryRow:
    if any(
        trial.historical_failure
        != bool(trial.collision or trial.unrecoverable_infeasible)
        for trial in trials
    ):
        raise AssertionError(
            "historical_failure must equal collision OR unrecoverable_infeasible"
        )
    n = len(trials)
    collisions = sum(int(t.collision) for t in trials)
    unrecoverable_infeasibles = sum(
        int(t.unrecoverable_infeasible) for t in trials
    )
    fail_count = sum(int(t.historical_failure) for t in trials)

    total_solve_sec = float(np.sum([t.solve_time_sum_sec for t in trials]))
    total_timed_steps = int(np.sum([t.timed_steps for t in trials]))
    avg_compute_ms = (
        1000.0 * total_solve_sec / total_timed_steps
        if total_timed_steps
        else float("nan")
    )

    return SummaryRow(
        key=algo_spec.key,
        label=algo_spec.label,
        n_trials=n,
        library_size=max((t.library_size for t in trials), default=0),
        collisions=collisions,
        unrecoverable_infeasibles=unrecoverable_infeasibles,
        fail_count=fail_count,
        collision_rate_pct=100.0 * collisions / max(n, 1),
        unrecoverable_infeasible_rate_pct=(
            100.0 * unrecoverable_infeasibles / max(n, 1)
        ),
        fail_rate_pct=100.0 * fail_count / max(n, 1),
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
                        f"failure={int(result.historical_failure)}"
                    )
    else:
        for index, payload in enumerate(payloads, start=1):
            result = _run_trial_payload(payload)
            trials.append(result)
            if verbose and index % max(1, args.progress_every) == 0:
                print(
                    f"  {algo_spec.key:22s} trial {index:3d}/{len(scenarios)} "
                    f"failure={int(result.historical_failure)}"
                )

    summary = summarize_trials(algo_spec, trials)
    elapsed = time.perf_counter() - started
    print(
        f"[Done] {algo_spec.label:<28} "
        f"failure={_fmt_count_rate(summary.fail_count, summary.n_trials)} "
        f"collision={_fmt_count_rate(summary.collisions, summary.n_trials)} "
        "unrecoverable="
        f"{_fmt_count_rate(summary.unrecoverable_infeasibles, summary.n_trials)} "
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
            "- The simulator applies the selected baseline's returned finite, "
            "bounded control unchanged."
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
        "| Algorithm | P | Library size | Failure | Avg Compute Time (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.label} | {args.num_angle_policies} | "
            f"{summary.library_size or args.num_angle_policies + 2} | "
            f"{_fmt_count_rate(summary.fail_count, summary.n_trials)} | "
            f"{summary.avg_compute_ms:.3f} |"
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
            "applied_control": "exact valid control returned by the selected baseline",
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
