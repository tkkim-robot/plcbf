"""Baseline-only drift-car black-ice benchmark.

This entry point intentionally exposes only MB-CBF-MI and Lib-PCBF-MI.  The
historical benchmark remains in ``benchmark_black_ice.py`` byte-for-byte.  The
simulator applies the exact valid action returned by the selected baseline.
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
from dataclasses import asdict, dataclass
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
HISTORICAL_FAILURE_DEFINITION = "collision OR unrecoverable_infeasibility"
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
    library_size: int
    collision: bool
    unrecoverable_infeasible: bool
    historical_failure: bool
    total_steps: int
    timed_steps: int
    solve_time_sum_sec: float


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


def controller_reported_runtime_error(shielding) -> bool:
    """Recognize an unrecoverable controller error without collecting metrics."""

    status_text = str(getattr(shielding, "status", "")).lower()
    return bool(getattr(shielding, "runtime_error", False)) or status_text.startswith(
        ("error", "value_error", "candidate_evaluation_error")
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
    collision = False
    runtime_error = False
    compute_times_s: List[float] = []
    solve_calls = 0
    timing_warmup_steps = 5
    quiet = not verbose

    for _step in range(n_steps):
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
            break

        try:
            u_nom_vec = np.asarray(u_nom, dtype=float).reshape(-1)
            if u_nom_vec.shape != (2,) or not np.all(np.isfinite(u_nom_vec)):
                raise ValueError("nominal controller returned an invalid control")
            u_safe_vec = validate_returned_control(u_safe, shielding)
        except (TypeError, ValueError):
            runtime_error = True
            break

        # A controller may explicitly classify an internal evaluation failure
        # after returning. Normal native fallback actions remain valid and are
        # passed to the simulator unchanged.
        if controller_reported_runtime_error(shielding):
            runtime_error = True
            break

        try:
            sim_res = simulator.step(u_safe_vec.reshape(-1, 1))
        except Exception:
            runtime_error = True
            break
        total_steps += 1
        if solve_calls > timing_warmup_steps:
            compute_times_s.append(solve_elapsed)
        collision = bool(sim_res["collision"])
        if collision:
            break

        # Track finished
        if car.get_position()[0] > env.track_length - 10.0:
            break

    historical_failure = bool(collision or runtime_error)
    if verbose:
        print(
            f"run={scenario.run_idx:02d} variant={variant.key} "
            f"obs={scenario.num_obstacles} collision={collision} "
            f"unrecoverable={runtime_error} steps={total_steps}"
        )
    return EpisodeResult(
        algorithm=variant.key,
        seed=scenario.seed,
        run_idx=scenario.run_idx,
        obstacle_geometry=list(scenario.obstacles),
        library_size=4,
        collision=collision,
        unrecoverable_infeasible=runtime_error,
        historical_failure=historical_failure,
        total_steps=total_steps,
        timed_steps=len(compute_times_s),
        solve_time_sum_sec=float(np.sum(compute_times_s)),
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
                            f"collision={int(ep.collision)} "
                            f"unrecoverable={int(ep.unrecoverable_infeasible)}"
                        )
        else:
            for payload in payloads:
                all_results[variant.key].append(_run_episode_payload(payload))

    summary_rows = []
    for variant in variants:
        rows = all_results[variant.key]
        if any(
            result.historical_failure
            != bool(result.collision or result.unrecoverable_infeasible)
            for result in rows
        ):
            raise AssertionError(
                "historical_failure must equal collision OR unrecoverable_infeasible"
            )
        n = len(rows)
        fail_count = sum(1 for r in rows if r.historical_failure)
        collision_count = sum(1 for r in rows if r.collision)
        unrecoverable_count = sum(
            1 for r in rows if r.unrecoverable_infeasible
        )
        timed_rows = [r for r in rows if r.timed_steps > 0]
        total_timed_steps = sum(r.timed_steps for r in timed_rows)
        mean_compute_ms = (
            1000.0
            * float(sum(r.solve_time_sum_sec for r in timed_rows))
            / total_timed_steps
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
                "unrecoverable_count": unrecoverable_count,
                "unrecoverable_rate": (
                    100.0 * unrecoverable_count / max(n, 1)
                ),
                "total_timed_steps": total_timed_steps,
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
    lines.append("- Policy library size: 4")
    lines.append(f"- Main failure: `{HISTORICAL_FAILURE_DEFINITION}`")
    lines.append(
        "- The simulator applies the selected baseline's exact finite, "
        "dimension-valid, actuator-valid returned control."
    )
    lines.append(
        "- Timing is end-to-end solve_control_problem wall time and excludes "
        "the first five calls of each trial."
    )
    lines.append(
        "- † MB-CBF-MI uses a sampled terminal proxy, not a proven "
        "control-invariant terminal set, and therefore does not inherit the formal "
        "guarantee of Chen et al."
    )
    lines.append("")
    lines.append("| Algorithm | Failure | Mean Time [ms] |")
    lines.append("|---|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['label']} | "
            f"{_fmt_count_rate(row['fail_count'], row['n'], row['fail_rate'])} | "
            f"{row['mean_compute_ms']:.2f} |"
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
            "failure_contract": {
                "failure": HISTORICAL_FAILURE_DEFINITION,
                "unrecoverable_infeasibility": (
                    UNRECOVERABLE_INFEASIBLE_DEFINITION
                ),
            },
            "control_contract": (
                "apply the exact finite, dimension-valid, actuator-valid control "
                "returned by the selected baseline"
            ),
            "timing": {
                "scope": "solve_control_problem wall-clock time",
                "candidate_execution": "sequential within each control step",
                "warmup_calls_excluded_per_trial": 5,
                "worker_count": max(1, int(args.num_workers)),
            },
            "policy_library_size": 4,
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
