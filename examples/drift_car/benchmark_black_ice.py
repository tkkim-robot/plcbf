"""
Benchmark for drift-car black-ice puddle-surprise scenario.

Compares 13 variants:
1) PLCBF (multi-policy)
2) BackupCBF with 3 fixed backup policies (stop/left/right)
3) MPS with 3 fixed backup policies (stop/left/right)
4) Gatekeeper with 3 fixed backup policies (stop/left/right)
5) PCBF with 3 fixed backup policies (stop/left/right)

Metrics:
- Collision/Infeasible rate
- Average nominal tracking percentage
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root and submodule path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "safe_control"))

from safe_control.envs.drifting_env import DriftingEnv
from safe_control.position_control.backup_cbf_qp import BackupCBF
from safe_control.position_control.backup_controller import LaneChangeController, StoppingController
from safe_control.position_control.mpcc import MPCC
from safe_control.robots.drifting_car import DriftingCar, DriftingCarSimulator
from safe_control.shielding.gatekeeper import Gatekeeper
from safe_control.shielding.mps import MPS

from examples.drift_car.algorithms.plcbf_drift import PLCBF
from examples.drift_car.algorithms.pcbf_drift import PCBF

QUIET_SINK = open(os.devnull, "w", encoding="utf-8")


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
    algo: str  # plcbf, backup_cbf, mps, gatekeeper, pcbf
    backup_policy: Optional[str]  # None for plcbf, else stop/lane_change_left/lane_change_right


@dataclass(frozen=True)
class Scenario:
    run_idx: int
    seed: int
    num_obstacles: int
    # tuples: (x, lane_name) lane_name in {"middle","left","right"}
    obstacles: Tuple[Tuple[float, str], ...]


@dataclass
class EpisodeResult:
    collision: bool
    infeasible: bool
    nominal_tracking_pct: float
    total_steps: int


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
    variants: List[AlgoVariant] = [
        AlgoVariant("plcbf", "PLCBF", "plcbf", None),
    ]

    families = [
        ("backup_cbf", "Backup CBF"),
        ("mps", "MPS"),
        ("gatekeeper", "Gatekeeper"),
        ("pcbf", "PCBF"),
    ]
    backups = [
        ("stop", "Stop"),
        ("lane_change_left", "Lane Left"),
        ("lane_change_right", "Lane Right"),
    ]
    for fam_key, fam_label in families:
        for backup_key, backup_label in backups:
            key = f"{fam_key}_{backup_key}"
            label = f"{fam_label} ({backup_label})"
            variants.append(AlgoVariant(key, label, fam_key, backup_key))
    return variants


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

        scenarios.append(
            Scenario(
                run_idx=run_idx,
                seed=int(rng.integers(0, 2**31 - 1)),
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


def create_backup_controller(
    backup_policy: str,
    vehicle_spec: Dict[str, float],
    lanes: Dict[str, float],
    cfg: SimConfig,
):
    if backup_policy == "stop":
        return StoppingController(vehicle_spec, cfg.dt), None
    if backup_policy == "lane_change_left":
        target = max(lanes["left"], 5.0)
        return LaneChangeController(vehicle_spec, cfg.dt, direction="left"), target
    if backup_policy == "lane_change_right":
        target = min(lanes["right"], -5.0)
        return LaneChangeController(vehicle_spec, cfg.dt, direction="right"), target
    raise ValueError(f"Unknown backup policy: {backup_policy}")


def setup_shielding(
    variant: AlgoVariant,
    car: DriftingCar,
    env: DriftingEnv,
    lanes: Dict[str, float],
    cfg: SimConfig,
):
    if variant.algo == "plcbf":
        shielding = PLCBF(
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
        shielding.set_environment(env)
        return shielding

    if variant.backup_policy is None:
        raise ValueError(f"Variant {variant.key} requires backup policy")

    backup_controller, backup_target = create_backup_controller(
        variant.backup_policy, car.robot_spec, lanes, cfg
    )

    if variant.algo == "backup_cbf":
        shielding = BackupCBF(
            robot=car,
            robot_spec=car.robot_spec,
            dt=cfg.dt,
            backup_horizon=cfg.backup_horizon_time,
            ax=None,
        )
        shielding.set_backup_controller(backup_controller, target=backup_target)
        shielding.set_environment(env)
        return shielding

    if variant.algo == "mps":
        shielding = MPS(
            robot=car,
            robot_spec=car.robot_spec,
            dt=cfg.dt,
            backup_horizon=cfg.backup_horizon_time,
            event_offset=cfg.event_offset,
            ax=None,
            safety_margin=cfg.safety_margin,
        )
        shielding.set_backup_controller(backup_controller, target=backup_target)
        shielding.set_environment(env)
        return shielding

    if variant.algo == "gatekeeper":
        shielding = Gatekeeper(
            robot=car,
            robot_spec=car.robot_spec,
            dt=cfg.dt,
            backup_horizon=cfg.backup_horizon_time,
            event_offset=cfg.event_offset,
            ax=None,
            safety_margin=cfg.safety_margin,
        )
        shielding.set_backup_controller(backup_controller, target=backup_target)
        shielding.set_environment(env)
        return shielding

    if variant.algo == "pcbf":
        shielding = PCBF(
            robot=car,
            robot_spec=car.robot_spec,
            dt=cfg.dt,
            backup_horizon=cfg.backup_horizon_time,
            cbf_alpha=5.0,
            safety_margin=1.0,
            use_cbf_slack=False,
            ax=None,
        )
        shielding.set_backup_controller(backup_controller, target=backup_target)
        shielding.set_environment(env)
        return shielding

    raise ValueError(f"Unknown algorithm: {variant.algo}")


def solve_safe_control(
    variant: AlgoVariant,
    shielding,
    state: np.ndarray,
    u_nom: np.ndarray,
    pred_states: Optional[np.ndarray],
    pred_controls: Optional[np.ndarray],
    friction: float,
):
    if variant.algo == "plcbf":
        return shielding.solve_control_problem(
            state,
            control_ref={"u_ref": u_nom},
            friction=friction,
            nominal_trajectory=pred_states.T if pred_states is not None else None,
            nominal_controls=pred_controls.T if pred_controls is not None else None,
        )
    if variant.algo == "pcbf":
        return shielding.solve_control_problem(
            state,
            control_ref={"u_ref": u_nom},
            friction=friction,
        )
    return shielding.solve_control_problem(state, friction=friction)


def call_quiet(quiet: bool, fn, *args, **kwargs):
    if quiet:
        with contextlib.redirect_stdout(QUIET_SINK), contextlib.redirect_stderr(QUIET_SINK):
            return fn(*args, **kwargs)
    return fn(*args, **kwargs)


def run_episode(variant: AlgoVariant, scenario: Scenario, cfg: SimConfig, verbose: bool = False) -> EpisodeResult:
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
            infeasible = True
            break

        # Baseline methods use externally supplied nominal trajectory.
        if variant.algo in ("backup_cbf", "mps", "gatekeeper"):
            if pred_states is not None and pred_controls is not None:
                shielding.set_nominal_trajectory(pred_states, pred_controls)

        try:
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

        # Track finished
        if car.get_position()[0] > env.track_length - 10.0:
            break

    nominal_pct = 100.0 * nominal_like_steps / max(total_steps, 1)
    if verbose:
        print(
            f"run={scenario.run_idx:02d} variant={variant.key} "
            f"obs={scenario.num_obstacles} collision={collision} infeasible={infeasible} "
            f"nominal={nominal_pct:.1f}% steps={total_steps}"
        )
    return EpisodeResult(
        collision=collision,
        infeasible=infeasible,
        nominal_tracking_pct=nominal_pct,
        total_steps=total_steps,
    )


def aggregate_results(
    variants: List[AlgoVariant],
    scenarios: List[Scenario],
    cfg: SimConfig,
    verbose: bool = False,
):
    all_results: Dict[str, List[EpisodeResult]] = {v.key: [] for v in variants}

    for variant in variants:
        if verbose:
            print(f"\n=== Running {variant.label} ===")
        for scenario in scenarios:
            ep = run_episode(variant, scenario, cfg, verbose=verbose)
            all_results[variant.key].append(ep)

    summary_rows = []
    for variant in variants:
        rows = all_results[variant.key]
        n = len(rows)
        fail_count = sum(1 for r in rows if (r.collision or r.infeasible))
        nominal_avg = float(np.mean([r.nominal_tracking_pct for r in rows])) if rows else 0.0
        summary_rows.append(
            {
                "key": variant.key,
                "label": variant.label,
                "n": n,
                "fail_count": fail_count,
                "fail_rate": 100.0 * fail_count / max(n, 1),
                "nominal_avg": nominal_avg,
            }
        )
    return summary_rows, all_results


def format_markdown_table(summary_rows: List[dict], num_runs: int, seed: int, cfg: SimConfig) -> str:
    lines: List[str] = []
    lines.append("# Drift Car Black-Ice Benchmark Results")
    lines.append("")
    lines.append(f"- Runs per algorithm: {num_runs}")
    lines.append(f"- Scenario seed: {seed}")
    lines.append(
        f"- Puddle: x={cfg.puddle_x:.1f}, radius={cfg.puddle_radius:.1f}, friction={cfg.puddle_friction:.2f}"
    )
    lines.append("- Obstacles per run: random 1 or 2, placed near and beyond puddle center (x in ~[72, 85])")
    lines.append("")
    lines.append("| Algorithm | Collision/Infeasible Rate | Avg Nominal Tracking (%) |")
    lines.append("|---|---:|---:|")
    for row in summary_rows:
        fail_text = f"{row['fail_count']}/{row['n']} ({row['fail_rate']:.1f}%)"
        lines.append(f"| {row['label']} | {fail_text} | {row['nominal_avg']:.1f} |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark drift-car black-ice scenario")
    parser.add_argument("--num-runs", type=int, default=10, help="Runs per algorithm")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for scenario generation")
    parser.add_argument(
        "--output-md",
        type=str,
        default="examples/drift_car/benchmark_black_ice_results.md",
        help="Path to output markdown report",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose per-run logging")
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore",
        message="Solution may be inaccurate.*",
        module="cvxpy",
    )

    cfg = SimConfig()
    variants = make_variants()
    scenarios = generate_scenarios(args.num_runs, args.seed)

    summary_rows, _ = aggregate_results(variants, scenarios, cfg, verbose=args.verbose)
    markdown = format_markdown_table(summary_rows, args.num_runs, args.seed, cfg)

    output_path = Path(args.output_md)
    if not output_path.is_absolute():
        output_path = Path(PROJECT_ROOT) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nSaved markdown report to: {output_path}")


if __name__ == "__main__":
    main()
