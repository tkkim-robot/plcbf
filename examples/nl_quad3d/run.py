"""Command-line entry point for deterministic nonlinear-quadrotor simulations."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .benchmark import seeded_scenario
from .config_io import (
    DEFAULT_CONTROLLER_CONFIG_PATH,
    load_controller_config_artifact,
)
from .controller import PLCBF_NLQuad3D
from .dynamics import NLQuad3D
from .rerun_logger import NLQuad3DRerunLogger
from .scenarios import (
    PLAYGROUND_STRESS_SCENARIO,
    get_scenario,
    scenario_names,
)
from .simulation import simulate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=scenario_names(),
        default=PLAYGROUND_STRESS_SCENARIO,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="replay a benchmark scenario seed (default: 0)",
    )
    parser.add_argument(
        "--obstacle-count",
        type=int,
        default=None,
        help=(
            "override generated playground sphere count (defaults: 48 for "
            "playground_stress, 32 for playground_crowded)"
        ),
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--controller",
        choices=("plcbf", "nominal"),
        default="plcbf",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="spawn the local Rerun viewer",
    )
    parser.add_argument(
        "--save-rrd",
        type=Path,
        help="write a headless Rerun recording (viewer is not spawned)",
    )
    parser.add_argument(
        "--config",
        "--config-json",
        dest="config_path",
        type=Path,
        help="PL-CBF controller YAML/JSON (default: packaged Optuna winner)",
    )
    parser.add_argument("--radial-policies", type=int, default=None)
    parser.add_argument("--backup-horizon", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scenario = seeded_scenario(
        get_scenario(args.scenario),
        args.seed,
        playground_obstacle_count=args.obstacle_count,
    )
    model = NLQuad3D()
    config_path = args.config_path or DEFAULT_CONTROLLER_CONFIG_PATH
    loaded_config, _ = load_controller_config_artifact(config_path)
    config = replace(
        loaded_config,
        dt=model.dt,
        num_radial_policies=(
            loaded_config.num_radial_policies
            if args.radial_policies is None
            else args.radial_policies
        ),
        backup_horizon=(
            loaded_config.backup_horizon
            if args.backup_horizon is None
            else args.backup_horizon
        ),
    )
    controller = PLCBF_NLQuad3D(model, config, bounds=scenario.bounds)
    logger = None
    if args.visualize or args.save_rrd:
        logger = NLQuad3DRerunLogger(
            model,
            spawn_viewer=args.visualize,
            save_path=args.save_rrd,
        )
    result = simulate(
        scenario,
        model=model,
        controller=controller,
        max_steps=args.steps,
        use_plcbf=args.controller == "plcbf",
        logger=logger,
    )
    print(
        json.dumps(
            {
                "scenario": result.scenario,
                "seed": args.seed,
                "obstacle_count": int(scenario.obstacles.shape[0]),
                "controller": args.controller,
                "controller_config_source": str(config_path),
                "steps": result.steps,
                "reached_goal": result.reached_goal,
                "collision": result.collision,
                "minimum_clearance": result.minimum_clearance,
                "final_position": result.final_state[:3].tolist(),
                "rrd": str(args.save_rrd.resolve()) if args.save_rrd else None,
            },
            indent=2,
        )
    )
    return 1 if result.collision else 0


if __name__ == "__main__":
    raise SystemExit(main())
