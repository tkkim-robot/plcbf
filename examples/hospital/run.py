"""Command-line entrypoint for the deterministic hospital refuge scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import build_benchmark_scenario
from .visualization import draw_simulation, export_simulation_visuals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stretchers", type=int, choices=(2, 3), default=3)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the crowded humans/stretchers and strict convoy",
    )
    parser.add_argument("--steps", type=int, default=1100)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    simulation = build_benchmark_scenario(
        f"blocked_{arguments.stretchers}_stretchers",
        seed=arguments.seed,
    )
    visual_artifacts = None
    if arguments.gif is not None or arguments.snapshot_dir is not None:
        visual_artifacts = export_simulation_visuals(
            simulation,
            arguments.steps,
            gif_path=arguments.gif,
            snapshot_dir=arguments.snapshot_dir,
            frame_stride=arguments.frame_stride,
            fps=arguments.fps,
            dpi=arguments.dpi,
        )
    else:
        simulation.run(arguments.steps)
    last = simulation.last_controller
    print(
        json.dumps(
            {
                "seed": arguments.seed,
                "time": round(simulation.time, 3),
                "collision": bool(simulation.collision),
                "reached_goal": bool(simulation.reached_goal),
                "position": simulation.state[:2].round(3).tolist(),
                "inside_refuge": bool(
                    simulation.environment.room_containing(
                        simulation.state[:2]
                    )
                    is not None
                ),
                "selected_policy": (
                    "nominal" if last is None else last.selected_policy
                ),
                "candidate_rollout_count": (
                    0 if last is None else len(last.policy_evaluations)
                ),
                "dynamic_obstacle_count": len(simulation.obstacles),
                "visuals": visual_artifacts,
            },
            indent=2,
        )
    )
    if arguments.save is not None or arguments.show:
        figure, _ = draw_simulation(simulation)
        if arguments.save is not None:
            arguments.save.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(arguments.save, dpi=160, bbox_inches="tight")
        if arguments.show:
            import matplotlib.pyplot as plt

            plt.show()
    return 1 if simulation.collision else 0


if __name__ == "__main__":
    raise SystemExit(main())
