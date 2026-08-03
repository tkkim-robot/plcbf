"""Command-line entrypoint for canonical hospital refuge stories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import (
    build_benchmark_scenario,
    default_hospital_benchmark_steps,
    publication_benchmark_config,
)
from .scenarios import (
    DEFAULT_HOSPITAL_TRAFFIC_SEEDS,
    HOSPITAL_STORY_IDS,
    build_hospital_story_scenario,
)
from .visualization import draw_simulation, export_simulation_visuals


def _traffic_seed(value: str) -> int:
    seed = int(value)
    if seed not in DEFAULT_HOSPITAL_TRAFFIC_SEEDS:
        raise argparse.ArgumentTypeError("seed must be an integer from 0 to 19")
    return seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    scenario = parser.add_mutually_exclusive_group()
    scenario.add_argument(
        "--story",
        choices=HOSPITAL_STORY_IDS,
        help="canonical fixed story (default: main_eastbound)",
    )
    scenario.add_argument(
        "--stretchers",
        type=int,
        choices=(2, 3),
        help="run the legacy blocked-main-hall case explicitly",
    )
    parser.add_argument(
        "--seed",
        type=_traffic_seed,
        default=0,
        help="canonical circular-traffic seed from 0 through 19",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=default_hospital_benchmark_steps(),
    )
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
    story_id = arguments.story
    if arguments.stretchers is None:
        story_id = story_id or "main_eastbound"
        scenario = build_hospital_story_scenario(
            story_id,
            traffic_seed=arguments.seed,
        )
        simulation = scenario.to_simulation(publication_benchmark_config())
        run_mode = "canonical_story"
    else:
        simulation = build_benchmark_scenario(
            f"blocked_{arguments.stretchers}_stretchers",
            seed=arguments.seed,
        )
        run_mode = "legacy_stretcher_case"
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
    provenance = getattr(simulation, "benchmark_scenario_metrics", {})
    print(
        json.dumps(
            {
                "run_mode": run_mode,
                "story": provenance.get("story_id", story_id),
                "seed": arguments.seed,
                "world_sha256": provenance.get("world_sha256"),
                "protocol_sha256": provenance.get(
                    "hospital_story_protocol_sha256"
                ),
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
