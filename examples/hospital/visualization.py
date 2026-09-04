"""Matplotlib visualization for hospital traces and current simulation state."""

from __future__ import annotations

from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, transforms
from PIL import Image

from .obstacles import Human, Stretcher

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .simulation import HospitalSimulation


_ROLLOUT_COLORS = {
    "nominal": "#0284c7",
    "angle": "#64748b",
    "reverse": "#d97706",
    "stop": "#dc2626",
    "room": "#0f766e",
}
_SELECTED_ROLLOUT_COLOR = "#7c3aed"
_BLOCKING_STRETCHER_COLOR = "#be123c"
_BLOCKING_STRETCHER_EDGE = "#4c0519"


def _benchmark_metadata(
    simulation: "HospitalSimulation",
) -> Mapping[str, object]:
    metadata = getattr(simulation, "benchmark_scenario_metrics", {})
    return metadata if isinstance(metadata, Mapping) else {}


def _provenance_label(simulation: "HospitalSimulation") -> str | None:
    """Return compact, observation-only canonical-world provenance."""

    metadata = _benchmark_metadata(simulation)
    story = metadata.get("story_id")
    seed = metadata.get("traffic_seed")
    world_hash = metadata.get("world_sha256")
    if story is None and seed is None and world_hash is None:
        return None
    parts = []
    if story is not None:
        parts.append(f"story: {story}")
    if seed is not None:
        parts.append(f"traffic seed: {seed}")
    if world_hash is not None:
        parts.append(f"world: {str(world_hash)[:12]}")
    return "   ".join(parts)


def _fixed_blockage_times(
    simulation: "HospitalSimulation",
) -> tuple[float, float] | None:
    """Read fixed geometry timings without exposing them to control."""

    metadata = _benchmark_metadata(simulation)
    start = metadata.get("blockage_started_at_s")
    clear = metadata.get("blockage_cleared_at_s")
    if isinstance(start, bool) or isinstance(clear, bool):
        return None
    try:
        start_time = float(start)
        clear_time = float(clear)
    except (TypeError, ValueError):
        return None
    if (
        not isfinite(start_time)
        or not isfinite(clear_time)
        or start_time < 0.0
        or clear_time < start_time
    ):
        return None
    return start_time, clear_time


def _selected_policy_id(simulation: "HospitalSimulation") -> str:
    """Return the active policy-certificate identifier on the latest step."""

    result = simulation.last_controller
    if result is None:
        return "nominal"
    decision = getattr(result, "decision", None)
    decision_policy = getattr(decision, "policy_id", None)
    if decision_policy:
        return str(decision_policy)
    # Keep rendering compatible with emergency/fallback annotations such as
    # ``"stop:emergency_stop"`` while matching the underlying rollout name.
    return str(getattr(result, "selected_policy", "nominal")).split(":", 1)[0]


def _control_source_label(simulation: "HospitalSimulation") -> str:
    """Describe whether the QP or an emergency path supplied the input."""

    result = simulation.last_controller
    if result is None:
        return "nominal"
    decision = getattr(result, "decision", None)
    diagnostics = getattr(decision, "diagnostics", None)
    if bool(getattr(diagnostics, "used_fallback", False)):
        return "selected-policy backup fallback"
    return "nominal-centered safety QP"


def _draw_policy_rollouts(
    simulation: "HospitalSimulation",
    axes: "Axes",
) -> tuple[int, bool]:
    """Draw every latest policy rollout and emphasize the selected one."""

    result = simulation.last_controller
    evaluations = () if result is None else result.policy_evaluations
    selected_policy = _selected_policy_id(simulation)
    selected_evaluation = None
    candidate_evaluations = []
    for evaluation in evaluations:
        if evaluation.policy.name == selected_policy:
            selected_evaluation = evaluation
        else:
            candidate_evaluations.append(evaluation)

    def draw(evaluation: object, *, selected: bool) -> None:
        trajectory = np.asarray(evaluation.trajectory, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[0] == 0:
            return
        policy = evaluation.policy
        if trajectory.shape[1] < 2:
            return
        line = axes.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            "-" if selected else "--",
            color=(
                _SELECTED_ROLLOUT_COLOR
                if selected
                else _ROLLOUT_COLORS.get(policy.kind, "#64748b")
            ),
            linewidth=3.2 if selected else 0.9,
            alpha=1.0 if selected else 0.32,
            zorder=7 if selected else 3,
            label=(
                f"active certificate rollout: {policy.name}"
                if selected
                else "_candidate_rollout"
            ),
        )[0]
        line.set_gid(f"hospital-policy-rollout:{policy.name}")

    for evaluation in candidate_evaluations:
        draw(evaluation, selected=False)
    if selected_evaluation is not None:
        # Render last so the QP-active certificate remains visible even when
        # several backup trajectories overlap.
        draw(selected_evaluation, selected=True)
    return len(evaluations), selected_evaluation is not None


def draw_simulation(
    simulation: "HospitalSimulation",
    axes: "Axes | None" = None,
    show_trace: bool = True,
) -> tuple["Figure", "Axes"]:
    if axes is None:
        figure, axes = plt.subplots(figsize=(14, 8))
    else:
        figure = axes.figure
    environment = simulation.environment

    for floor in environment.floor_rects:
        color = "#e5e7eb" if floor.kind == "corridor" else "#d5e5df"
        axes.add_patch(
            patches.Rectangle(
                (floor.x, floor.y),
                floor.width,
                floor.height,
                facecolor=color,
                edgecolor="none",
                zorder=0,
            )
        )
    for wall in environment.wall_rects:
        axes.add_patch(
            patches.Rectangle(
                (wall.x, wall.y),
                wall.width,
                wall.height,
                facecolor="#334155",
                edgecolor="#0f172a",
                linewidth=0.3,
                zorder=2,
            )
        )
    for room in environment.rooms:
        axes.text(
            *room.center,
            room.label,
            ha="center",
            va="center",
            fontsize=6,
            color="#475569",
            zorder=1,
        )

    for obstacle in simulation.obstacles:
        if isinstance(obstacle, Human):
            axes.add_patch(
                patches.Circle(
                    obstacle.center,
                    obstacle.radius,
                    facecolor="#ef4444",
                    edgecolor="#7f1d1d",
                    zorder=5,
                )
            )
        elif isinstance(obstacle, Stretcher):
            blocking = obstacle.identifier.startswith("blocking-stretcher-")
            rectangle = patches.Rectangle(
                (
                    obstacle.center[0] - obstacle.length / 2,
                    obstacle.center[1] - obstacle.width / 2,
                ),
                obstacle.length,
                obstacle.width,
                facecolor=(
                    _BLOCKING_STRETCHER_COLOR if blocking else "#f97316"
                ),
                edgecolor=(
                    _BLOCKING_STRETCHER_EDGE if blocking else "#9a3412"
                ),
                linewidth=2.0 if blocking else 1.0,
                hatch="////" if blocking else None,
                zorder=5,
            )
            rectangle.set_gid(
                (
                    "hospital-blocking-stretcher:"
                    if blocking
                    else "hospital-stretcher:"
                )
                + obstacle.identifier
            )
            rectangle.set_transform(
                transforms.Affine2D()
                .rotate_around(
                    obstacle.center[0],
                    obstacle.center[1],
                    obstacle.theta,
                )
                + axes.transData
            )
            axes.add_patch(rectangle)
            if blocking:
                axes.text(
                    *obstacle.center,
                    "BLOCKER",
                    ha="center",
                    va="center",
                    fontsize=5,
                    fontweight="bold",
                    color="white",
                    rotation=np.degrees(obstacle.theta),
                    clip_on=True,
                    zorder=6,
                )
        velocity = obstacle.velocity
        axes.arrow(
            obstacle.center[0],
            obstacle.center[1],
            velocity[0],
            velocity[1],
            width=0.04,
            color="#9a3412",
            length_includes_head=True,
            clip_on=True,
            zorder=6,
        )

    if show_trace and simulation.trace:
        trace = np.asarray([record.state[:2] for record in simulation.trace])
        axes.plot(trace[:, 0], trace[:, 1], color="#0284c7", linewidth=1.8, zorder=4)
    rollout_count, selected_rollout_visible = _draw_policy_rollouts(
        simulation,
        axes,
    )
    axes.add_patch(
        patches.Circle(
            simulation.state[:2],
            simulation.config.robot.radius,
            facecolor="#0ea5e9",
            edgecolor="#075985",
            zorder=8,
        )
    )
    axes.add_patch(
        patches.Circle(
            simulation.goal,
            1.35,
            facecolor="none",
            edgecolor="#16a34a",
            linewidth=2.0,
            zorder=7,
        )
    )
    inside_refuge = bool(
        environment.room_containing(simulation.state[:2]) is not None
    )
    axes.set(
        xlim=(0, environment.width),
        ylim=(0, environment.height),
        aspect="equal",
        xlabel="x [m]",
        ylabel="y [m]",
        title=(
            "Hospital PL-CBF — "
            f"inside room: {'yes' if inside_refuge else 'no'} "
            f"(t={simulation.time:.1f}s)"
        ),
    )
    selected_policy = _selected_policy_id(simulation)
    control_source = _control_source_label(simulation)
    axes.text(
        0.01,
        0.015,
        (
            f"active certificate: {selected_policy}   "
            f"control source: {control_source}   "
            f"candidate rollouts: {rollout_count}   "
            f"selected shown: {'yes' if selected_rollout_visible else 'no'}"
        ),
        transform=axes.transAxes,
        fontsize=8,
        color="#0f172a",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "alpha": 0.9,
        },
        zorder=20,
    )
    provenance = _provenance_label(simulation)
    if provenance is not None:
        provenance_artist = axes.text(
            0.99,
            0.015,
            provenance,
            transform=axes.transAxes,
            ha="right",
            fontsize=8,
            color="#0f172a",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#cbd5e1",
                "alpha": 0.9,
            },
            zorder=20,
        )
        provenance_artist.set_gid("hospital-world-provenance")
    return figure, axes


def _render_simulation_image(
    simulation: "HospitalSimulation",
    *,
    dpi: int,
) -> Image.Image:
    """Render one independent RGB frame using the standard scene drawer."""

    figure, _ = draw_simulation(simulation)
    buffer = BytesIO()
    try:
        # Keep the raster canvas identical across frames.  A tight bounding
        # box expands around a nonreflecting convoy label after the stretcher
        # leaves the modeled floor plan, which makes the GIF appear to pan
        # even though the data limits are fixed.
        figure.savefig(
            buffer,
            format="png",
            dpi=dpi,
        )
    finally:
        plt.close(figure)
    buffer.seek(0)
    with Image.open(buffer) as source:
        return source.convert("RGB")


def export_simulation_visuals(
    simulation: "HospitalSimulation",
    steps: int,
    *,
    gif_path: str | Path | None = None,
    snapshot_dir: str | Path | None = None,
    frame_stride: int = 10,
    fps: float = 12.0,
    dpi: int = 80,
) -> dict[str, object]:
    """Run a simulation while exporting GIF and observation-only snapshots.

    Events are inferred from the QP-selected policy, physical room occupancy,
    and immutable scenario provenance.  The initial room is treated as the
    start rather than a refuge: snapshots record leaving it, the first later
    refuge entry/exit, first room-policy selection, and goal arrival.  When
    fixed blockade start/clear times are available, those observation-only
    instants are captured too.  Nothing here feeds back into control.  GIF
    frames are sampled every ``frame_stride`` plant steps, with event and
    terminal frames included even when they fall between regular samples.
    """

    if gif_path is None and snapshot_dir is None:
        raise ValueError("gif_path or snapshot_dir must be provided")
    if int(steps) < 0:
        raise ValueError("steps must be nonnegative")
    if int(frame_stride) < 1:
        raise ValueError("frame_stride must be positive")
    if not isfinite(float(fps)) or float(fps) <= 0.0:
        raise ValueError("fps must be finite and positive")
    if int(dpi) < 1:
        raise ValueError("dpi must be positive")

    gif_destination = None if gif_path is None else Path(gif_path)
    snapshots_destination = (
        None if snapshot_dir is None else Path(snapshot_dir)
    )
    if gif_destination is not None:
        gif_destination.parent.mkdir(parents=True, exist_ok=True)
    if snapshots_destination is not None:
        snapshots_destination.mkdir(parents=True, exist_ok=True)

    gif_frames: list[Image.Image] = []
    snapshot_paths: dict[str, str] = {}
    captured_events: set[str] = set()

    def capture(events: tuple[str, ...], *, gif_frame: bool) -> None:
        image = _render_simulation_image(simulation, dpi=int(dpi))
        if gif_frame:
            gif_frames.append(image)
        for event in events:
            if event in captured_events:
                continue
            if snapshots_destination is not None:
                destination = snapshots_destination / f"{event}.png"
                image.save(destination, format="PNG")
                snapshot_paths[event] = str(destination)
            captured_events.add(event)
        if not gif_frame:
            image.close()

    initial_room = simulation.environment.room_containing(
        simulation.state[:2]
    )
    initial_room_label = (
        None if initial_room is None else str(initial_room.label)
    )
    previous_room_label = initial_room_label
    start_room_departed = initial_room_label is None
    refuge_room_label: str | None = None
    blockage_times = _fixed_blockage_times(simulation)
    initial_events = ["initial"]
    if (
        blockage_times is not None
        and simulation.time >= blockage_times[0]
        and simulation.time < blockage_times[1]
    ):
        initial_events.append("blockage_active")
    capture(tuple(initial_events), gif_frame=gif_destination is not None)

    executed_steps = 0
    for step_index in range(int(steps)):
        record = simulation.step()
        executed_steps = step_index + 1
        events: list[str] = []
        current_room = simulation.environment.room_containing(
            simulation.state[:2]
        )
        current_room_label = (
            None if current_room is None else str(current_room.label)
        )
        if (
            record.selected_policy.startswith("room")
            and "room_selected" not in captured_events
        ):
            events.append("room_selected")
        if (
            not start_room_departed
            and current_room_label != initial_room_label
            and "start_room_left" not in captured_events
        ):
            start_room_departed = True
            events.append("start_room_left")
        if (
            start_room_departed
            and refuge_room_label is None
            and current_room_label is not None
            and "refuge_entered" not in captured_events
        ):
            refuge_room_label = current_room_label
            events.append("refuge_entered")
        if (
            refuge_room_label is not None
            and previous_room_label == refuge_room_label
            and current_room_label != refuge_room_label
            and "refuge_left" not in captured_events
        ):
            events.append("refuge_left")
        if simulation.reached_goal and "goal" not in captured_events:
            events.append("goal")
        if blockage_times is not None:
            blockage_start, blockage_clear = blockage_times
            if (
                simulation.time >= blockage_start
                and simulation.time < blockage_clear
                and "blockage_active" not in captured_events
            ):
                events.append("blockage_active")
            if (
                simulation.time >= blockage_clear
                and "blockage_cleared" not in captured_events
            ):
                events.append("blockage_cleared")
        previous_room_label = current_room_label
        terminal = bool(simulation.collision or simulation.reached_goal)
        regular_frame = (step_index + 1) % int(frame_stride) == 0
        include_gif = bool(
            gif_destination is not None
            and (regular_frame or events or terminal)
        )
        if events or include_gif:
            capture(tuple(events), gif_frame=include_gif)
        if terminal:
            break

    if gif_destination is not None:
        if not gif_frames:
            raise RuntimeError("visual export produced no GIF frames")
        duration_ms = max(1, int(round(1000.0 / float(fps))))
        gif_frames[0].save(
            gif_destination,
            format="GIF",
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
        )
        for frame in gif_frames:
            frame.close()

    return {
        "gif": (
            None if gif_destination is None else str(gif_destination)
        ),
        "gif_frame_count": len(gif_frames),
        "snapshots": dict(sorted(snapshot_paths.items())),
        "events_captured": sorted(captured_events),
        "executed_steps": executed_steps,
        "collision": bool(simulation.collision),
        "reached_goal": bool(simulation.reached_goal),
    }


__all__ = ["draw_simulation", "export_simulation_visuals"]
