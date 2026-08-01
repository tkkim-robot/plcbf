from __future__ import annotations

from PIL import Image
import numpy as np
import pytest

from examples.hospital.benchmark import build_benchmark_scenario
from examples.hospital.simulation import TraceRecord
from examples.hospital.visualization import (
    draw_simulation,
    export_simulation_visuals,
)


def test_draw_simulation_shows_all_policy_rollouts_and_highlights_selected() -> None:
    simulation = build_benchmark_scenario(
        "blocked_3_stretchers",
        seed=0,
    )
    simulation.step()
    result = simulation.last_controller
    assert result is not None
    assert result.decision.policy_id is not None

    figure, axes = draw_simulation(simulation, show_trace=False)
    try:
        rollout_lines = {
            line.get_gid().split(":", 1)[1]: line
            for line in axes.lines
            if (line.get_gid() or "").startswith(
                "hospital-policy-rollout:"
            )
        }
        expected_names = {
            evaluation.policy.name
            for evaluation in result.policy_evaluations
        }
        assert set(rollout_lines) == expected_names

        selected_name = result.decision.policy_id
        selected_line = rollout_lines[selected_name]
        assert selected_line.get_linestyle() == "-"
        assert selected_line.get_linewidth() > 3.0
        assert selected_line.get_alpha() == 1.0

        for name, line in rollout_lines.items():
            if name == selected_name:
                continue
            assert line.get_linestyle() == "--"
            assert line.get_linewidth() < 1.0
            assert line.get_alpha() < 0.5
    finally:
        figure.clf()


def test_actual_simulation_visual_export_writes_gif_and_event_snapshots(
    tmp_path,
    monkeypatch,
) -> None:
    simulation = build_benchmark_scenario(
        "blocked_3_stretchers",
        seed=0,
    )
    positions = (
        np.array([58.0, 50.2]),
        np.array([59.0, 57.0]),
        np.array([59.0, 50.2]),
        simulation.goal.copy(),
    )

    def scripted_step() -> TraceRecord:
        index = len(simulation.trace)
        simulation.state = np.r_[positions[index], np.zeros(2)]
        simulation.time += simulation.config.dt
        simulation.reached_goal = index == len(positions) - 1
        inside_refuge = bool(
            simulation.environment.room_containing(
                simulation.state[:2]
            )
            is not None
        )
        record = TraceRecord(
            time=simulation.time,
            state=simulation.state.copy(),
            selected_policy=(
                "room_0" if index == 0 else "nominal"
            ),
            inside_refuge=inside_refuge,
            min_clearance=1.0,
            collision=False,
            reached_goal=simulation.reached_goal,
        )
        simulation.trace.append(record)
        return record

    monkeypatch.setattr(simulation, "step", scripted_step)
    gif_path = tmp_path / "hospital.gif"
    snapshot_dir = tmp_path / "snapshots"
    artifacts = export_simulation_visuals(
        simulation,
        4,
        gif_path=gif_path,
        snapshot_dir=snapshot_dir,
        frame_stride=99,
        fps=10.0,
        dpi=30,
    )

    assert artifacts["reached_goal"] is True
    assert artifacts["collision"] is False
    assert artifacts["executed_steps"] == 4
    assert artifacts["events_captured"] == [
        "goal",
        "initial",
        "inside_room",
        "room_left",
        "room_selected",
    ]
    assert artifacts["gif_frame_count"] == 5
    assert set(artifacts["snapshots"]) == {
        "initial",
        "room_selected",
        "inside_room",
        "room_left",
        "goal",
    }
    assert gif_path.is_file()
    with Image.open(gif_path) as animation:
        assert animation.format == "GIF"
        assert animation.n_frames == 5
        assert animation.size[0] > 100
        assert animation.size[1] > 50
    for event in (
        "initial",
        "room_selected",
        "inside_room",
        "room_left",
        "goal",
    ):
        path = snapshot_dir / f"{event}.png"
        assert path.is_file()
        with Image.open(path) as snapshot:
            assert snapshot.format == "PNG"
            assert snapshot.size[0] > 100
            assert snapshot.size[1] > 50


def test_visual_export_validates_sampling_arguments(tmp_path) -> None:
    simulation = build_benchmark_scenario("blocked_2_stretchers")
    with pytest.raises(ValueError, match="gif_path or snapshot_dir"):
        export_simulation_visuals(simulation, 1)
    with pytest.raises(ValueError, match="frame_stride"):
        export_simulation_visuals(
            simulation,
            1,
            gif_path=tmp_path / "bad.gif",
            frame_stride=0,
        )
    with pytest.raises(ValueError, match="fps"):
        export_simulation_visuals(
            simulation,
            1,
            gif_path=tmp_path / "bad.gif",
            fps=0.0,
        )
