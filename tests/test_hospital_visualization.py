from __future__ import annotations

import json
from types import SimpleNamespace

from PIL import Image
import numpy as np
import pytest

from examples.hospital import run
from examples.hospital.benchmark import build_benchmark_scenario
from examples.hospital.scenarios import build_hospital_story_scenario
from examples.hospital.simulation import TraceRecord
from examples.hospital.visualization import (
    _render_simulation_image,
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

        status = next(
            text.get_text()
            for text in axes.texts
            if "active certificate:" in text.get_text()
        )
        assert f"active certificate: {selected_name}" in status
        expected_source = (
            "selected-policy backup fallback"
            if result.decision.diagnostics.used_fallback
            else "nominal-centered safety QP"
        )
        assert f"control source: {expected_source}" in status
    finally:
        figure.clf()


def test_canonical_scene_shows_world_provenance_and_blockade_stretchers() -> None:
    scenario = build_hospital_story_scenario(
        "main_eastbound",
        traffic_seed=3,
        human_count=0,
    )
    simulation = scenario.to_simulation()

    figure, axes = draw_simulation(simulation, show_trace=False)
    try:
        blocker_patches = [
            patch
            for patch in axes.patches
            if (patch.get_gid() or "").startswith(
                "hospital-blocking-stretcher:"
            )
        ]
        assert len(blocker_patches) == len(scenario.blockers)
        assert all(patch.get_hatch() == "////" for patch in blocker_patches)
        assert all(patch.get_linewidth() == 2.0 for patch in blocker_patches)

        provenance = next(
            text
            for text in axes.texts
            if text.get_gid() == "hospital-world-provenance"
        )
        assert "story: main_eastbound" in provenance.get_text()
        assert "traffic seed: 3" in provenance.get_text()
        assert scenario.world_sha256[:12] in provenance.get_text()
    finally:
        figure.clf()


def test_gif_camera_stays_fixed_when_one_way_convoy_leaves_world() -> None:
    scenario = build_hospital_story_scenario(
        "main_eastbound",
        traffic_seed=0,
        human_count=0,
    )
    simulation = scenario.to_simulation()
    initial = _render_simulation_image(simulation, dpi=30)
    try:
        for obstacle in simulation.obstacles:
            if obstacle.identifier.startswith("blocking-stretcher-"):
                obstacle.coordinate = -150.0
        departed = _render_simulation_image(simulation, dpi=30)
        try:
            assert initial.size == departed.size
            assert initial.size == (420, 240)
        finally:
            departed.close()
    finally:
        initial.close()


def test_actual_simulation_visual_export_writes_gif_and_event_snapshots(
    tmp_path,
    monkeypatch,
) -> None:
    scenario = build_hospital_story_scenario(
        "main_eastbound",
        traffic_seed=0,
        human_count=0,
    )
    simulation = scenario.to_simulation()
    rooms = {
        room.label: room for room in simulation.environment.rooms
    }
    start_room = rooms[scenario.template.start_room]
    refuge_room = rooms[scenario.template.diagnostic_refuge_room]
    start_outside = simulation.environment.room_door_path(
        start_room,
        simulation.config.robot.radius,
        simulation.config.refuge.inside_door_offset,
        simulation.config.refuge.outside_door_offset,
    )[0]
    refuge_outside = simulation.environment.room_door_path(
        refuge_room,
        simulation.config.robot.radius,
        simulation.config.refuge.inside_door_offset,
        simulation.config.refuge.outside_door_offset,
    )[0]
    positions = (
        start_outside,
        refuge_room.center,
        refuge_outside,
        simulation.goal.copy(),
    )
    simulation.benchmark_scenario_metrics.update(
        {
            "blockage_started_at_s": 0.0,
            "blockage_cleared_at_s": 3.0 * simulation.config.dt,
            # Whole-corridor exit is deliberately unrelated to the witness
            # station's operational-footprint sweep interval.
            "convoy_clear_time_s": 99.0,
        }
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
                "room_0" if index == 1 else "nominal"
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
        "blockage_active",
        "blockage_cleared",
        "goal",
        "initial",
        "refuge_entered",
        "refuge_left",
        "room_selected",
        "start_room_left",
    ]
    assert artifacts["gif_frame_count"] == 5
    assert set(artifacts["snapshots"]) == {
        "blockage_active",
        "blockage_cleared",
        "initial",
        "start_room_left",
        "room_selected",
        "refuge_entered",
        "refuge_left",
        "goal",
    }
    assert gif_path.is_file()
    with Image.open(gif_path) as animation:
        assert animation.format == "GIF"
        assert animation.n_frames == 5
        assert animation.size[0] > 100
        assert animation.size[1] > 50
    for event in (
        "blockage_active",
        "blockage_cleared",
        "initial",
        "start_room_left",
        "room_selected",
        "refuge_entered",
        "refuge_left",
        "goal",
    ):
        path = snapshot_dir / f"{event}.png"
        assert path.is_file()
        with Image.open(path) as snapshot:
            assert snapshot.format == "PNG"
            assert snapshot.size[0] > 100
            assert snapshot.size[1] > 50


def test_cli_defaults_to_canonical_story_and_legacy_mode_is_explicit(
    monkeypatch,
    capsys,
) -> None:
    fake_simulation = SimpleNamespace(
        collision=False,
        reached_goal=False,
        time=0.0,
        state=np.zeros(4),
        last_controller=None,
        environment=SimpleNamespace(room_containing=lambda _point: None),
        obstacles=[],
        benchmark_scenario_metrics={
            "story_id": "main_eastbound",
            "traffic_seed": 0,
            "world_sha256": "a" * 64,
            "hospital_story_protocol_sha256": "b" * 64,
        },
        run=lambda _steps: [],
    )
    captured: dict[str, object] = {}

    def fake_story(story_id, *, traffic_seed):
        captured.update(story_id=story_id, traffic_seed=traffic_seed)

        def to_simulation(config=None):
            captured["config"] = config
            return fake_simulation

        return SimpleNamespace(to_simulation=to_simulation)

    monkeypatch.setattr(run, "build_hospital_story_scenario", fake_story)
    assert run.main(["--steps", "0"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured["story_id"] == "main_eastbound"
    assert captured["traffic_seed"] == 0
    assert captured["config"].policies.cbf_alpha == pytest.approx(
        2.1774542113693043
    )
    assert payload["run_mode"] == "canonical_story"
    assert payload["story"] == "main_eastbound"
    assert payload["world_sha256"] == "a" * 64
    assert payload["controller_config_provenance"]["study"][
        "best_trial_number"
    ] == 42

    parser = run.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--story", "main_eastbound", "--stretchers", "2"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(["--seed", "20"])


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
