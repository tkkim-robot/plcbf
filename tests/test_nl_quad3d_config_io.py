from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.nl_quad3d import benchmark
from examples.nl_quad3d import config_io
from examples.nl_quad3d import run as run_cli
from examples.nl_quad3d.config_io import (
    DEFAULT_CONTROLLER_CONFIG_PATH,
    controller_config_from_mapping,
    load_controller_config_artifact,
    load_default_controller_config,
    write_controller_config_artifact,
)
from examples.nl_quad3d.controller import NLQuad3DControllerConfig
from plcbf.benchmarking import (
    BenchmarkOutcome,
    BenchmarkReportPaths,
    BenchmarkResult,
)


def test_tuning_summary_replays_controller_and_protocol(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "best_controller": {
                    "num_radial_policies": 8,
                    "backup_horizon": 0.75,
                    "cbf_alpha": 3.0,
                },
                "configuration": {
                    "scenarios": ["head_on", "vertical_drop"],
                    "max_steps": 123,
                    "warmup": False,
                    "seed_perturbations": {
                        "position_uniform_half_width_m": 0.21,
                        "velocity_uniform_half_width_mps": 0.09,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    arguments = benchmark.build_parser().parse_args(
        ["--config-json", str(path)]
    )
    config = benchmark._config_from_args(arguments)

    assert config.plcbf_controller_config.num_radial_policies == 8
    assert config.plcbf_controller_config.backup_horizon == 0.75
    assert config.plcbf_controller_config.cbf_alpha == 3.0
    # A winning PL-CBF artifact must not retune comparison methods.
    assert config.controller_config == NLQuad3DControllerConfig()
    assert config.scenarios == ("head_on", "vertical_drop")
    assert config.max_steps == 123
    assert config.obstacle_position_perturbation == 0.21
    assert config.obstacle_velocity_perturbation == 0.09
    assert config.warmup is False
    assert config.configuration_source == str(path)


def test_controller_config_loader_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unknown"):
        controller_config_from_mapping({"not_a_controller_field": 1})

    direct = tmp_path / "controller.json"
    direct.write_text(
        json.dumps({"sensing_radius": 5.5}),
        encoding="utf-8",
    )
    config, payload = load_controller_config_artifact(direct)
    assert config.sensing_radius == 5.5
    assert payload == {"sensing_radius": 5.5}


def test_yaml_round_trip_preserves_controller_and_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "winner.yaml"
    expected = NLQuad3DControllerConfig(
        sensing_radius=6.25,
        max_obstacles=16,
        num_radial_policies=20,
    )
    write_controller_config_artifact(
        path,
        expected,
        provenance={"study": {"name": "test", "best_trial_number": 4}},
    )

    actual, payload = load_controller_config_artifact(path)

    assert actual == expected
    assert payload["schema_version"] == 1
    assert payload["provenance"]["study"]["best_trial_number"] == 4


def test_atomic_yaml_replace_failure_preserves_existing_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "winner.yaml"
    write_controller_config_artifact(
        path,
        NLQuad3DControllerConfig(cbf_alpha=2.0),
        provenance={"revision": "existing"},
    )
    original = path.read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(config_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic replace"):
        write_controller_config_artifact(
            path,
            NLQuad3DControllerConfig(cbf_alpha=6.0),
            provenance={"revision": "new"},
        )

    assert path.read_bytes() == original
    assert list(tmp_path.glob(".winner.yaml.*.tmp")) == []


def test_packaged_yaml_is_the_default_plcbf_config_only() -> None:
    expected, _ = load_default_controller_config()
    arguments = benchmark.build_parser().parse_args([])
    config = benchmark._config_from_args(arguments)

    assert DEFAULT_CONTROLLER_CONFIG_PATH.is_file()
    assert config.plcbf_controller_config == expected
    assert config.controller_config == NLQuad3DControllerConfig()
    assert config.configuration_source.startswith("packaged_plcbf:")


def test_single_run_defaults_to_frozen_stress_protocol() -> None:
    arguments = run_cli.build_parser().parse_args([])

    assert arguments.scenario == "playground_stress"


def test_explicit_config_then_cli_fields_have_highest_precedence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "explicit.yaml"
    write_controller_config_artifact(
        path,
        NLQuad3DControllerConfig(
            backup_horizon=0.75,
            num_radial_policies=8,
            cbf_alpha=4.5,
        ),
        provenance={"source": "test"},
    )
    arguments = benchmark.build_parser().parse_args(
        [
            "--config",
            str(path),
            "--backup-horizon",
            "1.5",
            "--radial-policies",
            "16",
        ]
    )

    config = benchmark._config_from_args(arguments)

    assert config.plcbf_controller_config.backup_horizon == 1.5
    assert config.plcbf_controller_config.num_radial_policies == 16
    assert config.plcbf_controller_config.cbf_alpha == 4.5
    assert config.controller_config == NLQuad3DControllerConfig()


def test_single_run_loads_yaml_then_applies_cli_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "single-run.yaml"
    write_controller_config_artifact(
        path,
        NLQuad3DControllerConfig(
            backup_horizon=0.75,
            num_radial_policies=8,
            cbf_alpha=4.25,
        ),
        provenance={"source": "test"},
    )
    captured = {}

    class FakeController:
        def __init__(self, _model, config, *, bounds):
            captured["config"] = config
            captured["bounds"] = bounds

    def fake_simulate(scenario, **_kwargs):
        return type(
            "Result",
            (),
            {
                "scenario": scenario.name,
                "steps": 1,
                "reached_goal": False,
                "collision": False,
                "minimum_clearance": 1.0,
                "final_state": scenario.initial_state,
            },
        )()

    monkeypatch.setattr(run_cli, "PLCBF_NLQuad3D", FakeController)
    monkeypatch.setattr(run_cli, "simulate", fake_simulate)

    assert run_cli.main(
        [
            "--scenario",
            "head_on",
            "--steps",
            "1",
            "--config",
            str(path),
            "--backup-horizon",
            "1.5",
            "--radial-policies",
            "16",
        ]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    assert captured["config"].backup_horizon == 1.5
    assert captured["config"].num_radial_policies == 16
    assert captured["config"].cbf_alpha == 4.25
    assert payload["controller_config_source"] == str(path)


def test_tuning_replay_rejects_a_mismatched_rollout_tilt_protocol(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mismatched-summary.json"
    path.write_text(
        json.dumps(
            {
                "best_controller": {
                    "rollout_tilt_max_rad": 1.2,
                }
            }
        ),
        encoding="utf-8",
    )
    arguments = benchmark.build_parser().parse_args(
        ["--config-json", str(path)]
    )

    with pytest.raises(ValueError, match="rollout_tilt_max_rad must equal"):
        benchmark._config_from_args(arguments)


def test_benchmark_cli_returns_nonzero_when_a_trial_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = BenchmarkResult(
        algorithm="plcbf",
        case_id="head_on/seed-0",
        seed=0,
        outcome=BenchmarkOutcome.ERROR,
        error="synthetic",
    )
    paths = BenchmarkReportPaths(
        csv=tmp_path / "results.csv",
        json=tmp_path / "results.json",
        markdown=tmp_path / "results.md",
    )
    monkeypatch.setattr(
        benchmark,
        "run_and_write",
        lambda _config, _output: ((result,), paths),
    )

    assert benchmark.main(["--quick"]) == 1
