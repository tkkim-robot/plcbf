from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from examples.nl_quad3d.benchmark import NLQuad3DBenchmarkConfig
from examples.nl_quad3d.controller import NLQuad3DControllerConfig
from plcbf.benchmarking import (
    BenchmarkOutcome,
    BenchmarkReportSchemaError,
    BenchmarkResult,
    results_to_json,
)
from plcbf.compare_benchmarks import (
    build_comparison_document,
    comparison_document_to_markdown,
    main,
)


def _metadata(
    methods: tuple[str, ...],
    *,
    seeds: tuple[int, ...] = (1, 2),
    controller_tag: str,
) -> dict[str, object]:
    return {
        "case_study": "nl_quad3d",
        "dynamics": "nonlinear_12_state_quadrotor",
        "collision_geometry": "shifted safety point against physical spheres",
        "methods": list(methods),
        "scenarios": ["playground_stress"],
        "seeds": list(seeds),
        "max_steps": 800,
        "safe_value_threshold": 0.0,
        "configuration_source": f"source-{controller_tag}",
        "stress_scenario": {
            "name": "playground_stress",
            "protocol_version": "balanced_six_axis_streams_v2",
            "obstacle_count": 48,
            "structured_stream_count": 24,
            "corridor_random_count": 24,
        },
        "seed_perturbations": {
            "position_uniform_half_width_m": 0.12,
            "velocity_uniform_half_width_mps": 0.08,
        },
        "warmup": True,
        "stop_on_collision": True,
        "state_validity": {"tilt_max_deg": 60.0},
        "timing": "native per-step wall time",
        "baseline_controller": {"tag": f"baseline-{controller_tag}"},
        "plcbf_controller": {"tag": f"plcbf-{controller_tag}"},
        "controller": {"tag": f"controller-{controller_tag}"},
        "controller_scope": "method-specific controllers",
        "merge": {"source_report_count": 10, "tag": controller_tag},
    }


def _result(
    method: str,
    seed: int,
    outcome: BenchmarkOutcome | str,
    clearance: float,
    solve_times_s: tuple[float, ...],
) -> BenchmarkResult:
    parsed = BenchmarkOutcome(outcome)
    return BenchmarkResult(
        algorithm=method,
        case_id=f"playground_stress/seed-{seed}",
        seed=seed,
        outcome=parsed,
        min_clearance=clearance,
        intervention=0.25,
        solve_times_s=solve_times_s,
        error="synthetic failure" if parsed is BenchmarkOutcome.ERROR else None,
    )


def _write_bundle(
    path: Path,
    methods: tuple[str, ...],
    rows: tuple[BenchmarkResult, ...],
    *,
    metadata: dict[str, object] | None = None,
    controller_tag: str,
) -> Path:
    payload = (
        _metadata(methods, controller_tag=controller_tag)
        if metadata is None
        else metadata
    )
    path.write_text(
        results_to_json(rows, metadata=payload),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def comparison_bundles(tmp_path: Path) -> tuple[Path, Path]:
    baseline = _write_bundle(
        tmp_path / "nonpl.json",
        ("mps", "pcbf"),
        (
            _result(
                "mps", 1, BenchmarkOutcome.SUCCESS, 0.2, (0.001, 0.003)
            ),
            _result(
                "mps", 2, BenchmarkOutcome.COLLISION, 0.1, (0.005, 0.007)
            ),
            _result(
                "pcbf", 1, BenchmarkOutcome.SUCCESS, 0.3, (0.002,)
            ),
            _result(
                "pcbf", 2, BenchmarkOutcome.INFEASIBLE, 0.0, (0.004,)
            ),
        ),
        controller_tag="baseline",
    )
    plcbf = _write_bundle(
        tmp_path / "plcbf.json",
        ("plcbf",),
        (
            _result(
                "plcbf", 1, BenchmarkOutcome.SUCCESS, 0.4, (0.008, 0.010)
            ),
            _result(
                "plcbf", 2, BenchmarkOutcome.TIMEOUT, 0.2, (0.012, 0.014)
            ),
        ),
        controller_tag="tuned",
    )
    return baseline, plcbf


def test_comparison_joins_disjoint_methods_and_pools_step_times(
    comparison_bundles: tuple[Path, Path],
) -> None:
    baseline, plcbf = comparison_bundles
    document = build_comparison_document(
        (baseline, plcbf),
        title="NL Quad3D paper comparison",
    )

    assert [item["method"] for item in document["methods"]] == [
        "mps",
        "pcbf",
        "plcbf",
    ]
    mps = document["methods"][0]
    assert mps["trials"] == 2
    assert mps["outcomes"]["success"] == {"count": 1, "rate": 0.5}
    assert mps["outcomes"]["collision"] == {"count": 1, "rate": 0.5}
    assert mps["outcomes"]["infeasible"] == {"count": 0, "rate": 0.0}
    assert mps["outcomes"]["timeout"] == {"count": 0, "rate": 0.0}
    assert mps["clearance_m"]["mean"] == pytest.approx(0.15)
    assert mps["clearance_m"]["min"] == pytest.approx(0.1)
    compute = mps["pooled_step_compute_time_s"]
    assert compute["sample_count"] == 4
    assert compute["mean"] == pytest.approx(0.004)
    assert compute["p95"] == pytest.approx(0.0067)
    assert compute["max"] == pytest.approx(0.007)

    shared = document["shared_protocol_metadata"]
    assert shared["max_steps"] == 800
    assert shared["stress_scenario"]["protocol_version"].endswith("_v2")
    for field in (
        "methods",
        "configuration_source",
        "baseline_controller",
        "plcbf_controller",
        "controller",
        "merge",
    ):
        assert field not in shared
    sources = document["sources"]
    assert sources[0]["method_controller_metadata"] != (
        sources[1]["method_controller_metadata"]
    )

    markdown = comparison_document_to_markdown(document)
    assert "Clearance mean / min (m)" in markdown
    assert "Pooled step compute mean / p95 / max (ms)" in markdown
    assert "1/2 (50.0%)" in markdown
    assert "4 / 6.7 / 7" in markdown


def test_comparison_accepts_real_nl_metadata_with_distinct_pl_controller(
    tmp_path: Path,
) -> None:
    base_controller = NLQuad3DControllerConfig()
    tuned_controller = replace(base_controller, cbf_alpha=3.75)
    common = {
        "scenarios": ("playground_stress",),
        "seeds": (1, 2),
        "max_steps": 800,
        "playground_obstacle_count": 48,
    }
    baseline_config = NLQuad3DBenchmarkConfig(
        methods=("mps", "pcbf"),
        controller_config=base_controller,
        plcbf_controller_config=base_controller,
        configuration_source="pre_tuning_packaged_controller",
        **common,
    )
    plcbf_config = NLQuad3DBenchmarkConfig(
        methods=("plcbf",),
        controller_config=base_controller,
        plcbf_controller_config=tuned_controller,
        configuration_source="frozen_optuna_winner",
        **common,
    )
    baseline = _write_bundle(
        tmp_path / "real-nonpl.json",
        ("mps", "pcbf"),
        tuple(
            _result(method, seed, BenchmarkOutcome.SUCCESS, 0.2, (0.001,))
            for method in ("mps", "pcbf")
            for seed in (1, 2)
        ),
        metadata=baseline_config.metadata(),
        controller_tag="unused",
    )
    plcbf = _write_bundle(
        tmp_path / "real-plcbf.json",
        ("plcbf",),
        tuple(
            _result("plcbf", seed, BenchmarkOutcome.SUCCESS, 0.3, (0.002,))
            for seed in (1, 2)
        ),
        metadata=plcbf_config.metadata(),
        controller_tag="unused",
    )

    document = build_comparison_document((baseline, plcbf))

    assert [item["method"] for item in document["methods"]] == [
        "mps",
        "pcbf",
        "plcbf",
    ]
    sources = document["sources"]
    assert sources[0]["method_controller_metadata"]["plcbf_controller"] != (
        sources[1]["method_controller_metadata"]["plcbf_controller"]
    )


def test_comparison_cli_writes_json_and_markdown_without_touching_inputs(
    comparison_bundles: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline, plcbf = comparison_bundles
    baseline_before = baseline.read_bytes()
    plcbf_before = plcbf.read_bytes()
    output = tmp_path / "paper-summary"

    assert (
        main(
            [
                str(baseline),
                str(plcbf),
                "--output",
                str(output),
                "--title",
                "Paper table",
            ]
        )
        == 0
    )
    assert baseline.read_bytes() == baseline_before
    assert plcbf.read_bytes() == plcbf_before
    payload = json.loads(output.with_suffix(".json").read_text())
    assert payload["title"] == "Paper table"
    assert len(payload["methods"]) == 3
    assert output.with_suffix(".md").read_text().startswith("# Paper table")
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["methods"] == ["mps", "pcbf", "plcbf"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda metadata: metadata.update({"case_study": "hospital"}),
        lambda metadata: metadata.update({"collision_geometry": "different"}),
        lambda metadata: metadata.update({"max_steps": 799}),
        lambda metadata: metadata.update({"warmup": False}),
        lambda metadata: metadata["stress_scenario"].update(
            {"protocol_version": "balanced_six_axis_streams_v3"}
        ),
        lambda metadata: metadata["seed_perturbations"].update(
            {"velocity_uniform_half_width_mps": 0.09}
        ),
    ],
)
def test_comparison_rejects_any_shared_protocol_mismatch(
    comparison_bundles: tuple[Path, Path],
    tmp_path: Path,
    mutation,
) -> None:
    baseline, original_plcbf = comparison_bundles
    metadata = _metadata(("plcbf",), controller_tag="tuned")
    mutation(metadata)
    changed = _write_bundle(
        tmp_path / "changed.json",
        ("plcbf",),
        (
            _result("plcbf", 1, BenchmarkOutcome.SUCCESS, 0.4, (0.01,)),
            _result("plcbf", 2, BenchmarkOutcome.SUCCESS, 0.3, (0.01,)),
        ),
        metadata=metadata,
        controller_tag="unused",
    )
    assert original_plcbf.is_file()

    with pytest.raises(BenchmarkReportSchemaError, match="protocol differs"):
        build_comparison_document((baseline, changed))


def test_comparison_rejects_seed_or_scenario_grid_mismatch(
    comparison_bundles: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    baseline, _ = comparison_bundles
    metadata = _metadata(
        ("plcbf",),
        seeds=(1, 3),
        controller_tag="tuned",
    )
    changed = _write_bundle(
        tmp_path / "changed-grid.json",
        ("plcbf",),
        (
            _result("plcbf", 1, BenchmarkOutcome.SUCCESS, 0.4, (0.01,)),
            _result("plcbf", 3, BenchmarkOutcome.SUCCESS, 0.3, (0.01,)),
        ),
        metadata=metadata,
        controller_tag="unused",
    )

    with pytest.raises(BenchmarkReportSchemaError, match="protocol differs"):
        build_comparison_document((baseline, changed))


def test_comparison_rejects_duplicate_methods_errors_and_grid_holes(
    comparison_bundles: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    baseline, _ = comparison_bundles
    duplicate = _write_bundle(
        tmp_path / "duplicate.json",
        ("mps",),
        (
            _result("mps", 1, BenchmarkOutcome.SUCCESS, 0.2, (0.01,)),
            _result("mps", 2, BenchmarkOutcome.SUCCESS, 0.2, (0.01,)),
        ),
        controller_tag="duplicate",
    )
    with pytest.raises(BenchmarkReportSchemaError, match="duplicate methods"):
        build_comparison_document((baseline, duplicate))

    error = _write_bundle(
        tmp_path / "error.json",
        ("plcbf",),
        (
            _result("plcbf", 1, BenchmarkOutcome.ERROR, 0.2, (0.01,)),
            _result("plcbf", 2, BenchmarkOutcome.SUCCESS, 0.2, (0.01,)),
        ),
        controller_tag="error",
    )
    with pytest.raises(BenchmarkReportSchemaError, match="ERROR"):
        build_comparison_document((baseline, error))

    hole = _write_bundle(
        tmp_path / "hole.json",
        ("plcbf",),
        (_result("plcbf", 1, BenchmarkOutcome.SUCCESS, 0.2, (0.01,)),),
        controller_tag="hole",
    )
    with pytest.raises(BenchmarkReportSchemaError, match="declared grid"):
        build_comparison_document((baseline, hole))


def test_comparison_output_must_not_replace_an_input_bundle(
    comparison_bundles: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline, plcbf = comparison_bundles
    original = baseline.read_bytes()

    assert (
        main(
            [
                str(baseline),
                str(plcbf),
                "--output",
                str(baseline.with_suffix("")),
            ]
        )
        == 2
    )
    assert "must not replace" in capsys.readouterr().err
    assert baseline.read_bytes() == original
