from __future__ import annotations

import json
from pathlib import Path

import pytest

import plcbf
from plcbf.benchmarking import (
    BenchmarkOutcome,
    BenchmarkReportSchemaError,
    BenchmarkResult,
    benchmark_report_from_json,
    benchmark_results_from_json,
    load_benchmark_report,
    load_benchmark_results,
    merge_benchmark_results,
    results_to_json,
)
from plcbf.merge_benchmarks import main


def _result(
    algorithm: str,
    case_id: str,
    seed: int,
    outcome: BenchmarkOutcome | str = BenchmarkOutcome.SUCCESS,
) -> BenchmarkResult:
    return BenchmarkResult(
        algorithm=algorithm,
        case_id=case_id,
        seed=seed,
        outcome=outcome,
        min_clearance=0.25,
        intervention=0.5,
        solve_times_s=(0.01, 0.02),
        case_metrics={"guarded": True, "progress": 1.0},
        error="solver failed" if outcome == BenchmarkOutcome.ERROR else None,
    )


def _rewrite(document: str, mutation) -> str:
    payload = json.loads(document)
    mutation(payload)
    return json.dumps(payload, allow_nan=True)


def test_public_report_loader_round_trips_writer_output(tmp_path: Path) -> None:
    expected = (_result("plcbf", "case-b", 2), _result("mps", "case-a", 1))
    document = results_to_json(expected, metadata={"shard": 3})
    path = tmp_path / "shard.json"
    path.write_text(document, encoding="utf-8")

    reconstructed = benchmark_results_from_json(document)
    report = benchmark_report_from_json(document)
    loaded = load_benchmark_results(path)

    assert loaded == reconstructed
    assert report.results == reconstructed
    assert dict(report.metadata) == {"shard": 3}
    assert load_benchmark_report(path) == report
    assert [
        (item.algorithm, item.case_id, item.seed) for item in loaded
    ] == [
        ("mps", "case-a", 1),
        ("plcbf", "case-b", 2),
    ]
    assert isinstance(loaded[0].outcome, BenchmarkOutcome)
    assert dict(loaded[0].case_metrics) == {
        "guarded": True,
        "progress": 1.0,
    }


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda payload: payload.update({"foreign": True}),
            "unexpected keys",
        ),
        (
            lambda payload: payload["results"][0].pop("case_id"),
            "missing keys",
        ),
        (
            lambda payload: payload["results"][0].update({"success": False}),
            "results\\[0\\]\\.success",
        ),
        (
            lambda payload: payload["aggregates"][0].update(
                {"success_count": 99}
            ),
            "aggregates\\[0\\]\\.success_count",
        ),
    ],
)
def test_report_loader_rejects_schema_or_derived_data_tampering(
    mutation,
    match: str,
) -> None:
    document = _rewrite(results_to_json((_result("plcbf", "case", 0),)), mutation)

    with pytest.raises(BenchmarkReportSchemaError, match=match):
        benchmark_results_from_json(document)


def test_report_loader_rejects_nonfinite_numbers_and_duplicate_json_keys() -> None:
    document = results_to_json((_result("plcbf", "case", 0),))
    nonfinite = document.replace('"min_clearance": 0.25', '"min_clearance": NaN')

    with pytest.raises(BenchmarkReportSchemaError, match="non-standard JSON"):
        benchmark_results_from_json(nonfinite)
    with pytest.raises(BenchmarkReportSchemaError, match="duplicate JSON"):
        benchmark_results_from_json(
            '{"metadata": {}, "metadata": {}, "aggregates": [], "results": []}'
        )


def test_report_loader_rejects_wrong_scalar_types_even_when_python_compares_equal() -> None:
    document = _rewrite(
        results_to_json((_result("plcbf", "case", 1),)),
        lambda payload: payload["results"][0].update({"seed": True}),
    )

    with pytest.raises(BenchmarkReportSchemaError, match="seed"):
        benchmark_results_from_json(document)


def test_merge_sorts_disjoint_shards_and_rejects_duplicate_identity() -> None:
    first = (_result("plcbf", "case-b", 2),)
    second = (_result("mps", "case-a", 1),)

    merged = merge_benchmark_results(first, second)
    assert [
        (item.algorithm, item.case_id, item.seed) for item in merged
    ] == [
        ("mps", "case-a", 1),
        ("plcbf", "case-b", 2),
    ]

    with pytest.raises(ValueError, match="duplicate benchmark row"):
        merge_benchmark_results(first, (_result("plcbf", "case-b", 2),))


def test_merge_rejects_error_outcomes_unless_explicitly_allowed() -> None:
    error = _result("plcbf", "case", 0, BenchmarkOutcome.ERROR)

    with pytest.raises(ValueError, match="ERROR row rejected"):
        merge_benchmark_results((error,))
    assert merge_benchmark_results((error,), allow_errors=True) == (error,)


def test_merge_cli_writes_valid_bundle_and_exact_optional_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    metadata = tmp_path / "metadata.json"
    output = tmp_path / "combined"
    first.write_text(
        results_to_json((_result("plcbf", "case", 0),), metadata={"part": 1}),
        encoding="utf-8",
    )
    second.write_text(
        results_to_json((_result("mps", "case", 0),), metadata={"part": 2}),
        encoding="utf-8",
    )
    metadata.write_text(
        json.dumps({"protocol": {"seed_count": 1}}), encoding="utf-8"
    )

    return_code = main(
        [
            str(first),
            "--input",
            str(second),
            "--output",
            str(output),
            "--title",
            "Sharded benchmark",
            "--metadata-json",
            str(metadata),
        ]
    )

    assert return_code == 0
    assert (tmp_path / "combined.csv").is_file()
    assert (tmp_path / "combined.md").read_text(encoding="utf-8").startswith(
        "# Sharded benchmark"
    )
    combined_json = tmp_path / "combined.json"
    assert len(load_benchmark_results(combined_json)) == 2
    assert json.loads(combined_json.read_text(encoding="utf-8"))["metadata"] == {
        "protocol": {"seed_count": 1}
    }
    output_payload = json.loads(capsys.readouterr().out)
    assert output_payload["trials"] == 2


def test_merge_cli_infers_matching_protocol_and_unions_methods(
    tmp_path: Path,
) -> None:
    first = tmp_path / "plcbf-shard.json"
    second = tmp_path / "mps-shard.json"
    output = tmp_path / "combined"
    shared = {
        "case_study": "synthetic",
        "seeds": [0, 1],
        "protocol": {"dt": 0.05},
    }
    first.write_text(
        results_to_json(
            (_result("plcbf", "case", 0),),
            metadata={**shared, "methods": ["plcbf"]},
        ),
        encoding="utf-8",
    )
    second.write_text(
        results_to_json(
            (_result("mps", "case", 0),),
            metadata={**shared, "methods": ["mps"]},
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                str(first),
                str(second),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    metadata = json.loads(
        output.with_suffix(".json").read_text(encoding="utf-8")
    )["metadata"]
    assert metadata["case_study"] == "synthetic"
    assert metadata["methods"] == ["mps", "plcbf"]
    assert metadata["merge"] == {
        "merged_result_count": 2,
        "source_files": ["plcbf-shard.json", "mps-shard.json"],
        "source_report_count": 2,
    }


def test_merge_cli_seed_shards_unions_seeds_and_validates_full_grid(
    tmp_path: Path,
) -> None:
    first = tmp_path / "seeds-1-2.json"
    second = tmp_path / "seeds-3-4.json"
    output = tmp_path / "combined"
    shared = {
        "case_study": "nl_quad3d",
        "methods": ["mps"],
        "scenarios": ["playground_stress"],
        "protocol": {"version": "stress-v2", "steps": 800},
    }
    first.write_text(
        results_to_json(
            (
                _result("mps", "playground_stress/seed-1", 1),
                _result("mps", "playground_stress/seed-2", 2),
            ),
            metadata={**shared, "seeds": [1, 2]},
        ),
        encoding="utf-8",
    )
    second.write_text(
        results_to_json(
            (
                _result("mps", "playground_stress/seed-3", 3),
                _result("mps", "playground_stress/seed-4", 4),
            ),
            metadata={**shared, "seeds": [3, 4]},
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                str(first),
                str(second),
                "--output",
                str(output),
                "--seed-shards",
            ]
        )
        == 0
    )
    report = load_benchmark_report(output.with_suffix(".json"))
    assert report.metadata["methods"] == ["mps"]
    assert report.metadata["seeds"] == [1, 2, 3, 4]
    assert report.metadata["merge"]["seed_shards"] is True
    assert {
        (item.algorithm, item.case_id, item.seed) for item in report.results
    } == {
        ("mps", f"playground_stress/seed-{seed}", seed)
        for seed in range(1, 5)
    }


def test_merge_cli_accepts_complete_method_by_seed_sharding(
    tmp_path: Path,
) -> None:
    inputs: list[str] = []
    shared = {
        "case_study": "nl_quad3d",
        "scenarios": ["playground_stress"],
        "protocol": {"version": "stress-v2", "steps": 800},
    }
    for method in ("mps", "pcbf"):
        for seeds in ((1, 2), (3, 4)):
            path = tmp_path / f"{method}-{seeds[0]}-{seeds[-1]}.json"
            path.write_text(
                results_to_json(
                    tuple(
                        _result(
                            method,
                            f"playground_stress/seed-{seed}",
                            seed,
                        )
                        for seed in seeds
                    ),
                    metadata={
                        **shared,
                        "methods": [method],
                        "seeds": list(seeds),
                    },
                ),
                encoding="utf-8",
            )
            inputs.append(str(path))

    output = tmp_path / "combined"
    assert (
        main(
            [
                *inputs,
                "--output",
                str(output),
                "--seed-shards",
            ]
        )
        == 0
    )
    report = load_benchmark_report(output.with_suffix(".json"))
    assert report.metadata["methods"] == ["mps", "pcbf"]
    assert report.metadata["seeds"] == [1, 2, 3, 4]
    assert len(report.results) == 8


def test_merge_cli_seed_shards_rejects_nonseed_protocol_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "combined"
    shared = {
        "methods": ["mps"],
        "scenarios": ["playground_stress"],
    }
    first.write_text(
        results_to_json(
            (_result("mps", "playground_stress/seed-1", 1),),
            metadata={
                **shared,
                "seeds": [1],
                "protocol": {"steps": 800},
            },
        ),
        encoding="utf-8",
    )
    second.write_text(
        results_to_json(
            (_result("mps", "playground_stress/seed-2", 2),),
            metadata={
                **shared,
                "seeds": [2],
                "protocol": {"steps": 799},
            },
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                str(first),
                str(second),
                "--output",
                str(output),
                "--seed-shards",
            ]
        )
        == 2
    )
    assert "metadata differs" in capsys.readouterr().err
    assert not output.with_suffix(".json").exists()


def test_merge_cli_seed_shards_rejects_declared_or_merged_grid_holes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad_declaration = tmp_path / "bad-declaration.json"
    output = tmp_path / "combined"
    bad_declaration.write_text(
        results_to_json(
            (_result("mps", "playground_stress/seed-1", 1),),
            metadata={
                "methods": ["mps"],
                "scenarios": ["playground_stress"],
                "seeds": [1, 2],
            },
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                str(bad_declaration),
                "--output",
                str(output),
                "--seed-shards",
            ]
        )
        == 2
    )
    assert "metadata.seeds" in capsys.readouterr().err

    mps = tmp_path / "mps-seed-1.json"
    plcbf = tmp_path / "plcbf-seed-2.json"
    shared = {"scenarios": ["playground_stress"], "protocol": "same"}
    mps.write_text(
        results_to_json(
            (_result("mps", "playground_stress/seed-1", 1),),
            metadata={**shared, "methods": ["mps"], "seeds": [1]},
        ),
        encoding="utf-8",
    )
    plcbf.write_text(
        results_to_json(
            (_result("plcbf", "playground_stress/seed-2", 2),),
            metadata={**shared, "methods": ["plcbf"], "seeds": [2]},
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                str(mps),
                str(plcbf),
                "--output",
                str(output),
                "--seed-shards",
            ]
        )
        == 2
    )
    assert "complete merged" in capsys.readouterr().err
    assert not output.with_suffix(".json").exists()


def test_merge_cli_seed_shards_forbids_metadata_override(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}", encoding="utf-8")

    assert (
        main(
            [
                str(tmp_path / "missing.json"),
                "--output",
                str(tmp_path / "combined"),
                "--seed-shards",
                "--metadata-json",
                str(metadata),
            ]
        )
        == 2
    )
    assert "cannot be combined" in capsys.readouterr().err


def test_merge_cli_rejects_protocol_mismatch_without_metadata_override(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "combined"
    first.write_text(
        results_to_json(
            (_result("plcbf", "case", 0),),
            metadata={"methods": ["plcbf"], "dt": 0.05},
        ),
        encoding="utf-8",
    )
    second.write_text(
        results_to_json(
            (_result("mps", "case", 0),),
            metadata={"methods": ["mps"], "dt": 0.1},
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                str(first),
                str(second),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert "metadata differs" in capsys.readouterr().err
    assert not output.with_suffix(".json").exists()


def test_merge_cli_fails_closed_before_writing_duplicate_or_error_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    duplicate_a = tmp_path / "duplicate-a.json"
    duplicate_b = tmp_path / "duplicate-b.json"
    output = tmp_path / "combined"
    row = _result("plcbf", "case", 0)
    duplicate_a.write_text(results_to_json((row,)), encoding="utf-8")
    duplicate_b.write_text(results_to_json((row,)), encoding="utf-8")

    assert (
        main(
            [
                str(duplicate_a),
                str(duplicate_b),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert "duplicate benchmark row" in capsys.readouterr().err
    assert not output.with_suffix(".json").exists()

    error_path = tmp_path / "error.json"
    error_path.write_text(
        results_to_json(
            (_result("plcbf", "other", 1, BenchmarkOutcome.ERROR),)
        ),
        encoding="utf-8",
    )
    assert main([str(error_path), "--output", str(output)]) == 2
    assert "ERROR row rejected" in capsys.readouterr().err
    assert (
        main(
            [
                str(error_path),
                "--output",
                str(output),
                "--allow-errors",
            ]
        )
        == 0
    )
    assert load_benchmark_results(output.with_suffix(".json"))[0].outcome is (
        BenchmarkOutcome.ERROR
    )


def test_loading_and_merge_apis_are_exported_from_package_root() -> None:
    assert plcbf.BenchmarkReportSchemaError is BenchmarkReportSchemaError
    assert plcbf.benchmark_report_from_json is benchmark_report_from_json
    assert plcbf.benchmark_results_from_json is benchmark_results_from_json
    assert plcbf.load_benchmark_report is load_benchmark_report
    assert plcbf.load_benchmark_results is load_benchmark_results
    assert plcbf.merge_benchmark_results is merge_benchmark_results
