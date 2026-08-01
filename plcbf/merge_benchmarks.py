"""Safely merge sharded PL-CBF benchmark JSON reports.

Example:

    python -m plcbf.merge_benchmarks \
        shard-00.json shard-01.json \
        --output results/combined \
        --title "Combined benchmark"

Seed-sharded reports require the explicit ``--seed-shards`` flag.  That mode
unions only ``metadata.seeds`` and ``metadata.methods``; every other metadata
field must match exactly, and both the input shards and merged result must be
complete declared method/scenario/seed grids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from plcbf.benchmarking import (
    BenchmarkResult,
    BenchmarkReportSchemaError,
    LoadedBenchmarkReport,
    _json_mismatch,
    _json_value,
    _strict_json_loads,
    load_benchmark_report,
    merge_benchmark_results,
    write_benchmark_reports,
)


def _load_metadata(path: Path | None) -> Mapping[str, Any]:
    if path is None:
        return {}
    parsed = _strict_json_loads(path.read_bytes(), source=str(path))
    if not isinstance(parsed, dict):
        raise BenchmarkReportSchemaError(
            f"{path}: metadata root must be a JSON object"
        )
    try:
        normalized = _json_value(parsed)
    except (TypeError, ValueError) as error:
        raise BenchmarkReportSchemaError(
            f"{path}: invalid metadata: {error}"
        ) from error
    mismatch = _json_mismatch(parsed, normalized, "metadata")
    if mismatch is not None:
        raise BenchmarkReportSchemaError(f"{path}: {mismatch}")
    return normalized


def _declared_methods(
    report: LoadedBenchmarkReport,
    *,
    source: Path,
    required: bool = False,
) -> set[str]:
    observed = {result.algorithm for result in report.results}
    declared = report.metadata.get("methods")
    if declared is None:
        if required:
            raise BenchmarkReportSchemaError(
                f"{source}: seed shards require metadata.methods"
            )
        return observed
    if (
        not isinstance(declared, list)
        or any(not isinstance(item, str) or not item for item in declared)
        or len(set(declared)) != len(declared)
    ):
        raise BenchmarkReportSchemaError(
            f"{source}: metadata.methods must be an array of unique, "
            "non-empty strings"
        )
    declared_set = set(declared)
    if declared_set != observed:
        raise BenchmarkReportSchemaError(
            f"{source}: metadata.methods {sorted(declared_set)} does not "
            f"match result algorithms {sorted(observed)}"
        )
    return declared_set


def _declared_seeds(
    report: LoadedBenchmarkReport,
    *,
    source: Path,
) -> tuple[int, ...]:
    declared = report.metadata.get("seeds")
    if (
        not isinstance(declared, list)
        or not declared
        or any(type(item) is not int for item in declared)
        or len(set(declared)) != len(declared)
    ):
        raise BenchmarkReportSchemaError(
            f"{source}: seed shards require metadata.seeds to be a "
            "non-empty array of unique integers"
        )
    observed = {result.seed for result in report.results}
    if set(declared) != observed:
        raise BenchmarkReportSchemaError(
            f"{source}: metadata.seeds {sorted(declared)} does not match "
            f"result seeds {sorted(observed)}"
        )
    return tuple(declared)


def _declared_scenarios(
    report: LoadedBenchmarkReport,
    *,
    source: Path,
) -> tuple[str, ...]:
    declared = report.metadata.get("scenarios")
    if (
        not isinstance(declared, list)
        or not declared
        or any(not isinstance(item, str) or not item for item in declared)
        or len(set(declared)) != len(declared)
    ):
        raise BenchmarkReportSchemaError(
            f"{source}: seed shards require metadata.scenarios to be a "
            "non-empty array of unique strings"
        )
    return tuple(declared)


def _result_identities(
    results: Sequence[BenchmarkResult],
) -> set[tuple[str, str, int]]:
    return {
        (result.algorithm, result.case_id, result.seed)
        for result in results
    }


def _expected_grid_identities(
    methods: set[str],
    scenarios: Sequence[str],
    seeds: Sequence[int],
) -> set[tuple[str, str, int]]:
    return {
        (method, f"{scenario}/seed-{seed}", seed)
        for method in methods
        for scenario in scenarios
        for seed in seeds
    }


def _identity_difference_summary(
    identities: set[tuple[str, str, int]],
) -> str:
    ordered = sorted(identities)
    preview = ordered[:5]
    suffix = "" if len(ordered) <= 5 else f" (+{len(ordered) - 5} more)"
    return f"{preview}{suffix}"


def _validate_declared_grid(
    report: LoadedBenchmarkReport,
    *,
    source: Path,
) -> tuple[set[str], tuple[str, ...], tuple[int, ...]]:
    methods = _declared_methods(report, source=source, required=True)
    scenarios = _declared_scenarios(report, source=source)
    seeds = _declared_seeds(report, source=source)
    expected = _expected_grid_identities(methods, scenarios, seeds)
    observed = _result_identities(report.results)
    if observed != expected:
        raise BenchmarkReportSchemaError(
            f"{source}: results do not exactly cover the declared "
            "method/scenario/seed grid; missing="
            f"{_identity_difference_summary(expected - observed)}, extra="
            f"{_identity_difference_summary(observed - expected)}"
        )
    return methods, scenarios, seeds


def _infer_merged_metadata(
    reports: Sequence[LoadedBenchmarkReport],
    input_paths: Sequence[Path],
    *,
    merged_results: Sequence[BenchmarkResult],
    seed_shards: bool = False,
) -> Mapping[str, Any]:
    if not reports:
        raise ValueError("at least one report is required")
    if len(reports) != len(input_paths):
        raise ValueError("every loaded report must have one input path")
    base = dict(reports[0].metadata)
    base.pop("methods", None)
    if seed_shards:
        base.pop("seeds", None)
    all_methods: set[str] = set()
    all_seeds: set[int] = set()
    declared_scenarios: tuple[str, ...] | None = None
    for index, (report, path) in enumerate(zip(reports, input_paths)):
        comparable = dict(report.metadata)
        comparable.pop("methods", None)
        if seed_shards:
            comparable.pop("seeds", None)
        mismatch = _json_mismatch(
            comparable,
            base,
            f"metadata for input {index + 1}",
        )
        if mismatch is not None:
            raise BenchmarkReportSchemaError(
                f"{path}: shard metadata differs after ignoring only "
                + (
                    "'methods' and 'seeds': "
                    if seed_shards
                    else "'methods': "
                )
                + mismatch
            )
        if seed_shards:
            methods, scenarios, seeds = _validate_declared_grid(
                report,
                source=path,
            )
            if declared_scenarios is None:
                declared_scenarios = scenarios
            all_methods.update(methods)
            all_seeds.update(seeds)
        else:
            all_methods.update(_declared_methods(report, source=path))

    source_identities = {
        identity
        for report in reports
        for identity in _result_identities(report.results)
    }
    merged_identities = _result_identities(merged_results)
    if merged_identities != source_identities:
        raise BenchmarkReportSchemaError(
            "merged result identities do not exactly equal the union of "
            "source report identities"
        )
    if seed_shards:
        assert declared_scenarios is not None
        expected_merged = _expected_grid_identities(
            all_methods,
            declared_scenarios,
            tuple(sorted(all_seeds)),
        )
        if merged_identities != expected_merged:
            raise BenchmarkReportSchemaError(
                "seed-sharded results do not form a complete merged "
                "method/scenario/seed grid; missing="
                f"{_identity_difference_summary(expected_merged - merged_identities)}, "
                "extra="
                f"{_identity_difference_summary(merged_identities - expected_merged)}"
            )
        base["seeds"] = sorted(all_seeds)
    base["methods"] = sorted(all_methods)
    base["merge"] = {
        "source_files": [path.name for path in input_paths],
        "source_report_count": len(input_paths),
        "merged_result_count": len(merged_results),
    }
    if seed_shards:
        base["merge"]["seed_shards"] = True
    return base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="benchmark report JSON files to merge",
    )
    parser.add_argument(
        "--input",
        dest="input_options",
        action="append",
        type=Path,
        default=[],
        help="benchmark report JSON file; repeat for multiple shards",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output prefix; .csv, .json, and .md are written",
    )
    parser.add_argument(
        "--title",
        default="Merged benchmark results",
        help="title for the aggregate Markdown report",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        help="optional JSON object used as the merged report metadata",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="retain ERROR outcomes (disabled by default)",
    )
    parser.add_argument(
        "--seed-shards",
        action="store_true",
        help=(
            "safely union metadata.seeds after exact protocol and complete "
            "method/scenario/seed-grid validation"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    input_paths = tuple(arguments.inputs) + tuple(arguments.input_options)
    if not input_paths:
        print("error: at least one input benchmark JSON is required", file=sys.stderr)
        return 2
    if arguments.seed_shards and arguments.metadata_json is not None:
        print(
            "error: --seed-shards cannot be combined with --metadata-json; "
            "the inferred protocol metadata is part of shard validation",
            file=sys.stderr,
        )
        return 2

    try:
        reports = tuple(load_benchmark_report(path) for path in input_paths)
        results = merge_benchmark_results(
            *(report.results for report in reports),
            allow_errors=arguments.allow_errors,
        )
        metadata = (
            _load_metadata(arguments.metadata_json)
            if arguments.metadata_json is not None
            else _infer_merged_metadata(
                reports,
                input_paths,
                merged_results=results,
                seed_shards=arguments.seed_shards,
            )
        )
        paths = write_benchmark_reports(
            arguments.output,
            results,
            metadata=metadata,
            title=arguments.title,
        )
    except (BenchmarkReportSchemaError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "inputs": [str(path) for path in input_paths],
                "reports": {
                    "csv": str(paths.csv),
                    "json": str(paths.json),
                    "markdown": str(paths.markdown),
                },
                "trials": len(results),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
