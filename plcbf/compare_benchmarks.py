"""Build a paper summary from independently audited benchmark bundles.

This utility intentionally does not merge raw reports or claim their
method-specific controller metadata is identical.  It validates the shared
physical/evaluation protocol, requires disjoint methods and complete result
grids, then writes a JSON and Markdown aggregate comparison.

Example:

    python -m plcbf.compare_benchmarks \
        results/nonpl.json results/plcbf.json \
        --output results/nl_quad3d_final_comparison
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

from plcbf.benchmarking import (
    BenchmarkOutcome,
    BenchmarkReportSchemaError,
    BenchmarkResult,
    LoadedBenchmarkReport,
    _json_mismatch,
    aggregate_results,
    load_benchmark_report,
    merge_benchmark_results,
)


COMPARISON_SCHEMA_VERSION = 1

# These fields describe which implementation produced a bundle, rather than
# the physical scenario/evaluation protocol shared by all compared methods.
_METHOD_SCOPED_METADATA_FIELDS = frozenset(
    {
        "methods",
        "configuration_source",
        "baseline_controller",
        "plcbf_controller",
        "controller",
        "merge",
    }
)

_REQUIRED_SHARED_METADATA_FIELDS = frozenset(
    {
        "case_study",
        "dynamics",
        "collision_geometry",
        "scenarios",
        "seeds",
        "max_steps",
        "seed_perturbations",
        "warmup",
        "stop_on_collision",
    }
)


@dataclass(frozen=True)
class ComparisonReportPaths:
    """Paths written by :func:`write_comparison_reports`."""

    json: Path
    markdown: Path


def _declared_strings(
    metadata: Mapping[str, Any],
    field: str,
    *,
    source: Path,
) -> tuple[str, ...]:
    values = metadata.get(field)
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(item, str) or not item for item in values)
        or len(set(values)) != len(values)
    ):
        raise BenchmarkReportSchemaError(
            f"{source}: metadata.{field} must be a non-empty array of "
            "unique strings"
        )
    return tuple(values)


def _declared_seeds(
    metadata: Mapping[str, Any],
    *,
    source: Path,
) -> tuple[int, ...]:
    values = metadata.get("seeds")
    if (
        not isinstance(values, list)
        or not values
        or any(type(item) is not int for item in values)
        or len(set(values)) != len(values)
    ):
        raise BenchmarkReportSchemaError(
            f"{source}: metadata.seeds must be a non-empty array of "
            "unique integers"
        )
    return tuple(values)


def _identity_preview(
    identities: set[tuple[str, str, int]],
) -> str:
    ordered = sorted(identities)
    preview = ordered[:5]
    suffix = "" if len(ordered) <= 5 else f" (+{len(ordered) - 5} more)"
    return f"{preview}{suffix}"


def _validate_report_grid(
    report: LoadedBenchmarkReport,
    *,
    source: Path,
) -> set[str]:
    metadata = report.metadata
    missing_shared = sorted(_REQUIRED_SHARED_METADATA_FIELDS - set(metadata))
    if missing_shared:
        raise BenchmarkReportSchemaError(
            f"{source}: metadata is missing shared protocol fields "
            f"{missing_shared}"
        )
    for field in ("case_study", "dynamics", "collision_geometry"):
        value = metadata[field]
        if not isinstance(value, str) or not value:
            raise BenchmarkReportSchemaError(
                f"{source}: metadata.{field} must be a non-empty string"
            )
    max_steps = metadata["max_steps"]
    if type(max_steps) is not int or max_steps < 1:
        raise BenchmarkReportSchemaError(
            f"{source}: metadata.max_steps must be a positive integer"
        )
    for field in ("warmup", "stop_on_collision"):
        if type(metadata[field]) is not bool:
            raise BenchmarkReportSchemaError(
                f"{source}: metadata.{field} must be a boolean"
            )
    if not isinstance(metadata["seed_perturbations"], Mapping):
        raise BenchmarkReportSchemaError(
            f"{source}: metadata.seed_perturbations must be an object"
        )
    methods = set(_declared_strings(metadata, "methods", source=source))
    scenarios = _declared_strings(metadata, "scenarios", source=source)
    seeds = _declared_seeds(metadata, source=source)
    if "playground_stress" in scenarios:
        stress = metadata.get("stress_scenario")
        if not isinstance(stress, Mapping):
            raise BenchmarkReportSchemaError(
                f"{source}: playground_stress requires an object-valued "
                "metadata.stress_scenario"
            )
        if (
            stress.get("name") != "playground_stress"
            or not isinstance(stress.get("protocol_version"), str)
            or not stress.get("protocol_version")
        ):
            raise BenchmarkReportSchemaError(
                f"{source}: metadata.stress_scenario must name the scenario "
                "and declare a non-empty protocol_version"
            )
    observed = {
        (result.algorithm, result.case_id, result.seed)
        for result in report.results
    }
    expected = {
        (method, f"{scenario}/seed-{seed}", seed)
        for method in methods
        for scenario in scenarios
        for seed in seeds
    }
    if observed != expected:
        raise BenchmarkReportSchemaError(
            f"{source}: results do not exactly cover the declared grid; "
            f"missing={_identity_preview(expected - observed)}, "
            f"extra={_identity_preview(observed - expected)}"
        )
    errors = [
        result
        for result in report.results
        if result.outcome is BenchmarkOutcome.ERROR
    ]
    if errors:
        raise BenchmarkReportSchemaError(
            f"{source}: comparison rejects {len(errors)} ERROR result rows"
        )
    return methods


def _protocol_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in _METHOD_SCOPED_METADATA_FIELDS
    }


def _controller_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in sorted(_METHOD_SCOPED_METADATA_FIELDS)
        if key in metadata
    }


def _outcome_value(count: int, total: int) -> dict[str, int | float]:
    return {
        "count": int(count),
        "rate": count / total if total else 0.0,
    }


def _method_summaries(
    results: Sequence[BenchmarkResult],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for aggregate in aggregate_results(results):
        clearance = aggregate.clearance
        compute = aggregate.solve_time_s
        summaries.append(
            {
                "method": aggregate.algorithm,
                "trials": aggregate.trial_count,
                "outcomes": {
                    "success": _outcome_value(
                        aggregate.success_count, aggregate.trial_count
                    ),
                    "collision": _outcome_value(
                        aggregate.collision_count, aggregate.trial_count
                    ),
                    "infeasible": _outcome_value(
                        aggregate.infeasible_count, aggregate.trial_count
                    ),
                    "timeout": _outcome_value(
                        aggregate.timeout_count, aggregate.trial_count
                    ),
                },
                "clearance_m": (
                    None
                    if clearance is None
                    else {
                        "sample_count": clearance.count,
                        "mean": clearance.mean,
                        "min": clearance.minimum,
                    }
                ),
                "pooled_step_compute_time_s": (
                    None
                    if compute is None
                    else {
                        "sample_count": compute.count,
                        "mean": compute.mean,
                        "p95": compute.p95,
                        "max": compute.maximum,
                    }
                ),
            }
        )
    return summaries


def build_comparison_document(
    input_paths: Sequence[str | Path],
    *,
    title: str = "Benchmark comparison",
) -> dict[str, Any]:
    """Load and validate disjoint bundles, returning an aggregate document."""

    paths = tuple(Path(path) for path in input_paths)
    if len(paths) < 2:
        raise ValueError("at least two benchmark bundles are required")
    resolved = tuple(path.resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("input benchmark bundle paths must be unique")

    reports = tuple(load_benchmark_report(path) for path in paths)
    base_protocol: dict[str, Any] | None = None
    all_methods: set[str] = set()
    sources: list[dict[str, Any]] = []
    for index, (path, report) in enumerate(zip(paths, reports)):
        methods = _validate_report_grid(report, source=path)
        duplicate_methods = methods & all_methods
        if duplicate_methods:
            raise BenchmarkReportSchemaError(
                "comparison bundles must contain disjoint methods; duplicate "
                f"methods in {path}: {sorted(duplicate_methods)}"
            )
        all_methods.update(methods)
        protocol = _protocol_metadata(report.metadata)
        if base_protocol is None:
            base_protocol = protocol
        else:
            mismatch = _json_mismatch(
                protocol,
                base_protocol,
                f"shared protocol for input {index + 1}",
            )
            if mismatch is not None:
                raise BenchmarkReportSchemaError(
                    f"{path}: shared benchmark protocol differs: {mismatch}"
                )
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "methods": sorted(methods),
                "method_controller_metadata": _controller_metadata(
                    report.metadata
                ),
            }
        )

    assert base_protocol is not None
    results = merge_benchmark_results(
        *(report.results for report in reports),
        allow_errors=False,
    )
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "report_kind": "independent_benchmark_bundle_comparison",
        "title": str(title),
        "shared_protocol_metadata": base_protocol,
        "sources": sources,
        "compute_time_pooling": (
            "Per-method mean, p95, and max pool every per-control-step "
            "solve_times_s sample across all trials in that method's bundle."
        ),
        "methods": _method_summaries(results),
    }


def _format_rate(value: Mapping[str, int | float], total: int) -> str:
    return f"{int(value['count'])}/{total} ({float(value['rate']):.1%})"


def comparison_document_to_markdown(document: Mapping[str, Any]) -> str:
    """Render the aggregate comparison document as a compact paper table."""

    title = str(document["title"]).replace("\n", " ").replace("|", "\\|")
    lines = [
        f"# {title}",
        "",
        (
            "| Method | Trials | Success | Collision | Infeasible | Timeout | "
            "Clearance mean / min (m) | Pooled step compute mean / p95 / max (ms) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    methods = document.get("methods")
    if not isinstance(methods, list):
        raise TypeError("comparison document methods must be a list")
    for raw_summary in methods:
        if not isinstance(raw_summary, Mapping):
            raise TypeError("comparison method summary must be an object")
        total = int(raw_summary["trials"])
        outcomes = raw_summary["outcomes"]
        if not isinstance(outcomes, Mapping):
            raise TypeError("comparison outcomes must be an object")
        clearance = raw_summary["clearance_m"]
        compute = raw_summary["pooled_step_compute_time_s"]
        clearance_text = (
            "—"
            if clearance is None
            else f"{clearance['mean']:.4g} / {clearance['min']:.4g}"
        )
        compute_text = (
            "—"
            if compute is None
            else (
                f"{compute['mean'] * 1000.0:.4g} / "
                f"{compute['p95'] * 1000.0:.4g} / "
                f"{compute['max'] * 1000.0:.4g}"
            )
        )
        row = [
            str(raw_summary["method"]).replace("|", "\\|"),
            str(total),
            _format_rate(outcomes["success"], total),
            _format_rate(outcomes["collision"], total),
            _format_rate(outcomes["infeasible"], total),
            _format_rate(outcomes["timeout"], total),
            clearance_text,
            compute_text,
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            (
                "Compute-time statistics pool all per-control-step samples "
                "within each method; they are not percentiles of per-trial means."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def write_comparison_reports(
    output_prefix: str | Path,
    document: Mapping[str, Any],
    *,
    input_paths: Sequence[str | Path],
) -> ComparisonReportPaths:
    """Atomically write JSON and Markdown without replacing an input bundle."""

    prefix = Path(output_prefix)
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    input_stems = {Path(path).resolve().with_suffix("") for path in input_paths}
    if prefix.resolve() in input_stems:
        raise ValueError("comparison output prefix must not replace an input bundle")
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    serialized = json.dumps(
        document,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    markdown = comparison_document_to_markdown(document)
    _atomic_write(json_path, serialized)
    _atomic_write(markdown_path, markdown)
    return ComparisonReportPaths(json=json_path, markdown=markdown_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="two or more independently audited benchmark JSON bundles",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output prefix for aggregate .json and .md files",
    )
    parser.add_argument(
        "--title",
        default="Benchmark comparison",
        help="title for the Markdown and JSON report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        document = build_comparison_document(
            arguments.inputs,
            title=arguments.title,
        )
        paths = write_comparison_reports(
            arguments.output,
            document,
            input_paths=arguments.inputs,
        )
    except (BenchmarkReportSchemaError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "inputs": [str(path) for path in arguments.inputs],
                "methods": [item["method"] for item in document["methods"]],
                "reports": {
                    "json": str(paths.json),
                    "markdown": str(paths.markdown),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPARISON_SCHEMA_VERSION",
    "ComparisonReportPaths",
    "build_comparison_document",
    "build_parser",
    "comparison_document_to_markdown",
    "main",
    "write_comparison_reports",
]
