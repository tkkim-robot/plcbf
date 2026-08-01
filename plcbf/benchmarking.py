"""Deterministic benchmark records, aggregation, and report writers.

The benchmark layer is intentionally independent of a particular dynamics
model.  Case studies produce one :class:`BenchmarkResult` per algorithm and
scenario; this module then provides stable CSV, JSON, and Markdown output.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, TypeAlias


MetricValue: TypeAlias = bool | int | float | str | None


class BenchmarkOutcome(str, Enum):
    """Exclusive terminal outcome for one algorithm/scenario run."""

    SUCCESS = "success"
    COLLISION = "collision"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    ERROR = "error"


class BenchmarkReportSchemaError(ValueError):
    """Raised when a benchmark report is not one of our valid JSON reports."""


def _finite_optional(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not _is_finite(result):
        raise ValueError(f"{name} must be finite when provided")
    return result


def _is_finite(value: float) -> bool:
    # Avoid a NumPy dependency in the reporting layer.
    return value == value and value not in (float("inf"), float("-inf"))


def _normalize_metric(value: Any, name: str) -> MetricValue:
    scalar_item = getattr(value, "item", None)
    if callable(scalar_item):
        scalar = scalar_item()
        if scalar is not value:
            return _normalize_metric(scalar, name)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not _is_finite(result):
            raise ValueError(f"case metric {name!r} must be finite")
        return result
    raise TypeError(
        f"case metric {name!r} must be a scalar string, bool, number, or None"
    )


@dataclass(frozen=True)
class BenchmarkResult:
    """Result of one algorithm on one deterministic benchmark case."""

    algorithm: str
    case_id: str
    seed: int
    outcome: BenchmarkOutcome | str
    min_clearance: float | None = None
    intervention: float | None = None
    solve_times_s: tuple[float, ...] = ()
    case_metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        algorithm = str(self.algorithm)
        case_id = str(self.case_id)
        if not algorithm:
            raise ValueError("algorithm must not be empty")
        if not case_id:
            raise ValueError("case_id must not be empty")
        if not isinstance(self.seed, Integral):
            raise TypeError("seed must be an integer")
        try:
            outcome = BenchmarkOutcome(self.outcome)
        except ValueError as error:
            choices = ", ".join(item.value for item in BenchmarkOutcome)
            raise ValueError(
                f"unknown outcome {self.outcome!r}; choose {choices}"
            ) from error

        solve_times = tuple(float(value) for value in self.solve_times_s)
        if any(not _is_finite(value) or value < 0.0 for value in solve_times):
            raise ValueError("solve_times_s must be finite and nonnegative")
        metrics: dict[str, MetricValue] = {}
        for raw_name, raw_value in self.case_metrics.items():
            name = str(raw_name)
            if not name:
                raise ValueError("case metric names must not be empty")
            metrics[name] = _normalize_metric(raw_value, name)

        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(
            self,
            "min_clearance",
            _finite_optional(self.min_clearance, "min_clearance"),
        )
        intervention = _finite_optional(self.intervention, "intervention")
        if intervention is not None and intervention < 0.0:
            raise ValueError("intervention must be nonnegative")
        object.__setattr__(self, "intervention", intervention)
        object.__setattr__(self, "solve_times_s", solve_times)
        object.__setattr__(
            self, "case_metrics", MappingProxyType(dict(sorted(metrics.items())))
        )
        object.__setattr__(
            self, "error", None if self.error is None else str(self.error)
        )

    @property
    def success(self) -> bool:
        return self.outcome is BenchmarkOutcome.SUCCESS

    @property
    def collision(self) -> bool:
        return self.outcome is BenchmarkOutcome.COLLISION

    @property
    def infeasible(self) -> bool:
        return self.outcome is BenchmarkOutcome.INFEASIBLE

    @property
    def timeout(self) -> bool:
        return self.outcome is BenchmarkOutcome.TIMEOUT

    @property
    def solve_time_mean_s(self) -> float | None:
        return _mean(self.solve_times_s)

    @property
    def solve_time_p95_s(self) -> float | None:
        return _percentile(self.solve_times_s, 0.95)

    @property
    def solve_time_max_s(self) -> float | None:
        return max(self.solve_times_s, default=None)


@dataclass(frozen=True)
class LoadedBenchmarkReport:
    """Strictly validated metadata and rows loaded from a JSON report."""

    metadata: Mapping[str, Any]
    results: tuple[BenchmarkResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        results = tuple(self.results)
        if any(not isinstance(item, BenchmarkResult) for item in results):
            raise TypeError("results must contain BenchmarkResult objects")
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )
        object.__setattr__(self, "results", results)


def _mean(values: Iterable[float]) -> float | None:
    sequence = tuple(values)
    if not sequence:
        return None
    return sum(sequence) / len(sequence)


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    sequence = sorted(float(value) for value in values)
    if not sequence:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    position = (len(sequence) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sequence) - 1)
    fraction = position - lower_index
    return (
        sequence[lower_index] * (1.0 - fraction)
        + sequence[upper_index] * fraction
    )


@dataclass(frozen=True)
class NumericSummary:
    """Deterministic descriptive statistics for finite scalar samples."""

    count: int
    mean: float
    minimum: float
    p95: float
    maximum: float

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "NumericSummary | None":
        sequence = tuple(float(value) for value in values)
        if not sequence:
            return None
        if any(not _is_finite(value) for value in sequence):
            raise ValueError("summary values must be finite")
        percentile = _percentile(sequence, 0.95)
        assert percentile is not None
        mean = _mean(sequence)
        assert mean is not None
        return cls(
            count=len(sequence),
            mean=mean,
            minimum=min(sequence),
            p95=percentile,
            maximum=max(sequence),
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "mean": self.mean,
            "min": self.minimum,
            "p95": self.p95,
            "max": self.maximum,
        }


@dataclass(frozen=True)
class BenchmarkAggregate:
    """Per-algorithm aggregate over a common set of benchmark cases."""

    algorithm: str
    trial_count: int
    success_count: int
    collision_count: int
    infeasible_count: int
    timeout_count: int
    error_count: int
    clearance: NumericSummary | None
    intervention: NumericSummary | None
    solve_time_s: NumericSummary | None
    case_metrics: Mapping[str, NumericSummary] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case_metrics",
            MappingProxyType(dict(sorted(self.case_metrics.items()))),
        )

    @property
    def success_rate(self) -> float:
        return self.success_count / self.trial_count if self.trial_count else 0.0

    @property
    def collision_rate(self) -> float:
        return (
            self.collision_count / self.trial_count if self.trial_count else 0.0
        )

    @property
    def infeasible_rate(self) -> float:
        return (
            self.infeasible_count / self.trial_count
            if self.trial_count
            else 0.0
        )

    @property
    def timeout_rate(self) -> float:
        return self.timeout_count / self.trial_count if self.trial_count else 0.0

    @property
    def error_rate(self) -> float:
        return self.error_count / self.trial_count if self.trial_count else 0.0

    @property
    def solve_time_mean_s(self) -> float | None:
        return None if self.solve_time_s is None else self.solve_time_s.mean

    @property
    def solve_time_p95_s(self) -> float | None:
        return None if self.solve_time_s is None else self.solve_time_s.p95

    @property
    def solve_time_max_s(self) -> float | None:
        return None if self.solve_time_s is None else self.solve_time_s.maximum


def _result_sort_key(
    result: BenchmarkResult,
) -> tuple[str, str, int, str]:
    return (
        result.algorithm,
        result.case_id,
        result.seed,
        result.outcome.value,
    )


def merge_benchmark_results(
    *result_sets: Iterable[BenchmarkResult],
    allow_errors: bool = False,
) -> tuple[BenchmarkResult, ...]:
    """Merge shards while enforcing one terminal row per method/case/seed.

    The identity deliberately excludes the outcome: two rows for the same
    algorithm, case, and seed are duplicate executions even if their outcomes
    disagree.  Error outcomes are rejected by default so a partially failed
    shard cannot silently enter a publication report.
    """

    merged: list[BenchmarkResult] = []
    seen: dict[tuple[str, str, int], tuple[int, int]] = {}
    for shard_index, result_set in enumerate(result_sets):
        for row_index, result in enumerate(result_set):
            if not isinstance(result, BenchmarkResult):
                raise TypeError(
                    "benchmark shards must contain BenchmarkResult objects"
                )
            identity = (result.algorithm, result.case_id, result.seed)
            previous = seen.get(identity)
            if previous is not None:
                previous_shard, previous_row = previous
                raise ValueError(
                    "duplicate benchmark row for "
                    f"algorithm={result.algorithm!r}, case_id={result.case_id!r}, "
                    f"seed={result.seed}; first seen in shard "
                    f"{previous_shard + 1}, row {previous_row + 1}"
                )
            if (
                result.outcome is BenchmarkOutcome.ERROR
                and not allow_errors
            ):
                raise ValueError(
                    "benchmark ERROR row rejected for "
                    f"algorithm={result.algorithm!r}, case_id={result.case_id!r}, "
                    f"seed={result.seed}; pass allow_errors=True only for "
                    "diagnostic reports"
                )
            seen[identity] = (shard_index, row_index)
            merged.append(result)
    return tuple(sorted(merged, key=_result_sort_key))


def aggregate_results(
    results: Iterable[BenchmarkResult],
) -> tuple[BenchmarkAggregate, ...]:
    """Aggregate results by algorithm in deterministic lexical order."""

    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        if not isinstance(result, BenchmarkResult):
            raise TypeError("results must contain BenchmarkResult objects")
        grouped.setdefault(result.algorithm, []).append(result)

    aggregates: list[BenchmarkAggregate] = []
    for algorithm in sorted(grouped):
        trials = sorted(grouped[algorithm], key=_result_sort_key)
        metric_values: dict[str, list[float]] = {}
        for trial in trials:
            for name, value in trial.case_metrics.items():
                if isinstance(value, bool):
                    metric_values.setdefault(name, []).append(float(value))
                elif isinstance(value, Real):
                    metric_values.setdefault(name, []).append(float(value))
        metric_summaries = {
            name: summary
            for name, values in sorted(metric_values.items())
            if (summary := NumericSummary.from_values(values)) is not None
        }
        aggregates.append(
            BenchmarkAggregate(
                algorithm=algorithm,
                trial_count=len(trials),
                success_count=sum(trial.success for trial in trials),
                collision_count=sum(trial.collision for trial in trials),
                infeasible_count=sum(trial.infeasible for trial in trials),
                timeout_count=sum(trial.timeout for trial in trials),
                error_count=sum(
                    trial.outcome is BenchmarkOutcome.ERROR for trial in trials
                ),
                clearance=NumericSummary.from_values(
                    trial.min_clearance
                    for trial in trials
                    if trial.min_clearance is not None
                ),
                intervention=NumericSummary.from_values(
                    trial.intervention
                    for trial in trials
                    if trial.intervention is not None
                ),
                solve_time_s=NumericSummary.from_values(
                    value
                    for trial in trials
                    for value in trial.solve_times_s
                ),
                case_metrics=metric_summaries,
            )
        )
    return tuple(aggregates)


def _format_float(value: float | None) -> str:
    return "" if value is None else format(value, ".12g")


def _format_metric(value: MetricValue) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _format_float(value)
    return str(value)


def results_to_csv(results: Iterable[BenchmarkResult]) -> str:
    """Serialize sorted per-trial results to deterministic CSV."""

    trials = sorted(tuple(results), key=_result_sort_key)
    metric_names = sorted(
        {name for result in trials for name in result.case_metrics}
    )
    base_fields = [
        "algorithm",
        "case_id",
        "seed",
        "outcome",
        "success",
        "collision",
        "infeasible",
        "timeout",
        "min_clearance",
        "intervention",
        "solve_time_count",
        "solve_time_mean_s",
        "solve_time_p95_s",
        "solve_time_max_s",
        "error",
    ]
    metric_fields = [f"metric.{name}" for name in metric_names]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=base_fields + metric_fields, lineterminator="\n"
    )
    writer.writeheader()
    for result in trials:
        row: dict[str, str | int] = {
            "algorithm": result.algorithm,
            "case_id": result.case_id,
            "seed": result.seed,
            "outcome": result.outcome.value,
            "success": str(result.success).lower(),
            "collision": str(result.collision).lower(),
            "infeasible": str(result.infeasible).lower(),
            "timeout": str(result.timeout).lower(),
            "min_clearance": _format_float(result.min_clearance),
            "intervention": _format_float(result.intervention),
            "solve_time_count": len(result.solve_times_s),
            "solve_time_mean_s": _format_float(result.solve_time_mean_s),
            "solve_time_p95_s": _format_float(result.solve_time_p95_s),
            "solve_time_max_s": _format_float(result.solve_time_max_s),
            "error": "" if result.error is None else result.error,
        }
        for name in metric_names:
            row[f"metric.{name}"] = _format_metric(
                result.case_metrics.get(name)
            )
        writer.writerow(row)
    return stream.getvalue()


def _summary_dict(summary: NumericSummary | None) -> dict[str, int | float] | None:
    return None if summary is None else summary.as_dict()


def _result_dict(result: BenchmarkResult) -> dict[str, Any]:
    return {
        "algorithm": result.algorithm,
        "case_id": result.case_id,
        "seed": result.seed,
        "outcome": result.outcome.value,
        "success": result.success,
        "collision": result.collision,
        "infeasible": result.infeasible,
        "timeout": result.timeout,
        "min_clearance": result.min_clearance,
        "intervention": result.intervention,
        "solve_times_s": list(result.solve_times_s),
        "solve_time_mean_s": result.solve_time_mean_s,
        "solve_time_p95_s": result.solve_time_p95_s,
        "solve_time_max_s": result.solve_time_max_s,
        "case_metrics": dict(result.case_metrics),
        "error": result.error,
    }


def _aggregate_dict(aggregate: BenchmarkAggregate) -> dict[str, Any]:
    return {
        "algorithm": aggregate.algorithm,
        "trial_count": aggregate.trial_count,
        "success_count": aggregate.success_count,
        "success_rate": aggregate.success_rate,
        "collision_count": aggregate.collision_count,
        "collision_rate": aggregate.collision_rate,
        "infeasible_count": aggregate.infeasible_count,
        "infeasible_rate": aggregate.infeasible_rate,
        "timeout_count": aggregate.timeout_count,
        "timeout_rate": aggregate.timeout_rate,
        "error_count": aggregate.error_count,
        "error_rate": aggregate.error_rate,
        "clearance": _summary_dict(aggregate.clearance),
        "intervention": _summary_dict(aggregate.intervention),
        "solve_time_s": _summary_dict(aggregate.solve_time_s),
        "case_metrics": {
            name: summary.as_dict()
            for name, summary in aggregate.case_metrics.items()
        },
    }


def _json_value(value: Any, path: str = "metadata") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not _is_finite(result):
            raise ValueError(f"{path} contains a non-finite number")
        return result
    if isinstance(value, Enum):
        return _json_value(value.value, path)
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item, f"{path}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains unsupported value {type(value).__name__}")


class _DuplicateJSONKey(ValueError):
    """Internal parse error used to reject ambiguous JSON objects."""


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def _strict_json_loads(
    document: str | bytes | bytearray,
    *,
    source: str,
) -> Any:
    try:
        return json.loads(
            document,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError, _DuplicateJSONKey, ValueError) as error:
        raise BenchmarkReportSchemaError(
            f"{source}: invalid JSON: {error}"
        ) from error


def _json_mismatch(actual: Any, expected: Any, path: str) -> str | None:
    """Return the first strict structural/value mismatch, if any."""

    if type(actual) is not type(expected):
        return (
            f"{path} must have type {type(expected).__name__}, "
            f"not {type(actual).__name__}"
        )
    if isinstance(expected, dict):
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            details = []
            if missing:
                details.append(f"missing keys {missing}")
            if unexpected:
                details.append(f"unexpected keys {unexpected}")
            return f"{path} has " + " and ".join(details)
        for key in expected:
            mismatch = _json_mismatch(
                actual[key], expected[key], f"{path}.{key}"
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(actual) != len(expected):
            return (
                f"{path} must contain {len(expected)} items, "
                f"not {len(actual)}"
            )
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            mismatch = _json_mismatch(
                actual_item, expected_item, f"{path}[{index}]"
            )
            if mismatch is not None:
                return mismatch
        return None
    if actual != expected:
        return f"{path} has value {actual!r}; expected {expected!r}"
    return None


def results_to_json(
    results: Iterable[BenchmarkResult],
    *,
    metadata: Mapping[str, Any] | None = None,
    indent: int = 2,
) -> str:
    """Serialize raw and aggregate results as stable, standards-compliant JSON."""

    if indent < 0:
        raise ValueError("indent must be nonnegative")
    trials = sorted(tuple(results), key=_result_sort_key)
    document = {
        "metadata": _json_value({} if metadata is None else metadata),
        "aggregates": [
            _aggregate_dict(item) for item in aggregate_results(trials)
        ],
        "results": [_result_dict(item) for item in trials],
    }
    return (
        json.dumps(
            document,
            indent=indent,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def benchmark_report_from_json(
    document: str | bytes | bytearray,
    *,
    source: str = "<benchmark JSON>",
) -> LoadedBenchmarkReport:
    """Strictly reconstruct a report from :func:`results_to_json` output.

    This validates the complete report, not only the raw rows: duplicate JSON
    keys and non-finite numbers are rejected, every result must match the exact
    writer schema (including derived flags/statistics), and the stored
    aggregates must exactly equal a fresh aggregation of the reconstructed
    rows.  As a result, corrupted or hand-edited reports fail closed.
    """

    parsed = _strict_json_loads(document, source=source)
    if not isinstance(parsed, dict):
        raise BenchmarkReportSchemaError(
            f"{source}: report root must be a JSON object"
        )
    expected_root_keys = {"metadata", "aggregates", "results"}
    parsed_root_keys = set(parsed)
    if parsed_root_keys != expected_root_keys:
        missing = sorted(expected_root_keys - parsed_root_keys)
        unexpected = sorted(parsed_root_keys - expected_root_keys)
        details = []
        if missing:
            details.append(f"missing keys {missing}")
        if unexpected:
            details.append(f"unexpected keys {unexpected}")
        raise BenchmarkReportSchemaError(
            f"{source}: report root has " + " and ".join(details)
        )

    metadata = parsed["metadata"]
    if not isinstance(metadata, dict):
        raise BenchmarkReportSchemaError(
            f"{source}: metadata must be a JSON object"
        )
    try:
        normalized_metadata = _json_value(metadata)
    except (TypeError, ValueError) as error:
        raise BenchmarkReportSchemaError(
            f"{source}: invalid metadata: {error}"
        ) from error
    mismatch = _json_mismatch(metadata, normalized_metadata, "metadata")
    if mismatch is not None:
        raise BenchmarkReportSchemaError(f"{source}: {mismatch}")

    raw_rows = parsed["results"]
    if not isinstance(raw_rows, list):
        raise BenchmarkReportSchemaError(
            f"{source}: results must be a JSON array"
        )
    reconstructed: list[BenchmarkResult] = []
    for index, raw_row in enumerate(raw_rows):
        row_path = f"results[{index}]"
        if not isinstance(raw_row, dict):
            raise BenchmarkReportSchemaError(
                f"{source}: {row_path} must be a JSON object"
            )
        required_fields = {
            "algorithm",
            "case_id",
            "seed",
            "outcome",
            "success",
            "collision",
            "infeasible",
            "timeout",
            "min_clearance",
            "intervention",
            "solve_times_s",
            "solve_time_mean_s",
            "solve_time_p95_s",
            "solve_time_max_s",
            "case_metrics",
            "error",
        }
        row_fields = set(raw_row)
        if row_fields != required_fields:
            missing = sorted(required_fields - row_fields)
            unexpected = sorted(row_fields - required_fields)
            details = []
            if missing:
                details.append(f"missing keys {missing}")
            if unexpected:
                details.append(f"unexpected keys {unexpected}")
            raise BenchmarkReportSchemaError(
                f"{source}: {row_path} has " + " and ".join(details)
            )
        try:
            result = BenchmarkResult(
                algorithm=raw_row["algorithm"],
                case_id=raw_row["case_id"],
                seed=raw_row["seed"],
                outcome=raw_row["outcome"],
                min_clearance=raw_row["min_clearance"],
                intervention=raw_row["intervention"],
                solve_times_s=raw_row["solve_times_s"],
                case_metrics=raw_row["case_metrics"],
                error=raw_row["error"],
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise BenchmarkReportSchemaError(
                f"{source}: invalid {row_path}: {error}"
            ) from error
        mismatch = _json_mismatch(raw_row, _result_dict(result), row_path)
        if mismatch is not None:
            raise BenchmarkReportSchemaError(f"{source}: {mismatch}")
        reconstructed.append(result)

    raw_aggregates = parsed["aggregates"]
    if not isinstance(raw_aggregates, list):
        raise BenchmarkReportSchemaError(
            f"{source}: aggregates must be a JSON array"
        )
    expected_aggregates = [
        _aggregate_dict(item) for item in aggregate_results(reconstructed)
    ]
    mismatch = _json_mismatch(
        raw_aggregates, expected_aggregates, "aggregates"
    )
    if mismatch is not None:
        raise BenchmarkReportSchemaError(f"{source}: {mismatch}")
    return LoadedBenchmarkReport(
        metadata=normalized_metadata,
        results=tuple(reconstructed),
    )


def benchmark_results_from_json(
    document: str | bytes | bytearray,
    *,
    source: str = "<benchmark JSON>",
) -> tuple[BenchmarkResult, ...]:
    """Strictly reconstruct result rows from our benchmark report JSON."""

    return benchmark_report_from_json(document, source=source).results


def load_benchmark_report(path: str | Path) -> LoadedBenchmarkReport:
    """Load and strictly validate a complete benchmark JSON report."""

    source = Path(path)
    return benchmark_report_from_json(
        source.read_bytes(),
        source=str(source),
    )


def load_benchmark_results(
    path: str | Path,
) -> tuple[BenchmarkResult, ...]:
    """Load and strictly validate a benchmark JSON report from disk."""

    return load_benchmark_report(path).results


def _markdown_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _format_rate(count: int, total: int) -> str:
    rate = count / total if total else 0.0
    return f"{count}/{total} ({rate:.1%})"


def _format_summary_pair(
    summary: NumericSummary | None,
    first: str,
    second: str,
    *,
    scale: float = 1.0,
) -> str:
    if summary is None:
        return "—"
    first_value = getattr(summary, first) * scale
    second_value = getattr(summary, second) * scale
    return f"{first_value:.4g} / {second_value:.4g}"


def results_to_markdown(
    results: Iterable[BenchmarkResult],
    *,
    title: str = "Benchmark results",
) -> str:
    """Render a deterministic aggregate Markdown table."""

    trials = tuple(results)
    aggregates = aggregate_results(trials)
    heading = _markdown_escape(str(title))
    if not aggregates:
        return f"# {heading}\n\n_No benchmark results._\n"

    metric_names = sorted(
        {name for aggregate in aggregates for name in aggregate.case_metrics}
    )
    headers = [
        "Algorithm",
        "Trials",
        "Success",
        "Collision",
        "Infeasible",
        "Timeout",
        "Error",
        "Clearance mean / min",
        "Intervention mean / p95",
        "Solve mean / p95 ms",
        "Solve max ms",
    ] + [f"{name} mean" for name in metric_names]
    lines = [
        f"# {heading}",
        "",
        "| " + " | ".join(_markdown_escape(item) for item in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for aggregate in aggregates:
        row = [
            _markdown_escape(aggregate.algorithm),
            str(aggregate.trial_count),
            _format_rate(aggregate.success_count, aggregate.trial_count),
            _format_rate(aggregate.collision_count, aggregate.trial_count),
            _format_rate(aggregate.infeasible_count, aggregate.trial_count),
            _format_rate(aggregate.timeout_count, aggregate.trial_count),
            _format_rate(aggregate.error_count, aggregate.trial_count),
            _format_summary_pair(
                aggregate.clearance, "mean", "minimum"
            ),
            _format_summary_pair(
                aggregate.intervention, "mean", "p95"
            ),
            _format_summary_pair(
                aggregate.solve_time_s, "mean", "p95", scale=1000.0
            ),
            (
                "—"
                if aggregate.solve_time_s is None
                else f"{aggregate.solve_time_s.maximum * 1000.0:.4g}"
            ),
        ]
        for name in metric_names:
            summary = aggregate.case_metrics.get(name)
            row.append("—" if summary is None else f"{summary.mean:.4g}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def _write_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8", newline="\n")
    return destination


def write_benchmark_csv(
    path: str | Path, results: Iterable[BenchmarkResult]
) -> Path:
    """Write deterministic per-trial CSV and return its path."""

    return _write_text(path, results_to_csv(results))


def write_benchmark_json(
    path: str | Path,
    results: Iterable[BenchmarkResult],
    *,
    metadata: Mapping[str, Any] | None = None,
    indent: int = 2,
) -> Path:
    """Write raw and aggregate benchmark JSON and return its path."""

    return _write_text(
        path, results_to_json(results, metadata=metadata, indent=indent)
    )


def write_benchmark_markdown(
    path: str | Path,
    results: Iterable[BenchmarkResult],
    *,
    title: str = "Benchmark results",
) -> Path:
    """Write an aggregate Markdown report and return its path."""

    return _write_text(path, results_to_markdown(results, title=title))


@dataclass(frozen=True)
class BenchmarkReportPaths:
    """Paths produced by :func:`write_benchmark_reports`."""

    csv: Path
    json: Path
    markdown: Path


def write_benchmark_reports(
    output_prefix: str | Path,
    results: Iterable[BenchmarkResult],
    *,
    metadata: Mapping[str, Any] | None = None,
    title: str = "Benchmark results",
) -> BenchmarkReportPaths:
    """Write CSV, JSON, and Markdown reports from one materialized result set."""

    trials = tuple(results)
    prefix = Path(output_prefix)
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    return BenchmarkReportPaths(
        csv=write_benchmark_csv(prefix.with_suffix(".csv"), trials),
        json=write_benchmark_json(
            prefix.with_suffix(".json"), trials, metadata=metadata
        ),
        markdown=write_benchmark_markdown(
            prefix.with_suffix(".md"), trials, title=title
        ),
    )


__all__ = [
    "BenchmarkAggregate",
    "BenchmarkOutcome",
    "BenchmarkReportSchemaError",
    "BenchmarkReportPaths",
    "BenchmarkResult",
    "LoadedBenchmarkReport",
    "MetricValue",
    "NumericSummary",
    "aggregate_results",
    "benchmark_report_from_json",
    "benchmark_results_from_json",
    "load_benchmark_report",
    "load_benchmark_results",
    "merge_benchmark_results",
    "results_to_csv",
    "results_to_json",
    "results_to_markdown",
    "write_benchmark_csv",
    "write_benchmark_json",
    "write_benchmark_markdown",
    "write_benchmark_reports",
]
