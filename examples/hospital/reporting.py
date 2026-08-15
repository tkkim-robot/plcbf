"""Concise publication reporting for the Hospital story benchmark."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping, Sequence

from plcbf.benchmarking import BenchmarkOutcome, BenchmarkResult

from .scenarios import HOSPITAL_STORIES


def hospital_story_id(case_id: str) -> str:
    """Extract the stable story identifier from ``story/seed-N``."""

    marker = "/seed-"
    if marker not in case_id:
        return str(case_id)
    return str(case_id).rsplit(marker, 1)[0]


def validate_paired_world_hashes(
    results: Iterable[BenchmarkResult],
) -> None:
    """Reject method rows that disagree about their paired physical world."""

    grouped: dict[tuple[str, int], set[str]] = defaultdict(set)
    for result in results:
        digest = result.case_metrics.get("world_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(
                "every Hospital result must contain a 64-character "
                "world_sha256"
            )
        grouped[(result.case_id, result.seed)].add(digest)
    mismatches = [identity for identity, hashes in grouped.items() if len(hashes) > 1]
    if mismatches:
        raise ValueError(
            "paired Hospital methods received different worlds for "
            + ", ".join(f"{case_id} seed={seed}" for case_id, seed in mismatches)
        )


def _percent(count: float, total: int) -> str:
    return "—" if total == 0 else f"{100.0 * count / total:.1f}%"


def _mean_metric(
    rows: Sequence[BenchmarkResult],
    name: str,
) -> float | None:
    values = [
        float(value)
        for row in rows
        if (value := row.case_metrics.get(name)) is not None
    ]
    return None if not values else mean(values)


def _trial_rate(
    rows: Sequence[BenchmarkResult],
    name: str,
    *,
    fallback_key: str | None = None,
) -> str:
    values = [
        bool(
            row.case_metrics.get(
                name,
                False
                if fallback_key is None
                else row.case_metrics.get(fallback_key, False),
            )
        )
        for row in rows
    ]
    return _percent(sum(values), len(values))


def _clearance_source_counts(
    rows: Sequence[BenchmarkResult],
    prefix: str,
    *,
    violations_only: bool = False,
) -> str:
    selected = [
        row
        for row in rows
        if not violations_only
        or bool(row.case_metrics.get("operational_safety_violation", False))
    ]
    sources = [
        row.case_metrics.get(f"{prefix}_source_kind") for row in selected
    ]
    known = {
        kind: sum(source == kind for source in sources)
        for kind in ("static", "human", "stretcher")
    }
    unknown = len(sources) - sum(known.values())
    if not sources or (sum(known.values()) == 0 and unknown == len(sources)):
        return "—"
    summary = (
        f"{known['static']} / {known['human']} / {known['stretcher']}"
    )
    return summary if unknown == 0 else f"{summary} / {unknown} unknown"


def _format_float(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _group_summary(rows: Sequence[BenchmarkResult]) -> Mapping[str, str | int]:
    total = len(rows)
    solve_times = [value for row in rows for value in row.solve_times_s]
    safety_clearances = [
        float(value)
        for row in rows
        if (value := row.case_metrics.get("minimum_safety_clearance"))
        is not None
    ]
    interventions = [
        float(row.intervention)
        for row in rows
        if row.intervention is not None
    ]
    selector_fallback_steps = sum(
        int(row.case_metrics.get("selector_fallback_count", 0)) for row in rows
    )
    infeasible_steps = sum(
        int(row.case_metrics.get("infeasible_count", 0)) for row in rows
    )
    backup_executed_steps = sum(
        int(row.case_metrics.get("backup_executed_count", 0)) for row in rows
    )
    executed_steps = sum(int(row.case_metrics.get("steps", 0)) for row in rows)
    return {
        "trials": total,
        "success": _percent(sum(row.success for row in rows), total),
        "collision": _percent(sum(row.collision for row in rows), total),
        "unsafe": _trial_rate(rows, "operational_safety_violation"),
        "timeout": _percent(sum(row.timeout for row in rows), total),
        "deadlock": _trial_rate(rows, "deadlocked_at_end"),
        "clearance": (
            "—"
            if not safety_clearances
            else f"{mean(safety_clearances):.3f} / {min(safety_clearances):.3f}"
        ),
        "intervention": _format_float(
            None if not interventions else mean(interventions)
        ),
        "solve": (
            "—"
            if not solve_times
            else (
                f"{1e3 * mean(solve_times):.2f} / "
                f"{1e3 * (_percentile(solve_times, 0.95) or 0.0):.2f}"
            )
        ),
        "room_during": _trial_rate(rows, "room_occupied_during_blockage"),
        "refuge_entered": _trial_rate(rows, "post_departure_room_entered"),
        "safe_block": _trial_rate(rows, "safe_through_blockage"),
        "goal_after": _trial_rate(rows, "goal_reached_after_clear"),
        "normal_room": _trial_rate(rows, "normal_qp_room_selected"),
        "fallback_room": _trial_rate(
            rows,
            "selector_fallback_room_selected",
            fallback_key="numerical_fallback_room_selected",
        ),
        "selector_fallback": _percent(selector_fallback_steps, executed_steps),
        "infeasible": _percent(infeasible_steps, executed_steps),
        "backup_executed": _percent(backup_executed_steps, executed_steps),
    }


def hospital_benchmark_markdown(
    results: Iterable[BenchmarkResult],
    *,
    title: str = "Hospital fixed-story refuge benchmark",
) -> str:
    """Render pooled quantitative and per-story narrative tables."""

    rows = tuple(results)
    validate_paired_world_hashes(rows)
    by_method: dict[str, list[BenchmarkResult]] = defaultdict(list)
    by_story_method: dict[tuple[str, str], list[BenchmarkResult]] = defaultdict(list)
    for row in rows:
        story = hospital_story_id(row.case_id)
        by_method[row.algorithm].append(row)
        by_story_method[(story, row.algorithm)].append(row)

    output = [
        f"# {title}",
        "",
        (
            "Task outcomes are exclusively goal success, physical collision, "
            "or horizon timeout. Operational safety clearance, solver "
            "infeasibility, deadlock, and room/blockage observations are "
            "reported separately and never terminate or control a trial; "
            "they are not extra success requirements."
        ),
        "",
        "## Pooled results",
        "",
        (
            "| Method | Trials | Success | Collision | Safety violation | "
            "Timeout | Deadlocked at end | Safety clearance mean / min [m] | "
            "Intervention | Decision mean / p95 [ms] | Selector fallback steps | "
            "Infeasible decisions | Backup/emergency steps |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in sorted(by_method):
        summary = _group_summary(by_method[method])
        output.append(
            f"| {method} | {summary['trials']} | {summary['success']} | "
            f"{summary['collision']} | {summary['unsafe']} | "
            f"{summary['timeout']} | {summary['deadlock']} | "
            f"{summary['clearance']} | {summary['intervention']} | "
            f"{summary['solve']} | "
            f"{summary['selector_fallback']} | {summary['infeasible']} | "
            f"{summary['backup_executed']} |"
        )

    output.extend(
        [
            "",
            "## JAX timing integrity",
            "",
            (
                "Warmup is outside every reported decision sample. Runtime "
                "compilation is detected from process-wide executable-cache "
                "misses between the post-warmup snapshot and trial end."
            ),
            "",
            (
                "| Method | JIT-warmed trials | Runtime compilation trials | "
                "Runtime cache-miss delta | Warmup mean [s] |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in sorted(by_method):
        method_rows = by_method[method]
        warmed = [
            row
            for row in method_rows
            if bool(row.case_metrics.get("jit_warmup_enabled", False))
        ]
        runtime_compilations = sum(
            bool(
                row.case_metrics.get(
                    "runtime_jit_compilation_detected", False
                )
            )
            for row in warmed
        )
        miss_delta = sum(
            int(row.case_metrics.get("runtime_jit_cache_miss_delta", 0))
            for row in warmed
        )
        warmup_mean = (
            None
            if not warmed
            else mean(
                float(row.case_metrics.get("jit_warmup_elapsed_s", 0.0))
                for row in warmed
            )
        )
        output.append(
            f"| {method} | {len(warmed)} | {runtime_compilations} | "
            f"{miss_delta} | {_format_float(warmup_mean, 2)} |"
        )

    output.extend(
        [
            "",
            "## Clearance-source diagnostics (pooled)",
            "",
            (
                "Counts below attribute each trial's global swept-clearance "
                "minimum. Source order is static / human / stretcher. The last "
                "column includes only trials with negative operational safety "
                "clearance."
            ),
            "",
            (
                "| Method | Physical minimum source counts | Operational "
                "minimum source counts | Safety-violation source counts |"
            ),
            "|---|---:|---:|---:|",
        ]
    )
    for method in sorted(by_method):
        method_rows = by_method[method]
        physical_sources = _clearance_source_counts(
            method_rows,
            "minimum_physical_clearance",
        )
        operational_sources = _clearance_source_counts(
            method_rows,
            "minimum_safety_clearance",
        )
        violation_sources = _clearance_source_counts(
            method_rows,
            "minimum_safety_clearance",
            violations_only=True,
        )
        output.append(
            f"| {method} | {physical_sources} | {operational_sources} | "
            f"{violation_sources} |"
        )

    output.extend(
        [
            "",
            "## Narrative diagnostics (pooled)",
            "",
            (
                "| Method | Post-departure room entry | Room occupied during "
                "blockage | Safe through blockage | Goal after clearance | "
                "Normal-QP room selection | Selector-backup room selection |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in sorted(by_method):
        summary = _group_summary(by_method[method])
        output.append(
            f"| {method} | {summary['refuge_entered']} | "
            f"{summary['room_during']} | {summary['safe_block']} | "
            f"{summary['goal_after']} | {summary['normal_room']} | "
            f"{summary['fallback_room']} |"
        )

    output.extend(
        [
            "",
            "## Per-story results",
            "",
            (
                "| Story | Method | Trials | Success | Safety violation | "
                "Post-departure room entry | Room during blockage | "
                "Safe through blockage | Decision mean / p95 [ms] |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    story_order = [story.story_id for story in HOSPITAL_STORIES]
    extras = sorted(
        {story for story, _method in by_story_method} - set(story_order)
    )
    for story in [*story_order, *extras]:
        methods = sorted(
            method
            for candidate_story, method in by_story_method
            if candidate_story == story
        )
        for method in methods:
            summary = _group_summary(by_story_method[(story, method)])
            output.append(
                f"| {story} | {method} | {summary['trials']} | "
                f"{summary['success']} | {summary['unsafe']} | "
                f"{summary['refuge_entered']} | {summary['room_during']} | "
                f"{summary['safe_block']} | {summary['solve']} |"
            )

    output.extend(
        [
            "",
            "## Fixed story registry",
            "",
            "| Story | Start room | Goal room | Hall | Blockers | Convoy |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for story in HOSPITAL_STORIES:
        convoy = (
            "opposes eastbound"
            if story.travel_direction > 0
            else "opposes westbound"
        )
        output.append(
            f"| {story.story_id} | {story.start_room} | {story.goal_room} | "
            f"{story.corridor_name} | {story.blocker_count} | {convoy} |"
        )
    output.append("")
    return "\n".join(output)


def write_hospital_benchmark_markdown(
    path: str | Path,
    results: Iterable[BenchmarkResult],
    *,
    title: str = "Hospital fixed-story refuge benchmark",
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        hospital_benchmark_markdown(results, title=title),
        encoding="utf-8",
        newline="\n",
    )
    return destination


__all__ = [
    "hospital_benchmark_markdown",
    "hospital_story_id",
    "validate_paired_world_hashes",
    "write_hospital_benchmark_markdown",
]
