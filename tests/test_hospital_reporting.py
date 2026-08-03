from __future__ import annotations

import json

import pytest

from examples.hospital.reporting import (
    hospital_benchmark_markdown,
    hospital_story_id,
    validate_paired_world_hashes,
)
from plcbf.benchmarking import (
    BenchmarkOutcome,
    BenchmarkResult,
    results_to_json,
)


def _row(method: str, story: str, seed: int, digest: str) -> BenchmarkResult:
    return BenchmarkResult(
        algorithm=method,
        case_id=f"{story}/seed-{seed}",
        seed=seed,
        outcome=BenchmarkOutcome.SUCCESS,
        min_clearance=0.4,
        intervention=0.2,
        solve_times_s=(0.001, 0.003),
        case_metrics={
            "world_sha256": digest,
            "clearance_diagnostic_schema_version": (
                "hospital_clearance_witness_v1"
            ),
            "minimum_physical_clearance_source_kind": "static",
            "minimum_physical_clearance_obstacle_identifier": None,
            "minimum_safety_clearance": 0.25,
            "minimum_safety_clearance_source_kind": (
                "human" if method == "plcbf" else "stretcher"
            ),
            "minimum_safety_clearance_obstacle_identifier": (
                "random-human-007"
                if method == "plcbf"
                else "blocking-stretcher-0"
            ),
            "minimum_safety_clearance_step_index": 2,
            "minimum_safety_clearance_time_s": 0.12,
            "operational_safety_violation": False,
            "deadlocked_at_end": False,
            "termination_reason": "goal_reached",
            "post_departure_room_entered": True,
            "room_occupied_during_blockage": True,
            "safe_through_blockage": True,
            "goal_reached_after_clear": True,
            "normal_qp_room_selected": method == "plcbf",
            "numerical_fallback_room_selected": False,
            "solver_fallback_count": 0,
            "steps": 2,
        },
    )


def test_story_id_uses_merge_compatible_case_identity() -> None:
    assert hospital_story_id("main_eastbound/seed-7") == "main_eastbound"
    assert hospital_story_id("legacy") == "legacy"


def test_paired_world_hash_validation_rejects_method_mismatch() -> None:
    first = _row("plcbf", "main_eastbound", 0, "a" * 64)
    second = _row("pcbf", "main_eastbound", 0, "b" * 64)
    with pytest.raises(ValueError, match="different worlds"):
        validate_paired_world_hashes((first, second))


def test_hospital_markdown_has_pooled_and_story_tables() -> None:
    digest = "a" * 64
    rows = (
        _row("plcbf", "main_eastbound", 0, digest),
        _row("pcbf", "main_eastbound", 0, digest),
    )
    markdown = hospital_benchmark_markdown(rows)

    assert "## Pooled results" in markdown
    assert "## JAX timing integrity" in markdown
    assert "## Clearance-source diagnostics (pooled)" in markdown
    assert "## Narrative diagnostics (pooled)" in markdown
    assert "## Per-story results" in markdown
    assert "## Fixed story registry" in markdown
    assert "Normal-QP room selection" in markdown
    assert "not extra success requirements" in markdown
    assert "Source order is static / human / stretcher" in markdown
    assert "Runtime cache-miss delta" in markdown
    assert "Task outcomes are exclusively goal success" in markdown
    assert "Deadlocked at end" in markdown
    assert "| plcbf | 1 / 0 / 0 | 0 / 1 / 0 | — |" in markdown
    assert "| main_eastbound | plcbf | 1 | 100.0%" in markdown


def test_clearance_attribution_survives_raw_json_serialization() -> None:
    row = _row("plcbf", "main_eastbound", 0, "a" * 64)
    payload = json.loads(results_to_json((row,)))
    metrics = payload["results"][0]["case_metrics"]

    assert (
        metrics["clearance_diagnostic_schema_version"]
        == "hospital_clearance_witness_v1"
    )
    assert metrics["minimum_physical_clearance_source_kind"] == "static"
    assert metrics["minimum_physical_clearance_obstacle_identifier"] is None
    assert metrics["minimum_safety_clearance_source_kind"] == "human"
    assert (
        metrics["minimum_safety_clearance_obstacle_identifier"]
        == "random-human-007"
    )
    assert metrics["minimum_safety_clearance_step_index"] == 2
    assert metrics["minimum_safety_clearance_time_s"] == 0.12
