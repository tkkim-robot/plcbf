"""Artifact-key and audit-label regressions for additional baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.drift_car import benchmark_additional_baselines as drift
from examples.warehouse import benchmark_additional_baselines_quad as warehouse


EXPECTED_KEYS = {"multi_backup_cbf_mi", "library_pcbf_mi"}
ROOT = Path(__file__).resolve().parents[1]


def _drift_result(variant, scenario):
    return drift.EpisodeResult(
        algorithm=variant.key,
        seed=scenario.seed,
        run_idx=scenario.run_idx,
        obstacle_geometry=list(scenario.obstacles),
        P_or_library_size=4,
        collision=False,
        infeasible=False,
        unrecoverable_infeasible=False,
        historical_failure=False,
        certificate_lost=True,
        qp_infeasible=False,
        runtime_error=False,
        reached_goal=False,
        survived_horizon=True,
        task_completed=False,
        completed_or_survived=True,
        filter_failure=True,
        union_failure=True,
        total_steps=1,
        timed_steps=1,
        mean_compute_ms=1.0,
        median_compute_ms=1.0,
        p95_compute_ms=1.0,
        max_compute_ms=1.0,
        nominal_tracking_fraction=0.0,
        mean_intervention_l2=1.0,
        max_intervention_l2=1.0,
        policy_switch_count=0,
    )


def test_drift_writer_emits_only_two_baseline_keys(tmp_path, monkeypatch):
    scenario = drift.Scenario(0, 7, 1, ((80.0, "middle"),))
    monkeypatch.setattr(drift, "generate_scenarios", lambda *args: [scenario])
    monkeypatch.setattr(
        drift,
        "run_episode",
        lambda variant, item, config, verbose=False: _drift_result(variant, item),
    )
    paths = {
        suffix: tmp_path / f"drift.{suffix}" for suffix in ("md", "json", "csv")
    }
    drift.main(
        [
            "--num-runs",
            "1",
            "--output-md",
            str(paths["md"]),
            "--output-json",
            str(paths["json"]),
            "--output-csv",
            str(paths["csv"]),
        ]
    )

    payload = json.loads(paths["json"].read_text())
    assert set(payload["algorithms"]) == EXPECTED_KEYS
    assert set(payload["trials"]) == EXPECTED_KEYS
    assert {row["key"] for row in payload["summary"]} == EXPECTED_KEYS
    with paths["csv"].open(newline="") as stream:
        assert {row["algorithm"] for row in csv.DictReader(stream)} == EXPECTED_KEYS


def _warehouse_result(algo, scenario):
    return warehouse.TrialResult(
        collision=False,
        infeasible=False,
        unrecoverable_infeasible=False,
        historical_failure=False,
        reached_goal=False,
        nominal_tracking_pct=0.0,
        solve_time_sum_sec=0.001,
        timed_steps=1,
        total_steps=1,
        algorithm=algo,
        seed=scenario.seed,
        run_idx=scenario.run_idx,
        obstacle_geometry=list(scenario.ghosts),
        p_or_library_size=66,
        certificate_lost=True,
        survived_horizon=True,
        completed_or_survived=True,
        filter_failure=True,
        union_failure=True,
    )


def test_warehouse_writer_emits_only_two_baseline_keys(tmp_path, monkeypatch):
    scenario = warehouse.TrialScenario(0, 11, tuple())
    monkeypatch.setattr(
        warehouse, "generate_random_scenarios", lambda **kwargs: [scenario]
    )
    monkeypatch.setattr(
        warehouse,
        "run_trial",
        lambda algo, scenario, **kwargs: _warehouse_result(algo, scenario),
    )
    paths = {
        suffix: tmp_path / f"warehouse.{suffix}"
        for suffix in ("md", "json", "csv")
    }
    warehouse.main(
        [
            "--num-trials",
            "1",
            "--output-md",
            str(paths["md"]),
            "--output-json",
            str(paths["json"]),
            "--output-csv",
            str(paths["csv"]),
        ]
    )

    payload = json.loads(paths["json"].read_text())
    assert set(payload["algorithms"]) == EXPECTED_KEYS
    assert set(payload["trials"]) == EXPECTED_KEYS
    assert {row["key"] for row in payload["summaries"]} == EXPECTED_KEYS
    with paths["csv"].open(newline="") as stream:
        assert {row["algorithm"] for row in csv.DictReader(stream)} == EXPECTED_KEYS


def test_superseded_directory_is_labeled_common_stop_audit():
    directory = (
        ROOT / "output/additional_baselines/reviewer_fix_2026-07-13"
    )
    manifest = json.loads((directory / "reproduction_manifest.json").read_text())
    readme = (directory / "README.md").read_text().lower()
    report = (directory / "corrected_results_report.md").read_text().lower()
    assert manifest["artifact_role"] == "common_stop_audit"
    assert manifest["not_for_publication"] is True
    assert "common-stop audit" in readme
    assert "not for publication" in readme
    assert "do not cite" in report
