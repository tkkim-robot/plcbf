"""Artifact-key and audit-label regressions for additional baselines."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from examples.drift_car import benchmark_additional_baselines as drift
from examples.warehouse import benchmark_additional_baselines_quad as warehouse


EXPECTED_KEYS = {"multi_backup_cbf_mi", "library_pcbf_mi"}
ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIRECTORY = (
    ROOT
    / "output/additional_baselines/native_control_baseline_only_2026-07-13"
)


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


def test_publication_manifest_and_artifacts_are_baseline_only():
    manifest = json.loads(
        (PUBLICATION_DIRECTORY / "reproduction_manifest.json").read_text()
    )
    assert manifest["artifact_role"] == "publication_baseline_only"
    assert manifest["not_for_publication"] is False
    assert set(manifest["algorithm_keys"]) == EXPECTED_KEYS
    assert manifest["scope"]["historical_controller_executed"] is False
    assert manifest["source_checkout"]["parent_status_porcelain"] == ""
    safe_control = manifest["source_checkout"]["submodules"]["safe_control"]
    assert safe_control["gitlink"] == safe_control["head"]
    assert safe_control["status_porcelain"] == ""

    for name, benchmark in manifest["benchmarks"].items():
        assert "plcbf" not in json.dumps(benchmark["command"]).lower(), name
        if benchmark["publication_timing"]:
            assert benchmark["workers"] == 1

    json_specs = {
        "drift_seed7_50.json": ("summary", 50),
        "drift_timing_serial3.json": ("summary", 3),
        "warehouse_seed11_100.json": ("summaries", 100),
        "warehouse_timing_serial3.json": ("summaries", 3),
    }
    for filename, (summary_key, trials_per_method) in json_specs.items():
        payload = json.loads((PUBLICATION_DIRECTORY / filename).read_text())
        assert set(payload["algorithms"]) == EXPECTED_KEYS
        assert set(payload["trials"]) == EXPECTED_KEYS
        assert {row["key"] for row in payload[summary_key]} == EXPECTED_KEYS
        assert all(
            len(payload["trials"][key]) == trials_per_method
            for key in EXPECTED_KEYS
        )

    csv_specs = {
        "drift_seed7_50.csv": 50,
        "drift_timing_serial3.csv": 3,
        "warehouse_seed11_100.csv": 100,
        "warehouse_timing_serial3.csv": 3,
    }
    for filename, trials_per_method in csv_specs.items():
        with (PUBLICATION_DIRECTORY / filename).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 2 * trials_per_method
        assert {row["algorithm"] for row in rows} == EXPECTED_KEYS

    for filename, expected_digest in manifest["artifacts_sha256"].items():
        digest = hashlib.sha256(
            (PUBLICATION_DIRECTORY / filename).read_bytes()
        ).hexdigest()
        assert digest == expected_digest


def test_publication_report_preserves_submitted_plcbf_rows_verbatim():
    report = (PUBLICATION_DIRECTORY / "publication_report.md").read_text()
    assert (
        "| PL-CBF (ours) — unchanged submitted value | $\\Pi$ | "
        "0/50 (0.0%) | 10.00 | 7.522 |"
    ) in report
    assert (
        "| PL-CBF (ours) — unchanged submitted value | 64 | "
        "0/100 (0.0%) | 27.986 |"
    ) in report
