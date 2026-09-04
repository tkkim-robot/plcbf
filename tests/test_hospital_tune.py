from __future__ import annotations

from dataclasses import asdict, replace
import json
import time

import pytest

from examples.hospital import tune
from examples.hospital.benchmark import (
    HOSPITAL_BENCHMARK_STORIES,
    PUBLICATION_MAX_SENSED_OBSTACLES,
)
from examples.hospital.config import DEFAULT_CONFIG, load_hospital_config
from examples.hospital.scenarios import hospital_story_protocol_metadata
from plcbf.benchmarking import BenchmarkOutcome, BenchmarkResult
from plcbf.benchmarking import load_benchmark_report


class FakeTrial:
    def __init__(self) -> None:
        self.number = 0
        self.params: dict[str, object] = {}
        self.float_ranges: dict[str, tuple[float, float, object, bool]] = {}

    def suggest_int(self, name, low, high, step=1):
        value = low + ((high - low) // (2 * step)) * step
        self.params[name] = value
        return value

    def suggest_float(self, name, low, high, step=None, log=False):
        self.float_ranges[name] = (low, high, step, log)
        value = (
            0.5 * (low + high)
            if step is None
            else low + (int(round((high - low) / step)) // 2) * step
        )
        self.params[name] = value
        return value


class ReportingTrial(FakeTrial):
    def __init__(self, *, prune_after: int | None = None) -> None:
        super().__init__()
        self.prune_after = prune_after
        self.reports: list[tuple[int, float]] = []
        self.user_attrs: dict[str, object] = {}

    def report(self, value, step) -> None:
        self.reports.append((int(step), float(value)))

    def should_prune(self) -> bool:
        return (
            self.prune_after is not None
            and len(self.reports) >= self.prune_after
        )

    def set_user_attr(self, name, value) -> None:
        self.user_attrs[name] = value


def _result(
    outcome: BenchmarkOutcome,
    *,
    progress: float,
) -> BenchmarkResult:
    return BenchmarkResult(
        "plcbf",
        "blocked_2_stretchers",
        0,
        outcome,
        min_clearance=0.2,
        intervention=0.1,
        case_metrics={
            "progress": progress,
            "solver_fallback_count": 0,
            "steps": 10,
            "oracle_and_solver_time_total_s": 0.1,
        },
    )


def _story_result(
    story: str,
    seed: int,
    *,
    runtime_s: float = 0.1,
) -> BenchmarkResult:
    return BenchmarkResult(
        "plcbf",
        f"{story}/seed-{seed}",
        seed,
        BenchmarkOutcome.SUCCESS,
        min_clearance=0.2,
        intervention=0.1,
        solve_times_s=(0.001, 0.002),
        case_metrics={
            "progress": 1.0,
            "solver_fallback_count": 0,
            "steps": 10,
            "oracle_and_solver_time_total_s": runtime_s,
            "world_sha256": "a" * 64,
        },
    )


def test_suggested_qp_refuge_config_can_be_reconstructed() -> None:
    trial = FakeTrial()
    suggested = tune.suggest_hospital_config(trial)
    reconstructed = tune.hospital_config_from_params(trial.params)
    assert reconstructed == suggested
    assert "guard_required_steps" not in trial.params
    assert "minimum_hold_time_s" not in trial.params
    assert "certificate_update_period" not in trial.params
    assert set(trial.params) == {
        "room_target_speed",
        "stop_gain",
        "cbf_alpha",
        "cbf_value_buffer",
        "component_temperature",
        "time_temperature",
        "max_gradient_norm",
        "hocbf_lambda1",
        "hocbf_lambda2",
    }
    assert suggested.refuge.terminal_speed_max > 0.0
    assert suggested.safety.max_obstacles == PUBLICATION_MAX_SENSED_OBSTACLES
    assert suggested.robot.sensing_range == 24.0
    assert suggested.policies.num_angle_policies == 12
    assert suggested.policies.room_policy_count == 7
    assert suggested.policies.room_rollout_dt == pytest.approx(0.24)
    assert {
        suggested.policies.nominal_horizon,
        suggested.policies.angle_horizon,
        suggested.policies.reverse_horizon,
        suggested.policies.stop_horizon,
        suggested.policies.room_horizon,
    } == {7.2}
    assert suggested.safety.safety_margin == pytest.approx(0.45)
    assert suggested.safety.stretcher_margin == pytest.approx(0.55)
    assert suggested.policies.cbf_value_buffer == trial.params[
        "cbf_value_buffer"
    ]
    low, high, step, log = trial.float_ranges["cbf_value_buffer"]
    assert low <= 0.9 <= high
    assert step == pytest.approx(0.05)
    assert log is False
    for name, specification in tune.hospital_tuning_search_space().items():
        observed_low, observed_high, observed_step, observed_log = (
            trial.float_ranges[name]
        )
        assert observed_low == specification["low"]
        assert observed_high == specification["high"]
        assert observed_step == specification.get("step")
        assert observed_log is bool(specification.get("log", False))
    with pytest.raises(ValueError, match="unknown tuned"):
        tune.hospital_config_from_params({"guard_required_steps": 8})


def test_exported_best_config_json_replays_exactly_and_is_hashable(
    tmp_path,
) -> None:
    trial = FakeTrial()
    suggested = tune.suggest_hospital_config(trial)
    path = tmp_path / "hospital_summary.json"
    path.write_text(
        json.dumps({"best_config": asdict(suggested)}),
        encoding="utf-8",
    )
    replayed = load_hospital_config(path)
    assert replayed == suggested
    assert isinstance(replayed.policies.gradient_steps, tuple)
    assert hash(replayed) == hash(suggested)


def test_tuning_score_uses_safety_and_task_metrics_not_fixed_dwell() -> None:
    success = _result(BenchmarkOutcome.SUCCESS, progress=1.0)
    timeout = _result(BenchmarkOutcome.TIMEOUT, progress=0.5)
    collision = _result(BenchmarkOutcome.COLLISION, progress=0.2)
    assert tune.score_results((success,)) < tune.score_results((timeout,))
    assert tune.score_results((timeout,)) < tune.score_results((collision,))


def test_tuning_score_maximizes_success_before_failure_type() -> None:
    success = _result(BenchmarkOutcome.SUCCESS, progress=1.0)
    timeout = _result(BenchmarkOutcome.TIMEOUT, progress=0.5)
    collision = _result(BenchmarkOutcome.COLLISION, progress=0.2)
    higher_success_with_collision = (success, success, collision)
    lower_success_without_collision = (success, timeout, timeout)
    assert tune.score_results(higher_success_with_collision) < (
        tune.score_results(lower_success_without_collision)
    )


def test_full_benchmark_grid_and_fingerprint_are_stable() -> None:
    first = tune.HospitalTuningConfig()
    second = replace(first)
    assert first.cases == tuple(HOSPITAL_BENCHMARK_STORIES)
    assert first.benchmark_seeds == tuple(range(20))
    assert first.base_config.safety.max_obstacles == (
        PUBLICATION_MAX_SENSED_OBSTACLES
    )
    assert tune.study_configuration_fingerprint(first) == (
        tune.study_configuration_fingerprint(second)
    )
    changed_buffer = replace(
        first,
        base_config=replace(
            first.base_config,
            policies=replace(
                first.base_config.policies,
                cbf_value_buffer=1.1,
            ),
        ),
    )
    assert tune.study_configuration_fingerprint(first) != (
        tune.study_configuration_fingerprint(changed_buffer)
    )
    metadata = first.metadata()
    assert metadata["external_refuge_state_machine"] is False
    assert metadata["oracle_period"] == "every_plant_step"
    assert metadata["oracle_period_s"] == first.base_config.dt
    assert metadata["protocol_kind"] == (
        "canonical_full_100_world_optimization_and_reporting_grid"
    )
    assert metadata["hospital_story_protocol"]["protocol_sha256"] == (
        hospital_story_protocol_metadata()["protocol_sha256"]
    )
    assert set(metadata["scenario_grid_sha256"]) == {"benchmark"}
    assert metadata["benchmark_world_order"] == (
        "seed_major_story_round_robin"
    )
    assert metadata["expected_benchmark_world_count"] == 100
    assert metadata["held_out_validation"] is False
    assert metadata["post_selection_simulation"] is False
    assert metadata["selection_grid_is_reported_grid"] is True
    assert set(metadata["tunable_parameters"]) == set(
        tune.base_controller_reference_params(first)
    )
    assert metadata["fixed_tuning_envelope"] == {
        "sensing_range_m": 24.0,
        "max_sensed_obstacles": PUBLICATION_MAX_SENSED_OBSTACLES,
        "safety_margin_m": 0.45,
        "human_margin_m": 0.0,
        "stretcher_margin_m": 0.55,
        "static_margin_m": 0.14,
        "num_angle_policies": 12,
        "room_policy_count": 7,
        "room_rollout_dt_s": 0.24,
        "common_policy_horizon_s": 7.2,
        "room_horizon_s": 7.2,
        "refuge_geometry": asdict(first.base_config.refuge),
    }


@pytest.mark.parametrize("seed", (-1, 20, 101, 1.5, True))
def test_tuning_rejects_seeds_outside_fixed_story_protocol(seed) -> None:
    with pytest.raises(ValueError, match="traffic seeds"):
        tune.HospitalTuningConfig(
            benchmark_seeds=(seed,),
        )


def test_cli_defaults_resolve_exact_five_by_twenty_grid() -> None:
    arguments = tune.build_parser().parse_args([])
    config = tune._config_from_args(arguments)
    assert config.cases == tuple(HOSPITAL_BENCHMARK_STORIES)
    assert config.benchmark_seeds == tuple(range(20))
    assert config.study_name == "hospital_plcbf_100"
    assert config.storage == "sqlite:///results/hospital_optuna_100.db"

    quick = tune._config_from_args(tune.build_parser().parse_args(["--quick"]))
    assert quick.cases == (HOSPITAL_BENCHMARK_STORIES[0],)
    assert quick.benchmark_seeds == (0,)
    assert quick.base_config.safety.max_obstacles == (
        PUBLICATION_MAX_SENSED_OBSTACLES
    )
    assert quick.base_config.robot.sensing_range == 24.0
    assert quick.base_config.policies.num_angle_policies == 12
    assert quick.base_config.policies.room_policy_count == 7


def test_cli_dry_run_cannot_create_or_execute_a_study(
    monkeypatch,
    capsys,
) -> None:
    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("dry run must not touch Optuna storage")

    monkeypatch.setattr(tune, "create_study", forbidden)
    monkeypatch.setattr(tune, "run_study", forbidden)
    assert tune.main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ready"
    assert "--run" in payload["message"]


def test_evaluation_applies_publication_sensor_capacity(monkeypatch) -> None:
    captured = {}

    def fake_benchmark(**kwargs):
        captured.update(kwargs)
        return (_result(BenchmarkOutcome.SUCCESS, progress=1.0),)

    monkeypatch.setattr(tune, "run_hospital_benchmark", fake_benchmark)
    tune.evaluate_plcbf_config(
        DEFAULT_CONFIG,
        cases=(HOSPITAL_BENCHMARK_STORIES[0],),
        seeds=(0,),
        steps=1,
    )
    assert captured["config"].safety.max_obstacles == (
        PUBLICATION_MAX_SENSED_OBSTACLES
    )
    assert captured["config"].robot.sensing_range == 24.0
    assert captured["oracle_period_s"] == captured["config"].dt

    custom = tune.HospitalTuningConfig(base_config=DEFAULT_CONFIG)
    assert custom.base_config.safety.max_obstacles == (
        PUBLICATION_MAX_SENSED_OBSTACLES
    )
    assert custom.base_config.robot.sensing_range == 24.0

    weakened = replace(
        DEFAULT_CONFIG,
        robot=replace(DEFAULT_CONFIG.robot, sensing_range=99.0),
        policies=replace(
            DEFAULT_CONFIG.policies,
            num_angle_policies=4,
            room_policy_count=1,
            room_rollout_dt=0.48,
        ),
        safety=replace(
            DEFAULT_CONFIG.safety,
            safety_margin=0.1,
            stretcher_margin=0.1,
            max_obstacles=999,
        ),
    )
    normalized = tune.HospitalTuningConfig(base_config=weakened).base_config
    assert normalized.robot.sensing_range == 24.0
    assert normalized.safety.max_obstacles == (
        PUBLICATION_MAX_SENSED_OBSTACLES
    )
    assert normalized.safety.safety_margin == pytest.approx(0.45)
    assert normalized.safety.stretcher_margin == pytest.approx(0.55)
    assert normalized.policies.num_angle_policies == 12
    assert normalized.policies.room_policy_count == 7
    assert normalized.policies.room_rollout_dt == pytest.approx(0.24)


def test_study_binding_records_protocol_and_scenario_fingerprints() -> None:
    class FakeStudy:
        def __init__(self) -> None:
            self.user_attrs = {}
            self.trials = []

        def set_user_attr(self, key, value) -> None:
            self.user_attrs[key] = value

    config = tune.HospitalTuningConfig()
    study = FakeStudy()
    tune.bind_study_configuration(study, config)
    assert study.user_attrs["hospital_story_protocol_sha256"] == (
        hospital_story_protocol_metadata()["protocol_sha256"]
    )
    assert study.user_attrs["scenario_grid_sha256"] == (
        config.metadata()["scenario_grid_sha256"]
    )
    assert study.user_attrs["pruner"] == config.metadata()["pruner"]
    assert study.user_attrs["sampler"] == config.metadata()["sampler"]
    assert study.user_attrs["source_content_sha256"]
    ordered = study.user_attrs["ordered_benchmark_case_ids"]
    assert len(ordered) == 100
    assert ordered[:5] == [
        f"{story}/seed-0" for story in HOSPITAL_BENCHMARK_STORIES
    ]
    assert study.user_attrs["benchmark_world_order"] == (
        "seed_major_story_round_robin"
    )
    assert set(study.user_attrs["search_space"]) == set(
        tune.base_controller_reference_params(config)
    )


def test_study_binding_rejects_incompatible_and_legacy_resume() -> None:
    class FakeStudy:
        def __init__(self) -> None:
            self.user_attrs = {}
            self.trials = []

        def set_user_attr(self, key, value) -> None:
            self.user_attrs[key] = value

    config = tune.HospitalTuningConfig()
    incompatible = FakeStudy()
    tune.bind_study_configuration(incompatible, config)
    incompatible.user_attrs["benchmark_world_order"] = "case_major"
    with pytest.raises(ValueError, match="benchmark_world_order"):
        tune.bind_study_configuration(incompatible, config)

    legacy = FakeStudy()
    legacy.trials.append(object())
    with pytest.raises(ValueError, match="legacy Optuna study"):
        tune.bind_study_configuration(legacy, config)


def test_objective_reports_ordered_complete_five_by_twenty_grid(
    monkeypatch,
    tmp_path,
) -> None:
    observed: list[tuple[str, int]] = []
    rows: list[BenchmarkResult] = []

    def fake_benchmark(**kwargs):
        story = kwargs["cases"][0]
        seed = kwargs["seeds"][0]
        observed.append((story, seed))
        result = _story_result(story, seed)
        rows.append(result)
        return (result,)

    monkeypatch.setattr(tune, "run_hospital_benchmark", fake_benchmark)
    trial = ReportingTrial()
    value = tune.objective(trial, result_archive_dir=tmp_path)

    expected = [
        (story, seed)
        for seed in range(20)
        for story in HOSPITAL_BENCHMARK_STORIES
    ]
    assert observed == expected
    assert [step for step, _value in trial.reports] == list(
        range(5, 101, 5)
    )
    assert value == tune.score_results(rows)
    assert trial.user_attrs["evaluated_world_count"] == 100
    assert trial.user_attrs["completed_world_count"] == 100
    assert trial.user_attrs["expected_complete_world_count"] == 100
    assert trial.user_attrs["completed_full_benchmark_grid"] is True
    descriptor = trial.user_attrs["exact_benchmark_result_archive"]
    assert descriptor["result_count"] == 100
    assert (tmp_path / "trial-0000.json").is_file()
    assert trial.user_attrs["mean_oracle_and_solver_time_total_s"] == (
        pytest.approx(0.1)
    )


def test_exact_result_archive_round_trips_and_rejects_tampering(
    monkeypatch,
    tmp_path,
) -> None:
    story = HOSPITAL_BENCHMARK_STORIES[0]

    def fake_benchmark(**kwargs):
        return (_story_result(kwargs["cases"][0], kwargs["seeds"][0]),)

    monkeypatch.setattr(tune, "run_hospital_benchmark", fake_benchmark)
    trial = ReportingTrial()
    score = tune.objective(
        trial,
        cases=(story,),
        seeds=(0, 1),
        steps=1,
        result_archive_dir=tmp_path,
    )
    trial.value = score
    case_ids = [f"{story}/seed-0", f"{story}/seed-1"]
    config = tune.hospital_config_from_params(trial.params)
    rows = tune._load_exact_result_archive(
        trial,
        expected_case_ids=case_ids,
        expected_grid_sha256=tune.scenario_grid_fingerprint(
            (story,), (0, 1)
        ),
        expected_config=config,
    )
    assert [row.case_id for row in rows] == case_ids
    assert rows[0].solve_times_s == (0.001, 0.002)
    archive_path = tmp_path / "trial-0000.json"
    archive_path.write_bytes(archive_path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="byte count"):
        tune._load_exact_result_archive(
            trial,
            expected_case_ids=case_ids,
            expected_grid_sha256=tune.scenario_grid_fingerprint(
                (story,), (0, 1)
            ),
            expected_config=config,
        )


def test_objective_prunes_only_after_complete_world_checkpoint(
    monkeypatch,
    tmp_path,
) -> None:
    optuna = pytest.importorskip("optuna")
    observed: list[tuple[str, int]] = []

    def fake_benchmark(**kwargs):
        story = kwargs["cases"][0]
        seed = kwargs["seeds"][0]
        observed.append((story, seed))
        return (_story_result(story, seed),)

    monkeypatch.setattr(tune, "run_hospital_benchmark", fake_benchmark)
    trial = ReportingTrial(prune_after=1)
    stories = tuple(HOSPITAL_BENCHMARK_STORIES[:2])

    with pytest.raises(optuna.TrialPruned):
        tune.objective(
            trial,
            cases=stories,
            seeds=(0, 1),
            steps=1,
            result_archive_dir=tmp_path,
        )

    assert observed == [(stories[0], 0), (stories[1], 0)]
    assert [step for step, _value in trial.reports] == [2]
    assert trial.user_attrs["evaluated_world_count"] == 2
    assert trial.user_attrs["expected_complete_world_count"] == 4
    assert "completed_full_benchmark_grid" not in trial.user_attrs


def test_final_completed_world_is_not_reclassified_as_pruned(
    monkeypatch,
    tmp_path,
) -> None:
    def fake_benchmark(**kwargs):
        return (
            _story_result(kwargs["cases"][0], kwargs["seeds"][0]),
        )

    monkeypatch.setattr(tune, "run_hospital_benchmark", fake_benchmark)
    trial = ReportingTrial(prune_after=2)
    tune.objective(
        trial,
        cases=(HOSPITAL_BENCHMARK_STORIES[0],),
        seeds=(0, 1),
        steps=1,
        result_archive_dir=tmp_path,
    )
    assert trial.user_attrs["completed_full_benchmark_grid"] is True


def test_prefix_metric_neutralizes_runtime_measurement() -> None:
    story = HOSPITAL_BENCHMARK_STORIES[0]
    fast = _story_result(story, 0, runtime_s=0.001)
    slow = _story_result(story, 0, runtime_s=100.0)
    assert tune._deterministic_prefix_score((fast,)) == (
        tune._deterministic_prefix_score((slow,))
    )
    assert tune.score_results((fast,)) == tune.score_results((slow,))


def test_create_study_wires_deterministic_exact_prefix_pruner() -> None:
    optuna = pytest.importorskip("optuna")
    study = tune.create_study(storage=None, load_if_exists=False)
    assert isinstance(
        study.pruner, tune.ExactWorldPrefixPatientMedianPruner
    )
    assert study.pruner.n_startup_trials == 5
    assert study.pruner.n_warmup_steps == 20
    assert study.pruner.interval_steps == 5
    assert study.pruner.n_min_trials == 5
    assert study.pruner.patience == 3

    quick = tune.create_study(
        storage=None,
        load_if_exists=False,
        quick=True,
    )
    assert isinstance(quick.pruner, optuna.pruners.NopPruner)


def test_default_pruner_cannot_prune_before_thirty_balanced_worlds() -> None:
    optuna = pytest.importorskip("optuna")
    pruner = tune.build_pruner()
    study = optuna.create_study(direction="minimize", pruner=pruner)
    expected_case_ids = [f"story-{index}" for index in range(100)]
    study.set_user_attr("ordered_benchmark_case_ids", expected_case_ids)
    study.set_user_attr(
        "scenario_grid_sha256", {"benchmark": "fixed-grid"}
    )

    def set_attrs(trial, evaluated, *, complete=False):
        trial.set_user_attr("benchmark_grid_sha256", "fixed-grid")
        trial.set_user_attr("expected_case_ids", expected_case_ids)
        trial.set_user_attr(
            "evaluated_case_ids", expected_case_ids[:evaluated]
        )
        trial.set_user_attr("evaluated_world_count", evaluated)
        trial.set_user_attr("expected_complete_world_count", 100)
        if complete:
            trial.set_user_attr("completed_full_benchmark_grid", True)
            trial.set_user_attr("completed_world_count", 100)

    for _ in range(5):
        reference = study.ask()
        set_attrs(reference, 100, complete=True)
        for checkpoint in (20, 25, 30):
            reference.report(1.0, checkpoint)
        study.tell(reference, 1.0)

    candidate = study.ask()
    for checkpoint in (20, 25):
        set_attrs(candidate, checkpoint)
        candidate.report(2.0, checkpoint)
        assert candidate.should_prune() is False
    set_attrs(candidate, 30)
    candidate.report(2.0, 30)
    assert candidate.should_prune() is True


def test_base_controller_is_enqueued_once_as_trial_zero() -> None:
    pytest.importorskip("optuna")
    config = tune.HospitalTuningConfig(
        cases=(HOSPITAL_BENCHMARK_STORIES[0],),
        benchmark_seeds=(0,),
        storage=None,
    )
    study = tune.create_study(
        study_name="base_reference",
        storage=None,
        load_if_exists=False,
    )
    tune.bind_study_configuration(study, config)
    assert tune.enqueue_base_controller_reference(study, config) is True
    assert tune.enqueue_base_controller_reference(study, config) is False
    assert len(study.trials) == 1
    assert tune._trial_state_name(study.trials[0]) == "waiting"

    study.optimize(
        lambda trial: (
            tune.suggest_hospital_config(trial, config.base_config),
            0.0,
        )[1],
        n_trials=1,
    )
    assert study.trials[0].number == 0
    assert study.trials[0].params == (
        tune.base_controller_reference_params(config)
    )
    assert study.trials[0].user_attrs[
        "hospital_base_controller_reference"
    ] is True


def test_real_optuna_pruner_uses_exact_prefix_and_patience() -> None:
    optuna = pytest.importorskip("optuna")
    pruner = tune.ExactWorldPrefixPatientMedianPruner(
        n_startup_trials=2,
        n_warmup_steps=1,
        interval_steps=1,
        n_min_trials=2,
        patience=2,
    )
    study = optuna.create_study(direction="minimize", pruner=pruner)
    expected_case_ids = ["story/seed-0", "story/seed-1"]
    study.set_user_attr("ordered_benchmark_case_ids", expected_case_ids)
    study.set_user_attr(
        "scenario_grid_sha256", {"benchmark": "fixed-grid"}
    )

    def set_attrs(trial, *, complete: bool, evaluated: int) -> None:
        values = {
            "benchmark_grid_sha256": "fixed-grid",
            "expected_case_ids": expected_case_ids,
            "evaluated_case_ids": expected_case_ids[:evaluated],
            "evaluated_world_count": evaluated,
            "expected_complete_world_count": 2,
        }
        if complete:
            values.update(
                {
                    "completed_full_benchmark_grid": True,
                    "completed_world_count": 2,
                }
            )
        for name, value in values.items():
            trial.set_user_attr(name, value)

    for _ in range(2):
        reference = study.ask()
        set_attrs(reference, complete=True, evaluated=2)
        reference.report(1.0, step=1)
        reference.report(1.0, step=2)
        study.tell(reference, 1.0)

    candidate = study.ask()
    set_attrs(candidate, complete=False, evaluated=1)
    candidate.report(2.0, step=1)
    assert candidate.should_prune() is False
    set_attrs(candidate, complete=False, evaluated=2)
    candidate.report(2.0, step=2)
    assert candidate.should_prune() is True


def test_pruner_ignores_complete_trials_without_full_grid_audit() -> None:
    optuna = pytest.importorskip("optuna")
    pruner = tune.ExactWorldPrefixPatientMedianPruner(
        n_startup_trials=1,
        n_warmup_steps=1,
        interval_steps=1,
        n_min_trials=1,
        patience=1,
    )
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.set_user_attr("ordered_benchmark_case_ids", ["story/seed-0"])
    study.set_user_attr(
        "scenario_grid_sha256", {"benchmark": "fixed-grid"}
    )
    unaudited = study.ask()
    unaudited.report(1.0, step=1)
    study.tell(unaudited, 1.0)

    candidate = study.ask()
    candidate.set_user_attr("benchmark_grid_sha256", "fixed-grid")
    candidate.set_user_attr("expected_case_ids", ["story/seed-0"])
    candidate.set_user_attr("evaluated_case_ids", ["story/seed-0"])
    candidate.set_user_attr("evaluated_world_count", 1)
    candidate.set_user_attr("expected_complete_world_count", 1)
    candidate.report(2.0, step=1)
    assert candidate.should_prune() is False


def test_trial_number_seeded_tpe_resume_matches_uninterrupted(
    tmp_path,
) -> None:
    optuna = pytest.importorskip("optuna")

    def execute(storage_path, count):
        study = optuna.create_study(
            study_name="restart_equivalence",
            storage=f"sqlite:///{storage_path}",
            load_if_exists=True,
            direction="minimize",
            sampler=tune.build_sampler(seed=17),
        )
        study.optimize(
            lambda trial: (
                trial.suggest_float("x", 0.0, 1.0) - 0.3
            )
            ** 2,
            n_trials=count,
        )
        return [dict(trial.params) for trial in study.trials]

    uninterrupted = execute(tmp_path / "continuous.db", 12)
    execute(tmp_path / "resumed.db", 6)
    resumed = execute(tmp_path / "resumed.db", 6)
    assert resumed == uninterrupted


def test_run_tuning_trials_are_total_target_across_resume(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("optuna")

    def cheap_objective(trial, **_kwargs):
        return float(trial.suggest_float("x", 0.0, 1.0))

    monkeypatch.setattr(tune, "objective", cheap_objective)
    storage = f"sqlite:///{tmp_path / 'target.db'}"
    common = {
        "storage": storage,
        "study_name": "terminal_target",
        "cases": (HOSPITAL_BENCHMARK_STORIES[0],),
        "seeds": (0,),
        "steps": 1,
    }
    first = tune.run_tuning(n_trials=2, **common)
    assert len(first.trials) == 2
    resumed = tune.run_tuning(n_trials=3, **common)
    assert len(resumed.trials) == 3


def test_timeout_does_not_export_an_incomplete_trial_target(
    monkeypatch,
) -> None:
    pytest.importorskip("optuna")

    def slow_objective(_trial, **_kwargs):
        time.sleep(0.02)
        return 0.0

    monkeypatch.setattr(tune, "objective", slow_objective)
    with pytest.raises(RuntimeError, match="target terminal trials unfinished"):
        tune.run_tuning(
            n_trials=2,
            timeout_s=0.001,
            storage=None,
            study_name="incomplete_timeout",
            cases=(HOSPITAL_BENCHMARK_STORIES[0],),
            seeds=(0,),
            steps=1,
        )


def test_audited_best_ignores_lower_unaudited_complete_trial(
    monkeypatch,
    tmp_path,
) -> None:
    optuna = pytest.importorskip("optuna")
    story = HOSPITAL_BENCHMARK_STORIES[0]

    def fake_benchmark(**kwargs):
        return (_story_result(kwargs["cases"][0], kwargs["seeds"][0]),)

    monkeypatch.setattr(tune, "run_hospital_benchmark", fake_benchmark)
    config = tune.HospitalTuningConfig(
        cases=(story,),
        benchmark_seeds=(0,),
        steps=1,
        n_trials=2,
        storage=None,
        quick=True,
    )
    study = tune.create_study(
        study_name="audited_best",
        storage=None,
        load_if_exists=False,
        quick=True,
    )
    tune.bind_study_configuration(study, config)
    study.optimize(
        lambda trial: tune.objective(
            trial,
            base=config.base_config,
            cases=config.cases,
            seeds=config.benchmark_seeds,
            steps=config.steps,
            result_archive_dir=tmp_path,
        ),
        n_trials=1,
    )
    audited = study.trials[0]
    malformed = study.ask()
    study.tell(malformed, -100.0)
    assert study.best_trial.number == malformed.number
    assert tune._select_audited_best_trial(study, config).number == (
        audited.number
    )


def test_run_study_reports_selected_archive_without_benchmark_rerun(
    monkeypatch,
    tmp_path,
) -> None:
    pytest.importorskip("optuna")
    story = HOSPITAL_BENCHMARK_STORIES[0]

    def fake_benchmark(**kwargs):
        return (_story_result(kwargs["cases"][0], kwargs["seeds"][0]),)

    monkeypatch.setattr(tune, "run_hospital_benchmark", fake_benchmark)
    config = tune.HospitalTuningConfig(
        cases=(story,),
        benchmark_seeds=(0,),
        steps=1,
        n_trials=1,
        study_name="archive_only_reporting",
        storage=f"sqlite:///{tmp_path / 'study.db'}",
        output_prefix=tmp_path / "winner",
        quick=True,
    )
    study = tune.create_study(
        study_name=config.study_name,
        storage=config.storage,
        load_if_exists=False,
        quick=True,
    )
    fingerprint = tune.bind_study_configuration(study, config)
    archive_dir = tune._trial_archive_directory(config, fingerprint)
    study.optimize(
        lambda trial: tune.objective(
            trial,
            base=config.base_config,
            cases=config.cases,
            seeds=config.benchmark_seeds,
            steps=config.steps,
            result_archive_dir=archive_dir,
        ),
        n_trials=1,
    )

    def forbidden(**_kwargs):
        raise AssertionError("winner reporting must not rerun the benchmark")

    monkeypatch.setattr(tune, "run_hospital_benchmark", forbidden)
    result = tune.run_study(config)
    assert len(result.benchmark_results) == 1
    assert result.benchmark_results[0].case_id == f"{story}/seed-0"
    assert result.summary_path.is_file()
    report = load_benchmark_report(result.benchmark_reports.json)
    assert report.results == result.benchmark_results
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["benchmark_results_source"] == (
        "selected_complete_trial_exact_archive_no_rerun"
    )
    assert "validation_score" not in summary


def test_fingerprint_binds_relevant_source_content(monkeypatch) -> None:
    config = tune.HospitalTuningConfig()
    original = tune.study_configuration_fingerprint(config)
    monkeypatch.setattr(
        tune,
        "_relevant_source_content_sha256",
        lambda: {
            "objective": "changed",
            "controller": "changed",
            "benchmark": "changed",
            "scenarios": "changed",
        },
    )
    assert tune.study_configuration_fingerprint(config) != original


def test_source_fingerprint_covers_baseline_dispatch() -> None:
    hashes = tune._relevant_source_content_sha256()
    assert "hospital_baselines" in hashes
    assert "core_baselines" in hashes
