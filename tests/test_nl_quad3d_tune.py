from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.nl_quad3d import tune
from examples.nl_quad3d.config_io import load_controller_config_artifact
from examples.nl_quad3d.config_io import write_controller_config_artifact
from examples.nl_quad3d.controller import NLQuad3DControllerConfig
from plcbf.benchmarking import (
    BenchmarkOutcome,
    BenchmarkReportPaths,
    BenchmarkResult,
)


class FakeTrial:
    def __init__(self, *, prune_after: int | None = None) -> None:
        self.params = {}
        self.user_attrs = {}
        self.reports = []
        self.prune_after = prune_after

    def suggest_categorical(self, name, choices):
        value = choices[0]
        self.params[name] = value
        return value

    def suggest_float(self, name, low, high, *, log=False):
        value = (low * high) ** 0.5 if log else 0.5 * (low + high)
        self.params[name] = value
        return value

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value

    def report(self, value, step):
        self.reports.append((step, value))

    def should_prune(self):
        return (
            self.prune_after is not None
            and len(self.reports) >= self.prune_after
        )


class FakeStudy:
    def __init__(self, *, trials=(), user_attrs=None) -> None:
        self.trials = list(trials)
        self.user_attrs = dict(user_attrs or {})

    def set_user_attr(self, name, value) -> None:
        self.user_attrs[name] = value

    def enqueue_trial(self, params, *, user_attrs=None) -> None:
        self.trials.append(
            SimpleNamespace(params=dict(params), user_attrs=dict(user_attrs or {}))
        )


def _pruner_trial(
    values,
    *,
    state: str = "RUNNING",
):
    intermediate = {int(step): float(value) for step, value in values.items()}
    return SimpleNamespace(
        state=SimpleNamespace(name=state),
        intermediate_values=intermediate,
        last_step=max(intermediate) if intermediate else None,
    )


def _pruner_study(trials, *, direction: str = "MINIMIZE"):
    return SimpleNamespace(
        direction=SimpleNamespace(name=direction),
        trials=list(trials),
    )


def _prefix_pruner(
    *,
    startup: int = 8,
    warmup: int = 20,
    interval: int = 5,
    minimum: int = 3,
    patience: int = 5,
):
    return tune.ExactSeedPrefixPatientMedianPruner(
        n_startup_trials=startup,
        n_warmup_steps=warmup,
        interval_steps=interval,
        n_min_trials=minimum,
        patience=patience,
    )


def _result(
    outcome: BenchmarkOutcome | str,
    *,
    scenario: str = "scenario",
    seed: int = 0,
    clearance: float = 0.2,
    intervention: float = 1.0,
    fallback_count: int = 0,
) -> BenchmarkResult:
    return BenchmarkResult(
        algorithm="plcbf",
        case_id=f"{scenario}/seed-{seed}",
        seed=seed,
        outcome=outcome,
        min_clearance=clearance,
        intervention=intervention,
        solve_times_s=(0.01, 0.02),
        case_metrics={
            "control_steps": 2,
            "solver_fallback_count": fallback_count,
            "reached_goal": outcome == BenchmarkOutcome.SUCCESS,
        },
    )


def test_suggested_controller_is_valid_and_reconstructable() -> None:
    trial = FakeTrial()
    base = NLQuad3DControllerConfig()
    suggested = tune.suggest_controller_config(
        trial, base=base, quick=True
    )
    reconstructed = tune.controller_config_from_params(
        trial.params, base=base, quick=True
    )

    assert suggested == reconstructed
    assert suggested.backup_horizon in (0.1, 0.2)
    assert suggested.num_radial_policies in (2, 4)
    assert suggested.max_obstacles == 2
    assert suggested.nominal_prefix_steps == 0
    assert suggested.max_operator == "input_space"
    assert "nominal_prefix_steps" not in trial.params


def test_unknown_reconstructed_parameter_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown tuned"):
        tune.controller_config_from_params({"not_a_parameter": 1.0})


def test_tuning_score_has_strict_failure_priority() -> None:
    success = tune.score_results([_result("success")]).score
    timeout = tune.score_results([_result("timeout")]).score
    infeasible = tune.score_results([_result("infeasible")]).score
    collision = tune.score_results([_result("collision")]).score
    error = tune.score_results([_result("error")]).score

    assert success < timeout < infeasible < collision < error


def test_tuning_score_includes_clearance_fallback_intervention_and_timing() -> None:
    better = tune.score_results(
        [_result("success", clearance=0.4, intervention=0.1)]
    )
    worse = tune.score_results(
        [
            BenchmarkResult(
                "plcbf",
                "scenario/seed-0",
                0,
                "success",
                min_clearance=-0.05,
                intervention=10.0,
                solve_times_s=(0.5,),
                case_metrics={
                    "control_steps": 1,
                    "solver_fallback_count": 1,
                },
            )
        ]
    )

    assert better.score < worse.score
    assert worse.clearance_shortfall > 0.0
    assert worse.solver_fallback_rate == 1.0
    assert worse.mean_intervention == 10.0
    assert worse.mean_step_time_s == 0.5


def test_wall_time_is_audited_but_does_not_change_objective() -> None:
    fast = _result("success")
    slow = BenchmarkResult(
        fast.algorithm,
        fast.case_id,
        fast.seed,
        fast.outcome,
        min_clearance=fast.min_clearance,
        intervention=fast.intervention,
        solve_times_s=(10.0, 20.0),
        case_metrics=fast.case_metrics,
    )

    fast_score = tune.score_results((fast,))
    slow_score = tune.score_results((slow,))

    assert fast_score.mean_step_time_s != slow_score.mean_step_time_s
    assert fast_score.score == slow_score.score


def test_success_count_dominates_failure_composition() -> None:
    one_collision = tune.score_results(
        [_result("success"), _result("success"), _result("collision")]
    )
    two_timeouts = tune.score_results(
        [_result("success"), _result("timeout"), _result("timeout")]
    )

    assert one_collision.success_count == 2
    assert two_timeouts.success_count == 1
    assert one_collision.score < two_timeouts.score


def test_objective_runs_only_plcbf_on_fixed_training_split() -> None:
    captured = {}

    def fake_runner(config):
        captured["config"] = config
        return (
            _result(
                "success",
                scenario=config.scenarios[0],
                seed=config.seeds[0],
            ),
        )

    config = tune.NLQuad3DTuningConfig(
        scenarios=("head_on",),
        train_seeds=(3,),
        validation_seeds=(9,),
        max_steps=2,
        n_trials=1,
        storage=None,
        quick=True,
    )
    trial = FakeTrial()
    value = tune.build_objective(
        config, benchmark_runner=fake_runner
    )(trial)

    assert value >= 0.0
    assert captured["config"].methods == ("plcbf",)
    assert captured["config"].scenarios == ("head_on",)
    assert captured["config"].seeds == (3,)
    assert captured["config"].max_steps == 2
    assert "controller_config" in trial.user_attrs
    assert trial.user_attrs["outcomes"] == ["success"]
    assert trial.user_attrs["score_components"]["success_count"] == 1
    assert trial.user_attrs["completed_case_count"] == 1
    assert trial.user_attrs["completed_full_protocol"] is True


def test_full_defaults_are_frozen_stress_protocol() -> None:
    config = tune.NLQuad3DTuningConfig()

    assert config.scenarios == ("playground_stress",)
    assert config.train_seeds == tuple(range(1, 101))
    assert config.validation_seeds == tuple(range(101, 201))
    assert config.max_steps is None
    assert config.playground_obstacle_count == 48
    assert config.n_jobs == 1
    assert config.metadata()["resolved_max_steps"] == {
        "playground_stress": 800
    }
    assert config.metadata()["protocol_kind"] == "canonical_full"


def test_parallel_optuna_threads_are_rejected_until_rng_is_local() -> None:
    with pytest.raises(ValueError, match="n_jobs must equal one"):
        tune.NLQuad3DTuningConfig(n_jobs=2)


def test_objective_reports_each_seed_and_a_complete_trial_audits_all_cases() -> None:
    observed_seeds = []

    def fake_runner(config):
        seed = config.seeds[0]
        observed_seeds.append(seed)
        result = _result("success")
        return (
            BenchmarkResult(
                result.algorithm,
                f"playground_stress/seed-{seed}",
                seed,
                result.outcome,
                min_clearance=result.min_clearance,
                intervention=result.intervention,
                solve_times_s=result.solve_times_s,
                case_metrics=result.case_metrics,
            ),
        )

    config = tune.NLQuad3DTuningConfig(
        train_seeds=(1, 2, 3),
        validation_seeds=(101,),
        n_trials=1,
        storage=None,
        quick=True,
    )
    trial = FakeTrial()

    tune.build_objective(config, benchmark_runner=fake_runner)(trial)

    assert observed_seeds == [1, 2, 3]
    assert [step for step, _ in trial.reports] == [1, 2, 3]
    assert trial.user_attrs["evaluated_seed_count"] == 3
    assert trial.user_attrs["completed_case_count"] == 3
    assert trial.user_attrs["expected_complete_case_count"] == 3


def test_canonical_completed_trial_contains_exactly_100_cases() -> None:
    observed_seeds = []

    def fake_runner(config):
        observed_seeds.extend(config.seeds)
        return (
            _result(
                "success",
                scenario=config.scenarios[0],
                seed=config.seeds[0],
            ),
        )

    config = tune.NLQuad3DTuningConfig(
        n_trials=1,
        storage=None,
    )
    trial = FakeTrial()

    tune.build_objective(config, benchmark_runner=fake_runner)(trial)

    assert observed_seeds == list(range(1, 101))
    assert trial.user_attrs["completed_case_count"] == 100
    assert trial.user_attrs["expected_complete_case_count"] == 100
    assert trial.user_attrs["evaluated_seed_count"] == 100
    assert trial.user_attrs["completed_full_protocol"] is True


def test_pruned_objective_stops_before_remaining_seeds() -> None:
    observed_seeds = []

    def fake_runner(config):
        observed_seeds.append(config.seeds[0])
        return (
            _result(
                "timeout",
                scenario=config.scenarios[0],
                seed=config.seeds[0],
            ),
        )

    config = tune.NLQuad3DTuningConfig(
        train_seeds=(1, 2, 3, 4),
        validation_seeds=(101,),
        n_trials=1,
        storage=None,
        quick=True,
    )
    trial = FakeTrial(prune_after=2)

    with pytest.raises(Exception) as captured:
        tune.build_objective(config, benchmark_runner=fake_runner)(trial)

    assert type(captured.value).__name__ == "TrialPruned"
    assert observed_seeds == [1, 2]
    assert trial.user_attrs["evaluated_seed_count"] == 2
    assert trial.user_attrs["evaluated_case_count"] == 2
    assert "completed_case_count" not in trial.user_attrs


def test_prefix_pruner_waits_for_eight_completed_startup_trials() -> None:
    references = [
        _pruner_trial({20: 1.0}, state="COMPLETE") for _ in range(7)
    ]
    candidate = _pruner_trial({20: 2.0})

    assert not _prefix_pruner(patience=1).prune(
        _pruner_study(references), candidate
    )


def test_prefix_pruner_does_not_compare_before_warmup_step() -> None:
    references = [
        _pruner_trial({15: 1.0}, state="COMPLETE") for _ in range(8)
    ]
    candidate = _pruner_trial({15: 2.0})

    assert not _prefix_pruner(patience=1).prune(
        _pruner_study(references), candidate
    )


def test_prefix_pruner_uses_exact_step_not_best_earlier_value() -> None:
    references = [
        _pruner_trial({20: 100.0, 25: 10.0}, state="COMPLETE")
        for _ in range(8)
    ]
    # The candidate's earlier 0.0 is better than every reference's earlier
    # value, but its current seed-25 prefix is strictly worse than the seed-25
    # median.  A best-so-far median comparison would incorrectly retain it.
    candidate = _pruner_trial({20: 0.0, 25: 20.0})

    assert _prefix_pruner(patience=1).prune(
        _pruner_study(references), candidate
    )


def test_prefix_pruner_requires_five_consecutive_bad_checkpoints() -> None:
    checkpoints = (20, 25, 30, 35, 40)
    references = [
        _pruner_trial(
            {step: 10.0 for step in checkpoints}, state="COMPLETE"
        )
        for _ in range(8)
    ]
    study = _pruner_study(references)

    assert not _prefix_pruner().prune(
        study,
        _pruner_trial({step: 11.0 for step in checkpoints[:-1]}),
    )
    assert _prefix_pruner().prune(
        study,
        _pruner_trial({step: 11.0 for step in checkpoints}),
    )


def test_prefix_pruner_recovery_checkpoint_resets_patience() -> None:
    checkpoints = (20, 25, 30, 35, 40)
    references = [
        _pruner_trial(
            {step: 10.0 for step in checkpoints}, state="COMPLETE"
        )
        for _ in range(8)
    ]
    candidate_values = {step: 11.0 for step in checkpoints}
    candidate_values[30] = 9.0

    assert not _prefix_pruner().prune(
        _pruner_study(references), _pruner_trial(candidate_values)
    )


def test_prefix_pruner_requires_three_complete_exact_step_references() -> None:
    complete = [
        _pruner_trial({20: 10.0}, state="COMPLETE") for _ in range(2)
    ] + [
        _pruner_trial({15: 10.0}, state="COMPLETE") for _ in range(6)
    ]
    pruned = [
        _pruner_trial({20: 1.0}, state="PRUNED") for _ in range(5)
    ]

    assert not _prefix_pruner(patience=1).prune(
        _pruner_study(complete + pruned), _pruner_trial({20: 20.0})
    )


def test_prefix_pruner_retains_equal_or_better_candidate() -> None:
    references = [
        _pruner_trial({20: 10.0}, state="COMPLETE") for _ in range(8)
    ]
    study = _pruner_study(references)
    pruner = _prefix_pruner(patience=1)

    assert not pruner.prune(study, _pruner_trial({20: 10.0}))
    assert not pruner.prune(study, _pruner_trial({20: 9.0}))


def test_build_pruner_uses_audited_same_prefix_settings() -> None:
    pruner = tune.build_pruner()

    assert isinstance(pruner, tune.ExactSeedPrefixPatientMedianPruner)
    assert pruner.n_startup_trials == 8
    assert pruner.n_warmup_steps == 20
    assert pruner.interval_steps == 5
    assert pruner.n_min_trials == 3
    assert pruner.patience == 5
    assert tune.DEFAULT_PRUNER_SETTINGS["direction"] == "minimize"
    assert tune.DEFAULT_PRUNER_SETTINGS["reference_trials"] == "complete_only"


def test_real_optuna_study_accepts_pruner_and_prunes_bad_prefix() -> None:
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study(
        direction="minimize",
        pruner=tune.build_pruner(),
    )
    checkpoints = (20, 25, 30, 35, 40)
    for _ in range(8):
        reference = study.ask()
        for step in checkpoints:
            reference.report(10.0, step=step)
        study.tell(reference, 10.0)

    candidate = study.ask()
    decisions = []
    for step in checkpoints:
        candidate.report(11.0, step=step)
        decisions.append(candidate.should_prune())

    assert decisions == [False, False, False, False, True]
    study.tell(candidate, state=optuna.trial.TrialState.PRUNED)
    assert study.trials[-1].state is optuna.trial.TrialState.PRUNED


def test_objective_rejects_missing_or_misidentified_cases() -> None:
    config = tune.NLQuad3DTuningConfig(
        scenarios=("head_on", "vertical_drop"),
        train_seeds=(3,),
        validation_seeds=(9,),
        max_steps=1,
        n_trials=1,
        storage=None,
        quick=True,
    )

    def incomplete(_config):
        return (_result("success", scenario="head_on", seed=3),)

    with pytest.raises(RuntimeError, match="exact configured grid"):
        tune.build_objective(config, benchmark_runner=incomplete)(FakeTrial())

    def wrong_seed(_config):
        return (
            _result("success", scenario="head_on", seed=4),
            _result("success", scenario="vertical_drop", seed=4),
        )

    with pytest.raises(RuntimeError, match="exact configured grid"):
        tune.build_objective(config, benchmark_runner=wrong_seed)(FakeTrial())


def test_training_and_validation_seeds_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        tune.NLQuad3DTuningConfig(
            train_seeds=(1, 2),
            validation_seeds=(2, 3),
        )


def test_study_fingerprint_rejects_incompatible_resume() -> None:
    original = tune.NLQuad3DTuningConfig(
        scenarios=("head_on",),
        train_seeds=(0,),
        validation_seeds=(101,),
        max_steps=2,
        n_trials=1,
        storage=None,
        quick=True,
    )
    study = FakeStudy()
    fingerprint = tune.bind_study_configuration(study, original)

    assert fingerprint == tune.study_configuration_fingerprint(original)
    assert study.user_attrs["objective_configuration_fingerprint"] == fingerprint
    tune.bind_study_configuration(study, original)

    incompatible = tune.NLQuad3DTuningConfig(
        scenarios=("vertical_drop",),
        train_seeds=(0,),
        validation_seeds=(101,),
        max_steps=2,
        n_trials=1,
        storage=None,
        quick=True,
    )
    with pytest.raises(ValueError, match="does not match"):
        tune.bind_study_configuration(study, incompatible)


def test_stress_protocol_version_is_bound_into_study_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tune.NLQuad3DTuningConfig()
    original = tune.study_configuration_fingerprint(config)
    monkeypatch.setattr(
        tune,
        "PLAYGROUND_STRESS_PROTOCOL_VERSION",
        "synthetic_protocol_revision",
    )
    assert tune.study_configuration_fingerprint(config) != original


def test_resume_target_counts_terminal_states_but_not_waiting() -> None:
    state = lambda name: SimpleNamespace(name=name)
    trials = [
        SimpleNamespace(state=state("COMPLETE")),
        SimpleNamespace(state=state("PRUNED")),
        SimpleNamespace(state=state("FAIL")),
        SimpleNamespace(state=state("WAITING")),
    ]
    config = tune.NLQuad3DTuningConfig(
        n_trials=6,
        storage=None,
    )
    assert tune._remaining_target_trials(
        FakeStudy(trials=trials), config
    ) == 3

    running = FakeStudy(
        trials=(SimpleNamespace(state=state("RUNNING")),)
    )
    with pytest.raises(RuntimeError, match="RUNNING"):
        tune._remaining_target_trials(running, config)


def test_best_trial_selection_ignores_unaudited_complete_trial() -> None:
    config = tune.NLQuad3DTuningConfig(storage=None)
    base = asdict(config.base_controller_config)
    expected_cases = 100
    audited_attrs = {
        "completed_full_protocol": True,
        "evaluated_seed_count": 100,
        "completed_case_count": expected_cases,
        "expected_complete_case_count": expected_cases,
        "evaluated_seeds": list(config.train_seeds),
        "score_components": {"total_cases": expected_cases},
        "outcomes": ["success"] * expected_cases,
        "controller_config": base,
    }
    complete_state = SimpleNamespace(name="COMPLETE")
    unaudited = SimpleNamespace(
        state=complete_state,
        value=0.0,
        number=1,
        params={},
        user_attrs={},
    )
    audited = SimpleNamespace(
        state=complete_state,
        value=1.0,
        number=2,
        params={},
        user_attrs=audited_attrs,
    )
    study = FakeStudy(trials=(unaudited, audited))

    assert tune._select_audited_best_trial(study, config) is audited


def test_legacy_study_with_trials_requires_new_name() -> None:
    config = tune.NLQuad3DTuningConfig(
        scenarios=("head_on",),
        train_seeds=(0,),
        validation_seeds=(101,),
        max_steps=2,
        n_trials=1,
        storage=None,
        quick=True,
    )
    with pytest.raises(ValueError, match="legacy Optuna study"):
        tune.bind_study_configuration(FakeStudy(trials=(object(),)), config)


def test_new_full_study_enqueues_exact_base_once() -> None:
    config = tune.NLQuad3DTuningConfig(storage=None)
    study = FakeStudy()

    assert tune.enqueue_reference_trial(study, config) is True
    assert len(study.trials) == 1
    assert study.trials[0].params["backup_horizon"] == (
        config.base_controller_config.backup_horizon
    )
    assert study.trials[0].params["max_obstacles"] == (
        config.base_controller_config.max_obstacles
    )
    assert "nominal_prefix_steps" not in study.trials[0].params
    assert study.user_attrs["base_controller_reference_enqueued"] is True
    assert tune.enqueue_reference_trial(study, config) is False


def test_quick_study_does_not_enqueue_reference() -> None:
    config = tune.NLQuad3DTuningConfig(
        train_seeds=(0,),
        validation_seeds=(101,),
        max_steps=1,
        n_trials=1,
        storage=None,
        quick=True,
    )
    study = FakeStudy()

    assert tune.enqueue_reference_trial(study, config) is False
    assert study.trials == []


@pytest.mark.parametrize(
    "config",
    [
        tune.NLQuad3DTuningConfig(n_trials=1),
        tune.NLQuad3DTuningConfig(scenarios=("head_on",)),
        tune.NLQuad3DTuningConfig(train_seeds=tuple(range(1, 100))),
        tune.NLQuad3DTuningConfig(validation_seeds=(101,)),
        tune.NLQuad3DTuningConfig(max_steps=800),
        tune.NLQuad3DTuningConfig(playground_obstacle_count=47),
        tune.NLQuad3DTuningConfig(warmup=False),
        tune.NLQuad3DTuningConfig(quick=True),
        tune.NLQuad3DTuningConfig(
            base_controller_config=NLQuad3DControllerConfig(cbf_alpha=3.0)
        ),
        tune.NLQuad3DTuningConfig(
            base_controller_config=NLQuad3DControllerConfig(
                backup_horizon=1.0
            )
        ),
    ],
)
def test_incomplete_protocol_cannot_target_packaged_default(config) -> None:
    with pytest.raises(ValueError, match="may not overwrite the packaged"):
        tune.run_study(config)


def test_cli_does_not_run_study_without_explicit_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    called = False

    def forbidden(_):
        nonlocal called
        called = True
        raise AssertionError("study should not run")

    monkeypatch.setattr(tune, "run_study", forbidden)
    return_code = tune.main(["--quick", "--storage", "none"])
    payload = json.loads(capsys.readouterr().out)

    assert return_code == 0
    assert not called
    assert payload["status"] == "ready"
    assert "Pass --run" in payload["message"]
    assert payload["configuration"]["n_trials"] == 1
    assert payload["configuration"]["train_seeds"] == [0]
    assert payload["configuration"]["validation_seeds"] == [101]


def test_cli_run_flag_dispatches_explicit_study(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured = {}
    reports = BenchmarkReportPaths(
        csv=tmp_path / "validation.csv",
        json=tmp_path / "validation.json",
        markdown=tmp_path / "validation.md",
    )

    def fake_run(config):
        captured["config"] = config
        return tune.TuningRunResult(
            study=SimpleNamespace(
                study_name=config.study_name,
                best_value=0.25,
            ),
            best_controller_config=config.base_controller_config,
            validation_results=(_result("success"),),
            validation_reports=reports,
            summary_path=tmp_path / "summary.json",
        )

    monkeypatch.setattr(tune, "run_study", fake_run)
    return_code = tune.main(
        [
            "--run",
            "--quick",
            "--storage",
            "none",
            "--output",
            str(tmp_path / "tuning"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert return_code == 0
    assert captured["config"].quick
    assert captured["config"].n_trials == 1
    assert payload["study"] == "nl_quad3d_plcbf_quick"
    assert payload["best_value"] == 0.25


def test_run_study_exports_best_yaml_summary_and_auditable_complete_trial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_benchmark(config):
        return tuple(
            BenchmarkResult(
                algorithm="plcbf",
                case_id=f"{scenario}/seed-{seed}",
                seed=seed,
                outcome="success",
                min_clearance=0.25,
                intervention=0.1,
                solve_times_s=(0.001,),
                case_metrics={
                    "control_steps": 1,
                    "solver_fallback_count": 0,
                },
            )
            for scenario in config.scenarios
            for seed in config.seeds
        )

    monkeypatch.setattr(tune, "run_benchmark", fake_benchmark)
    config = tune.NLQuad3DTuningConfig(
        scenarios=("head_on",),
        train_seeds=(0,),
        validation_seeds=(101,),
        max_steps=1,
        n_trials=1,
        storage=None,
        study_name="unit_test_export",
        output_prefix=tmp_path / "study",
        best_config_output=tmp_path / "winner.yaml",
        quick=True,
    )

    result = tune.run_study(config)

    assert result.best_config_path == tmp_path / "winner.yaml"
    assert result.best_config_path.is_file()
    assert result.summary_path.is_file()
    exported, payload = load_controller_config_artifact(
        result.best_config_path
    )
    assert exported == result.best_controller_config
    assert payload["provenance"]["study"]["best_trial_number"] == 0
    assert payload["provenance"]["held_out_validation"]["error_count"] == 0
    complete = result.study.best_trial.user_attrs
    assert complete["completed_case_count"] == 1
    assert complete["expected_complete_case_count"] == 1
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["best_controller_yaml"] == str(result.best_config_path)


def test_resume_runs_only_remaining_terminal_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_benchmark(config):
        return tuple(
            _result("success", scenario=scenario, seed=seed)
            for scenario in config.scenarios
            for seed in config.seeds
        )

    monkeypatch.setattr(tune, "run_benchmark", fake_benchmark)
    storage = f"sqlite:///{tmp_path / 'resume.db'}"
    initial = tune.NLQuad3DTuningConfig(
        scenarios=("head_on",),
        train_seeds=(0,),
        validation_seeds=(101,),
        max_steps=1,
        n_trials=2,
        storage=storage,
        study_name="resume_target_test",
        output_prefix=tmp_path / "resume",
        best_config_output=tmp_path / "winner.yaml",
        quick=True,
    )
    first = tune.run_study(initial)
    assert sum(
        tune._trial_state_name(item) in tune._TARGET_TERMINAL_STATES
        for item in first.study.trials
    ) == 2

    resumed = tune.run_study(replace(initial, n_trials=3))
    assert sum(
        tune._trial_state_name(item) in tune._TARGET_TERMINAL_STATES
        for item in resumed.study.trials
    ) == 3


def test_validation_error_never_replaces_existing_controller_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    winner = tmp_path / "winner.yaml"
    write_controller_config_artifact(
        winner,
        NLQuad3DControllerConfig(cbf_alpha=6.0),
        provenance={"source": "preexisting_valid_controller"},
    )
    original = winner.read_bytes()

    def fake_benchmark(config):
        outcome = "error" if config.seeds == (101,) else "success"
        return tuple(
            _result(outcome, scenario=scenario, seed=seed)
            for scenario in config.scenarios
            for seed in config.seeds
        )

    monkeypatch.setattr(tune, "run_benchmark", fake_benchmark)
    config = tune.NLQuad3DTuningConfig(
        scenarios=("head_on",),
        train_seeds=(0,),
        validation_seeds=(101,),
        max_steps=1,
        n_trials=1,
        storage=None,
        study_name="validation_export_guard",
        output_prefix=tmp_path / "guard",
        best_config_output=winner,
        quick=True,
    )

    with pytest.raises(RuntimeError, match="held-out validation"):
        tune.run_study(config)
    assert winner.read_bytes() == original


def test_cli_returns_nonzero_for_held_out_validation_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reports = BenchmarkReportPaths(
        csv=tmp_path / "validation.csv",
        json=tmp_path / "validation.json",
        markdown=tmp_path / "validation.md",
    )
    monkeypatch.setattr(
        tune,
        "run_study",
        lambda config: tune.TuningRunResult(
            study=SimpleNamespace(
                study_name=config.study_name,
                best_value=1.0,
            ),
            best_controller_config=config.base_controller_config,
            validation_results=(_result("error"),),
            validation_reports=reports,
            summary_path=tmp_path / "summary.json",
        ),
    )

    assert tune.main(["--run", "--quick", "--storage", "none"]) == 1
