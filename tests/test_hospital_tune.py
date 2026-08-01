from __future__ import annotations

from dataclasses import replace

import pytest

from examples.hospital import tune
from examples.hospital.config import DEFAULT_CONFIG
from plcbf.benchmarking import BenchmarkOutcome, BenchmarkResult


class FakeTrial:
    def __init__(self) -> None:
        self.params: dict[str, object] = {}

    def suggest_int(self, name, low, high, step=1):
        value = low + ((high - low) // (2 * step)) * step
        self.params[name] = value
        return value

    def suggest_float(self, name, low, high, step=None, log=False):
        del log
        value = (
            0.5 * (low + high)
            if step is None
            else low + (int(round((high - low) / step)) // 2) * step
        )
        self.params[name] = value
        return value


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


def test_suggested_qp_refuge_config_can_be_reconstructed() -> None:
    trial = FakeTrial()
    suggested = tune.suggest_hospital_config(trial)
    reconstructed = tune.hospital_config_from_params(trial.params)
    assert reconstructed == suggested
    assert "guard_required_steps" not in trial.params
    assert "minimum_hold_time_s" not in trial.params
    assert "certificate_update_period" not in trial.params
    assert suggested.refuge.terminal_speed_max > 0.0
    with pytest.raises(ValueError, match="unknown tuned"):
        tune.hospital_config_from_params({"guard_required_steps": 8})


def test_tuning_score_uses_safety_and_task_metrics_not_fixed_dwell() -> None:
    success = _result(BenchmarkOutcome.SUCCESS, progress=1.0)
    timeout = _result(BenchmarkOutcome.TIMEOUT, progress=0.5)
    collision = _result(BenchmarkOutcome.COLLISION, progress=0.2)
    assert tune.score_results((success,)) < tune.score_results((timeout,))
    assert tune.score_results((timeout,)) < tune.score_results((collision,))


def test_tuning_split_and_fingerprint_are_stable() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        tune.HospitalTuningConfig(
            train_seeds=(1, 2),
            validation_seeds=(2, 3),
        )
    first = tune.HospitalTuningConfig()
    second = replace(first)
    assert tune.study_configuration_fingerprint(first) == (
        tune.study_configuration_fingerprint(second)
    )
    assert first.metadata()["external_refuge_state_machine"] is False
    assert first.metadata()["oracle_period"] == "every_plant_step"
    assert first.metadata()["oracle_period_s"] == first.base_config.dt
