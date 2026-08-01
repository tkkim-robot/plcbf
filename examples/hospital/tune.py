"""Reproducible Optuna tuning for the strict hospital refuge benchmark.

Importing this module creates no study and runs no trials. The CLI prints its
resolved configuration by default and requires ``--run`` before optimization.
Every completed study evaluates its best configuration on disjoint held-out
seeds and writes raw CSV/JSON plus an aggregate Markdown report.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

from plcbf.benchmarking import (
    BenchmarkOutcome,
    BenchmarkReportPaths,
    BenchmarkResult,
    write_benchmark_reports,
)

from .benchmark import STRICT_HOSPITAL_CASES, run_hospital_benchmark
from .config import DEFAULT_CONFIG, HospitalConfig
from .simulation import validate_strict_refuge_protocol

if TYPE_CHECKING:
    import optuna


_STUDY_FINGERPRINT_ATTRIBUTE = "objective_configuration_fingerprint"
_SEARCH_SPACE_VERSION = "hospital_plcbf_v4_per_step_oracle"
_TUNABLE_NAMES = frozenset(
    {
        "num_angle_policies",
        "room_policy_count",
        "angle_target_speed",
        "room_target_speed",
        "room_horizon",
        "room_rollout_dt",
        "cbf_alpha",
        "component_temperature",
        "time_temperature",
        "safety_margin",
        "stretcher_margin",
        "hocbf_lambda1",
        "hocbf_lambda2",
        "inside_door_offset",
        "terminal_interior_margin",
        "terminal_speed_max",
    }
)


@dataclass(frozen=True)
class HospitalTuningConfig:
    """Fixed train/validation split and Optuna execution settings."""

    cases: tuple[str, ...] = tuple(STRICT_HOSPITAL_CASES)
    train_seeds: tuple[int, ...] = (0, 1)
    validation_seeds: tuple[int, ...] = (101, 102)
    steps: int = 1100
    n_trials: int = 50
    timeout_s: float | None = None
    sampler_seed: int = 0
    study_name: str = "hospital_plcbf"
    storage: str | None = "sqlite:///results/hospital_optuna.db"
    output_prefix: Path = Path("results/hospital_optuna")
    quick: bool = False
    base_config: HospitalConfig = field(default_factory=HospitalConfig)

    def __post_init__(self) -> None:
        validate_strict_refuge_protocol(self.base_config)
        cases = tuple(str(case) for case in self.cases)
        train_seeds = tuple(int(seed) for seed in self.train_seeds)
        validation_seeds = tuple(int(seed) for seed in self.validation_seeds)
        if not cases:
            raise ValueError("at least one hospital tuning case is required")
        unknown = set(cases) - set(STRICT_HOSPITAL_CASES)
        if unknown:
            raise ValueError(
                "unknown hospital tuning cases: " + ", ".join(sorted(unknown))
            )
        if not train_seeds:
            raise ValueError("at least one training seed is required")
        if not validation_seeds:
            raise ValueError("at least one held-out validation seed is required")
        if set(train_seeds) & set(validation_seeds):
            raise ValueError("training and validation seeds must be disjoint")
        if self.steps < 1 or self.n_trials < 1:
            raise ValueError("steps and n_trials must be positive")
        if self.timeout_s is not None and self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive when provided")
        study_name = str(self.study_name)
        if not study_name:
            raise ValueError("study_name must not be empty")
        output_prefix = Path(self.output_prefix)
        if output_prefix.suffix:
            output_prefix = output_prefix.with_suffix("")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "train_seeds", train_seeds)
        object.__setattr__(self, "validation_seeds", validation_seeds)
        object.__setattr__(self, "sampler_seed", int(self.sampler_seed))
        object.__setattr__(self, "study_name", study_name)
        object.__setattr__(self, "output_prefix", output_prefix)
        object.__setattr__(self, "quick", bool(self.quick))

    def metadata(self) -> dict[str, object]:
        return {
            "case_study": "hospital_refuge",
            "method": "plcbf",
            "cases": list(self.cases),
            "train_seeds": list(self.train_seeds),
            "validation_seeds": list(self.validation_seeds),
            "steps": self.steps,
            "n_trials": self.n_trials,
            "timeout_s": self.timeout_s,
            "sampler_seed": self.sampler_seed,
            "study_name": self.study_name,
            "storage": self.storage,
            "quick": self.quick,
            "policy_library": "full",
            "oracle_period": "every_plant_step",
            "oracle_period_s": self.base_config.dt,
            "seed_zero_is_exact_reference": False,
            "external_refuge_state_machine": False,
        }


def suggest_hospital_config(
    trial: Any,
    base: HospitalConfig = DEFAULT_CONFIG,
) -> HospitalConfig:
    """Materialize one internally consistent PL-CBF configuration."""

    policies = replace(
        base.policies,
        num_angle_policies=trial.suggest_int(
            "num_angle_policies", 4, 10, step=2
        ),
        room_policy_count=trial.suggest_int("room_policy_count", 1, 3),
        angle_target_speed=trial.suggest_float(
            "angle_target_speed", 1.2, 2.2
        ),
        room_target_speed=trial.suggest_float(
            "room_target_speed", 1.5, 2.6
        ),
        room_horizon=trial.suggest_float(
            "room_horizon", 6.0, 10.0, step=0.4
        ),
        room_rollout_dt=trial.suggest_float(
            "room_rollout_dt", 0.24, 0.48, step=0.06
        ),
        cbf_alpha=trial.suggest_float("cbf_alpha", 0.4, 2.0, log=True),
        component_temperature=trial.suggest_float(
            "component_temperature", 24.0, 60.0
        ),
        time_temperature=trial.suggest_float(
            "time_temperature", 20.0, 55.0
        ),
    )
    safety = replace(
        base.safety,
        safety_margin=trial.suggest_float(
            "safety_margin", 0.30, 0.70
        ),
        stretcher_margin=trial.suggest_float(
            "stretcher_margin", 0.35, 0.80
        ),
        hocbf_lambda1=trial.suggest_float(
            "hocbf_lambda1", 0.20, 0.80
        ),
        hocbf_lambda2=trial.suggest_float(
            "hocbf_lambda2", 0.40, 1.60
        ),
    )
    refuge = replace(
        base.refuge,
        inside_door_offset=trial.suggest_float(
            "inside_door_offset", 1.8, 3.0
        ),
        terminal_interior_margin=trial.suggest_float(
            "terminal_interior_margin", 1.6, 2.6
        ),
        terminal_speed_max=trial.suggest_float(
            "terminal_speed_max", 1.5, 2.85
        ),
    )
    config = replace(base, policies=policies, safety=safety, refuge=refuge)
    validate_strict_refuge_protocol(config)
    return config


def hospital_config_from_params(
    params: Mapping[str, Any],
    *,
    base: HospitalConfig = DEFAULT_CONFIG,
) -> HospitalConfig:
    """Reconstruct the best hospital configuration from Optuna parameters."""

    unknown = set(params) - _TUNABLE_NAMES
    if unknown:
        raise ValueError(
            "unknown tuned hospital parameters: " + ", ".join(sorted(unknown))
        )

    def values(names: set[str], source: Any) -> dict[str, Any]:
        return {
            name: params.get(name, getattr(source, name))
            for name in names
        }

    policy_names = {
        "num_angle_policies",
        "room_policy_count",
        "angle_target_speed",
        "room_target_speed",
        "room_horizon",
        "room_rollout_dt",
        "cbf_alpha",
        "component_temperature",
        "time_temperature",
    }
    safety_names = {
        "safety_margin",
        "stretcher_margin",
        "hocbf_lambda1",
        "hocbf_lambda2",
    }
    refuge_names = {
        "inside_door_offset",
        "terminal_interior_margin",
        "terminal_speed_max",
    }
    config = replace(
        base,
        policies=replace(
            base.policies,
            **values(policy_names, base.policies),
        ),
        safety=replace(
            base.safety,
            **values(safety_names, base.safety),
        ),
        refuge=replace(
            base.refuge,
            **values(refuge_names, base.refuge),
        ),
    )
    validate_strict_refuge_protocol(config)
    return config


def score_results(results: Iterable[BenchmarkResult]) -> float:
    """Return a safety-first task score; lower is better."""

    trials = tuple(results)
    if not trials:
        raise ValueError("at least one benchmark result is required")
    errors = sum(
        result.outcome is BenchmarkOutcome.ERROR for result in trials
    )
    collisions = sum(
        result.outcome is BenchmarkOutcome.COLLISION for result in trials
    )
    infeasible = sum(
        result.outcome is BenchmarkOutcome.INFEASIBLE for result in trials
    )
    timeouts = sum(
        result.outcome is BenchmarkOutcome.TIMEOUT for result in trials
    )
    base = len(trials) + 1
    failure_rank = (
        errors * base**3
        + collisions * base**2
        + infeasible * base
        + timeouts
    )

    progress_shortfall = float(
        np.mean(
            [
                max(
                    0.0,
                    1.0 - float(result.case_metrics.get("progress", 0.0)),
                )
                for result in trials
            ]
        )
    )
    clearance_shortfall = float(
        np.mean(
            [
                max(0.0, -float(result.min_clearance))
                for result in trials
                if result.min_clearance is not None
            ]
            or [0.0]
        )
    )
    intervention = float(
        np.mean(
            [
                float(result.intervention)
                for result in trials
                if result.intervention is not None
            ]
            or [0.0]
        )
    )
    timing = float(
        np.mean(
            [
                float(
                    result.case_metrics.get(
                        "oracle_and_solver_time_total_s",
                        0.0,
                    )
                )
                for result in trials
            ]
        )
    )
    solver_fallback_count = sum(
        int(result.case_metrics.get("solver_fallback_count", 0))
        for result in trials
    )
    control_steps = sum(
        int(result.case_metrics.get("steps", 0)) for result in trials
    )
    solver_fallback_rate = solver_fallback_count / max(1, control_steps)
    secondary = (
        0.35 * min(progress_shortfall, 1.0)
        + 0.25 * clearance_shortfall / (1.0 + clearance_shortfall)
        + 0.15 * intervention / (1.0 + intervention)
        + 0.15 * timing / (1.0 + timing)
        + 0.10 * min(solver_fallback_rate, 1.0)
    )
    return float(failure_rank + secondary)


def evaluate_plcbf_config(
    config: HospitalConfig,
    *,
    cases: Sequence[str] = tuple(STRICT_HOSPITAL_CASES),
    seeds: Sequence[int] = (0,),
    steps: int = 1100,
) -> tuple[float, tuple[BenchmarkResult, ...]]:
    """Evaluate a proposed full-library configuration without creating a study."""

    results = run_hospital_benchmark(
        methods=("plcbf",),
        cases=cases,
        seeds=seeds,
        steps=steps,
        config=config,
        oracle_period_s=config.dt,
        compact_policy_library=False,
        progress=False,
    )
    return score_results(results), results


def objective(
    trial: Any,
    *,
    base: HospitalConfig = DEFAULT_CONFIG,
    cases: Sequence[str] = tuple(STRICT_HOSPITAL_CASES),
    seeds: Sequence[int] = (0,),
    steps: int = 1100,
) -> float:
    """Optuna objective callable over an explicit training split."""

    config = suggest_hospital_config(trial, base)
    score, results = evaluate_plcbf_config(
        config,
        cases=cases,
        seeds=seeds,
        steps=steps,
    )
    if hasattr(trial, "set_user_attr"):
        trial.set_user_attr(
            "collision_count",
            sum(result.collision for result in results),
        )
        trial.set_user_attr(
            "mean_progress",
            sum(
                float(item.case_metrics.get("progress", 0.0))
                for item in results
            )
            / len(results),
        )
        trial.set_user_attr(
            "outcomes",
            [result.outcome.value for result in results],
        )
        trial.set_user_attr("config", asdict(config))
    return float(score)


def study_configuration_fingerprint(
    config: HospitalTuningConfig,
) -> str:
    """Hash every setting that changes the training objective/search space."""

    payload = {
        "search_space_version": _SEARCH_SPACE_VERSION,
        "cases": list(config.cases),
        "train_seeds": list(config.train_seeds),
        "steps": config.steps,
        "sampler_seed": config.sampler_seed,
        "quick": config.quick,
        "base_config": asdict(config.base_config),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_study_configuration(
    study: Any,
    config: HospitalTuningConfig,
) -> str:
    """Bind a study to one objective configuration, rejecting unsafe resumes."""

    fingerprint = study_configuration_fingerprint(config)
    existing = study.user_attrs.get(_STUDY_FINGERPRINT_ATTRIBUTE)
    if existing is None:
        if study.trials:
            raise ValueError(
                "refusing to resume a legacy Optuna study without an objective "
                "configuration fingerprint; choose a new --study-name"
            )
        study.set_user_attr(_STUDY_FINGERPRINT_ATTRIBUTE, fingerprint)
        study.set_user_attr("search_space_version", _SEARCH_SPACE_VERSION)
    elif existing != fingerprint:
        raise ValueError(
            "Optuna study configuration does not match this training objective; "
            "choose a new --study-name or restore the original configuration"
        )
    return fingerprint


def _prepare_storage(storage: str | None) -> None:
    if storage is None or not storage.startswith("sqlite:///"):
        return
    raw_path = storage[len("sqlite:///") :]
    if raw_path and raw_path != ":memory:":
        Path(raw_path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def create_study(
    *,
    study_name: str = "hospital_plcbf",
    storage: str | None = None,
    seed: int = 0,
    load_if_exists: bool = True,
) -> "optuna.Study":
    """Create, but do not optimize, a reproducible Optuna study."""

    try:
        import optuna
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Hospital tuning requires the 'optuna' dependency."
        ) from exc
    _prepare_storage(storage)
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(seed)),
    )


def run_tuning(
    *,
    n_trials: int,
    timeout_s: float | None = None,
    study_name: str = "hospital_plcbf",
    storage: str | None = None,
    sampler_seed: int = 0,
    cases: Sequence[str] = tuple(STRICT_HOSPITAL_CASES),
    seeds: Sequence[int] = (0,),
    steps: int = 1100,
    base: HospitalConfig = DEFAULT_CONFIG,
) -> "optuna.Study":
    """Compatibility API that explicitly optimizes and returns a study."""

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    training_seeds = tuple(int(seed) for seed in seeds)
    validation_seed = 101
    if validation_seed in training_seeds:
        validation_seed = max(training_seeds) + 10_001
    config = HospitalTuningConfig(
        cases=tuple(cases),
        train_seeds=training_seeds,
        validation_seeds=(validation_seed,),
        steps=steps,
        n_trials=n_trials,
        timeout_s=timeout_s,
        sampler_seed=sampler_seed,
        study_name=study_name,
        storage=storage,
        base_config=base,
    )
    study = create_study(
        study_name=config.study_name,
        storage=config.storage,
        seed=config.sampler_seed,
    )
    bind_study_configuration(study, config)
    study.optimize(
        lambda trial: objective(
            trial,
            base=config.base_config,
            cases=config.cases,
            seeds=config.train_seeds,
            steps=config.steps,
        ),
        n_trials=config.n_trials,
        timeout=config.timeout_s,
        n_jobs=1,
    )
    return study


@dataclass(frozen=True)
class TuningRunResult:
    """Artifacts from an explicitly executed tuning and validation run."""

    study: Any
    best_config: HospitalConfig
    validation_results: tuple[BenchmarkResult, ...]
    validation_reports: BenchmarkReportPaths
    summary_path: Path


def _write_run_summary(
    path: Path,
    *,
    config: HospitalTuningConfig,
    study: Any,
    best_config: HospitalConfig,
    validation_results: tuple[BenchmarkResult, ...],
    reports: BenchmarkReportPaths,
) -> Path:
    payload = {
        "study": {
            "name": study.study_name,
            "best_value": float(study.best_value),
            "best_params": dict(sorted(study.best_params.items())),
            "completed_trials": len(study.trials),
            "objective_configuration_fingerprint": (
                study_configuration_fingerprint(config)
            ),
        },
        "configuration": config.metadata(),
        "best_config": asdict(best_config),
        "validation_score": score_results(validation_results),
        "validation_outcomes": [
            result.outcome.value for result in validation_results
        ],
        "validation_reports": {
            "csv": str(reports.csv),
            "json": str(reports.json),
            "markdown": str(reports.markdown),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def run_study(config: HospitalTuningConfig) -> TuningRunResult:
    """Create/resume, optimize, reconstruct, and validate one study."""

    study = create_study(
        study_name=config.study_name,
        storage=config.storage,
        seed=config.sampler_seed,
    )
    bind_study_configuration(study, config)
    study.optimize(
        lambda trial: objective(
            trial,
            base=config.base_config,
            cases=config.cases,
            seeds=config.train_seeds,
            steps=config.steps,
        ),
        n_trials=config.n_trials,
        timeout=config.timeout_s,
        n_jobs=1,
    )
    best_config = hospital_config_from_params(
        study.best_params,
        base=config.base_config,
    )
    _, validation_results = evaluate_plcbf_config(
        best_config,
        cases=config.cases,
        seeds=config.validation_seeds,
        steps=config.steps,
    )
    validation_prefix = config.output_prefix.with_name(
        config.output_prefix.name + "_validation"
    )
    validation_reports = write_benchmark_reports(
        validation_prefix,
        validation_results,
        metadata={
            **config.metadata(),
            "split": "held_out_validation",
            "best_params": dict(sorted(study.best_params.items())),
            "best_config": asdict(best_config),
        },
        title="Hospital tuned PL-CBF held-out validation",
    )
    summary_path = config.output_prefix.with_name(
        config.output_prefix.name + "_summary.json"
    )
    _write_run_summary(
        summary_path,
        config=config,
        study=study,
        best_config=best_config,
        validation_results=validation_results,
        reports=validation_reports,
    )
    return TuningRunResult(
        study=study,
        best_config=best_config,
        validation_results=validation_results,
        validation_reports=validation_reports,
        summary_path=summary_path,
    )


def write_tuning_summary(path: str | Path, study: "optuna.Study") -> Path:
    """Write the best trial from a study created through the compatibility API."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    best = study.best_trial
    payload = {
        "study_name": study.study_name,
        "best_value": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "trial_number": best.number,
        "trial_count": len(study.trials),
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="explicitly start or resume the Optuna study",
    )
    parser.add_argument("--trials", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--study-name")
    parser.add_argument(
        "--storage",
        default="sqlite:///results/hospital_optuna.db",
    )
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(STRICT_HOSPITAL_CASES),
        default=list(STRICT_HOSPITAL_CASES),
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--validation-seeds", nargs="+", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hospital_optuna"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="one trial, one step, one case, and disjoint single-seed splits",
    )
    return parser


def _config_from_args(arguments: argparse.Namespace) -> HospitalTuningConfig:
    quick = bool(arguments.quick)
    cases = tuple(arguments.cases[:1] if quick else arguments.cases)
    train_seeds = tuple(
        arguments.seeds
        if arguments.seeds
        else ((0,) if quick else (0, 1))
    )
    validation_seeds = tuple(
        arguments.validation_seeds
        if arguments.validation_seeds
        else ((101,) if quick else (101, 102))
    )
    base = DEFAULT_CONFIG
    if quick:
        base = replace(
            base,
            policies=replace(
                base.policies,
                num_angle_policies=4,
                room_policy_count=1,
            ),
        )
    return HospitalTuningConfig(
        cases=cases,
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        steps=(
            arguments.steps
            if arguments.steps is not None
            else (1 if quick else 1100)
        ),
        n_trials=(
            arguments.trials
            if arguments.trials is not None
            else (1 if quick else 50)
        ),
        timeout_s=arguments.timeout,
        sampler_seed=arguments.sampler_seed,
        study_name=(
            arguments.study_name
            or ("hospital_plcbf_quick" if quick else "hospital_plcbf")
        ),
        storage=None if arguments.storage == "none" else arguments.storage,
        output_prefix=arguments.output,
        quick=quick,
        base_config=base,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    config = _config_from_args(arguments)
    if not arguments.run:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "message": "Pass --run to start or resume Optuna tuning.",
                    "configuration": config.metadata(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = run_study(config)
    print(
        json.dumps(
            {
                "study": result.study.study_name,
                "best_value": result.study.best_value,
                "summary": str(result.summary_path.resolve()),
                "validation_reports": {
                    "csv": str(result.validation_reports.csv.resolve()),
                    "json": str(result.validation_reports.json.resolve()),
                    "markdown": str(
                        result.validation_reports.markdown.resolve()
                    ),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return int(
        any(
            item.outcome is BenchmarkOutcome.ERROR
            for item in result.validation_results
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HospitalTuningConfig",
    "TuningRunResult",
    "bind_study_configuration",
    "build_parser",
    "create_study",
    "evaluate_plcbf_config",
    "hospital_config_from_params",
    "main",
    "objective",
    "run_study",
    "run_tuning",
    "score_results",
    "study_configuration_fingerprint",
    "suggest_hospital_config",
    "write_tuning_summary",
]
