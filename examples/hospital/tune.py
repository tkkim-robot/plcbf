"""Reproducible Optuna tuning for the fixed Hospital story benchmark.

Importing this module creates no study and runs no trials. The CLI prints its
resolved configuration by default and requires ``--run`` before optimization.
Every completed trial covers the same canonical 100-world benchmark used for
selection and reporting.  The selected trial's exact archived rows are
reported directly; there is no held-out split or post-selection simulation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

from plcbf.benchmarking import (
    BenchmarkOutcome,
    BenchmarkReportPaths,
    BenchmarkResult,
    benchmark_report_from_json,
    results_to_json,
    write_benchmark_reports,
)

from .benchmark import (
    HOSPITAL_BENCHMARK_STORIES,
    PUBLICATION_MAX_SENSED_OBSTACLES,
    PUBLICATION_SENSING_RANGE_M,
    default_hospital_benchmark_steps,
    publication_benchmark_config,
    run_hospital_benchmark,
)
from .config import (
    DEFAULT_CONFIG,
    HospitalConfig,
    hospital_config_from_mapping,
)
from .scenarios import (
    DEFAULT_HOSPITAL_TRAFFIC_SEEDS,
    get_hospital_publication_trial,
    hospital_story_protocol_metadata,
)
from .reporting import write_hospital_benchmark_markdown
from .simulation import validate_strict_refuge_protocol

if TYPE_CHECKING:
    import optuna


_STUDY_FINGERPRINT_ATTRIBUTE = "objective_configuration_fingerprint"
_BASE_REFERENCE_ATTRIBUTE = "base_controller_reference_enqueued"
_RESULT_ARCHIVE_ATTRIBUTE = "exact_benchmark_result_archive"
_RESULT_ARCHIVE_SCHEMA = "hospital_optuna_benchmark_results_v1"
_OBJECTIVE_SCORE_VERSION = "success_first_lexicographic_v1"
_SEARCH_SPACE_VERSION = "hospital_plcbf_v10_100world_no_holdout_success_first"
_CANONICAL_BENCHMARK_SEEDS = tuple(DEFAULT_HOSPITAL_TRAFFIC_SEEDS)


def hospital_tuning_base_config(
    base: HospitalConfig = DEFAULT_CONFIG,
) -> HospitalConfig:
    """Normalize every non-tunable publication/evaluation invariant.

    Optuna may change controller gains but cannot weaken the scored safe set,
    truncate the fallback library, coarsen the room rollout, or enlarge the
    perception envelope.  Keeping the shape-defining fields fixed also avoids
    compiling a different JAX program for each trial.
    """

    published = publication_benchmark_config(base)
    reference = publication_benchmark_config(DEFAULT_CONFIG)
    return replace(
        published,
        robot=replace(
            published.robot,
            sensing_range=PUBLICATION_SENSING_RANGE_M,
        ),
        policies=replace(
            published.policies,
            num_angle_policies=reference.policies.num_angle_policies,
            room_policy_count=reference.policies.room_policy_count,
            room_rollout_dt=reference.policies.room_rollout_dt,
            room_horizon=reference.policies.room_horizon,
        ),
        safety=replace(
            published.safety,
            safety_margin=reference.safety.safety_margin,
            human_margin=reference.safety.human_margin,
            stretcher_margin=reference.safety.stretcher_margin,
            static_margin=reference.safety.static_margin,
            max_obstacles=PUBLICATION_MAX_SENSED_OBSTACLES,
        ),
        refuge=reference.refuge,
    )


_PUBLICATION_BASE_CONFIG = hospital_tuning_base_config(DEFAULT_CONFIG)
_DEFAULT_TUNING_STEPS = default_hospital_benchmark_steps(
    _PUBLICATION_BASE_CONFIG
)
DEFAULT_SAMPLER_SETTINGS: dict[str, object] = {
    "type": "trial_number_seeded_tpe",
    "seed_derivation": "sha256(base_seed,trial_number)",
    "n_jobs": 1,
}
DEFAULT_PRUNER_SETTINGS: dict[str, object] = {
    "type": "exact_world_prefix_patient_median",
    "direction": "minimize",
    "reference_trials": "complete_only",
    "prefix_metric": "success_first_task_score_without_runtime",
    "checkpoint_unit": "one_complete_five_story_seed_block",
    "n_startup_trials": 5,
    "n_warmup_world_steps": 20,
    "interval_world_steps": 5,
    "n_min_trials": 5,
    "patience_reports": 3,
    "earliest_prune_world": 30,
}
_TUNABLE_NAMES = frozenset(
    {
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
)


def hospital_tuning_search_space() -> dict[str, dict[str, object]]:
    """Return the JSON-compatible, shape-stable phase-one search space."""

    return {
        "cbf_alpha": {"low": 0.3, "high": 3.0, "log": True},
        "cbf_value_buffer": {
            "low": 0.25,
            "high": 1.05,
            "step": 0.05,
        },
        "component_temperature": {
            "low": 24.0,
            "high": 80.0,
            "log": True,
        },
        "hocbf_lambda1": {"low": 0.2, "high": 1.0},
        "hocbf_lambda2": {"low": 0.4, "high": 2.0},
        "max_gradient_norm": {
            "low": 30.0,
            "high": 200.0,
            "log": True,
        },
        "room_target_speed": {"low": 1.4, "high": 2.85},
        "stop_gain": {"low": 1.5, "high": 4.5},
        "time_temperature": {
            "low": 20.0,
            "high": 80.0,
            "log": True,
        },
    }


def ordered_hospital_tuning_worlds(
    cases: Sequence[str],
    seeds: Sequence[int],
) -> tuple[tuple[str, int], ...]:
    """Return seed-major story round-robin order for balanced prefixes."""

    return tuple(
        (str(case), int(seed))
        for seed in seeds
        for case in cases
    )


@dataclass(frozen=True)
class HospitalTuningConfig:
    """One fixed optimization/reporting grid and Optuna execution settings."""

    cases: tuple[str, ...] = tuple(HOSPITAL_BENCHMARK_STORIES)
    benchmark_seeds: tuple[int, ...] = _CANONICAL_BENCHMARK_SEEDS
    steps: int = _DEFAULT_TUNING_STEPS
    n_trials: int = 50
    timeout_s: float | None = None
    sampler_seed: int = 0
    study_name: str = "hospital_plcbf_100"
    storage: str | None = "sqlite:///results/hospital_optuna_100.db"
    output_prefix: Path = Path("results/hospital_optuna_100")
    quick: bool = False
    base_config: HospitalConfig = field(
        default_factory=lambda: _PUBLICATION_BASE_CONFIG
    )

    def __post_init__(self) -> None:
        base_config = hospital_tuning_base_config(self.base_config)
        validate_strict_refuge_protocol(base_config)
        cases = tuple(str(case) for case in self.cases)
        raw_benchmark_seeds = tuple(self.benchmark_seeds)
        if any(
            isinstance(seed, bool) or int(seed) != seed
            for seed in raw_benchmark_seeds
        ):
            raise ValueError(
                "hospital traffic seeds must be integers from 0 to 19"
            )
        benchmark_seeds = tuple(int(seed) for seed in raw_benchmark_seeds)
        if not cases:
            raise ValueError("at least one hospital tuning case is required")
        unknown = set(cases) - set(HOSPITAL_BENCHMARK_STORIES)
        if unknown:
            raise ValueError(
                "unknown hospital tuning stories: " + ", ".join(sorted(unknown))
            )
        if len(set(cases)) != len(cases):
            raise ValueError(
                "hospital tuning stories must not contain duplicates"
            )
        if not benchmark_seeds:
            raise ValueError("at least one benchmark seed is required")
        allowed_seeds = set(DEFAULT_HOSPITAL_TRAFFIC_SEEDS)
        invalid_seeds = set(benchmark_seeds) - allowed_seeds
        if invalid_seeds:
            raise ValueError(
                "hospital traffic seeds must be integers from 0 to 19; got "
                + ", ".join(str(seed) for seed in sorted(invalid_seeds))
            )
        if len(set(benchmark_seeds)) != len(benchmark_seeds):
            raise ValueError("hospital benchmark seeds must not contain duplicates")
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
        object.__setattr__(self, "benchmark_seeds", benchmark_seeds)
        object.__setattr__(self, "sampler_seed", int(self.sampler_seed))
        object.__setattr__(self, "study_name", study_name)
        object.__setattr__(self, "output_prefix", output_prefix)
        object.__setattr__(self, "quick", bool(self.quick))
        object.__setattr__(self, "base_config", base_config)

    def metadata(self) -> dict[str, object]:
        protocol = hospital_story_protocol_metadata()
        return {
            "case_study": "hospital_refuge",
            "method": "plcbf",
            "cases": list(self.cases),
            "benchmark_seeds": list(self.benchmark_seeds),
            "steps": self.steps,
            "n_trials": self.n_trials,
            "target_terminal_trial_count": self.n_trials,
            "terminal_trial_states": ["complete", "pruned", "fail"],
            "base_controller_reference": (
                "enqueued_once_as_trial_zero_for_non_quick_fresh_study"
            ),
            "timeout_s": self.timeout_s,
            "sampler_seed": self.sampler_seed,
            "sampler": dict(DEFAULT_SAMPLER_SETTINGS),
            "study_name": self.study_name,
            "storage": self.storage,
            "quick": self.quick,
            "policy_library": "full",
            "benchmark_world_order": "seed_major_story_round_robin",
            "objective_score_version": _OBJECTIVE_SCORE_VERSION,
            "search_space_version": _SEARCH_SPACE_VERSION,
            "tunable_parameters": hospital_tuning_search_space(),
            "fixed_tuning_envelope": {
                "sensing_range_m": self.base_config.robot.sensing_range,
                "max_sensed_obstacles": (
                    self.base_config.safety.max_obstacles
                ),
                "safety_margin_m": self.base_config.safety.safety_margin,
                "human_margin_m": self.base_config.safety.human_margin,
                "stretcher_margin_m": (
                    self.base_config.safety.stretcher_margin
                ),
                "static_margin_m": self.base_config.safety.static_margin,
                "num_angle_policies": (
                    self.base_config.policies.num_angle_policies
                ),
                "room_policy_count": (
                    self.base_config.policies.room_policy_count
                ),
                "room_rollout_dt_s": (
                    self.base_config.policies.room_rollout_dt
                ),
                "room_horizon_s": self.base_config.policies.room_horizon,
                "refuge_geometry": asdict(self.base_config.refuge),
            },
            "oracle_period": "every_plant_step",
            "oracle_period_s": self.base_config.dt,
            "seed_zero_is_exact_reference": False,
            "external_refuge_state_machine": False,
            "protocol_kind": (
                "canonical_full_100_world_optimization_and_reporting_grid"
                if self.cases == tuple(HOSPITAL_BENCHMARK_STORIES)
                and self.benchmark_seeds == _CANONICAL_BENCHMARK_SEEDS
                and not self.quick
                else "explicit_custom_optimization_and_reporting_grid"
            ),
            "held_out_validation": False,
            "post_selection_simulation": False,
            "selection_grid_is_reported_grid": True,
            "reported_results_source": (
                "selected_complete_trial_exact_archive_no_rerun"
            ),
            "hospital_story_protocol": protocol,
            "source_content_sha256": _relevant_source_content_sha256(),
            "scenario_grid_sha256": {
                "benchmark": scenario_grid_fingerprint(
                    self.cases, self.benchmark_seeds
                ),
            },
            "publication_sensor_capacity": (
                self.base_config.safety.max_obstacles
            ),
            "publication_sensing_range_m": (
                self.base_config.robot.sensing_range
            ),
            "expected_benchmark_world_count": (
                len(self.cases) * len(self.benchmark_seeds)
            ),
            "completed_trial_requires_full_benchmark_grid": True,
            "optimization_objective_includes_runtime": False,
            "pruner": (
                {"type": "none"}
                if self.quick
                else dict(DEFAULT_PRUNER_SETTINGS)
            ),
        }


def scenario_grid_fingerprint(
    cases: Sequence[str], seeds: Sequence[int]
) -> str:
    """Fingerprint an ordered story/traffic-seed grid and its protocol."""

    protocol = hospital_story_protocol_metadata()
    trials = []
    for case, seed in ordered_hospital_tuning_worlds(cases, seeds):
        trial = get_hospital_publication_trial(case, seed)
        trials.append(
            {
                "case_id": trial.case_id,
                "ordinal": trial.ordinal,
                "generator_seed": trial.generator_seed,
            }
        )
    payload = {
        "protocol_sha256": protocol["protocol_sha256"],
        "geometry_sha256": protocol["geometry_sha256"],
        "trials": trials,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def suggest_hospital_config(
    trial: Any,
    base: HospitalConfig = _PUBLICATION_BASE_CONFIG,
) -> HospitalConfig:
    """Materialize one internally consistent PL-CBF configuration."""

    base = hospital_tuning_base_config(base)
    policies = replace(
        base.policies,
        room_target_speed=trial.suggest_float(
            "room_target_speed", 1.4, 2.85
        ),
        stop_gain=trial.suggest_float("stop_gain", 1.5, 4.5),
        cbf_alpha=trial.suggest_float("cbf_alpha", 0.3, 3.0, log=True),
        cbf_value_buffer=trial.suggest_float(
            "cbf_value_buffer", 0.25, 1.05, step=0.05
        ),
        component_temperature=trial.suggest_float(
            "component_temperature", 24.0, 80.0, log=True
        ),
        time_temperature=trial.suggest_float(
            "time_temperature", 20.0, 80.0, log=True
        ),
        max_gradient_norm=trial.suggest_float(
            "max_gradient_norm", 30.0, 200.0, log=True
        ),
    )
    safety = replace(
        base.safety,
        hocbf_lambda1=trial.suggest_float(
            "hocbf_lambda1", 0.20, 1.00
        ),
        hocbf_lambda2=trial.suggest_float(
            "hocbf_lambda2", 0.40, 2.00
        ),
    )
    config = replace(base, policies=policies, safety=safety)
    validate_strict_refuge_protocol(config)
    return config


def hospital_config_from_params(
    params: Mapping[str, Any],
    *,
    base: HospitalConfig = _PUBLICATION_BASE_CONFIG,
) -> HospitalConfig:
    """Reconstruct the best hospital configuration from Optuna parameters."""

    base = hospital_tuning_base_config(base)
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
        "room_target_speed",
        "stop_gain",
        "cbf_alpha",
        "cbf_value_buffer",
        "component_temperature",
        "time_temperature",
        "max_gradient_norm",
    }
    safety_names = {
        "hocbf_lambda1",
        "hocbf_lambda2",
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
    )
    validate_strict_refuge_protocol(config)
    return config


def score_results(results: Iterable[BenchmarkResult]) -> float:
    """Return a deterministic success-first task score; lower is better.

    Wall-clock timing is intentionally excluded.  It remains available in the
    raw benchmark rows and trial diagnostics, but JIT warm-up and host load must
    never change the selected controller.  The integer portion is
    lexicographic: maximize success count first, then minimize errors,
    collisions, timeouts, operational violations, and terminal deadlock.
    """

    trials = tuple(results)
    if not trials:
        raise ValueError("at least one benchmark result is required")
    errors = sum(
        result.outcome is BenchmarkOutcome.ERROR for result in trials
    )
    collisions = sum(
        result.outcome is BenchmarkOutcome.COLLISION for result in trials
    )
    timeouts = sum(
        result.outcome is BenchmarkOutcome.TIMEOUT for result in trials
    )
    invalid_legacy_outcomes = sum(
        result.outcome is BenchmarkOutcome.INFEASIBLE for result in trials
    )
    operational_violations = sum(
        bool(
            result.case_metrics.get(
                "operational_safety_violation", False
            )
        )
        for result in trials
    )
    deadlocked_at_end = sum(
        bool(result.case_metrics.get("deadlocked_at_end", False))
        for result in trials
    )
    base = len(trials) + 1
    non_successes = (
        errors + invalid_legacy_outcomes + collisions + timeouts
    )
    failure_rank = (
        non_successes * base**5
        + (errors + invalid_legacy_outcomes) * base**4
        + collisions * base**3
        + timeouts * base**2
        + operational_violations * base
        + deadlocked_at_end
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
                max(
                    0.0,
                    -float(
                        result.case_metrics.get(
                            "minimum_safety_clearance",
                            result.min_clearance,
                        )
                    ),
                )
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
        + 0.10 * min(solver_fallback_rate, 1.0)
    )
    return float(failure_rank + secondary)


def evaluate_plcbf_config(
    config: HospitalConfig,
    *,
    cases: Sequence[str] = tuple(HOSPITAL_BENCHMARK_STORIES),
    seeds: Sequence[int] = _CANONICAL_BENCHMARK_SEEDS,
    steps: int = _DEFAULT_TUNING_STEPS,
) -> tuple[float, tuple[BenchmarkResult, ...]]:
    """Evaluate a proposed full-library configuration without creating a study."""

    config = hospital_tuning_base_config(config)
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
    base: HospitalConfig = _PUBLICATION_BASE_CONFIG,
    cases: Sequence[str] = tuple(HOSPITAL_BENCHMARK_STORIES),
    seeds: Sequence[int] = _CANONICAL_BENCHMARK_SEEDS,
    steps: int = _DEFAULT_TUNING_STEPS,
    result_archive_dir: str | Path | None = None,
) -> float:
    """Evaluate ordered worlds, allowing pruning only between complete worlds.

    The returned objective is always the runtime-neutral
    :func:`score_results` value over the complete Cartesian grid.
    """

    if result_archive_dir is None:
        raise ValueError(
            "result_archive_dir is required so a completed Optuna trial can "
            "be reported without rerunning the benchmark"
        )
    config = suggest_hospital_config(trial, base)
    case_values = tuple(str(case) for case in cases)
    seed_values = tuple(int(seed) for seed in seeds)
    expected_world_count = len(case_values) * len(seed_values)
    if expected_world_count < 1:
        raise ValueError("the Hospital tuning grid must not be empty")
    ordered_worlds = ordered_hospital_tuning_worlds(
        case_values,
        seed_values,
    )
    expected_case_ids = [
        f"{case}/seed-{seed}"
        for case, seed in ordered_worlds
    ]
    benchmark_grid_sha256 = scenario_grid_fingerprint(
        case_values,
        seed_values,
    )
    _set_trial_attr(
        trial, "benchmark_grid_sha256", benchmark_grid_sha256
    )
    _set_trial_attr(
        trial, "expected_case_ids", expected_case_ids
    )
    _set_trial_attr(
        trial, "expected_complete_world_count", expected_world_count
    )

    results: list[BenchmarkResult] = []
    report = getattr(trial, "report", None)
    should_prune = getattr(trial, "should_prune", None)
    for case, seed in ordered_worlds:
        world_results = run_hospital_benchmark(
            methods=("plcbf",),
            cases=(case,),
            seeds=(seed,),
            steps=steps,
            config=config,
            oracle_period_s=config.dt,
            compact_policy_library=False,
            progress=False,
        )
        result = _validate_single_world_result(
            world_results,
            case=case,
            seed=seed,
        )
        results.append(result)
        world_index = len(results)
        evaluated_case_ids = [item.case_id for item in results]
        _set_trial_attr(
            trial, "evaluated_case_ids", evaluated_case_ids
        )
        _set_trial_attr(
            trial, "evaluated_world_count", world_index
        )
        completed_story_block = world_index % len(case_values) == 0
        prefix_score = _deterministic_prefix_score(results)
        if completed_story_block and callable(report):
            report(prefix_score, step=world_index)
        if (
            world_index < expected_world_count
            and completed_story_block
            and callable(should_prune)
            and should_prune()
        ):
            _set_trial_attr(trial, "config", asdict(config))
            _set_trial_attr(
                trial, "partial_prefix_score", prefix_score
            )
            _set_trial_attr(
                trial,
                "expected_complete_world_count",
                expected_world_count,
            )
            optuna = _require_optuna()
            raise optuna.TrialPruned(
                "pruned after "
                f"{world_index}/{expected_world_count} complete worlds"
            )

    completed = tuple(results)
    if len(completed) != expected_world_count:
        raise RuntimeError("completed Hospital trial did not cover its full grid")
    score = score_results(completed)
    _set_trial_attr(
        trial,
        "success_count",
        sum(result.success for result in completed),
    )
    _set_trial_attr(
        trial,
        "collision_count",
        sum(result.collision for result in completed),
    )
    _set_trial_attr(
        trial,
        "timeout_count",
        sum(result.timeout for result in completed),
    )
    _set_trial_attr(
        trial,
        "error_count",
        sum(result.outcome is BenchmarkOutcome.ERROR for result in completed),
    )
    _set_trial_attr(
        trial,
        "mean_progress",
        sum(
            float(item.case_metrics.get("progress", 0.0))
            for item in completed
        )
        / len(completed),
    )
    _set_trial_attr(
        trial,
        "outcomes",
        [result.outcome.value for result in completed],
    )
    _set_trial_attr(trial, "config", asdict(config))
    _set_trial_attr(trial, "evaluated_world_count", len(completed))
    _set_trial_attr(
        trial,
        "evaluated_case_ids",
        [item.case_id for item in completed],
    )
    _set_trial_attr(trial, "completed_world_count", len(completed))
    _set_trial_attr(
        trial, "expected_complete_world_count", expected_world_count
    )
    archive = _write_exact_result_archive(
        result_archive_dir,
        trial=trial,
        config=config,
        results=completed,
        expected_case_ids=expected_case_ids,
        benchmark_grid_sha256=benchmark_grid_sha256,
        score=score,
    )
    _set_trial_attr(trial, _RESULT_ARCHIVE_ATTRIBUTE, archive)
    _set_trial_attr(trial, "completed_full_benchmark_grid", True)
    _set_trial_attr(trial, "final_full_grid_score", float(score))
    timing_totals = [
        float(
            result.case_metrics.get(
                "oracle_and_solver_time_total_s",
                0.0,
            )
        )
        for result in completed
    ]
    _set_trial_attr(
        trial,
        "mean_oracle_and_solver_time_total_s",
        float(np.mean(timing_totals)),
    )
    _set_trial_attr(
        trial,
        "sum_oracle_and_solver_time_total_s",
        float(np.sum(timing_totals)),
    )
    return float(score)


def _set_trial_attr(trial: Any, name: str, value: object) -> None:
    setter = getattr(trial, "set_user_attr", None)
    if callable(setter):
        setter(name, value)


def _write_exact_result_archive(
    directory: str | Path,
    *,
    trial: Any,
    config: HospitalConfig,
    results: Sequence[BenchmarkResult],
    expected_case_ids: Sequence[str],
    benchmark_grid_sha256: str,
    score: float,
) -> dict[str, object]:
    """Atomically archive one completed trial's exact benchmark rows."""

    try:
        trial_number = int(trial.number)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "completed Optuna trials require an integer trial number for "
            "their exact result archive"
        ) from error
    rows = tuple(results)
    identities = [result.case_id for result in rows]
    if identities != list(expected_case_ids):
        raise RuntimeError(
            "cannot archive a completed trial whose rows are not in the "
            "declared seed-major benchmark order"
        )
    metadata = {
        "archive_schema": _RESULT_ARCHIVE_SCHEMA,
        "benchmark_grid_sha256": str(benchmark_grid_sha256),
        "config": asdict(config),
        "objective_score": float(score),
        "objective_score_version": _OBJECTIVE_SCORE_VERSION,
        "ordered_case_ids": list(expected_case_ids),
        "params": dict(sorted(getattr(trial, "params", {}).items())),
        "trial_number": trial_number,
    }
    document = results_to_json(rows, metadata=metadata, indent=0).encode(
        "utf-8"
    )
    digest = hashlib.sha256(document).hexdigest()
    destination_directory = Path(directory).expanduser().resolve()
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / f"trial-{trial_number:04d}.json"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination_directory,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists():
            if destination.read_bytes() != document:
                raise RuntimeError(
                    f"refusing to overwrite conflicting trial archive "
                    f"{destination}"
                )
            temporary_path.unlink()
            temporary_path = None
        else:
            os.replace(temporary_path, destination)
            temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return {
        "byte_count": len(document),
        "path": str(destination),
        "result_count": len(rows),
        "schema": _RESULT_ARCHIVE_SCHEMA,
        "sha256": digest,
    }


def _load_exact_result_archive(
    trial: Any,
    *,
    expected_case_ids: Sequence[str],
    expected_grid_sha256: str,
    expected_config: HospitalConfig,
) -> tuple[BenchmarkResult, ...]:
    """Load and strictly verify a completed trial's immutable result rows."""

    attrs = dict(getattr(trial, "user_attrs", {}))
    descriptor = attrs.get(_RESULT_ARCHIVE_ATTRIBUTE)
    expected_descriptor_keys = {
        "byte_count",
        "path",
        "result_count",
        "schema",
        "sha256",
    }
    if not isinstance(descriptor, Mapping) or set(descriptor) != (
        expected_descriptor_keys
    ):
        raise RuntimeError("trial lacks a complete exact-result archive descriptor")
    if descriptor.get("schema") != _RESULT_ARCHIVE_SCHEMA:
        raise RuntimeError("trial result archive uses an unknown schema")
    path = Path(str(descriptor.get("path")))
    try:
        document = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"cannot read trial result archive {path}") from error
    if len(document) != descriptor.get("byte_count"):
        raise RuntimeError("trial result archive byte count does not match")
    if hashlib.sha256(document).hexdigest() != descriptor.get("sha256"):
        raise RuntimeError("trial result archive SHA-256 does not match")
    report = benchmark_report_from_json(document, source=str(path))
    metadata = dict(report.metadata)
    expected_metadata = {
        "archive_schema": _RESULT_ARCHIVE_SCHEMA,
        "benchmark_grid_sha256": expected_grid_sha256,
        "config": asdict(expected_config),
        "objective_score": float(trial.value),
        "objective_score_version": _OBJECTIVE_SCORE_VERSION,
        "ordered_case_ids": list(expected_case_ids),
        "params": dict(sorted(getattr(trial, "params", {}).items())),
        "trial_number": int(trial.number),
    }
    if _json_compatible(metadata) != _json_compatible(expected_metadata):
        raise RuntimeError("trial result archive metadata does not match the trial")
    rows_by_case_id: dict[str, BenchmarkResult] = {}
    for result in report.results:
        if result.algorithm != "plcbf" or result.case_id in rows_by_case_id:
            raise RuntimeError(
                "trial result archive has a wrong method or duplicate world"
            )
        rows_by_case_id[result.case_id] = result
    if set(rows_by_case_id) != set(expected_case_ids):
        raise RuntimeError("trial result archive does not cover the exact grid")
    ordered = tuple(rows_by_case_id[case_id] for case_id in expected_case_ids)
    if len(ordered) != descriptor.get("result_count"):
        raise RuntimeError("trial result archive row count does not match")
    if [result.outcome.value for result in ordered] != attrs.get("outcomes"):
        raise RuntimeError("trial result archive outcomes do not match")
    recomputed_score = score_results(ordered)
    if not np.isclose(
        recomputed_score,
        float(trial.value),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("trial result archive score does not match")
    return ordered


def _validate_single_world_result(
    results: Sequence[BenchmarkResult],
    *,
    case: str,
    seed: int,
) -> BenchmarkResult:
    rows = tuple(results)
    expected_case_id = f"{case}/seed-{seed}"
    if len(rows) != 1:
        raise RuntimeError(
            "each Hospital tuning checkpoint must return exactly one world"
        )
    result = rows[0]
    if (
        result.algorithm != "plcbf"
        or result.case_id != expected_case_id
        or result.seed != seed
    ):
        raise RuntimeError(
            "Hospital tuning checkpoint returned the wrong method/story/seed: "
            f"expected plcbf {expected_case_id}, got "
            f"{result.algorithm} {result.case_id} seed={result.seed}"
        )
    return result


def _deterministic_prefix_score(
    results: Sequence[BenchmarkResult],
) -> float:
    """Return the normal safety/task score with measured runtime neutralized."""

    deterministic = tuple(
        replace(
            result,
            case_metrics={
                **dict(result.case_metrics),
                "oracle_and_solver_time_total_s": 0.0,
            },
        )
        for result in results
    )
    return score_results(deterministic)


def _require_optuna() -> Any:
    try:
        import optuna
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Hospital tuning requires the 'optuna' dependency."
        ) from exc
    return optuna


class ExactWorldPrefixPatientMedianPruner:
    """Deterministically compare identical completed-world prefixes.

    Only COMPLETE reference trials contribute.  Patience is reconstructed from
    the current trial's intermediate values, so pruning decisions carry no
    mutable process-local state and remain stable across study resumes.
    """

    def __init__(
        self,
        *,
        n_startup_trials: int,
        n_warmup_steps: int,
        interval_steps: int,
        n_min_trials: int,
        patience: int,
    ) -> None:
        if int(n_startup_trials) < 0:
            raise ValueError("n_startup_trials must be nonnegative")
        if int(n_warmup_steps) < 0:
            raise ValueError("n_warmup_steps must be nonnegative")
        if int(interval_steps) < 1:
            raise ValueError("interval_steps must be positive")
        if int(n_min_trials) < 1:
            raise ValueError("n_min_trials must be positive")
        if int(patience) < 1:
            raise ValueError("patience must be positive")
        self.n_startup_trials = int(n_startup_trials)
        self.n_warmup_steps = int(n_warmup_steps)
        self.interval_steps = int(interval_steps)
        self.n_min_trials = int(n_min_trials)
        self.patience = int(patience)

    @staticmethod
    def _state_name(trial: Any) -> str:
        state = getattr(trial, "state", None)
        name = getattr(state, "name", state)
        return str(name).rsplit(".", 1)[-1].upper()

    @staticmethod
    def _study_trials(study: Any) -> tuple[Any, ...]:
        get_trials = getattr(study, "get_trials", None)
        if callable(get_trials):
            return tuple(get_trials(deepcopy=False))
        return tuple(getattr(study, "trials", ()))

    @staticmethod
    def _require_minimize(study: Any) -> None:
        try:
            direction = study.direction
        except (AttributeError, RuntimeError) as error:
            raise ValueError(
                "exact world-prefix pruning requires a single-objective "
                "minimize study"
            ) from error
        name = getattr(direction, "name", direction)
        if str(name).rsplit(".", 1)[-1].upper() != "MINIMIZE":
            raise ValueError(
                "exact world-prefix pruning requires minimize direction"
            )

    def prune(self, study: Any, trial: Any) -> bool:
        """Prune after a patient run of exact-prefix median losses."""

        self._require_minimize(study)
        intermediate = dict(getattr(trial, "intermediate_values", {}))
        if not intermediate:
            return False
        last_step = getattr(trial, "last_step", None)
        if last_step is None:
            last_step = max(intermediate)
        step = int(last_step)
        if step < self.n_warmup_steps:
            return False
        if (step - self.n_warmup_steps) % self.interval_steps != 0:
            return False

        candidate_attrs = dict(getattr(trial, "user_attrs", {}))
        grid_sha256 = candidate_attrs.get("benchmark_grid_sha256")
        expected_case_ids = candidate_attrs.get("expected_case_ids")
        evaluated_case_ids = candidate_attrs.get("evaluated_case_ids")
        study_attrs = dict(getattr(study, "user_attrs", {}))
        bound_case_ids = study_attrs.get("ordered_benchmark_case_ids")
        bound_grid_hashes = study_attrs.get("scenario_grid_sha256", {})
        bound_grid_sha256 = (
            bound_grid_hashes.get("benchmark")
            if isinstance(bound_grid_hashes, Mapping)
            else None
        )
        if not (
            isinstance(grid_sha256, str)
            and grid_sha256
            and isinstance(expected_case_ids, list)
            and all(isinstance(item, str) for item in expected_case_ids)
            and expected_case_ids == bound_case_ids
            and grid_sha256 == bound_grid_sha256
            and isinstance(evaluated_case_ids, list)
            and evaluated_case_ids == expected_case_ids[:step]
            and len(evaluated_case_ids) == step
            and candidate_attrs.get("evaluated_world_count") == step
            and candidate_attrs.get("expected_complete_world_count")
            == len(expected_case_ids)
        ):
            return False

        def audited_complete_reference(item: Any) -> bool:
            attrs = dict(getattr(item, "user_attrs", {}))
            return bool(
                self._state_name(item) == "COMPLETE"
                and attrs.get("completed_full_benchmark_grid") is True
                and attrs.get("benchmark_grid_sha256") == grid_sha256
                and attrs.get("expected_case_ids") == expected_case_ids
                and attrs.get("evaluated_case_ids") == expected_case_ids
                and attrs.get("evaluated_world_count")
                == len(expected_case_ids)
                and attrs.get("completed_world_count")
                == len(expected_case_ids)
                and attrs.get("expected_complete_world_count")
                == len(expected_case_ids)
            )

        completed = tuple(
            item
            for item in self._study_trials(study)
            if audited_complete_reference(item)
        )
        if len(completed) < self.n_startup_trials:
            return False

        first_step = step - (self.patience - 1) * self.interval_steps
        if first_step < self.n_warmup_steps:
            return False
        checkpoints = range(first_step, step + 1, self.interval_steps)
        for checkpoint in checkpoints:
            candidate = intermediate.get(checkpoint)
            if candidate is None or not np.isfinite(float(candidate)):
                return False
            references = [
                float(item.intermediate_values[checkpoint])
                for item in completed
                if checkpoint in getattr(item, "intermediate_values", {})
                and np.isfinite(float(item.intermediate_values[checkpoint]))
            ]
            if len(references) < self.n_min_trials:
                return False
            if float(candidate) <= float(np.median(references)):
                return False
        return True


def build_pruner(*, quick: bool = False) -> Any:
    """Build the deterministic exact-world-prefix Optuna pruner."""

    optuna = _require_optuna()
    if quick:
        return optuna.pruners.NopPruner()
    return ExactWorldPrefixPatientMedianPruner(
        n_startup_trials=int(DEFAULT_PRUNER_SETTINGS["n_startup_trials"]),
        n_warmup_steps=int(
            DEFAULT_PRUNER_SETTINGS["n_warmup_world_steps"]
        ),
        interval_steps=int(
            DEFAULT_PRUNER_SETTINGS["interval_world_steps"]
        ),
        n_min_trials=int(DEFAULT_PRUNER_SETTINGS["n_min_trials"]),
        patience=int(DEFAULT_PRUNER_SETTINGS["patience_reports"]),
    )


class TrialNumberSeededTPESampler:
    """Restart-stable TPE sampler using only Optuna's public sampler API.

    Optuna deliberately does not persist a sampler object's RNG state in study
    storage.  Reconstructing a normally seeded ``TPESampler`` therefore repeats
    its initial random stream after a process restart.  This adapter gives each
    trial its own TPE instance whose seed is a pure function of the declared
    sampler seed and Optuna trial number.  Given the same completed history,
    uninterrupted and resumed studies consequently make identical suggestions.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._delegates: dict[int, Any] = {}

    def _trial_seed(self, trial_number: int) -> int:
        payload = json.dumps(
            {
                "schema": "hospital_trial_number_tpe_v1",
                "base_seed": self.seed,
                "trial_number": int(trial_number),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        # Optuna accepts a NumPy-compatible unsigned 32-bit seed.
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")

    def _delegate(self, trial: Any) -> Any:
        number = int(trial.number)
        delegate = self._delegates.get(number)
        if delegate is None:
            optuna = _require_optuna()
            delegate = optuna.samplers.TPESampler(
                seed=self._trial_seed(number)
            )
            self._delegates[number] = delegate
        return delegate

    def infer_relative_search_space(
        self, study: Any, trial: Any
    ) -> dict[str, Any]:
        return self._delegate(trial).infer_relative_search_space(study, trial)

    def sample_relative(
        self,
        study: Any,
        trial: Any,
        search_space: dict[str, Any],
    ) -> dict[str, Any]:
        return self._delegate(trial).sample_relative(
            study,
            trial,
            search_space,
        )

    def sample_independent(
        self,
        study: Any,
        trial: Any,
        param_name: str,
        param_distribution: Any,
    ) -> Any:
        return self._delegate(trial).sample_independent(
            study,
            trial,
            param_name,
            param_distribution,
        )

    def before_trial(self, study: Any, trial: Any) -> None:
        self._delegate(trial).before_trial(study, trial)

    def after_trial(
        self,
        study: Any,
        trial: Any,
        state: Any,
        values: Sequence[float] | None,
    ) -> None:
        number = int(trial.number)
        delegate = self._delegate(trial)
        delegate.after_trial(study, trial, state, values)
        self._delegates.pop(number, None)

    def reseed_rng(self) -> None:
        # Hospital optimization is intentionally single-worker.  Seeds are
        # already namespaced by trial number, so process-local reseeding would
        # break restart equivalence.
        return None


def build_sampler(*, seed: int = 0) -> TrialNumberSeededTPESampler:
    """Build the restart-stable sampler used by Hospital studies."""

    _require_optuna()
    return TrialNumberSeededTPESampler(seed=seed)


def _relevant_source_content_sha256() -> dict[str, str]:
    """Hash implementation sources whose changes invalidate a study resume."""

    directory = Path(__file__).resolve().parent
    repository = directory.parent.parent
    sources = {
        "objective": Path(__file__).resolve(),
        "hospital_config": directory / "config.py",
        "controller": directory / "controller.py",
        "policies": directory / "policies.py",
        "dynamics": directory / "dynamics.py",
        "environment": directory / "environment.py",
        "jax_rollout": directory / "jax_rollout.py",
        "obstacles": directory / "obstacles.py",
        "planner": directory / "planner.py",
        "benchmark": directory / "benchmark.py",
        "simulation": directory / "simulation.py",
        "scenario_generation": directory / "scenario_generation.py",
        "scenarios": directory / "scenarios.py",
        "core_policy_library": repository / "plcbf" / "policy_library.py",
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sorted(sources.items())
    }


def study_configuration_fingerprint(
    config: HospitalTuningConfig,
) -> str:
    """Hash every setting that changes the benchmark objective/search space."""

    metadata = config.metadata()
    payload = {
        "search_space_version": _SEARCH_SPACE_VERSION,
        "objective_score_version": _OBJECTIVE_SCORE_VERSION,
        "search_space": metadata["tunable_parameters"],
        "fixed_tuning_envelope": metadata["fixed_tuning_envelope"],
        "benchmark_world_order": metadata["benchmark_world_order"],
        "cases": list(config.cases),
        "benchmark_seeds": list(config.benchmark_seeds),
        "hospital_story_protocol": metadata["hospital_story_protocol"],
        "benchmark_scenario_grid_sha256": scenario_grid_fingerprint(
            config.cases, config.benchmark_seeds
        ),
        "steps": config.steps,
        "sampler_seed": config.sampler_seed,
        "sampler": dict(DEFAULT_SAMPLER_SETTINGS),
        "quick": config.quick,
        "pruner": (
            {"type": "none"}
            if config.quick
            else dict(DEFAULT_PRUNER_SETTINGS)
        ),
        "base_config": asdict(config.base_config),
        "source_content_sha256": metadata["source_content_sha256"],
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
    metadata = config.metadata()
    audit_attributes = {
        "search_space_version": _SEARCH_SPACE_VERSION,
        "objective_score_version": _OBJECTIVE_SCORE_VERSION,
        "search_space": metadata["tunable_parameters"],
        "fixed_tuning_envelope": metadata["fixed_tuning_envelope"],
        "benchmark_world_order": metadata["benchmark_world_order"],
        "hospital_story_protocol_sha256": metadata[
            "hospital_story_protocol"
        ]["protocol_sha256"],
        "scenario_grid_sha256": metadata["scenario_grid_sha256"],
        "pruner": metadata["pruner"],
        "sampler": metadata["sampler"],
        "source_content_sha256": metadata["source_content_sha256"],
        "ordered_benchmark_case_ids": [
            f"{case}/seed-{seed}"
            for case, seed in ordered_hospital_tuning_worlds(
                config.cases,
                config.benchmark_seeds,
            )
        ],
    }
    existing = study.user_attrs.get(_STUDY_FINGERPRINT_ATTRIBUTE)
    if existing is None:
        if study.trials:
            raise ValueError(
                "refusing to resume a legacy Optuna study without an objective "
                "configuration fingerprint; choose a new --study-name"
            )
        study.set_user_attr(_STUDY_FINGERPRINT_ATTRIBUTE, fingerprint)
    elif existing != fingerprint:
        raise ValueError(
            "Optuna study configuration does not match this training objective; "
            "choose a new --study-name or restore the original configuration"
        )
    for name, value in audit_attributes.items():
        recorded = study.user_attrs.get(name)
        if recorded is None:
            study.set_user_attr(name, value)
        elif recorded != value:
            raise ValueError(
                f"Optuna study audit attribute {name!r} does not match this "
                "training objective; choose a new --study-name"
            )
    return fingerprint


def _trial_archive_directory(
    config: HospitalTuningConfig,
    fingerprint: str,
) -> Path:
    """Return the fingerprint-namespaced exact-result archive directory."""

    return config.output_prefix.with_name(
        f"{config.output_prefix.name}_trials_{fingerprint[:16]}"
    )


def _prepare_storage(storage: str | None) -> None:
    if storage is None or not storage.startswith("sqlite:///"):
        return
    raw_path = storage[len("sqlite:///") :]
    if raw_path and raw_path != ":memory:":
        Path(raw_path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def create_study(
    *,
    study_name: str = "hospital_plcbf_100",
    storage: str | None = None,
    seed: int = 0,
    load_if_exists: bool = True,
    quick: bool = False,
) -> "optuna.Study":
    """Create, but do not optimize, a reproducible Optuna study."""

    optuna = _require_optuna()
    _prepare_storage(storage)
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        direction="minimize",
        sampler=build_sampler(seed=int(seed)),
        pruner=build_pruner(quick=quick),
    )


_TARGET_TERMINAL_STATES = frozenset({"complete", "pruned", "fail"})


def _trial_state_name(trial: Any) -> str:
    state = getattr(trial, "state", "")
    return str(getattr(state, "name", state)).lower()


def _remaining_target_trials(
    study: Any,
    config: HospitalTuningConfig,
) -> int:
    """Return executions needed to reach the declared terminal-trial target."""

    states = [_trial_state_name(trial) for trial in study.trials]
    if states.count("running"):
        raise RuntimeError(
            "refusing to resume with RUNNING Optuna trials; resolve stale or "
            "concurrently owned trials before launching the deterministic study"
        )
    terminal = sum(state in _TARGET_TERMINAL_STATES for state in states)
    if terminal > config.n_trials:
        raise RuntimeError(
            f"study already has {terminal} terminal trials, exceeding the "
            f"configured target {config.n_trials}"
        )
    remaining = config.n_trials - terminal
    waiting = states.count("waiting")
    if waiting > remaining:
        raise RuntimeError(
            f"study has {waiting} WAITING trials but only {remaining} target "
            "executions remain"
        )
    return remaining


def base_controller_reference_params(
    config: HospitalTuningConfig,
) -> dict[str, float]:
    """Return all tunable values from the untuned publication controller."""

    policies = config.base_config.policies
    safety = config.base_config.safety
    params = {
        "room_target_speed": policies.room_target_speed,
        "stop_gain": policies.stop_gain,
        "cbf_alpha": policies.cbf_alpha,
        "cbf_value_buffer": policies.cbf_value_buffer,
        "component_temperature": policies.component_temperature,
        "time_temperature": policies.time_temperature,
        "max_gradient_norm": policies.max_gradient_norm,
        "hocbf_lambda1": safety.hocbf_lambda1,
        "hocbf_lambda2": safety.hocbf_lambda2,
    }
    if set(params) != _TUNABLE_NAMES:
        raise AssertionError("base controller does not cover the search space")
    return {name: float(value) for name, value in sorted(params.items())}


def enqueue_base_controller_reference(
    study: Any,
    config: HospitalTuningConfig,
) -> bool:
    """Enqueue the exact untuned controller once in a fresh full study."""

    if config.quick:
        return False
    params = base_controller_reference_params(config)
    recorded = study.user_attrs.get(_BASE_REFERENCE_ATTRIBUTE)
    if recorded is not None:
        if recorded != params:
            raise ValueError(
                "Optuna study base-controller reference does not match this "
                "configuration; choose a new --study-name"
            )
        return False
    if study.trials:
        raise ValueError(
            "refusing to add the required base-controller reference after "
            "study trials already exist; choose a new --study-name"
        )
    study.enqueue_trial(
        params,
        user_attrs={"hospital_base_controller_reference": True},
    )
    study.set_user_attr(_BASE_REFERENCE_ATTRIBUTE, params)
    return True


def _json_compatible(value: object) -> object:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _select_audited_best_trial(
    study: Any,
    config: HospitalTuningConfig | None = None,
) -> Any:
    """Select only COMPLETE trials proving the exact benchmark grid finished."""

    if config is None:
        expected_case_ids = list(
            study.user_attrs.get("ordered_benchmark_case_ids", ())
        )
        scenario_hashes = study.user_attrs.get("scenario_grid_sha256", {})
        expected_grid_sha256 = scenario_hashes.get("benchmark")
    else:
        expected_case_ids = [
            f"{case}/seed-{seed}"
            for case, seed in ordered_hospital_tuning_worlds(
                config.cases,
                config.benchmark_seeds,
            )
        ]
        expected_grid_sha256 = scenario_grid_fingerprint(
            config.cases,
            config.benchmark_seeds,
        )
    if not expected_case_ids or not isinstance(expected_grid_sha256, str):
        raise RuntimeError(
            "study lacks the ordered benchmark-grid audit metadata"
        )
    expected_world_count = len(expected_case_ids)
    audited: list[Any] = []
    for trial in study.trials:
        if _trial_state_name(trial) != "complete":
            continue
        attrs = dict(getattr(trial, "user_attrs", {}))
        params = dict(getattr(trial, "params", {}))
        try:
            base_config = (
                config.base_config
                if config is not None
                else hospital_config_from_mapping(attrs.get("config", {}))
            )
            reconstructed = hospital_config_from_params(
                params,
                base=base_config,
            )
            recorded_config_matches = _json_compatible(
                attrs.get("config")
            ) == _json_compatible(asdict(reconstructed))
            value = float(trial.value)
            recorded_score = float(attrs.get("final_full_grid_score"))
            archived_results = _load_exact_result_archive(
                trial,
                expected_case_ids=expected_case_ids,
                expected_grid_sha256=expected_grid_sha256,
                expected_config=reconstructed,
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        outcomes = attrs.get("outcomes")
        if not (
            set(params) == _TUNABLE_NAMES
            and attrs.get("completed_full_benchmark_grid") is True
            and attrs.get("benchmark_grid_sha256") == expected_grid_sha256
            and attrs.get("expected_case_ids") == expected_case_ids
            and attrs.get("evaluated_case_ids") == expected_case_ids
            and attrs.get("evaluated_world_count") == expected_world_count
            and attrs.get("completed_world_count") == expected_world_count
            and attrs.get("expected_complete_world_count")
            == expected_world_count
            and isinstance(outcomes, list)
            and len(outcomes) == expected_world_count
            and recorded_config_matches
            and np.isfinite(value)
            and np.isfinite(recorded_score)
            and np.isclose(value, recorded_score, rtol=0.0, atol=1e-12)
            and len(archived_results) == expected_world_count
        ):
            continue
        audited.append(trial)
    if not audited:
        raise RuntimeError(
            "no COMPLETE Optuna trial has an exact, auditable full-grid record"
        )
    return min(
        audited,
        key=lambda trial: (float(trial.value), int(trial.number)),
    )


def run_tuning(
    *,
    n_trials: int,
    timeout_s: float | None = None,
    study_name: str = "hospital_plcbf_100",
    storage: str | None = None,
    sampler_seed: int = 0,
    cases: Sequence[str] = tuple(HOSPITAL_BENCHMARK_STORIES),
    seeds: Sequence[int] = _CANONICAL_BENCHMARK_SEEDS,
    steps: int = _DEFAULT_TUNING_STEPS,
    base: HospitalConfig = _PUBLICATION_BASE_CONFIG,
    output_prefix: str | Path = Path("results/hospital_optuna_100"),
) -> "optuna.Study":
    """Compatibility API that explicitly optimizes and returns a study."""

    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    benchmark_seeds = tuple(int(seed) for seed in seeds)
    config = HospitalTuningConfig(
        cases=tuple(cases),
        benchmark_seeds=benchmark_seeds,
        steps=steps,
        n_trials=n_trials,
        timeout_s=timeout_s,
        sampler_seed=sampler_seed,
        study_name=study_name,
        storage=storage,
        output_prefix=Path(output_prefix),
        base_config=base,
    )
    study = create_study(
        study_name=config.study_name,
        storage=config.storage,
        seed=config.sampler_seed,
        quick=config.quick,
    )
    fingerprint = bind_study_configuration(study, config)
    result_archive_dir = _trial_archive_directory(config, fingerprint)
    enqueue_base_controller_reference(study, config)
    remaining_trials = _remaining_target_trials(study, config)
    if remaining_trials:
        study.optimize(
            lambda trial: objective(
                trial,
                base=config.base_config,
                cases=config.cases,
                seeds=config.benchmark_seeds,
                steps=config.steps,
                result_archive_dir=result_archive_dir,
            ),
            n_trials=remaining_trials,
            timeout=config.timeout_s,
            n_jobs=1,
        )
    remaining_after = _remaining_target_trials(study, config)
    if remaining_after:
        raise RuntimeError(
            f"Optuna stopped with {remaining_after} of {config.n_trials} "
            "target terminal trials unfinished; resume the same study before "
            "selecting a winner"
        )
    bind_study_configuration(study, config)
    return study


@dataclass(frozen=True)
class TuningRunResult:
    """Artifacts reported directly from the selected completed trial."""

    study: Any
    best_trial: Any
    best_config: HospitalConfig
    benchmark_results: tuple[BenchmarkResult, ...]
    benchmark_reports: BenchmarkReportPaths
    summary_path: Path


def _write_run_summary(
    path: Path,
    *,
    config: HospitalTuningConfig,
    study: Any,
    best_trial: Any,
    best_config: HospitalConfig,
    benchmark_results: tuple[BenchmarkResult, ...],
    reports: BenchmarkReportPaths,
) -> Path:
    state_counts: dict[str, int] = {}
    for trial in study.trials:
        state = _trial_state_name(trial)
        state_counts[state] = state_counts.get(state, 0) + 1
    payload = {
        "study": {
            "name": study.study_name,
            "best_value": float(best_trial.value),
            "best_params": dict(sorted(best_trial.params.items())),
            "best_trial_number": int(best_trial.number),
            "best_trial_timing_diagnostics": {
                "mean_oracle_and_solver_time_total_s": (
                    best_trial.user_attrs.get(
                        "mean_oracle_and_solver_time_total_s"
                    )
                ),
                "sum_oracle_and_solver_time_total_s": (
                    best_trial.user_attrs.get(
                        "sum_oracle_and_solver_time_total_s"
                    )
                ),
            },
            "completed_trials": state_counts.get("complete", 0),
            "terminal_trial_count": sum(
                state_counts.get(state, 0)
                for state in _TARGET_TERMINAL_STATES
            ),
            "target_terminal_trial_count": config.n_trials,
            "trial_state_counts": dict(sorted(state_counts.items())),
            "objective_configuration_fingerprint": (
                study_configuration_fingerprint(config)
            ),
        },
        "configuration": config.metadata(),
        "best_config": asdict(best_config),
        "benchmark_score": score_results(benchmark_results),
        "benchmark_outcomes": [
            result.outcome.value for result in benchmark_results
        ],
        "benchmark_outcome_counts": {
            outcome.value: sum(
                result.outcome is outcome for result in benchmark_results
            )
            for outcome in BenchmarkOutcome
        },
        "benchmark_results_source": (
            "selected_complete_trial_exact_archive_no_rerun"
        ),
        "benchmark_reports": {
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
    """Optimize and report the selected trial without another simulation."""

    study = create_study(
        study_name=config.study_name,
        storage=config.storage,
        seed=config.sampler_seed,
        quick=config.quick,
    )
    fingerprint = bind_study_configuration(study, config)
    result_archive_dir = _trial_archive_directory(config, fingerprint)
    enqueue_base_controller_reference(study, config)
    remaining_trials = _remaining_target_trials(study, config)
    if remaining_trials:
        study.optimize(
            lambda trial: objective(
                trial,
                base=config.base_config,
                cases=config.cases,
                seeds=config.benchmark_seeds,
                steps=config.steps,
                result_archive_dir=result_archive_dir,
            ),
            n_trials=remaining_trials,
            timeout=config.timeout_s,
            n_jobs=1,
        )
    remaining_after = _remaining_target_trials(study, config)
    if remaining_after:
        raise RuntimeError(
            f"Optuna stopped with {remaining_after} of {config.n_trials} "
            "target terminal trials unfinished; resume the same study before "
            "exporting a winner"
        )
    # A long-running study must not export results if implementation files or
    # protocol settings changed after the study was bound.
    bind_study_configuration(study, config)
    best_trial = _select_audited_best_trial(study, config)
    best_config = hospital_config_from_params(
        best_trial.params,
        base=config.base_config,
    )
    expected_case_ids = [
        f"{case}/seed-{seed}"
        for case, seed in ordered_hospital_tuning_worlds(
            config.cases,
            config.benchmark_seeds,
        )
    ]
    benchmark_results = _load_exact_result_archive(
        best_trial,
        expected_case_ids=expected_case_ids,
        expected_grid_sha256=scenario_grid_fingerprint(
            config.cases,
            config.benchmark_seeds,
        ),
        expected_config=best_config,
    )
    benchmark_reports = write_benchmark_reports(
        config.output_prefix,
        benchmark_results,
        metadata={
            **config.metadata(),
            "split": "optimization_and_reporting_full_100_world_grid",
            "best_params": dict(sorted(best_trial.params.items())),
            "best_trial_number": int(best_trial.number),
            "best_config": asdict(best_config),
            "post_selection_simulation": False,
            "results_source": "selected_trial_exact_archive",
        },
        title="Hospital tuned PL-CBF 100-world benchmark",
    )
    write_hospital_benchmark_markdown(
        benchmark_reports.markdown,
        benchmark_results,
        title="Hospital tuned PL-CBF 100-world benchmark",
    )
    summary_path = config.output_prefix.with_name(
        config.output_prefix.name + "_summary.json"
    )
    _write_run_summary(
        summary_path,
        config=config,
        study=study,
        best_trial=best_trial,
        best_config=best_config,
        benchmark_results=benchmark_results,
        reports=benchmark_reports,
    )
    return TuningRunResult(
        study=study,
        best_trial=best_trial,
        best_config=best_config,
        benchmark_results=benchmark_results,
        benchmark_reports=benchmark_reports,
        summary_path=summary_path,
    )


def write_tuning_summary(path: str | Path, study: "optuna.Study") -> Path:
    """Write the audited winner from the compatibility tuning API."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    best = _select_audited_best_trial(study)
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
    parser.add_argument(
        "--trials",
        type=int,
        help=(
            "total COMPLETE/PRUNED/FAIL trial target across fresh and resumed "
            "runs"
        ),
    )
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--study-name")
    parser.add_argument(
        "--storage",
        default="sqlite:///results/hospital_optuna_100.db",
    )
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(HOSPITAL_BENCHMARK_STORIES),
        default=list(HOSPITAL_BENCHMARK_STORIES),
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hospital_optuna_100"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="one trial, one step, one case, and one benchmark seed",
    )
    return parser


def _config_from_args(arguments: argparse.Namespace) -> HospitalTuningConfig:
    quick = bool(arguments.quick)
    cases = tuple(arguments.cases[:1] if quick else arguments.cases)
    benchmark_seeds = tuple(
        arguments.seeds
        if arguments.seeds
        else ((0,) if quick else _CANONICAL_BENCHMARK_SEEDS)
    )
    base = _PUBLICATION_BASE_CONFIG
    return HospitalTuningConfig(
        cases=cases,
        benchmark_seeds=benchmark_seeds,
        steps=(
            arguments.steps
            if arguments.steps is not None
            else (1 if quick else _DEFAULT_TUNING_STEPS)
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
            or (
                "hospital_plcbf_100_quick"
                if quick
                else "hospital_plcbf_100"
            )
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
                "best_value": result.best_trial.value,
                "best_trial_number": result.best_trial.number,
                "summary": str(result.summary_path.resolve()),
                "benchmark_reports": {
                    "csv": str(result.benchmark_reports.csv.resolve()),
                    "json": str(result.benchmark_reports.json.resolve()),
                    "markdown": str(
                        result.benchmark_reports.markdown.resolve()
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
            for item in result.benchmark_results
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_PRUNER_SETTINGS",
    "DEFAULT_SAMPLER_SETTINGS",
    "ExactWorldPrefixPatientMedianPruner",
    "HospitalTuningConfig",
    "TuningRunResult",
    "base_controller_reference_params",
    "bind_study_configuration",
    "build_pruner",
    "build_sampler",
    "build_parser",
    "create_study",
    "enqueue_base_controller_reference",
    "evaluate_plcbf_config",
    "hospital_config_from_params",
    "hospital_tuning_base_config",
    "hospital_tuning_search_space",
    "main",
    "objective",
    "ordered_hospital_tuning_worlds",
    "run_study",
    "run_tuning",
    "scenario_grid_fingerprint",
    "score_results",
    "study_configuration_fingerprint",
    "suggest_hospital_config",
    "TrialNumberSeededTPESampler",
    "write_tuning_summary",
]
