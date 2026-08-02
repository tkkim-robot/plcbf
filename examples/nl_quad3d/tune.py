"""Optuna-ready tuning for nonlinear Quad3D PL-CBF parameters.

Importing this module never creates or runs a study.  The command-line entry
point also requires an explicit ``--run`` flag, so merely inspecting the
configuration cannot accidentally start an expensive optimization.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from plcbf.baselines import BenchmarkMethod
from plcbf.benchmarking import (
    BenchmarkOutcome,
    BenchmarkReportPaths,
    BenchmarkResult,
    write_benchmark_reports,
)

from .benchmark import NLQuad3DBenchmarkConfig, run_benchmark
from .config_io import (
    DEFAULT_CONTROLLER_CONFIG_PATH,
    write_controller_config_artifact,
)
from .controller import NLQuad3DControllerConfig
from .scenarios import (
    PLAYGROUND_CROWDED_SCENARIO,
    PLAYGROUND_STRESS_OBSTACLE_COUNT,
    PLAYGROUND_STRESS_PROTOCOL_VERSION,
    PLAYGROUND_STRESS_SCENARIO,
    get_scenario,
    make_playground_stress_scenario,
    scenario_names,
)


BenchmarkRunner = Callable[
    [NLQuad3DBenchmarkConfig], tuple[BenchmarkResult, ...]
]

_STUDY_FINGERPRINT_ATTRIBUTE = "objective_configuration_fingerprint"
_SEARCH_SPACE_VERSION = (
    "nl_quad3d_plcbf_v4_exact_seed_prefix_patient_median"
)
FULL_TUNING_SEEDS = tuple(range(1, 101))
FULL_VALIDATION_SEEDS = tuple(range(101, 201))
FULL_BENCHMARK_STEPS = 800
DEFAULT_PRUNER_SETTINGS: Mapping[str, int | str] = {
    "type": "exact_seed_prefix_patient_median",
    "direction": "minimize",
    "reference_trials": "complete_only",
    "comparison": "strictly_worse_than_same_step_median",
    "n_startup_trials": 8,
    "n_warmup_seed_steps": 20,
    "interval_seed_steps": 5,
    "n_min_trials": 3,
    "patience_reports": 5,
}
_TUNABLE_PARAMETER_NAMES = (
    "backup_horizon",
    "cbf_alpha",
    "cbf_value_buffer",
    "safety_margin",
    "safety_scale",
    "sensing_radius",
    "max_obstacles",
    "num_radial_policies",
    "target_speed",
    "radial_gain",
    "stop_gain",
    "obstacle_temperature",
    "time_temperature",
    "max_gradient_norm",
    "min_lg_norm",
    "constraint_tolerance",
)


@dataclass(frozen=True)
class NLQuad3DTuningConfig:
    """Frozen full benchmark protocol and Optuna execution settings.

    A completed full-mode trial always covers seeds 1--100 in the registered
    stress scenario.  ``max_steps=None`` is intentional: it delegates to the
    scenario's full episode limit (currently 800) instead of truncating it.
    The cheap protocol is available only through the explicit ``quick`` mode.
    """

    scenarios: tuple[str, ...] = (PLAYGROUND_STRESS_SCENARIO,)
    train_seeds: tuple[int, ...] = FULL_TUNING_SEEDS
    validation_seeds: tuple[int, ...] = FULL_VALIDATION_SEEDS
    max_steps: int | None = None
    playground_obstacle_count: int = PLAYGROUND_STRESS_OBSTACLE_COUNT
    n_trials: int = 50
    timeout_s: float | None = None
    sampler_seed: int = 0
    study_name: str = "nl_quad3d_plcbf_stress_v2_prefix_median"
    storage: str | None = "sqlite:///results/nl_quad3d_optuna.db"
    n_jobs: int = 1
    output_prefix: Path = Path("results/nl_quad3d_optuna")
    best_config_output: Path = DEFAULT_CONTROLLER_CONFIG_PATH
    clearance_target: float = 0.1
    obstacle_position_perturbation: float = 0.12
    obstacle_velocity_perturbation: float = 0.08
    warmup: bool = True
    quick: bool = False
    base_controller_config: NLQuad3DControllerConfig = field(
        default_factory=NLQuad3DControllerConfig
    )

    def __post_init__(self) -> None:
        scenarios = tuple(str(name) for name in self.scenarios)
        train_seeds = tuple(int(seed) for seed in self.train_seeds)
        validation_seeds = tuple(int(seed) for seed in self.validation_seeds)
        if not scenarios or any(not name for name in scenarios):
            raise ValueError("at least one tuning scenario is required")
        if not train_seeds:
            raise ValueError("at least one training seed is required")
        if not validation_seeds:
            raise ValueError("at least one held-out validation seed is required")
        if set(train_seeds) & set(validation_seeds):
            raise ValueError("training and validation seeds must be disjoint")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when provided")
        if self.n_trials < 1:
            raise ValueError("n_trials must be positive")
        if int(self.n_jobs) != 1:
            raise ValueError(
                "n_jobs must equal one: run_trial isolates legacy stochastic "
                "providers through process-global RNG state, which is not "
                "thread-safe under Optuna's n_jobs parallelism"
            )
        obstacle_count = int(self.playground_obstacle_count)
        if obstacle_count < 5:
            raise ValueError("playground_obstacle_count must be at least five")
        if self.timeout_s is not None and self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive when provided")
        clearance_target = float(self.clearance_target)
        if not np.isfinite(clearance_target):
            raise ValueError("clearance_target must be finite")
        position_perturbation = float(self.obstacle_position_perturbation)
        velocity_perturbation = float(self.obstacle_velocity_perturbation)
        if (
            not np.isfinite(position_perturbation)
            or not np.isfinite(velocity_perturbation)
            or position_perturbation < 0.0
            or velocity_perturbation < 0.0
        ):
            raise ValueError("obstacle perturbation magnitudes must be nonnegative")
        study_name = str(self.study_name)
        if not study_name:
            raise ValueError("study_name must not be empty")
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "train_seeds", train_seeds)
        object.__setattr__(self, "validation_seeds", validation_seeds)
        object.__setattr__(self, "sampler_seed", int(self.sampler_seed))
        object.__setattr__(self, "n_jobs", int(self.n_jobs))
        object.__setattr__(self, "study_name", study_name)
        object.__setattr__(self, "output_prefix", Path(self.output_prefix))
        object.__setattr__(self, "best_config_output", Path(self.best_config_output))
        object.__setattr__(self, "playground_obstacle_count", obstacle_count)
        object.__setattr__(self, "clearance_target", clearance_target)
        object.__setattr__(
            self, "obstacle_position_perturbation", position_perturbation
        )
        object.__setattr__(
            self, "obstacle_velocity_perturbation", velocity_perturbation
        )
        object.__setattr__(self, "warmup", bool(self.warmup))
        object.__setattr__(self, "quick", bool(self.quick))

    def metadata(self) -> dict[str, object]:
        resolved_steps = {
            name: (
                self.max_steps
                if self.max_steps is not None
                else get_scenario(name).default_steps
            )
            for name in self.scenarios
        }
        canonical_full = (
            not self.quick
            and self.scenarios == (PLAYGROUND_STRESS_SCENARIO,)
            and self.train_seeds == FULL_TUNING_SEEDS
            and self.validation_seeds == FULL_VALIDATION_SEEDS
            and self.max_steps is None
            and resolved_steps
            == {PLAYGROUND_STRESS_SCENARIO: FULL_BENCHMARK_STEPS}
            and self.playground_obstacle_count
            == PLAYGROUND_STRESS_OBSTACLE_COUNT
            and self.warmup
            and self.base_controller_config == NLQuad3DControllerConfig()
            and self.n_trials >= 50
        )
        return {
            "case_study": "nl_quad3d",
            "method": BenchmarkMethod.PLCBF.value,
            "scenarios": list(self.scenarios),
            "train_seeds": list(self.train_seeds),
            "validation_seeds": list(self.validation_seeds),
            "max_steps": self.max_steps,
            "resolved_max_steps": resolved_steps,
            "playground_obstacle_count": self.playground_obstacle_count,
            "n_trials": self.n_trials,
            "target_terminal_trial_count": self.n_trials,
            "terminal_trial_states": ["complete", "pruned", "fail"],
            "timeout_s": self.timeout_s,
            "sampler_seed": self.sampler_seed,
            "study_name": self.study_name,
            "storage": self.storage,
            "n_jobs": self.n_jobs,
            "best_config_output": str(self.best_config_output),
            "clearance_target": self.clearance_target,
            "seed_perturbations": {
                "seed_zero_is_exact_reference": True,
                "generated_playground_stress": (
                    "The seed procedurally regenerates the complete stress "
                    "field; the generic perturbation widths apply only to "
                    "non-generated scenarios."
                ),
                "position_uniform_half_width_m": (
                    self.obstacle_position_perturbation
                ),
                "velocity_uniform_half_width_mps": (
                    self.obstacle_velocity_perturbation
                ),
            },
            "warmup": self.warmup,
            "quick": self.quick,
            "protocol_kind": (
                "canonical_full" if canonical_full else "explicit_custom"
            ),
            "completed_trial_requires_all_training_seeds": True,
            "pruner": (
                {"type": "none"}
                if self.quick
                else dict(DEFAULT_PRUNER_SETTINGS)
            ),
        }


def suggest_controller_config(
    trial: Any,
    *,
    base: NLQuad3DControllerConfig | None = None,
    quick: bool = False,
) -> NLQuad3DControllerConfig:
    """Suggest the PL-CBF parameters that materially affect safety/performance."""

    base_config = base or NLQuad3DControllerConfig()
    horizon_choices = (
        [0.1, 0.2]
        if quick
        else [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    )
    radial_choices = [2, 4] if quick else [8, 12, 16, 20]
    obstacle_choices = [2] if quick else [8, 12, 16, 24]
    updates = {
        "backup_horizon": trial.suggest_categorical(
            "backup_horizon", horizon_choices
        ),
        "cbf_alpha": trial.suggest_float(
            "cbf_alpha", 0.5, 8.0, log=True
        ),
        "cbf_value_buffer": trial.suggest_float(
            "cbf_value_buffer", 0.0, 0.6
        ),
        "safety_margin": trial.suggest_float(
            "safety_margin", 0.05, 0.5
        ),
        "safety_scale": trial.suggest_float(
            "safety_scale", 1.0, 1.4
        ),
        "sensing_radius": trial.suggest_float(
            "sensing_radius", 2.5, 9.0
        ),
        "max_obstacles": trial.suggest_categorical(
            "max_obstacles", obstacle_choices
        ),
        "num_radial_policies": trial.suggest_categorical(
            "num_radial_policies", radial_choices
        ),
        "target_speed": trial.suggest_float("target_speed", 1.0, 4.0),
        "radial_gain": trial.suggest_float(
            "radial_gain", 1.0, 5.0, log=True
        ),
        "stop_gain": trial.suggest_float(
            "stop_gain", 1.0, 6.0, log=True
        ),
        "obstacle_temperature": trial.suggest_float(
            "obstacle_temperature", 20.0, 140.0, log=True
        ),
        "time_temperature": trial.suggest_float(
            "time_temperature", 20.0, 140.0, log=True
        ),
        "max_gradient_norm": trial.suggest_float(
            "max_gradient_norm", 60.0, 320.0, log=True
        ),
        "min_lg_norm": trial.suggest_float(
            "min_lg_norm", 1e-6, 1e-2, log=True
        ),
        "constraint_tolerance": trial.suggest_float(
            "constraint_tolerance", 1e-6, 1e-3, log=True
        ),
    }
    if quick:
        updates["nominal_prefix_steps"] = 0
    return replace(base_config, **updates)


def controller_config_from_params(
    params: Mapping[str, Any],
    *,
    base: NLQuad3DControllerConfig | None = None,
    quick: bool = False,
) -> NLQuad3DControllerConfig:
    """Reconstruct a controller from an Optuna best-parameter mapping."""

    base_config = base or NLQuad3DControllerConfig()
    tunable_names = set(_TUNABLE_PARAMETER_NAMES)
    unknown = set(params) - tunable_names
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown tuned controller parameters: {names}")
    updates = {
        name: params.get(name, getattr(base_config, name))
        for name in tunable_names
    }
    if quick:
        updates["nominal_prefix_steps"] = 0
    return replace(base_config, **updates)


@dataclass(frozen=True)
class TuningScore:
    """Lexicographic safety counts plus bounded secondary performance score."""

    score: float
    total_cases: int
    unsuccessful_count: int
    error_count: int
    collision_count: int
    infeasible_count: int
    timeout_count: int
    success_count: int
    mean_clearance: float | None
    clearance_shortfall: float
    solver_fallback_rate: float
    mean_intervention: float
    mean_step_time_s: float

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def score_results(
    results: Iterable[BenchmarkResult],
    *,
    clearance_target: float = 0.1,
) -> TuningScore:
    """Score a trial with exact success-first lexicographic priority.

    Counts are encoded in base ``N+1``.  Errors have highest priority because
    they invalidate a trial.  Subject to equal errors, one fewer unsuccessful
    episode dominates every possible change in failure composition, preventing
    Optuna from trading success away for an early tilt/infeasible termination.
    Collision, infeasible, and timeout counts break ties in that order.  The
    secondary score is strictly below one and cannot reverse the ordering.
    """

    trials = tuple(results)
    if not trials:
        raise ValueError("cannot score an empty benchmark")
    clearance_target = float(clearance_target)
    if not np.isfinite(clearance_target):
        raise ValueError("clearance_target must be finite")
    errors = sum(item.outcome is BenchmarkOutcome.ERROR for item in trials)
    collisions = sum(
        item.outcome is BenchmarkOutcome.COLLISION for item in trials
    )
    infeasible = sum(
        item.outcome is BenchmarkOutcome.INFEASIBLE for item in trials
    )
    timeouts = sum(
        item.outcome is BenchmarkOutcome.TIMEOUT for item in trials
    )
    successes = sum(
        item.outcome is BenchmarkOutcome.SUCCESS for item in trials
    )
    unsuccessful = len(trials) - successes
    base = len(trials) + 1
    failure_rank = (
        errors * base**4
        + unsuccessful * base**3
        + collisions * base**2
        + infeasible * base
        + timeouts
    )

    clearances = [
        item.min_clearance
        for item in trials
        if item.min_clearance is not None
    ]
    mean_clearance = (
        float(np.mean(clearances)) if clearances else None
    )
    shortfalls = [
        max(0.0, clearance_target - value) for value in clearances
    ]
    clearance_shortfall = (
        float(np.mean(shortfalls)) if shortfalls else 0.0
    )
    interventions = [
        item.intervention
        for item in trials
        if item.intervention is not None
    ]
    mean_intervention = (
        float(np.mean(interventions)) if interventions else 0.0
    )
    step_times = [
        value for item in trials for value in item.solve_times_s
    ]
    mean_step_time = float(np.mean(step_times)) if step_times else 0.0
    total_steps = sum(
        int(item.case_metrics.get("control_steps", 0)) for item in trials
    )
    solver_fallback_count = sum(
        int(item.case_metrics.get("solver_fallback_count", 0))
        for item in trials
    )
    solver_fallback_rate = solver_fallback_count / max(1, total_steps)

    clearance_component = min(clearance_shortfall / 10.0, 1.0)
    intervention_component = mean_intervention / (1.0 + mean_intervention)
    # Wall time remains an audited metric but is deliberately excluded from
    # optimization.  Host contention/JIT scheduling is not a controller-quality
    # signal and must not decide the winning safety parameters.
    secondary = (
        0.5 * clearance_component
        + 0.25 * min(solver_fallback_rate, 1.0)
        + 0.15 * intervention_component
    )
    return TuningScore(
        score=float(failure_rank + secondary),
        total_cases=len(trials),
        unsuccessful_count=unsuccessful,
        error_count=errors,
        collision_count=collisions,
        infeasible_count=infeasible,
        timeout_count=timeouts,
        success_count=successes,
        mean_clearance=mean_clearance,
        clearance_shortfall=clearance_shortfall,
        solver_fallback_rate=solver_fallback_rate,
        mean_intervention=mean_intervention,
        mean_step_time_s=mean_step_time,
    )


def _validate_result_grid(
    results: Iterable[BenchmarkResult],
    *,
    scenarios: Iterable[str],
    seeds: Iterable[int],
    method: BenchmarkMethod = BenchmarkMethod.PLCBF,
) -> tuple[BenchmarkResult, ...]:
    """Require exactly one correctly identified result for every grid case."""

    materialized = tuple(results)
    scenario_tuple = tuple(str(item) for item in scenarios)
    seed_tuple = tuple(int(item) for item in seeds)
    expected = {
        (scenario, seed): f"{scenario}/seed-{seed}"
        for scenario in scenario_tuple
        for seed in seed_tuple
    }
    observed: dict[tuple[str, int], str] = {}
    for item in materialized:
        if item.algorithm != method.value:
            raise RuntimeError(
                f"expected only {method.value!r} results, got {item.algorithm!r}"
            )
        matching = [
            scenario
            for scenario in scenario_tuple
            if item.case_id == f"{scenario}/seed-{item.seed}"
        ]
        if len(matching) != 1:
            raise RuntimeError(
                f"unexpected benchmark case identity {item.case_id!r} "
                f"for seed {item.seed}"
            )
        key = (matching[0], item.seed)
        if key in observed:
            raise RuntimeError(
                f"duplicate benchmark case {item.case_id!r}"
            )
        observed[key] = item.case_id
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise RuntimeError(
            "benchmark runner did not return the exact configured grid; "
            f"missing={missing}, extra={extra}"
        )
    if len(materialized) != len(expected):
        raise RuntimeError(
            f"expected {len(expected)} benchmark cases, got {len(materialized)}"
        )
    return materialized


def build_objective(
    config: NLQuad3DTuningConfig,
    *,
    benchmark_runner: BenchmarkRunner | None = None,
) -> Callable[[Any], float]:
    """Create an Optuna objective with seed-prefix pruning reports.

    One benchmark call is made per seed so Optuna can terminate an evidently
    poor trial without paying for the remaining episodes.  If the objective
    returns normally, every configured seed has necessarily completed; Optuna
    therefore cannot label a partial full-protocol evaluation ``COMPLETE``.
    """

    runner = run_benchmark if benchmark_runner is None else benchmark_runner

    def objective(trial: Any) -> float:
        controller = suggest_controller_config(
            trial,
            base=config.base_controller_config,
            quick=config.quick,
        )
        accumulated: list[BenchmarkResult] = []
        report = getattr(trial, "report", None)
        should_prune = getattr(trial, "should_prune", None)
        for seed_index, seed in enumerate(config.train_seeds, start=1):
            benchmark_config = NLQuad3DBenchmarkConfig(
                methods=(BenchmarkMethod.PLCBF.value,),
                scenarios=config.scenarios,
                seeds=(seed,),
                max_steps=config.max_steps,
                controller_config=controller,
                obstacle_position_perturbation=(
                    config.obstacle_position_perturbation
                ),
                obstacle_velocity_perturbation=(
                    config.obstacle_velocity_perturbation
                ),
                playground_obstacle_count=(
                    config.playground_obstacle_count
                ),
                warmup=config.warmup,
                configuration_source="optuna_trial",
            )
            seed_results = _validate_result_grid(
                runner(benchmark_config),
                scenarios=config.scenarios,
                seeds=(seed,),
            )
            accumulated.extend(seed_results)
            interim = score_results(
                accumulated,
                clearance_target=config.clearance_target,
            )
            if callable(report):
                report(interim.score, step=seed_index)
            if callable(should_prune) and should_prune():
                trial.set_user_attr("controller_config", asdict(controller))
                trial.set_user_attr("partial_score_components", interim.as_dict())
                trial.set_user_attr("evaluated_seed_count", seed_index)
                trial.set_user_attr("evaluated_case_count", len(accumulated))
                trial.set_user_attr(
                    "evaluated_seeds", list(config.train_seeds[:seed_index])
                )
                optuna = _require_optuna()
                raise optuna.TrialPruned(
                    f"pruned after {seed_index}/{len(config.train_seeds)} seeds"
                )

        results = _validate_result_grid(
            accumulated,
            scenarios=config.scenarios,
            seeds=config.train_seeds,
        )
        scored = score_results(
            results, clearance_target=config.clearance_target
        )
        trial.set_user_attr("controller_config", asdict(controller))
        trial.set_user_attr("score_components", scored.as_dict())
        trial.set_user_attr("evaluated_seed_count", len(config.train_seeds))
        trial.set_user_attr("completed_case_count", len(results))
        trial.set_user_attr(
            "expected_complete_case_count",
            len(config.train_seeds) * len(config.scenarios),
        )
        trial.set_user_attr("completed_full_protocol", True)
        trial.set_user_attr("evaluated_seeds", list(config.train_seeds))
        trial.set_user_attr(
            "outcomes", [item.outcome.value for item in results]
        )
        return scored.score

    return objective


def _require_optuna() -> Any:
    try:
        import optuna
    except ImportError as error:
        raise RuntimeError(
            "Optuna is required to run tuning; install the project dependencies"
        ) from error
    return optuna


class ExactSeedPrefixPatientMedianPruner:
    """Prune against COMPLETE trials at identical seed-prefix lengths.

    Optuna's :class:`~optuna.pruners.MedianPruner` compares a trial's best
    intermediate value with reference trials.  That is not valid for this
    objective because its cumulative score changes scale as each seed is
    appended: an unusually good short prefix can permanently shield a poor
    longer prefix.  This pruner compares only values reported at the exact
    same seed count instead.

    Patience is reconstructed from the trial's intermediate-value history,
    rather than stored as mutable process state.  It is therefore deterministic
    across SQLite resumes and parallel worker-process reconstruction.
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
                "exact seed-prefix pruning requires a single-objective "
                "minimize study"
            ) from error
        name = getattr(direction, "name", direction)
        if str(name).rsplit(".", 1)[-1].upper() != "MINIMIZE":
            raise ValueError(
                "exact seed-prefix pruning requires minimize direction"
            )

    def prune(self, study: Any, trial: Any) -> bool:
        """Return true after ``patience`` same-prefix median losses."""

        self._require_minimize(study)
        completed = tuple(
            item
            for item in self._study_trials(study)
            if self._state_name(item) == "COMPLETE"
        )
        if len(completed) < self.n_startup_trials:
            return False

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
                and np.isfinite(
                    float(item.intermediate_values[checkpoint])
                )
            ]
            if len(references) < self.n_min_trials:
                return False
            if float(candidate) <= float(np.median(references)):
                return False
        return True


def build_pruner(*, quick: bool = False) -> Any:
    """Build the deterministic seed-prefix pruner used by the study.

    The median comparison waits for eight COMPLETE startup trials and twenty
    benchmark seeds.  Each comparison uses only reference values reported at
    that exact seed-prefix length.  Five consecutive worse-than-median
    checkpoints are required, so one unusually difficult prefix cannot discard
    an otherwise competitive controller.  Quick smoke studies use Optuna's
    no-op pruner.
    """

    optuna = _require_optuna()
    if quick:
        return optuna.pruners.NopPruner()
    return ExactSeedPrefixPatientMedianPruner(
        n_startup_trials=int(DEFAULT_PRUNER_SETTINGS["n_startup_trials"]),
        n_warmup_steps=int(DEFAULT_PRUNER_SETTINGS["n_warmup_seed_steps"]),
        interval_steps=int(DEFAULT_PRUNER_SETTINGS["interval_seed_steps"]),
        n_min_trials=int(DEFAULT_PRUNER_SETTINGS["n_min_trials"]),
        patience=int(DEFAULT_PRUNER_SETTINGS["patience_reports"]),
    )


def _prepare_storage(storage: str | None) -> None:
    if storage is None or not storage.startswith("sqlite:///"):
        return
    raw_path = storage[len("sqlite:///") :]
    if raw_path and raw_path != ":memory:":
        Path(raw_path).expanduser().parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TuningRunResult:
    """Artifacts from an explicitly executed tuning study."""

    study: Any
    best_controller_config: NLQuad3DControllerConfig
    validation_results: tuple[BenchmarkResult, ...]
    validation_reports: BenchmarkReportPaths
    summary_path: Path
    best_config_path: Path | None = None
    best_trial: Any | None = None


def study_configuration_fingerprint(
    config: NLQuad3DTuningConfig,
) -> str:
    """Hash every setting that changes the training objective/search space."""

    stress_protocol: dict[str, object] | None = None
    if PLAYGROUND_STRESS_SCENARIO in config.scenarios:
        protocol_config = NLQuad3DBenchmarkConfig(
            methods=(BenchmarkMethod.PLCBF.value,),
            scenarios=(PLAYGROUND_STRESS_SCENARIO,),
            seeds=(config.train_seeds[0],),
            max_steps=config.max_steps,
            controller_config=config.base_controller_config,
            playground_obstacle_count=config.playground_obstacle_count,
            obstacle_position_perturbation=(
                config.obstacle_position_perturbation
            ),
            obstacle_velocity_perturbation=(
                config.obstacle_velocity_perturbation
            ),
            warmup=config.warmup,
            configuration_source="optuna_fingerprint",
        )
        stress_protocol = {
            "version": PLAYGROUND_STRESS_PROTOCOL_VERSION,
            "metadata": protocol_config.metadata()["stress_scenario"],
            "generator_source_sha256": hashlib.sha256(
                inspect.getsource(make_playground_stress_scenario).encode("utf-8")
            ).hexdigest(),
        }
    payload = {
        "search_space_version": _SEARCH_SPACE_VERSION,
        "scenarios": list(config.scenarios),
        "train_seeds": list(config.train_seeds),
        "validation_seeds": list(config.validation_seeds),
        "max_steps": config.max_steps,
        "resolved_max_steps": {
            name: (
                config.max_steps
                if config.max_steps is not None
                else get_scenario(name).default_steps
            )
            for name in config.scenarios
        },
        "playground_obstacle_count": config.playground_obstacle_count,
        "sampler_seed": config.sampler_seed,
        "n_jobs": config.n_jobs,
        "clearance_target": config.clearance_target,
        "obstacle_position_perturbation": (
            config.obstacle_position_perturbation
        ),
        "obstacle_velocity_perturbation": (
            config.obstacle_velocity_perturbation
        ),
        "warmup": config.warmup,
        "quick": config.quick,
        "base_controller_config": asdict(config.base_controller_config),
        "pruner": (
            {"type": "none"}
            if config.quick
            else dict(DEFAULT_PRUNER_SETTINGS)
        ),
        "reference_trial": "exact_base_controller_once_on_new_full_study",
        "stress_protocol": stress_protocol,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bind_study_configuration(study: Any, config: NLQuad3DTuningConfig) -> str:
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


def enqueue_reference_trial(
    study: Any,
    config: NLQuad3DTuningConfig,
) -> bool:
    """Seed a brand-new full study with the exact declared base controller."""

    if config.quick or study.trials:
        return False
    params = {
        name: getattr(config.base_controller_config, name)
        for name in _TUNABLE_PARAMETER_NAMES
    }
    study.enqueue_trial(
        params,
        user_attrs={
            "seeded_reference_controller": True,
            "reference_kind": "exact_base_controller",
        },
    )
    study.set_user_attr("base_controller_reference_enqueued", True)
    study.set_user_attr("base_controller_reference_params", params)
    return True


_TARGET_TERMINAL_STATES = frozenset({"complete", "pruned", "fail"})


def _trial_state_name(trial: Any) -> str:
    state = getattr(trial, "state", "")
    return str(getattr(state, "name", state)).lower()


def _remaining_target_trials(study: Any, config: NLQuad3DTuningConfig) -> int:
    """Return executions needed to reach the terminal-trial target.

    COMPLETE, PRUNED, and FAIL are auditable terminal executions.  WAITING
    trials are not counted because Optuna consumes them within ``n_trials``.
    A RUNNING trial on entry is stale or owned by another process, either of
    which makes a deterministic single-worker resume unsafe.
    """

    states = [_trial_state_name(trial) for trial in study.trials]
    running = states.count("running")
    if running:
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


def _select_audited_best_trial(
    study: Any,
    config: NLQuad3DTuningConfig,
) -> Any:
    """Select only a COMPLETE trial that proves the configured grid finished."""

    expected_cases = len(config.train_seeds) * len(config.scenarios)
    canonical_full = config.metadata()["protocol_kind"] == "canonical_full"
    if canonical_full and expected_cases != 100:
        raise RuntimeError(
            "canonical full tuning must contain exactly 100 cases, "
            f"got {expected_cases}"
        )
    audited: list[Any] = []
    for trial in study.trials:
        if _trial_state_name(trial) != "complete":
            continue
        attrs = getattr(trial, "user_attrs", {})
        score = attrs.get("score_components")
        outcomes = attrs.get("outcomes")
        controller_data = attrs.get("controller_config")
        try:
            expected_controller = asdict(
                controller_config_from_params(
                    getattr(trial, "params", {}),
                    base=config.base_controller_config,
                    quick=config.quick,
                )
            )
        except (TypeError, ValueError):
            continue
        if not (
            attrs.get("completed_full_protocol") is True
            and attrs.get("evaluated_seed_count") == len(config.train_seeds)
            and attrs.get("completed_case_count") == expected_cases
            and attrs.get("expected_complete_case_count") == expected_cases
            and attrs.get("evaluated_seeds") == list(config.train_seeds)
            and isinstance(score, Mapping)
            and score.get("total_cases") == expected_cases
            and isinstance(outcomes, list)
            and len(outcomes) == expected_cases
            and controller_data == expected_controller
            and getattr(trial, "value", None) is not None
            and np.isfinite(float(trial.value))
        ):
            continue
        audited.append(trial)
    if not audited:
        raise RuntimeError(
            "no COMPLETE Optuna trial has an exact, auditable full-protocol record"
        )
    return min(
        audited,
        key=lambda item: (float(item.value), int(item.number)),
    )


def _write_summary(
    path: Path,
    *,
    config: NLQuad3DTuningConfig,
    study: Any,
    best_controller: NLQuad3DControllerConfig,
    validation_results: tuple[BenchmarkResult, ...],
    reports: BenchmarkReportPaths,
    best_config_path: Path,
    fingerprint: str,
    best_trial: Any,
) -> Path:
    validation_score = score_results(
        validation_results, clearance_target=config.clearance_target
    )
    state_counts: dict[str, int] = {}
    for item in study.trials:
        name = str(getattr(item.state, "name", item.state)).lower()
        state_counts[name] = state_counts.get(name, 0) + 1
    document = {
        "study": {
            "name": study.study_name,
            "best_value": float(best_trial.value),
            "best_params": dict(sorted(best_trial.params.items())),
            "best_trial_number": int(best_trial.number),
            "trial_state_counts": dict(sorted(state_counts.items())),
            "objective_configuration_fingerprint": fingerprint,
        },
        "configuration": config.metadata(),
        "best_controller": asdict(best_controller),
        "best_controller_yaml": str(best_config_path),
        "best_trial_score": best_trial.user_attrs.get(
            "score_components"
        ),
        "validation_score": validation_score.as_dict(),
        "validation_reports": {
            "csv": str(reports.csv),
            "json": str(reports.json),
            "markdown": str(reports.markdown),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _winning_config_provenance(
    *,
    config: NLQuad3DTuningConfig,
    study: Any,
    fingerprint: str,
    best_trial: Any,
    validation_results: tuple[BenchmarkResult, ...],
) -> dict[str, object]:
    state_counts: dict[str, int] = {}
    for item in study.trials:
        name = str(getattr(item.state, "name", item.state)).lower()
        state_counts[name] = state_counts.get(name, 0) + 1
    return {
        "artifact_kind": "optuna_winning_plcbf_controller",
        "study": {
            "name": study.study_name,
            "storage": config.storage,
            "best_trial_number": int(best_trial.number),
            "best_value": float(best_trial.value),
            "best_params": dict(sorted(best_trial.params.items())),
            "trial_state_counts": dict(sorted(state_counts.items())),
            "objective_configuration_fingerprint": fingerprint,
            "search_space_version": _SEARCH_SPACE_VERSION,
        },
        "optimization_protocol": {
            "stress_protocol_version": PLAYGROUND_STRESS_PROTOCOL_VERSION,
            "stress_generator_source_sha256": hashlib.sha256(
                inspect.getsource(make_playground_stress_scenario).encode("utf-8")
            ).hexdigest(),
            "scenarios": list(config.scenarios),
            "seed_range_inclusive": [
                min(config.train_seeds),
                max(config.train_seeds),
            ],
            "seed_count": len(config.train_seeds),
            "seeds": list(config.train_seeds),
            "max_steps": config.max_steps,
            "resolved_max_steps": {
                name: (
                    config.max_steps
                    if config.max_steps is not None
                    else get_scenario(name).default_steps
                )
                for name in config.scenarios
            },
            "playground_obstacle_count": config.playground_obstacle_count,
            "position_perturbation_m": (
                config.obstacle_position_perturbation
            ),
            "velocity_perturbation_mps": (
                config.obstacle_velocity_perturbation
            ),
            "generated_stress_uses_generic_perturbations": False,
            "generated_stress_randomization": (
                "Each seed procedurally regenerates all structured and "
                "corridor hazards; generic perturbation widths are retained "
                "only for non-generated scenarios."
            ),
            "warmup": config.warmup,
            "completed_trial_requires_all_seeds": True,
        },
        "objective": {
            "direction": "minimize",
            "strict_priority": [
                "errors",
                "total_unsuccessful_cases",
                "collisions",
                "infeasibilities",
                "timeouts",
                "bounded_secondary_quality",
            ],
            "best_score_components": best_trial.user_attrs.get(
                "score_components"
            ),
            "wall_time": (
                "recorded for post-hoc audit; excluded from Optuna objective"
            ),
            "pruner": (
                {"type": "none"}
                if config.quick
                else dict(DEFAULT_PRUNER_SETTINGS)
            ),
        },
        "held_out_validation": {
            "seeds": list(config.validation_seeds),
            "seed_count": len(config.validation_seeds),
            "case_count": len(validation_results),
            "error_count": sum(
                item.outcome is BenchmarkOutcome.ERROR
                for item in validation_results
            ),
            "outcomes": [item.outcome.value for item in validation_results],
            "score": score_results(
                validation_results,
                clearance_target=config.clearance_target,
            ).as_dict(),
        },
    }


def run_study(config: NLQuad3DTuningConfig) -> TuningRunResult:
    """Explicitly create/resume, optimize, and validate one Optuna study."""

    canonical_full = config.metadata()["protocol_kind"] == "canonical_full"
    if (
        config.best_config_output.resolve()
        == DEFAULT_CONTROLLER_CONFIG_PATH.resolve()
        and not canonical_full
    ):
        raise ValueError(
            "quick/custom studies may not overwrite the packaged default; "
            "pass --best-config-output with a separate results path"
        )
    optuna = _require_optuna()
    _prepare_storage(config.storage)
    sampler = optuna.samplers.TPESampler(seed=config.sampler_seed)
    pruner = build_pruner(quick=config.quick)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=config.study_name,
        storage=config.storage,
        load_if_exists=True,
    )
    fingerprint = bind_study_configuration(study, config)
    enqueue_reference_trial(study, config)
    remaining_trials = _remaining_target_trials(study, config)
    if remaining_trials:
        study.optimize(
            build_objective(config),
            n_trials=remaining_trials,
            timeout=config.timeout_s,
            n_jobs=config.n_jobs,
        )
    remaining_after = _remaining_target_trials(study, config)
    if remaining_after:
        raise RuntimeError(
            f"Optuna stopped with {remaining_after} of "
            f"{config.n_trials} target terminal trials unfinished; resume the "
            "same study before exporting a winner"
        )
    best_trial = _select_audited_best_trial(study, config)
    best_controller = controller_config_from_params(
        best_trial.params,
        base=config.base_controller_config,
        quick=config.quick,
    )
    best_config_path = config.best_config_output
    validation_config = NLQuad3DBenchmarkConfig(
        methods=(BenchmarkMethod.PLCBF.value,),
        scenarios=config.scenarios,
        seeds=config.validation_seeds,
        max_steps=config.max_steps,
        controller_config=best_controller,
        obstacle_position_perturbation=(
            config.obstacle_position_perturbation
        ),
        obstacle_velocity_perturbation=(
            config.obstacle_velocity_perturbation
        ),
        playground_obstacle_count=config.playground_obstacle_count,
        warmup=config.warmup,
        configuration_source=f"audited_optuna_trial:{best_trial.number}",
    )
    validation_results = _validate_result_grid(
        run_benchmark(validation_config),
        scenarios=config.scenarios,
        seeds=config.validation_seeds,
    )
    validation_error_count = sum(
        item.outcome is BenchmarkOutcome.ERROR
        for item in validation_results
    )
    if validation_error_count:
        raise RuntimeError(
            f"held-out validation produced {validation_error_count} ERROR "
            "results; the existing default controller was not changed"
        )
    validation_prefix = config.output_prefix.with_name(
        config.output_prefix.name + "_validation"
    )
    reports = write_benchmark_reports(
        validation_prefix,
        validation_results,
        metadata={
            **validation_config.metadata(),
            "split": "held_out_validation",
            "study_name": config.study_name,
        },
        title="Nonlinear Quad3D tuned PL-CBF validation",
    )
    summary_path = config.output_prefix.with_name(
        config.output_prefix.name + "_summary.json"
    )
    _write_summary(
        summary_path,
        config=config,
        study=study,
        best_controller=best_controller,
        validation_results=validation_results,
        reports=reports,
        best_config_path=best_config_path,
        fingerprint=fingerprint,
        best_trial=best_trial,
    )
    # Export is deliberately last.  The atomic writer preserves any existing
    # valid default if optimization, audit, validation, or report generation
    # fails.
    best_config_path = write_controller_config_artifact(
        best_config_path,
        best_controller,
        provenance=_winning_config_provenance(
            config=config,
            study=study,
            fingerprint=fingerprint,
            best_trial=best_trial,
            validation_results=validation_results,
        ),
    )
    return TuningRunResult(
        study=study,
        best_controller_config=best_controller,
        validation_results=validation_results,
        validation_reports=reports,
        summary_path=summary_path,
        best_config_path=best_config_path,
        best_trial=best_trial,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="explicitly start/resume the Optuna study",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=scenario_names(),
        help="training/validation scenario; repeat for a suite",
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--validation-seed", action="append", type=int)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help=(
            "target total COMPLETE/PRUNED/FAIL executions in the study; "
            "resumes run only the remaining count"
        ),
    )
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "must be 1: run_trial's process-global RNG isolation is not "
            "thread-safe under Optuna parallel jobs"
        ),
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name (quick mode uses a separate default study)",
    )
    parser.add_argument(
        "--storage",
        default="sqlite:///results/nl_quad3d_optuna.db",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/nl_quad3d_optuna"),
    )
    parser.add_argument(
        "--best-config-output",
        type=Path,
        default=DEFAULT_CONTROLLER_CONFIG_PATH,
        help=(
            "winning controller YAML (default: the packaged config consumed "
            "by NL-Quad3D benchmark and single-run entry points)"
        ),
    )
    parser.add_argument(
        "--obstacle-count",
        type=int,
        default=PLAYGROUND_STRESS_OBSTACLE_COUNT,
        help="fixed number of spheres in the generated stress scenario",
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--position-perturbation",
        type=float,
        default=0.12,
        help=(
            "bounded offset for static scenarios; canonical playground_stress "
            "uses seed-driven procedural regeneration instead"
        ),
    )
    parser.add_argument(
        "--velocity-perturbation",
        type=float,
        default=0.08,
        help=(
            "bounded offset for static scenarios; canonical playground_stress "
            "uses seed-driven procedural regeneration instead"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="one trial, one step, and a small policy library",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> NLQuad3DTuningConfig:
    scenarios = (
        tuple(args.scenario)
        if args.scenario
        else (
            (PLAYGROUND_CROWDED_SCENARIO,)
            if args.quick
            else (PLAYGROUND_STRESS_SCENARIO,)
        )
    )
    train_seeds = (
        tuple(args.seed)
        if args.seed
        else ((0,) if args.quick else FULL_TUNING_SEEDS)
    )
    validation_seeds = (
        tuple(args.validation_seed)
        if args.validation_seed
        else ((101,) if args.quick else FULL_VALIDATION_SEEDS)
    )
    steps = args.steps if args.steps is not None else (1 if args.quick else None)
    trials = (
        args.trials if args.trials is not None else (1 if args.quick else 50)
    )
    base = NLQuad3DControllerConfig(
        num_radial_policies=2 if args.quick else 12,
        backup_horizon=0.1 if args.quick else 1.25,
        max_obstacles=2 if args.quick else 8,
        nominal_prefix_steps=0 if args.quick else 1,
    )
    return NLQuad3DTuningConfig(
        scenarios=scenarios,
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        max_steps=steps,
        n_trials=trials,
        timeout_s=args.timeout,
        sampler_seed=args.sampler_seed,
        n_jobs=args.jobs,
        study_name=(
            args.study_name
            or (
                "nl_quad3d_plcbf_quick"
                if args.quick
                else "nl_quad3d_plcbf_stress_v2_prefix_median"
            )
        ),
        storage=None if args.storage == "none" else args.storage,
        output_prefix=args.output,
        best_config_output=args.best_config_output,
        playground_obstacle_count=(
            5 if args.quick else args.obstacle_count
        ),
        obstacle_position_perturbation=args.position_perturbation,
        obstacle_velocity_perturbation=args.velocity_perturbation,
        warmup=not args.no_warmup,
        quick=args.quick,
        base_controller_config=base,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    if not args.run:
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
    audited_best_value = (
        float(result.best_trial.value)
        if result.best_trial is not None
        else float(result.study.best_value)
    )
    print(
        json.dumps(
            {
                "study": result.study.study_name,
                "best_value": audited_best_value,
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
    "DEFAULT_PRUNER_SETTINGS",
    "ExactSeedPrefixPatientMedianPruner",
    "FULL_BENCHMARK_STEPS",
    "FULL_TUNING_SEEDS",
    "FULL_VALIDATION_SEEDS",
    "NLQuad3DTuningConfig",
    "TuningRunResult",
    "TuningScore",
    "build_objective",
    "build_parser",
    "build_pruner",
    "bind_study_configuration",
    "controller_config_from_params",
    "enqueue_reference_trial",
    "main",
    "run_study",
    "score_results",
    "study_configuration_fingerprint",
    "suggest_controller_config",
]
