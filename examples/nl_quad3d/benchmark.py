"""Headless, deterministic benchmark for the nonlinear Quad3D case study.

Every method uses the same nonlinear plant, route, moving obstacles, and
shifted-safety-point collision geometry.  PL-CBF uses its native
playground-matched max operator and selected-policy QP; trajectory baselines
use their warehouse-style native algorithms rather than certificate-selector
surrogates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import random
import time
from typing import Callable, Iterable, Mapping

import numpy as np

from plcbf.baselines import (
    BENCHMARK_METHODS,
    BaselineDecision,
    BenchmarkMethod,
)
from plcbf.benchmarking import (
    BenchmarkOutcome,
    BenchmarkReportPaths,
    BenchmarkResult,
    aggregate_results,
    write_benchmark_reports,
)

from .controller import NLQuad3DControllerConfig, PLCBF_NLQuad3D
from .baselines import NLQuad3DBaselineSuite
from .config_io import (
    DEFAULT_CONTROLLER_CONFIG_PATH,
    load_controller_config_artifact,
)
from .dynamics import NLQuad3D
from .scenarios import (
    NLQuad3DScenario,
    PLAYGROUND_CROWDED_SCENARIO,
    PLAYGROUND_OBSTACLE_COUNT,
    PLAYGROUND_REFERENCE_OBSTACLE_COUNT,
    PLAYGROUND_START_GOAL_PROTECTION,
    PLAYGROUND_STRESS_CORRIDOR_LOWER,
    PLAYGROUND_STRESS_CORRIDOR_UPPER,
    PLAYGROUND_STRESS_CROSS_FLOW_COUNT,
    PLAYGROUND_STRESS_OBSTACLE_COUNT,
    PLAYGROUND_STRESS_OBSTACLE_SPEED_MAX,
    PLAYGROUND_STRESS_OBSTACLE_SPEED_MIN,
    PLAYGROUND_STRESS_PAIR_CLEARANCE,
    PLAYGROUND_STRESS_PROTOCOL_VERSION,
    PLAYGROUND_STRESS_SCENARIO,
    PLAYGROUND_STRESS_STREAM_DIRECTIONS,
    advance_obstacles,
    get_scenario,
    make_playground_crowded_scenario,
    make_playground_stress_scenario,
    minimum_clearance,
    scenario_names,
)


ModelFactory = Callable[[], NLQuad3D]
ControllerFactory = Callable[
    [NLQuad3D, NLQuad3DControllerConfig, object | None],
    PLCBF_NLQuad3D,
]
ScenarioLoader = Callable[[str], NLQuad3DScenario]
Clock = Callable[[], float]

DEFAULT_BENCHMARK_SCENARIOS = (PLAYGROUND_STRESS_SCENARIO,)


def _default_controller_factory(
    model: NLQuad3D,
    config: NLQuad3DControllerConfig,
    bounds: object | None,
) -> PLCBF_NLQuad3D:
    return PLCBF_NLQuad3D(model, config, bounds=bounds)


def _validate_tilt_protocol(
    controller_config: NLQuad3DControllerConfig,
    tilt_max_rad: float,
) -> None:
    if not np.isclose(
        controller_config.rollout_tilt_max_rad,
        tilt_max_rad,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "controller rollout_tilt_max_rad must equal benchmark tilt_max_rad"
        )


@dataclass(frozen=True)
class NLQuad3DBenchmarkConfig:
    """Reproducible scenario grid and controller configuration."""

    methods: tuple[str, ...] = BENCHMARK_METHODS
    scenarios: tuple[str, ...] = DEFAULT_BENCHMARK_SCENARIOS
    seeds: tuple[int, ...] = (0,)
    max_steps: int | None = None
    controller_config: NLQuad3DControllerConfig = field(
        default_factory=NLQuad3DControllerConfig
    )
    plcbf_controller_config: NLQuad3DControllerConfig | None = None
    safe_value_threshold: float = 0.0
    obstacle_position_perturbation: float = 0.12
    obstacle_velocity_perturbation: float = 0.08
    playground_obstacle_count: int | None = None
    tilt_max_rad: float = np.deg2rad(60.0)
    warmup: bool = True
    stop_on_collision: bool = True
    configuration_source: str = "defaults"

    def __post_init__(self) -> None:
        if not self.methods:
            raise ValueError("at least one benchmark method is required")
        methods = tuple(BenchmarkMethod(method).value for method in self.methods)
        if len(set(methods)) != len(methods):
            raise ValueError("benchmark methods must be unique")
        scenarios = tuple(str(name) for name in self.scenarios)
        if not scenarios or any(not name for name in scenarios):
            raise ValueError("at least one non-empty scenario name is required")
        if len(set(scenarios)) != len(scenarios):
            raise ValueError("scenario names must be unique")
        seeds = tuple(int(seed) for seed in self.seeds)
        if not seeds:
            raise ValueError("at least one seed is required")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when provided")
        threshold = float(self.safe_value_threshold)
        if not np.isfinite(threshold):
            raise ValueError("safe_value_threshold must be finite")
        position_perturbation = float(self.obstacle_position_perturbation)
        velocity_perturbation = float(self.obstacle_velocity_perturbation)
        if (
            not np.isfinite(position_perturbation)
            or not np.isfinite(velocity_perturbation)
            or position_perturbation < 0.0
            or velocity_perturbation < 0.0
        ):
            raise ValueError("obstacle perturbation magnitudes must be nonnegative")
        generated_scenarios = {
            PLAYGROUND_CROWDED_SCENARIO,
            PLAYGROUND_STRESS_SCENARIO,
        }.intersection(scenarios)
        if (
            len(generated_scenarios) > 1
            and self.playground_obstacle_count is None
        ):
            raise ValueError(
                "mixed crowded/stress grids require an explicit shared "
                "playground_obstacle_count; their defaults are 32 and 48"
            )
        playground_obstacle_count = (
            PLAYGROUND_STRESS_OBSTACLE_COUNT
            if self.playground_obstacle_count is None
            and PLAYGROUND_STRESS_SCENARIO in scenarios
            else (
                PLAYGROUND_OBSTACLE_COUNT
                if self.playground_obstacle_count is None
                else int(self.playground_obstacle_count)
            )
        )
        if playground_obstacle_count < PLAYGROUND_REFERENCE_OBSTACLE_COUNT:
            raise ValueError(
                "playground_obstacle_count must be at least five"
            )
        tilt_max_rad = float(self.tilt_max_rad)
        if (
            not np.isfinite(tilt_max_rad)
            or tilt_max_rad <= 0.0
            or tilt_max_rad > np.pi
        ):
            raise ValueError("tilt_max_rad must be in the interval (0, pi]")
        _validate_tilt_protocol(self.controller_config, tilt_max_rad)
        plcbf_controller_config = (
            self.controller_config
            if self.plcbf_controller_config is None
            else self.plcbf_controller_config
        )
        _validate_tilt_protocol(plcbf_controller_config, tilt_max_rad)
        configuration_source = str(self.configuration_source)
        if not configuration_source:
            raise ValueError("configuration_source must not be empty")
        object.__setattr__(self, "methods", methods)
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "safe_value_threshold", threshold)
        object.__setattr__(
            self, "obstacle_position_perturbation", position_perturbation
        )
        object.__setattr__(
            self, "obstacle_velocity_perturbation", velocity_perturbation
        )
        object.__setattr__(
            self, "playground_obstacle_count", playground_obstacle_count
        )
        object.__setattr__(self, "tilt_max_rad", tilt_max_rad)
        object.__setattr__(
            self,
            "plcbf_controller_config",
            plcbf_controller_config,
        )
        object.__setattr__(
            self,
            "configuration_source",
            configuration_source,
        )
        object.__setattr__(self, "warmup", bool(self.warmup))
        object.__setattr__(
            self, "stop_on_collision", bool(self.stop_on_collision)
        )

    def metadata(self) -> dict[str, object]:
        """Return stable JSON-compatible report metadata."""

        return {
            "case_study": "nl_quad3d",
            "dynamics": "nonlinear_12_state_quadrotor",
            "collision_geometry": (
                "playground-matched physical spheres and PL-CBF sphere "
                "clearance at the shifted safety point, without extra rho_z "
                "inflation; world bounds reflect prescribed hazards only and "
                "are not robot walls"
            ),
            "methods": list(self.methods),
            "scenarios": list(self.scenarios),
            "seeds": list(self.seeds),
            "max_steps": self.max_steps,
            "safe_value_threshold": self.safe_value_threshold,
            "configuration_source": self.configuration_source,
            "headline_scenario": {
                "name": PLAYGROUND_CROWDED_SCENARIO,
                "world_bounds_m": [20.0, 20.0, 10.0],
                "obstacle_count": (
                    self.playground_obstacle_count
                    if PLAYGROUND_CROWDED_SCENARIO in self.scenarios
                    else PLAYGROUND_OBSTACLE_COUNT
                ),
                "reference_obstacle_count": (
                    PLAYGROUND_REFERENCE_OBSTACLE_COUNT
                ),
                "random_obstacle_count": (
                    (
                        self.playground_obstacle_count
                        if PLAYGROUND_CROWDED_SCENARIO in self.scenarios
                        else PLAYGROUND_OBSTACLE_COUNT
                    )
                    - PLAYGROUND_REFERENCE_OBSTACLE_COUNT
                ),
                "start_goal_protection_m": (
                    PLAYGROUND_START_GOAL_PROTECTION
                ),
                "initial_velocity_mps": [1.0, 0.0, 0.0],
                "seed_replay": (
                    "The seed regenerates the random spheres; the five "
                    "reference threats remain fixed."
                ),
            },
            "stress_scenario": {
                "name": PLAYGROUND_STRESS_SCENARIO,
                "protocol_version": PLAYGROUND_STRESS_PROTOCOL_VERSION,
                "world_bounds_m": [20.0, 20.0, 10.0],
                "obstacle_count": (
                    self.playground_obstacle_count
                    if PLAYGROUND_STRESS_SCENARIO in self.scenarios
                    else PLAYGROUND_STRESS_OBSTACLE_COUNT
                ),
                "structured_cross_flow_count": min(
                    PLAYGROUND_STRESS_CROSS_FLOW_COUNT,
                    (
                        self.playground_obstacle_count
                        if PLAYGROUND_STRESS_SCENARIO in self.scenarios
                        else PLAYGROUND_STRESS_OBSTACLE_COUNT
                    ),
                ),
                "structured_stream_count": min(
                    PLAYGROUND_STRESS_CROSS_FLOW_COUNT,
                    (
                        self.playground_obstacle_count
                        if PLAYGROUND_STRESS_SCENARIO in self.scenarios
                        else PLAYGROUND_STRESS_OBSTACLE_COUNT
                    ),
                ),
                "corridor_random_count": max(
                    0,
                    (
                        self.playground_obstacle_count
                        if PLAYGROUND_STRESS_SCENARIO in self.scenarios
                        else PLAYGROUND_STRESS_OBSTACLE_COUNT
                    )
                    - PLAYGROUND_STRESS_CROSS_FLOW_COUNT,
                ),
                "cross_flow_directions": list(
                    PLAYGROUND_STRESS_STREAM_DIRECTIONS
                ),
                "structured_stream_directions": list(
                    PLAYGROUND_STRESS_STREAM_DIRECTIONS
                ),
                "corridor_sampling_bounds_m": [
                    list(PLAYGROUND_STRESS_CORRIDOR_LOWER),
                    list(PLAYGROUND_STRESS_CORRIDOR_UPPER),
                ],
                "random_speed_bounds_mps": [
                    PLAYGROUND_STRESS_OBSTACLE_SPEED_MIN,
                    PLAYGROUND_STRESS_OBSTACLE_SPEED_MAX,
                ],
                "minimum_initial_pair_surface_clearance_m": (
                    PLAYGROUND_STRESS_PAIR_CLEARANCE
                ),
                "start_goal_protection_m": (
                    PLAYGROUND_START_GOAL_PROTECTION
                ),
                "initial_velocity_mps": [1.0, 0.0, 0.0],
                "seed_replay": (
                    "The seed generates every cross-flow and corridor sphere "
                    "before the episode; all methods receive the same array."
                ),
                "method_independence": (
                    "Generation never observes a controller, policy, rollout, "
                    "or outcome."
                ),
                "obstacle_interaction": (
                    "Independent prescribed spheres reflect at world bounds; "
                    "they may overlap or pass through one another."
                ),
            },
            "seed_perturbations": {
                "seed_zero_is_exact_reference": True,
                "playground_crowded": (
                    "Seeded procedural regeneration; perturbation widths do "
                    "not apply."
                ),
                "playground_stress": (
                    "Seeded cross-flow/corridor regeneration; perturbation "
                    "widths do not apply."
                ),
                "position_uniform_half_width_m": (
                    self.obstacle_position_perturbation
                ),
                "velocity_uniform_half_width_mps": (
                    self.obstacle_velocity_perturbation
                ),
                "position_clipping": "inside bounds when bounds are defined",
            },
            "warmup": self.warmup,
            "stop_on_collision": self.stop_on_collision,
            "baseline_controller": asdict(self.controller_config),
            "plcbf_controller": asdict(self.plcbf_controller_config),
            "controller": asdict(self.controller_config),
            "controller_scope": (
                "The packaged/tuned controller applies only to PL-CBF; "
                "comparison methods retain baseline_controller."
            ),
            "state_validity": {
                "body_tilt_rad": (
                    "acos(clip(cos(phi) * cos(theta), -1, 1))"
                ),
                "nominal_attitude_envelope": (
                    "dynamics.attitude_bound is reported as a soft tilt-excess "
                    "diagnostic, not an episode termination threshold"
                ),
                "tilt_max_rad": self.tilt_max_rad,
                "tilt_max_deg": float(np.rad2deg(self.tilt_max_rad)),
                "tilt_termination_reference": (
                    "dpcbf/benchmark/run_experiment.py::_quad3D_spec and "
                    "dpcbf/benchmark/controller/rerun_tracking_controller.py"
                ),
                "tilt_violation_outcome": "infeasible",
                "body_rates": (
                    "The dynamics clamp ||x[9:12]|| to body_rate_max. "
                    "abs(x[11]) is reported separately as a diagnostic."
                ),
                "nominal_yaw_slew": (
                    "dynamics.nominal_yaw_slew_max caps a desired rotate-to "
                    "reference; it is not a hard state-validity bound"
                ),
                "integration_behavior": (
                    "Tilt is never silently clamped; crossing tilt_max_rad "
                    "terminates the trial as infeasible."
                ),
            },
            "decision_metrics": {
                "selector_fallback": (
                    "The shared policy selector explicitly used its fallback "
                    "control."
                ),
                "solver_fallback": (
                    "Decision infeasible or a policy selector explicitly used "
                    "its fallback control."
                ),
                "backup_executed": (
                    "A method entered an executable backup/emergency path: "
                    "PL-CBF or Library-PCBF-MI direct policy backup, "
                    "Backup-CBF/Multi-Backup-CBF-MI emergency backup, "
                    "MPS/Gatekeeper committed retrace, or MI-MPC solver "
                    "fallback. Normal MI-MPC continuous controls and ordinary "
                    "filtered-QP policy selections are excluded."
                ),
                "shield_active": (
                    "A feasible normal MPS backup_shield or Gatekeeper "
                    "commit/committed backup was executed."
                ),
                "fallback_count": (
                    "Deprecated alias of solver_fallback_count; it never mixes "
                    "normal shield activation into solver fallback."
                ),
                "mi_mpc_requested_safety_feasible": (
                    "The solved MI-MPC selected a branch meeting the original "
                    "requested safety threshold. This is distinct from MILP "
                    "trajectory feasibility."
                ),
                "mi_mpc_safety_threshold_relaxed": (
                    "No branch met the requested threshold, so the warehouse "
                    "max-safety emergency admission lowered the effective "
                    "threshold. A feasible relaxed solve does not count as "
                    "requested-safety feasible."
                ),
            },
            "timing": (
                "Each solve_times_s sample is native certificate/trajectory "
                "construction plus native solver wall time for one control step."
            ),
            "intervention": (
                "Mean squared Euclidean distance from nominal rotor thrust."
            ),
        }


def _copy_scenario(scenario: NLQuad3DScenario) -> NLQuad3DScenario:
    return NLQuad3DScenario(
        name=scenario.name,
        waypoints=scenario.waypoints,
        obstacles=scenario.obstacles,
        bounds=scenario.bounds,
        description=scenario.description,
        reach_threshold=scenario.reach_threshold,
        default_steps=scenario.default_steps,
        initial_velocity=scenario.initial_velocity,
    )


def seeded_scenario(
    scenario: NLQuad3DScenario,
    seed: int,
    *,
    position_perturbation: float = 0.12,
    velocity_perturbation: float = 0.08,
    playground_obstacle_count: int | None = None,
) -> NLQuad3DScenario:
    """Return a bounded deterministic obstacle variant.

    ``playground_crowded`` and ``playground_stress`` use ``seed`` to
    procedurally regenerate their complete obstacle fields; the generic
    perturbation widths do not apply to those generated protocols.  For every
    other scenario, seed zero is the unmodified reference and nonzero seeds
    perturb each obstacle coordinate and velocity independently with a
    uniform bounded offset.  When world bounds exist, sphere centers are
    clipped so their full physical radius remains inside the world.
    """

    if scenario.name == PLAYGROUND_CROWDED_SCENARIO:
        return make_playground_crowded_scenario(
            seed,
            obstacle_count=(
                PLAYGROUND_OBSTACLE_COUNT
                if playground_obstacle_count is None
                else playground_obstacle_count
            ),
        )
    if scenario.name == PLAYGROUND_STRESS_SCENARIO:
        return make_playground_stress_scenario(
            seed,
            obstacle_count=(
                PLAYGROUND_STRESS_OBSTACLE_COUNT
                if playground_obstacle_count is None
                else playground_obstacle_count
            ),
        )

    result = _copy_scenario(scenario)
    position_perturbation = float(position_perturbation)
    velocity_perturbation = float(velocity_perturbation)
    if (
        not np.isfinite(position_perturbation)
        or not np.isfinite(velocity_perturbation)
        or position_perturbation < 0.0
        or velocity_perturbation < 0.0
    ):
        raise ValueError("obstacle perturbation magnitudes must be nonnegative")
    if int(seed) == 0 or result.obstacles.shape[0] == 0:
        return result

    generator = np.random.default_rng(int(seed) % (2**64))
    obstacles = result.obstacles.copy()
    obstacles[:, :3] += generator.uniform(
        -position_perturbation,
        position_perturbation,
        size=(obstacles.shape[0], 3),
    )
    obstacles[:, 4:7] += generator.uniform(
        -velocity_perturbation,
        velocity_perturbation,
        size=(obstacles.shape[0], 3),
    )
    if result.bounds is not None:
        lower = np.asarray(result.bounds.lower, dtype=float)
        upper = np.asarray(result.bounds.upper, dtype=float)
        radii = obstacles[:, 3:4]
        obstacles[:, :3] = np.clip(
            obstacles[:, :3], lower[None, :] + radii, upper[None, :] - radii
        )
    return NLQuad3DScenario(
        name=result.name,
        waypoints=result.waypoints,
        obstacles=obstacles,
        bounds=result.bounds,
        description=result.description,
        reach_threshold=result.reach_threshold,
        default_steps=result.default_steps,
        initial_velocity=result.initial_velocity,
    )


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _maximum(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _p95(values: list[float]) -> float:
    return float(np.percentile(values, 95.0)) if values else 0.0


def _finite_clearance(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _degrees_or_none(value: float | None) -> float | None:
    return None if value is None else float(np.rad2deg(value))


def _initial_obstacle_protocol_metrics(
    scenario: NLQuad3DScenario,
) -> dict[str, int | float | str | None]:
    """Return auditable realized-scene statistics without controller data."""

    obstacles = np.asarray(scenario.obstacles, dtype=float).reshape(-1, 7)
    speeds = np.linalg.norm(obstacles[:, 4:7], axis=1)
    if obstacles.shape[0] < 2:
        pair_clearance = None
    else:
        positions = obstacles[:, :3]
        center_distances = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :],
            axis=2,
        )
        surface_clearances = (
            center_distances
            - obstacles[:, None, 3]
            - obstacles[None, :, 3]
        )
        np.fill_diagonal(surface_clearances, np.inf)
        pair_clearance = float(np.min(surface_clearances))
    is_stress = scenario.name == PLAYGROUND_STRESS_SCENARIO
    structured_stream_count = (
        min(PLAYGROUND_STRESS_CROSS_FLOW_COUNT, obstacles.shape[0])
        if is_stress
        else 0
    )
    return {
        "scenario_generator": (
            "seeded_balanced_six_axis_stream_corridor"
            if is_stress
            else (
                "seeded_exact_playground"
                if scenario.name == PLAYGROUND_CROWDED_SCENARIO
                else "authored_with_bounded_seed_perturbation"
            )
        ),
        "scenario_protocol_version": (
            PLAYGROUND_STRESS_PROTOCOL_VERSION if is_stress else None
        ),
        "structured_cross_flow_count": structured_stream_count,
        "structured_stream_count": structured_stream_count,
        "corridor_random_count": (
            obstacles.shape[0] - structured_stream_count if is_stress else 0
        ),
        "initial_obstacle_speed_min_mps": (
            float(np.min(speeds)) if speeds.size else None
        ),
        "initial_obstacle_speed_mean_mps": (
            float(np.mean(speeds)) if speeds.size else None
        ),
        "initial_obstacle_speed_max_mps": (
            float(np.max(speeds)) if speeds.size else None
        ),
        "initial_min_pair_surface_clearance_m": pair_clearance,
    }


def _case_id(scenario: NLQuad3DScenario, seed: int) -> str:
    return f"{scenario.name}/seed-{seed}"


def _body_tilt_rad(state: np.ndarray) -> float:
    """Return body-z tilt from world-z using the reference benchmark formula."""

    phi, theta = np.asarray(state, dtype=float).reshape(-1)[6:8]
    return float(
        np.arccos(np.clip(np.cos(phi) * np.cos(theta), -1.0, 1.0))
    )


def _decision_metric_flags(
    method: BenchmarkMethod,
    decision: BaselineDecision,
) -> tuple[bool, bool, bool, bool]:
    """Return selector, solver, direct-backup, and normal-shield flags."""

    selector_fallback = bool(
        (
            decision.policy_decision is not None
            and decision.policy_decision.diagnostics.used_fallback
        )
        or (
            decision.policy_decision is None
            and decision.status.startswith("fallback:")
        )
    )
    solver_fallback = bool(not decision.feasible or selector_fallback)
    mps_backup = (
        method is BenchmarkMethod.MPS and decision.used_fallback
    )
    gatekeeper_backup = (
        method is BenchmarkMethod.GATEKEEPER and decision.used_fallback
    )
    selected_policy_backup = bool(
        selector_fallback
        and method
        in (
            BenchmarkMethod.PLCBF,
            BenchmarkMethod.LIBRARY_PCBF_MI,
        )
    )
    fixed_backup_emergency = bool(
        method is BenchmarkMethod.BACKUP_CBF
        and decision.used_fallback
        and decision.status.startswith("fallback_after:")
    )
    multi_backup_emergency = bool(
        method is BenchmarkMethod.MULTI_BACKUP_CBF_MI
        and decision.used_fallback
    )
    mi_mpc_emergency = bool(
        method is BenchmarkMethod.MI_MPC and decision.used_fallback
    )
    backup_executed = bool(
        selected_policy_backup
        or mps_backup
        or gatekeeper_backup
        or fixed_backup_emergency
        or multi_backup_emergency
        or mi_mpc_emergency
    )
    shield_active = bool(
        decision.feasible
        and (
            mps_backup
            or gatekeeper_backup
        )
    )
    return (
        selector_fallback,
        solver_fallback,
        backup_executed,
        shield_active,
    )


def _mi_mpc_safety_flags(
    method: BenchmarkMethod,
    baseline_suite: object,
) -> tuple[bool, bool] | None:
    """Return post-decision MI-MPC safety diagnostics without changing control."""

    if method is not BenchmarkMethod.MI_MPC:
        return None
    result = getattr(baseline_suite, "last_mi_mpc_result", None)
    if result is None:
        return None
    return (
        bool(result.safety_feasible),
        bool(result.safety_threshold_relaxed),
    )


def _trial_metrics(
    *,
    scenario: NLQuad3DScenario,
    seed: int,
    control_steps: int,
    reached_goal: bool,
    completed_waypoints: int,
    final_goal_distance: float,
    path_length: float,
    intervention_values: list[float],
    certificate_times: list[float],
    solver_times: list[float],
    full_step_times: list[float],
    certificate_counts: list[int],
    infeasible_count: int,
    selector_fallback_count: int,
    solver_fallback_count: int,
    backup_executed_count: int,
    shield_active_count: int,
    mi_mpc_result_count: int | None,
    mi_mpc_result_missing_count: int | None,
    mi_mpc_requested_safety_feasible_count: int | None,
    mi_mpc_safety_threshold_relaxed_count: int | None,
    policy_switches: int,
    max_tilt_rad: float,
    attitude_bound_rad: float | None,
    tilt_max_rad: float,
    tilt_max_violated: bool,
    max_body_rate_norm_rad_s: float,
    body_rate_max_rad_s: float | None,
    max_abs_body_yaw_rate_rad_s: float,
    nominal_yaw_slew_max_rad_s: float | None,
) -> dict[str, bool | int | float | str | None]:
    denominator = max(1, control_steps)
    attitude_excess = (
        None
        if attitude_bound_rad is None
        else max(0.0, max_tilt_rad - attitude_bound_rad)
    )
    body_rate_bound_violated = bool(
        body_rate_max_rad_s is not None
        and max_body_rate_norm_rad_s > body_rate_max_rad_s
    )
    return {
        "scenario": scenario.name,
        "seed": seed,
        "obstacle_count": int(scenario.obstacles.shape[0]),
        **_initial_obstacle_protocol_metrics(scenario),
        "initial_velocity_x_mps": float(scenario.initial_velocity[0]),
        "initial_velocity_y_mps": float(scenario.initial_velocity[1]),
        "initial_velocity_z_mps": float(scenario.initial_velocity[2]),
        "control_steps": control_steps,
        "reached_goal": reached_goal,
        "completed_waypoints": completed_waypoints,
        "final_goal_distance": final_goal_distance,
        "path_length": path_length,
        "intervention_total": float(np.sum(intervention_values)),
        "intervention_max": _maximum(intervention_values),
        "infeasible_count": infeasible_count,
        "infeasible_rate": infeasible_count / denominator,
        "selector_fallback_count": selector_fallback_count,
        "selector_fallback_rate": selector_fallback_count / denominator,
        "solver_fallback_count": solver_fallback_count,
        "solver_fallback_rate": solver_fallback_count / denominator,
        "backup_executed_count": backup_executed_count,
        "backup_executed_rate": backup_executed_count / denominator,
        "shield_active_count": shield_active_count,
        "shield_active_rate": shield_active_count / denominator,
        "mi_mpc_result_count": mi_mpc_result_count,
        "mi_mpc_result_missing_count": mi_mpc_result_missing_count,
        "mi_mpc_requested_safety_feasible_count": (
            mi_mpc_requested_safety_feasible_count
        ),
        "mi_mpc_requested_safety_feasible_rate": (
            None
            if mi_mpc_requested_safety_feasible_count is None
            or mi_mpc_result_count is None
            else (
                mi_mpc_requested_safety_feasible_count
                / max(1, mi_mpc_result_count)
            )
        ),
        "mi_mpc_safety_threshold_relaxed_count": (
            mi_mpc_safety_threshold_relaxed_count
        ),
        "mi_mpc_safety_threshold_relaxed_rate": (
            None
            if mi_mpc_safety_threshold_relaxed_count is None
            or mi_mpc_result_count is None
            else (
                mi_mpc_safety_threshold_relaxed_count
                / max(1, mi_mpc_result_count)
            )
        ),
        # Backward-compatible alias with corrected, non-mixed semantics.
        "fallback_count": solver_fallback_count,
        "fallback_rate": solver_fallback_count / denominator,
        "policy_switches": policy_switches,
        "max_tilt_rad": max_tilt_rad,
        "max_tilt_deg": float(np.rad2deg(max_tilt_rad)),
        "attitude_bound_rad": attitude_bound_rad,
        "attitude_bound_deg": _degrees_or_none(attitude_bound_rad),
        "max_attitude_excess_rad": attitude_excess,
        "max_attitude_excess_deg": _degrees_or_none(attitude_excess),
        "tilt_max_rad": tilt_max_rad,
        "tilt_max_deg": float(np.rad2deg(tilt_max_rad)),
        "tilt_max_violated": tilt_max_violated,
        "max_body_rate_norm_rad_s": max_body_rate_norm_rad_s,
        "body_rate_max_rad_s": body_rate_max_rad_s,
        "body_rate_bound_violated": body_rate_bound_violated,
        "max_abs_body_yaw_rate_rad_s": max_abs_body_yaw_rate_rad_s,
        "nominal_yaw_slew_max_rad_s": nominal_yaw_slew_max_rad_s,
        "state_bound_violation": (
            "tilt_max" if tilt_max_violated else "none"
        ),
        "mean_certificate_count": _mean(
            [float(value) for value in certificate_counts]
        ),
        "certificate_time_mean_s": _mean(certificate_times),
        "certificate_time_p95_s": _p95(certificate_times),
        "certificate_time_max_s": _maximum(certificate_times),
        "solver_time_mean_s": _mean(solver_times),
        "solver_time_p95_s": _p95(solver_times),
        "solver_time_max_s": _maximum(solver_times),
        "full_step_time_mean_s": _mean(full_step_times),
        "full_step_time_p95_s": _p95(full_step_times),
        "full_step_time_max_s": _maximum(full_step_times),
    }


def run_trial(
    method: BenchmarkMethod | str,
    scenario: NLQuad3DScenario | str,
    *,
    seed: int = 0,
    max_steps: int | None = None,
    controller_config: NLQuad3DControllerConfig | None = None,
    safe_value_threshold: float = 0.0,
    obstacle_position_perturbation: float = 0.12,
    obstacle_velocity_perturbation: float = 0.08,
    playground_obstacle_count: int | None = None,
    tilt_max_rad: float = np.deg2rad(60.0),
    warmup: bool = True,
    stop_on_collision: bool = True,
    model_factory: ModelFactory = NLQuad3D,
    controller_factory: ControllerFactory = _default_controller_factory,
    scenario_loader: ScenarioLoader = get_scenario,
    clock: Clock = time.perf_counter,
) -> BenchmarkResult:
    """Run one comparison method on one fresh nonlinear scenario.

    Seed zero preserves the exact registered scenario.  Nonzero seeds produce
    bounded obstacle variants and also isolate Python/NumPy random state for
    stochastic policy providers.  Equal scenario/seed/config tuples therefore
    reproduce identical physical trajectories.
    """

    parsed_method = (
        method if isinstance(method, BenchmarkMethod) else BenchmarkMethod(method)
    )
    source = scenario_loader(scenario) if isinstance(scenario, str) else scenario
    current_scenario = seeded_scenario(
        source,
        seed,
        position_perturbation=obstacle_position_perturbation,
        velocity_perturbation=obstacle_velocity_perturbation,
        playground_obstacle_count=playground_obstacle_count,
    )
    limit = (
        current_scenario.default_steps
        if max_steps is None
        else int(max_steps)
    )
    if limit < 1:
        raise ValueError("max_steps must be positive")
    seed = int(seed)
    tilt_max_rad = float(tilt_max_rad)
    if (
        not np.isfinite(tilt_max_rad)
        or tilt_max_rad <= 0.0
        or tilt_max_rad > np.pi
    ):
        raise ValueError("tilt_max_rad must be in the interval (0, pi]")
    if controller_config is not None:
        _validate_tilt_protocol(controller_config, tilt_max_rad)

    state = current_scenario.initial_state
    obstacles = current_scenario.obstacles.copy()
    waypoint_index = 1 if current_scenario.waypoints.shape[0] > 1 else 0
    intervention_values: list[float] = []
    certificate_times: list[float] = []
    solver_times: list[float] = []
    full_step_times: list[float] = []
    certificate_counts: list[int] = []
    selected_policies: list[str] = []
    infeasible_count = 0
    selector_fallback_count = 0
    solver_fallback_count = 0
    backup_executed_count = 0
    shield_active_count = 0
    mi_mpc_result_count = 0
    mi_mpc_result_missing_count = 0
    mi_mpc_requested_safety_feasible_count = 0
    mi_mpc_safety_threshold_relaxed_count = 0
    policy_switches = 0
    path_length = 0.0
    reached_goal = False
    collision = False
    tilt_max_violated = False
    max_tilt_rad = _body_tilt_rad(state)
    max_body_rate_norm_rad_s = float(np.linalg.norm(state[9:12]))
    max_abs_body_yaw_rate_rad_s = abs(float(state[11]))
    attitude_bound_rad: float | None = None
    body_rate_max_rad_s: float | None = None
    nominal_yaw_slew_max_rad_s: float | None = None
    error_message: str | None = None
    min_clearance_value = float("inf")
    model: NLQuad3D | None = None
    numpy_random_state = np.random.get_state()
    python_random_state = random.getstate()
    np.random.seed(seed % (2**32))
    random.seed(seed)

    try:
        model = model_factory()
        config = controller_config or NLQuad3DControllerConfig(dt=model.dt)
        _validate_tilt_protocol(config, tilt_max_rad)
        if not np.isclose(config.dt, model.dt):
            raise ValueError("controller dt and nonlinear dynamics dt must match")
        attitude_bound_rad = float(model.config.attitude_bound)
        body_rate_max_rad_s = float(model.config.body_rate_max)
        nominal_yaw_slew_max_rad_s = float(
            model.config.nominal_yaw_slew_max
        )
        if not np.isfinite(attitude_bound_rad) or attitude_bound_rad <= 0.0:
            raise ValueError("dynamics attitude_bound must be finite and positive")
        if (
            not np.isfinite(body_rate_max_rad_s)
            or body_rate_max_rad_s <= 0.0
        ):
            raise ValueError("dynamics body_rate_max must be finite and positive")
        if (
            not np.isfinite(nominal_yaw_slew_max_rad_s)
            or nominal_yaw_slew_max_rad_s <= 0.0
        ):
            raise ValueError(
                "dynamics nominal_yaw_slew_max must be finite and positive"
            )
        if tilt_max_rad <= attitude_bound_rad:
            raise ValueError(
                "tilt_max_rad must exceed the nominal attitude_bound"
            )
        if max_body_rate_norm_rad_s > body_rate_max_rad_s:
            raise ValueError(
                "initial state exceeds the dynamics body-rate norm bound"
            )
        if not np.isfinite(max_abs_body_yaw_rate_rad_s):
            raise ValueError("initial body yaw rate must be finite")
        if not np.isfinite(max_body_rate_norm_rad_s):
            raise ValueError("initial body rates must be finite")
        if not np.isfinite(max_tilt_rad):
            raise ValueError("initial body tilt must be finite")
        if max_tilt_rad > np.pi:
            raise ValueError("initial body tilt exceeds pi")
        if max_tilt_rad < 0.0:
            raise ValueError("initial body tilt must be nonnegative")
        tilt_max_violated = max_tilt_rad > tilt_max_rad
        controller = controller_factory(model, config, current_scenario.bounds)
        baseline_suite = NLQuad3DBaselineSuite(
            controller,
            model,
            current_scenario.bounds,
            waypoints=current_scenario.waypoints,
        )
        min_clearance_value = minimum_clearance(
            model.safety_point(state),
            obstacles,
            model.config.robot_radius,
        )
        collision = min_clearance_value < 0.0

        if (
            warmup
            and not collision
            and not tilt_max_violated
        ):
            first_goal = current_scenario.waypoints[waypoint_index]
            # Compile the common differentiable rollout oracle without
            # mutating MPS/Gatekeeper committed-trajectory state.
            if parsed_method in {
                BenchmarkMethod.PLCBF,
                BenchmarkMethod.LIBRARY_PCBF_MI,
            }:
                if type(controller) is PLCBF_NLQuad3D:
                    controller.warmup(state, first_goal, obstacles)
                else:
                    controller.policy_certificates(
                        state, first_goal, obstacles
                    )

        for _ in range(limit):
            if collision and stop_on_collision:
                break
            if tilt_max_violated:
                break
            goal = current_scenario.waypoints[waypoint_index]
            nominal = model.nominal_input(state, goal)

            certificate_started = clock()
            if parsed_method is BenchmarkMethod.LIBRARY_PCBF_MI:
                certificates = (
                    controller.decision_certificates(state, goal, obstacles)
                    if type(controller) is PLCBF_NLQuad3D
                    else controller.policy_certificates(state, goal, obstacles)
                )
            else:
                certificates = ()
            certificate_elapsed = max(0.0, clock() - certificate_started)
            solver_started = clock()
            decision = baseline_suite.solve(
                parsed_method,
                state,
                goal,
                obstacles,
                nominal,
                certificates=certificates,
                active_waypoint_index=waypoint_index,
                safe_value_threshold=safe_value_threshold,
            )
            mi_mpc_safety = _mi_mpc_safety_flags(
                parsed_method,
                baseline_suite,
            )
            if parsed_method is BenchmarkMethod.MI_MPC:
                if mi_mpc_safety is None:
                    mi_mpc_result_missing_count += 1
                else:
                    requested_safe, threshold_relaxed = mi_mpc_safety
                    mi_mpc_result_count += 1
                    mi_mpc_requested_safety_feasible_count += int(
                        requested_safe
                    )
                    mi_mpc_safety_threshold_relaxed_count += int(
                        threshold_relaxed
                    )
            solver_elapsed = max(0.0, clock() - solver_started)
            if parsed_method is BenchmarkMethod.PLCBF:
                certificate_count = len(controller.last_certificates)
            elif parsed_method in {
                BenchmarkMethod.BACKUP_CBF,
                BenchmarkMethod.POLICY_PCBF,
                BenchmarkMethod.MPS,
                BenchmarkMethod.GATEKEEPER,
            }:
                certificate_count = 1
            elif parsed_method in {
                BenchmarkMethod.MULTI_BACKUP_CBF_MI,
            }:
                certificate_count = len(controller.candidates(goal))
            elif parsed_method is BenchmarkMethod.MI_MPC:
                certificate_count = len(baseline_suite.mi_mpc_candidates())
            else:
                certificate_count = len(certificates)

            control = model.saturate_rotors(decision.control)
            if not np.all(np.isfinite(control)):
                raise FloatingPointError("baseline returned non-finite control")
            intervention = float(np.sum((control - nominal) ** 2))
            policy = decision.policy_id or "none"
            if selected_policies and policy != selected_policies[-1]:
                policy_switches += 1
            selected_policies.append(policy)
            infeasible_count += int(not decision.feasible)
            (
                selector_fallback,
                solver_fallback,
                backup_executed,
                shield_active,
            ) = _decision_metric_flags(parsed_method, decision)
            selector_fallback_count += int(selector_fallback)
            solver_fallback_count += int(solver_fallback)
            backup_executed_count += int(backup_executed)
            shield_active_count += int(shield_active)
            intervention_values.append(intervention)
            certificate_times.append(certificate_elapsed)
            solver_times.append(solver_elapsed)
            full_step_times.append(certificate_elapsed + solver_elapsed)
            certificate_counts.append(certificate_count)

            previous_position = state[:3].copy()
            state = model.step(state, control)
            if not np.all(np.isfinite(state)):
                raise FloatingPointError("nonlinear dynamics produced non-finite state")
            tilt = _body_tilt_rad(state)
            body_rate_norm = float(np.linalg.norm(state[9:12]))
            abs_body_yaw_rate = abs(float(state[11]))
            max_tilt_rad = max(max_tilt_rad, tilt)
            max_body_rate_norm_rad_s = max(
                max_body_rate_norm_rad_s, body_rate_norm
            )
            max_abs_body_yaw_rate_rad_s = max(
                max_abs_body_yaw_rate_rad_s, abs_body_yaw_rate
            )
            tilt_max_violated = tilt_max_violated or tilt > tilt_max_rad
            path_length += float(np.linalg.norm(state[:3] - previous_position))
            obstacles = advance_obstacles(
                obstacles, model.dt, current_scenario.bounds
            )
            clearance = minimum_clearance(
                model.safety_point(state),
                obstacles,
                model.config.robot_radius,
            )
            min_clearance_value = min(min_clearance_value, clearance)
            collision = collision or clearance < 0.0
            if collision and stop_on_collision:
                break
            if tilt_max_violated:
                break

            if np.linalg.norm(state[:3] - goal) <= current_scenario.reach_threshold:
                if waypoint_index + 1 < current_scenario.waypoints.shape[0]:
                    waypoint_index += 1
                else:
                    reached_goal = True
                    break
    except Exception as error:  # A failed method must not abort the comparison grid.
        error_message = f"{type(error).__name__}: {error}"
    finally:
        np.random.set_state(numpy_random_state)
        random.setstate(python_random_state)

    control_steps = len(intervention_values)
    completed_waypoints = waypoint_index + int(reached_goal)
    final_goal_distance = float(
        np.linalg.norm(state[:3] - current_scenario.goal)
    )
    metrics = _trial_metrics(
        scenario=current_scenario,
        seed=seed,
        control_steps=control_steps,
        reached_goal=reached_goal,
        completed_waypoints=completed_waypoints,
        final_goal_distance=final_goal_distance,
        path_length=path_length,
        intervention_values=intervention_values,
        certificate_times=certificate_times,
        solver_times=solver_times,
        full_step_times=full_step_times,
        certificate_counts=certificate_counts,
        infeasible_count=infeasible_count,
        selector_fallback_count=selector_fallback_count,
        solver_fallback_count=solver_fallback_count,
        backup_executed_count=backup_executed_count,
        shield_active_count=shield_active_count,
        mi_mpc_result_count=(
            mi_mpc_result_count
            if parsed_method is BenchmarkMethod.MI_MPC
            else None
        ),
        mi_mpc_result_missing_count=(
            mi_mpc_result_missing_count
            if parsed_method is BenchmarkMethod.MI_MPC
            else None
        ),
        mi_mpc_requested_safety_feasible_count=(
            mi_mpc_requested_safety_feasible_count
            if parsed_method is BenchmarkMethod.MI_MPC
            else None
        ),
        mi_mpc_safety_threshold_relaxed_count=(
            mi_mpc_safety_threshold_relaxed_count
            if parsed_method is BenchmarkMethod.MI_MPC
            else None
        ),
        policy_switches=policy_switches,
        max_tilt_rad=max_tilt_rad,
        attitude_bound_rad=attitude_bound_rad,
        tilt_max_rad=tilt_max_rad,
        tilt_max_violated=tilt_max_violated,
        max_body_rate_norm_rad_s=max_body_rate_norm_rad_s,
        body_rate_max_rad_s=body_rate_max_rad_s,
        max_abs_body_yaw_rate_rad_s=max_abs_body_yaw_rate_rad_s,
        nominal_yaw_slew_max_rad_s=nominal_yaw_slew_max_rad_s,
    )

    if error_message is not None:
        outcome = BenchmarkOutcome.ERROR
    elif collision:
        outcome = BenchmarkOutcome.COLLISION
    elif tilt_max_violated:
        outcome = BenchmarkOutcome.INFEASIBLE
    elif reached_goal:
        outcome = BenchmarkOutcome.SUCCESS
    elif solver_fallback_count:
        outcome = BenchmarkOutcome.INFEASIBLE
    else:
        outcome = BenchmarkOutcome.TIMEOUT

    return BenchmarkResult(
        algorithm=parsed_method.value,
        case_id=_case_id(current_scenario, seed),
        seed=seed,
        outcome=outcome,
        min_clearance=_finite_clearance(min_clearance_value),
        intervention=(
            _mean(intervention_values) if intervention_values else None
        ),
        solve_times_s=tuple(full_step_times),
        case_metrics=metrics,
        error=error_message,
    )


def run_benchmark(
    config: NLQuad3DBenchmarkConfig,
    *,
    model_factory: ModelFactory = NLQuad3D,
    controller_factory: ControllerFactory = _default_controller_factory,
    scenario_loader: ScenarioLoader = get_scenario,
    clock: Clock = time.perf_counter,
) -> tuple[BenchmarkResult, ...]:
    """Run the common method grid, materializing a fresh scenario per method."""

    results = []
    for scenario_name in config.scenarios:
        for seed in config.seeds:
            for method in config.methods:
                trial_controller_config = (
                    config.plcbf_controller_config
                    if method == BenchmarkMethod.PLCBF.value
                    else config.controller_config
                )
                results.append(
                    run_trial(
                        method,
                        scenario_name,
                        seed=seed,
                        max_steps=config.max_steps,
                        controller_config=trial_controller_config,
                        safe_value_threshold=config.safe_value_threshold,
                        obstacle_position_perturbation=(
                            config.obstacle_position_perturbation
                        ),
                        obstacle_velocity_perturbation=(
                            config.obstacle_velocity_perturbation
                        ),
                        playground_obstacle_count=(
                            config.playground_obstacle_count
                        ),
                        tilt_max_rad=config.tilt_max_rad,
                        warmup=config.warmup,
                        stop_on_collision=config.stop_on_collision,
                        model_factory=model_factory,
                        controller_factory=controller_factory,
                        scenario_loader=scenario_loader,
                        clock=clock,
                    )
                )
    return tuple(results)


def run_and_write(
    config: NLQuad3DBenchmarkConfig,
    output_prefix: str | Path,
) -> tuple[tuple[BenchmarkResult, ...], BenchmarkReportPaths]:
    """Run the benchmark and write deterministic CSV, JSON, and Markdown."""

    results = run_benchmark(config)
    paths = write_benchmark_reports(
        output_prefix,
        results,
        metadata=config.metadata(),
        title="Nonlinear Quad3D benchmark",
    )
    return results, paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        action="append",
        choices=BENCHMARK_METHODS,
        help="method to run; repeat to select several (default: all eight)",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=scenario_names(),
        help="scenario to run; repeat for a grid",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="deterministic seed/provenance value; repeat for a grid",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument(
        "--config",
        "--config-json",
        dest="config_json",
        type=Path,
        help=(
            "PL-CBF controller YAML/JSON or tuning summary (default: "
            "packaged Optuna winner); comparison baselines are unchanged"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/nl_quad3d_benchmark"),
        help="report prefix; .csv, .json, and .md are written",
    )
    parser.add_argument("--radial-policies", type=int, default=None)
    parser.add_argument("--backup-horizon", type=float, default=None)
    parser.add_argument(
        "--position-perturbation",
        type=float,
        help=(
            "bounded offset for static scenarios only; seeded crowded/stress "
            "scenarios procedurally regenerate the full field"
        ),
    )
    parser.add_argument(
        "--velocity-perturbation",
        type=float,
        help=(
            "bounded offset for static scenarios only; seeded crowded/stress "
            "scenarios procedurally regenerate the full field"
        ),
    )
    parser.add_argument(
        "--obstacle-count",
        type=int,
        help=(
            "total spheres in a generated playground scenario (defaults: "
            "48 for playground_stress, 32 for playground_crowded)"
        ),
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="one-step, small-policy smoke configuration",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> NLQuad3DBenchmarkConfig:
    artifact: Mapping[str, object] = {}
    baseline_controller = NLQuad3DControllerConfig()
    config_path = (
        args.config_json
        if args.config_json is not None
        else DEFAULT_CONTROLLER_CONFIG_PATH
    )
    plcbf_controller, artifact = load_controller_config_artifact(config_path)
    tuning_metadata = artifact.get("configuration", {})
    if not isinstance(tuning_metadata, Mapping):
        tuning_metadata = {}
    methods = tuple(args.method) if args.method else BENCHMARK_METHODS
    artifact_scenarios = tuning_metadata.get("scenarios")
    scenarios = (
        tuple(args.scenario)
        if args.scenario
        else (
            (PLAYGROUND_CROWDED_SCENARIO,)
            if args.quick
            else (
                tuple(str(item) for item in artifact_scenarios)
                if isinstance(artifact_scenarios, list)
                else DEFAULT_BENCHMARK_SCENARIOS
            )
        )
    )
    seeds = tuple(args.seed) if args.seed else (0,)
    artifact_steps = tuning_metadata.get("max_steps")
    steps = (
        args.steps
        if args.steps is not None
        else (
            int(artifact_steps)
            if isinstance(artifact_steps, int)
            else None
        )
    )
    radial_policies = args.radial_policies
    backup_horizon = args.backup_horizon
    plcbf_max_obstacles = plcbf_controller.max_obstacles
    plcbf_nominal_prefix_steps = plcbf_controller.nominal_prefix_steps
    if args.quick:
        steps = 1 if args.steps is None else args.steps
        radial_policies = 2 if radial_policies is None else radial_policies
        backup_horizon = 0.1 if backup_horizon is None else backup_horizon
        plcbf_max_obstacles = 2
        plcbf_nominal_prefix_steps = 0
        baseline_controller = replace(
            baseline_controller,
            num_radial_policies=radial_policies,
            backup_horizon=backup_horizon,
            max_obstacles=2,
            nominal_prefix_steps=0,
        )
    plcbf_controller = replace(
        plcbf_controller,
        num_radial_policies=(
            plcbf_controller.num_radial_policies
            if radial_policies is None
            else radial_policies
        ),
        backup_horizon=(
            plcbf_controller.backup_horizon
            if backup_horizon is None
            else backup_horizon
        ),
        max_obstacles=plcbf_max_obstacles,
        nominal_prefix_steps=plcbf_nominal_prefix_steps,
    )
    perturbations = tuning_metadata.get("seed_perturbations", {})
    if not isinstance(perturbations, Mapping):
        perturbations = {}
    position_perturbation = (
        args.position_perturbation
        if args.position_perturbation is not None
        else float(perturbations.get("position_uniform_half_width_m", 0.12))
    )
    velocity_perturbation = (
        args.velocity_perturbation
        if args.velocity_perturbation is not None
        else float(perturbations.get("velocity_uniform_half_width_mps", 0.08))
    )
    artifact_warmup = tuning_metadata.get("warmup", True)
    headline_metadata = tuning_metadata.get("headline_scenario", {})
    if not isinstance(headline_metadata, Mapping):
        headline_metadata = {}
    artifact_obstacle_count = tuning_metadata.get(
        "playground_obstacle_count",
        headline_metadata.get("obstacle_count"),
    )
    playground_obstacle_count = (
        args.obstacle_count
        if args.obstacle_count is not None
        else (
            int(artifact_obstacle_count)
            if isinstance(artifact_obstacle_count, int)
            else None
        )
    )
    return NLQuad3DBenchmarkConfig(
        methods=methods,
        scenarios=scenarios,
        seeds=seeds,
        max_steps=steps,
        controller_config=baseline_controller,
        plcbf_controller_config=plcbf_controller,
        obstacle_position_perturbation=position_perturbation,
        obstacle_velocity_perturbation=velocity_perturbation,
        playground_obstacle_count=playground_obstacle_count,
        warmup=bool(artifact_warmup) and not args.no_warmup,
        configuration_source=(
            f"packaged_plcbf:{config_path}"
            if args.config_json is None
            else str(config_path)
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    results, paths = run_and_write(config, args.output)
    aggregates = aggregate_results(results)
    print(
        json.dumps(
            {
                "trials": len(results),
                "methods": list(config.methods),
                "scenarios": list(config.scenarios),
                "seeds": list(config.seeds),
                "successes": {
                    item.algorithm: item.success_count for item in aggregates
                },
                "reports": {
                    "csv": str(paths.csv.resolve()),
                    "json": str(paths.json.resolve()),
                    "markdown": str(paths.markdown.resolve()),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return int(
        any(result.outcome is BenchmarkOutcome.ERROR for result in results)
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BENCHMARK_SCENARIOS",
    "NLQuad3DBenchmarkConfig",
    "build_parser",
    "main",
    "run_and_write",
    "run_benchmark",
    "run_trial",
    "seeded_scenario",
]
