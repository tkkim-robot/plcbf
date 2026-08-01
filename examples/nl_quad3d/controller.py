"""PLCBF controller with batched differentiable nonlinear-quadrotor rollouts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import importlib
import math
from typing import Any, Callable, Iterable, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from .dynamics import NLQuad3D, NLQuad3DConfig
from .dynamics_jax import (
    NLQuad3DJaxParams,
    control_matrix_jax,
    drift_jax,
    jax_params,
    safety_point_jax,
    step_jax,
)
from .policies import (
    PolicyBatch,
    PolicyCandidate,
    candidates_to_batch,
    make_default_candidates,
    policy_control_jax,
)
from .scenarios import WorldBounds, advance_obstacles
from plcbf.policy_library import (
    DecisionDiagnostics,
    PolicyCertificate,
    PolicyDecision,
    PolicyEvaluation,
    SelectionMode,
    solve_box_halfspace_qp,
)


class PolicyProvider(Protocol):
    """Narrow extension point shared policy libraries can implement."""

    def __call__(
        self,
        *,
        goal: np.ndarray,
        model: NLQuad3D,
        controller_config: "NLQuad3DControllerConfig",
    ) -> Iterable[PolicyCandidate]:
        ...


@dataclass(frozen=True)
class NLQuad3DControllerConfig:
    """PLCBF rollout, selection, and filtering parameters."""

    dt: float = 0.05
    backup_horizon: float = 1.25
    cbf_alpha: float = 2.2
    cbf_value_buffer: float = 0.2
    safety_margin: float = 0.18
    safety_scale: float = 1.15
    sensing_radius: float = 3.8
    max_obstacles: int = 8
    num_radial_policies: int = 12
    target_speed: float = 2.8
    radial_gain: float = 2.6
    stop_gain: float = 3.0
    obstacle_temperature: float = 75.0
    time_temperature: float = 70.0
    max_gradient_norm: float = 180.0
    min_lg_norm: float = 1e-4
    constraint_tolerance: float = 1e-4
    rollout_tilt_max_rad: float = np.deg2rad(60.0)
    max_operator: str = "input_space"
    nominal_prefix_steps: int = 1

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.backup_horizon <= 0.0:
            raise ValueError("dt and backup_horizon must be positive")
        if self.max_obstacles < 1 or self.num_radial_policies < 1:
            raise ValueError("policy and obstacle counts must be positive")
        if self.safety_margin < 0.0 or self.safety_scale < 1.0:
            raise ValueError(
                "safety_margin must be non-negative and safety_scale must be at least one"
            )
        if self.nominal_prefix_steps < 0:
            raise ValueError("nominal_prefix_steps must be non-negative")
        if (
            not np.isfinite(self.rollout_tilt_max_rad)
            or self.rollout_tilt_max_rad <= 0.0
            or self.rollout_tilt_max_rad > np.pi
        ):
            raise ValueError(
                "rollout_tilt_max_rad must be in the interval (0, pi]"
            )
        if self.max_operator not in {"value", "input_space"}:
            raise ValueError("max_operator must be 'value' or 'input_space'")

    @property
    def horizon_steps(self) -> int:
        # Avoid losing a step to binary roundoff (for example 0.3 / 0.05).
        return max(1, math.floor(self.backup_horizon / self.dt + 1e-9))


@dataclass(frozen=True)
class RolloutEvaluation:
    """Host-side policy evaluation and CBF data for one control cycle."""

    names: tuple[str, ...]
    values: np.ndarray
    state_gradients: np.ndarray
    obstacle_gradients: np.ndarray
    trajectories: np.ndarray
    time_derivatives: np.ndarray
    control_directions: np.ndarray
    constraint_rhs: np.ndarray
    scores: np.ndarray
    selected_index: int

    @property
    def selected_name(self) -> str:
        return self.names[self.selected_index]

    @property
    def selected_value(self) -> float:
        return float(self.values[self.selected_index])


def _advance_obstacles_jax(
    obstacles: jnp.ndarray,
    dt: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    bounce: jnp.ndarray,
) -> jnp.ndarray:
    positions = obstacles[:, :3]
    radii = obstacles[:, 3:4]
    velocities = obstacles[:, 4:7]
    linear = positions + velocities * dt
    low = lower[None, :] + radii
    high = upper[None, :] - radii
    span = high - low
    phase = jnp.mod(linear - low, 2.0 * span)
    forward = phase <= span
    reflected = low + jnp.where(forward, phase, 2.0 * span - phase)
    reflected_velocity = jnp.where(forward, velocities, -velocities)
    positions_next = jnp.where(bounce, reflected, linear)
    velocities_next = jnp.where(bounce, reflected_velocity, velocities)
    return jnp.concatenate(
        [positions_next, radii, velocities_next],
        axis=1,
    )


def smooth_min_jax(
    values: jnp.ndarray,
    temperature: float | jnp.ndarray,
) -> jnp.ndarray:
    """Numerically stable differentiable minimum."""

    minimum = jnp.min(values)
    return minimum - jnp.log(
        jnp.sum(jnp.exp(-temperature * (values - minimum)))
    ) / temperature


def rollout_policy_jax(
    initial_state: jnp.ndarray,
    obstacles: jnp.ndarray,
    kind: jnp.ndarray,
    direction: jnp.ndarray,
    target_speed: jnp.ndarray,
    gain: jnp.ndarray,
    goal: jnp.ndarray,
    dynamics_params: NLQuad3DJaxParams,
    dt: float | jnp.ndarray,
    horizon_steps: int,
    bounds_lower: jnp.ndarray,
    bounds_upper: jnp.ndarray,
    bounce_obstacles: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Roll out a policy and the same moving obstacle field with ``lax.scan``."""

    def scan_step(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        _: None,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
        state, current_obstacles = carry
        control = policy_control_jax(
            state,
            kind,
            direction,
            target_speed,
            gain,
            goal,
            dynamics_params,
        )
        next_state = step_jax(state, control, dt, dynamics_params)
        next_obstacles = _advance_obstacles_jax(
            current_obstacles,
            jnp.asarray(dt),
            bounds_lower,
            bounds_upper,
            bounce_obstacles,
        )
        return (next_state, next_obstacles), (next_state, next_obstacles)

    _, (state_tail, obstacle_tail) = jax.lax.scan(
        scan_step,
        (initial_state, obstacles),
        xs=None,
        length=horizon_steps,
    )
    return (
        jnp.concatenate([initial_state[None, :], state_tail], axis=0),
        jnp.concatenate([obstacles[None, :, :], obstacle_tail], axis=0),
    )


def trajectory_value_jax(
    state_trajectory: jnp.ndarray,
    obstacle_trajectory: jnp.ndarray,
    dynamics_params: NLQuad3DJaxParams,
    safety_margin: float | jnp.ndarray,
    safety_scale: float | jnp.ndarray,
    obstacle_temperature: float | jnp.ndarray,
    time_temperature: float | jnp.ndarray,
    obstacle_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Smooth minimum signed sphere clearance along a rollout."""

    points = jax.vmap(safety_point_jax, in_axes=(0, None))(
        state_trajectory,
        dynamics_params,
    )
    relative_positions = (
        points[:, None, :] - obstacle_trajectory[:, :, :3]
    )
    # ``sqrt(sum(x²) + eps)`` keeps the clearance derivative finite even if a
    # rollout passes exactly through an obstacle center.
    distances = jnp.sqrt(
        jnp.sum(jnp.square(relative_positions), axis=-1) + 1e-12
    )
    safe_radii = (
        obstacle_trajectory[:, :, 3]
        + dynamics_params.robot_radius
        + safety_margin
    ) * safety_scale
    clearances = distances - safe_radii
    if obstacle_mask is not None:
        # The production evaluator pads every sensed set to ``max_obstacles``
        # so changing the number of nearby spheres never changes an XLA input
        # shape.  ``+inf`` contributes exactly zero weight to the smooth min;
        # the real-obstacle values and derivatives are therefore unchanged.
        clearances = jnp.where(
            jnp.asarray(obstacle_mask, dtype=bool)[None, :],
            clearances,
            jnp.inf,
        )
    obstacle_values = jax.vmap(smooth_min_jax, in_axes=(0, None))(
        clearances,
        obstacle_temperature,
    )
    return smooth_min_jax(obstacle_values, time_temperature)


def _policy_value_with_aux(
    state: jnp.ndarray,
    obstacles: jnp.ndarray,
    kind: jnp.ndarray,
    direction: jnp.ndarray,
    target_speed: jnp.ndarray,
    gain: jnp.ndarray,
    goal: jnp.ndarray,
    dynamics_params: NLQuad3DJaxParams,
    dt: float | jnp.ndarray,
    horizon_steps: int,
    bounds_lower: jnp.ndarray,
    bounds_upper: jnp.ndarray,
    bounce_obstacles: jnp.ndarray,
    safety_margin: float | jnp.ndarray,
    safety_scale: float | jnp.ndarray,
    obstacle_temperature: float | jnp.ndarray,
    time_temperature: float | jnp.ndarray,
    obstacle_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    states, obstacle_states = rollout_policy_jax(
        state,
        obstacles,
        kind,
        direction,
        target_speed,
        gain,
        goal,
        dynamics_params,
        dt,
        horizon_steps,
        bounds_lower,
        bounds_upper,
        bounce_obstacles,
    )
    value = trajectory_value_jax(
        states,
        obstacle_states,
        dynamics_params,
        safety_margin,
        safety_scale,
        obstacle_temperature,
        time_temperature,
        obstacle_mask,
    )
    return value, states


def batched_policy_values_and_gradients_jax(
    state: jnp.ndarray,
    obstacles: jnp.ndarray,
    policy_batch: PolicyBatch,
    goal: jnp.ndarray,
    dynamics_params: NLQuad3DJaxParams,
    *,
    dt: float,
    horizon_steps: int,
    bounds_lower: jnp.ndarray,
    bounds_upper: jnp.ndarray,
    bounce_obstacles: jnp.ndarray,
    safety_margin: float,
    safety_scale: float,
    obstacle_temperature: float,
    time_temperature: float,
    obstacle_mask: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate all candidates and differentiate values w.r.t. state/obstacles."""

    value_grad = jax.value_and_grad(
        _policy_value_with_aux,
        argnums=(0, 1),
        has_aux=True,
    )
    # ``policy_control_jax`` uses a finite branchless selection, so policies
    # can be evaluated in parallel.  This preserves independent-policy
    # gradients while avoiding the sequential runtime of ``lax.map``.
    def evaluate_one(
        policy: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[
        tuple[jnp.ndarray, jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray],
    ]:
        kind, direction, target_speed, gain = policy
        return value_grad(
            state,
            obstacles,
            kind,
            direction,
            target_speed,
            gain,
            goal,
            dynamics_params,
            dt,
            horizon_steps,
            bounds_lower,
            bounds_upper,
            bounce_obstacles,
            safety_margin,
            safety_scale,
            obstacle_temperature,
            time_temperature,
            obstacle_mask,
        )

    (values, trajectories), (state_gradients, obstacle_gradients) = jax.vmap(
        evaluate_one,
    )(
        (
            policy_batch.kinds,
            policy_batch.directions,
            policy_batch.target_speeds,
            policy_batch.gains,
        )
    )
    return values, state_gradients, obstacle_gradients, trajectories


@lru_cache(maxsize=16)
def _shared_compiled_evaluator(
    policy_count: int,
    max_obstacles: int,
    horizon_steps: int,
) -> Callable[..., tuple[jnp.ndarray, ...]]:
    """Return one process-wide executable for a fixed array structure.

    Controllers are recreated for every benchmark trial.  Keeping the jitted
    function on an instance therefore recompiles identical XLA programs for
    every seed.  Only loop length and array shapes are structural; controller
    gains, temperatures, margins, time step, and obstacle-bounce mode remain
    dynamic arguments so Optuna trials also reuse this executable.
    """

    del policy_count, max_obstacles  # encoded by the input shapes/cache key

    def evaluate(
        state: jnp.ndarray,
        obstacles: jnp.ndarray,
        policy_batch: PolicyBatch,
        goal: jnp.ndarray,
        dynamics_params: NLQuad3DJaxParams,
        bounds_lower: jnp.ndarray,
        bounds_upper: jnp.ndarray,
        obstacle_mask: jnp.ndarray,
        dt: jnp.ndarray,
        bounce_obstacles: jnp.ndarray,
        safety_margin: jnp.ndarray,
        safety_scale: jnp.ndarray,
        obstacle_temperature: jnp.ndarray,
        time_temperature: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return batched_policy_values_and_gradients_jax(
            state,
            obstacles,
            policy_batch,
            goal,
            dynamics_params,
            dt=dt,
            horizon_steps=horizon_steps,
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            bounce_obstacles=bounce_obstacles,
            safety_margin=safety_margin,
            safety_scale=safety_scale,
            obstacle_temperature=obstacle_temperature,
            time_temperature=time_temperature,
            obstacle_mask=obstacle_mask,
        )

    return jax.jit(evaluate)


def _project_box_halfspace(
    reference: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    normal: np.ndarray,
    rhs: float,
) -> tuple[np.ndarray, bool, str]:
    """Euclidean projection onto ``box ∩ {normal @ u >= rhs}``."""

    clipped = np.clip(reference, lower, upper)
    if float(normal @ clipped) >= rhs:
        return clipped, True, "nominal" if np.allclose(clipped, reference) else "box"
    maximizing_corner = np.where(normal >= 0.0, upper, lower)
    if float(normal @ maximizing_corner) < rhs - 1e-10:
        return maximizing_corner, False, "infeasible"
    if float(np.linalg.norm(normal)) < 1e-12:
        return clipped, rhs <= 0.0, "degenerate"

    def point(multiplier: float) -> np.ndarray:
        return np.clip(clipped + multiplier * normal, lower, upper)

    low_multiplier, high_multiplier = 0.0, 1.0
    while float(normal @ point(high_multiplier)) < rhs:
        high_multiplier *= 2.0
    for _ in range(80):
        midpoint = 0.5 * (low_multiplier + high_multiplier)
        if float(normal @ point(midpoint)) < rhs:
            low_multiplier = midpoint
        else:
            high_multiplier = midpoint
    return point(high_multiplier), True, "filtered"


def _halfspace_box_measure(
    normal: np.ndarray,
    rhs: float,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    epsilon: float = 1e-9,
) -> float:
    """Exact 4-D feasible volume used by the input-space max operator."""

    a = np.asarray(normal, dtype=float).copy()
    lo = np.asarray(lower, dtype=float).copy()
    hi = np.asarray(upper, dtype=float).copy()
    widths = hi - lo
    full_volume = float(np.prod(widths))
    if np.linalg.norm(a) < epsilon:
        return full_volume if rhs <= 0.0 else 0.0
    negative = a < 0.0
    old_lo = lo.copy()
    lo[negative] = -hi[negative]
    hi[negative] = -old_lo[negative]
    a[negative] *= -1.0
    widths = hi - lo
    active = (a > epsilon) & (widths > epsilon)
    if not np.any(active):
        return full_volume if rhs <= 0.0 else 0.0
    inactive_volume = (
        float(np.prod(widths[~active])) if np.any(~active) else 1.0
    )
    active_a = a[active]
    active_lo = lo[active]
    active_widths = widths[active]
    weighted_widths = active_a * active_widths
    threshold = rhs - float(active_a @ active_lo)
    total_weight = float(np.sum(weighted_widths))
    if threshold <= 0.0:
        return full_volume
    if threshold >= total_weight:
        return 0.0
    dimension = active_a.size
    cdf_numerator = 0.0
    for mask in range(1 << dimension):
        shifted = threshold
        bits = 0
        for axis in range(dimension):
            if mask & (1 << axis):
                shifted -= weighted_widths[axis]
                bits += 1
        if shifted > 0.0:
            cdf_numerator += (-1.0) ** bits * shifted**dimension
    denominator = math.factorial(dimension) * float(np.prod(weighted_widths))
    feasible_probability = float(
        np.clip(1.0 - cdf_numerator / denominator, 0.0, 1.0)
    )
    return feasible_probability * float(np.prod(active_widths)) * inactive_volume


def _normalize_candidate(candidate: Any) -> PolicyCandidate:
    if isinstance(candidate, PolicyCandidate):
        return candidate
    if isinstance(candidate, dict):
        return PolicyCandidate(**candidate)
    required = ("name", "kind")
    if all(hasattr(candidate, attribute) for attribute in required):
        return PolicyCandidate(
            name=str(candidate.name),
            kind=str(candidate.kind),
            direction=tuple(getattr(candidate, "direction", (0.0, 0.0, 0.0))),
            target_speed=float(getattr(candidate, "target_speed", 0.0)),
            gain=float(getattr(candidate, "gain", 1.0)),
        )
    raise TypeError(f"cannot convert {type(candidate).__name__} to PolicyCandidate")


class PLCBF_NLQuad3D:
    """Multiple-policy CBF safety filter for the nonlinear quadrotor."""

    def __init__(
        self,
        model: NLQuad3D | None = None,
        controller_config: NLQuad3DControllerConfig | None = None,
        *,
        bounds: WorldBounds | None = WorldBounds(),
        candidate_provider: PolicyProvider | None = None,
        use_shared_policy_library: bool = True,
    ):
        self.model = model or NLQuad3D()
        self.config = controller_config or NLQuad3DControllerConfig(dt=self.model.dt)
        self.bounds = bounds
        self._dynamics_params = jax_params(self.model.config)
        self._candidate_provider = candidate_provider
        self.policy_provider_source = (
            "custom" if candidate_provider is not None else "local"
        )
        if candidate_provider is None and use_shared_policy_library:
            self._candidate_provider = self._discover_shared_provider()
        self.last_evaluation: RolloutEvaluation | None = None
        self.last_certificates: tuple[PolicyCertificate, ...] = ()
        self.last_decision: PolicyDecision | None = None
        self.last_status = "uninitialized"
        self.last_control = self.model.hover_input.copy()
        self._compiled_evaluators: dict[
            tuple[int, int, bool],
            Callable[..., tuple[jnp.ndarray, ...]],
        ] = {}
        self._policy_batches: dict[tuple[PolicyCandidate, ...], PolicyBatch] = {}
        if self.bounds is None:
            bounds_lower = np.full(3, -1e6)
            bounds_upper = np.full(3, 1e6)
        else:
            bounds_lower = np.asarray(self.bounds.lower, dtype=float)
            bounds_upper = np.asarray(self.bounds.upper, dtype=float)
        # Immutable device values are reused at every decision.  Besides
        # reducing host dispatch overhead this standardizes dtypes/signatures.
        self._bounds_lower_jax = jnp.asarray(bounds_lower)
        self._bounds_upper_jax = jnp.asarray(bounds_upper)
        self._dt_jax = jnp.asarray(self.config.dt)
        self._bounce_jax = jnp.asarray(self.bounds is not None)
        self._safety_margin_jax = jnp.asarray(self.config.safety_margin)
        self._safety_scale_jax = jnp.asarray(self.config.safety_scale)
        self._obstacle_temperature_jax = jnp.asarray(
            self.config.obstacle_temperature
        )
        self._time_temperature_jax = jnp.asarray(self.config.time_temperature)

    def _discover_shared_provider(self) -> PolicyProvider | None:
        """Use a shared provider when it exposes the documented narrow hook."""

        try:
            module = importlib.import_module("plcbf.policy_library")
        except (ImportError, ModuleNotFoundError):
            return None
        provider = getattr(module, "build_nl_quad3d_candidates", None)
        if callable(provider):
            self.policy_provider_source = "plcbf.policy_library"
            return provider
        return None

    def _local_candidates(self) -> tuple[PolicyCandidate, ...]:
        return make_default_candidates(
            num_radial=self.config.num_radial_policies,
            target_speed=self.config.target_speed,
            radial_gain=self.config.radial_gain,
            stop_gain=self.config.stop_gain,
            velocity_limit=self.model.config.v_max,
        )

    def candidates(self, goal: np.ndarray) -> tuple[PolicyCandidate, ...]:
        if self._candidate_provider is None:
            return self._local_candidates()
        supplied = self._candidate_provider(
            goal=np.asarray(goal, dtype=float).reshape(3),
            model=self.model,
            controller_config=self.config,
        )
        normalized = tuple(_normalize_candidate(item) for item in supplied)
        if not normalized:
            raise ValueError("candidate provider returned an empty policy library")
        return normalized

    def _active_obstacles(
        self,
        state: np.ndarray,
        obstacles: np.ndarray,
    ) -> np.ndarray:
        obstacle_array = np.asarray(obstacles, dtype=float)
        if obstacle_array.size == 0:
            return np.zeros((0, 7), dtype=float)
        obstacle_array = obstacle_array.reshape(-1, 7)
        safety_point = self.model.safety_point(state)
        ranges = np.linalg.norm(
            obstacle_array[:, :3] - safety_point[None, :],
            axis=1,
        )
        sensed = ranges - obstacle_array[:, 3] <= self.config.sensing_radius
        indices = np.flatnonzero(sensed)
        indices = indices[np.argsort(ranges[indices])]
        return obstacle_array[indices[: self.config.max_obstacles]].copy()

    def _compiled_evaluator(
        self,
        policy_count: int,
    ) -> Callable[..., tuple[jnp.ndarray, ...]]:
        bounce = self.bounds is not None
        key = (
            policy_count,
            self.config.horizon_steps,
            bounce,
        )
        if key not in self._compiled_evaluators:
            self._compiled_evaluators[key] = _shared_compiled_evaluator(
                policy_count,
                self.config.max_obstacles,
                self.config.horizon_steps,
            )
        return self._compiled_evaluators[key]

    def evaluate_policies(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
        *,
        candidates: Iterable[PolicyCandidate] | None = None,
    ) -> RolloutEvaluation:
        """Return rollout values, exact JAX gradients, and CBF halfspaces."""

        state_array = np.asarray(state, dtype=float).reshape(12)
        goal_array = np.asarray(goal, dtype=float).reshape(3)
        active = self._active_obstacles(state_array, obstacles)
        if active.shape[0] == 0:
            raise ValueError("policy evaluation requires at least one sensed obstacle")
        candidate_tuple = (
            self.candidates(goal_array)
            if candidates is None
            else tuple(_normalize_candidate(item) for item in candidates)
        )
        if not candidate_tuple:
            raise ValueError("at least one policy candidate is required")
        policy_batch = self._policy_batches.get(candidate_tuple)
        if policy_batch is None:
            policy_batch = candidates_to_batch(candidate_tuple)
            self._policy_batches[candidate_tuple] = policy_batch
        padded_obstacles = np.zeros((self.config.max_obstacles, 7), dtype=float)
        padded_obstacles[: active.shape[0]] = active
        obstacle_mask = np.arange(self.config.max_obstacles) < active.shape[0]
        evaluator = self._compiled_evaluator(len(candidate_tuple))
        values_jax, gradients_jax, obstacle_gradients_jax, trajectories_jax = (
            evaluator(
                jnp.asarray(state_array),
                jnp.asarray(padded_obstacles),
                policy_batch,
                jnp.asarray(goal_array),
                self._dynamics_params,
                self._bounds_lower_jax,
                self._bounds_upper_jax,
                jnp.asarray(obstacle_mask),
                self._dt_jax,
                self._bounce_jax,
                self._safety_margin_jax,
                self._safety_scale_jax,
                self._obstacle_temperature_jax,
                self._time_temperature_jax,
            )
        )
        values = np.asarray(values_jax, dtype=float)
        gradients = np.asarray(gradients_jax, dtype=float)
        obstacle_gradients = np.asarray(
            obstacle_gradients_jax[:, : active.shape[0], :],
            dtype=float,
        )
        trajectories = np.asarray(trajectories_jax, dtype=float)
        gradient_norms = np.linalg.norm(gradients, axis=1)
        scales = np.minimum(
            1.0,
            self.config.max_gradient_norm / np.maximum(gradient_norms, 1e-12),
        )
        # The playground clips only dV/dx.  The explicit time derivative is
        # evaluated independently, so do not rescale dV/d(obstacle) with dV/dx.
        gradients = gradients * scales[:, None]

        drift = self.model.f(state_array)
        control_matrix = self.model.g(state_array)
        time_derivatives = np.einsum(
            "nmi,mi->n",
            obstacle_gradients[:, :, :3],
            active[:, 4:7],
        )
        control_directions = gradients @ control_matrix
        constraint_rhs = (
            -(gradients @ drift)
            - time_derivatives
            - self.config.cbf_alpha
            * (values - self.config.cbf_value_buffer)
        )
        lower = self.model.input_lower_bound
        upper = self.model.input_upper_bound
        if self.config.max_operator == "input_space":
            scores = np.asarray(
                [
                    _halfspace_box_measure(normal, rhs, lower, upper)
                    if np.linalg.norm(normal) >= self.config.min_lg_norm
                    else value
                    for normal, rhs, value in zip(
                        control_directions,
                        constraint_rhs,
                        values,
                        strict=True,
                    )
                ]
            )
        else:
            scores = values.copy()

        selected = 0
        for index in range(1, len(values)):
            value = float(values[index])
            best_value = float(values[selected])
            score = float(scores[index])
            best_score = float(scores[selected])
            if value > 0.0 and best_value > 0.0:
                if score > best_score + 1e-6:
                    selected = index
            elif value > 0.0 and best_value <= 0.0:
                selected = index
            elif value <= 0.0 and best_value <= 0.0:
                if value > best_value + 1e-6 or (
                    abs(value - best_value) <= 1e-6
                    and score > best_score + 1e-6
                ):
                    selected = index
        evaluation = RolloutEvaluation(
            names=tuple(candidate.name for candidate in candidate_tuple),
            values=values,
            state_gradients=gradients,
            obstacle_gradients=obstacle_gradients,
            trajectories=trajectories,
            time_derivatives=time_derivatives,
            control_directions=control_directions,
            constraint_rhs=constraint_rhs,
            scores=scores,
            selected_index=selected,
        )
        self.last_evaluation = evaluation
        return evaluation

    def _host_rollout(
        self,
        state: np.ndarray,
        obstacles: np.ndarray,
        goal: np.ndarray,
        candidate: PolicyCandidate,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy rollout used only for discrete safety metadata."""

        current_state = np.asarray(state, dtype=float).reshape(12).copy()
        current_obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 7).copy()
        states = [current_state.copy()]
        obstacle_states = [current_obstacles.copy()]
        for _ in range(self.config.horizon_steps):
            control = self._candidate_control(
                current_state,
                goal,
                candidate,
            )
            current_state = self.model.step(current_state, control)
            current_obstacles = advance_obstacles(
                current_obstacles,
                self.config.dt,
                self.bounds,
            )
            states.append(current_state.copy())
            obstacle_states.append(current_obstacles.copy())
        return np.asarray(states), np.asarray(obstacle_states)

    def _clearance_history(
        self,
        states: np.ndarray,
        obstacle_states: np.ndarray,
    ) -> np.ndarray:
        if obstacle_states.shape[1] == 0:
            return np.full((states.shape[0], 1), np.inf)
        points = np.asarray(
            [self.model.safety_point(state) for state in states]
        )
        distances = np.linalg.norm(
            points[:, None, :] - obstacle_states[:, :, :3],
            axis=-1,
        )
        safe_radii = (
            obstacle_states[:, :, 3]
            + self.model.config.robot_radius
            + self.config.safety_margin
        ) * self.config.safety_scale
        return distances - safe_radii

    def _rollout_domain(
        self,
        states: np.ndarray,
    ) -> tuple[bool, float | None, float | None, str]:
        """Check the finite physical envelope used by benchmark certificates."""

        trajectory = np.asarray(states, dtype=float)
        if (
            trajectory.ndim != 2
            or trajectory.shape[1] != 12
            or not np.all(np.isfinite(trajectory))
        ):
            return False, None, None, "nonfinite_rollout"
        tilt = np.arccos(
            np.clip(
                np.cos(trajectory[:, 6]) * np.cos(trajectory[:, 7]),
                -1.0,
                1.0,
            )
        )
        body_rate_norm = np.linalg.norm(trajectory[:, 9:12], axis=1)
        max_tilt = float(np.max(tilt))
        max_body_rate = float(np.max(body_rate_norm))
        violations = []
        if max_tilt > self.config.rollout_tilt_max_rad:
            violations.append("tilt_max")
        if max_body_rate > self.model.config.body_rate_max + 1e-9:
            violations.append("body_rate_max")
        return (
            not violations,
            max_tilt,
            max_body_rate,
            "+".join(violations) if violations else "",
        )

    def _terminal_safe(
        self,
        states: np.ndarray,
        obstacle_states: np.ndarray,
        goal: np.ndarray,
        candidate: PolicyCandidate,
    ) -> bool:
        terminal_state = states[-1]
        terminal_obstacles = obstacle_states[-1]
        next_control = self._candidate_control(
            terminal_state,
            goal,
            candidate,
        )
        next_state = self.model.step(terminal_state, next_control)
        next_obstacles = advance_obstacles(
            terminal_obstacles,
            self.config.dt,
            self.bounds,
        )
        terminal_pair = np.stack([terminal_state, next_state])
        obstacle_pair = np.stack([terminal_obstacles, next_obstacles])
        terminal_domain_valid, _, _, _ = self._rollout_domain(terminal_pair)
        if not terminal_domain_valid:
            return False
        if obstacle_pair.shape[1] == 0:
            return True
        clearances = self._clearance_history(terminal_pair, obstacle_pair)
        return bool(
            np.all(np.isfinite(clearances)) and np.min(clearances) >= 0.0
        )

    def _nominal_prefix_safe(
        self,
        state: np.ndarray,
        obstacles: np.ndarray,
        goal: np.ndarray,
        candidate: PolicyCandidate,
    ) -> int:
        """Count tested nominal steps from which this backup remains safe."""

        prefix_state = np.asarray(state, dtype=float).reshape(12).copy()
        prefix_obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 7).copy()
        safe_steps = 0
        for _ in range(self.config.nominal_prefix_steps):
            nominal = self.model.nominal_input(prefix_state, goal)
            prefix_state = self.model.step(prefix_state, nominal)
            prefix_obstacles = advance_obstacles(
                prefix_obstacles,
                self.config.dt,
                self.bounds,
            )
            states, obstacle_states = self._host_rollout(
                prefix_state,
                prefix_obstacles,
                goal,
                candidate,
            )
            rollout_domain_valid, _, _, _ = self._rollout_domain(states)
            rollout_safe = bool(
                rollout_domain_valid
                and np.min(
                    self._clearance_history(states, obstacle_states)
                )
                >= 0.0
            )
            if not rollout_safe or not self._terminal_safe(
                states,
                obstacle_states,
                goal,
                candidate,
            ):
                break
            safe_steps += 1
        return safe_steps

    def _obstacle_history(
        self,
        obstacles: np.ndarray,
        count: int,
    ) -> np.ndarray:
        current = np.asarray(obstacles, dtype=float).reshape(-1, 7).copy()
        history = [current.copy()]
        for _ in range(count - 1):
            current = advance_obstacles(
                current,
                self.config.dt,
                self.bounds,
            )
            history.append(current.copy())
        return np.asarray(history)

    def _unconstrained_evaluation(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        candidates: tuple[PolicyCandidate, ...],
        *,
        include_trajectories: bool = True,
    ) -> RolloutEvaluation:
        count = len(candidates)
        if include_trajectories:
            trajectories = []
            for candidate in candidates:
                states, _ = self._host_rollout(
                    state,
                    np.zeros((0, 7)),
                    goal,
                    candidate,
                )
                trajectories.append(states)
            trajectory_array = np.asarray(trajectories)
        else:
            # Decision-only certificates never inspect a no-obstacle rollout.
            # Retain a well-shaped initial-state sample for diagnostics APIs.
            trajectory_array = np.broadcast_to(
                np.asarray(state, dtype=float).reshape(1, 1, 12),
                (count, 1, 12),
            ).copy()
        values = np.full(count, 100.0)
        gradients = np.zeros((count, 12))
        control_directions = np.zeros((count, 4))
        rhs = np.full(
            count,
            -self.config.cbf_alpha * (100.0 - self.config.cbf_value_buffer),
        )
        scores = (
            values.copy()
            if self.config.max_operator == "value"
            else np.full(
                count,
                float(
                    np.prod(
                        self.model.input_upper_bound
                        - self.model.input_lower_bound
                    )
                ),
            )
        )
        nominal_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate.kind == "nominal"
        ]
        selected = nominal_indices[0] if nominal_indices else 0
        result = RolloutEvaluation(
            names=tuple(candidate.name for candidate in candidates),
            values=values,
            state_gradients=gradients,
            obstacle_gradients=np.zeros((count, 0, 7)),
            trajectories=trajectory_array,
            time_derivatives=np.zeros(count),
            control_directions=control_directions,
            constraint_rhs=rhs,
            scores=scores,
            selected_index=selected,
        )
        self.last_evaluation = result
        return result

    def _decision_certificates(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        candidates: tuple[PolicyCandidate, ...],
        evaluation: RolloutEvaluation,
    ) -> tuple[PolicyCertificate, ...]:
        """Build only data used by the PL-CBF and Library-PCBF-MI QPs.

        Rollout-domain, terminal, and nominal-prefix fields are visualization
        diagnostics.  Computing them with scalar NumPy re-rollouts used to
        dominate every controller decision even though neither algorithm
        reads them.  This path retains exactly the same value, derivative,
        affine halfspace, validity rule, and direct backup action.
        """

        state_array = np.asarray(state, dtype=float).reshape(12)
        goal_array = np.asarray(goal, dtype=float).reshape(3)
        drift = self.model.f(state_array)
        control_matrix = self.model.g(state_array)
        certificates: list[PolicyCertificate] = []
        for index, candidate in enumerate(candidates):
            backup_control = self._candidate_control(
                state_array,
                goal_array,
                candidate,
            )
            value = float(evaluation.values[index])
            gradient = evaluation.state_gradients[index]
            value_time_derivative = float(evaluation.time_derivatives[index])
            cbf_finite = bool(
                np.isfinite(value)
                and np.all(np.isfinite(gradient))
                and np.all(np.isfinite(evaluation.obstacle_gradients[index]))
                and np.isfinite(value_time_derivative)
                and np.all(np.isfinite(evaluation.control_directions[index]))
                and np.isfinite(evaluation.constraint_rhs[index])
                and np.all(np.isfinite(backup_control))
            )
            metadata = {
                "kind": candidate.kind,
                "state_gradient": gradient.copy(),
                "value_time_derivative": value_time_derivative,
                "Lg": evaluation.control_directions[index].copy(),
                "rhs": float(evaluation.constraint_rhs[index]),
            }
            if cbf_finite:
                certificate = PolicyCertificate.from_cbf(
                    candidate.name,
                    value=value,
                    gradient=gradient,
                    drift=drift,
                    control_matrix=control_matrix,
                    value_time_derivative=value_time_derivative,
                    alpha=self.config.cbf_alpha,
                    buffer=self.config.cbf_value_buffer,
                    backup_control=backup_control,
                    valid=True,
                    metadata=metadata,
                )
            else:
                certificate = PolicyCertificate(
                    policy_id=candidate.name,
                    value=-1e12,
                    backup_control=backup_control,
                    valid=False,
                    diagnostic="nonfinite_cbf",
                    metadata=metadata,
                )
            certificates.append(certificate)
        self.last_certificates = tuple(certificates)
        return self.last_certificates

    def policy_certificates(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
        *,
        include_rollout_metadata: bool = True,
    ) -> tuple[PolicyCertificate, ...]:
        """Build the shared per-step certificate oracle for all baselines.

        The halfspace, direct backup action, rollout-safety labels, terminal
        check, nominal-prefix recoverability, and terminal cost all come from
        the same nonlinear policy branch used by this controller.
        """

        state_array = np.asarray(state, dtype=float).reshape(12)
        goal_array = np.asarray(goal, dtype=float).reshape(3)
        active = self._active_obstacles(state_array, obstacles)
        candidates = self.candidates(goal_array)
        evaluation = (
            self._unconstrained_evaluation(
                state_array,
                goal_array,
                candidates,
                include_trajectories=include_rollout_metadata,
            )
            if active.shape[0] == 0
            else self.evaluate_policies(
                state_array,
                goal_array,
                active,
                candidates=candidates,
            )
        )
        if not include_rollout_metadata:
            return self._decision_certificates(
                state_array,
                goal_array,
                candidates,
                evaluation,
            )
        obstacle_history = self._obstacle_history(
            active,
            evaluation.trajectories.shape[1],
        )
        drift = self.model.f(state_array)
        control_matrix = self.model.g(state_array)
        certificates = []
        for index, candidate in enumerate(candidates):
            trajectory = evaluation.trajectories[index]
            (
                rollout_domain_valid,
                rollout_max_tilt,
                rollout_max_body_rate,
                domain_diagnostic,
            ) = self._rollout_domain(trajectory)
            if rollout_domain_valid and active.shape[0] == 0:
                clearances = np.full((trajectory.shape[0], 1), 1e12)
                minimum_clearance = 1e12
            elif rollout_domain_valid and np.all(np.isfinite(obstacle_history)):
                clearances = self._clearance_history(
                    trajectory,
                    obstacle_history,
                )
                minimum_clearance = float(np.min(clearances))
            else:
                clearances = np.full(
                    (trajectory.shape[0], max(1, active.shape[0])),
                    -np.inf,
                )
                minimum_clearance = -1e12
            rollout_safe = bool(
                rollout_domain_valid
                and np.all(np.isfinite(clearances))
                and minimum_clearance >= 0.0
            )
            terminal_safe = bool(
                rollout_domain_valid
                and self._terminal_safe(
                    trajectory,
                    obstacle_history,
                    goal_array,
                    candidate,
                )
            )
            terminal_state = trajectory[-1]
            terminal_cost = (
                float(
                    np.sum((terminal_state[:3] - goal_array) ** 2)
                    + 0.1 * np.sum(terminal_state[3:6] ** 2)
                )
                if rollout_domain_valid
                else 1e12
            )
            backup_control = self._candidate_control(
                state_array,
                goal_array,
                candidate,
            )
            value = float(evaluation.values[index])
            gradient = evaluation.state_gradients[index]
            value_time_derivative = float(
                evaluation.time_derivatives[index]
            )
            cbf_finite = bool(
                np.isfinite(value)
                and np.all(np.isfinite(gradient))
                and np.all(
                    np.isfinite(evaluation.obstacle_gradients[index])
                )
                and np.isfinite(value_time_derivative)
                and np.all(
                    np.isfinite(evaluation.control_directions[index])
                )
                and np.isfinite(evaluation.constraint_rhs[index])
                and np.all(np.isfinite(backup_control))
            )
            diagnostic_parts = []
            if domain_diagnostic:
                diagnostic_parts.append(domain_diagnostic)
            if not cbf_finite:
                diagnostic_parts.append("nonfinite_cbf")
            diagnostic = "+".join(diagnostic_parts)
            # Rollout-domain checks are benchmark diagnostics.  The PL-CBF
            # certificate itself matches the playground/warehouse construction:
            # a finite V and dV define the certificate, without a heuristic
            # pre-filter on the candidate trajectory.
            certificate_valid = cbf_finite
            metadata = {
                "kind": candidate.kind,
                "rollout_safe": rollout_safe,
                "terminal_safe": terminal_safe,
                "nominal_prefix_safe": (
                    self._nominal_prefix_safe(
                        state_array,
                        active,
                        goal_array,
                        candidate,
                    )
                    if rollout_domain_valid
                    else 0
                ),
                "terminal_cost": terminal_cost,
                "mi_cost": terminal_cost,
                "minimum_rollout_clearance": minimum_clearance,
                "rollout_domain_valid": rollout_domain_valid,
                "rollout_max_tilt_rad": rollout_max_tilt,
                "rollout_tilt_max_rad": self.config.rollout_tilt_max_rad,
                "rollout_max_body_rate_rad_s": rollout_max_body_rate,
                "rollout_body_rate_max_rad_s": (
                    self.model.config.body_rate_max
                ),
                "state_gradient": gradient.copy(),
                "value_time_derivative": value_time_derivative,
                "Lg": evaluation.control_directions[index].copy(),
                "rhs": float(evaluation.constraint_rhs[index]),
            }
            if cbf_finite:
                certificate = PolicyCertificate.from_cbf(
                    candidate.name,
                    value=value,
                    gradient=gradient,
                    drift=drift,
                    control_matrix=control_matrix,
                    value_time_derivative=value_time_derivative,
                    alpha=self.config.cbf_alpha,
                    buffer=self.config.cbf_value_buffer,
                    backup_control=backup_control,
                    valid=certificate_valid,
                    diagnostic=diagnostic,
                    metadata=metadata,
                )
            else:
                certificate = PolicyCertificate(
                    policy_id=candidate.name,
                    value=-1e12,
                    backup_control=backup_control,
                    valid=False,
                    diagnostic=diagnostic,
                    metadata=metadata,
                )
            certificates.append(certificate)
        self.last_certificates = tuple(certificates)
        return self.last_certificates

    def decision_certificates(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
    ) -> tuple[PolicyCertificate, ...]:
        """Fast algorithm certificate oracle without unused diagnostics."""

        return self.policy_certificates(
            state,
            goal,
            obstacles,
            include_rollout_metadata=False,
        )

    def warmup(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
    ) -> None:
        """Compile and synchronize the one static-shape rollout executable."""

        state_array = np.asarray(state, dtype=float).reshape(12)
        goal_array = np.asarray(goal, dtype=float).reshape(3)
        active = self._active_obstacles(state_array, obstacles)
        if active.shape[0] == 0:
            # Compilation depends on shape, never values.  A finite synthetic
            # sphere makes the smooth minimum defined even when the initial
            # benchmark state has no sensed obstacle; runtime calls of every
            # real sensed count still reuse the same padded signature.
            point = self.model.safety_point(state_array)
            active = np.zeros((1, 7), dtype=float)
            active[0, :3] = point + np.asarray(
                [max(0.25, 0.5 * self.config.sensing_radius), 0.0, 0.0]
            )
            active[0, 3] = 0.1
        self.evaluate_policies(state_array, goal_array, active)

    # Aliases make the oracle easy to pass directly into benchmark harnesses.
    get_policy_certificates = policy_certificates
    certificate_oracle = policy_certificates

    def _candidate_control(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        candidate: PolicyCandidate,
    ) -> np.ndarray:
        if candidate.kind == "radial":
            from .dynamics import velocity_input

            return velocity_input(
                state,
                np.asarray(candidate.direction) * candidate.target_speed,
                self.model.config,
                gain=candidate.gain,
            )
        if candidate.kind == "stop":
            return self.model.stop_input(state, candidate.gain)
        return self.model.nominal_input(state, goal)

    def solve_control_problem(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
        control_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        """Filter a nominal rotor command through the selected PLCBF constraint."""

        state_array = np.asarray(state, dtype=float).reshape(12)
        goal_array = np.asarray(goal, dtype=float).reshape(3)
        nominal = (
            self.model.nominal_input(state_array, goal_array)
            if control_ref is None
            else self.model.saturate_rotors(control_ref)
        )
        certificates = self.decision_certificates(
            state_array,
            goal_array,
            obstacles,
        )
        if self.last_evaluation is None:
            raise RuntimeError("policy evaluation did not produce diagnostics")
        selected_index = self.last_evaluation.selected_index
        selected = certificates[selected_index]
        if len(selected.halfspaces) != 1:
            raise RuntimeError("NL-Quad3D PL-CBF requires one affine constraint")
        lower = self.model.input_lower_bound
        upper = self.model.input_upper_bound
        solution = solve_box_halfspace_qp(
            nominal,
            lower,
            upper,
            selected.halfspaces[0],
        )
        filtered = (
            None
            if solution.control is None
            else self.model.saturate_rotors(solution.control)
        )
        margin = (
            float("-inf")
            if filtered is None
            else selected.halfspaces[0].residual(filtered)
        )
        used_fallback = bool(
            not solution.feasible
            or filtered is None
            or margin < -self.config.constraint_tolerance
        )
        if used_fallback:
            fallback = (
                nominal
                if selected.backup_control is None
                else selected.backup_control
            )
            control = self.model.saturate_rotors(fallback)
        else:
            assert filtered is not None
            control = filtered

        policy_evaluations = []
        for index, certificate in enumerate(certificates):
            if certificate.valid and len(certificate.halfspaces) == 1:
                halfspace = certificate.halfspaces[0]
                volume = (
                    float(self.last_evaluation.scores[index])
                    if self.config.max_operator == "input_space"
                    else _halfspace_box_measure(
                        halfspace.normal,
                        halfspace.offset,
                        lower,
                        upper,
                    )
                )
                maximizing_corner = np.where(
                    halfspace.normal >= 0.0,
                    upper,
                    lower,
                )
                candidate_feasible = bool(
                    halfspace.residual(maximizing_corner)
                    >= -self.config.constraint_tolerance
                )
                if index == selected_index:
                    candidate_control = control
                    candidate_objective = solution.objective
                    candidate_status = (
                        "fallback" if used_fallback else solution.status
                    )
                else:
                    candidate_control = None
                    candidate_objective = float("inf")
                    candidate_status = "not_solved"
            else:
                volume = 0.0
                candidate_control = None
                candidate_objective = float("inf")
                candidate_feasible = False
                candidate_status = "invalid_certificate"
            policy_evaluations.append(
                PolicyEvaluation(
                    policy_id=certificate.policy_id,
                    value=certificate.value,
                    safe_value=bool(
                        certificate.valid and certificate.value > 0.0
                    ),
                    feasible=candidate_feasible,
                    input_volume=volume,
                    intervention_cost=candidate_objective,
                    control=candidate_control,
                    status=candidate_status,
                )
            )
        diagnostics = DecisionDiagnostics(
            mode=(
                SelectionMode.INPUT_VOLUME
                if self.config.max_operator == "input_space"
                else SelectionMode.VALUE
            ),
            selected_policy_id=selected.policy_id,
            used_fallback=used_fallback,
            fallback_reason="selected_policy_qp_infeasible" if used_fallback else None,
            fallback_source="selected_policy_backup" if used_fallback else None,
            safe_policy_count=sum(item.safe_value for item in policy_evaluations),
            feasible_policy_count=sum(item.feasible for item in policy_evaluations),
            eligible_policy_count=sum(
                item.safe_value and item.feasible for item in policy_evaluations
            ),
            evaluations=tuple(policy_evaluations),
        )
        decision = PolicyDecision(selected, control, diagnostics)
        self.last_decision = decision
        if (
            decision.policy_id is not None
            and self.last_evaluation is not None
            and decision.policy_id in self.last_evaluation.names
        ):
            self.last_evaluation = replace(
                self.last_evaluation,
                selected_index=self.last_evaluation.names.index(decision.policy_id),
            )
        if decision.diagnostics.used_fallback:
            self.last_status = "fallback"
        elif np.allclose(decision.control, nominal, atol=1e-9):
            self.last_status = (
                "nominal_no_obstacles"
                if np.asarray(obstacles).size == 0
                else "nominal"
            )
        else:
            self.last_status = "filtered"
        self.last_control = self.model.saturate_rotors(decision.control)
        return self.last_control.copy()

    # Existing example code commonly uses ``solve`` for controller dispatch.
    solve = solve_control_problem


__all__ = [
    "NLQuad3DControllerConfig",
    "PLCBF_NLQuad3D",
    "PolicyCandidate",
    "RolloutEvaluation",
    "batched_policy_values_and_gradients_jax",
    "rollout_policy_jax",
    "smooth_min_jax",
    "trajectory_value_jax",
]
