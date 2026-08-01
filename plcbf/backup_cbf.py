"""Path-wise Backup-CBF and multi-backup minimum-intervention solvers.

Unlike a policy PCBF, a Backup-CBF constrains every sampled point of the
closed-loop backup flow and a terminal set.  The scenario callback supplied
here returns those path and terminal margins.  Central differences of the
composed rollout maps are the numerical equivalent of propagating the flow
sensitivity and multiplying it by each safety-function gradient.

This module intentionally contains no policy switching or case-study modes.
The single-policy solver receives one fixed backup.  The multi-policy solver
evaluates every supplied backup independently and selects the feasible QP with
minimum intervention, as in the warehouse ``Multi-Backup-CBF-MI`` baseline
(``MI`` means minimum intervention).
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Literal, Mapping, Sequence

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from plcbf.policy_library import CBFHalfspace


FloatArray = NDArray[np.float64]
RolloutMarginOracle = Callable[
    [FloatArray, float], tuple[FloatArray, float]
]
BackupCbfFormulation = Literal["single", "strict_multi"]


def _vector(value: ArrayLike, name: str, size: int | None = None) -> FloatArray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if size is not None and result.size != size:
        raise ValueError(f"{name} must contain {size} values")
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result.copy()


@dataclass(frozen=True)
class BackupCbfCandidate:
    """All path/terminal constraints and the associated QP result."""

    policy_id: str
    path_values: FloatArray
    terminal_value: float
    nominal_control: FloatArray
    halfspaces: tuple[CBFHalfspace, ...]
    control: FloatArray | None
    feasible: bool
    rollout_safe: bool
    terminal_safe: bool
    objective: float
    status: str
    solve_time_s: float

    def __post_init__(self) -> None:
        path_values = np.asarray(self.path_values, dtype=float).reshape(-1).copy()
        path_values.setflags(write=False)
        object.__setattr__(self, "path_values", path_values)
        nominal_control = np.asarray(
            self.nominal_control, dtype=float
        ).reshape(-1).copy()
        nominal_control.setflags(write=False)
        object.__setattr__(self, "nominal_control", nominal_control)
        if self.control is not None:
            control = np.asarray(self.control, dtype=float).reshape(-1).copy()
            control.setflags(write=False)
            object.__setattr__(self, "control", control)
        object.__setattr__(self, "policy_id", str(self.policy_id))
        object.__setattr__(self, "terminal_value", float(self.terminal_value))
        object.__setattr__(self, "objective", float(self.objective))
        object.__setattr__(self, "solve_time_s", float(self.solve_time_s))


@dataclass(frozen=True)
class BackupCbfRolloutDerivatives:
    """Precomputed central-difference data for one backup rollout.

    Case studies may batch the otherwise independent policy and perturbation
    rollouts, then pass the resulting arrays back through the exact common
    warehouse row builder and QP solver.  This object changes only how the
    numerical samples are evaluated; it does not change any Backup-CBF row,
    certification threshold, or solver rule.
    """

    path_values: FloatArray
    terminal_value: float
    path_gradients: FloatArray
    path_time_derivatives: FloatArray
    terminal_gradient: FloatArray


@dataclass(frozen=True)
class BackupCbfDecision:
    """Executable single- or multi-backup decision."""

    control: FloatArray
    policy_id: str
    feasible: bool
    safety_feasible: bool
    used_fallback: bool
    objective: float
    status: str
    candidates: tuple[BackupCbfCandidate, ...]
    solve_time_s: float

    def __post_init__(self) -> None:
        control = np.asarray(self.control, dtype=float).reshape(-1).copy()
        if control.size == 0 or not np.all(np.isfinite(control)):
            raise ValueError("decision control must be finite")
        control.setflags(write=False)
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "policy_id", str(self.policy_id))
        object.__setattr__(self, "safety_feasible", bool(self.safety_feasible))
        object.__setattr__(self, "objective", float(self.objective))
        object.__setattr__(self, "solve_time_s", float(self.solve_time_s))


def _rollout_derivatives(
    state: FloatArray,
    oracle: RolloutMarginOracle,
    gradient_steps: FloatArray,
    time_derivative_step: float,
) -> tuple[
    FloatArray,
    float,
    FloatArray,
    FloatArray,
    FloatArray,
]:
    path, terminal = oracle(state.copy(), 0.0)
    path = np.asarray(path, dtype=float).reshape(-1)
    terminal = float(terminal)
    if path.size == 0:
        raise ValueError("a Backup-CBF rollout must contain path margins")
    if not np.all(np.isfinite(path)) or not np.isfinite(terminal):
        raise FloatingPointError("backup rollout returned non-finite margins")

    path_gradient = np.zeros((path.size, state.size), dtype=float)
    terminal_gradient = np.zeros(state.size, dtype=float)
    for index, step in enumerate(gradient_steps):
        plus = state.copy()
        minus = state.copy()
        plus[index] += step
        minus[index] -= step
        plus_path, plus_terminal = oracle(plus, 0.0)
        minus_path, minus_terminal = oracle(minus, 0.0)
        plus_path = np.asarray(plus_path, dtype=float).reshape(-1)
        minus_path = np.asarray(minus_path, dtype=float).reshape(-1)
        if plus_path.shape != path.shape or minus_path.shape != path.shape:
            raise ValueError("rollout margin count changed under differentiation")
        path_gradient[:, index] = (plus_path - minus_path) / (2.0 * step)
        terminal_gradient[index] = (
            float(plus_terminal) - float(minus_terminal)
        ) / (2.0 * step)

    advanced_path, _ = oracle(
        state.copy(), time_derivative_step
    )
    advanced_path = np.asarray(advanced_path, dtype=float).reshape(-1)
    if advanced_path.shape != path.shape:
        raise ValueError("rollout margin count changed under time advancement")
    path_time_derivative = (
        advanced_path - path
    ) / time_derivative_step
    return (
        path,
        terminal,
        path_gradient,
        path_time_derivative,
        terminal_gradient,
    )


def _solve_qp(
    nominal: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    constraints: Sequence[CBFHalfspace],
    control_scales: FloatArray,
    control_weights: FloatArray,
) -> tuple[FloatArray | None, float, str, float]:
    nominal = np.clip(nominal, lower, upper)
    control = cp.Variable(nominal.size)
    problem_constraints = [control >= lower, control <= upper]
    problem_constraints.extend(
        constraint.normal @ control >= constraint.offset
        for constraint in constraints
    )
    normalized_error = cp.multiply(
        control_weights,
        cp.multiply(1.0 / control_scales, control - nominal),
    )
    objective = cp.Minimize(cp.sum_squares(normalized_error))
    problem = cp.Problem(objective, problem_constraints)
    started_at = time.perf_counter()
    try:
        problem.solve(
            solver=cp.OSQP,
            verbose=False,
            warm_start=False,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=100_000,
            polishing=True,
        )
        status = str(problem.status)
    except Exception as error:
        return (
            None,
            float("inf"),
            f"solver_error:{type(error).__name__}",
            time.perf_counter() - started_at,
        )
    solve_time = time.perf_counter() - started_at
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return None, float("inf"), status, solve_time
    if control.value is None:
        return None, float("inf"), "missing_solution", solve_time
    result = np.asarray(control.value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(result)):
        return None, float("inf"), "nonfinite_solution", solve_time
    result = np.clip(result, lower, upper)
    residuals = [
        constraint.residual(result) for constraint in constraints
    ]
    if residuals and min(residuals) < -1e-4:
        return None, float("inf"), "postsolve_constraint_violation", solve_time
    normalized_delta = control_weights * (
        (result - nominal) / control_scales
    )
    return (
        result,
        float(normalized_delta @ normalized_delta),
        status,
        solve_time,
    )


def evaluate_backup_cbf_candidate(
    *,
    policy_id: str,
    state: ArrayLike,
    nominal_control: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    drift: ArrayLike,
    control_matrix: ArrayLike,
    backup_closed_loop_drift: ArrayLike,
    rollout_margins: RolloutMarginOracle,
    gradient_steps: ArrayLike,
    time_derivative_step: float,
    alpha: float,
    terminal_alpha: float,
    path_flow_derivatives: ArrayLike | None = None,
    formulation: BackupCbfFormulation = "strict_multi",
    path_constraint_start_index: int = 0,
    control_scales: ArrayLike | None = None,
    control_weights: ArrayLike | None = None,
    common_halfspaces: Sequence[CBFHalfspace] = (),
    safety_threshold: float = 0.0,
    terminal_threshold: float = 0.0,
    precomputed_derivatives: BackupCbfRolloutDerivatives | None = None,
) -> BackupCbfCandidate:
    """Build and solve one full path-wise Backup-CBF candidate QP.

    ``formulation="single"`` reproduces the legacy warehouse Backup-CBF: it
    solves the QP even for an uncertified open-loop rollout and silently skips
    rows whose control coefficient has norm at most ``1e-6``.

    ``formulation="strict_multi"`` reproduces the warehouse
    Multi-Backup-CBF-MI candidate: it rejects an uncertified rollout before
    solving and rejects a zero-control row when its right-hand side is greater
    than ``1e-8``.  Safe redundant zero-control rows are skipped.

    ``path_constraint_start_index`` controls only which rollout samples
    generate QP rows.  All returned path values still participate in safety
    certification and the fixed-controller failure margin.  Thus the legacy
    single formulation can use ``1`` while strict multi uses ``0``.
    """

    state_array = _vector(state, "state")
    nominal = _vector(nominal_control, "nominal_control")
    low = np.broadcast_to(np.asarray(lower, dtype=float), nominal.shape).copy()
    high = np.broadcast_to(np.asarray(upper, dtype=float), nominal.shape).copy()
    if np.any(low > high):
        raise ValueError("lower bounds must not exceed upper bounds")
    if formulation not in ("single", "strict_multi"):
        raise ValueError(
            "formulation must be 'single' or 'strict_multi'"
        )
    if (
        not isinstance(path_constraint_start_index, (int, np.integer))
        or path_constraint_start_index < 0
    ):
        raise ValueError(
            "path_constraint_start_index must be a nonnegative integer"
        )
    if control_scales is None:
        scales = np.maximum(np.abs(low), np.abs(high))
    else:
        scales = np.broadcast_to(
            np.asarray(control_scales, dtype=float), nominal.shape
        ).copy()
    if control_weights is None:
        weights = np.ones_like(nominal)
    else:
        weights = np.broadcast_to(
            np.asarray(control_weights, dtype=float), nominal.shape
        ).copy()
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("control_scales must be finite and positive")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("control_weights must be finite and positive")
    drift_array = _vector(drift, "drift", state_array.size)
    matrix = np.asarray(control_matrix, dtype=float)
    if matrix.shape != (state_array.size, nominal.size):
        raise ValueError("control_matrix has incompatible dimensions")
    backup_drift = _vector(
        backup_closed_loop_drift,
        "backup_closed_loop_drift",
        state_array.size,
    )
    steps = _vector(gradient_steps, "gradient_steps", state_array.size)
    if np.any(steps <= 0.0):
        raise ValueError("gradient_steps must be positive")
    if time_derivative_step <= 0.0:
        raise ValueError("time_derivative_step must be positive")
    if alpha < 0.0 or terminal_alpha < 0.0:
        raise ValueError("CBF gains must be nonnegative")

    if precomputed_derivatives is None:
        (
            path,
            terminal,
            path_gradients,
            path_time_derivatives,
            terminal_gradient,
        ) = _rollout_derivatives(
            state_array,
            rollout_margins,
            steps,
            float(time_derivative_step),
        )
    else:
        path = np.asarray(
            precomputed_derivatives.path_values,
            dtype=float,
        ).reshape(-1)
        terminal = float(precomputed_derivatives.terminal_value)
        path_gradients = np.asarray(
            precomputed_derivatives.path_gradients,
            dtype=float,
        )
        path_time_derivatives = np.asarray(
            precomputed_derivatives.path_time_derivatives,
            dtype=float,
        ).reshape(-1)
        terminal_gradient = np.asarray(
            precomputed_derivatives.terminal_gradient,
            dtype=float,
        ).reshape(-1)
        if path.size == 0:
            raise ValueError("a Backup-CBF rollout must contain path margins")
        if path_gradients.shape != (path.size, state_array.size):
            raise ValueError(
                "precomputed path gradients have incompatible dimensions"
            )
        if path_time_derivatives.shape != path.shape:
            raise ValueError(
                "precomputed path time derivatives have incompatible dimensions"
            )
        if terminal_gradient.shape != state_array.shape:
            raise ValueError(
                "precomputed terminal gradient has incompatible dimensions"
            )
        if not (
            np.all(np.isfinite(path))
            and np.isfinite(terminal)
            and np.all(np.isfinite(path_gradients))
            and np.all(np.isfinite(path_time_derivatives))
            and np.all(np.isfinite(terminal_gradient))
        ):
            raise FloatingPointError(
                "precomputed backup rollout derivatives must be finite"
            )
    if path_flow_derivatives is None:
        # Exact for an autonomous continuous closed-loop backup flow by the
        # semigroup identity Q(t, x0) f_backup(x0) = f_backup(phi(t, x0)).
        # Time-indexed or maneuver-then-tail policies should instead provide
        # the warehouse row values grad_h(phi_i) @ f_policy_i explicitly.
        flow_derivatives = path_gradients @ backup_drift
    else:
        flow_derivatives = _vector(
            path_flow_derivatives,
            "path_flow_derivatives",
            path.size,
        )
    if path_constraint_start_index > path.size:
        raise ValueError(
            "path_constraint_start_index exceeds rollout path length"
        )
    rollout_safe = bool(np.min(path) >= safety_threshold)
    terminal_safe = bool(terminal >= terminal_threshold)
    constraints = list(common_halfspaces)
    clipped_nominal = np.clip(nominal, low, high)

    if formulation == "strict_multi" and not (
        rollout_safe and terminal_safe
    ):
        return BackupCbfCandidate(
            policy_id=policy_id,
            path_values=path,
            terminal_value=terminal,
            nominal_control=clipped_nominal,
            halfspaces=tuple(constraints),
            control=None,
            feasible=False,
            rollout_safe=rollout_safe,
            terminal_safe=terminal_safe,
            objective=float("inf"),
            status=(
                "uncertified_rollout"
                if not rollout_safe
                else "uncertified_terminal"
            ),
            solve_time_s=0.0,
        )

    for index, (value, gradient, time_derivative, flow_derivative) in enumerate(
        zip(
            path[path_constraint_start_index:],
            path_gradients[path_constraint_start_index:],
            path_time_derivatives[path_constraint_start_index:],
            flow_derivatives[path_constraint_start_index:],
            strict=True,
        ),
        start=path_constraint_start_index,
    ):
        normal = gradient @ matrix
        # Match the warehouse Backup-CBF flow-sensitivity row:
        #
        # Dh(phi) Q g u >= -Dh(phi) Q f + Dh(phi) f_backup(phi)
        #                       - partial_t h - alpha(h).
        #
        # Central differences of the composed rollout margin give
        # ``Dh(phi) Q``.  For an autonomous closed-loop backup flow, the
        # semigroup identity ``Q f_backup(x0) = f_backup(phi)`` supplies the
        # policy-flow correction from the initial closed-loop drift.
        offset = (
            -float(gradient @ drift_array)
            + float(flow_derivative)
            - float(time_derivative)
            - float(alpha) * (float(value) - float(safety_threshold))
        )
        if np.linalg.norm(normal) <= 1e-6:
            if formulation == "strict_multi" and offset > 1e-8:
                return BackupCbfCandidate(
                    policy_id=policy_id,
                    path_values=path,
                    terminal_value=terminal,
                    nominal_control=clipped_nominal,
                    halfspaces=tuple(constraints),
                    control=None,
                    feasible=False,
                    rollout_safe=rollout_safe,
                    terminal_safe=terminal_safe,
                    objective=float("inf"),
                    status="infeasible_constant_path_constraint",
                    solve_time_s=0.0,
                )
            continue
        constraints.append(
            CBFHalfspace(normal, offset, f"{policy_id}:path[{index}]")
        )
    terminal_normal = terminal_gradient @ matrix
    # Warehouse Backup-CBF intentionally has neither the policy-flow
    # correction nor an explicit obstacle-time derivative in its terminal
    # invariant-set row (backup_cbf_qp.py:655-664).
    terminal_offset = -(
        float(terminal_gradient @ drift_array)
        + float(terminal_alpha)
        * (terminal - float(terminal_threshold))
    )
    if np.linalg.norm(terminal_normal) <= 1e-6:
        if formulation == "strict_multi" and terminal_offset > 1e-8:
            return BackupCbfCandidate(
                policy_id=policy_id,
                path_values=path,
                terminal_value=terminal,
                nominal_control=clipped_nominal,
                halfspaces=tuple(constraints),
                control=None,
                feasible=False,
                rollout_safe=rollout_safe,
                terminal_safe=terminal_safe,
                objective=float("inf"),
                status="infeasible_constant_terminal_constraint",
                solve_time_s=0.0,
            )
    else:
        constraints.append(
            CBFHalfspace(
                terminal_normal,
                terminal_offset,
                f"{policy_id}:terminal",
            )
        )
    control, objective, status, solve_time = _solve_qp(
        nominal,
        low,
        high,
        constraints,
        scales,
        weights,
    )
    feasible = bool(
        control is not None and rollout_safe and terminal_safe
    )
    if control is not None and not (rollout_safe and terminal_safe):
        status = "unsafe_rollout_or_terminal"
    return BackupCbfCandidate(
        policy_id=policy_id,
        path_values=path,
        terminal_value=terminal,
        nominal_control=clipped_nominal,
        halfspaces=tuple(constraints),
        control=control,
        feasible=feasible,
        rollout_safe=rollout_safe,
        terminal_safe=terminal_safe,
        objective=objective,
        status=status,
        solve_time_s=solve_time,
    )


def solve_fixed_backup_cbf(
    candidate: BackupCbfCandidate,
    *,
    direct_backup_control: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    safe_failure_nominal_margin: float = 0.01,
) -> BackupCbfDecision:
    """Execute one predeclared fixed-backup candidate."""

    started_at = time.perf_counter()
    direct = _vector(direct_backup_control, "direct_backup_control")
    low = np.broadcast_to(np.asarray(lower, dtype=float), direct.shape)
    high = np.broadcast_to(np.asarray(upper, dtype=float), direct.shape)
    if safe_failure_nominal_margin < 0.0:
        raise ValueError("safe_failure_nominal_margin must be nonnegative")
    certified = bool(candidate.rollout_safe and candidate.terminal_safe)
    if candidate.control is not None:
        return BackupCbfDecision(
            control=candidate.control,
            policy_id=candidate.policy_id,
            feasible=True,
            safety_feasible=certified,
            used_fallback=False,
            objective=candidate.objective,
            status=candidate.status,
            candidates=(candidate,),
            solve_time_s=(
                candidate.solve_time_s + time.perf_counter() - started_at
            ),
        )
    minimum_margin = min(
        float(np.min(candidate.path_values)),
        float(candidate.terminal_value),
    )
    if minimum_margin > float(safe_failure_nominal_margin):
        return BackupCbfDecision(
            control=np.clip(candidate.nominal_control, low, high),
            policy_id=candidate.policy_id,
            feasible=False,
            safety_feasible=True,
            used_fallback=True,
            objective=float("inf"),
            status=f"nominal_after_qp_failure:{candidate.status}",
            candidates=(candidate,),
            solve_time_s=(
                candidate.solve_time_s + time.perf_counter() - started_at
            ),
        )
    return BackupCbfDecision(
        control=np.clip(direct, low, high),
        policy_id=candidate.policy_id,
        feasible=False,
        safety_feasible=False,
        used_fallback=True,
        objective=float("inf"),
        status=f"fallback_after:{candidate.status}",
        candidates=(candidate,),
        solve_time_s=candidate.solve_time_s + time.perf_counter() - started_at,
    )


def solve_multi_backup_cbf_min_intervention(
    candidates: Sequence[BackupCbfCandidate],
    *,
    direct_backup_controls: Mapping[str, ArrayLike],
    lower: ArrayLike,
    upper: ArrayLike,
    emergency_policy_id: str = "stop",
    tie_tolerance: float = 1e-8,
) -> BackupCbfDecision:
    """Select the feasible full Backup-CBF QP with minimum intervention.

    If no candidate is certified and QP-feasible, execute the same fixed stop
    emergency used by the warehouse MB-CBF-MI implementation.  This fallback
    is not a second policy-selection heuristic.
    """

    started_at = time.perf_counter()
    candidate_tuple = tuple(candidates)
    if not candidate_tuple:
        raise ValueError("multi-backup CBF requires at least one candidate")
    if tie_tolerance < 0.0 or not np.isfinite(tie_tolerance):
        raise ValueError("tie_tolerance must be finite and nonnegative")
    feasible = [
        candidate
        for candidate in candidate_tuple
        if candidate.feasible and candidate.control is not None
    ]
    if feasible:
        # Warehouse resolves objectives within 1e-8 by original policy-library
        # order; it does not introduce a second safety-margin selector.
        selected = feasible[0]
        for candidate in feasible[1:]:
            if candidate.objective < selected.objective - tie_tolerance:
                selected = candidate
        assert selected.control is not None
        return BackupCbfDecision(
            control=selected.control,
            policy_id=selected.policy_id,
            feasible=True,
            safety_feasible=True,
            used_fallback=False,
            objective=selected.objective,
            status=selected.status,
            candidates=candidate_tuple,
            solve_time_s=(
                sum(item.solve_time_s for item in candidate_tuple)
                + time.perf_counter()
                - started_at
            ),
        )

    emergency_id = str(emergency_policy_id)
    if not emergency_id:
        raise ValueError("emergency_policy_id must not be empty")
    if emergency_id not in direct_backup_controls:
        raise KeyError(
            f"missing direct backup control for {emergency_id!r}"
        )
    direct = _vector(
        direct_backup_controls[emergency_id],
        "direct_backup_control",
    )
    low = np.broadcast_to(np.asarray(lower, dtype=float), direct.shape)
    high = np.broadcast_to(np.asarray(upper, dtype=float), direct.shape)
    return BackupCbfDecision(
        control=np.clip(direct, low, high),
        policy_id=emergency_id,
        feasible=False,
        safety_feasible=False,
        used_fallback=True,
        objective=float("inf"),
        status="no_feasible_backup_cbf_candidate",
        candidates=candidate_tuple,
        solve_time_s=(
            sum(item.solve_time_s for item in candidate_tuple)
            + time.perf_counter()
            - started_at
        ),
    )


__all__ = [
    "BackupCbfFormulation",
    "BackupCbfCandidate",
    "BackupCbfRolloutDerivatives",
    "BackupCbfDecision",
    "RolloutMarginOracle",
    "evaluate_backup_cbf_candidate",
    "solve_fixed_backup_cbf",
    "solve_multi_backup_cbf_min_intervention",
]
