"""Reusable Big-M trajectory MPC with a one-hot backup-policy disjunction.

This module contains the optimization core used by the warehouse MI-MPC
baseline, without any scenario-specific dynamics, geometry, or policy code.
The caller is responsible for:

* computing affine prediction dynamics,
* rolling out each candidate backup policy, and
* assigning each rollout a safety value (and optional eligibility flag).

The optimizer then chooses one branch jointly with a continuous state and
control trajectory.  To match the warehouse MI-MPC exactly, if no branch
reaches ``safety_threshold`` it admits only the maximum-safety branch and sets
the effective selector threshold to ``max_safety - 1e-6``.  Result metadata
keeps that emergency relaxation distinct from satisfying the requested safety
threshold.  If the MILP has no feasible incumbent, the warehouse fallback
blends the maximum-safety branch's first action with the nominal action.
Caller-supplied emergency control is reserved for the reusable extension where
all branches are hard-excluded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class BigMTrajectoryMPCProblem:
    """Numerical data for one receding-horizon Big-M MPC solve.

    ``branch_states`` has shape ``(P, N + 1, n)`` and ``branch_controls`` has
    shape ``(P, N, m)``.  These are full candidate-policy rollouts, although
    only the coordinates named by ``position_indices`` are placed in Big-M
    state tubes.

    ``A``, ``B``, and ``c`` describe

    ``x[k + 1] = A[k] @ x[k] + B[k] @ u[k] + c[k]``.

    Constant ``A``/``B`` matrices and a constant ``c`` vector are broadcast
    over the horizon.  Bounds may likewise be constant or time-varying.
    """

    x0: ArrayLike
    A: ArrayLike
    B: ArrayLike
    branch_states: ArrayLike
    branch_controls: ArrayLike
    branch_safety: ArrayLike
    state_lower: ArrayLike
    state_upper: ArrayLike
    control_lower: ArrayLike
    control_upper: ArrayLike
    position_indices: Sequence[int]
    c: ArrayLike | None = None
    tracking_target: ArrayLike | None = None
    terminal_target: ArrayLike | None = None
    velocity_indices: Sequence[int] = ()
    nominal_control: ArrayLike | None = None
    branch_eligible: ArrayLike | None = None
    fallback_control: ArrayLike | None = None


@dataclass(frozen=True)
class BigMTrajectoryMPCConfig:
    """Configuration matching the warehouse baseline's L1 MILP objective."""

    safety_threshold: float = 0.0
    position_tube: ArrayLike = 3.0
    early_control_tube: ArrayLike = 6.0
    early_control_steps: int = 2
    big_m_position: ArrayLike | None = None
    big_m_control: ArrayLike | None = None
    big_m_safety: ArrayLike = 50.0
    tracking_weight: float = 8.0
    terminal_weight: float = 16.0
    velocity_weight: float = 0.15
    control_weight: float = 0.02
    nominal_weight: float = 0.5
    safety_tiebreak_weight: float = 0.01
    fallback_branch_weight: float = 0.75
    time_limit_s: float = 1.0
    mip_rel_gap: float = 0.05
    feasibility_tolerance: float = 1e-7
    integrality_tolerance: float = 1e-6


@dataclass(frozen=True)
class BigMVariableLayout:
    """Indices of the semantic variable blocks in the flat MILP vector."""

    state: IntArray
    control: IntArray
    selector: IntArray
    auxiliary: Mapping[str, IntArray] = field(default_factory=dict)

    @property
    def variable_count(self) -> int:
        blocks = [self.state.reshape(-1), self.control.reshape(-1), self.selector]
        blocks.extend(block.reshape(-1) for block in self.auxiliary.values())
        return 0 if not blocks else int(max(int(np.max(block)) for block in blocks if block.size) + 1)


@dataclass(frozen=True)
class BigMMILPModel:
    """A built SciPy MILP and metadata useful for auditing its structure."""

    objective: FloatArray
    integrality: IntArray
    bounds: Bounds
    constraints: LinearConstraint
    layout: BigMVariableLayout
    row_groups: Mapping[str, slice]
    eligible_branches: BoolArray
    admissible_branches: BoolArray
    effective_safety_threshold: float
    safety_threshold_relaxed: bool
    promoted_branch: int | None
    position_big_m: FloatArray
    control_big_m: FloatArray
    safety_big_m: FloatArray
    horizon: int
    state_dimension: int
    control_dimension: int
    branch_count: int


@dataclass(frozen=True)
class BigMTrajectoryMPCResult:
    """Outcome of a trajectory MILP.

    ``feasible`` is true only when the optimizer returned a validated feasible
    trajectory.  ``safety_feasible`` additionally requires the selected
    branch to meet the originally requested threshold; it is false for the
    warehouse max-safety emergency admission.  If ``used_fallback`` is true,
    ``control`` comes from ``fallback_source`` (normally the warehouse
    max-safety-branch/nominal blend); it does not make the solve feasible and
    carries no safety claim.
    """

    status: str
    feasible: bool
    safety_feasible: bool
    used_fallback: bool
    control: FloatArray | None
    selected_branch: int | None
    selected_branch_safety: float | None
    state_trajectory: FloatArray | None
    control_trajectory: FloatArray | None
    selector: FloatArray | None
    objective: float | None
    solve_time_s: float
    solver_status: int | None
    solver_message: str
    safety_threshold: float
    effective_safety_threshold: float
    safety_threshold_relaxed: bool
    selected_branch_meets_safety_threshold: bool | None
    eligible_branches: tuple[int, ...]
    admissible_branches: tuple[int, ...]
    promoted_branch: int | None
    fallback_source: str | None
    fallback_branch: int | None
    mip_gap: float | None = None
    mip_node_count: int | None = None


@dataclass(frozen=True)
class _NormalizedProblem:
    x0: FloatArray
    A: FloatArray
    B: FloatArray
    c: FloatArray
    branch_states: FloatArray
    branch_controls: FloatArray
    branch_safety: FloatArray
    state_lower: FloatArray
    state_upper: FloatArray
    control_lower: FloatArray
    control_upper: FloatArray
    position_indices: tuple[int, ...]
    velocity_indices: tuple[int, ...]
    tracking_target: FloatArray | None
    terminal_target: FloatArray | None
    nominal_control: FloatArray | None
    branch_eligible: BoolArray
    fallback_control: FloatArray | None
    horizon: int
    state_dimension: int
    control_dimension: int
    branch_count: int


@dataclass(frozen=True)
class _BranchAdmission:
    """Requested-safe and warehouse-emergency branch masks."""

    eligible: BoolArray
    admissible: BoolArray
    effective_threshold: float
    threshold_relaxed: bool
    promoted_branch: int | None


_WAREHOUSE_THRESHOLD_EPSILON = 1e-6


def _finite_array(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=float, copy=True)


def _bounds_array(
    value: ArrayLike,
    shape: tuple[int, ...],
    name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=float)
    try:
        result = np.broadcast_to(array, shape)
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to shape {shape}") from exc
    if np.any(np.isnan(result)):
        raise ValueError(f"{name} must not contain NaN")
    return np.array(result, dtype=float, copy=True)


def _time_varying_matrix(
    value: ArrayLike,
    horizon: int,
    rows: int,
    columns: int,
    name: str,
) -> FloatArray:
    array = _finite_array(value, name)
    if array.shape == (rows, columns):
        return np.broadcast_to(array, (horizon, rows, columns)).copy()
    if array.shape != (horizon, rows, columns):
        raise ValueError(
            f"{name} must have shape {(rows, columns)} or "
            f"{(horizon, rows, columns)}"
        )
    return array


def _indices(
    value: Sequence[int],
    dimension: int,
    name: str,
    *,
    allow_empty: bool,
) -> tuple[int, ...]:
    result = tuple(int(index) for index in value)
    if not allow_empty and not result:
        raise ValueError(f"{name} must contain at least one index")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate indices")
    if any(index < 0 or index >= dimension for index in result):
        raise ValueError(f"{name} contains an out-of-range state index")
    return result


def _target_over_horizon(
    value: ArrayLike | None,
    horizon: int,
    dimension: int,
    name: str,
) -> FloatArray | None:
    if value is None:
        return None
    target = _finite_array(value, name)
    if target.shape == (dimension,):
        return np.broadcast_to(target, (horizon, dimension)).copy()
    if target.shape == (horizon + 1, dimension):
        return target[1:].copy()
    if target.shape != (horizon, dimension):
        raise ValueError(
            f"{name} must have shape {(dimension,)}, {(horizon, dimension)}, "
            f"or {(horizon + 1, dimension)}"
        )
    return target


def _normalize_problem(problem: BigMTrajectoryMPCProblem) -> _NormalizedProblem:
    branch_states = _finite_array(problem.branch_states, "branch_states")
    branch_controls = _finite_array(problem.branch_controls, "branch_controls")
    if branch_states.ndim != 3:
        raise ValueError("branch_states must have shape (P, N + 1, n)")
    if branch_controls.ndim != 3:
        raise ValueError("branch_controls must have shape (P, N, m)")

    branch_count, state_steps, state_dimension = branch_states.shape
    controls_count, horizon, control_dimension = branch_controls.shape
    if branch_count <= 0:
        raise ValueError("at least one branch rollout is required")
    if horizon <= 0:
        raise ValueError("the MPC horizon must be positive")
    if controls_count != branch_count or state_steps != horizon + 1:
        raise ValueError(
            "branch_states and branch_controls must share P and N dimensions"
        )
    if state_dimension <= 0 or control_dimension <= 0:
        raise ValueError("state and control dimensions must be positive")

    x0 = _finite_array(problem.x0, "x0").reshape(-1)
    if x0.shape != (state_dimension,):
        raise ValueError(f"x0 must have shape {(state_dimension,)}")

    A = _time_varying_matrix(
        problem.A,
        horizon,
        state_dimension,
        state_dimension,
        "A",
    )
    B = _time_varying_matrix(
        problem.B,
        horizon,
        state_dimension,
        control_dimension,
        "B",
    )
    if problem.c is None:
        affine = np.zeros((horizon, state_dimension), dtype=float)
    else:
        c_value = _finite_array(problem.c, "c")
        if c_value.shape == (state_dimension,):
            affine = np.broadcast_to(
                c_value, (horizon, state_dimension)
            ).copy()
        elif c_value.shape == (horizon, state_dimension):
            affine = c_value
        else:
            raise ValueError(
                f"c must have shape {(state_dimension,)} or "
                f"{(horizon, state_dimension)}"
            )

    state_lower = _bounds_array(
        problem.state_lower,
        (horizon + 1, state_dimension),
        "state_lower",
    )
    state_upper = _bounds_array(
        problem.state_upper,
        (horizon + 1, state_dimension),
        "state_upper",
    )
    control_lower = _bounds_array(
        problem.control_lower,
        (horizon, control_dimension),
        "control_lower",
    )
    control_upper = _bounds_array(
        problem.control_upper,
        (horizon, control_dimension),
        "control_upper",
    )
    if np.any(state_lower > state_upper):
        raise ValueError("state_lower must not exceed state_upper")
    if np.any(control_lower > control_upper):
        raise ValueError("control_lower must not exceed control_upper")
    if not np.all(np.isfinite(control_lower)) or not np.all(
        np.isfinite(control_upper)
    ):
        raise ValueError("control bounds must be finite")

    position_indices = _indices(
        problem.position_indices,
        state_dimension,
        "position_indices",
        allow_empty=False,
    )
    velocity_indices = _indices(
        problem.velocity_indices,
        state_dimension,
        "velocity_indices",
        allow_empty=True,
    )
    position_lower = state_lower[1:, position_indices]
    position_upper = state_upper[1:, position_indices]
    if not np.all(np.isfinite(position_lower)) or not np.all(
        np.isfinite(position_upper)
    ):
        raise ValueError(
            "state bounds must be finite for every position tube coordinate"
        )

    branch_safety = _finite_array(
        problem.branch_safety, "branch_safety"
    ).reshape(-1)
    if branch_safety.shape != (branch_count,):
        raise ValueError(f"branch_safety must have shape {(branch_count,)}")

    if problem.branch_eligible is None:
        branch_eligible = np.ones(branch_count, dtype=bool)
    else:
        eligible_array = np.asarray(problem.branch_eligible)
        if eligible_array.shape != (branch_count,):
            raise ValueError(
                f"branch_eligible must have shape {(branch_count,)}"
            )
        branch_eligible = np.array(eligible_array, dtype=bool, copy=True)

    position_dimension = len(position_indices)
    tracking_target = _target_over_horizon(
        problem.tracking_target,
        horizon,
        position_dimension,
        "tracking_target",
    )
    if problem.terminal_target is None:
        terminal_target = None
    else:
        terminal_target = _finite_array(
            problem.terminal_target, "terminal_target"
        ).reshape(-1)
        if terminal_target.shape != (position_dimension,):
            raise ValueError(
                f"terminal_target must have shape {(position_dimension,)}"
            )

    nominal_control = None
    if problem.nominal_control is not None:
        nominal_control = _finite_array(
            problem.nominal_control, "nominal_control"
        ).reshape(-1)
        if nominal_control.shape != (control_dimension,):
            raise ValueError(
                f"nominal_control must have shape {(control_dimension,)}"
            )

    fallback_control = None
    if problem.fallback_control is not None:
        fallback_control = _finite_array(
            problem.fallback_control, "fallback_control"
        ).reshape(-1)
        if fallback_control.shape != (control_dimension,):
            raise ValueError(
                f"fallback_control must have shape {(control_dimension,)}"
            )

    return _NormalizedProblem(
        x0=x0,
        A=A,
        B=B,
        c=affine,
        branch_states=branch_states,
        branch_controls=branch_controls,
        branch_safety=branch_safety,
        state_lower=state_lower,
        state_upper=state_upper,
        control_lower=control_lower,
        control_upper=control_upper,
        position_indices=position_indices,
        velocity_indices=velocity_indices,
        tracking_target=tracking_target,
        terminal_target=terminal_target,
        nominal_control=nominal_control,
        branch_eligible=branch_eligible,
        fallback_control=fallback_control,
        horizon=horizon,
        state_dimension=state_dimension,
        control_dimension=control_dimension,
        branch_count=branch_count,
    )


def _validate_config(config: BigMTrajectoryMPCConfig) -> None:
    scalar_values = {
        "safety_threshold": config.safety_threshold,
        "tracking_weight": config.tracking_weight,
        "terminal_weight": config.terminal_weight,
        "velocity_weight": config.velocity_weight,
        "control_weight": config.control_weight,
        "nominal_weight": config.nominal_weight,
        "safety_tiebreak_weight": config.safety_tiebreak_weight,
        "fallback_branch_weight": config.fallback_branch_weight,
        "time_limit_s": config.time_limit_s,
        "mip_rel_gap": config.mip_rel_gap,
        "feasibility_tolerance": config.feasibility_tolerance,
        "integrality_tolerance": config.integrality_tolerance,
    }
    for name, value in scalar_values.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    for name in (
        "tracking_weight",
        "terminal_weight",
        "velocity_weight",
        "control_weight",
        "nominal_weight",
        "safety_tiebreak_weight",
    ):
        if scalar_values[name] < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if config.early_control_steps < 0:
        raise ValueError("early_control_steps must be nonnegative")
    if config.time_limit_s <= 0.0:
        raise ValueError("time_limit_s must be positive")
    if config.mip_rel_gap < 0.0:
        raise ValueError("mip_rel_gap must be nonnegative")
    if config.feasibility_tolerance <= 0.0:
        raise ValueError("feasibility_tolerance must be positive")
    if config.integrality_tolerance <= 0.0:
        raise ValueError("integrality_tolerance must be positive")
    if not 0.0 <= config.fallback_branch_weight <= 1.0:
        raise ValueError("fallback_branch_weight must lie in [0, 1]")
    for name, value in (
        ("position_tube", config.position_tube),
        ("early_control_tube", config.early_control_tube),
        ("big_m_position", config.big_m_position),
        ("big_m_control", config.big_m_control),
        ("big_m_safety", config.big_m_safety),
    ):
        if value is None:
            continue
        array = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(f"{name} must be finite and nonnegative")


def _tube_array(
    value: ArrayLike,
    shape: tuple[int, ...],
    name: str,
) -> FloatArray:
    tube = _bounds_array(value, shape, name)
    if not np.all(np.isfinite(tube)) or np.any(tube < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return tube


def _big_m_array(
    configured: ArrayLike | None,
    required: FloatArray,
    name: str,
) -> FloatArray:
    """Return a valid M, deriving a tight one when the caller omits it."""

    required = np.maximum(required, 0.0)
    if configured is None:
        return required
    candidate = _tube_array(configured, required.shape, name)
    if np.any(candidate + 1e-12 < required):
        worst = float(np.max(required - candidate))
        raise ValueError(
            f"{name} is too small to deactivate unselected branch tubes "
            f"(largest shortfall {worst:.6g})"
        )
    return candidate


class _MILPBuilder:
    def __init__(self, variable_count: int):
        self.variable_count = int(variable_count)
        self.rows: list[int] = []
        self.columns: list[int] = []
        self.values: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.groups: dict[str, slice] = {}

    def add_row(
        self,
        coefficients: Mapping[int, float],
        lower: float = -np.inf,
        upper: float = np.inf,
    ) -> None:
        row = len(self.lower)
        for column, value in coefficients.items():
            if abs(value) > 1e-14:
                self.rows.append(row)
                self.columns.append(int(column))
                self.values.append(float(value))
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    def group(self, name: str, start: int) -> None:
        self.groups[name] = slice(start, len(self.lower))

    def constraint(self) -> LinearConstraint:
        matrix = sparse.coo_matrix(
            (self.values, (self.rows, self.columns)),
            shape=(len(self.lower), self.variable_count),
        ).tocsr()
        return LinearConstraint(
            matrix,
            np.asarray(self.lower, dtype=float),
            np.asarray(self.upper, dtype=float),
        )


def _allocate_layout(
    normalized: _NormalizedProblem,
    config: BigMTrajectoryMPCConfig,
) -> BigMVariableLayout:
    N = normalized.horizon
    n = normalized.state_dimension
    m = normalized.control_dimension
    P = normalized.branch_count
    position_dimension = len(normalized.position_indices)
    velocity_dimension = len(normalized.velocity_indices)

    offset = 0

    def allocate(shape: tuple[int, ...]) -> IntArray:
        nonlocal offset
        size = int(np.prod(shape, dtype=int))
        block = np.arange(offset, offset + size, dtype=np.int64).reshape(shape)
        offset += size
        return block

    state = allocate((N + 1, n))
    control = allocate((N, m))
    selector = allocate((P,))
    auxiliary: dict[str, IntArray] = {}
    if normalized.tracking_target is not None and config.tracking_weight > 0.0:
        auxiliary["tracking_abs"] = allocate((N, position_dimension))
    if velocity_dimension and config.velocity_weight > 0.0:
        auxiliary["velocity_abs"] = allocate((N, velocity_dimension))
    if config.control_weight > 0.0:
        auxiliary["control_abs"] = allocate((N, m))
    if normalized.nominal_control is not None and config.nominal_weight > 0.0:
        auxiliary["nominal_abs"] = allocate((m,))
    if normalized.terminal_target is not None and config.terminal_weight > 0.0:
        auxiliary["terminal_abs"] = allocate((position_dimension,))
    return BigMVariableLayout(
        state=state,
        control=control,
        selector=selector,
        auxiliary=auxiliary,
    )


def _branch_admission(
    normalized: _NormalizedProblem,
    config: BigMTrajectoryMPCConfig,
) -> _BranchAdmission:
    """Apply the warehouse MI-MPC branch-admission rule.

    ``branch_eligible`` is an optional caller-side hard exclusion and is never
    overridden.  Among the remaining branches, the requested threshold is
    used whenever possible.  Only when none reaches it is the first
    maximum-safety branch promoted, exactly matching ``np.argmax`` in the
    warehouse implementation.
    """

    eligible = (
        normalized.branch_eligible
        & (normalized.branch_safety >= config.safety_threshold)
    )
    if np.any(eligible):
        return _BranchAdmission(
            eligible=eligible,
            admissible=eligible.copy(),
            effective_threshold=float(config.safety_threshold),
            threshold_relaxed=False,
            promoted_branch=None,
        )

    permitted = np.flatnonzero(normalized.branch_eligible)
    if permitted.size == 0:
        return _BranchAdmission(
            eligible=eligible,
            admissible=np.zeros(normalized.branch_count, dtype=bool),
            effective_threshold=float(config.safety_threshold),
            threshold_relaxed=False,
            promoted_branch=None,
        )

    local_best = int(np.argmax(normalized.branch_safety[permitted]))
    promoted_branch = int(permitted[local_best])
    admissible = np.zeros(normalized.branch_count, dtype=bool)
    admissible[promoted_branch] = True
    effective_threshold = (
        float(normalized.branch_safety[promoted_branch])
        - _WAREHOUSE_THRESHOLD_EPSILON
    )
    return _BranchAdmission(
        eligible=eligible,
        admissible=admissible,
        effective_threshold=effective_threshold,
        threshold_relaxed=True,
        promoted_branch=promoted_branch,
    )


def _build_model_from_normalized(
    normalized: _NormalizedProblem,
    config: BigMTrajectoryMPCConfig,
    admission: _BranchAdmission | None = None,
) -> BigMMILPModel:
    N = normalized.horizon
    n = normalized.state_dimension
    m = normalized.control_dimension
    P = normalized.branch_count
    position_indices = normalized.position_indices
    position_dimension = len(position_indices)
    control_tube_steps = min(config.early_control_steps, N)

    layout = _allocate_layout(normalized, config)
    variable_count = layout.variable_count
    lower = np.full(variable_count, -np.inf, dtype=float)
    upper = np.full(variable_count, np.inf, dtype=float)
    integrality = np.zeros(variable_count, dtype=np.int64)

    lower[layout.state.reshape(-1)] = normalized.state_lower.reshape(-1)
    upper[layout.state.reshape(-1)] = normalized.state_upper.reshape(-1)
    lower[layout.control.reshape(-1)] = normalized.control_lower.reshape(-1)
    upper[layout.control.reshape(-1)] = normalized.control_upper.reshape(-1)

    branch_admission = admission or _branch_admission(normalized, config)
    lower[layout.selector] = 0.0
    upper[layout.selector] = branch_admission.admissible.astype(float)
    integrality[layout.selector] = 1
    for block in layout.auxiliary.values():
        lower[block.reshape(-1)] = 0.0

    position_tube = _tube_array(
        config.position_tube,
        (P, N, position_dimension),
        "position_tube",
    )
    position_reference = normalized.branch_states[
        :, 1:, position_indices
    ]
    position_lower = normalized.state_lower[1:, position_indices][None, :, :]
    position_upper = normalized.state_upper[1:, position_indices][None, :, :]
    required_position_m = np.maximum(
        position_upper - position_reference - position_tube,
        position_reference - position_lower - position_tube,
    )
    position_big_m = _big_m_array(
        config.big_m_position,
        required_position_m,
        "big_m_position",
    )

    if control_tube_steps:
        control_tube = _tube_array(
            config.early_control_tube,
            (P, control_tube_steps, m),
            "early_control_tube",
        )
        control_reference = normalized.branch_controls[:, :control_tube_steps, :]
        control_lower = normalized.control_lower[:control_tube_steps][None, :, :]
        control_upper = normalized.control_upper[:control_tube_steps][None, :, :]
        required_control_m = np.maximum(
            control_upper - control_reference - control_tube,
            control_reference - control_lower - control_tube,
        )
        control_big_m = _big_m_array(
            config.big_m_control,
            required_control_m,
            "big_m_control",
        )
    else:
        control_tube = np.empty((P, 0, m), dtype=float)
        control_big_m = np.empty((P, 0, m), dtype=float)
    safety_big_m = _tube_array(
        config.big_m_safety,
        (P,),
        "big_m_safety",
    )

    builder = _MILPBuilder(variable_count)

    start = len(builder.lower)
    for dimension in range(n):
        index = int(layout.state[0, dimension])
        value = float(normalized.x0[dimension])
        builder.add_row({index: 1.0}, lower=value, upper=value)
    builder.group("initial_state", start)

    start = len(builder.lower)
    for step in range(N):
        for row in range(n):
            coefficients: dict[int, float] = {
                int(layout.state[step + 1, row]): 1.0
            }
            for column in range(n):
                value = -float(normalized.A[step, row, column])
                if value:
                    index = int(layout.state[step, column])
                    coefficients[index] = coefficients.get(index, 0.0) + value
            for column in range(m):
                value = -float(normalized.B[step, row, column])
                if value:
                    index = int(layout.control[step, column])
                    coefficients[index] = coefficients.get(index, 0.0) + value
            affine = float(normalized.c[step, row])
            builder.add_row(coefficients, lower=affine, upper=affine)
    builder.group("dynamics", start)

    start = len(builder.lower)
    builder.add_row(
        {int(index): 1.0 for index in layout.selector},
        lower=1.0,
        upper=1.0,
    )
    builder.group("one_hot", start)

    start = len(builder.lower)
    for branch in range(P):
        # safety_i + M_h * (1 - z_i) >= effective_threshold
        # <=> M_h * z_i <= safety_i + M_h - effective_threshold.
        builder.add_row(
            {
                int(layout.selector[branch]): float(
                    safety_big_m[branch]
                )
            },
            upper=(
                float(normalized.branch_safety[branch])
                + float(safety_big_m[branch])
                - float(branch_admission.effective_threshold)
            ),
        )
    builder.group("safety_admission", start)

    start = len(builder.lower)
    for branch in range(P):
        selector = int(layout.selector[branch])
        for step in range(N):
            for local_dimension, state_dimension in enumerate(position_indices):
                state = int(layout.state[step + 1, state_dimension])
                reference = float(
                    position_reference[branch, step, local_dimension]
                )
                tube = float(position_tube[branch, step, local_dimension])
                big_m = float(position_big_m[branch, step, local_dimension])
                builder.add_row(
                    {state: 1.0, selector: big_m},
                    upper=reference + tube + big_m,
                )
                builder.add_row(
                    {state: -1.0, selector: big_m},
                    upper=-reference + tube + big_m,
                )
    builder.group("position_tubes", start)

    start = len(builder.lower)
    for branch in range(P):
        selector = int(layout.selector[branch])
        for step in range(control_tube_steps):
            for dimension in range(m):
                control = int(layout.control[step, dimension])
                reference = float(
                    normalized.branch_controls[branch, step, dimension]
                )
                tube = float(control_tube[branch, step, dimension])
                big_m = float(control_big_m[branch, step, dimension])
                builder.add_row(
                    {control: 1.0, selector: big_m},
                    upper=reference + tube + big_m,
                )
                builder.add_row(
                    {control: -1.0, selector: big_m},
                    upper=-reference + tube + big_m,
                )
    builder.group("control_tubes", start)

    objective = np.zeros(variable_count, dtype=float)

    start = len(builder.lower)
    tracking_abs = layout.auxiliary.get("tracking_abs")
    if tracking_abs is not None:
        assert normalized.tracking_target is not None
        for step in range(N):
            for local_dimension, state_dimension in enumerate(position_indices):
                state = int(layout.state[step + 1, state_dimension])
                auxiliary = int(tracking_abs[step, local_dimension])
                target = float(
                    normalized.tracking_target[step, local_dimension]
                )
                builder.add_row(
                    {state: 1.0, auxiliary: -1.0}, upper=target
                )
                builder.add_row(
                    {state: -1.0, auxiliary: -1.0}, upper=-target
                )
                objective[auxiliary] = config.tracking_weight
    builder.group("tracking_absolute_value", start)

    start = len(builder.lower)
    velocity_abs = layout.auxiliary.get("velocity_abs")
    if velocity_abs is not None:
        for step in range(N):
            for local_dimension, state_dimension in enumerate(
                normalized.velocity_indices
            ):
                state = int(layout.state[step + 1, state_dimension])
                auxiliary = int(velocity_abs[step, local_dimension])
                builder.add_row(
                    {state: 1.0, auxiliary: -1.0}, upper=0.0
                )
                builder.add_row(
                    {state: -1.0, auxiliary: -1.0}, upper=0.0
                )
                objective[auxiliary] = config.velocity_weight
    builder.group("velocity_absolute_value", start)

    start = len(builder.lower)
    control_abs = layout.auxiliary.get("control_abs")
    if control_abs is not None:
        for step in range(N):
            for dimension in range(m):
                control = int(layout.control[step, dimension])
                auxiliary = int(control_abs[step, dimension])
                builder.add_row(
                    {control: 1.0, auxiliary: -1.0}, upper=0.0
                )
                builder.add_row(
                    {control: -1.0, auxiliary: -1.0}, upper=0.0
                )
                objective[auxiliary] = config.control_weight
    builder.group("control_absolute_value", start)

    start = len(builder.lower)
    nominal_abs = layout.auxiliary.get("nominal_abs")
    if nominal_abs is not None:
        assert normalized.nominal_control is not None
        for dimension in range(m):
            control = int(layout.control[0, dimension])
            auxiliary = int(nominal_abs[dimension])
            reference = float(normalized.nominal_control[dimension])
            builder.add_row(
                {control: 1.0, auxiliary: -1.0}, upper=reference
            )
            builder.add_row(
                {control: -1.0, auxiliary: -1.0}, upper=-reference
            )
            objective[auxiliary] = config.nominal_weight
    builder.group("nominal_absolute_value", start)

    start = len(builder.lower)
    terminal_abs = layout.auxiliary.get("terminal_abs")
    if terminal_abs is not None:
        assert normalized.terminal_target is not None
        for local_dimension, state_dimension in enumerate(position_indices):
            state = int(layout.state[N, state_dimension])
            auxiliary = int(terminal_abs[local_dimension])
            target = float(normalized.terminal_target[local_dimension])
            builder.add_row(
                {state: 1.0, auxiliary: -1.0}, upper=target
            )
            builder.add_row(
                {state: -1.0, auxiliary: -1.0}, upper=-target
            )
            objective[auxiliary] = config.terminal_weight
    builder.group("terminal_absolute_value", start)

    objective[layout.selector] = (
        -config.safety_tiebreak_weight * normalized.branch_safety
    )

    return BigMMILPModel(
        objective=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=builder.constraint(),
        layout=layout,
        row_groups=dict(builder.groups),
        eligible_branches=branch_admission.eligible,
        admissible_branches=branch_admission.admissible,
        effective_safety_threshold=branch_admission.effective_threshold,
        safety_threshold_relaxed=branch_admission.threshold_relaxed,
        promoted_branch=branch_admission.promoted_branch,
        position_big_m=position_big_m,
        control_big_m=control_big_m,
        safety_big_m=safety_big_m,
        horizon=N,
        state_dimension=n,
        control_dimension=m,
        branch_count=P,
    )


def build_big_m_trajectory_milp(
    problem: BigMTrajectoryMPCProblem,
    config: BigMTrajectoryMPCConfig | None = None,
) -> BigMMILPModel:
    """Build, but do not solve, a SciPy MILP for structural inspection."""

    actual_config = config or BigMTrajectoryMPCConfig()
    _validate_config(actual_config)
    normalized = _normalize_problem(problem)
    return _build_model_from_normalized(normalized, actual_config)


def _solution_is_feasible(
    solution: FloatArray,
    model: BigMMILPModel,
    config: BigMTrajectoryMPCConfig,
) -> bool:
    tolerance = config.feasibility_tolerance
    if solution.shape != model.objective.shape or not np.all(
        np.isfinite(solution)
    ):
        return False

    lower = np.asarray(model.bounds.lb, dtype=float)
    upper = np.asarray(model.bounds.ub, dtype=float)
    if np.any(solution < lower - tolerance) or np.any(
        solution > upper + tolerance
    ):
        return False

    value = np.asarray(model.constraints.A @ solution, dtype=float).reshape(-1)
    constraint_lower = np.asarray(model.constraints.lb, dtype=float)
    constraint_upper = np.asarray(model.constraints.ub, dtype=float)
    if np.any(value < constraint_lower - tolerance) or np.any(
        value > constraint_upper + tolerance
    ):
        return False

    selector = solution[model.layout.selector]
    if np.any(
        np.abs(selector - np.rint(selector)) > config.integrality_tolerance
    ):
        return False
    rounded = np.rint(selector).astype(int)
    if int(np.sum(rounded)) != 1:
        return False
    selected = int(np.argmax(rounded))
    return bool(model.admissible_branches[selected])


def _optional_result_number(result: object, name: str) -> float | None:
    value = getattr(result, name, None)
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _fallback_result(
    normalized: _NormalizedProblem,
    config: BigMTrajectoryMPCConfig,
    admission: _BranchAdmission,
    *,
    status: str,
    solver_status: int | None,
    solver_message: str,
    solve_time_s: float,
    use_warehouse_branch_fallback: bool = True,
    mip_gap: float | None = None,
    mip_node_count: int | None = None,
) -> BigMTrajectoryMPCResult:
    control = None
    fallback_source = None
    fallback_branch = None
    if use_warehouse_branch_fallback:
        permitted = np.flatnonzero(normalized.branch_eligible)
        if permitted.size:
            local_best = int(
                np.argmax(normalized.branch_safety[permitted])
            )
            fallback_branch = int(permitted[local_best])
            control = normalized.branch_controls[fallback_branch, 0].copy()
            if normalized.nominal_control is not None:
                weight = float(config.fallback_branch_weight)
                control = (
                    weight * control
                    + (1.0 - weight) * normalized.nominal_control
                )
            control = np.clip(
                control,
                normalized.control_lower[0],
                normalized.control_upper[0],
            )
            fallback_source = "warehouse_max_safety_branch_blend"
    if control is None and normalized.fallback_control is not None:
        control = np.clip(
            normalized.fallback_control,
            normalized.control_lower[0],
            normalized.control_upper[0],
        )
        fallback_source = "caller_emergency_control"
    used_fallback = control is not None
    return BigMTrajectoryMPCResult(
        status=status,
        feasible=False,
        safety_feasible=False,
        used_fallback=used_fallback,
        control=control,
        selected_branch=None,
        selected_branch_safety=None,
        state_trajectory=None,
        control_trajectory=None,
        selector=None,
        objective=None,
        solve_time_s=float(solve_time_s),
        solver_status=solver_status,
        solver_message=str(solver_message),
        safety_threshold=float(config.safety_threshold),
        effective_safety_threshold=float(admission.effective_threshold),
        safety_threshold_relaxed=admission.threshold_relaxed,
        selected_branch_meets_safety_threshold=None,
        eligible_branches=tuple(np.flatnonzero(admission.eligible).tolist()),
        admissible_branches=tuple(
            np.flatnonzero(admission.admissible).tolist()
        ),
        promoted_branch=admission.promoted_branch,
        fallback_source=fallback_source,
        fallback_branch=fallback_branch,
        mip_gap=mip_gap,
        mip_node_count=mip_node_count,
    )


def solve_big_m_trajectory_mpc(
    problem: BigMTrajectoryMPCProblem,
    config: BigMTrajectoryMPCConfig | None = None,
) -> BigMTrajectoryMPCResult:
    """Solve one faithful receding-horizon Big-M trajectory MILP.

    Only the first control in a feasible returned trajectory should be applied
    before the caller rebuilds and resolves the problem at the next plant
    step.
    """

    actual_config = config or BigMTrajectoryMPCConfig()
    _validate_config(actual_config)
    normalized = _normalize_problem(problem)
    admission = _branch_admission(normalized, actual_config)
    if not np.any(admission.admissible):
        return _fallback_result(
            normalized,
            actual_config,
            admission,
            status="infeasible_no_admissible_branch",
            solver_status=None,
            solver_message=(
                "Every branch was hard-excluded by branch_eligible; "
                "warehouse emergency promotion could not be applied."
            ),
            solve_time_s=0.0,
            use_warehouse_branch_fallback=False,
        )

    model = _build_model_from_normalized(
        normalized, actual_config, admission
    )
    started_at = time.perf_counter()
    try:
        scipy_result = milp(
            c=model.objective,
            integrality=model.integrality,
            bounds=model.bounds,
            constraints=model.constraints,
            options={
                "disp": False,
                "time_limit": actual_config.time_limit_s,
                "mip_rel_gap": actual_config.mip_rel_gap,
            },
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        return _fallback_result(
            normalized,
            actual_config,
            admission,
            status="solver_exception",
            solver_status=None,
            solver_message=f"{exc.__class__.__name__}: {exc}",
            solve_time_s=elapsed,
        )
    elapsed = time.perf_counter() - started_at

    solver_status = int(scipy_result.status)
    solver_message = str(scipy_result.message)
    mip_gap = _optional_result_number(scipy_result, "mip_gap")
    node_number = _optional_result_number(scipy_result, "mip_node_count")
    mip_node_count = None if node_number is None else int(node_number)
    solution_value = getattr(scipy_result, "x", None)
    solution = (
        None
        if solution_value is None
        else np.asarray(solution_value, dtype=float).reshape(-1)
    )

    if (
        solution is not None
        and solver_status in (0, 1)
        and _solution_is_feasible(solution, model, actual_config)
    ):
        state_trajectory = solution[model.layout.state]
        control_trajectory = solution[model.layout.control]
        selector = solution[model.layout.selector]
        selected_branch = int(np.argmax(selector))
        selected_meets_requested_threshold = bool(
            admission.eligible[selected_branch]
        )
        if admission.threshold_relaxed:
            status = (
                "optimal_safety_threshold_relaxed"
                if solver_status == 0
                else (
                    "time_limit_feasible_incumbent_"
                    "safety_threshold_relaxed"
                )
            )
            solver_message = (
                f"{solver_message}; warehouse emergency admission promoted "
                f"branch {selected_branch} at effective threshold "
                f"{admission.effective_threshold:.12g}"
            )
        else:
            status = (
                "optimal"
                if solver_status == 0
                else "time_limit_feasible_incumbent"
            )
        return BigMTrajectoryMPCResult(
            status=status,
            feasible=True,
            safety_feasible=selected_meets_requested_threshold,
            used_fallback=False,
            control=control_trajectory[0].copy(),
            selected_branch=selected_branch,
            selected_branch_safety=float(
                normalized.branch_safety[selected_branch]
            ),
            state_trajectory=state_trajectory.copy(),
            control_trajectory=control_trajectory.copy(),
            selector=selector.copy(),
            objective=float(scipy_result.fun),
            solve_time_s=elapsed,
            solver_status=solver_status,
            solver_message=solver_message,
            safety_threshold=float(actual_config.safety_threshold),
            effective_safety_threshold=float(admission.effective_threshold),
            safety_threshold_relaxed=admission.threshold_relaxed,
            selected_branch_meets_safety_threshold=(
                selected_meets_requested_threshold
            ),
            eligible_branches=tuple(
                np.flatnonzero(admission.eligible).tolist()
            ),
            admissible_branches=tuple(
                np.flatnonzero(admission.admissible).tolist()
            ),
            promoted_branch=admission.promoted_branch,
            fallback_source=None,
            fallback_branch=None,
            mip_gap=mip_gap,
            mip_node_count=mip_node_count,
        )

    if solver_status == 2:
        status = "infeasible"
    elif solver_status == 3:
        status = "unbounded"
    elif solver_status == 1:
        status = "time_limit_without_feasible_incumbent"
    else:
        status = "solver_failure"
    if solution is not None and solver_status in (0, 1):
        status = "invalid_solver_incumbent"
        solver_message = (
            f"{solver_message}; incumbent failed independent feasibility "
            "or integrality validation"
        )
    return _fallback_result(
        normalized,
        actual_config,
        admission,
        status=status,
        solver_status=solver_status,
        solver_message=solver_message,
        solve_time_s=elapsed,
        mip_gap=mip_gap,
        mip_node_count=mip_node_count,
    )


__all__ = [
    "BigMMILPModel",
    "BigMTrajectoryMPCConfig",
    "BigMTrajectoryMPCProblem",
    "BigMTrajectoryMPCResult",
    "BigMVariableLayout",
    "build_big_m_trajectory_milp",
    "solve_big_m_trajectory_mpc",
]
