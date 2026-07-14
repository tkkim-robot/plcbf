"""Numerical control helpers used only by the additional baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np


SOLVER_INPUT_TOL = 1e-5


@dataclass(frozen=True)
class SolverControlProjection:
    """A tolerance-valid solver control and its exact-bound projection."""

    control: Optional[np.ndarray]
    projection_applied: bool
    projection_delta_inf: float


@dataclass(frozen=True)
class ConstraintAudit:
    """Post-projection residual audit for the QP's original inequalities."""

    passed: bool
    constraint_count: int
    max_violation: float
    max_tolerance: float
    max_violation_ratio: float
    absolute_tolerance: float
    relative_tolerance: float


def project_solver_control(
    control,
    lower,
    upper,
    *,
    expected_shape: Tuple[int, ...],
    tolerance: float = SOLVER_INPUT_TOL,
) -> Optional[np.ndarray]:
    """Return an exactly bounded copy of a tolerance-feasible QP result.

    Validation happens before projection, so clipping only removes a small
    actuator-bound residual. Callers must separately recheck every other QP
    inequality before accepting the projected candidate.
    """

    return project_solver_control_with_diagnostics(
        control,
        lower,
        upper,
        expected_shape=expected_shape,
        tolerance=tolerance,
    ).control


def project_solver_control_with_diagnostics(
    control,
    lower,
    upper,
    *,
    expected_shape: Tuple[int, ...],
    tolerance: float = SOLVER_INPUT_TOL,
) -> SolverControlProjection:
    """Project a solver-tolerance-valid input and retain projection metadata."""

    invalid = SolverControlProjection(
        control=None,
        projection_applied=False,
        projection_delta_inf=float("nan"),
    )
    if control is None:
        return invalid
    value = np.asarray(control, dtype=float).reshape(-1)
    if value.shape != expected_shape or not np.all(np.isfinite(value)):
        return invalid
    try:
        lower_bound = np.broadcast_to(np.asarray(lower, dtype=float), value.shape)
        upper_bound = np.broadcast_to(np.asarray(upper, dtype=float), value.shape)
    except ValueError:
        return invalid
    if np.any(value < lower_bound - tolerance) or np.any(
        value > upper_bound + tolerance
    ):
        return invalid

    projected = np.clip(value, lower_bound, upper_bound)
    delta_inf = float(np.max(np.abs(projected - value))) if value.size else 0.0
    return SolverControlProjection(
        control=projected,
        projection_applied=bool(delta_inf > 0.0),
        projection_delta_inf=delta_inf,
    )


def audit_cvxpy_inequalities(
    constraints: Sequence[Any],
    assignments: Sequence[Tuple[Any, Any]],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> ConstraintAudit:
    """Audit the original affine CVXPY inequalities at assigned values.

    CVXPY stores every inequality in canonical form ``expr <= 0``. For each
    scalar row ``c.T @ z + d <= 0``, this function uses the declared scale-aware
    post-projection audit threshold

    ``absolute_tolerance + relative_tolerance * max(1, |d|, |c.T @ z|)``.

    Variable attributes are not included in ``problem.constraints``. The
    nonnegative attribute used by the drift PCBF slack is therefore audited
    explicitly as ``-z <= 0``.
    """

    absolute_tolerance = float(absolute_tolerance)
    relative_tolerance = float(relative_tolerance)
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("constraint audit tolerances must be nonnegative")

    variables = [
        (variable, np.asarray(value, dtype=float))
        for variable, value in assignments
    ]
    original_values = {
        variable: None if variable.value is None else np.array(variable.value, copy=True)
        for variable, _ in variables
    }
    violations: list[np.ndarray] = []
    tolerances: list[np.ndarray] = []

    try:
        for variable, value in variables:
            if variable.shape == ():
                if value.size != 1:
                    raise ValueError("scalar CVXPY assignment must contain one value")
                assigned_value = float(value.reshape(-1)[0])
            else:
                if value.size != variable.size:
                    raise ValueError("CVXPY assignment has the wrong dimension")
                assigned_value = value.reshape(variable.shape, order="F")
            # ``save_value`` intentionally bypasses variable-attribute projection
            # so a solver-returned nonnegative slack is audited exactly as returned.
            variable.save_value(assigned_value)

        for constraint in constraints:
            if not isinstance(constraint, cp.constraints.nonpos.Inequality):
                raise TypeError("only affine CVXPY inequalities can be audited")
            expression = constraint.expr
            if not expression.is_affine():
                raise TypeError("post-projection audit requires affine constraints")
            expression_value = np.asarray(expression.value, dtype=float).reshape(
                -1, order="F"
            )
            linear_value = np.zeros_like(expression_value)

            for variable, value in variables:
                gradient = expression.grad.get(variable)
                if gradient is None:
                    continue
                gradient_array = (
                    gradient.toarray()
                    if hasattr(gradient, "toarray")
                    else np.asarray(gradient)
                )
                jacobian = np.asarray(gradient_array, dtype=float).reshape(
                    (variable.size, expression.size), order="F"
                ).T
                flat_value = value.reshape(-1, order="F")
                linear_value += jacobian @ flat_value

            constant = expression_value - linear_value
            scale = np.maximum.reduce(
                [
                    np.ones_like(expression_value),
                    np.abs(constant),
                    np.abs(linear_value),
                ]
            )
            violations.append(np.maximum(expression_value, 0.0))
            tolerances.append(absolute_tolerance + relative_tolerance * scale)

        for variable, value in variables:
            if bool(variable.attributes.get("nonneg", False)):
                flat_value = value.reshape(-1, order="F")
                scale = np.maximum(1.0, np.abs(flat_value))
                violations.append(np.maximum(-flat_value, 0.0))
                tolerances.append(
                    absolute_tolerance + relative_tolerance * scale
                )
    finally:
        for variable, _ in variables:
            variable.save_value(original_values[variable])

    constraint_count = int(sum(values.size for values in violations))
    if constraint_count == 0:
        return ConstraintAudit(
            passed=True,
            constraint_count=0,
            max_violation=0.0,
            max_tolerance=0.0,
            max_violation_ratio=0.0,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )

    all_violations = np.concatenate(violations)
    all_tolerances = np.concatenate(tolerances)
    finite = bool(
        np.all(np.isfinite(all_violations))
        and np.all(np.isfinite(all_tolerances))
        and np.all(all_tolerances > 0.0)
    )
    ratios = (
        all_violations / all_tolerances
        if finite
        else np.full_like(all_violations, np.inf)
    )
    return ConstraintAudit(
        passed=bool(finite and np.all(all_violations <= all_tolerances)),
        constraint_count=constraint_count,
        max_violation=float(np.max(all_violations)) if finite else float("inf"),
        max_tolerance=float(np.max(all_tolerances)) if finite else float("inf"),
        max_violation_ratio=float(np.max(ratios)) if finite else float("inf"),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
