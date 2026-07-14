"""Actuator-validity regression tests for the additional baselines."""

import cvxpy as cp
import numpy as np
import pytest

from examples.additional_baseline_control_utils import (
    SOLVER_INPUT_TOL,
    audit_cvxpy_inequalities,
    project_solver_control,
    project_solver_control_with_diagnostics,
)


@pytest.mark.parametrize(
    ("control", "expected"),
    [
        (
            np.array([1.0 + 0.5 * SOLVER_INPUT_TOL, -2.0]),
            np.array([1.0, -2.0]),
        ),
        (
            np.array([-1.0 - 0.5 * SOLVER_INPUT_TOL, 2.0]),
            np.array([-1.0, 2.0]),
        ),
    ],
)
def test_tolerance_feasible_solver_control_is_projected_exactly(control, expected):
    projected = project_solver_control(
        control,
        np.array([-1.0, -2.0]),
        np.array([1.0, 2.0]),
        expected_shape=(2,),
    )

    np.testing.assert_array_equal(projected, expected)


@pytest.mark.parametrize(
    "control",
    [
        np.array([1.0 + 2.0 * SOLVER_INPUT_TOL, 0.0]),
        np.array([np.nan, 0.0]),
        np.zeros(3),
        None,
    ],
)
def test_invalid_solver_control_is_not_projected_into_feasibility(control):
    assert project_solver_control(
        control,
        -1.0,
        1.0,
        expected_shape=(2,),
    ) is None


def test_projection_diagnostics_record_exact_infinity_norm():
    raw = np.array([1.0 + 0.25 * SOLVER_INPUT_TOL, -2.0])
    result = project_solver_control_with_diagnostics(
        raw,
        np.array([-1.0, -2.0]),
        np.array([1.0, 2.0]),
        expected_shape=(2,),
    )

    assert result.projection_applied is True
    assert result.projection_delta_inf == pytest.approx(0.25 * SOLVER_INPUT_TOL)
    np.testing.assert_array_equal(result.control, np.array([1.0, -2.0]))


def test_post_projection_audit_rejects_worsened_tight_constraint():
    control = cp.Variable(1)
    constraints = [control >= 1.0 + 4e-6, control <= 1.0]
    projection = project_solver_control_with_diagnostics(
        np.array([1.0 + 5e-6]),
        -1.0,
        1.0,
        expected_shape=(1,),
    )

    audit = audit_cvxpy_inequalities(
        constraints,
        [(control, projection.control)],
        absolute_tolerance=1e-7,
        relative_tolerance=1e-7,
    )

    assert projection.projection_applied is True
    assert audit.passed is False
    assert audit.constraint_count == 2
    assert audit.max_violation == pytest.approx(4e-6)
    assert audit.max_violation_ratio > 1.0


def test_post_projection_audit_accepts_violation_within_declared_tolerance():
    control = cp.Variable(1)
    constraints = [control >= 1.0 + 4e-6, control <= 1.0]
    audit = audit_cvxpy_inequalities(
        constraints,
        [(control, np.array([1.0]))],
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )

    assert audit.passed is True
    assert audit.max_violation == pytest.approx(4e-6)
    assert audit.max_violation_ratio < 1.0
