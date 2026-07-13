"""Actuator-validity regression tests for the additional baselines."""

import numpy as np
import pytest

from examples.additional_baseline_control_utils import (
    SOLVER_INPUT_TOL,
    project_solver_control,
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
