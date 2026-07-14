"""Strict actuator projection shared by the additional Quad3D baselines."""

from __future__ import annotations

from typing import Optional

import numpy as np

from examples.additional_baseline_control_utils import (
    SOLVER_INPUT_TOL,
    SolverControlProjection,
    project_solver_control,
    project_solver_control_with_diagnostics,
)


def project_quad3d_solver_control(
    control,
    lower,
    upper,
    *,
    expected_dimension: int = 4,
    tolerance: float = SOLVER_INPUT_TOL,
) -> Optional[np.ndarray]:
    """Project a tolerance-feasible Quad3D QP result to exact bounds."""

    return project_solver_control(
        control,
        lower,
        upper,
        expected_shape=(expected_dimension,),
        tolerance=tolerance,
    )


def project_quad3d_solver_control_with_diagnostics(
    control,
    lower,
    upper,
    *,
    expected_dimension: int = 4,
    tolerance: float = SOLVER_INPUT_TOL,
) -> SolverControlProjection:
    """Project a Quad3D QP result and retain the numerical correction."""

    return project_solver_control_with_diagnostics(
        control,
        lower,
        upper,
        expected_shape=(expected_dimension,),
        tolerance=tolerance,
    )
