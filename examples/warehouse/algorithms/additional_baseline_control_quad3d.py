"""Control projection for the additional Quad3D baselines."""

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
    return project_solver_control_with_diagnostics(
        control,
        lower,
        upper,
        expected_shape=(expected_dimension,),
        tolerance=tolerance,
    )
