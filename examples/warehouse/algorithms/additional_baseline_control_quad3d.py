"""Strict actuator projection shared by the additional Quad3D baselines."""

from __future__ import annotations

from typing import Optional

import numpy as np

from examples.additional_baseline_control_utils import (
    SOLVER_INPUT_TOL,
    project_solver_control,
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
