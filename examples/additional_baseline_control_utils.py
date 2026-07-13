"""Numerical control helpers used only by the additional baselines."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


SOLVER_INPUT_TOL = 1e-5


def project_solver_control(
    control,
    lower,
    upper,
    *,
    expected_shape: Tuple[int, ...],
    tolerance: float = SOLVER_INPUT_TOL,
) -> Optional[np.ndarray]:
    """Return an exactly bounded copy of a tolerance-feasible QP result.

    Validation happens before projection. Consequently, clipping only removes
    a solver feasibility residual; it cannot make a materially invalid result
    into a feasible candidate.
    """

    if control is None:
        return None
    value = np.asarray(control, dtype=float).reshape(-1)
    if value.shape != expected_shape or not np.all(np.isfinite(value)):
        return None
    try:
        lower_bound = np.broadcast_to(np.asarray(lower, dtype=float), value.shape)
        upper_bound = np.broadcast_to(np.asarray(upper, dtype=float), value.shape)
    except ValueError:
        return None
    if np.any(value < lower_bound - tolerance) or np.any(
        value > upper_bound + tolerance
    ):
        return None
    return np.clip(value, lower_bound, upper_bound)
