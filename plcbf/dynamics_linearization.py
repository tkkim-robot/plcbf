"""Local affine discrete-time models for trajectory MPC."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
DiscreteStep = Callable[[FloatArray, FloatArray], FloatArray]


def linearize_discrete_trajectory(
    step: DiscreteStep,
    reference_states: ArrayLike,
    reference_controls: ArrayLike,
    *,
    state_steps: ArrayLike | float = 1e-5,
    control_steps: ArrayLike | float = 1e-5,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Linearize ``x+ = step(x, u)`` along a reference trajectory.

    Returns time-varying matrices satisfying

    ``x[k+1] ≈ A[k] x[k] + B[k] u[k] + c[k]``.

    Central differences include state-dependent feedback, integration,
    saturation, and other behavior implemented by the supplied plant step.
    """

    states = np.asarray(reference_states, dtype=float)
    controls = np.asarray(reference_controls, dtype=float)
    if states.ndim != 2 or controls.ndim != 2:
        raise ValueError("reference states and controls must be matrices")
    if states.shape[0] != controls.shape[0] + 1:
        raise ValueError("reference states must contain one extra sample")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls)):
        raise ValueError("reference trajectories must be finite")
    horizon, control_dimension = controls.shape
    state_dimension = states.shape[1]
    state_eps = np.broadcast_to(
        np.asarray(state_steps, dtype=float), (state_dimension,)
    ).copy()
    control_eps = np.broadcast_to(
        np.asarray(control_steps, dtype=float), (control_dimension,)
    ).copy()
    if (
        not np.all(np.isfinite(state_eps))
        or not np.all(np.isfinite(control_eps))
        or np.any(state_eps <= 0.0)
        or np.any(control_eps <= 0.0)
    ):
        raise ValueError("finite-difference steps must be finite and positive")

    matrices_a = np.zeros((horizon, state_dimension, state_dimension))
    matrices_b = np.zeros((horizon, state_dimension, control_dimension))
    affine = np.zeros((horizon, state_dimension))
    for time_index in range(horizon):
        state = states[time_index].copy()
        control = controls[time_index].copy()
        next_reference = np.asarray(step(state, control), dtype=float).reshape(-1)
        if next_reference.shape != (state_dimension,):
            raise ValueError("step returned a state with the wrong dimension")
        for index, epsilon in enumerate(state_eps):
            plus = state.copy()
            minus = state.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            matrices_a[time_index, :, index] = (
                np.asarray(step(plus, control), dtype=float).reshape(-1)
                - np.asarray(step(minus, control), dtype=float).reshape(-1)
            ) / (2.0 * epsilon)
        for index, epsilon in enumerate(control_eps):
            plus = control.copy()
            minus = control.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            matrices_b[time_index, :, index] = (
                np.asarray(step(state, plus), dtype=float).reshape(-1)
                - np.asarray(step(state, minus), dtype=float).reshape(-1)
            ) / (2.0 * epsilon)
        affine[time_index] = (
            next_reference
            - matrices_a[time_index] @ state
            - matrices_b[time_index] @ control
        )
    return matrices_a, matrices_b, affine


__all__ = ["linearize_discrete_trajectory"]
