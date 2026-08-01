"""Backup-policy definitions for the nonlinear quadrotor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, NamedTuple

import jax.numpy as jnp
import numpy as np

from .dynamics_jax import (
    NLQuad3DJaxParams,
    nominal_input_jax,
    velocity_input_jax,
)


POLICY_RADIAL = 0
POLICY_STOP = 1
POLICY_NOMINAL = 2
_POLICY_KIND = {
    "radial": POLICY_RADIAL,
    "stop": POLICY_STOP,
    "nominal": POLICY_NOMINAL,
}


@dataclass(frozen=True)
class PolicyCandidate:
    """Serializable policy description accepted by the rollout engine."""

    name: str
    kind: str
    direction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_speed: float = 0.0
    gain: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in _POLICY_KIND:
            raise ValueError(f"unknown policy kind {self.kind!r}")
        direction = np.asarray(self.direction, dtype=float)
        if direction.shape != (3,):
            raise ValueError("policy direction must contain three values")
        if self.kind == "radial" and not np.isclose(np.linalg.norm(direction), 1.0):
            raise ValueError("radial policy directions must have unit norm")
        if self.target_speed < 0.0 or self.gain <= 0.0:
            raise ValueError("target_speed must be non-negative and gain positive")


class PolicyBatch(NamedTuple):
    kinds: jnp.ndarray
    directions: jnp.ndarray
    target_speeds: jnp.ndarray
    gains: jnp.ndarray


def fibonacci_directions(count: int) -> np.ndarray:
    """Evenly distribute unit directions over the full 3-D sphere."""

    if count < 1:
        raise ValueError("count must be at least one")
    if count == 1:
        return np.asarray([[1.0, 0.0, 0.0]])
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    directions = []
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radius = np.sqrt(max(0.0, 1.0 - z * z))
        theta = index * golden_angle
        directions.append(
            [np.cos(theta) * radius, np.sin(theta) * radius, z]
        )
    return np.asarray(directions)


def make_default_candidates(
    *,
    num_radial: int = 12,
    target_speed: float = 2.8,
    radial_gain: float = 2.6,
    stop_gain: float = 3.0,
    velocity_limit: float = 3.5,
) -> tuple[PolicyCandidate, ...]:
    """Build Fibonacci-sphere evasive policies plus stop and nominal."""

    speed = min(float(target_speed), float(velocity_limit))
    candidates = [
        PolicyCandidate(
            name=f"radial_{index}",
            kind="radial",
            direction=tuple(float(value) for value in direction),
            target_speed=speed,
            gain=radial_gain,
        )
        for index, direction in enumerate(fibonacci_directions(num_radial))
    ]
    candidates.extend(
        [
            PolicyCandidate("stop", "stop", gain=stop_gain),
            PolicyCandidate("nominal", "nominal"),
        ]
    )
    return tuple(candidates)


def candidates_to_batch(
    candidates: Iterable[PolicyCandidate],
) -> PolicyBatch:
    candidates_tuple = tuple(candidates)
    if not candidates_tuple:
        raise ValueError("at least one policy candidate is required")
    return PolicyBatch(
        kinds=jnp.asarray([_POLICY_KIND[item.kind] for item in candidates_tuple]),
        directions=jnp.asarray([item.direction for item in candidates_tuple]),
        target_speeds=jnp.asarray(
            [item.target_speed for item in candidates_tuple]
        ),
        gains=jnp.asarray([item.gain for item in candidates_tuple]),
    )


def policy_control_jax(
    state: jnp.ndarray,
    kind: jnp.ndarray,
    direction: jnp.ndarray,
    target_speed: jnp.ndarray,
    gain: jnp.ndarray,
    goal: jnp.ndarray,
    dynamics_params: NLQuad3DJaxParams,
) -> jnp.ndarray:
    """Evaluate one policy with a vectorization-safe finite selection.

    ``lax.switch`` is attractive for a scalar policy, but reverse-mode AD of a
    vmapped switch materializes cotangents for inactive branches.  The
    nonlinear attitude controller can then produce ``NaN`` cotangents in an
    inactive branch even though every primal policy output is finite.  All
    three controls below are finite on the controller's physical domain, so a
    branchless selection is both semantically identical and safe to batch with
    ``vmap``.
    """

    radial_control = velocity_input_jax(
        state,
        direction * target_speed,
        gain,
        dynamics_params,
    )
    stop_control = velocity_input_jax(
        state,
        jnp.zeros(3),
        gain,
        dynamics_params,
    )
    nominal_control = nominal_input_jax(state, goal, dynamics_params)
    return jnp.where(
        kind == POLICY_RADIAL,
        radial_control,
        jnp.where(kind == POLICY_STOP, stop_control, nominal_control),
    )
