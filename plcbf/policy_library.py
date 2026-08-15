"""Numerical building blocks for policy-library control barrier functions.

The objects in this module deliberately contain no robot- or scenario-specific
logic.  A case study evaluates its backup policies, turns each value function
into one or more affine control constraints, and passes the resulting
``PolicyCertificate`` objects to :func:`select_policy`.

Halfspaces use the convention ``normal @ control >= offset`` throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import factorial, fsum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray: TypeAlias = NDArray[np.float64]


def _as_vector(value: ArrayLike, name: str, size: int | None = None) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if size is not None and array.size != size:
        raise ValueError(f"{name} must contain {size} elements, got {array.size}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=float, copy=True)


def _as_bound(value: ArrayLike, name: str, size: int) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=float)
    elif array.ndim == 1 and array.size == size:
        array = np.array(array, dtype=float, copy=True)
    else:
        raise ValueError(f"{name} must be scalar or contain {size} elements")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_box(
    reference: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    reference_array = _as_vector(reference, "reference")
    lower_array = _as_bound(lower, "lower", reference_array.size)
    upper_array = _as_bound(upper, "upper", reference_array.size)
    if np.any(lower_array > upper_array):
        raise ValueError("lower bounds must not exceed upper bounds")
    return reference_array, lower_array, upper_array


def _readonly(array: ArrayLike) -> FloatArray:
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _validate_tolerance(tolerance: float) -> float:
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    return tolerance


@dataclass(frozen=True)
class CBFHalfspace:
    """Affine control constraint in the form ``normal @ u >= offset``."""

    normal: FloatArray
    offset: float
    label: str = ""

    def __post_init__(self) -> None:
        normal = _as_vector(self.normal, "normal")
        offset = float(self.offset)
        if not np.isfinite(offset):
            raise ValueError("offset must be finite")
        object.__setattr__(self, "normal", _readonly(normal))
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "label", str(self.label))

    def residual(self, control: ArrayLike) -> float:
        """Return the signed feasibility margin at ``control``."""

        control_array = _as_vector(control, "control", self.normal.size)
        return float(self.normal @ control_array - self.offset)

    def is_satisfied(self, control: ArrayLike, tolerance: float = 1e-9) -> bool:
        """Return whether ``control`` satisfies the halfspace."""

        tolerance = _validate_tolerance(tolerance)
        return self.residual(control) >= -tolerance


HalfspaceLike: TypeAlias = CBFHalfspace | tuple[ArrayLike, float]


def _coerce_halfspace(
    halfspace: HalfspaceLike | ArrayLike,
    offset: float | None = None,
    *,
    dimension: int | None = None,
) -> CBFHalfspace:
    if isinstance(halfspace, CBFHalfspace):
        if offset is not None:
            raise ValueError("offset must be omitted when halfspace is CBFHalfspace")
        result = halfspace
    elif offset is None:
        if not isinstance(halfspace, tuple) or len(halfspace) != 2:
            raise TypeError("halfspace must be CBFHalfspace or a (normal, offset) tuple")
        result = CBFHalfspace(halfspace[0], float(halfspace[1]))
    else:
        result = CBFHalfspace(halfspace, offset)
    if dimension is not None and result.normal.size != dimension:
        raise ValueError(
            f"halfspace normal must contain {dimension} elements, "
            f"got {result.normal.size}"
        )
    return result


def cbf_halfspace(
    gradient: ArrayLike,
    drift: ArrayLike,
    control_matrix: ArrayLike,
    *,
    value: float,
    value_time_derivative: float = 0.0,
    alpha: float = 1.0,
    buffer: float = 0.0,
    label: str = "",
) -> CBFHalfspace:
    """Construct the affine control constraint for a CBF value.

    The source inequality is

    ``gradient @ (drift + control_matrix @ u)
      + value_time_derivative + alpha * (value - buffer) >= 0``.

    The returned object represents the equivalent constraint
    ``normal @ u >= offset``.
    """

    gradient_array = _as_vector(gradient, "gradient")
    drift_array = _as_vector(drift, "drift", gradient_array.size)
    matrix = np.asarray(control_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != gradient_array.size:
        raise ValueError(
            "control_matrix must be two-dimensional with one row per state"
        )
    if matrix.shape[1] == 0:
        raise ValueError("control_matrix must have at least one control column")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("control_matrix must contain only finite values")

    value = float(value)
    value_time_derivative = float(value_time_derivative)
    alpha = float(alpha)
    buffer = float(buffer)
    if not all(
        np.isfinite(item)
        for item in (value, value_time_derivative, alpha, buffer)
    ):
        raise ValueError("CBF scalar inputs must be finite")
    if alpha < 0.0:
        raise ValueError("alpha must be nonnegative")

    normal = gradient_array @ matrix
    constant = (
        float(gradient_array @ drift_array)
        + value_time_derivative
        + alpha * (value - buffer)
    )
    return CBFHalfspace(normal=normal, offset=-constant, label=label)


@dataclass(frozen=True)
class QPSolution:
    """Result of one of the small, dependency-free projection QPs."""

    control: FloatArray | None
    objective: float
    feasible: bool
    status: str
    max_violation: float
    active_constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.control is not None:
            object.__setattr__(self, "control", _readonly(self.control))
        object.__setattr__(self, "objective", float(self.objective))
        object.__setattr__(self, "max_violation", float(self.max_violation))
        object.__setattr__(
            self,
            "active_constraints",
            tuple(str(item) for item in self.active_constraints),
        )


def _box_active_constraints(
    point: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    tolerance: float,
) -> list[str]:
    active: list[str] = []
    for index, (coordinate, low, high) in enumerate(zip(point, lower, upper)):
        if abs(coordinate - low) <= tolerance:
            active.append(f"lower[{index}]")
        if abs(coordinate - high) <= tolerance:
            active.append(f"upper[{index}]")
    return active


def solve_box_halfspace_qp(
    reference: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    halfspace: HalfspaceLike | ArrayLike,
    offset: float | None = None,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> QPSolution:
    """Project ``reference`` onto a box intersected with one halfspace.

    This solves ``min 0.5 * ||u-reference||²`` without an external optimizer.
    When the halfspace is active, the KKT solution is
    ``clip(reference + multiplier * normal, lower, upper)``.  Its left-hand
    side is monotone in the nonnegative multiplier, so a bracketed bisection
    gives a robust solution even when box faces become active.
    """

    tolerance = _validate_tolerance(tolerance)
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    reference_array, lower_array, upper_array = _validated_box(
        reference, lower, upper
    )
    constraint = _coerce_halfspace(
        halfspace, offset, dimension=reference_array.size
    )

    clipped = np.clip(reference_array, lower_array, upper_array)
    residual = constraint.residual(clipped)
    if residual >= -tolerance:
        active = _box_active_constraints(
            clipped, lower_array, upper_array, tolerance
        )
        if abs(residual) <= tolerance:
            active.append(constraint.label or "halfspace")
        delta = clipped - reference_array
        return QPSolution(
            control=clipped,
            objective=0.5 * float(delta @ delta),
            feasible=True,
            status="optimal",
            max_violation=max(0.0, -residual),
            active_constraints=tuple(active),
        )

    normal = constraint.normal
    if np.linalg.norm(normal, ord=np.inf) <= tolerance:
        return QPSolution(
            control=None,
            objective=float("inf"),
            feasible=False,
            status="infeasible_zero_normal",
            max_violation=max(0.0, -residual),
        )

    maximizer = np.array(clipped, copy=True)
    maximizer[normal > 0.0] = upper_array[normal > 0.0]
    maximizer[normal < 0.0] = lower_array[normal < 0.0]
    maximum = float(normal @ maximizer)
    if maximum < constraint.offset - tolerance:
        return QPSolution(
            control=None,
            objective=float("inf"),
            feasible=False,
            status="infeasible",
            max_violation=constraint.offset - maximum,
        )

    def point_at(multiplier: float) -> FloatArray:
        return np.clip(
            reference_array + multiplier * normal,
            lower_array,
            upper_array,
        )

    low_multiplier = 0.0
    high_multiplier = 1.0
    high_point = point_at(high_multiplier)
    while (
        float(normal @ high_point) < constraint.offset
        and high_multiplier < np.finfo(float).max / 4.0
    ):
        high_multiplier *= 2.0
        high_point = point_at(high_multiplier)

    if float(normal @ high_point) < constraint.offset - tolerance:
        # This can only occur at a numerically extreme multiplier.  The box
        # maximizer is still a valid closest limiting candidate.
        high_point = maximizer

    solution = high_point
    for _ in range(max_iterations):
        multiplier = 0.5 * (low_multiplier + high_multiplier)
        candidate = point_at(multiplier)
        candidate_value = float(normal @ candidate)
        if candidate_value >= constraint.offset:
            high_multiplier = multiplier
            solution = candidate
        else:
            low_multiplier = multiplier
        if abs(candidate_value - constraint.offset) <= tolerance:
            solution = candidate
            break
        if high_multiplier - low_multiplier <= tolerance * max(
            1.0, high_multiplier
        ):
            solution = point_at(high_multiplier)
            break

    final_residual = constraint.residual(solution)
    if final_residual < -tolerance:
        solution = maximizer
        final_residual = constraint.residual(solution)
    delta = solution - reference_array
    active = _box_active_constraints(
        solution, lower_array, upper_array, tolerance * 10.0
    )
    if abs(final_residual) <= tolerance * 10.0:
        active.append(constraint.label or "halfspace")
    return QPSolution(
        control=solution,
        objective=0.5 * float(delta @ delta),
        feasible=final_residual >= -tolerance,
        status="optimal" if final_residual >= -tolerance else "numerical_failure",
        max_violation=max(0.0, -final_residual),
        active_constraints=tuple(active),
    )


def box_halfspace_volume(
    lower: ArrayLike,
    upper: ArrayLike,
    halfspace: HalfspaceLike | ArrayLike,
    offset: float | None = None,
    *,
    max_active_dimension: int = 10,
) -> float:
    """Return the exact box/halfspace intersection volume in low dimensions.

    The inclusion-exclusion formula is the CDF of a weighted sum of independent
    uniform variables.  Complexity is exponential only in the number of
    nonzero halfspace coefficients, which is intentionally capped.
    """

    lower_array = _as_vector(lower, "lower")
    upper_array = _as_vector(upper, "upper", lower_array.size)
    if np.any(lower_array > upper_array):
        raise ValueError("lower bounds must not exceed upper bounds")
    if max_active_dimension <= 0:
        raise ValueError("max_active_dimension must be positive")
    constraint = _coerce_halfspace(
        halfspace, offset, dimension=lower_array.size
    )

    widths = upper_array - lower_array
    full_volume = float(np.prod(widths))
    if full_volume == 0.0:
        return 0.0

    normal = constraint.normal
    minimum_contributions = np.where(
        normal >= 0.0, normal * lower_array, normal * upper_array
    )
    weights = np.abs(normal) * widths
    active_weights = weights[weights > 0.0]
    if active_weights.size == 0:
        return full_volume if constraint.offset <= 0.0 else 0.0
    if active_weights.size > max_active_dimension:
        raise ValueError(
            "exact volume requested with "
            f"{active_weights.size} active dimensions; cap is "
            f"{max_active_dimension}"
        )

    threshold = constraint.offset - float(np.sum(minimum_contributions))
    total_weight = float(np.sum(active_weights))
    if threshold <= 0.0:
        return full_volume
    if threshold >= total_weight:
        return 0.0

    normalized_threshold = threshold / total_weight
    normalized_weights = active_weights / total_weight
    dimension = int(active_weights.size)
    terms: list[float] = []
    for mask in range(1 << dimension):
        shifted = normalized_threshold
        bit_count = 0
        for index, weight in enumerate(normalized_weights):
            if mask & (1 << index):
                shifted -= float(weight)
                bit_count += 1
        if shifted > 0.0:
            terms.append((-1.0 if bit_count % 2 else 1.0) * shifted**dimension)

    denominator = float(factorial(dimension) * np.prod(normalized_weights))
    cdf = fsum(terms) / denominator
    feasible_fraction = float(np.clip(1.0 - cdf, 0.0, 1.0))
    return feasible_fraction * full_volume


def polygon_area(vertices: ArrayLike) -> float:
    """Return the unsigned shoelace area of a 2D polygon."""

    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("vertices must have shape (number_of_vertices, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("vertices must contain only finite values")
    if points.shape[0] < 3:
        return 0.0
    return 0.5 * abs(
        float(
            np.dot(points[:, 0], np.roll(points[:, 1], -1))
            - np.dot(points[:, 1], np.roll(points[:, 0], -1))
        )
    )


@dataclass(frozen=True)
class ClippedPolygon2D:
    """Vertices and area of a clipped 2D control rectangle."""

    vertices: FloatArray
    area: float

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("vertices must have shape (number_of_vertices, 2)")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices must contain only finite values")
        area = float(self.area)
        if not np.isfinite(area) or area < 0.0:
            raise ValueError("area must be finite and nonnegative")
        object.__setattr__(self, "vertices", _readonly(vertices))
        object.__setattr__(self, "area", area)

    @property
    def is_empty(self) -> bool:
        return self.vertices.shape[0] == 0 or self.area == 0.0


def _deduplicate_polygon(vertices: list[FloatArray], tolerance: float) -> list[FloatArray]:
    if not vertices:
        return []
    result = [vertices[0]]
    for vertex in vertices[1:]:
        if np.linalg.norm(vertex - result[-1], ord=np.inf) > tolerance:
            result.append(vertex)
    if (
        len(result) > 1
        and np.linalg.norm(result[0] - result[-1], ord=np.inf) <= tolerance
    ):
        result.pop()
    return result


def clip_rectangle_halfspaces(
    lower: ArrayLike,
    upper: ArrayLike,
    halfspaces: Iterable[HalfspaceLike],
    *,
    tolerance: float = 1e-10,
) -> ClippedPolygon2D:
    """Clip a 2D axis-aligned rectangle by affine halfspaces.

    Sutherland-Hodgman clipping preserves counter-clockwise vertex order and
    works for empty, lower-dimensional, and redundant intersections.
    """

    tolerance = _validate_tolerance(tolerance)
    lower_array = _as_vector(lower, "lower", 2)
    upper_array = _as_vector(upper, "upper", 2)
    if np.any(lower_array > upper_array):
        raise ValueError("lower bounds must not exceed upper bounds")
    constraints = tuple(
        _coerce_halfspace(item, dimension=2) for item in halfspaces
    )
    polygon: list[FloatArray] = [
        np.array([lower_array[0], lower_array[1]], dtype=float),
        np.array([upper_array[0], lower_array[1]], dtype=float),
        np.array([upper_array[0], upper_array[1]], dtype=float),
        np.array([lower_array[0], upper_array[1]], dtype=float),
    ]

    for constraint in constraints:
        if not polygon:
            break
        normal = constraint.normal
        if np.linalg.norm(normal, ord=np.inf) <= tolerance:
            if constraint.offset > tolerance:
                polygon = []
            continue
        clipped: list[FloatArray] = []
        for start, end in zip(polygon, polygon[1:] + polygon[:1]):
            start_residual = float(normal @ start - constraint.offset)
            end_residual = float(normal @ end - constraint.offset)
            start_inside = start_residual >= -tolerance
            end_inside = end_residual >= -tolerance
            if start_inside and end_inside:
                clipped.append(np.array(end, copy=True))
            elif start_inside != end_inside:
                direction = end - start
                denominator = float(normal @ direction)
                if abs(denominator) > np.finfo(float).eps:
                    fraction = (
                        constraint.offset - float(normal @ start)
                    ) / denominator
                    intersection = start + np.clip(fraction, 0.0, 1.0) * direction
                    clipped.append(intersection)
                if end_inside:
                    clipped.append(np.array(end, copy=True))
        polygon = _deduplicate_polygon(clipped, tolerance * 10.0)

    if polygon:
        vertices = np.vstack(polygon)
    else:
        vertices = np.empty((0, 2), dtype=float)
    return ClippedPolygon2D(vertices=vertices, area=polygon_area(vertices))


def rectangle_halfspaces_area(
    lower: ArrayLike,
    upper: ArrayLike,
    halfspaces: Iterable[HalfspaceLike],
    *,
    tolerance: float = 1e-10,
) -> float:
    """Convenience wrapper returning only the clipped rectangle area."""

    return clip_rectangle_halfspaces(
        lower, upper, halfspaces, tolerance=tolerance
    ).area


def _weight_matrix(weights: ArrayLike | None) -> FloatArray:
    if weights is None:
        return np.eye(2, dtype=float)
    matrix = np.asarray(weights, dtype=float)
    if matrix.ndim == 1:
        if matrix.size != 2:
            raise ValueError("weight vector must contain two elements")
        if np.any(matrix <= 0.0) or not np.all(np.isfinite(matrix)):
            raise ValueError("weight vector must be finite and positive")
        return np.diag(matrix)
    if matrix.shape != (2, 2):
        raise ValueError("weight matrix must have shape (2, 2)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("weight matrix must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError("weight matrix must be symmetric")
    if np.min(np.linalg.eigvalsh(matrix)) <= 0.0:
        raise ValueError("weight matrix must be positive definite")
    return np.array(matrix, dtype=float, copy=True)


def solve_weighted_box_halfspaces_qp_2d(
    reference: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    halfspaces: Iterable[HalfspaceLike] = (),
    *,
    weights: ArrayLike | None = None,
    tolerance: float = 1e-10,
) -> QPSolution:
    """Solve a weighted 2D projection QP by finite candidate enumeration.

    The objective is
    ``0.5 * (u-reference).T @ weights @ (u-reference)``.
    For a positive-definite weight matrix, the unique optimum is either the
    unconstrained reference, a projection onto one active boundary, or an
    intersection of two active boundaries.  Enumerating those cases is exact
    up to floating-point tolerance and avoids an optimizer dependency.
    """

    tolerance = _validate_tolerance(tolerance)
    reference_array, lower_array, upper_array = _validated_box(
        reference, lower, upper
    )
    if reference_array.size != 2:
        raise ValueError("the weighted candidate-enumeration solver is 2D only")
    matrix = _weight_matrix(weights)
    inverse = np.linalg.inv(matrix)
    extra_constraints = tuple(
        _coerce_halfspace(item, dimension=2) for item in halfspaces
    )

    normals: list[FloatArray] = [
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
    ]
    offsets = [
        float(lower_array[0]),
        float(-upper_array[0]),
        float(lower_array[1]),
        float(-upper_array[1]),
    ]
    labels = ["lower[0]", "upper[0]", "lower[1]", "upper[1]"]
    for index, constraint in enumerate(extra_constraints):
        normals.append(np.array(constraint.normal, copy=True))
        offsets.append(constraint.offset)
        labels.append(constraint.label or f"halfspace[{index}]")

    constraint_matrix = np.vstack(normals)
    offset_array = np.asarray(offsets, dtype=float)

    def feasible(point: FloatArray) -> bool:
        return bool(
            np.all(constraint_matrix @ point >= offset_array - tolerance)
        )

    def objective(point: FloatArray) -> float:
        delta = point - reference_array
        return 0.5 * float(delta @ matrix @ delta)

    candidates: list[FloatArray] = []
    if feasible(reference_array):
        candidates.append(np.array(reference_array, copy=True))

    for normal, boundary in zip(constraint_matrix, offset_array):
        inverse_normal = inverse @ normal
        denominator = float(normal @ inverse_normal)
        if denominator <= np.finfo(float).eps:
            continue
        multiplier = (boundary - float(normal @ reference_array)) / denominator
        candidate = reference_array + multiplier * inverse_normal
        if feasible(candidate):
            candidates.append(candidate)

    constraint_count = constraint_matrix.shape[0]
    for first in range(constraint_count):
        for second in range(first + 1, constraint_count):
            pair = constraint_matrix[[first, second], :]
            determinant = float(np.linalg.det(pair))
            if abs(determinant) <= tolerance:
                continue
            candidate = np.linalg.solve(
                pair, offset_array[[first, second]]
            )
            if feasible(candidate):
                candidates.append(candidate)

    if not candidates:
        # A finite violation at clipped nominal is more useful than ``nan``.
        clipped = np.clip(reference_array, lower_array, upper_array)
        violation = float(
            np.max(np.maximum(offset_array - constraint_matrix @ clipped, 0.0))
        )
        return QPSolution(
            control=None,
            objective=float("inf"),
            feasible=False,
            status="infeasible",
            max_violation=violation,
        )

    solution = min(
        candidates,
        key=lambda point: (
            objective(point),
            float(point[0]),
            float(point[1]),
        ),
    )
    residuals = constraint_matrix @ solution - offset_array
    active = tuple(
        labels[index]
        for index, residual in enumerate(residuals)
        if abs(float(residual)) <= tolerance * 10.0
    )
    return QPSolution(
        control=solution,
        objective=objective(solution),
        feasible=True,
        status="optimal",
        max_violation=max(0.0, -float(np.min(residuals))),
        active_constraints=active,
    )


def _project_onto_clipped_polygon_2d(
    reference: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    halfspaces: tuple[CBFHalfspace, ...],
    polygon: ClippedPolygon2D,
    *,
    weights: ArrayLike | None,
    tolerance: float,
) -> QPSolution | None:
    """Project onto a previously clipped feasible polygon.

    ``select_policy`` ranks two-dimensional certificates by both feasible
    input area and their minimum-intervention QP.  Reusing the polygon avoids
    solving the same rectangle/halfspace intersection twice.  ``None`` asks
    the caller to use the general active-set solver for empty or
    lower-dimensional intersections, whose feasibility semantics are retained.
    """

    vertices = np.asarray(polygon.vertices, dtype=float)
    if (
        0 < vertices.shape[0] < 3
        or (vertices.shape[0] > 0 and polygon.area <= tolerance * tolerance)
    ):
        return None
    matrix = _weight_matrix(weights)
    normals: list[FloatArray] = [
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
    ]
    offsets = [
        float(lower[0]),
        float(-upper[0]),
        float(lower[1]),
        float(-upper[1]),
    ]
    labels = ["lower[0]", "upper[0]", "lower[1]", "upper[1]"]
    for index, constraint in enumerate(halfspaces):
        normals.append(np.asarray(constraint.normal, dtype=float))
        offsets.append(float(constraint.offset))
        labels.append(constraint.label or f"halfspace[{index}]")
    constraint_matrix = np.vstack(normals)
    offset_array = np.asarray(offsets, dtype=float)

    def feasible(point: FloatArray) -> bool:
        return bool(
            np.all(constraint_matrix @ point >= offset_array - tolerance)
        )

    def objective(point: FloatArray) -> float:
        delta = point - reference
        return 0.5 * float(delta @ matrix @ delta)

    if vertices.shape[0] == 0:
        clipped = np.clip(reference, lower, upper)
        violation = float(
            np.max(
                np.maximum(
                    offset_array - constraint_matrix @ clipped,
                    0.0,
                )
            )
        )
        return QPSolution(
            control=None,
            objective=float("inf"),
            feasible=False,
            status="infeasible",
            max_violation=violation,
        )

    candidates: list[FloatArray] = []
    if feasible(reference):
        candidates.append(np.asarray(reference, dtype=float).copy())
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        direction = end - start
        denominator = float(direction @ matrix @ direction)
        if denominator <= np.finfo(float).eps:
            candidate = np.asarray(start, dtype=float).copy()
        else:
            fraction = -float(direction @ matrix @ (start - reference)) / denominator
            candidate = start + np.clip(fraction, 0.0, 1.0) * direction
        if feasible(candidate):
            candidates.append(candidate)
    if not candidates:
        return None

    solution = min(
        candidates,
        key=lambda point: (
            objective(point),
            float(point[0]),
            float(point[1]),
        ),
    )
    residuals = constraint_matrix @ solution - offset_array
    active = tuple(
        labels[index]
        for index, residual in enumerate(residuals)
        if abs(float(residual)) <= tolerance * 10.0
    )
    return QPSolution(
        control=solution,
        objective=objective(solution),
        feasible=True,
        status="optimal",
        max_violation=max(0.0, -float(np.min(residuals))),
        active_constraints=active,
    )


@dataclass(frozen=True)
class PolicyCertificate:
    """A backup policy's rollout value and affine control certificate(s).

    ``backup_control`` is the policy's direct action.  It is used only when no
    safe, QP-feasible policy exists; normal selections return the
    minimum-intervention control satisfying the selected certificate.
    """

    policy_id: str
    value: float
    halfspaces: tuple[CBFHalfspace, ...] = ()
    backup_control: FloatArray | None = None
    valid: bool = True
    diagnostic: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        policy_id = str(self.policy_id)
        if not policy_id:
            raise ValueError("policy_id must not be empty")
        value = float(self.value)
        if self.valid and not np.isfinite(value):
            raise ValueError("a valid policy certificate must have a finite value")
        constraints = tuple(
            _coerce_halfspace(item) for item in self.halfspaces
        )
        dimensions = {constraint.normal.size for constraint in constraints}
        if len(dimensions) > 1:
            raise ValueError("all policy halfspaces must have the same dimension")
        backup = None
        if self.backup_control is not None:
            backup = _as_vector(self.backup_control, "backup_control")
            if dimensions and backup.size not in dimensions:
                raise ValueError(
                    "backup_control dimension must match policy halfspaces"
                )
            backup = _readonly(backup)
        metadata = MappingProxyType(dict(sorted(self.metadata.items())))
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "halfspaces", constraints)
        object.__setattr__(self, "backup_control", backup)
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "diagnostic", str(self.diagnostic))
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_cbf(
        cls,
        policy_id: str,
        *,
        value: float,
        gradient: ArrayLike,
        drift: ArrayLike,
        control_matrix: ArrayLike,
        value_time_derivative: float = 0.0,
        alpha: float = 1.0,
        buffer: float = 0.0,
        backup_control: ArrayLike | None = None,
        valid: bool = True,
        diagnostic: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "PolicyCertificate":
        """Build a single-halfspace certificate directly from CBF terms."""

        constraint = cbf_halfspace(
            gradient,
            drift,
            control_matrix,
            value=value,
            value_time_derivative=value_time_derivative,
            alpha=alpha,
            buffer=buffer,
            label=policy_id,
        )
        return cls(
            policy_id=policy_id,
            value=value,
            halfspaces=(constraint,),
            backup_control=backup_control,
            valid=valid,
            diagnostic=diagnostic,
            metadata={} if metadata is None else metadata,
        )


class SelectionMode(str, Enum):
    """Supported policy-library maximization/minimization rules."""

    VALUE = "value"
    INPUT_VOLUME = "input_volume"
    INTERVENTION = "intervention"


@dataclass(frozen=True)
class PolicyEvaluation:
    """Diagnostics for one candidate policy."""

    policy_id: str
    value: float
    safe_value: bool
    feasible: bool
    input_volume: float
    intervention_cost: float
    control: FloatArray | None
    status: str

    def __post_init__(self) -> None:
        if self.control is not None:
            object.__setattr__(self, "control", _readonly(self.control))


@dataclass(frozen=True)
class DecisionDiagnostics:
    """Selection and fallback information suitable for logs and benchmarks."""

    mode: SelectionMode
    selected_policy_id: str | None
    used_fallback: bool
    fallback_reason: str | None
    fallback_source: str | None
    safe_policy_count: int
    feasible_policy_count: int
    eligible_policy_count: int
    evaluations: tuple[PolicyEvaluation, ...]


@dataclass(frozen=True)
class PolicyDecision:
    """Selected policy, executable control, and complete diagnostics."""

    certificate: PolicyCertificate | None
    control: FloatArray
    diagnostics: DecisionDiagnostics

    def __post_init__(self) -> None:
        object.__setattr__(self, "control", _readonly(self.control))

    @property
    def policy_id(self) -> str | None:
        return None if self.certificate is None else self.certificate.policy_id


def _parse_selection_mode(mode: SelectionMode | str) -> SelectionMode:
    if isinstance(mode, SelectionMode):
        return mode
    aliases = {
        "v": SelectionMode.VALUE,
        "value": SelectionMode.VALUE,
        "input_space": SelectionMode.INPUT_VOLUME,
        "input_volume": SelectionMode.INPUT_VOLUME,
        "minimum_intervention": SelectionMode.INTERVENTION,
        "per_policy_intervention": SelectionMode.INTERVENTION,
        "intervention": SelectionMode.INTERVENTION,
    }
    try:
        return aliases[str(mode)]
    except KeyError as error:
        choices = ", ".join(item.value for item in SelectionMode)
        raise ValueError(f"unknown selection mode {mode!r}; choose {choices}") from error


def _certificate_volume(
    certificate: PolicyCertificate,
    lower: FloatArray,
    upper: FloatArray,
    tolerance: float,
) -> float:
    if not certificate.halfspaces:
        return float(np.prod(upper - lower))
    if len(certificate.halfspaces) == 1:
        return box_halfspace_volume(
            lower, upper, certificate.halfspaces[0]
        )
    if lower.size == 2:
        return rectangle_halfspaces_area(
            lower, upper, certificate.halfspaces, tolerance=tolerance
        )
    raise ValueError(
        "multiple-halfspace input volume is supported only for 2D controls"
    )


def _certificate_qp(
    certificate: PolicyCertificate,
    reference: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    weights: ArrayLike | None,
    tolerance: float,
) -> QPSolution:
    if reference.size == 2:
        return solve_weighted_box_halfspaces_qp_2d(
            reference,
            lower,
            upper,
            certificate.halfspaces,
            weights=weights,
            tolerance=tolerance,
        )
    if weights is not None:
        raise ValueError("weighted policy intervention is supported only in 2D")
    if not certificate.halfspaces:
        clipped = np.clip(reference, lower, upper)
        delta = clipped - reference
        return QPSolution(
            control=clipped,
            objective=0.5 * float(delta @ delta),
            feasible=True,
            status="optimal",
            max_violation=0.0,
            active_constraints=tuple(
                _box_active_constraints(clipped, lower, upper, tolerance)
            ),
        )
    if len(certificate.halfspaces) > 1:
        raise ValueError(
            "multiple-halfspace QP is supported only for 2D controls"
        )
    return solve_box_halfspace_qp(
        reference,
        lower,
        upper,
        certificate.halfspaces[0],
        tolerance=tolerance,
    )


def select_policy(
    certificates: Sequence[PolicyCertificate],
    nominal_control: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    mode: SelectionMode | str = SelectionMode.INPUT_VOLUME,
    safe_value_threshold: float = 0.0,
    fallback_control: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    tolerance: float = 1e-9,
) -> PolicyDecision:
    """Select a safe policy and return its minimum-intervention control.

    Policies are eligible only when their certificate is valid, their value is
    at least ``safe_value_threshold``, and their affine control constraints are
    feasible inside the input box.  Each selection mode uses only the criteria
    in the warehouse implementation: value then library order; input volume,
    value, then library order; or intervention cost then library order.

    If no policy is eligible, the least-unsafe valid policy is identified and
    its ``backup_control`` is executed.  A global fallback is the second
    choice, and clipped nominal control is the explicit last resort.  The
    diagnostics always state why and where that fallback came from.
    """

    tolerance = _validate_tolerance(tolerance)
    parsed_mode = _parse_selection_mode(mode)
    nominal, lower_array, upper_array = _validated_box(
        nominal_control, lower, upper
    )
    threshold = float(safe_value_threshold)
    if not np.isfinite(threshold):
        raise ValueError("safe_value_threshold must be finite")
    global_fallback = None
    if fallback_control is not None:
        global_fallback = _as_vector(
            fallback_control, "fallback_control", nominal.size
        )

    certificate_tuple = tuple(certificates)
    evaluations: list[PolicyEvaluation] = []
    solutions: dict[str, QPSolution] = {}
    if len({item.policy_id for item in certificate_tuple}) != len(
        certificate_tuple
    ):
        raise ValueError("policy_id values must be unique")

    for certificate in certificate_tuple:
        if certificate.backup_control is not None and (
            certificate.backup_control.size != nominal.size
        ):
            raise ValueError(
                f"backup control for {certificate.policy_id!r} has the wrong dimension"
            )
        for constraint in certificate.halfspaces:
            if constraint.normal.size != nominal.size:
                raise ValueError(
                    f"halfspace for {certificate.policy_id!r} has the wrong dimension"
                )
        safe_value = bool(
            certificate.valid
            and np.isfinite(certificate.value)
            and certificate.value >= threshold - tolerance
        )
        if not certificate.valid:
            evaluations.append(
                PolicyEvaluation(
                    policy_id=certificate.policy_id,
                    value=certificate.value,
                    safe_value=False,
                    feasible=False,
                    input_volume=0.0,
                    intervention_cost=float("inf"),
                    control=None,
                    status="invalid_certificate",
                )
            )
            continue

        clipped_polygon: ClippedPolygon2D | None = None
        solution: QPSolution | None = None
        if nominal.size == 2 and len(certificate.halfspaces) > 1:
            clipped_polygon = clip_rectangle_halfspaces(
                lower_array,
                upper_array,
                certificate.halfspaces,
                tolerance=tolerance,
            )
            solution = _project_onto_clipped_polygon_2d(
                nominal,
                lower_array,
                upper_array,
                certificate.halfspaces,
                clipped_polygon,
                weights=weights,
                tolerance=tolerance,
            )
        if solution is None:
            solution = _certificate_qp(
                certificate,
                nominal,
                lower_array,
                upper_array,
                weights,
                tolerance,
            )
        solutions[certificate.policy_id] = solution
        volume = (
            clipped_polygon.area
            if clipped_polygon is not None
            else _certificate_volume(
                certificate, lower_array, upper_array, tolerance
            )
        )
        if not safe_value:
            status = "below_safe_value_threshold"
        else:
            status = solution.status
        evaluations.append(
            PolicyEvaluation(
                policy_id=certificate.policy_id,
                value=certificate.value,
                safe_value=safe_value,
                feasible=solution.feasible,
                input_volume=volume,
                intervention_cost=solution.objective,
                control=solution.control,
                status=status,
            )
        )

    by_id = {item.policy_id: item for item in certificate_tuple}
    library_index = {
        item.policy_id: index
        for index, item in enumerate(certificate_tuple)
    }
    eligible = [
        item for item in evaluations if item.safe_value and item.feasible
    ]

    def ordering(item: PolicyEvaluation) -> tuple[float | int, ...]:
        if parsed_mode is SelectionMode.VALUE:
            return (
                -item.value,
                library_index[item.policy_id],
            )
        if parsed_mode is SelectionMode.INPUT_VOLUME:
            return (
                -item.input_volume,
                -item.value,
                library_index[item.policy_id],
            )
        return (
            item.intervention_cost,
            library_index[item.policy_id],
        )

    if eligible:
        selected_evaluation = min(eligible, key=ordering)
        selected = by_id[selected_evaluation.policy_id]
        solution = solutions[selected.policy_id]
        assert solution.control is not None
        diagnostics = DecisionDiagnostics(
            mode=parsed_mode,
            selected_policy_id=selected.policy_id,
            used_fallback=False,
            fallback_reason=None,
            fallback_source=None,
            safe_policy_count=sum(item.safe_value for item in evaluations),
            feasible_policy_count=sum(item.feasible for item in evaluations),
            eligible_policy_count=len(eligible),
            evaluations=tuple(evaluations),
        )
        return PolicyDecision(selected, solution.control, diagnostics)

    valid_certificates = [
        item
        for item in certificate_tuple
        if item.valid and np.isfinite(item.value)
    ]
    selected = (
        min(
            valid_certificates,
            key=lambda item: (-item.value, library_index[item.policy_id]),
        )
        if valid_certificates
        else None
    )
    if not certificate_tuple:
        reason = "empty_policy_library"
    elif not any(item.safe_value for item in evaluations):
        reason = "no_safe_policy"
    else:
        reason = "safe_policies_infeasible"

    if selected is not None and selected.backup_control is not None:
        control = np.clip(selected.backup_control, lower_array, upper_array)
        source = "selected_policy_backup"
    elif global_fallback is not None:
        control = np.clip(global_fallback, lower_array, upper_array)
        source = "global_fallback"
    else:
        control = np.clip(nominal, lower_array, upper_array)
        source = "clipped_nominal"

    diagnostics = DecisionDiagnostics(
        mode=parsed_mode,
        selected_policy_id=None if selected is None else selected.policy_id,
        used_fallback=True,
        fallback_reason=reason,
        fallback_source=source,
        safe_policy_count=sum(item.safe_value for item in evaluations),
        feasible_policy_count=sum(item.feasible for item in evaluations),
        eligible_policy_count=0,
        evaluations=tuple(evaluations),
    )
    return PolicyDecision(selected, control, diagnostics)


__all__ = [
    "CBFHalfspace",
    "ClippedPolygon2D",
    "DecisionDiagnostics",
    "PolicyCertificate",
    "PolicyDecision",
    "PolicyEvaluation",
    "QPSolution",
    "SelectionMode",
    "box_halfspace_volume",
    "cbf_halfspace",
    "clip_rectangle_halfspaces",
    "polygon_area",
    "rectangle_halfspaces_area",
    "select_policy",
    "solve_box_halfspace_qp",
    "solve_weighted_box_halfspaces_qp_2d",
]
