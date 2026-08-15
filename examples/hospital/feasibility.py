"""Controller-independent dynamic feasibility witnesses for Hospital worlds.

The publication benchmark conditions randomized human traffic on the existence
of at least one dynamically feasible room-refuge trajectory.  This module is
deliberately below the controller layer: it imports no Hospital controller,
policy library, PL-CBF implementation, QP, or compared baseline.  A witness is
an offline construction certificate only and is never executed by a benchmark
method.

Each witness starts from the exact story initial state, follows a deterministic
geometry-derived route into any eligible room, remains there until the fixed
convoy has cleared, then exits and reaches the nominal goal.  Candidate states
are propagated with the benchmark double-integrator plant and checked against
static geometry and synchronized deterministic obstacle motion at every plant
step.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from math import ceil, isfinite
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
from jax.experimental import enable_x64
import numpy as np

from .config import HospitalConfig
from .dynamics import step_double_integrator, waypoint_control
from .environment import HospitalEnvironment, Rect, Room
from .obstacles import Human, Stretcher
from .planner import HospitalGridPlanner


DYNAMIC_FEASIBILITY_SCHEMA = "hospital_dynamic_refuge_witness_v1"
MINIMUM_ROOM_ENTRY_LEAD_S = 1.0
POST_CONVOY_HOLD_BUFFER_S = 0.6
# The benchmark's operational safe set is nonnegative clearance.  The witness
# evaluates the same synchronized ``Human.predicted`` samples as the simulator,
# so no interpolation-error buffer is needed.
MINIMUM_OPERATIONAL_CLEARANCE_M = 0.0
MAXIMUM_WITNESS_TIME_S = 180.0
GOAL_TOLERANCE_M = 1.35
WITNESS_DYNAMIC_SUBSTEPS = 8
WITNESS_NOMINAL_BLOCK_STEPS = 32
POST_CONVOY_DEPARTURE_DELAYS_S = tuple(float(value) for value in range(0, 22, 2))
WITNESS_HUMAN_CAUTION_CLEARANCE_M = 2.8
WITNESS_PREFERRED_CLEARANCE_M = 0.6


@dataclass(frozen=True)
class DynamicFeasibilityAudit:
    """Method-independent proof that one complete refuge behavior exists."""

    schema: str
    valid: bool
    blocker_only: bool
    candidate_room_count: int
    witness_room_label: str | None
    room_entry_time_s: float | None
    room_blockage_started_at_s: float | None
    room_entry_lead_margin_s: float | None
    convoy_cleared_at_s: float
    room_exit_time_s: float | None
    goal_reached_time_s: float | None
    post_convoy_departure_delay_s: float | None
    minimum_physical_clearance_m: float | None
    minimum_operational_clearance_m: float | None
    maximum_speed_mps: float | None
    maximum_control_component_mps2: float | None
    trajectory_sha256: str | None
    failure_reason: str | None

    def require_valid(self) -> None:
        if not self.valid:
            raise RuntimeError(
                "hospital world lacks a controller-independent dynamic "
                f"refuge witness: {self.failure_reason or 'unknown failure'}"
            )

    def public_metadata(self) -> dict[str, object]:
        """Return witness facts that reveal no privileged room or controls."""

        return {
            "dynamic_refuge_feasibility_schema": self.schema,
            "dynamic_refuge_feasibility_verified": self.valid,
            "dynamic_refuge_candidate_room_count": self.candidate_room_count,
            "dynamic_refuge_entry_lead_margin_s": (
                self.room_entry_lead_margin_s
            ),
            "dynamic_refuge_minimum_physical_clearance_m": (
                self.minimum_physical_clearance_m
            ),
            "dynamic_refuge_minimum_operational_clearance_m": (
                self.minimum_operational_clearance_m
            ),
            "dynamic_refuge_goal_reached_time_s": self.goal_reached_time_s,
            "dynamic_refuge_trajectory_sha256": self.trajectory_sha256,
            "dynamic_refuge_witness_is_controller_input": False,
        }


@dataclass(frozen=True)
class _RouteCandidate:
    room: Room
    entry_waypoints: tuple[np.ndarray, ...]
    exit_waypoints: tuple[np.ndarray, ...]
    doorway_station_m: float
    blockage_started_at_s: float


@dataclass
class _Replay:
    state: np.ndarray
    time_s: float = 0.0
    step_index: int = 0
    minimum_physical_clearance_m: float = float("inf")
    minimum_operational_clearance_m: float = float("inf")
    maximum_speed_mps: float = 0.0
    maximum_control_component_mps2: float = 0.0
    failed: bool = False
    failure_reason: str | None = None


@dataclass(frozen=True)
class _TrafficPrediction:
    """Plant-step obstacle centers shared by every room witness replay."""

    human_centers: np.ndarray
    human_swept_centers: np.ndarray
    human_radii: np.ndarray
    blocker_centers: np.ndarray
    blocker_half_extents: np.ndarray
    human_centers_jax: jax.Array
    human_swept_centers_jax: jax.Array
    human_radii_jax: jax.Array
    blocker_centers_jax: jax.Array
    blocker_half_extents_jax: jax.Array
    floor_bounds_jax: jax.Array
    floor_boundary_starts_jax: jax.Array
    floor_boundary_ends_jax: jax.Array
    wall_centers_jax: jax.Array
    wall_half_jax: jax.Array
    clearance_parameters_jax: jax.Array
    witness_plant_parameters_jax: jax.Array


@dataclass(frozen=True)
class _WitnessStep:
    """One already-propagated and already-certified plant transition."""

    control: np.ndarray
    following_state: np.ndarray
    physical_clearance_m: float
    operational_clearance_m: float


_ROUTE_CACHE: dict[tuple[object, ...], tuple[_RouteCandidate, ...]] = {}


@partial(
    jax.jit,
    static_argnames=(
        "step_count",
        "prediction_chunk_count",
    ),
)
def _precompute_human_centers_jax(
    initial_positions: jax.Array,
    initial_velocities: jax.Array,
    radii: jax.Array,
    floor_bounds: jax.Array,
    floor_boundary_starts: jax.Array,
    floor_boundary_ends: jax.Array,
    wall_centers: jax.Array,
    wall_half: jax.Array,
    dt: jax.Array,
    *,
    step_count: int,
    prediction_chunk_count: int,
) -> tuple[jax.Array, jax.Array]:
    """Compile production carries and exact simulator swept predictions."""

    def collisions(
        points: jax.Array,
        query_radii: jax.Array,
    ) -> jax.Array:
        lower = floor_bounds[:, :2]
        upper = lower + floor_bounds[:, 2:]
        on_floor = jnp.any(
            jnp.all(
                (points[:, None, :] >= lower[None, :, :])
                & (points[:, None, :] <= upper[None, :, :]),
                axis=2,
            ),
            axis=1,
        )
        segments = floor_boundary_ends - floor_boundary_starts
        lengths_squared = jnp.sum(segments * segments, axis=1)
        relative = points[:, None, :] - floor_boundary_starts[None, :, :]
        fractions = jnp.clip(
            jnp.sum(relative * segments[None, :, :], axis=2)
            / jnp.maximum(lengths_squared[None, :], 1.0e-18),
            0.0,
            1.0,
        )
        closest = (
            floor_boundary_starts[None, :, :]
            + fractions[:, :, None] * segments[None, :, :]
        )
        floor_distances = jnp.min(
            jnp.linalg.norm(points[:, None, :] - closest, axis=2),
            axis=1,
        )
        floor_margin = jnp.where(
            on_floor, floor_distances, -floor_distances
        ) - query_radii
        wall_q = (
            jnp.abs(points[:, None, :] - wall_centers[None, :, :])
            - wall_half[None, :, :]
        )
        wall_signed = (
            jnp.linalg.norm(jnp.maximum(wall_q, 0.0), axis=2)
            + jnp.minimum(
                jnp.maximum(wall_q[:, :, 0], wall_q[:, :, 1]), 0.0
            )
        )
        wall_margin = jnp.min(wall_signed, axis=1) - query_radii
        return jnp.minimum(floor_margin, wall_margin) <= 0.0

    def advance_once(
        positions: jax.Array,
        velocities: jax.Array,
        elapsed: jax.Array,
        query_radii: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Vectorized, float64-equivalent implementation of Human.advance."""

        elapsed_values = jnp.broadcast_to(elapsed, (positions.shape[0],))
        next_positions = (
            positions + velocities * elapsed_values[:, None]
        )
        full_collision = collisions(next_positions, query_radii)

        def resolve_collisions(
            _unused: None,
        ) -> tuple[jax.Array, jax.Array]:
            next_x = jnp.column_stack(
                (
                    positions[:, 0]
                    + velocities[:, 0] * elapsed_values,
                    positions[:, 1],
                )
            )
            next_y = jnp.column_stack(
                (
                    positions[:, 0],
                    positions[:, 1]
                    + velocities[:, 1] * elapsed_values,
                )
            )
            free_x = ~collisions(next_x, query_radii)
            free_y = ~collisions(next_y, query_radii)
            neither = ~(free_x | free_y)
            collided_positions = jnp.column_stack(
                (
                    jnp.where(free_x, next_x[:, 0], positions[:, 0]),
                    jnp.where(free_y, next_y[:, 1], positions[:, 1]),
                )
            )
            collided_velocities = jnp.column_stack(
                (
                    jnp.where(
                        free_y | neither,
                        -velocities[:, 0],
                        velocities[:, 0],
                    ),
                    jnp.where(
                        free_x | neither,
                        -velocities[:, 1],
                        velocities[:, 1],
                    ),
                )
            )
            return (
                jnp.where(
                    full_collision[:, None],
                    collided_positions,
                    next_positions,
                ),
                jnp.where(
                    full_collision[:, None],
                    collided_velocities,
                    velocities,
                ),
            )

        return jax.lax.cond(
            jnp.any(full_collision),
            resolve_collisions,
            lambda _unused: (next_positions, velocities),
            operand=None,
        )

    prediction_elapsed = (
        jnp.linspace(
            0.0,
            1.0,
            WITNESS_DYNAMIC_SUBSTEPS + 1,
            dtype=dt.dtype,
        )
        * dt
    )

    def predicted_centers(
        positions: jax.Array,
        velocities: jax.Array,
    ) -> jax.Array:
        # Human.predicted repeatedly calls advance with chunks <= 0.05 s.
        sample_count = WITNESS_DYNAMIC_SUBSTEPS + 1
        human_count = positions.shape[0]
        flat_positions = jnp.broadcast_to(
            positions[None, :, :],
            (sample_count, human_count, 2),
        ).reshape(-1, 2)
        flat_velocities = jnp.broadcast_to(
            velocities[None, :, :],
            (sample_count, human_count, 2),
        ).reshape(-1, 2)
        flat_remaining = jnp.broadcast_to(
            prediction_elapsed[:, None],
            (sample_count, human_count),
        ).reshape(-1)
        flat_radii = jnp.broadcast_to(
            radii[None, :],
            (sample_count, human_count),
        ).reshape(-1)

        def chunk(
            _index: int,
            carry: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            chunk_positions, chunk_velocities, remaining = carry
            chunk_elapsed = jnp.minimum(remaining, 0.05)
            next_positions, next_velocities = advance_once(
                chunk_positions,
                chunk_velocities,
                chunk_elapsed,
                flat_radii,
            )
            return (
                next_positions,
                next_velocities,
                jnp.maximum(remaining - chunk_elapsed, 0.0),
            )

        following_positions, _, _ = jax.lax.fori_loop(
            0,
            prediction_chunk_count,
            chunk,
            (flat_positions, flat_velocities, flat_remaining),
        )
        return following_positions.reshape(sample_count, human_count, 2)

    def advance(
        carry: tuple[jax.Array, jax.Array],
        _unused: None,
    ) -> tuple[
        tuple[jax.Array, jax.Array],
        tuple[jax.Array, jax.Array],
    ]:
        positions, velocities = carry
        swept_centers = predicted_centers(positions, velocities)
        following_positions, following_velocities = advance_once(
            positions,
            velocities,
            dt,
            radii,
        )
        return (
            following_positions,
            following_velocities,
        ), (following_positions, swept_centers)

    (_, _), (suffix, swept_centers) = jax.lax.scan(
        advance,
        (initial_positions, initial_velocities),
        xs=None,
        length=step_count - 1,
    )
    centers = jnp.concatenate(
        (initial_positions[None, :, :], suffix),
        axis=0,
    )
    return centers, swept_centers


@jax.jit
def _swept_clearances_jax(
    transitions: jax.Array,
    human_centers: jax.Array,
    human_swept_centers: jax.Array,
    human_radii: jax.Array,
    blocker_centers: jax.Array,
    blocker_half_extents: jax.Array,
    floor_bounds: jax.Array,
    floor_boundary_starts: jax.Array,
    floor_boundary_ends: jax.Array,
    wall_centers: jax.Array,
    wall_half: jax.Array,
    parameters: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Exact batched swept clearance for one double-integrator plant step.

    This is the same floor-union, explicit-wall, circular-human, and
    axis-aligned stretcher geometry used by the scalar audit.  JIT changes
    only evaluation cost; it does not introduce a clearance approximation.
    """

    starts = transitions[:, :4]
    ends = transitions[:, 4:8]
    step_indices = transitions[:, 8].astype(jnp.int32)
    (
        robot_radius,
        safety_margin,
        human_margin,
        stretcher_margin,
        static_margin,
    ) = parameters
    fractions = jnp.linspace(
        0.0,
        1.0,
        WITNESS_DYNAMIC_SUBSTEPS + 1,
        dtype=starts.dtype,
    )
    robot_positions = (
        starts[:, None, :2]
        + fractions[None, :, None]
        * (ends[:, None, :2] - starts[:, None, :2])
    )
    points = robot_positions.reshape(-1, 2)

    lower = floor_bounds[:, :2]
    upper = lower + floor_bounds[:, 2:]
    on_floor = jnp.any(
        jnp.all(
            (points[:, None, :] >= lower[None, :, :])
            & (points[:, None, :] <= upper[None, :, :]),
            axis=2,
        ),
        axis=1,
    )
    segments = floor_boundary_ends - floor_boundary_starts
    lengths_squared = jnp.sum(segments * segments, axis=1)
    relative = points[:, None, :] - floor_boundary_starts[None, :, :]
    segment_fractions = jnp.clip(
        jnp.sum(relative * segments[None, :, :], axis=2)
        / jnp.maximum(lengths_squared[None, :], 1.0e-18),
        0.0,
        1.0,
    )
    closest = (
        floor_boundary_starts[None, :, :]
        + segment_fractions[:, :, None] * segments[None, :, :]
    )
    floor_distances = jnp.min(
        jnp.linalg.norm(points[:, None, :] - closest, axis=2),
        axis=1,
    )
    floor_signed = jnp.where(on_floor, floor_distances, -floor_distances)

    wall_q = (
        jnp.abs(points[:, None, :] - wall_centers[None, :, :])
        - wall_half[None, :, :]
    )
    wall_signed = (
        jnp.linalg.norm(jnp.maximum(wall_q, 0.0), axis=2)
        + jnp.minimum(
            jnp.maximum(wall_q[:, :, 0], wall_q[:, :, 1]),
            0.0,
        )
    )
    static_signed = jnp.minimum(
        floor_signed,
        jnp.min(wall_signed, axis=1),
    ).reshape(starts.shape[0], WITNESS_DYNAMIC_SUBSTEPS + 1)
    physical = jnp.min(static_signed - robot_radius, axis=1)
    operational = jnp.min(
        static_signed - robot_radius - static_margin,
        axis=1,
    )

    if human_centers.shape[1]:
        swept_humans = human_swept_centers[step_indices]
        human_distance = jnp.linalg.norm(
            robot_positions[:, :, None, :]
            - swept_humans,
            axis=3,
        ) - human_radii[None, None, :]
        physical = jnp.minimum(
            physical,
            jnp.min(human_distance - robot_radius, axis=(1, 2)),
        )
        operational = jnp.minimum(
            operational,
            jnp.min(
                human_distance
                - robot_radius
                - safety_margin
                - human_margin,
                axis=(1, 2),
            ),
        )

    if blocker_centers.shape[1]:
        blocker_start = blocker_centers[step_indices]
        blocker_end = blocker_centers[step_indices + 1]
        swept_blockers = (
            blocker_start[:, None, :, :]
            + fractions[None, :, None, None]
            * (blocker_end - blocker_start)[:, None, :, :]
        )
        blocker_q = (
            jnp.abs(
                robot_positions[:, :, None, :]
                - swept_blockers
            )
            - blocker_half_extents[None, None, :, :]
        )
        blocker_distance = (
            jnp.linalg.norm(jnp.maximum(blocker_q, 0.0), axis=3)
            + jnp.minimum(
                jnp.maximum(blocker_q[:, :, :, 0], blocker_q[:, :, :, 1]),
                0.0,
            )
        )
        physical = jnp.minimum(
            physical,
            jnp.min(blocker_distance - robot_radius, axis=(1, 2)),
        )
        operational = jnp.minimum(
            operational,
            jnp.min(
                blocker_distance
                - robot_radius
                - safety_margin
                - stretcher_margin,
                axis=(1, 2),
            ),
        )
    return physical, operational


@partial(jax.jit, static_argnames=("block_steps",))
def _nominal_block_jax(
    request: jax.Array,
    human_centers: jax.Array,
    human_swept_centers: jax.Array,
    human_radii: jax.Array,
    blocker_centers: jax.Array,
    blocker_half_extents: jax.Array,
    floor_bounds: jax.Array,
    floor_boundary_starts: jax.Array,
    floor_boundary_ends: jax.Array,
    wall_centers: jax.Array,
    wall_half: jax.Array,
    clearance_parameters: jax.Array,
    plant_parameters: jax.Array,
    *,
    block_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Propagate a nominal-only prefix for amortized exact verification.

    The caller accepts states only until the first traffic-caution or unsafe
    transition.  That boundary step is then evaluated by the existing scalar
    witness rule, including its acceleration lattice, so this block changes no
    controller-independent witness semantics.
    """

    initial_state = request[:4]
    target = request[4:6]
    target_speed = request[6]
    first_step = request[7].astype(jnp.int32)
    dt, v_max, a_max, k_position, k_velocity = plant_parameters
    robot_radius, safety_margin, human_margin, _, _ = clearance_parameters

    def nominal_control(state: jax.Array) -> jax.Array:
        delta = target - state[:2]
        distance = jnp.linalg.norm(delta)
        speed = jnp.minimum(target_speed, k_position * distance)
        desired_velocity = jnp.where(
            distance < 1.0e-9,
            jnp.zeros(2, dtype=state.dtype),
            delta * (speed / jnp.maximum(distance, 1.0e-18)),
        )
        return jnp.clip(
            k_velocity * (desired_velocity - state[2:]),
            -a_max,
            a_max,
        )

    def plant_step(state: jax.Array, control: jax.Array) -> jax.Array:
        acceleration = jnp.clip(control, -a_max, a_max)
        velocity = state[2:] + acceleration * dt
        speed = jnp.linalg.norm(velocity)
        velocity = jnp.where(
            speed > v_max,
            velocity * (v_max / jnp.maximum(speed, 1.0e-18)),
            velocity,
        )
        return jnp.concatenate((state[:2] + velocity * dt, velocity))

    def advance(
        state: jax.Array,
        offset: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        step_index = first_step + offset
        control = nominal_control(state)
        if human_centers.shape[1]:
            centers = human_centers[step_index]
            next_centers = human_centers[step_index + 1]
            relative = centers - state[None, :2]
            distances = jnp.linalg.norm(relative, axis=1)
            safe_distances = (
                distances
                - human_radii
                - robot_radius
                - safety_margin
                - human_margin
            )
            directions = relative / jnp.maximum(
                distances[:, None],
                1.0e-9,
            )
            human_velocities = (next_centers - centers) / dt
            closing_rates = jnp.sum(
                directions * (human_velocities - state[None, 2:]),
                axis=1,
            )
            cautious = jnp.any(
                (safe_distances < WITNESS_HUMAN_CAUTION_CLEARANCE_M)
                & (closing_rates < 0.15)
            )
        else:
            cautious = jnp.asarray(False)
        following = plant_step(state, control)
        return following, (following, control, cautious)

    _, (states, controls, cautious) = jax.lax.scan(
        advance,
        initial_state,
        jnp.arange(block_steps, dtype=jnp.int32),
    )
    starts = jnp.concatenate((initial_state[None, :], states[:-1]), axis=0)
    step_indices = first_step + jnp.arange(block_steps, dtype=jnp.int32)
    transitions = jnp.column_stack(
        (starts, states, step_indices.astype(initial_state.dtype))
    )
    physical, operational = _swept_clearances_jax(
        transitions,
        human_centers,
        human_swept_centers,
        human_radii,
        blocker_centers,
        blocker_half_extents,
        floor_bounds,
        floor_boundary_starts,
        floor_boundary_ends,
        wall_centers,
        wall_half,
        clearance_parameters,
    )
    return states, controls, physical, operational, cautious


def validate_dynamic_refuge_feasibility(
    *,
    story_id: str,
    start_room_label: str,
    goal_room_label: str,
    corridor_name: str,
    nonroom_escape_interval_m: tuple[float, float],
    initial_state: Sequence[float],
    goal: Sequence[float],
    blockers: Sequence[Stretcher],
    humans: Sequence[Human],
    environment: HospitalEnvironment,
    config: HospitalConfig,
) -> DynamicFeasibilityAudit:
    """Find and exactly replay a sufficient room-refuge construction witness.

    The deterministic search enumerates every statically accessible room whose
    doorway lies in the story's blocked corridor segment.  Rooms are ordered by
    entry-path length, not by a controller value or benchmark outcome.
    """

    state = np.asarray(initial_state, dtype=float)
    goal_point = np.asarray(goal, dtype=float)
    if state.shape != (4,) or goal_point.shape != (2,):
        raise ValueError("dynamic feasibility requires state (4,) and goal (2,)")
    rooms = {room.label: room for room in environment.rooms}
    corridors = {corridor.name: corridor for corridor in environment.corridor_rects}
    try:
        start_room = rooms[start_room_label]
        goal_room = rooms[goal_room_label]
        corridor = corridors[corridor_name]
    except KeyError as exc:
        raise RuntimeError(
            f"story {story_id!r} references missing Hospital geometry"
        ) from exc

    convoy_clear_time_s = _convoy_clear_time(blockers, corridor)
    route_key = _route_cache_key(
        story_id,
        start_room,
        goal_room,
        corridor,
        nonroom_escape_interval_m,
        blockers,
        environment,
        config,
    )
    candidates = _ROUTE_CACHE.get(route_key)
    if candidates is None:
        candidates = _build_route_candidates(
            start_room=start_room,
            goal_room=goal_room,
            active_corridor=corridor,
            nonroom_escape_interval_m=nonroom_escape_interval_m,
            blockers=blockers,
            environment=environment,
            config=config,
        )
        _ROUTE_CACHE[route_key] = candidates
    def search(traffic: _TrafficPrediction) -> DynamicFeasibilityAudit:
        failure_reasons: list[str] = []
        for candidate in candidates:
            audits = _replay_candidate_departure_grid(
                candidate=candidate,
                initial_state=state,
                goal=goal_point,
                humans=humans,
                traffic=traffic,
                convoy_clear_time_s=convoy_clear_time_s,
                environment=environment,
                config=config,
            )
            for audit in audits:
                if audit.valid:
                    return replace(
                        audit,
                        blocker_only=not humans,
                        candidate_room_count=len(candidates),
                    )
                if audit.failure_reason:
                    delay = audit.post_convoy_departure_delay_s
                    delay_label = "none" if delay is None else f"{delay:g}"
                    failure_reasons.append(
                        f"{candidate.room.label}/delay-{delay_label}: "
                        f"{audit.failure_reason}"
                    )
                    if audit.failure_reason.startswith(("entry:", "hold:")):
                        break

        return DynamicFeasibilityAudit(
            schema=DYNAMIC_FEASIBILITY_SCHEMA,
            valid=False,
            blocker_only=not humans,
            candidate_room_count=len(candidates),
            witness_room_label=None,
            room_entry_time_s=None,
            room_blockage_started_at_s=None,
            room_entry_lead_margin_s=None,
            convoy_cleared_at_s=float(convoy_clear_time_s),
            room_exit_time_s=None,
            goal_reached_time_s=None,
            post_convoy_departure_delay_s=None,
            minimum_physical_clearance_m=None,
            minimum_operational_clearance_m=None,
            maximum_speed_mps=None,
            maximum_control_component_mps2=None,
            trajectory_sha256=None,
            failure_reason=(
                "; ".join(failure_reasons[:4])
                if failure_reasons
                else "no statically eligible refuge room"
            ),
        )

    return search(
        _precompute_traffic(
            humans,
            blockers,
            environment,
            config,
        )
    )


def _build_route_candidates(
    *,
    start_room: Room,
    goal_room: Room,
    active_corridor: Rect,
    nonroom_escape_interval_m: tuple[float, float],
    blockers: Sequence[Stretcher],
    environment: HospitalEnvironment,
    config: HospitalConfig,
) -> tuple[_RouteCandidate, ...]:
    planner = HospitalGridPlanner(
        environment,
        resolution=config.planner.resolution,
        clearance=config.robot.radius + config.planner.clearance_buffer,
        preferred_clearance=config.planner.preferred_clearance,
        clearance_weight=config.planner.clearance_weight,
    )
    start_access = environment.room_door_path(
        start_room,
        config.robot.radius,
        config.refuge.inside_door_offset,
        config.refuge.outside_door_offset,
    )
    goal_access = environment.room_door_path(
        goal_room,
        config.robot.radius,
        config.refuge.inside_door_offset,
        config.refuge.outside_door_offset,
    )
    longitudinal_index = 0 if active_corridor.width >= active_corridor.height else 1
    lower, upper = nonroom_escape_interval_m
    output: list[tuple[float, _RouteCandidate]] = []
    for room in environment.rooms:
        if room.label in {start_room.label, goal_room.label}:
            continue
        try:
            access = environment.room_door_path(
                room,
                config.robot.radius,
                config.refuge.inside_door_offset,
                config.refuge.outside_door_offset,
            )
        except ValueError:
            continue
        station = float(access[0][longitudinal_index])
        if not lower < station < upper or not active_corridor.contains(
            access[0], margin=1e-9
        ):
            continue
        blockage = _blockage_interval_at_station(station, blockers, config)
        if blockage is None:
            continue
        entry_middle = _static_route(
            start_access[0], access[0], environment, planner, config
        )
        exit_middle = _static_route(
            access[0], goal_access[0], environment, planner, config
        )
        if entry_middle is None or exit_middle is None:
            continue
        entry = _unique_waypoints(
            (
                start_access[2],
                start_access[1],
                start_access[0],
                *entry_middle,
                access[1],
                access[2],
                access[3],
            )
        )
        exit_path = _unique_waypoints(
            (
                access[2],
                access[1],
                access[0],
                *exit_middle,
                goal_access[1],
                goal_access[2],
                goal_access[3],
            )
        )
        length = _polyline_length((start_room.center, *entry))
        output.append(
            (
                length,
                _RouteCandidate(
                    room=room,
                    entry_waypoints=entry,
                    exit_waypoints=exit_path,
                    doorway_station_m=station,
                    blockage_started_at_s=float(blockage[0]),
                ),
            )
        )
    output.sort(key=lambda item: (item[0], item[1].room.label))
    return tuple(item[1] for item in output)


def _static_route(
    start: np.ndarray,
    end: np.ndarray,
    environment: HospitalEnvironment,
    planner: HospitalGridPlanner,
    config: HospitalConfig,
) -> tuple[np.ndarray, ...] | None:
    clearance = config.robot.radius + config.safety.static_margin
    if environment.segment_is_free(start, end, clearance):
        return (np.asarray(end, dtype=float).copy(),)
    try:
        path = planner.plan(start, end)
    except ValueError:
        return None
    if any(
        not environment.segment_is_free(first, second, clearance)
        for first, second in zip(path, path[1:])
    ):
        return None
    return tuple(np.asarray(point, dtype=float).copy() for point in path[1:])


def _replay_candidate_departure_grid(
    *,
    candidate: _RouteCandidate,
    initial_state: np.ndarray,
    goal: np.ndarray,
    humans: Sequence[Human],
    traffic: _TrafficPrediction,
    convoy_clear_time_s: float,
    environment: HospitalEnvironment,
    config: HospitalConfig,
) -> tuple[DynamicFeasibilityAudit, ...]:
    replay = _Replay(state=initial_state.copy())
    digest = hashlib.sha256()
    _hash_state(digest, replay.time_s, replay.state)

    entered = _follow_waypoints(
        replay,
        candidate.entry_waypoints,
        environment,
        config,
        digest,
        traffic,
        deadline_s=candidate.blockage_started_at_s - MINIMUM_ROOM_ENTRY_LEAD_S,
        terminal_room=candidate.room,
    )
    entry_time = replay.time_s if entered else None
    entry_lead = (
        candidate.blockage_started_at_s - replay.time_s if entered else None
    )
    if entered and entry_lead is not None and entry_lead < MINIMUM_ROOM_ENTRY_LEAD_S:
        replay.failed = True
        replay.failure_reason = "insufficient room-entry lead margin"
    if replay.failed and replay.failure_reason is not None:
        replay.failure_reason = f"entry: {replay.failure_reason}"
        return (
            _audit_from_replay(
                replay,
                candidate,
                humans,
                convoy_clear_time_s,
                entry_time,
                entry_lead,
                None,
                None,
                None,
                digest,
                config,
            ),
        )

    base_hold_until = convoy_clear_time_s + POST_CONVOY_HOLD_BUFFER_S
    _hold_in_room(
        replay,
        candidate.room,
        candidate.entry_waypoints[-1],
        base_hold_until,
        environment,
        config,
        digest,
        traffic,
    )
    if replay.failed:
        if replay.failure_reason is not None:
            replay.failure_reason = f"hold: {replay.failure_reason}"
        return (
            _audit_from_replay(
                replay,
                candidate,
                humans,
                convoy_clear_time_s,
                entry_time,
                entry_lead,
                None,
                None,
                None,
                digest,
                config,
            ),
        )

    audits: list[DynamicFeasibilityAudit] = []
    for departure_delay_s in POST_CONVOY_DEPARTURE_DELAYS_S:
        _hold_in_room(
            replay,
            candidate.room,
            candidate.entry_waypoints[-1],
            base_hold_until + departure_delay_s,
            environment,
            config,
            digest,
            traffic,
        )
        if replay.failed:
            if replay.failure_reason is not None:
                replay.failure_reason = f"hold: {replay.failure_reason}"
            audits.append(
                _audit_from_replay(
                    replay,
                    candidate,
                    humans,
                    convoy_clear_time_s,
                    entry_time,
                    entry_lead,
                    None,
                    None,
                    departure_delay_s,
                    digest,
                    config,
                )
            )
            break
        exit_replay = _copy_replay(replay)
        exit_digest = digest.copy()
        exited, reached_goal = _follow_exit_to_goal(
            exit_replay,
            candidate,
            goal,
            environment,
            config,
            exit_digest,
            traffic,
        )
        goal_time = exit_replay.time_s if reached_goal else None
        if exit_replay.failed and exit_replay.failure_reason is not None:
            exit_replay.failure_reason = f"exit: {exit_replay.failure_reason}"
        audit = _audit_from_replay(
            exit_replay,
            candidate,
            humans,
            convoy_clear_time_s,
            entry_time,
            entry_lead,
            exited,
            goal_time,
            departure_delay_s,
            exit_digest,
            config,
        )
        audits.append(audit)
        if audit.valid:
            break
    return tuple(audits)


def _audit_from_replay(
    replay: _Replay,
    candidate: _RouteCandidate,
    humans: Sequence[Human],
    convoy_clear_time_s: float,
    entry_time: float | None,
    entry_lead: float | None,
    exit_time: float | None,
    goal_time: float | None,
    departure_delay_s: float | None,
    digest: "hashlib._Hash",
    config: HospitalConfig,
) -> DynamicFeasibilityAudit:
    valid = bool(
        not replay.failed
        and entry_time is not None
        and entry_lead is not None
        and entry_lead >= MINIMUM_ROOM_ENTRY_LEAD_S - 1e-12
        and exit_time is not None
        and goal_time is not None
        and replay.minimum_physical_clearance_m > 0.0
        and replay.minimum_operational_clearance_m
        >= MINIMUM_OPERATIONAL_CLEARANCE_M
        and replay.maximum_speed_mps <= config.robot.v_max + 1e-9
        and replay.maximum_control_component_mps2
        <= config.robot.a_max + 1e-9
    )
    return DynamicFeasibilityAudit(
        schema=DYNAMIC_FEASIBILITY_SCHEMA,
        valid=valid,
        blocker_only=not humans,
        candidate_room_count=1,
        witness_room_label=candidate.room.label if valid else None,
        room_entry_time_s=entry_time,
        room_blockage_started_at_s=candidate.blockage_started_at_s,
        room_entry_lead_margin_s=entry_lead,
        convoy_cleared_at_s=float(convoy_clear_time_s),
        room_exit_time_s=exit_time,
        goal_reached_time_s=goal_time,
        post_convoy_departure_delay_s=(
            float(departure_delay_s)
            if departure_delay_s is not None
            else None
        ),
        minimum_physical_clearance_m=(
            float(replay.minimum_physical_clearance_m)
            if isfinite(replay.minimum_physical_clearance_m)
            else None
        ),
        minimum_operational_clearance_m=(
            float(replay.minimum_operational_clearance_m)
            if isfinite(replay.minimum_operational_clearance_m)
            else None
        ),
        maximum_speed_mps=float(replay.maximum_speed_mps),
        maximum_control_component_mps2=float(
            replay.maximum_control_component_mps2
        ),
        trajectory_sha256=digest.hexdigest() if valid else None,
        failure_reason=None if valid else replay.failure_reason,
    )


def _copy_replay(replay: _Replay) -> _Replay:
    return _Replay(
        state=replay.state.copy(),
        time_s=replay.time_s,
        step_index=replay.step_index,
        minimum_physical_clearance_m=replay.minimum_physical_clearance_m,
        minimum_operational_clearance_m=replay.minimum_operational_clearance_m,
        maximum_speed_mps=replay.maximum_speed_mps,
        maximum_control_component_mps2=(
            replay.maximum_control_component_mps2
        ),
        failed=replay.failed,
        failure_reason=replay.failure_reason,
    )


def _follow_waypoints(
    replay: _Replay,
    waypoints: Sequence[np.ndarray],
    environment: HospitalEnvironment,
    config: HospitalConfig,
    digest: "hashlib._Hash",
    traffic: _TrafficPrediction,
    *,
    deadline_s: float,
    terminal_room: Room,
) -> bool:
    index = 0
    radius = min(0.42, config.refuge.waypoint_radius)
    limit_s = min(deadline_s, MAXIMUM_WITNESS_TIME_S)
    while replay.time_s < limit_s - 1e-12:
        while index < len(waypoints) - 1 and np.linalg.norm(
            replay.state[:2] - waypoints[index]
        ) <= radius:
            index += 1
        final = index == len(waypoints) - 1
        # A zero target-speed cap means "brake in place" in waypoint_control;
        # it cannot drive a robot from rest to the final interior waypoint.
        # The geometry tracker must remain active through room entry, while
        # the proportional distance term still decelerates it to rest.
        target_speed = config.robot.v_max
        block = _nominal_block_steps(
            replay,
            waypoints[index],
            target_speed,
            int(ceil((limit_s - replay.time_s) / config.dt)),
            traffic,
        )
        if block:
            for block_step in block:
                _advance_replay(replay, block_step, config, digest)
                if replay.failed:
                    return False
                if (
                    final
                    and terminal_room.interior_margin(replay.state[:2])
                    >= config.refuge.terminal_interior_margin
                    and np.linalg.norm(replay.state[2:])
                    <= config.refuge.terminal_speed_max
                ):
                    return True
                if (
                    not final
                    and np.linalg.norm(
                        replay.state[:2] - waypoints[index]
                    )
                    <= radius
                ):
                    break
            continue
        step = _witness_step(
            replay,
            waypoints[index],
            target_speed,
            traffic,
            environment,
            config,
        )
        _advance_replay(replay, step, config, digest)
        if replay.failed:
            return False
        if (
            final
            and terminal_room.interior_margin(replay.state[:2])
            >= config.refuge.terminal_interior_margin
            and np.linalg.norm(replay.state[2:])
            <= config.refuge.terminal_speed_max
        ):
            return True
    if not replay.failed:
        replay.failed = True
        replay.failure_reason = "room terminal set not reached before deadline"
    return False


def _hold_in_room(
    replay: _Replay,
    room: Room,
    target: np.ndarray,
    hold_until_s: float,
    environment: HospitalEnvironment,
    config: HospitalConfig,
    digest: "hashlib._Hash",
    traffic: _TrafficPrediction,
) -> None:
    limit_s = min(hold_until_s, MAXIMUM_WITNESS_TIME_S)
    while replay.time_s < limit_s - 1e-12:
        block = _nominal_block_steps(
            replay,
            target,
            config.robot.v_max,
            int(ceil((limit_s - replay.time_s) / config.dt)),
            traffic,
        )
        if block:
            for block_step in block:
                _advance_replay(replay, block_step, config, digest)
                if replay.failed:
                    return
                if (
                    room.interior_margin(replay.state[:2])
                    < config.robot.radius
                ):
                    replay.failed = True
                    replay.failure_reason = (
                        "witness left refuge before convoy clearance"
                    )
                    return
            continue
        step = _witness_step(
            replay,
            target,
            config.robot.v_max,
            traffic,
            environment,
            config,
        )
        _advance_replay(replay, step, config, digest)
        if replay.failed:
            return
        if room.interior_margin(replay.state[:2]) < config.robot.radius:
            replay.failed = True
            replay.failure_reason = "witness left refuge before convoy clearance"
            return


def _follow_exit_to_goal(
    replay: _Replay,
    candidate: _RouteCandidate,
    goal: np.ndarray,
    environment: HospitalEnvironment,
    config: HospitalConfig,
    digest: "hashlib._Hash",
    traffic: _TrafficPrediction,
) -> tuple[float | None, bool]:
    index = 0
    exit_time: float | None = None
    radius = min(0.42, config.refuge.waypoint_radius)
    while replay.time_s < MAXIMUM_WITNESS_TIME_S - 1e-12:
        if exit_time is None and not candidate.room.contains(replay.state[:2]):
            exit_time = replay.time_s
        if np.linalg.norm(replay.state[:2] - goal) <= GOAL_TOLERANCE_M:
            return exit_time, exit_time is not None
        while index < len(candidate.exit_waypoints) - 1 and np.linalg.norm(
            replay.state[:2] - candidate.exit_waypoints[index]
        ) <= radius:
            index += 1
        block = _nominal_block_steps(
            replay,
            candidate.exit_waypoints[index],
            config.robot.v_max,
            int(
                ceil(
                    (MAXIMUM_WITNESS_TIME_S - replay.time_s) / config.dt
                )
            ),
            traffic,
        )
        if block:
            for block_step in block:
                _advance_replay(replay, block_step, config, digest)
                if replay.failed:
                    return exit_time, False
                if (
                    exit_time is None
                    and not candidate.room.contains(replay.state[:2])
                ):
                    exit_time = replay.time_s
                if (
                    np.linalg.norm(replay.state[:2] - goal)
                    <= GOAL_TOLERANCE_M
                ):
                    return exit_time, exit_time is not None
                if (
                    index < len(candidate.exit_waypoints) - 1
                    and np.linalg.norm(
                        replay.state[:2] - candidate.exit_waypoints[index]
                    )
                    <= radius
                ):
                    break
            continue
        step = _witness_step(
            replay,
            candidate.exit_waypoints[index],
            config.robot.v_max,
            traffic,
            environment,
            config,
        )
        _advance_replay(replay, step, config, digest)
        if replay.failed:
            return exit_time, False
    if not replay.failed:
        replay.failed = True
        replay.failure_reason = "goal not reached within witness horizon"
    return exit_time, False


def _advance_replay(
    replay: _Replay,
    step: _WitnessStep,
    config: HospitalConfig,
    digest: "hashlib._Hash",
) -> None:
    control_value = step.control
    following = step.following_state
    physical = step.physical_clearance_m
    operational = step.operational_clearance_m
    replay.minimum_physical_clearance_m = min(
        replay.minimum_physical_clearance_m, physical
    )
    replay.minimum_operational_clearance_m = min(
        replay.minimum_operational_clearance_m, operational
    )
    replay.maximum_speed_mps = max(
        replay.maximum_speed_mps,
        float(np.linalg.norm(following[2:])),
    )
    replay.maximum_control_component_mps2 = max(
        replay.maximum_control_component_mps2,
        float(np.max(np.abs(control_value))),
    )
    if physical <= 0.0:
        replay.failed = True
        replay.failure_reason = "physical collision along witness"
        return
    if operational < MINIMUM_OPERATIONAL_CLEARANCE_M:
        replay.failed = True
        replay.failure_reason = "insufficient operational clearance along witness"
        return
    replay.state = following
    replay.time_s += config.dt
    replay.step_index += 1
    _hash_state(digest, replay.time_s, replay.state)


def _nominal_block_steps(
    replay: _Replay,
    target: np.ndarray,
    target_speed: float,
    max_steps: int,
    traffic: _TrafficPrediction,
) -> tuple[_WitnessStep, ...]:
    """Return the exact safe prefix before local traffic logic can branch."""

    count = min(WITNESS_NOMINAL_BLOCK_STEPS, max(0, int(max_steps)))
    if count <= 0 or (
        replay.step_index + WITNESS_NOMINAL_BLOCK_STEPS
        >= traffic.human_centers.shape[0]
    ):
        return ()
    request = np.r_[
        replay.state,
        np.asarray(target, dtype=float),
        float(target_speed),
        float(replay.step_index),
    ]
    with enable_x64():
        states, controls, physical, operational, cautious = (
            _nominal_block_jax(
                jnp.asarray(request, dtype=jnp.float64),
                traffic.human_centers_jax,
                traffic.human_swept_centers_jax,
                traffic.human_radii_jax,
                traffic.blocker_centers_jax,
                traffic.blocker_half_extents_jax,
                traffic.floor_bounds_jax,
                traffic.floor_boundary_starts_jax,
                traffic.floor_boundary_ends_jax,
                traffic.wall_centers_jax,
                traffic.wall_half_jax,
                traffic.clearance_parameters_jax,
                traffic.witness_plant_parameters_jax,
                block_steps=WITNESS_NOMINAL_BLOCK_STEPS,
            )
        )
        state_values = np.asarray(states, dtype=float)
        control_values = np.asarray(controls, dtype=float)
        physical_values = np.asarray(physical, dtype=float)
        operational_values = np.asarray(operational, dtype=float)
        caution_values = np.asarray(cautious, dtype=bool)

    output: list[_WitnessStep] = []
    for index in range(count):
        # The first branch-sensitive transition is deliberately left to
        # `_witness_step`, which applies the exact brake/lattice rule.
        if (
            caution_values[index]
            or operational_values[index] < MINIMUM_OPERATIONAL_CLEARANCE_M
        ):
            break
        output.append(
            _WitnessStep(
                control=control_values[index],
                following_state=state_values[index],
                physical_clearance_m=float(physical_values[index]),
                operational_clearance_m=float(operational_values[index]),
            )
        )
    return tuple(output)


def _witness_step(
    replay: _Replay,
    target: np.ndarray,
    target_speed: float,
    traffic: _TrafficPrediction,
    environment: HospitalEnvironment,
    config: HospitalConfig,
) -> _WitnessStep:
    """Fixed local traffic rule used only by the offline witness search.

    The rule is method-independent and published.  It tracks the geometry
    waypoint until nearby traffic is not clearly receding, then evaluates the
    nominal control, braking, and a symmetric bounded acceleration lattice.
    Among transitions preserving a preferred positive clearance it maximizes
    route progress (clearance breaks ties); otherwise it maximizes clearance.
    No blocker identity, room state, policy value, QP, or benchmark method
    enters this calculation.
    """

    nominal = waypoint_control(
        replay.state,
        target,
        config.robot,
        target_speed=target_speed,
    )
    traffic_caution = False
    if (
        traffic.human_centers.shape[1]
        and replay.step_index + 1 < traffic.human_centers.shape[0]
    ):
        centers = traffic.human_centers[replay.step_index]
        next_centers = traffic.human_centers[replay.step_index + 1]
        relative = centers - replay.state[None, :2]
        distances = np.linalg.norm(relative, axis=1)
        safe_distances = (
            distances
            - traffic.human_radii
            - config.robot.radius
            - config.safety.safety_margin
            - config.safety.human_margin
        )
        directions = relative / np.maximum(distances[:, None], 1e-9)
        human_velocities = (next_centers - centers) / config.dt
        closing_rates = np.sum(
            directions * (human_velocities - replay.state[None, 2:]),
            axis=1,
        )
        if np.any(
            (safe_distances < WITNESS_HUMAN_CAUTION_CLEARANCE_M)
            & (closing_rates < 0.15)
        ):
            traffic_caution = True

    if not traffic_caution:
        following = step_double_integrator(
            replay.state,
            nominal,
            config.dt,
            config.robot,
        )
        physical, operational = _swept_clearances(
            replay.state,
            following,
            replay.step_index,
            traffic,
            environment,
            config,
        )
        if operational >= MINIMUM_OPERATIONAL_CLEARANCE_M:
            return _WitnessStep(
                control=np.asarray(nominal, dtype=float),
                following_state=following,
                physical_clearance_m=physical,
                operational_clearance_m=operational,
            )

    acceleration = config.robot.a_max
    candidates = [
        nominal,
        waypoint_control(
            replay.state,
            replay.state[:2],
            config.robot,
            target_speed=0.0,
        ),
        *(
            np.array([x_value, y_value], dtype=float)
            for x_value in (-acceleration, 0.0, acceleration)
            for y_value in (-acceleration, 0.0, acceleration)
        ),
    ]
    candidate_controls = tuple(
        np.asarray(candidate_control, dtype=float)
        for candidate_control in candidates
    )
    candidate_states = tuple(
        step_double_integrator(
            replay.state,
            candidate_control,
            config.dt,
            config.robot,
        )
        for candidate_control in candidate_controls
    )
    candidate_physical, candidate_operational = _swept_clearances_batch(
        np.broadcast_to(replay.state, (len(candidate_states), 4)),
        np.asarray(candidate_states),
        replay.step_index,
        traffic,
        config,
    )
    current_distance = float(np.linalg.norm(target - replay.state[:2]))
    progress_values = np.asarray(
        [
            current_distance
            - float(np.linalg.norm(target - candidate_state[:2]))
            for candidate_state in candidate_states
        ]
    )
    preferred = np.flatnonzero(
        candidate_operational >= WITNESS_PREFERRED_CLEARANCE_M
    )
    if preferred.size:
        best_index = int(preferred[0])
        for index in preferred[1:]:
            index = int(index)
            if (
                progress_values[index] > progress_values[best_index] + 1e-12
                or (
                    abs(
                        progress_values[index]
                        - progress_values[best_index]
                    )
                    <= 1e-12
                    and candidate_operational[index]
                    > candidate_operational[best_index] + 1e-12
                )
            ):
                best_index = index
    else:
        best_index = 0
        for index in range(1, len(candidate_states)):
            if (
                candidate_operational[index]
                > candidate_operational[best_index] + 1e-12
                or (
                    abs(
                        candidate_operational[index]
                        - candidate_operational[best_index]
                    )
                    <= 1e-12
                    and progress_values[index]
                    > progress_values[best_index] + 1e-12
                )
            ):
                best_index = index
    return _WitnessStep(
        control=candidate_controls[best_index],
        following_state=candidate_states[best_index],
        physical_clearance_m=float(candidate_physical[best_index]),
        operational_clearance_m=float(candidate_operational[best_index]),
    )


def _swept_clearances(
    start: np.ndarray,
    end: np.ndarray,
    step_index: int,
    traffic: _TrafficPrediction,
    environment: HospitalEnvironment,
    config: HospitalConfig,
) -> tuple[float, float]:
    del environment
    physical, operational = _swept_clearances_batch(
        np.asarray(start, dtype=float).reshape(1, 4),
        np.asarray(end, dtype=float).reshape(1, 4),
        step_index,
        traffic,
        config,
    )
    return float(physical[0]), float(operational[0])


def _swept_clearances_batch(
    starts: np.ndarray,
    ends: np.ndarray,
    step_index: int,
    traffic: _TrafficPrediction,
    config: HospitalConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate one or more candidate transitions in one synchronized call."""

    if step_index + 1 >= traffic.human_centers.shape[0]:
        count = np.asarray(starts).shape[0]
        invalid = np.full(count, -np.inf, dtype=float)
        return invalid, invalid.copy()
    # The feasibility audit is intentionally float64 even though production
    # PL-CBF rollouts use float32.  ``enable_x64`` is a context-local JAX
    # setting, so this does not change the controller's JIT configuration.
    with enable_x64():
        transition_payload = np.column_stack(
            (
                np.asarray(starts, dtype=float),
                np.asarray(ends, dtype=float),
                np.full(np.asarray(starts).shape[0], float(step_index)),
            )
        )
        physical, operational = _swept_clearances_jax(
            jnp.asarray(transition_payload, dtype=jnp.float64),
            traffic.human_centers_jax,
            traffic.human_swept_centers_jax,
            traffic.human_radii_jax,
            traffic.blocker_centers_jax,
            traffic.blocker_half_extents_jax,
            traffic.floor_bounds_jax,
            traffic.floor_boundary_starts_jax,
            traffic.floor_boundary_ends_jax,
            traffic.wall_centers_jax,
            traffic.wall_half_jax,
            traffic.clearance_parameters_jax,
        )
        return np.asarray(physical, dtype=float), np.asarray(
            operational,
            dtype=float,
        )


def _precompute_traffic(
    humans: Sequence[Human],
    blockers: Sequence[Stretcher],
    environment: HospitalEnvironment,
    config: HospitalConfig,
) -> _TrafficPrediction:
    """Precompute exact plant-step obstacle states once per candidate crowd."""

    step_count = int(ceil(MAXIMUM_WITNESS_TIME_S / config.dt)) + 1
    human_count = len(humans)
    human_centers = np.zeros((step_count, human_count, 2), dtype=float)
    human_swept_centers = np.zeros(
        (
            step_count - 1,
            WITNESS_DYNAMIC_SUBSTEPS + 1,
            human_count,
            2,
        ),
        dtype=float,
    )
    human_radii = np.asarray(
        [human.radius for human in humans],
        dtype=float,
    )
    positions = np.asarray(
        [[human.x, human.y] for human in humans],
        dtype=float,
    ).reshape(-1, 2)
    velocities = np.asarray(
        [[human.vx, human.vy] for human in humans],
        dtype=float,
    ).reshape(-1, 2)
    if human_count:
        with enable_x64():
            centers_jax, swept_centers_jax = (
                _precompute_human_centers_jax(
                    jnp.asarray(positions, dtype=jnp.float64),
                    jnp.asarray(velocities, dtype=jnp.float64),
                    jnp.asarray(human_radii, dtype=jnp.float64),
                    jnp.asarray(
                        environment._floor_bounds,
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        environment._floor_boundary_starts,
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        environment._floor_boundary_ends,
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        environment._wall_centers,
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        environment._wall_half,
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(config.dt, dtype=jnp.float64),
                    step_count=step_count,
                    prediction_chunk_count=max(
                        1,
                        int(ceil(config.dt / 0.05)),
                    ),
                )
            )
            human_centers = np.asarray(centers_jax, dtype=float)
            human_swept_centers = np.asarray(
                swept_centers_jax,
                dtype=float,
            )

    times = np.arange(step_count, dtype=float) * config.dt
    blocker_centers = np.zeros((step_count, len(blockers), 2), dtype=float)
    blocker_half = np.zeros((len(blockers), 2), dtype=float)
    for obstacle_index, blocker in enumerate(blockers):
        if blocker.reflect_at_route_bounds:
            raise ValueError(
                "publication dynamic-feasibility blockers must be nonreflecting"
            )
        coordinate = blocker.coordinate + blocker.speed * times
        if blocker.axis == "x":
            blocker_centers[:, obstacle_index, 0] = coordinate
            blocker_centers[:, obstacle_index, 1] = blocker.lateral
            blocker_half[obstacle_index] = (
                0.5 * blocker.length,
                0.5 * blocker.width,
            )
        else:
            blocker_centers[:, obstacle_index, 0] = blocker.lateral
            blocker_centers[:, obstacle_index, 1] = coordinate
            blocker_half[obstacle_index] = (
                0.5 * blocker.width,
                0.5 * blocker.length,
            )
    with enable_x64():
        return _TrafficPrediction(
            human_centers=human_centers,
            human_swept_centers=human_swept_centers,
            human_radii=human_radii,
            blocker_centers=blocker_centers,
            blocker_half_extents=blocker_half,
            human_centers_jax=jnp.asarray(
                human_centers,
                dtype=jnp.float64,
            ),
            human_swept_centers_jax=jnp.asarray(
                human_swept_centers,
                dtype=jnp.float64,
            ),
            human_radii_jax=jnp.asarray(human_radii, dtype=jnp.float64),
            blocker_centers_jax=jnp.asarray(
                blocker_centers,
                dtype=jnp.float64,
            ),
            blocker_half_extents_jax=jnp.asarray(
                blocker_half,
                dtype=jnp.float64,
            ),
            floor_bounds_jax=jnp.asarray(
                environment._floor_bounds,
                dtype=jnp.float64,
            ),
            floor_boundary_starts_jax=jnp.asarray(
                environment._floor_boundary_starts,
                dtype=jnp.float64,
            ),
            floor_boundary_ends_jax=jnp.asarray(
                environment._floor_boundary_ends,
                dtype=jnp.float64,
            ),
            wall_centers_jax=jnp.asarray(
                environment._wall_centers,
                dtype=jnp.float64,
            ),
            wall_half_jax=jnp.asarray(
                environment._wall_half,
                dtype=jnp.float64,
            ),
            clearance_parameters_jax=jnp.asarray(
                (
                    config.robot.radius,
                    config.safety.safety_margin,
                    config.safety.human_margin,
                    config.safety.stretcher_margin,
                    config.safety.static_margin,
                ),
                dtype=jnp.float64,
            ),
            witness_plant_parameters_jax=jnp.asarray(
                (
                    config.dt,
                    config.robot.v_max,
                    config.robot.a_max,
                    config.robot.k_position,
                    config.robot.k_velocity,
                ),
                dtype=jnp.float64,
            ),
        )


def _route_cache_key(
    story_id: str,
    start_room: Room,
    goal_room: Room,
    corridor: Rect,
    interval: tuple[float, float],
    blockers: Sequence[Stretcher],
    environment: HospitalEnvironment,
    config: HospitalConfig,
) -> tuple[object, ...]:
    """Content key allowing route reuse across regenerated equivalent worlds."""

    rects = tuple(
        (rect.x, rect.y, rect.width, rect.height, rect.name, rect.kind)
        for rect in (
            *environment.floor_rects,
            *environment.wall_rects,
            *environment.corridor_rects,
        )
    )
    rooms = tuple(
        (
            room.label,
            room.rect.x,
            room.rect.y,
            room.rect.width,
            room.rect.height,
            room.door.side,
            room.door.rect.x,
            room.door.rect.y,
            room.door.rect.width,
            room.door.rect.height,
        )
        for room in environment.rooms
    )
    blocker_signature = tuple(
        (
            blocker.coordinate,
            blocker.lateral,
            blocker.speed,
            blocker.axis,
            blocker.length,
            blocker.width,
            blocker.reflect_at_route_bounds,
        )
        for blocker in blockers
    )
    return (
        story_id,
        start_room.label,
        goal_room.label,
        corridor.name,
        interval,
        environment.width,
        environment.height,
        rects,
        rooms,
        blocker_signature,
        config,
    )


def _blockage_interval_at_station(
    station: float,
    blockers: Sequence[Stretcher],
    config: HospitalConfig,
) -> tuple[float, float] | None:
    intervals: list[tuple[float, float]] = []
    for blocker in blockers:
        if abs(blocker.speed) <= 1e-12:
            return None
        footprint = (
            0.5 * blocker.length
            + config.robot.radius
            + config.safety.safety_margin
            + config.safety.stretcher_margin
        )
        roots = (
            (station - footprint - blocker.coordinate) / blocker.speed,
            (station + footprint - blocker.coordinate) / blocker.speed,
        )
        start, end = min(roots), max(roots)
        if end >= 0.0:
            intervals.append((max(0.0, float(start)), float(end)))
    if not intervals:
        return None
    intervals.sort()
    merged_start, merged_end = intervals[0]
    for start, end in intervals[1:]:
        if start > merged_end + 1e-9:
            return None
        merged_end = max(merged_end, end)
    return float(merged_start), float(merged_end)


def _convoy_clear_time(
    blockers: Sequence[Stretcher],
    corridor: Rect,
) -> float:
    clear: list[float] = []
    for blocker in blockers:
        low, high = (
            (corridor.x, corridor.x1)
            if blocker.axis == "x"
            else (corridor.y, corridor.y1)
        )
        if blocker.speed < 0.0:
            clear.append(
                (blocker.coordinate + 0.5 * blocker.length - low)
                / abs(blocker.speed)
            )
        elif blocker.speed > 0.0:
            clear.append(
                (high - blocker.coordinate + 0.5 * blocker.length)
                / blocker.speed
            )
        else:
            return float("inf")
    return float(max(clear, default=float("inf")))


def _unique_waypoints(
    points: Sequence[Sequence[float]],
) -> tuple[np.ndarray, ...]:
    output: list[np.ndarray] = []
    for point in points:
        value = np.asarray(point, dtype=float).copy()
        if not output or np.linalg.norm(value - output[-1]) > 0.08:
            output.append(value)
    return tuple(output)


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    return float(
        sum(
            np.linalg.norm(np.asarray(second) - np.asarray(first))
            for first, second in zip(points, points[1:])
        )
    )


def _hash_state(
    digest: "hashlib._Hash",
    time_s: float,
    state: np.ndarray,
) -> None:
    payload = np.r_[float(time_s), np.asarray(state, dtype="<f8")].astype("<f8")
    digest.update(payload.tobytes())


__all__ = [
    "DYNAMIC_FEASIBILITY_SCHEMA",
    "DynamicFeasibilityAudit",
    "GOAL_TOLERANCE_M",
    "MAXIMUM_WITNESS_TIME_S",
    "MINIMUM_OPERATIONAL_CLEARANCE_M",
    "MINIMUM_ROOM_ENTRY_LEAD_S",
    "POST_CONVOY_HOLD_BUFFER_S",
    "WITNESS_DYNAMIC_SUBSTEPS",
    "validate_dynamic_refuge_feasibility",
]
