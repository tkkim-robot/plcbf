"""Static 2-D hospital floor plan and geometry queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Iterable, Literal, Sequence

import numpy as np


DoorSide = Literal["top", "bottom", "left", "right"]


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def _floor_union_boundary_segments(
    rectangles: Sequence["Rect"],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the true exterior segments of an axis-aligned rectangle union.

    Individual room and corridor rectangles overlap at doors and junctions.
    Treating each rectangle edge as a wall therefore creates fictitious
    barriers at those internal seams.  Splitting every source edge at all
    rectangle coordinates and retaining only pieces whose two sides have
    different union membership gives the exact boundary of this floor plan.
    """

    if not rectangles:
        empty = np.empty((0, 2), dtype=float)
        return empty, empty.copy()

    x_breaks = sorted({value for rect in rectangles for value in (rect.x, rect.x1)})
    y_breaks = sorted({value for rect in rectangles for value in (rect.y, rect.y1)})
    scale = max(
        1.0,
        max(
            max(abs(rect.x), abs(rect.x1), abs(rect.y), abs(rect.y1))
            for rect in rectangles
        ),
    )
    probe = 1.0e-8 * scale

    def inside(x: float, y: float) -> bool:
        return any(rect.contains((x, y)) for rect in rectangles)

    # Store axis, fixed coordinate, and the increasing interval.  Rounding is
    # only a deterministic de-duplication key; returned coordinates remain the
    # original floor-plan values.
    raw: dict[tuple[str, float, float, float], tuple[str, float, float, float]] = {}

    def retain(axis: str, fixed: float, lower: float, upper: float) -> None:
        if upper - lower <= 1.0e-12:
            return
        midpoint = 0.5 * (lower + upper)
        if axis == "v":
            first = inside(fixed - probe, midpoint)
            second = inside(fixed + probe, midpoint)
        else:
            first = inside(midpoint, fixed - probe)
            second = inside(midpoint, fixed + probe)
        if first == second:
            return
        key = (axis, round(fixed, 10), round(lower, 10), round(upper, 10))
        raw[key] = (axis, float(fixed), float(lower), float(upper))

    for rect in rectangles:
        vertical_cuts = [
            value for value in y_breaks if rect.y <= value <= rect.y1
        ]
        horizontal_cuts = [
            value for value in x_breaks if rect.x <= value <= rect.x1
        ]
        for fixed in (rect.x, rect.x1):
            for lower, upper in zip(
                vertical_cuts, vertical_cuts[1:], strict=False
            ):
                retain("v", fixed, lower, upper)
        for fixed in (rect.y, rect.y1):
            for lower, upper in zip(
                horizontal_cuts, horizontal_cuts[1:], strict=False
            ):
                retain("h", fixed, lower, upper)

    grouped: dict[tuple[str, float], list[tuple[float, float]]] = {}
    coordinates: dict[tuple[str, float], float] = {}
    for axis, fixed, lower, upper in raw.values():
        key = (axis, round(fixed, 10))
        coordinates[key] = fixed
        grouped.setdefault(key, []).append((lower, upper))

    merged: list[tuple[str, float, float, float]] = []
    for key in sorted(grouped):
        intervals = sorted(grouped[key])
        lower, upper = intervals[0]
        for following_lower, following_upper in intervals[1:]:
            if following_lower <= upper + 1.0e-10:
                upper = max(upper, following_upper)
            else:
                merged.append((key[0], coordinates[key], lower, upper))
                lower, upper = following_lower, following_upper
        merged.append((key[0], coordinates[key], lower, upper))

    starts: list[tuple[float, float]] = []
    ends: list[tuple[float, float]] = []
    for axis, fixed, lower, upper in merged:
        if axis == "v":
            starts.append((fixed, lower))
            ends.append((fixed, upper))
        else:
            starts.append((lower, fixed))
            ends.append((upper, fixed))
    return np.asarray(starts, dtype=float), np.asarray(ends, dtype=float)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float
    name: str = ""
    kind: str = ""

    @property
    def x1(self) -> float:
        return self.x + self.width

    @property
    def y1(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> np.ndarray:
        return np.array(
            [self.x + 0.5 * self.width, self.y + 0.5 * self.height],
            dtype=float,
        )

    def contains(self, point: Sequence[float], margin: float = 0.0) -> bool:
        px, py = float(point[0]), float(point[1])
        return (
            self.x - margin <= px <= self.x1 + margin
            and self.y - margin <= py <= self.y1 + margin
        )

    def signed_distance(self, point: Sequence[float], padding: float = 0.0) -> float:
        """Signed distance to the rectangle (positive outside)."""

        px, py = float(point[0]), float(point[1])
        qx = abs(px - (self.x + 0.5 * self.width)) - (
            0.5 * self.width + padding
        )
        qy = abs(py - (self.y + 0.5 * self.height)) - (
            0.5 * self.height + padding
        )
        outside = hypot(max(qx, 0.0), max(qy, 0.0))
        inside = min(max(qx, qy), 0.0)
        return outside + inside


@dataclass(frozen=True)
class Door:
    rect: Rect
    side: DoorSide

    @property
    def center(self) -> np.ndarray:
        return self.rect.center


@dataclass(frozen=True)
class Room:
    rect: Rect
    label: str
    door: Door

    @property
    def center(self) -> np.ndarray:
        return self.rect.center

    def contains(self, point: Sequence[float], margin: float = 0.0) -> bool:
        return self.rect.contains(point, margin)

    def interior_margin(self, point: Sequence[float]) -> float:
        px, py = float(point[0]), float(point[1])
        return min(
            px - self.rect.x,
            self.rect.x1 - px,
            py - self.rect.y,
            self.rect.y1 - py,
        )


@dataclass
class HospitalEnvironment:
    width: float
    height: float
    floor_rects: list[Rect] = field(default_factory=list)
    corridor_rects: list[Rect] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    wall_rects: list[Rect] = field(default_factory=list)
    door_rects: list[Rect] = field(default_factory=list)
    _floor_bounds: np.ndarray = field(init=False, repr=False)
    _wall_bounds: np.ndarray = field(init=False, repr=False)
    _floor_centers: np.ndarray = field(init=False, repr=False)
    _floor_half: np.ndarray = field(init=False, repr=False)
    _wall_centers: np.ndarray = field(init=False, repr=False)
    _wall_half: np.ndarray = field(init=False, repr=False)
    _floor_boundary_starts: np.ndarray = field(init=False, repr=False)
    _floor_boundary_ends: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._floor_bounds = np.asarray(
            [[rect.x, rect.y, rect.width, rect.height] for rect in self.floor_rects],
            dtype=float,
        ).reshape(-1, 4)
        self._wall_bounds = np.asarray(
            [[rect.x, rect.y, rect.width, rect.height] for rect in self.wall_rects],
            dtype=float,
        ).reshape(-1, 4)
        self._floor_centers = (
            self._floor_bounds[:, :2]
            + 0.5 * self._floor_bounds[:, 2:]
        )
        self._floor_half = 0.5 * self._floor_bounds[:, 2:]
        self._wall_centers = (
            self._wall_bounds[:, :2]
            + 0.5 * self._wall_bounds[:, 2:]
        )
        self._wall_half = 0.5 * self._wall_bounds[:, 2:]
        (
            self._floor_boundary_starts,
            self._floor_boundary_ends,
        ) = _floor_union_boundary_segments(self.floor_rects)

    def is_on_floor(self, point: Sequence[float]) -> bool:
        return any(rect.contains(point) for rect in self.floor_rects)

    def is_collision(self, point: Sequence[float], radius: float = 0.0) -> bool:
        return self.static_clearance(point, radius) <= 0.0

    def static_clearance(self, point: Sequence[float], radius: float = 0.0) -> float:
        """Exact signed disc clearance to the floor union and explicit walls."""

        return float(
            self._static_clearances_with_radii(
                np.asarray(point, dtype=float).reshape(1, 2),
                np.asarray([radius], dtype=float),
            )[0]
        )

    def _points_on_floor(self, points: np.ndarray) -> np.ndarray:
        if not self._floor_bounds.size:
            return np.zeros(points.shape[0], dtype=bool)
        lower = self._floor_bounds[:, :2]
        upper = lower + self._floor_bounds[:, 2:]
        return np.any(
            np.all(
                (points[:, None, :] >= lower[None, :, :])
                & (points[:, None, :] <= upper[None, :, :]),
                axis=2,
            ),
            axis=1,
        )

    def _floor_union_signed_clearances(self, points: np.ndarray) -> np.ndarray:
        if not self._floor_boundary_starts.size:
            return np.full(points.shape[0], -np.inf)
        segments = self._floor_boundary_ends - self._floor_boundary_starts
        lengths_squared = np.sum(segments * segments, axis=1)
        relative = points[:, None, :] - self._floor_boundary_starts[None, :, :]
        fractions = np.clip(
            np.sum(relative * segments[None, :, :], axis=2)
            / np.maximum(lengths_squared[None, :], 1.0e-18),
            0.0,
            1.0,
        )
        closest = (
            self._floor_boundary_starts[None, :, :]
            + fractions[:, :, None] * segments[None, :, :]
        )
        distances = np.min(
            np.linalg.norm(points[:, None, :] - closest, axis=2), axis=1
        )
        return np.where(self._points_on_floor(points), distances, -distances)

    def _static_clearances_with_radii(
        self,
        points: np.ndarray,
        radii: np.ndarray,
    ) -> np.ndarray:
        floor_margin = self._floor_union_signed_clearances(points) - radii
        if self._wall_bounds.size:
            q = (
                np.abs(points[:, None, :] - self._wall_centers[None, :, :])
                - self._wall_half[None, :, :]
            )
            wall_signed = (
                np.linalg.norm(np.maximum(q, 0.0), axis=2)
                + np.minimum(np.maximum(q[:, :, 0], q[:, :, 1]), 0.0)
            )
            wall_margin = np.min(wall_signed, axis=1) - radii
        else:
            wall_margin = np.full(points.shape[0], np.inf)
        return np.minimum(floor_margin, wall_margin)

    def collisions(
        self,
        points: Sequence[Sequence[float]] | np.ndarray,
        radii: float | Sequence[float] | np.ndarray = 0.0,
    ) -> np.ndarray:
        """Vectorized equivalent of :meth:`is_collision`.

        The center, 16 perimeter samples, inclusive floor membership, and
        radius-padded wall tests are intentionally identical to the scalar
        geometry query.  This is used only to amortize Python overhead while
        predicting a crowd of independent humans.
        """

        point_array = np.asarray(points, dtype=float)
        if point_array.ndim == 1:
            point_array = point_array.reshape(1, 2)
        if point_array.ndim != 2 or point_array.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        radius_array = np.broadcast_to(
            np.asarray(radii, dtype=float),
            (point_array.shape[0],),
        )
        return self._static_clearances_with_radii(
            point_array,
            radius_array,
        ) <= 0.0

    def static_clearances(
        self,
        points: Sequence[Sequence[float]] | np.ndarray,
        radius: float = 0.0,
    ) -> np.ndarray:
        """Vectorized equivalent of :meth:`static_clearance`."""

        point_array = np.asarray(points, dtype=float)
        if point_array.ndim == 1:
            point_array = point_array.reshape(1, 2)
        if point_array.ndim != 2 or point_array.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")

        return self._static_clearances_with_radii(
            point_array,
            np.full(point_array.shape[0], float(radius)),
        )

    def segments_are_free(
        self,
        starts: Sequence[Sequence[float]] | np.ndarray,
        ends: Sequence[Sequence[float]] | np.ndarray,
        radius: float,
        step: float = 0.45,
    ) -> np.ndarray:
        """Evaluate the exact scalar segment samples in one geometry batch."""

        start_array = np.asarray(starts, dtype=float)
        end_array = np.asarray(ends, dtype=float)
        if start_array.ndim == 1:
            start_array = start_array.reshape(1, 2)
        if end_array.ndim == 1:
            end_array = end_array.reshape(1, 2)
        if (
            start_array.shape != end_array.shape
            or start_array.ndim != 2
            or start_array.shape[1] != 2
        ):
            raise ValueError("starts and ends must both have shape (N, 2)")

        segment_ids: list[int] = []
        samples: list[np.ndarray] = []
        for index, (start, end) in enumerate(
            zip(start_array, end_array, strict=True)
        ):
            distance = float(np.linalg.norm(end - start))
            count = max(2, int(np.ceil(distance / step)))
            for sample_index in range(count + 1):
                segment_ids.append(index)
                samples.append(
                    start
                    + (end - start) * (sample_index / count)
                )
        collision = self.collisions(np.asarray(samples), radius)
        free = np.ones(start_array.shape[0], dtype=bool)
        np.logical_and.at(
            free,
            np.asarray(segment_ids, dtype=int),
            ~collision,
        )
        return free

    def room_containing(self, point: Sequence[float]) -> Room | None:
        return next((room for room in self.rooms if room.contains(point)), None)

    def nearest_corridor(self, point: Sequence[float]) -> Rect:
        px, py = float(point[0]), float(point[1])

        def distance(rect: Rect) -> float:
            cx = _clamp(px, rect.x, rect.x1)
            cy = _clamp(py, rect.y, rect.y1)
            return hypot(px - cx, py - cy)

        return min(self.corridor_rects, key=distance)

    def room_door_path(
        self,
        room: Room,
        robot_radius: float,
        inside_offset: float = 2.3,
        outside_offset: float = 1.75,
        *,
        segment_clearance_buffer: float = 0.06,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a collision-free outside/door/inside room path.

        This mirrors the hospital playground's ``roomDoorPath`` geometry.  A
        centered doorway route is preferred, with the same two small tangent
        offsets tried when the centered line is unavailable.  The final point
        is a diagnostic room-center projection and is not part of the backup
        path.
        """

        door = room.door.center
        if room.door.side == "bottom":
            outward = np.array([0.0, -1.0])
        elif room.door.side == "top":
            outward = np.array([0.0, 1.0])
        elif room.door.side == "left":
            outward = np.array([-1.0, 0.0])
        else:
            outward = np.array([1.0, 0.0])

        horizontal_door = room.door.rect.width >= room.door.rect.height
        tangent = (
            np.array([1.0, 0.0])
            if horizontal_door
            else np.array([0.0, 1.0])
        )
        span = (
            room.door.rect.width
            if horizontal_door
            else room.door.rect.height
        )
        point_radius = robot_radius + 0.08
        segment_radius = robot_radius + segment_clearance_buffer
        for tangent_offset in (0.0, -0.28 * span, 0.28 * span):
            candidate_door = door + tangent * tangent_offset
            outside = candidate_door + outward * (
                robot_radius + outside_offset
            )
            inside = candidate_door - outward * (
                robot_radius + inside_offset
            )
            if self.is_collision(inside, point_radius) or self.is_collision(
                outside, point_radius
            ):
                continue
            if not self.segment_is_free(
                outside,
                candidate_door,
                segment_radius,
                step=0.55,
            ) or not self.segment_is_free(
                candidate_door,
                inside,
                segment_radius,
                step=0.55,
            ):
                continue

            terminal_center = room.center.copy()
            # Preserve a direct line through the selected door for narrow
            # rooms, as in the playground diagnostic rendering.
            if horizontal_door:
                terminal_center[0] = candidate_door[0]
            else:
                terminal_center[1] = candidate_door[1]
            return (
                outside,
                candidate_door.copy(),
                inside,
                terminal_center,
            )
        raise ValueError(f"no collision-free doorway route for {room.label}")

    def segment_is_free(
        self,
        start: Sequence[float],
        end: Sequence[float],
        radius: float,
        step: float = 0.45,
    ) -> bool:
        start_arr = np.asarray(start, dtype=float)
        end_arr = np.asarray(end, dtype=float)
        distance = float(np.linalg.norm(end_arr - start_arr))
        count = max(2, int(np.ceil(distance / step)))
        return all(
            not self.is_collision(
                start_arr + (end_arr - start_arr) * (index / count), radius
            )
            for index in range(count + 1)
        )


def _wall_with_gap(
    walls: list[Rect],
    x0: float,
    y0: float,
    length: float,
    horizontal: bool,
    gap_center: float | None,
    gap_width: float,
    wall_thickness: float,
) -> None:
    if gap_center is None:
        if horizontal:
            walls.append(Rect(x0, y0 - wall_thickness / 2, length, wall_thickness))
        else:
            walls.append(Rect(x0 - wall_thickness / 2, y0, wall_thickness, length))
        return

    start = x0 if horizontal else y0
    end = start + length
    gap0 = max(start, gap_center - gap_width / 2)
    gap1 = min(end, gap_center + gap_width / 2)
    if gap0 - start > 0.05:
        if horizontal:
            walls.append(
                Rect(start, y0 - wall_thickness / 2, gap0 - start, wall_thickness)
            )
        else:
            walls.append(
                Rect(x0 - wall_thickness / 2, start, wall_thickness, gap0 - start)
            )
    if end - gap1 > 0.05:
        if horizontal:
            walls.append(
                Rect(gap1, y0 - wall_thickness / 2, end - gap1, wall_thickness)
            )
        else:
            walls.append(
                Rect(x0 - wall_thickness / 2, gap1, wall_thickness, end - gap1)
            )


def build_hospital_environment() -> HospitalEnvironment:
    """Build the same 140 x 95 traversable topology as the web playground."""

    width, height = 140.0, 95.0
    wall_t = 0.48
    floors: list[Rect] = []
    corridors: list[Rect] = []
    rooms: list[Room] = []
    walls: list[Rect] = []
    doors: list[Rect] = []

    corridor_specs = [
        Rect(4, 43, 132, 9, "Main corridor", "corridor"),
        Rect(14, 72, 112, 8, "North corridor", "corridor"),
        Rect(14, 15, 112, 9, "South corridor", "corridor"),
        Rect(20, 8, 8, 80, "West vertical hall", "corridor"),
        Rect(62, 8, 8, 80, "Center vertical hall", "corridor"),
        Rect(104, 8, 8, 80, "East vertical hall", "corridor"),
        Rect(6, 60, 32, 8, "Waiting spur", "corridor"),
        Rect(102, 30, 30, 8, "Emergency spur", "corridor"),
    ]
    corridors.extend(corridor_specs)
    floors.extend(corridor_specs)

    walls.extend(
        [
            Rect(0, 0, width, 1.2, kind="outer_wall"),
            Rect(0, height - 1.2, width, 1.2, kind="outer_wall"),
            Rect(0, 0, 1.2, height, kind="outer_wall"),
            Rect(width - 1.2, 0, 1.2, height, kind="outer_wall"),
        ]
    )

    def add_room(
        x: float,
        y: float,
        room_width: float,
        room_height: float,
        side: DoorSide,
        label: str,
        door_width: float = 8.0,
    ) -> None:
        room_rect = Rect(x, y, room_width, room_height, label, "room")
        center = (
            x + room_width / 2
            if side in {"top", "bottom"}
            else y + room_height / 2
        )
        _wall_with_gap(
            walls,
            x,
            y,
            room_width,
            True,
            center if side == "bottom" else None,
            door_width,
            wall_t,
        )
        _wall_with_gap(
            walls,
            x,
            y + room_height,
            room_width,
            True,
            center if side == "top" else None,
            door_width,
            wall_t,
        )
        _wall_with_gap(
            walls,
            x,
            y,
            room_height,
            False,
            center if side == "left" else None,
            door_width,
            wall_t,
        )
        _wall_with_gap(
            walls,
            x + room_width,
            y,
            room_height,
            False,
            center if side == "right" else None,
            door_width,
            wall_t,
        )
        if side == "bottom":
            door_rect = Rect(
                center - door_width / 2,
                y - wall_t / 2,
                door_width,
                wall_t,
                f"{label} door",
                "door",
            )
        elif side == "top":
            door_rect = Rect(
                center - door_width / 2,
                y + room_height - wall_t / 2,
                door_width,
                wall_t,
                f"{label} door",
                "door",
            )
        elif side == "left":
            door_rect = Rect(
                x - wall_t / 2,
                center - door_width / 2,
                wall_t,
                door_width,
                f"{label} door",
                "door",
            )
        else:
            door_rect = Rect(
                x + room_width - wall_t / 2,
                center - door_width / 2,
                wall_t,
                door_width,
                f"{label} door",
                "door",
            )
        doors.append(door_rect)
        floors.extend([room_rect, door_rect])
        rooms.append(Room(room_rect, label, Door(door_rect, side)))

    for x in (30, 44, 72, 86):
        add_room(x, 52, 12, 16, "bottom", f"Ward {x}")
        add_room(x, 27, 12, 16, "top", f"Exam {x}")
    for x in (8, 34, 50, 76, 92, 118):
        room_width = 14 if x in (8, 118) else 12
        add_room(x, 80, room_width, 12, "bottom", f"North Patient {x}")
        add_room(x, 3, room_width, 12, "top", f"South Patient {x}")

    add_room(4, 37, 16, 6, "top", "Pharmacy", 8)
    add_room(56, 52, 6, 12, "bottom", "Nurse", 6)
    floors.append(Rect(56, 50.4, 6, 1.6, "Nurse apron", "room"))
    add_room(112, 52, 8, 12, "bottom", "Supply", 5)
    floors.append(Rect(121.5, 50.4, 6, 6, "Waiting apron", "room"))
    add_room(120, 52, 15, 16, "bottom", "Waiting", 12)
    add_room(118, 24, 16, 6, "bottom", "Trauma", 8)

    return HospitalEnvironment(
        width=width,
        height=height,
        floor_rects=floors,
        corridor_rects=corridors,
        rooms=rooms,
        wall_rects=walls,
        door_rects=doors,
    )


def unique_points(points: Iterable[Sequence[float]]) -> list[np.ndarray]:
    """Return stable unique points; useful when composing doorway paths."""

    output: list[np.ndarray] = []
    for point in points:
        candidate = np.asarray(point, dtype=float)
        if not output or all(
            np.linalg.norm(candidate - item) > 1e-6 for item in output
        ):
            output.append(candidate)
    return output
