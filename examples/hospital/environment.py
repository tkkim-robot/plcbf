"""Static 2-D hospital floor plan and geometry queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, hypot, pi, sin
from typing import Iterable, Literal, Sequence

import numpy as np


DoorSide = Literal["top", "bottom", "left", "right"]


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


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
    _collision_perimeter: np.ndarray = field(init=False, repr=False)
    _clearance_perimeter: np.ndarray = field(init=False, repr=False)

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
        self._collision_perimeter = np.asarray(
            [
                (
                    cos(index * 2.0 * pi / 16.0),
                    sin(index * 2.0 * pi / 16.0),
                )
                for index in range(16)
            ],
            dtype=float,
        )
        self._clearance_perimeter = np.asarray(
            [
                (
                    cos(index * 2.0 * pi / 12.0),
                    sin(index * 2.0 * pi / 12.0),
                )
                for index in range(12)
            ],
            dtype=float,
        )

    def is_on_floor(self, point: Sequence[float]) -> bool:
        return any(rect.contains(point) for rect in self.floor_rects)

    def is_collision(self, point: Sequence[float], radius: float = 0.0) -> bool:
        px, py = float(point[0]), float(point[1])
        if px < 0.0 or px > self.width or py < 0.0 or py > self.height:
            return True

        samples = [(px, py)]
        if radius > 1e-9:
            samples.extend(
                (
                    px + radius * cos(index * 2.0 * pi / 16.0),
                    py + radius * sin(index * 2.0 * pi / 16.0),
                )
                for index in range(16)
            )
        if any(not self.is_on_floor(sample) for sample in samples):
            return True
        return any(wall.contains((px, py), radius) for wall in self.wall_rects)

    def static_clearance(self, point: Sequence[float], radius: float = 0.0) -> float:
        """Conservative signed clearance to floor-union and explicit walls."""

        px, py = float(point[0]), float(point[1])
        sample_points = [(px, py)]
        if radius > 1e-9:
            sample_points.extend(
                (
                    px + radius * cos(index * 2.0 * pi / 12.0),
                    py + radius * sin(index * 2.0 * pi / 12.0),
                )
                for index in range(12)
            )
        samples = np.asarray(sample_points, dtype=float)
        floor_centers = self._floor_bounds[:, :2] + 0.5 * self._floor_bounds[:, 2:]
        floor_half = 0.5 * self._floor_bounds[:, 2:]
        floor_q = np.abs(samples[:, None, :] - floor_centers[None, :, :]) - floor_half[None, :, :]
        floor_signed = (
            np.linalg.norm(np.maximum(floor_q, 0.0), axis=2)
            + np.minimum(np.maximum(floor_q[:, :, 0], floor_q[:, :, 1]), 0.0)
        )
        floor_margin = float(np.min(np.max(-floor_signed, axis=1)))
        if self._wall_bounds.size:
            wall_centers = self._wall_bounds[:, :2] + 0.5 * self._wall_bounds[:, 2:]
            wall_half = 0.5 * self._wall_bounds[:, 2:] + radius
            wall_q = np.abs(np.array([px, py]) - wall_centers) - wall_half
            wall_signed = (
                np.linalg.norm(np.maximum(wall_q, 0.0), axis=1)
                + np.minimum(np.maximum(wall_q[:, 0], wall_q[:, 1]), 0.0)
            )
            wall_margin = float(np.min(wall_signed))
        else:
            wall_margin = float("inf")
        return min(floor_margin, wall_margin)

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
        sampled_radii = np.where(radius_array > 1e-9, radius_array, 0.0)
        outside_bounds = (
            (point_array[:, 0] < 0.0)
            | (point_array[:, 0] > self.width)
            | (point_array[:, 1] < 0.0)
            | (point_array[:, 1] > self.height)
        )

        samples = np.concatenate(
            (
                point_array[:, None, :],
                point_array[:, None, :]
                + sampled_radii[:, None, None]
                * self._collision_perimeter[None, :, :],
            ),
            axis=1,
        )
        floor_x0 = self._floor_bounds[:, 0]
        floor_y0 = self._floor_bounds[:, 1]
        floor_x1 = floor_x0 + self._floor_bounds[:, 2]
        floor_y1 = floor_y0 + self._floor_bounds[:, 3]
        on_a_floor = np.any(
            (samples[:, :, None, 0] >= floor_x0)
            & (samples[:, :, None, 0] <= floor_x1)
            & (samples[:, :, None, 1] >= floor_y0)
            & (samples[:, :, None, 1] <= floor_y1),
            axis=2,
        )
        outside_floor = np.any(~on_a_floor, axis=1)

        if self._wall_bounds.size:
            wall_x0 = self._wall_bounds[:, 0] - radius_array[:, None]
            wall_y0 = self._wall_bounds[:, 1] - radius_array[:, None]
            wall_x1 = (
                self._wall_bounds[:, 0]
                + self._wall_bounds[:, 2]
                + radius_array[:, None]
            )
            wall_y1 = (
                self._wall_bounds[:, 1]
                + self._wall_bounds[:, 3]
                + radius_array[:, None]
            )
            in_wall = np.any(
                (point_array[:, None, 0] >= wall_x0)
                & (point_array[:, None, 0] <= wall_x1)
                & (point_array[:, None, 1] >= wall_y0)
                & (point_array[:, None, 1] <= wall_y1),
                axis=1,
            )
        else:
            in_wall = np.zeros(point_array.shape[0], dtype=bool)
        return outside_bounds | outside_floor | in_wall

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

        perimeter = (
            radius * self._clearance_perimeter
            if radius > 1e-9
            else np.zeros_like(self._clearance_perimeter)
        )
        samples = np.concatenate(
            (
                point_array[:, None, :],
                point_array[:, None, :] + perimeter[None, :, :],
            ),
            axis=1,
        )
        floor_q = (
            np.abs(
                samples[:, :, None, :]
                - self._floor_centers[None, None, :, :]
            )
            - self._floor_half[None, None, :, :]
        )
        floor_signed = (
            np.linalg.norm(np.maximum(floor_q, 0.0), axis=3)
            + np.minimum(
                np.maximum(floor_q[:, :, :, 0], floor_q[:, :, :, 1]),
                0.0,
            )
        )
        floor_margin = np.min(np.max(-floor_signed, axis=2), axis=1)

        if self._wall_bounds.size:
            wall_half = self._wall_half + radius
            wall_q = (
                np.abs(
                    point_array[:, None, :]
                    - self._wall_centers[None, :, :]
                )
                - wall_half[None, :, :]
            )
            wall_signed = (
                np.linalg.norm(np.maximum(wall_q, 0.0), axis=2)
                + np.minimum(
                    np.maximum(wall_q[:, :, 0], wall_q[:, :, 1]),
                    0.0,
                )
            )
            wall_margin = np.min(wall_signed, axis=1)
        else:
            wall_margin = np.full(point_array.shape[0], np.inf)
        return np.minimum(floor_margin, wall_margin)

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
        if not output or all(np.linalg.norm(candidate - item) > 1e-6 for item in output):
            output.append(candidate)
    return output
