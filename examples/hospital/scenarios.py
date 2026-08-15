"""Fixed-story publication protocol for the Hospital refuge case study.

The scientific variable in this protocol is a mandatory, full-width stretcher
encounter.  Five immutable stories vary the room from which the ego departs,
its nominal destination, and the direction of the oncoming convoy.  Twenty
traffic seeds per story randomize *only* circular human traffic.

Each story also names a diagnostic refuge witness.  That label is used only by
method-independent scenario validation to prove that a lateral escape exists.
It is deliberately excluded from simulator/controller construction, policy
ranking, success classification, and benchmark case metadata.  A controller
may enter any safe room or use any other behavior allowed by its native method.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
import hashlib
import json
from math import isfinite
from typing import Iterable, Mapping, Sequence

import numpy as np

from .config import DEFAULT_CONFIG, HospitalConfig
from .environment import (
    HospitalEnvironment,
    Rect,
    Room,
    build_hospital_environment,
)
from .feasibility import (
    DYNAMIC_FEASIBILITY_SCHEMA,
    DynamicFeasibilityAudit,
    GOAL_TOLERANCE_M,
    MAXIMUM_WITNESS_TIME_S,
    MINIMUM_OPERATIONAL_CLEARANCE_M,
    MINIMUM_ROOM_ENTRY_LEAD_S,
    POST_CONVOY_DEPARTURE_DELAYS_S,
    POST_CONVOY_HOLD_BUFFER_S,
    WITNESS_DYNAMIC_SUBSTEPS,
    WITNESS_HUMAN_CAUTION_CLEARANCE_M,
    WITNESS_NOMINAL_BLOCK_STEPS,
    WITNESS_PREFERRED_CLEARANCE_M,
    validate_dynamic_refuge_feasibility,
)
from .obstacles import DynamicObstacle, Human, Stretcher, stretcher_route
from .scenario_generation import (
    DEFAULT_HUMAN_COUNT,
    DEFAULT_PROTECTED_CLEARANCE,
    HUMAN_RADIUS,
    HUMAN_SPEED_RANGE,
    HospitalCrowdMetadata,
    INITIAL_PAIRWISE_CLEARANCE,
    MAX_PLACEMENT_ATTEMPTS_PER_OBSTACLE,
    STATIC_PLACEMENT_MARGIN,
    TRAFFIC_SPEED_CAP,
    generate_hospital_crowd,
    obstacle_pair_clearance,
)


HOSPITAL_STORY_PROTOCOL_VERSION = "hospital_fixed_refuge_v4"
HOSPITAL_GEOMETRY_VERSION = "hospital_floorplan_140x95_v1"
HOSPITAL_STORY_SEED_NAMESPACE = 0x48535031
DEFAULT_HOSPITAL_TRAFFIC_SEEDS = tuple(range(20))
PUBLICATION_HUMAN_COUNT = DEFAULT_HUMAN_COUNT
PUBLICATION_SENSING_RANGE_M = 24.0
BLOCKER_LENGTH_M = 5.4
BLOCKER_WIDTH_M = 7.1
# The playground UI caps obstacle motion at 3 m/s.  The publication convoy
# uses that ceiling while remaining strictly faster than the 2.85 m/s ego, so
# pure corridor retreat is still swept and a lateral room route is necessary.
BLOCKER_SPEED_MPS = 3.0
MAX_DYNAMIC_FEASIBILITY_GENERATION_ATTEMPTS = 256


@dataclass(frozen=True)
class HospitalStoryTemplate:
    """Immutable, method-independent definition of one narrative encounter."""

    story_id: str
    start_room: str
    goal_room: str
    diagnostic_refuge_room: str
    corridor_name: str
    travel_direction: int
    convoy_coordinates_m: tuple[float, ...]
    convoy_speed_mps: float
    necessity_witness_station_m: float
    necessity_witness_time_s: float
    nonroom_escape_interval_m: tuple[float, float]
    human_corridor_names: tuple[str, ...]
    narrative: str

    def __post_init__(self) -> None:
        if not self.story_id:
            raise ValueError("story_id must not be empty")
        if self.travel_direction not in (-1, 1):
            raise ValueError("travel_direction must be -1 or 1")
        if len(self.convoy_coordinates_m) not in (2, 3):
            raise ValueError("each story must contain two or three blockers")
        if not all(isfinite(value) for value in self.convoy_coordinates_m):
            raise ValueError("convoy coordinates must be finite")
        if not isfinite(self.convoy_speed_mps) or self.convoy_speed_mps == 0.0:
            raise ValueError("convoy speed must be finite and nonzero")
        if self.travel_direction * self.convoy_speed_mps >= 0.0:
            raise ValueError("convoy must approach opposite nominal travel")
        if (
            not isfinite(self.necessity_witness_station_m)
            or not isfinite(self.necessity_witness_time_s)
            or self.necessity_witness_time_s < 0.0
        ):
            raise ValueError("necessity witness station/time must be finite")
        if (
            len(self.nonroom_escape_interval_m) != 2
            or not all(
                isfinite(value) for value in self.nonroom_escape_interval_m
            )
            or self.nonroom_escape_interval_m[0]
            >= self.nonroom_escape_interval_m[1]
            or not (
                self.nonroom_escape_interval_m[0]
                < self.necessity_witness_station_m
                < self.nonroom_escape_interval_m[1]
            )
        ):
            raise ValueError("necessity witness must lie between escape stations")
        if len(
            {self.start_room, self.goal_room, self.diagnostic_refuge_room}
        ) != 3:
            raise ValueError("start, goal, and diagnostic refuge must differ")
        if not self.human_corridor_names:
            raise ValueError("at least one human-traffic corridor is required")

    @property
    def blocker_count(self) -> int:
        return len(self.convoy_coordinates_m)


_MAIN_HUMAN_REGION = (
    "Main corridor",
    "West vertical hall",
    "Center vertical hall",
    "East vertical hall",
    "Waiting spur",
    "Emergency spur",
)
_NORTH_HUMAN_REGION = (
    "North corridor",
    "West vertical hall",
    "Center vertical hall",
    "East vertical hall",
)
_SOUTH_HUMAN_REGION = (
    "South corridor",
    "West vertical hall",
    "Center vertical hall",
    "East vertical hall",
)


HOSPITAL_STORIES: tuple[HospitalStoryTemplate, ...] = (
    HospitalStoryTemplate(
        story_id="main_eastbound",
        start_room="Ward 30",
        goal_room="Waiting",
        diagnostic_refuge_room="Ward 44",
        corridor_name="Main corridor",
        travel_direction=1,
        convoy_coordinates_m=(96.0, 102.4, 108.8),
        convoy_speed_mps=-BLOCKER_SPEED_MPS,
        necessity_witness_station_m=50.0,
        necessity_witness_time_s=10.8,
        nonroom_escape_interval_m=(28.0, 62.0),
        human_corridor_names=_MAIN_HUMAN_REGION,
        narrative=(
            "Depart Ward 30 toward Waiting; Ward 44 is the geometric room "
            "witness before the center-hall escape."
        ),
    ),
    HospitalStoryTemplate(
        story_id="main_westbound",
        start_room="Ward 86",
        goal_room="Pharmacy",
        diagnostic_refuge_room="Exam 72",
        corridor_name="Main corridor",
        travel_direction=-1,
        convoy_coordinates_m=(44.0, 37.6),
        convoy_speed_mps=BLOCKER_SPEED_MPS,
        necessity_witness_station_m=78.0,
        necessity_witness_time_s=6.7,
        nonroom_escape_interval_m=(70.0, 104.0),
        human_corridor_names=_MAIN_HUMAN_REGION,
        narrative=(
            "Depart Ward 86 toward Pharmacy while an eastbound emergency "
            "convoy closes the main corridor."
        ),
    ),
    HospitalStoryTemplate(
        story_id="north_eastbound",
        start_room="North Patient 8",
        goal_room="North Patient 118",
        diagnostic_refuge_room="North Patient 34",
        corridor_name="North corridor",
        travel_direction=1,
        # Shift the complete immutable convoy and witness together by 6 s.
        # This preserves the relative no-room necessity encounter while giving
        # the actual bounded-DI start-to-room construction witness >1 s lead.
        convoy_coordinates_m=(107.0, 113.4, 119.8),
        convoy_speed_mps=-BLOCKER_SPEED_MPS,
        necessity_witness_station_m=40.0,
        necessity_witness_time_s=17.9,
        nonroom_escape_interval_m=(28.0, 62.0),
        human_corridor_names=_NORTH_HUMAN_REGION,
        narrative=(
            "Cross the north ward eastbound while a westbound stretcher "
            "convoy occupies the complete corridor cross-section."
        ),
    ),
    HospitalStoryTemplate(
        story_id="north_westbound",
        start_room="North Patient 92",
        goal_room="North Patient 8",
        diagnostic_refuge_room="North Patient 76",
        corridor_name="North corridor",
        travel_direction=-1,
        # The westbound story needs 7.5 s more actual start-to-room travel.
        # Positions and witness time move together, so encounter geometry and
        # the stop/continue/max-retreat necessity proof remain unchanged.
        convoy_coordinates_m=(24.5, 18.1),
        convoy_speed_mps=BLOCKER_SPEED_MPS,
        necessity_witness_station_m=82.0,
        necessity_witness_time_s=14.6,
        nonroom_escape_interval_m=(70.0, 104.0),
        human_corridor_names=_NORTH_HUMAN_REGION,
        narrative=(
            "Cross the north ward westbound while an eastbound stretcher "
            "convoy occupies the complete corridor cross-section."
        ),
    ),
    HospitalStoryTemplate(
        story_id="south_eastbound",
        start_room="South Patient 8",
        goal_room="South Patient 118",
        diagnostic_refuge_room="South Patient 34",
        corridor_name="South corridor",
        travel_direction=1,
        convoy_coordinates_m=(107.0, 113.4, 119.8),
        convoy_speed_mps=-BLOCKER_SPEED_MPS,
        necessity_witness_station_m=40.0,
        necessity_witness_time_s=17.9,
        nonroom_escape_interval_m=(28.0, 62.0),
        human_corridor_names=_SOUTH_HUMAN_REGION,
        narrative=(
            "Cross the south ward eastbound while a westbound stretcher "
            "convoy occupies the complete corridor cross-section."
        ),
    ),
)

HOSPITAL_STORY_IDS = tuple(story.story_id for story in HOSPITAL_STORIES)
_STORY_BY_ID = {story.story_id: story for story in HOSPITAL_STORIES}
HOSPITAL_ACCEPTED_ATTEMPT_MANIFEST_SCHEMA = (
    "hospital_accepted_generation_attempts_v1"
)
# Frozen after exact, controller-independent replay of all 100 publication
# worlds. Rows follow HOSPITAL_STORY_IDS; columns follow traffic seeds 0..19.
HOSPITAL_ACCEPTED_GENERATION_ATTEMPTS: tuple[tuple[int, ...], ...] = (
    (2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),
    (1, 2, 1, 0, 0, 0, 2, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0),
    (34, 39, 54, 6, 42, 1, 35, 7, 56, 29, 29, 33, 4, 5, 13, 23, 28, 140, 42, 38),
    (6, 15, 5, 3, 42, 15, 5, 2, 12, 7, 8, 1, 9, 0, 4, 0, 10, 2, 2, 3),
    (21, 4, 7, 25, 0, 7, 1, 8, 35, 6, 11, 21, 15, 5, 36, 5, 31, 21, 35, 20),
)
if (
    len(HOSPITAL_ACCEPTED_GENERATION_ATTEMPTS) != len(HOSPITAL_STORY_IDS)
    or any(
        len(attempts) != len(DEFAULT_HOSPITAL_TRAFFIC_SEEDS)
        or any(
            attempt < 0
            or attempt >= MAX_DYNAMIC_FEASIBILITY_GENERATION_ATTEMPTS
            for attempt in attempts
        )
        for attempts in HOSPITAL_ACCEPTED_GENERATION_ATTEMPTS
    )
):
    raise AssertionError("Hospital accepted-attempt manifest must be 5x20")


@dataclass(frozen=True)
class HospitalPublicationTrial:
    """One member of the frozen five-story by twenty-seed trial grid."""

    ordinal: int
    story_index: int
    story_id: str
    traffic_seed: int
    generator_seed: int
    generation_attempt: int = 0

    @property
    def case_id(self) -> str:
        return f"{self.story_id}/seed-{self.traffic_seed}"


@dataclass(frozen=True)
class OpenLoopStrategyAudit:
    """Outcome of one controller-free corridor behavior at the witness."""

    strategy: str
    swept_by_convoy: bool
    reached_nonroom_escape: bool
    event_time_s: float
    event_coordinate_m: float


@dataclass(frozen=True)
class HospitalNecessityAudit:
    """Fixed geometric witness that a lateral room route is necessary."""

    story_id: str
    witness_station_m: float
    witness_time_s: float
    nonroom_escape_interval_m: tuple[float, float]
    diagnostic_room_route_is_reachable: bool
    blockade_covers_full_cross_section: bool
    nonroom_escape_stations_are_verified: bool
    blockage_interval_is_contiguous: bool
    blockage_started_at_s: float
    blockage_cleared_at_s: float
    blocker_observable_at_witness: bool
    minimum_lateral_room_entry_time_s: float
    lateral_room_entry_lead_margin_s: float
    strategies: tuple[OpenLoopStrategyAudit, ...]

    @property
    def valid(self) -> bool:
        return bool(
            self.diagnostic_room_route_is_reachable
            and self.blockade_covers_full_cross_section
            and self.nonroom_escape_stations_are_verified
            and self.blockage_interval_is_contiguous
            and 0.0 <= self.blockage_started_at_s < self.blockage_cleared_at_s
            and self.blocker_observable_at_witness
            and self.minimum_lateral_room_entry_time_s > 0.0
            and self.lateral_room_entry_lead_margin_s >= 0.0
            and len(self.strategies) == 3
            and {result.strategy for result in self.strategies}
            == {"stop", "continue", "max_retreat"}
            and all(
                result.swept_by_convoy
                and not result.reached_nonroom_escape
                for result in self.strategies
            )
        )

    def require_valid(self) -> None:
        if not self.valid:
            raise RuntimeError(
                f"story {self.story_id!r} lacks a valid open-loop "
                "room-necessity witness"
            )


@dataclass(frozen=True)
class HospitalGeometryContract:
    """Method-independent audit of a generated story world."""

    story_rooms_are_distinct: bool
    initial_state_is_static_safe: bool
    goal_is_static_safe: bool
    start_exit_is_reachable: bool
    diagnostic_refuge_is_reachable: bool
    blockers_are_inside_corridor: bool
    blockade_covers_full_cross_section: bool
    convoy_is_nonreflecting: bool
    convoy_approaches_ego: bool
    convoy_overtakes_bounded_corridor_motion: bool
    convoy_eventually_clears: bool
    initial_obstacles_are_separated: bool
    background_contains_humans_only: bool
    humans_are_in_fixed_active_region: bool
    nonroom_escape_stations_are_verified: bool
    stop_is_swept_before_nonroom_escape: bool
    continue_is_swept_before_nonroom_escape: bool
    max_retreat_is_swept_before_nonroom_escape: bool
    initial_operational_clearance_m: float
    minimum_obstacle_pair_clearance_m: float | None
    convoy_clear_time_s: float
    blockage_started_at_s: float
    blockage_cleared_at_s: float

    @property
    def valid(self) -> bool:
        return all(
            value
            for name, value in asdict(self).items()
            if isinstance(value, bool)
        )

    def require_valid(self) -> None:
        failures = [
            name
            for name, value in asdict(self).items()
            if isinstance(value, bool) and not value
        ]
        if failures:
            raise RuntimeError(
                "hospital story violates its geometric contract: "
                + ", ".join(failures)
            )


@dataclass(frozen=True)
class HospitalStoryScenario:
    """Generated plant inputs plus diagnostics that never influence control."""

    trial: HospitalPublicationTrial
    template: HospitalStoryTemplate
    initial_state: np.ndarray = field(repr=False, compare=False)
    goal: np.ndarray = field(repr=False, compare=False)
    blockers: tuple[Stretcher, ...] = field(repr=False, compare=False)
    humans: tuple[Human, ...] = field(repr=False, compare=False)
    crowd_metadata: HospitalCrowdMetadata
    contract: HospitalGeometryContract
    necessity_audit: HospitalNecessityAudit
    blocker_only_feasibility_audit: DynamicFeasibilityAudit
    feasibility_audit: DynamicFeasibilityAudit
    world_sha256: str
    config: HospitalConfig = field(repr=False, compare=False)
    environment: HospitalEnvironment = field(repr=False, compare=False)

    @property
    def obstacles(self) -> tuple[DynamicObstacle, ...]:
        return (*self.blockers, *self.humans)

    def diagnostic_metadata(self) -> dict[str, object]:
        """Return the separate refuge witness used only to audit geometry."""

        return {
            "diagnostic_only": True,
            "diagnostic_refuge_room": self.template.diagnostic_refuge_room,
            "used_as_controller_input": False,
            "used_for_policy_ranking": False,
            "used_for_success_classification": False,
            "dynamic_witness_room": self.feasibility_audit.witness_room_label,
            "dynamic_witness_used_as_controller_input": False,
        }

    def benchmark_metadata(self) -> dict[str, object]:
        """Return controller-safe case metadata, excluding the refuge label."""

        return {
            "hospital_story_protocol_version": HOSPITAL_STORY_PROTOCOL_VERSION,
            "hospital_story_protocol_sha256": hospital_story_protocol_metadata()[
                "protocol_sha256"
            ],
            "case_id": self.trial.case_id,
            "story_id": self.trial.story_id,
            "story_index": self.trial.story_index,
            "traffic_seed": self.trial.traffic_seed,
            "traffic_generator_seed": self.trial.generator_seed,
            "traffic_generation_attempt": self.trial.generation_attempt,
            "world_sha256": self.world_sha256,
            "hospital_world_sha256": self.world_sha256,
            "human_count": len(self.humans),
            "ordinary_stretcher_count": 0,
            "blocking_stretcher_count": len(self.blockers),
            "dynamic_obstacle_count": len(self.obstacles),
            "full_width_blockade": self.contract.blockade_covers_full_cross_section,
            "convoy_reflects": False,
            "necessity_witness_station_m": (
                self.necessity_audit.witness_station_m
            ),
            "necessity_witness_time_s": self.necessity_audit.witness_time_s,
            "blockage_started_at_s": self.contract.blockage_started_at_s,
            "blockage_cleared_at_s": self.contract.blockage_cleared_at_s,
            "convoy_clear_time_s": self.contract.convoy_clear_time_s,
            "blocker_observable_at_necessity_witness": (
                self.necessity_audit.blocker_observable_at_witness
            ),
            "minimum_lateral_room_entry_time_s": (
                self.necessity_audit.minimum_lateral_room_entry_time_s
            ),
            "lateral_room_entry_lead_margin_s": (
                self.necessity_audit.lateral_room_entry_lead_margin_s
            ),
            "open_loop_room_necessity_verified": self.necessity_audit.valid,
            "traffic_randomizes_humans_only": True,
            "external_refuge_state_machine": False,
            "diagnostic_refuge_supplied_to_controller": False,
            "blocker_only_dynamic_refuge_feasibility_verified": (
                self.blocker_only_feasibility_audit.valid
            ),
            **self.feasibility_audit.public_metadata(),
        }

    def to_simulation(
        self,
        config: HospitalConfig | None = None,
    ):
        """Create a simulator without passing the diagnostic witness anywhere."""

        from .simulation import HospitalSimulation

        runtime_config = self.config if config is None else config
        simulation = HospitalSimulation(
            initial_state=self.initial_state.copy(),
            goal=self.goal.copy(),
            obstacles=tuple(replace(obstacle) for obstacle in self.obstacles),
            config=runtime_config,
            environment=self.environment,
        )
        simulation.benchmark_scenario_metrics = self.benchmark_metadata()
        return simulation


def hospital_accepted_generation_attempt_manifest() -> dict[str, object]:
    """Return the canonical accepted-attempt table for protocol v4."""

    return {
        "schema": HOSPITAL_ACCEPTED_ATTEMPT_MANIFEST_SCHEMA,
        "story_order": list(HOSPITAL_STORY_IDS),
        "traffic_seeds": list(DEFAULT_HOSPITAL_TRAFFIC_SEEDS),
        "attempts": [
            list(attempts)
            for attempts in HOSPITAL_ACCEPTED_GENERATION_ATTEMPTS
        ],
    }


@lru_cache(maxsize=1)
def hospital_accepted_generation_attempt_manifest_sha256() -> str:
    """Hash the canonical JSON encoding of the accepted-attempt table."""

    encoded = json.dumps(
        hospital_accepted_generation_attempt_manifest(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hospital_publication_trial_grid() -> tuple[HospitalPublicationTrial, ...]:
    """Return the exact, stable five-story × twenty-traffic-seed grid."""

    output: list[HospitalPublicationTrial] = []
    for story_index, story in enumerate(HOSPITAL_STORIES):
        for traffic_seed in DEFAULT_HOSPITAL_TRAFFIC_SEEDS:
            generation_attempt = HOSPITAL_ACCEPTED_GENERATION_ATTEMPTS[
                story_index
            ][traffic_seed]
            output.append(
                HospitalPublicationTrial(
                    ordinal=len(output),
                    story_index=story_index,
                    story_id=story.story_id,
                    traffic_seed=traffic_seed,
                    generator_seed=_derive_generator_seed(
                        story_index,
                        traffic_seed,
                        generation_attempt,
                    ),
                    generation_attempt=generation_attempt,
                )
            )
    if len(output) != 100 or len({trial.case_id for trial in output}) != 100:
        raise AssertionError("hospital publication grid must contain 100 cases")
    return tuple(output)


def get_hospital_story(story_id: str) -> HospitalStoryTemplate:
    """Resolve a frozen story by its stable identifier."""

    try:
        return _STORY_BY_ID[str(story_id)]
    except KeyError as exc:
        raise ValueError(
            f"unknown hospital story {story_id!r}; choose "
            + ", ".join(HOSPITAL_STORY_IDS)
        ) from exc


def get_hospital_publication_trial(
    story_id: str,
    traffic_seed: int,
) -> HospitalPublicationTrial:
    """Resolve one exact publication trial, rejecting out-of-grid seeds."""

    story = get_hospital_story(story_id)
    if isinstance(traffic_seed, bool) or int(traffic_seed) != traffic_seed:
        raise ValueError("traffic_seed must be an integer from 0 through 19")
    seed = int(traffic_seed)
    if seed not in DEFAULT_HOSPITAL_TRAFFIC_SEEDS:
        raise ValueError("traffic_seed must be an integer from 0 through 19")
    story_index = HOSPITAL_STORIES.index(story)
    generation_attempt = HOSPITAL_ACCEPTED_GENERATION_ATTEMPTS[
        story_index
    ][seed]
    return HospitalPublicationTrial(
        ordinal=story_index * len(DEFAULT_HOSPITAL_TRAFFIC_SEEDS) + seed,
        story_index=story_index,
        story_id=story.story_id,
        traffic_seed=seed,
        generator_seed=_derive_generator_seed(
            story_index,
            seed,
            generation_attempt,
        ),
        generation_attempt=generation_attempt,
    )


def build_hospital_story_scenario(
    story_id: str,
    *,
    traffic_seed: int,
    config: HospitalConfig = DEFAULT_CONFIG,
    human_count: int = PUBLICATION_HUMAN_COUNT,
    environment: HospitalEnvironment | None = None,
) -> HospitalStoryScenario:
    """Build one story, caching the 100 strict publication worlds."""

    if (
        environment is None
        and human_count == PUBLICATION_HUMAN_COUNT
        and _scenario_construction_projection(config)
        == _scenario_construction_projection(DEFAULT_CONFIG)
    ):
        canonical = _cached_canonical_hospital_story_scenario(
            str(story_id),
            traffic_seed,
        )
        return replace(canonical, config=config)
    return _build_hospital_story_scenario_uncached(
        story_id,
        traffic_seed=traffic_seed,
        config=config,
        human_count=human_count,
        environment=environment,
    )


def _scenario_construction_projection(
    config: HospitalConfig,
) -> tuple[object, ...]:
    """Return fields that affect generated geometry or exact replay."""

    return (
        config.width,
        config.height,
        config.dt,
        config.robot,
        config.planner,
        config.refuge,
        config.safety.safety_margin,
        config.safety.human_margin,
        config.safety.stretcher_margin,
        config.safety.static_margin,
    )


@lru_cache(maxsize=100)
def _cached_canonical_hospital_story_scenario(
    story_id: str,
    traffic_seed: int,
) -> HospitalStoryScenario:
    """Validate one frozen plant world once per process."""

    return _build_hospital_story_scenario_uncached(
        story_id,
        traffic_seed=traffic_seed,
        config=DEFAULT_CONFIG,
        human_count=PUBLICATION_HUMAN_COUNT,
        environment=None,
    )


def _build_hospital_story_scenario_uncached(
    story_id: str,
    *,
    traffic_seed: int,
    config: HospitalConfig,
    human_count: int,
    environment: HospitalEnvironment | None,
) -> HospitalStoryScenario:
    """Build and validate one fixed story with seeded circular traffic.

    ``traffic_seed`` affects only the generated :class:`Human` objects.  Ego,
    goal, room metadata, blocker geometry, blocker positions, and blocker
    velocities are fixed by the story template.
    """

    trial = get_hospital_publication_trial(story_id, traffic_seed)
    template = get_hospital_story(story_id)
    uses_canonical_environment = environment is None
    hospital = environment or build_hospital_environment()
    rooms = _rooms_by_label(hospital)
    corridors = _corridors_by_name(hospital)
    start_room = _require_key(rooms, template.start_room, "room")
    goal_room = _require_key(rooms, template.goal_room, "room")
    corridor = _require_key(corridors, template.corridor_name, "corridor")
    active_corridors = tuple(
        _require_key(corridors, name, "human corridor")
        for name in template.human_corridor_names
    )

    initial_state = np.r_[start_room.center, 0.0, 0.0].astype(float)
    goal = goal_room.center.astype(float)
    blockers = _build_fixed_blockers(template, corridor)
    # Protect only the task endpoints during random traffic placement.  The
    # diagnostic refuge is an audit witness, not a privileged route through
    # the sampled plant; clearing its doorway here would silently make the
    # witness room easier than the other candidate backup rooms.
    protected_points = _protected_room_points(
        hospital,
        (start_room, goal_room),
        config,
    )
    necessity_audit = (
        _cached_canonical_necessity_audit(template.story_id, config)
        if uses_canonical_environment
        else validate_hospital_story_necessity(
            template,
            blockers,
            hospital,
            config,
        )
    )
    necessity_audit.require_valid()
    blocker_only_feasibility_audit = (
        _cached_canonical_blocker_only_feasibility_audit(
            template.story_id,
            config,
        )
        if uses_canonical_environment
        else validate_dynamic_refuge_feasibility(
            story_id=template.story_id,
            start_room_label=template.start_room,
            goal_room_label=template.goal_room,
            corridor_name=template.corridor_name,
            nonroom_escape_interval_m=template.nonroom_escape_interval_m,
            initial_state=initial_state,
            goal=goal,
            blockers=blockers,
            humans=(),
            environment=hospital,
            config=config,
        )
    )
    blocker_only_feasibility_audit.require_valid()

    strict_publication_world = bool(
        uses_canonical_environment and human_count == PUBLICATION_HUMAN_COUNT
    )
    attempts = (
        (trial.generation_attempt,) if strict_publication_world else (0,)
    )
    crowd = None
    contract = None
    feasibility_audit = None
    accepted_trial = trial
    for generation_attempt in attempts:
        generator_seed = _derive_generator_seed(
            trial.story_index,
            trial.traffic_seed,
            generation_attempt,
        )
        candidate_crowd = generate_hospital_crowd(
            hospital,
            config,
            seed=generator_seed,
            ego_position=initial_state[:2],
            goal_position=goal,
            human_count=human_count,
            ordinary_stretcher_count=0,
            protected_points=protected_points,
            protected_clearance=DEFAULT_PROTECTED_CLEARANCE,
            existing_obstacles=blockers,
            human_corridors=active_corridors,
        )
        if candidate_crowd.stretchers:
            raise AssertionError(
                "publication traffic must not generate stretchers"
            )
        candidate_contract = validate_hospital_story_geometry(
            template,
            initial_state,
            goal,
            blockers,
            candidate_crowd.humans,
            hospital,
            config,
            necessity_audit=necessity_audit,
        )
        candidate_contract.require_valid()
        candidate_feasibility_audit = validate_dynamic_refuge_feasibility(
            story_id=template.story_id,
            start_room_label=template.start_room,
            goal_room_label=template.goal_room,
            corridor_name=template.corridor_name,
            nonroom_escape_interval_m=template.nonroom_escape_interval_m,
            initial_state=initial_state,
            goal=goal,
            blockers=blockers,
            humans=candidate_crowd.humans,
            environment=hospital,
            config=config,
        )
        if strict_publication_world and not candidate_feasibility_audit.valid:
            raise RuntimeError(
                f"pinned publication world {trial.case_id!r} at generation "
                f"attempt {trial.generation_attempt} failed its exact "
                "dynamic-feasibility audit; protocol v4 never re-searches "
                "at runtime"
            )
        crowd = candidate_crowd
        contract = candidate_contract
        feasibility_audit = candidate_feasibility_audit
        accepted_trial = replace(
            trial,
            generator_seed=generator_seed,
            generation_attempt=generation_attempt,
        )
        break
    if crowd is None or contract is None or feasibility_audit is None:
        raise RuntimeError(
            f"story {template.story_id!r}/seed-{trial.traffic_seed} did not "
            "produce a valid Hospital world"
        )
    if strict_publication_world:
        feasibility_audit.require_valid()
    world_sha256 = hospital_story_world_sha256(
        initial_state,
        goal,
        (*blockers, *crowd.humans),
    )
    return HospitalStoryScenario(
        trial=accepted_trial,
        template=template,
        initial_state=initial_state,
        goal=goal,
        blockers=blockers,
        humans=crowd.humans,
        crowd_metadata=crowd.metadata,
        contract=contract,
        necessity_audit=necessity_audit,
        blocker_only_feasibility_audit=blocker_only_feasibility_audit,
        feasibility_audit=feasibility_audit,
        world_sha256=world_sha256,
        config=config,
        environment=hospital,
    )


def validate_hospital_story_geometry(
    template: HospitalStoryTemplate,
    initial_state: Sequence[float],
    goal: Sequence[float],
    blockers: Sequence[Stretcher],
    humans: Sequence[Human],
    environment: HospitalEnvironment,
    config: HospitalConfig = DEFAULT_CONFIG,
    *,
    necessity_audit: HospitalNecessityAudit | None = None,
) -> HospitalGeometryContract:
    """Audit only geometry and plant limits; no controller is instantiated."""

    state = np.asarray(initial_state, dtype=float)
    goal_point = np.asarray(goal, dtype=float)
    rooms = _rooms_by_label(environment)
    corridors = _corridors_by_name(environment)
    start_room = _require_key(rooms, template.start_room, "room")
    goal_room = _require_key(rooms, template.goal_room, "room")
    refuge_room = _require_key(
        rooms,
        template.diagnostic_refuge_room,
        "room",
    )
    corridor = _require_key(corridors, template.corridor_name, "corridor")
    active_corridors = tuple(
        _require_key(corridors, name, "human corridor")
        for name in template.human_corridor_names
    )
    audit = necessity_audit or validate_hospital_story_necessity(
        template,
        blockers,
        environment,
        config,
    )

    start_access = environment.room_door_path(
        start_room,
        config.robot.radius,
        config.refuge.inside_door_offset,
        config.refuge.outside_door_offset,
    )
    refuge_access = environment.room_door_path(
        refuge_room,
        config.robot.radius,
        config.refuge.inside_door_offset,
        config.refuge.outside_door_offset,
    )
    start_exit_is_reachable = _door_path_is_free(
        environment,
        start_access,
        config.robot.radius,
    )
    refuge_door_is_reachable = _door_path_is_free(
        environment,
        refuge_access,
        config.robot.radius,
    )
    same_corridor_segment_is_free = environment.segment_is_free(
        start_access[0],
        refuge_access[0],
        config.robot.radius,
    )

    full_width = all(
        blocker.cross_section_width
        + 2.0 * (config.robot.radius + config.safety.stretcher_margin)
        >= min(corridor.width, corridor.height)
        for blocker in blockers
    )
    blockers_inside = (
        len(blockers) == template.blocker_count
        and all(
            blocker.identifier.startswith(
                f"blocking-stretcher-{template.story_id}-"
            )
            and blocker.axis
            == ("x" if corridor.width >= corridor.height else "y")
            and _stretcher_is_inside_corridor(blocker, corridor)
            for blocker in blockers
        )
    )
    convoy_clear_time = _convoy_clear_time(blockers, corridor)
    all_obstacles: tuple[DynamicObstacle, ...] = (*blockers, *humans)
    pair_clearances = [
        obstacle_pair_clearance(first, second)
        for index, first in enumerate(all_obstacles)
        for second in all_obstacles[index + 1 :]
    ]
    minimum_pair_clearance = min(pair_clearances) if pair_clearances else None
    operational_radius = config.robot.radius + config.safety.safety_margin
    initial_clearances = [
        obstacle.signed_clearance(
            state[:2],
            operational_radius,
            (
                config.safety.human_margin
                if isinstance(obstacle, Human)
                else config.safety.stretcher_margin
            ),
        )
        for obstacle in all_obstacles
    ]
    initial_clearances.append(
        environment.static_clearance(
            state[:2],
            config.robot.radius + config.safety.static_margin,
        )
    )
    initial_operational_clearance = min(initial_clearances)

    report = HospitalGeometryContract(
        story_rooms_are_distinct=(
            len(
                {
                    template.start_room,
                    template.goal_room,
                    template.diagnostic_refuge_room,
                }
            )
            == 3
            and start_room.contains(state[:2])
            and goal_room.contains(goal_point)
        ),
        initial_state_is_static_safe=not environment.is_collision(
            state[:2],
            config.robot.radius,
        ),
        goal_is_static_safe=not environment.is_collision(
            goal_point,
            config.robot.radius,
        ),
        start_exit_is_reachable=start_exit_is_reachable,
        diagnostic_refuge_is_reachable=(
            refuge_door_is_reachable and same_corridor_segment_is_free
        ),
        blockers_are_inside_corridor=blockers_inside,
        blockade_covers_full_cross_section=full_width,
        convoy_is_nonreflecting=bool(blockers)
        and all(not blocker.reflect_at_route_bounds for blocker in blockers),
        convoy_approaches_ego=bool(blockers)
        and all(
            template.travel_direction * blocker.speed < 0.0
            for blocker in blockers
        ),
        convoy_overtakes_bounded_corridor_motion=bool(blockers)
        and min(abs(blocker.speed) for blocker in blockers)
        > config.robot.v_max,
        convoy_eventually_clears=isfinite(convoy_clear_time)
        and convoy_clear_time > 0.0,
        initial_obstacles_are_separated=(
            initial_operational_clearance > 0.0
            and (
                minimum_pair_clearance is None
                or minimum_pair_clearance
                >= INITIAL_PAIRWISE_CLEARANCE - 1e-12
            )
        ),
        background_contains_humans_only=(
            all(isinstance(human, Human) for human in humans)
            and all(isinstance(blocker, Stretcher) for blocker in blockers)
        ),
        humans_are_in_fixed_active_region=all(
            any(region.contains(human.center) for region in active_corridors)
            for human in humans
        ),
        nonroom_escape_stations_are_verified=(
            audit.nonroom_escape_stations_are_verified
        ),
        stop_is_swept_before_nonroom_escape=_strategy_is_swept(
            audit,
            "stop",
        ),
        continue_is_swept_before_nonroom_escape=_strategy_is_swept(
            audit,
            "continue",
        ),
        max_retreat_is_swept_before_nonroom_escape=_strategy_is_swept(
            audit,
            "max_retreat",
        ),
        initial_operational_clearance_m=float(initial_operational_clearance),
        minimum_obstacle_pair_clearance_m=(
            float(minimum_pair_clearance)
            if minimum_pair_clearance is not None
            else None
        ),
        convoy_clear_time_s=float(convoy_clear_time),
        blockage_started_at_s=float(audit.blockage_started_at_s),
        blockage_cleared_at_s=float(audit.blockage_cleared_at_s),
    )
    return report


def validate_hospital_story_necessity(
    template: HospitalStoryTemplate,
    blockers: Sequence[Stretcher],
    environment: HospitalEnvironment,
    config: HospitalConfig = DEFAULT_CONFIG,
) -> HospitalNecessityAudit:
    """Verify the fixed room-necessity witness without running a controller.

    At the template's immutable witness time, a virtual ego is placed at the
    diagnostic room's corridor station with maximum bounded corridor speed.
    Three deliberately simple open-loop behaviors are checked: maximum braking
    to a stop, continuing at bounded speed, and maximum acceleration into
    corridor retreat. A behavior counts as swept only if exact rectangle-circle
    geometry collides before the ego reaches either non-room junction bounding
    the witness interval. The room route is checked independently against only
    static hospital geometry.
    """

    rooms = _rooms_by_label(environment)
    corridors = _corridors_by_name(environment)
    refuge = _require_key(
        rooms,
        template.diagnostic_refuge_room,
        "room",
    )
    corridor = _require_key(corridors, template.corridor_name, "corridor")
    horizontal = corridor.width >= corridor.height
    witness = (
        np.array(
            [template.necessity_witness_station_m, corridor.center[1]],
            dtype=float,
        )
        if horizontal
        else np.array(
            [corridor.center[0], template.necessity_witness_station_m],
            dtype=float,
        )
    )
    refuge_access = environment.room_door_path(
        refuge,
        config.robot.radius,
        config.refuge.inside_door_offset,
        config.refuge.outside_door_offset,
    )
    longitudinal_index = 0 if horizontal else 1
    witness_matches_room_station = bool(
        abs(
            refuge_access[0][longitudinal_index]
            - template.necessity_witness_station_m
        )
        <= 1e-9
    )
    room_route_reachable = bool(
        witness_matches_room_station
        and environment.segment_is_free(
            witness,
            refuge_access[0],
            config.robot.radius,
        )
        and _door_path_is_free(
            environment,
            refuge_access,
            config.robot.radius,
        )
    )
    full_width = bool(blockers) and all(
        blocker.cross_section_width + 4.0 * config.robot.radius
        >= min(corridor.width, corridor.height)
        for blocker in blockers
    )
    escape_stations_are_verified = all(
        _is_nonroom_escape_boundary(
            environment,
            corridor,
            coordinate,
        )
        for coordinate in template.nonroom_escape_interval_m
    )
    encounter_blockers = tuple(
        blocker.predicted(template.necessity_witness_time_s, environment)
        for blocker in blockers
    )
    strategies = tuple(
        _simulate_open_loop_strategy(
            strategy,
            template,
            encounter_blockers,
            corridor,
            environment,
            config,
        )
        for strategy in ("stop", "continue", "max_retreat")
    )
    (
        blockage_started_at_s,
        blockage_cleared_at_s,
        blockage_interval_is_contiguous,
    ) = _blockage_interval_at_witness(
        template,
        blockers,
        config,
    )
    predicted_at_witness = tuple(
        blocker.predicted(template.necessity_witness_time_s, environment)
        for blocker in blockers
    )
    blocker_observable_at_witness = any(
        float(np.linalg.norm(blocker.center - witness))
        - 0.5 * float(np.hypot(blocker.length, blocker.width))
        <= PUBLICATION_SENSING_RANGE_M
        for blocker in predicted_at_witness
    )
    lateral_room_distance = float(np.linalg.norm(refuge_access[2] - witness))
    minimum_lateral_room_entry_time_s = float(
        np.sqrt(2.0 * lateral_room_distance / config.robot.a_max)
    )
    lateral_room_entry_lead_margin_s = float(
        blockage_started_at_s
        - template.necessity_witness_time_s
        - minimum_lateral_room_entry_time_s
    )
    return HospitalNecessityAudit(
        story_id=template.story_id,
        witness_station_m=template.necessity_witness_station_m,
        witness_time_s=template.necessity_witness_time_s,
        nonroom_escape_interval_m=template.nonroom_escape_interval_m,
        diagnostic_room_route_is_reachable=room_route_reachable,
        blockade_covers_full_cross_section=full_width,
        nonroom_escape_stations_are_verified=escape_stations_are_verified,
        blockage_interval_is_contiguous=blockage_interval_is_contiguous,
        blockage_started_at_s=blockage_started_at_s,
        blockage_cleared_at_s=blockage_cleared_at_s,
        blocker_observable_at_witness=blocker_observable_at_witness,
        minimum_lateral_room_entry_time_s=(
            minimum_lateral_room_entry_time_s
        ),
        lateral_room_entry_lead_margin_s=(
            lateral_room_entry_lead_margin_s
        ),
        strategies=strategies,
    )


@lru_cache(maxsize=64)
def _cached_canonical_necessity_audit(
    story_id: str,
    config: HospitalConfig,
) -> HospitalNecessityAudit:
    template = get_hospital_story(story_id)
    environment = build_hospital_environment()
    corridor = _require_key(
        _corridors_by_name(environment),
        template.corridor_name,
        "corridor",
    )
    blockers = _build_fixed_blockers(template, corridor)
    return validate_hospital_story_necessity(
        template,
        blockers,
        environment,
        config,
    )


@lru_cache(maxsize=64)
def _cached_canonical_blocker_only_feasibility_audit(
    story_id: str,
    config: HospitalConfig,
) -> DynamicFeasibilityAudit:
    """Replay the fixed story before any randomized traffic is admitted."""

    template = get_hospital_story(story_id)
    environment = build_hospital_environment()
    rooms = _rooms_by_label(environment)
    corridor = _require_key(
        _corridors_by_name(environment),
        template.corridor_name,
        "corridor",
    )
    blockers = _build_fixed_blockers(template, corridor)
    initial_state = np.r_[rooms[template.start_room].center, 0.0, 0.0]
    return validate_dynamic_refuge_feasibility(
        story_id=template.story_id,
        start_room_label=template.start_room,
        goal_room_label=template.goal_room,
        corridor_name=template.corridor_name,
        nonroom_escape_interval_m=template.nonroom_escape_interval_m,
        initial_state=initial_state,
        goal=rooms[template.goal_room].center,
        blockers=blockers,
        humans=(),
        environment=environment,
        config=config,
    )


def hospital_story_world_sha256(
    initial_state: Sequence[float],
    goal: Sequence[float],
    obstacles: Sequence[DynamicObstacle],
) -> str:
    """Hash exact plant inputs and ordered obstacle geometry for one world."""

    state = np.asarray(initial_state, dtype=float)
    goal_point = np.asarray(goal, dtype=float)
    if state.shape != (4,) or goal_point.shape != (2,):
        raise ValueError("world hash requires state shape (4,) and goal shape (2,)")
    obstacle_payloads: list[dict[str, object]] = []
    for index, obstacle in enumerate(obstacles):
        if isinstance(obstacle, Human):
            payload: dict[str, object] = {
                "index": index,
                "type": "human",
                "identifier": obstacle.identifier,
                "x": _float_token(obstacle.x),
                "y": _float_token(obstacle.y),
                "vx": _float_token(obstacle.vx),
                "vy": _float_token(obstacle.vy),
                "radius": _float_token(obstacle.radius),
            }
        elif isinstance(obstacle, Stretcher):
            payload = {
                "index": index,
                "type": "stretcher",
                "identifier": obstacle.identifier,
                "coordinate": _float_token(obstacle.coordinate),
                "lateral": _float_token(obstacle.lateral),
                "speed": _float_token(obstacle.speed),
                "axis": obstacle.axis,
                "route_min": _float_token(obstacle.route_min),
                "route_max": _float_token(obstacle.route_max),
                "length": _float_token(obstacle.length),
                "width": _float_token(obstacle.width),
                "reflect_at_route_bounds": obstacle.reflect_at_route_bounds,
            }
        else:
            raise TypeError(
                "world hash supports only hospital Human and Stretcher obstacles"
            )
        obstacle_payloads.append(payload)
    payload = {
        "schema": "hospital_world_v1",
        "initial_state": [_float_token(value) for value in state],
        "goal": [_float_token(value) for value in goal_point],
        "obstacles": obstacle_payloads,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=1)
def hospital_geometry_sha256() -> str:
    """Return an exact digest of static Hospital floor-plan geometry."""

    environment = build_hospital_environment()
    payload = {
        "geometry_version": HOSPITAL_GEOMETRY_VERSION,
        "width": _float_token(environment.width),
        "height": _float_token(environment.height),
        "floors": [_rect_payload(rect) for rect in environment.floor_rects],
        "corridors": [
            _rect_payload(rect) for rect in environment.corridor_rects
        ],
        "walls": [_rect_payload(rect) for rect in environment.wall_rects],
        "doors": [_rect_payload(rect) for rect in environment.door_rects],
        "rooms": [
            {
                "label": room.label,
                "rect": _rect_payload(room.rect),
                "door": _rect_payload(room.door.rect),
                "door_side": room.door.side,
            }
            for room in environment.rooms
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hospital_story_protocol_metadata() -> dict[str, object]:
    """Return canonical, versioned provenance for the default 100-case grid."""

    payload = _protocol_payload()
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **payload,
        "protocol_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _protocol_payload() -> dict[str, object]:
    return {
        "protocol_version": HOSPITAL_STORY_PROTOCOL_VERSION,
        "geometry_version": HOSPITAL_GEOMETRY_VERSION,
        "geometry_sha256": hospital_geometry_sha256(),
        "world_sha256_schema": "hospital_world_v1",
        "seed_namespace": HOSPITAL_STORY_SEED_NAMESPACE,
        "story_count": len(HOSPITAL_STORIES),
        "traffic_seeds_per_story": len(DEFAULT_HOSPITAL_TRAFFIC_SEEDS),
        "trial_count": len(HOSPITAL_STORIES)
        * len(DEFAULT_HOSPITAL_TRAFFIC_SEEDS),
        "traffic_seeds": list(DEFAULT_HOSPITAL_TRAFFIC_SEEDS),
        "human_count": PUBLICATION_HUMAN_COUNT,
        "ordinary_stretcher_count": 0,
        "accepted_generation_attempt_manifest": (
            hospital_accepted_generation_attempt_manifest()
        ),
        "accepted_generation_attempt_manifest_sha256": (
            hospital_accepted_generation_attempt_manifest_sha256()
        ),
        "traffic_generator": {
            "traffic_speed_cap_mps": TRAFFIC_SPEED_CAP,
            "human_speed_range_mps": list(HUMAN_SPEED_RANGE),
            "human_radius_m": HUMAN_RADIUS,
            "protected_clearance_m": DEFAULT_PROTECTED_CLEARANCE,
            "protected_room_roles": ["start", "goal"],
            "diagnostic_refuge_protected": False,
            "initial_pairwise_clearance_m": INITIAL_PAIRWISE_CLEARANCE,
            "static_placement_margin_m": STATIC_PLACEMENT_MARGIN,
            "maximum_attempts_per_obstacle": (
                MAX_PLACEMENT_ATTEMPTS_PER_OBSTACLE
            ),
        },
        "dynamic_feasibility_conditioning": {
            "schema": DYNAMIC_FEASIBILITY_SCHEMA,
            "method_independent": True,
            "witness_room_is_controller_input": False,
            "witness_controls_are_controller_input": False,
            "accepted_generation_attempt_recorded": True,
            "strict_build_runtime_research": False,
            "canonical_scenario_cache_size": 100,
            "maximum_generation_attempts": (
                MAX_DYNAMIC_FEASIBILITY_GENERATION_ATTEMPTS
            ),
            "minimum_room_entry_lead_s": MINIMUM_ROOM_ENTRY_LEAD_S,
            "post_convoy_hold_buffer_s": POST_CONVOY_HOLD_BUFFER_S,
            "minimum_operational_clearance_m": (
                MINIMUM_OPERATIONAL_CLEARANCE_M
            ),
            "dynamic_substeps_per_plant_step": WITNESS_DYNAMIC_SUBSTEPS,
            "maximum_witness_time_s": MAXIMUM_WITNESS_TIME_S,
            "goal_tolerance_m": GOAL_TOLERANCE_M,
            "witness_human_caution_clearance_m": (
                WITNESS_HUMAN_CAUTION_CLEARANCE_M
            ),
            "witness_preferred_clearance_m": (
                WITNESS_PREFERRED_CLEARANCE_M
            ),
            "nominal_jit_block_steps": WITNESS_NOMINAL_BLOCK_STEPS,
            "exact_synchronized_dynamic_replay": True,
            "clearance_approximation_used": False,
            "post_convoy_departure_delays_s": list(
                POST_CONVOY_DEPARTURE_DELAYS_S
            ),
        },
        "blocker_geometry": {
            "length_m": BLOCKER_LENGTH_M,
            "width_m": BLOCKER_WIDTH_M,
            "speed_mps": BLOCKER_SPEED_MPS,
            "reflect_at_route_bounds": False,
        },
        "plant_geometry_contract": {
            "robot_radius_m": DEFAULT_CONFIG.robot.radius,
            "robot_max_speed_mps": DEFAULT_CONFIG.robot.v_max,
            "robot_max_acceleration_mps2": DEFAULT_CONFIG.robot.a_max,
            "robot_safety_margin_m": DEFAULT_CONFIG.safety.safety_margin,
            "stretcher_margin_m": DEFAULT_CONFIG.safety.stretcher_margin,
            "blockage_interval_footprint": (
                "half_stretcher_length_plus_robot_radius_plus_safety_margin_"
                "plus_stretcher_margin"
            ),
            "room_inside_door_offset_m": (
                DEFAULT_CONFIG.refuge.inside_door_offset
            ),
            "room_outside_door_offset_m": (
                DEFAULT_CONFIG.refuge.outside_door_offset
            ),
        },
        "publication_perception": {
            "sensing_range_m": PUBLICATION_SENSING_RANGE_M,
            "line_of_sight_filtering": True,
            "obstacle_id_priority": False,
        },
        "randomized_entities": ["human"],
        "randomized_fields": ["x", "y", "vx", "vy"],
        "fixed_entities": [
            "ego_initial_state",
            "nominal_goal",
            "blocking_stretchers",
            "hospital_geometry",
        ],
        "paired_world_shared_across_methods": True,
        "diagnostic_refuge_is_controller_input": False,
        "diagnostic_refuge_is_success_target": False,
        "external_refuge_state_machine": False,
        "stories": [asdict(story) for story in HOSPITAL_STORIES],
    }


def _derive_generator_seed(
    story_index: int,
    traffic_seed: int,
    generation_attempt: int = 0,
) -> int:
    if generation_attempt < 0:
        raise ValueError("generation_attempt must be nonnegative")
    entropy = [
        HOSPITAL_STORY_SEED_NAMESPACE,
        int(story_index),
        int(traffic_seed),
    ]
    # Attempt zero intentionally retains the v3 generator seed.  Subsequent
    # attempts occupy a disjoint, versioned deterministic namespace.
    if generation_attempt:
        entropy.extend((0x46454153, int(generation_attempt)))
    sequence = np.random.SeedSequence(
        entropy
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _rooms_by_label(environment: HospitalEnvironment) -> dict[str, Room]:
    return {room.label: room for room in environment.rooms}


def _corridors_by_name(environment: HospitalEnvironment) -> dict[str, Rect]:
    return {corridor.name: corridor for corridor in environment.corridor_rects}


def _require_key(
    values: Mapping[str, Room | Rect],
    name: str,
    kind: str,
):
    try:
        return values[name]
    except KeyError as exc:
        raise RuntimeError(f"hospital environment is missing {kind} {name!r}") from exc


def _build_fixed_blockers(
    template: HospitalStoryTemplate,
    corridor: Rect,
) -> tuple[Stretcher, ...]:
    blockers: list[Stretcher] = []
    for index, requested_coordinate in enumerate(
        template.convoy_coordinates_m
    ):
        axis, coordinate, lateral, route_min, route_max = stretcher_route(
            corridor,
            coordinate=requested_coordinate,
            lateral=(
                corridor.center[1]
                if corridor.width >= corridor.height
                else corridor.center[0]
            ),
            length=BLOCKER_LENGTH_M,
            width=BLOCKER_WIDTH_M,
        )
        if abs(coordinate - requested_coordinate) > 1e-12:
            raise RuntimeError(
                f"story blocker coordinate {requested_coordinate} lies outside "
                f"{corridor.name}"
            )
        blockers.append(
            Stretcher(
                identifier=f"blocking-stretcher-{template.story_id}-{index}",
                coordinate=float(coordinate),
                lateral=float(lateral),
                speed=float(template.convoy_speed_mps),
                axis=axis,
                route_min=float(route_min),
                route_max=float(route_max),
                length=BLOCKER_LENGTH_M,
                width=BLOCKER_WIDTH_M,
                reflect_at_route_bounds=False,
            )
        )
    return tuple(blockers)


def _protected_room_points(
    environment: HospitalEnvironment,
    rooms: Iterable[Room],
    config: HospitalConfig,
) -> tuple[np.ndarray, ...]:
    points: list[np.ndarray] = []
    for room in rooms:
        points.extend(
            environment.room_door_path(
                room,
                config.robot.radius,
                config.refuge.inside_door_offset,
                config.refuge.outside_door_offset,
            )
        )
    return tuple(point.copy() for point in points)


def _door_path_is_free(
    environment: HospitalEnvironment,
    points: Sequence[np.ndarray],
    robot_radius: float,
) -> bool:
    return all(
        environment.segment_is_free(start, end, robot_radius)
        for start, end in zip(points, points[1:])
    )


def _stretcher_is_inside_corridor(
    blocker: Stretcher,
    corridor: Rect,
) -> bool:
    if blocker.axis == "x":
        half = np.array([0.5 * blocker.length, 0.5 * blocker.width])
    else:
        half = np.array([0.5 * blocker.width, 0.5 * blocker.length])
    center = blocker.center
    return bool(
        center[0] - half[0] >= corridor.x - 1e-12
        and center[0] + half[0] <= corridor.x1 + 1e-12
        and center[1] - half[1] >= corridor.y - 1e-12
        and center[1] + half[1] <= corridor.y1 + 1e-12
    )


def _convoy_clear_time(
    blockers: Sequence[Stretcher],
    corridor: Rect,
) -> float:
    clear_times: list[float] = []
    for blocker in blockers:
        if blocker.axis == "x":
            low, high = corridor.x, corridor.x1
        else:
            low, high = corridor.y, corridor.y1
        if blocker.speed < 0.0:
            clear_times.append(
                (blocker.coordinate + 0.5 * blocker.length - low)
                / abs(blocker.speed)
            )
        elif blocker.speed > 0.0:
            clear_times.append(
                (high - blocker.coordinate + 0.5 * blocker.length)
                / blocker.speed
            )
        else:
            return float("inf")
    return max(clear_times, default=float("inf"))


def _simulate_open_loop_strategy(
    strategy: str,
    template: HospitalStoryTemplate,
    blockers: Sequence[Stretcher],
    corridor: Rect,
    environment: HospitalEnvironment,
    config: HospitalConfig,
) -> OpenLoopStrategyAudit:
    if strategy not in {"stop", "continue", "max_retreat"}:
        raise ValueError(f"unknown necessity-audit strategy {strategy!r}")
    horizontal = corridor.width >= corridor.height
    coordinate = float(template.necessity_witness_station_m)
    # Velocity is expressed along nominal travel, so +v_max always means
    # continuing toward the goal regardless of map-axis direction.
    signed_velocity = float(config.robot.v_max)
    dt = min(float(config.dt), 0.02)
    elapsed = 0.0
    lower, upper = template.nonroom_escape_interval_m
    maximum_time = max(30.0, _convoy_clear_time(blockers, corridor) + 10.0)

    while elapsed <= maximum_time + 1e-12:
        next_signed_velocity = signed_velocity
        if strategy == "stop":
            next_signed_velocity = max(
                0.0,
                signed_velocity - config.robot.a_max * dt,
            )
        elif strategy == "max_retreat":
            next_signed_velocity = max(
                -config.robot.v_max,
                signed_velocity - config.robot.a_max * dt,
            )
        displacement = (
            template.travel_direction
            * 0.5
            * (signed_velocity + next_signed_velocity)
            * dt
        )
        next_coordinate = coordinate + displacement

        for substep in range(1, 9):
            alpha = substep / 8.0
            sample_time = elapsed + alpha * dt
            sample_coordinate = coordinate + alpha * displacement
            position = (
                np.array([sample_coordinate, corridor.center[1]], dtype=float)
                if horizontal
                else np.array(
                    [corridor.center[0], sample_coordinate],
                    dtype=float,
                )
            )
            swept = any(
                blocker.predicted(sample_time, environment).signed_clearance(
                    position,
                    config.robot.radius,
                )
                <= 0.0
                for blocker in blockers
            )
            if swept:
                return OpenLoopStrategyAudit(
                    strategy=strategy,
                    swept_by_convoy=True,
                    reached_nonroom_escape=False,
                    event_time_s=float(sample_time),
                    event_coordinate_m=float(sample_coordinate),
                )
            if sample_coordinate <= lower or sample_coordinate >= upper:
                return OpenLoopStrategyAudit(
                    strategy=strategy,
                    swept_by_convoy=False,
                    reached_nonroom_escape=True,
                    event_time_s=float(sample_time),
                    event_coordinate_m=float(sample_coordinate),
                )
        coordinate = next_coordinate
        signed_velocity = next_signed_velocity
        elapsed += dt

    return OpenLoopStrategyAudit(
        strategy=strategy,
        swept_by_convoy=False,
        reached_nonroom_escape=False,
        event_time_s=float(maximum_time),
        event_coordinate_m=float(coordinate),
    )


def _blockage_interval_at_witness(
    template: HospitalStoryTemplate,
    blockers: Sequence[Stretcher],
    config: HospitalConfig,
) -> tuple[float, float, bool]:
    """Return the fixed-station operational blockade interval.

    The longitudinal footprint is the rectangle half-length plus the complete
    operational robot/stretcher clearance used by the benchmark safe set. The
    result is purely a prediction of immutable obstacle motion and is never a
    controller phase or switching signal.
    """

    intervals: list[tuple[float, float]] = []
    station = template.necessity_witness_station_m
    for blocker in blockers:
        if abs(blocker.speed) <= 1e-12:
            return 0.0, float("inf"), False
        half_footprint = (
            0.5 * blocker.length
            + config.robot.radius
            + config.safety.safety_margin
            + config.safety.stretcher_margin
        )
        roots = (
            (station - half_footprint - blocker.coordinate) / blocker.speed,
            (station + half_footprint - blocker.coordinate) / blocker.speed,
        )
        start, end = min(roots), max(roots)
        if end < 0.0:
            continue
        intervals.append((max(0.0, float(start)), float(end)))
    if not intervals:
        return float("inf"), float("inf"), False

    intervals.sort()
    merged_start, merged_end = intervals[0]
    contiguous = True
    for start, end in intervals[1:]:
        if start > merged_end + 1e-12:
            contiguous = False
        merged_end = max(merged_end, end)
    return float(merged_start), float(merged_end), contiguous


def _is_nonroom_escape_boundary(
    environment: HospitalEnvironment,
    active_corridor: Rect,
    coordinate: float,
) -> bool:
    """Whether a witness-interval endpoint is a perpendicular hall junction."""

    horizontal = active_corridor.width >= active_corridor.height
    point = (
        np.array([coordinate, active_corridor.center[1]], dtype=float)
        if horizontal
        else np.array([active_corridor.center[0], coordinate], dtype=float)
    )
    for candidate in environment.corridor_rects:
        if candidate is active_corridor:
            continue
        candidate_horizontal = candidate.width >= candidate.height
        if candidate_horizontal == horizontal or not candidate.contains(
            point,
            margin=1e-9,
        ):
            continue
        candidate_low = candidate.x if horizontal else candidate.y
        candidate_high = candidate.x1 if horizontal else candidate.y1
        if abs(coordinate - candidate_low) <= 1e-9 or abs(
            coordinate - candidate_high
        ) <= 1e-9:
            return True
    return False


def _strategy_is_swept(
    audit: HospitalNecessityAudit,
    strategy: str,
) -> bool:
    result = next(
        (item for item in audit.strategies if item.strategy == strategy),
        None,
    )
    return bool(
        result is not None
        and result.swept_by_convoy
        and not result.reached_nonroom_escape
    )


def _float_token(value: float) -> str:
    number = float(value)
    if not isfinite(number):
        raise ValueError("hashed world and geometry values must be finite")
    return number.hex()


def _rect_payload(rect: Rect) -> dict[str, str]:
    return {
        "x": _float_token(rect.x),
        "y": _float_token(rect.y),
        "width": _float_token(rect.width),
        "height": _float_token(rect.height),
        "name": rect.name,
        "kind": rect.kind,
    }


__all__ = [
    "BLOCKER_LENGTH_M",
    "BLOCKER_SPEED_MPS",
    "BLOCKER_WIDTH_M",
    "DEFAULT_HOSPITAL_TRAFFIC_SEEDS",
    "HOSPITAL_STORIES",
    "HOSPITAL_GEOMETRY_VERSION",
    "HOSPITAL_STORY_IDS",
    "HOSPITAL_STORY_PROTOCOL_VERSION",
    "HospitalGeometryContract",
    "HospitalNecessityAudit",
    "OpenLoopStrategyAudit",
    "HospitalPublicationTrial",
    "HospitalStoryScenario",
    "HospitalStoryTemplate",
    "PUBLICATION_HUMAN_COUNT",
    "PUBLICATION_SENSING_RANGE_M",
    "build_hospital_story_scenario",
    "get_hospital_publication_trial",
    "get_hospital_story",
    "hospital_publication_trial_grid",
    "hospital_geometry_sha256",
    "hospital_story_protocol_metadata",
    "hospital_story_world_sha256",
    "validate_hospital_story_necessity",
    "validate_hospital_story_geometry",
]
