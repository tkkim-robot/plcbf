from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np
import pytest

from examples.hospital.config import DEFAULT_CONFIG
from examples.hospital.obstacles import Human, Stretcher
from examples.hospital.scenarios import (
    DEFAULT_HOSPITAL_TRAFFIC_SEEDS,
    HOSPITAL_STORIES,
    HOSPITAL_STORY_IDS,
    HOSPITAL_STORY_PROTOCOL_VERSION,
    PUBLICATION_HUMAN_COUNT,
    build_hospital_story_scenario,
    get_hospital_publication_trial,
    get_hospital_story,
    hospital_publication_trial_grid,
    hospital_geometry_sha256,
    hospital_story_protocol_metadata,
    hospital_story_world_sha256,
)


EXPECTED_PROTOCOL_SHA256 = (
    "adac3eb688a4b4a7f05fdf8f1092b9b517349de5dee3af7a11ad99e721719a57"
)
EXPECTED_GEOMETRY_SHA256 = (
    "202a75d9bb05381249ed1fc12d96f5ca0020c1b45029434c12b2f305afa15923"
)
EXPECTED_SEED_ZERO_WORLD_SHA256 = {
    "main_eastbound": (
        "295874eb0079a20f2ba08c817748108121e58d96595754339c67927a9272c7f7"
    ),
    "main_westbound": (
        "7b353f706cf54ccc31d688ba21aa96d5b0baa5aa76a17039a9d5f0c6a7226e9e"
    ),
    "north_eastbound": (
        "85440f0f950556dd9c63ed9d1bf0dc0baca34446973adc29a3857d85e195117c"
    ),
    "north_westbound": (
        "ee4a69c5457377d7b94834a55713becaf11092b1f1dda4bed6ffa9c82feb36a3"
    ),
    "south_eastbound": (
        "80ad91d9429e7fd75e98fa60c82c9fc9e852f1828b34d36c41a55e8d74da31c3"
    ),
}


def _obstacle_signature(scenario) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            obstacle.identifier,
            *np.round(obstacle.center, 12),
            *np.round(obstacle.velocity, 12),
        )
        for obstacle in scenario.obstacles
    )


def _blocker_signature(scenario) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            blocker.identifier,
            blocker.axis,
            blocker.coordinate,
            blocker.lateral,
            blocker.speed,
            blocker.length,
            blocker.width,
            blocker.reflect_at_route_bounds,
        )
        for blocker in scenario.blockers
    )


def test_publication_grid_is_exact_stable_five_by_twenty() -> None:
    grid = hospital_publication_trial_grid()

    assert HOSPITAL_STORY_PROTOCOL_VERSION == "hospital_fixed_refuge_v3"
    assert len(HOSPITAL_STORIES) == len(HOSPITAL_STORY_IDS) == 5
    assert DEFAULT_HOSPITAL_TRAFFIC_SEEDS == tuple(range(20))
    assert len(grid) == 100
    assert tuple(trial.ordinal for trial in grid) == tuple(range(100))
    assert len({trial.case_id for trial in grid}) == 100
    assert len({trial.generator_seed for trial in grid}) == 100
    assert grid[0].case_id == "main_eastbound/seed-0"
    assert grid[-1].case_id == "south_eastbound/seed-19"
    for story_index, story_id in enumerate(HOSPITAL_STORY_IDS):
        trials = [trial for trial in grid if trial.story_id == story_id]
        assert len(trials) == 20
        assert {trial.traffic_seed for trial in trials} == set(range(20))
        assert {trial.story_index for trial in trials} == {story_index}


def test_protocol_provenance_is_canonical_and_declares_randomization_boundary() -> None:
    first = hospital_story_protocol_metadata()
    repeated = hospital_story_protocol_metadata()

    assert first == repeated
    assert first["protocol_version"] == HOSPITAL_STORY_PROTOCOL_VERSION
    assert first["story_count"] == 5
    assert first["traffic_seeds_per_story"] == 20
    assert first["trial_count"] == 100
    assert first["human_count"] == PUBLICATION_HUMAN_COUNT == 50
    assert first["ordinary_stretcher_count"] == 0
    assert first["randomized_entities"] == ["human"]
    assert first["randomized_fields"] == ["x", "y", "vx", "vy"]
    assert first["paired_world_shared_across_methods"] is True
    assert first["diagnostic_refuge_is_controller_input"] is False
    assert first["diagnostic_refuge_is_success_target"] is False
    assert first["external_refuge_state_machine"] is False
    assert first["publication_perception"] == {
        "sensing_range_m": 24.0,
        "line_of_sight_filtering": True,
        "obstacle_id_priority": False,
    }
    assert first["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256
    assert first["geometry_sha256"] == EXPECTED_GEOMETRY_SHA256
    assert hospital_geometry_sha256() == EXPECTED_GEOMETRY_SHA256
    assert first["geometry_version"] == "hospital_floorplan_140x95_v1"
    assert first["world_sha256_schema"] == "hospital_world_v1"
    assert first["traffic_generator"] == {
        "traffic_speed_cap_mps": 1.45,
        "human_speed_range_mps": [0.609, 1.45],
        "human_radius_m": 0.52,
        "protected_clearance_m": 5.0,
        "protected_room_roles": ["start", "goal"],
        "diagnostic_refuge_protected": False,
        "initial_pairwise_clearance_m": 0.16,
        "static_placement_margin_m": 0.06,
        "maximum_attempts_per_obstacle": 400,
    }
    json.dumps(first, sort_keys=True)


@pytest.mark.parametrize("story_id", HOSPITAL_STORY_IDS)
def test_story_templates_name_five_distinct_room_encounters(story_id: str) -> None:
    story = get_hospital_story(story_id)

    assert len(
        {
            story.start_room,
            story.goal_room,
            story.diagnostic_refuge_room,
        }
    ) == 3
    assert story.blocker_count in {2, 3}
    assert story.travel_direction * story.convoy_speed_mps < 0.0
    assert story.corridor_name in story.human_corridor_names
    assert story.narrative


@pytest.mark.parametrize("story_id", HOSPITAL_STORY_IDS)
def test_fixed_necessity_witness_has_transparent_convoy_timing(
    story_id: str,
) -> None:
    scenario = build_hospital_story_scenario(
        story_id,
        traffic_seed=0,
        human_count=0,
    )
    story = scenario.template
    predicted_coordinates = [
        blocker.predicted(
            story.necessity_witness_time_s,
            scenario.environment,
        ).coordinate
        for blocker in scenario.blockers
    ]

    nearest_at_witness = min(
        abs(coordinate - story.necessity_witness_station_m)
        for coordinate in predicted_coordinates
    )
    assert 13.0 <= nearest_at_witness <= 14.0
    assert scenario.necessity_audit.valid
    witness_lead_s = (
        scenario.necessity_audit.blockage_started_at_s
        - story.necessity_witness_time_s
    )
    assert 3.0 <= witness_lead_s <= 3.25
    assert scenario.necessity_audit.witness_station_m == (
        story.necessity_witness_station_m
    )
    assert scenario.necessity_audit.witness_time_s == (
        story.necessity_witness_time_s
    )
    for result in scenario.necessity_audit.strategies:
        assert result.swept_by_convoy
        assert not result.reached_nonroom_escape
        assert (
            story.nonroom_escape_interval_m[0]
            < result.event_coordinate_m
            < story.nonroom_escape_interval_m[1]
        )


@pytest.mark.parametrize("story_id", HOSPITAL_STORY_IDS)
def test_only_humans_change_with_traffic_seed(story_id: str) -> None:
    first = build_hospital_story_scenario(story_id, traffic_seed=3)
    repeated = build_hospital_story_scenario(story_id, traffic_seed=3)
    changed = build_hospital_story_scenario(story_id, traffic_seed=4)

    assert _obstacle_signature(first) == _obstacle_signature(repeated)
    assert _obstacle_signature(first) != _obstacle_signature(changed)
    assert _blocker_signature(first) == _blocker_signature(changed)
    assert np.array_equal(first.initial_state, changed.initial_state)
    assert np.array_equal(first.goal, changed.goal)
    assert all(isinstance(obstacle, Human) for obstacle in first.humans)
    assert all(isinstance(obstacle, Stretcher) for obstacle in first.blockers)
    assert all(not blocker.reflect_at_route_bounds for blocker in first.blockers)
    assert not any(
        isinstance(obstacle, Stretcher)
        and not obstacle.identifier.startswith("blocking-stretcher-")
        for obstacle in first.obstacles
    )
    assert len(first.humans) == PUBLICATION_HUMAN_COUNT
    assert first.crowd_metadata.ordinary_stretcher_count == 0


def test_every_default_world_satisfies_the_method_independent_contract() -> None:
    minimum_pair_clearance = float("inf")
    for trial in hospital_publication_trial_grid():
        scenario = build_hospital_story_scenario(
            trial.story_id,
            traffic_seed=trial.traffic_seed,
        )
        report = scenario.contract
        report.require_valid()
        assert report.valid
        assert report.blockade_covers_full_cross_section
        assert report.convoy_is_nonreflecting
        assert report.convoy_approaches_ego
        assert report.convoy_overtakes_bounded_corridor_motion
        assert report.convoy_eventually_clears
        assert report.diagnostic_refuge_is_reachable
        assert report.stop_is_swept_before_nonroom_escape
        assert report.continue_is_swept_before_nonroom_escape
        assert report.max_retreat_is_swept_before_nonroom_escape
        assert scenario.necessity_audit.valid
        assert scenario.necessity_audit.diagnostic_room_route_is_reachable
        assert scenario.necessity_audit.nonroom_escape_stations_are_verified
        assert scenario.necessity_audit.blockage_interval_is_contiguous
        assert (
            0.0
            <= scenario.contract.blockage_started_at_s
            < scenario.contract.blockage_cleared_at_s
            < scenario.contract.convoy_clear_time_s
        )
        assert all(
            result.swept_by_convoy
            and not result.reached_nonroom_escape
            for result in scenario.necessity_audit.strategies
        )
        assert report.initial_operational_clearance_m > 0.0
        assert report.minimum_obstacle_pair_clearance_m is not None
        minimum_pair_clearance = min(
            minimum_pair_clearance,
            report.minimum_obstacle_pair_clearance_m,
        )
    assert minimum_pair_clearance >= 0.16 - 1e-12


def test_diagnostic_refuge_witness_is_excluded_from_simulation_inputs() -> None:
    scenario = build_hospital_story_scenario(
        "main_eastbound",
        traffic_seed=0,
        human_count=6,
    )
    diagnostic = scenario.diagnostic_metadata()
    benchmark_metadata = scenario.benchmark_metadata()
    simulation = scenario.to_simulation(DEFAULT_CONFIG)

    assert diagnostic == {
        "diagnostic_only": True,
        "diagnostic_refuge_room": "Ward 44",
        "used_as_controller_input": False,
        "used_for_policy_ranking": False,
        "used_for_success_classification": False,
    }
    assert "diagnostic_refuge_room" not in benchmark_metadata
    assert "Ward 44" not in json.dumps(benchmark_metadata, sort_keys=True)
    assert benchmark_metadata["world_sha256"] == scenario.world_sha256
    assert benchmark_metadata["hospital_world_sha256"] == scenario.world_sha256
    assert (
        benchmark_metadata["blockage_started_at_s"]
        == scenario.contract.blockage_started_at_s
    )
    assert (
        benchmark_metadata["blockage_cleared_at_s"]
        == scenario.contract.blockage_cleared_at_s
    )
    assert benchmark_metadata["convoy_clear_time_s"] > 0.0
    assert benchmark_metadata["open_loop_room_necessity_verified"] is True
    # The generator protects the ego/goal plus four door-route points for
    # each of the start and goal rooms.  The diagnostic refuge contributes no
    # protected point and therefore cannot bias randomized human placement.
    assert scenario.crowd_metadata.protected_point_count == 10
    assert simulation.benchmark_scenario_metrics == benchmark_metadata
    assert not hasattr(simulation, "diagnostic_refuge_room")
    assert not hasattr(simulation.controller, "diagnostic_refuge_room")
    assert np.array_equal(simulation.state, scenario.initial_state)
    assert np.array_equal(simulation.goal, scenario.goal)
    assert len(simulation.obstacles) == len(scenario.obstacles)


def test_trial_resolution_rejects_nonpublication_cases() -> None:
    with pytest.raises(ValueError, match="unknown hospital story"):
        get_hospital_story("invented")
    for invalid_seed in (-1, 20, 1.5, True):
        with pytest.raises(ValueError, match="traffic_seed"):
            get_hospital_publication_trial(
                "main_eastbound",
                invalid_seed,
            )


def test_templates_are_serializable_provenance_not_runtime_state() -> None:
    payloads = [asdict(story) for story in HOSPITAL_STORIES]
    assert {payload["story_id"] for payload in payloads} == set(
        HOSPITAL_STORY_IDS
    )
    json.dumps(payloads, sort_keys=True)


@pytest.mark.parametrize(
    ("story_id", "expected_sha256"),
    tuple(EXPECTED_SEED_ZERO_WORLD_SHA256.items()),
)
def test_seed_zero_world_digest_is_pinned(
    story_id: str,
    expected_sha256: str,
) -> None:
    scenario = build_hospital_story_scenario(story_id, traffic_seed=0)
    assert scenario.world_sha256 == expected_sha256
    assert hospital_story_world_sha256(
        scenario.initial_state,
        scenario.goal,
        scenario.obstacles,
    ) == expected_sha256


def test_world_digest_changes_with_any_traffic_world_change() -> None:
    first = build_hospital_story_scenario("main_eastbound", traffic_seed=0)
    changed_seed = build_hospital_story_scenario(
        "main_eastbound",
        traffic_seed=1,
    )
    changed_goal = first.goal.copy()
    changed_goal[0] += 0.125

    assert first.world_sha256 != changed_seed.world_sha256
    assert first.world_sha256 != hospital_story_world_sha256(
        first.initial_state,
        changed_goal,
        first.obstacles,
    )
    assert first.world_sha256 != hospital_story_world_sha256(
        first.initial_state,
        first.goal,
        tuple(reversed(first.obstacles)),
    )
