from __future__ import annotations

from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
import os

import numpy as np
import pytest

from examples.hospital import scenarios as hospital_scenarios
from examples.hospital.config import DEFAULT_CONFIG
from examples.hospital.feasibility import (
    DYNAMIC_FEASIBILITY_SCHEMA,
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
)
from examples.hospital.obstacles import Human, Stretcher
from examples.hospital.scenarios import (
    DEFAULT_HOSPITAL_TRAFFIC_SEEDS,
    HOSPITAL_ACCEPTED_ATTEMPT_MANIFEST_SCHEMA,
    HOSPITAL_ACCEPTED_GENERATION_ATTEMPTS,
    HOSPITAL_STORIES,
    HOSPITAL_STORY_IDS,
    HOSPITAL_STORY_PROTOCOL_VERSION,
    MAX_DYNAMIC_FEASIBILITY_GENERATION_ATTEMPTS,
    PUBLICATION_HUMAN_COUNT,
    build_hospital_story_scenario,
    get_hospital_publication_trial,
    get_hospital_story,
    hospital_publication_trial_grid,
    hospital_geometry_sha256,
    hospital_accepted_generation_attempt_manifest,
    hospital_accepted_generation_attempt_manifest_sha256,
    hospital_story_protocol_metadata,
    hospital_story_world_sha256,
)


EXPECTED_PROTOCOL_SHA256 = (
    "5aa5a772d901f682226f4eb597f89da125a3909108bfa6d2435a3b99d6529021"
)
EXPECTED_MANIFEST_SHA256 = (
    "b9f6bf2cf4653343e3a4af1d7980edaab45056eb7a5a90011621ccc0f1cb696b"
)
EXPECTED_GEOMETRY_SHA256 = (
    "202a75d9bb05381249ed1fc12d96f5ca0020c1b45029434c12b2f305afa15923"
)
EXPECTED_SEED_ZERO_WORLD_SHA256 = {
    "main_eastbound": (
        "fef6f32b2eeb6f64bb21fa329c2735338be8d7ea417d75a0ee881e895da21394"
    ),
    "main_westbound": (
        "502adf396ed05bed3ef7ee69c69739dea50bce3c5d6314136687e4594f6e114a"
    ),
    "north_eastbound": (
        "3ccd25fdd7e2afb831de1ca5faac872689733a456c814b1696a22daeb86649c7"
    ),
    "north_westbound": (
        "5ae1137e1c08f48a7c8ebbaf111301898082e8738a1cbceba738151760a4e25e"
    ),
    "south_eastbound": (
        "806bc720230f4d8060e85b2469c79c3c952a4c08a9e2e80a695929b22a23fa1b"
    ),
}
EXPECTED_ACCEPTED_GENERATION_ATTEMPTS = (
    (2, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),
    (1, 2, 1, 0, 0, 0, 2, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0),
    (34, 39, 54, 6, 42, 1, 35, 7, 56, 29, 29, 33, 4, 5, 13, 23, 28, 140, 42, 38),
    (6, 15, 5, 3, 42, 15, 5, 2, 12, 7, 8, 1, 9, 0, 4, 0, 10, 2, 2, 3),
    (21, 4, 7, 25, 0, 7, 1, 8, 35, 6, 11, 21, 15, 5, 36, 5, 31, 21, 35, 20),
)


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


def _assert_valid_dynamic_refuge_audit(audit, *, blocker_only: bool) -> None:
    audit.require_valid()
    assert audit.schema == DYNAMIC_FEASIBILITY_SCHEMA
    assert audit.valid
    assert audit.blocker_only is blocker_only
    assert audit.candidate_room_count >= 1
    assert audit.witness_room_label
    assert audit.room_entry_time_s is not None
    assert audit.room_blockage_started_at_s is not None
    assert audit.room_entry_lead_margin_s is not None
    assert audit.room_exit_time_s is not None
    assert audit.goal_reached_time_s is not None
    assert audit.post_convoy_departure_delay_s in (
        POST_CONVOY_DEPARTURE_DELAYS_S
    )
    assert audit.minimum_physical_clearance_m is not None
    assert audit.minimum_operational_clearance_m is not None
    assert audit.maximum_speed_mps is not None
    assert audit.maximum_control_component_mps2 is not None
    assert audit.trajectory_sha256 is not None
    assert len(audit.trajectory_sha256) == 64
    assert audit.failure_reason is None
    assert audit.room_entry_lead_margin_s >= MINIMUM_ROOM_ENTRY_LEAD_S
    assert audit.room_entry_time_s < audit.room_blockage_started_at_s
    assert audit.room_exit_time_s >= (
        audit.convoy_cleared_at_s + POST_CONVOY_HOLD_BUFFER_S
    )
    assert audit.goal_reached_time_s <= MAXIMUM_WITNESS_TIME_S
    assert audit.minimum_physical_clearance_m > 0.0
    assert (
        audit.minimum_operational_clearance_m
        >= MINIMUM_OPERATIONAL_CLEARANCE_M
    )
    assert audit.maximum_speed_mps <= DEFAULT_CONFIG.robot.v_max + 1e-12
    assert (
        audit.maximum_control_component_mps2
        <= DEFAULT_CONFIG.robot.a_max + 1e-12
    )


@lru_cache(maxsize=len(HOSPITAL_STORY_IDS))
def _seed_zero_publication_scenario(story_id: str):
    return build_hospital_story_scenario(story_id, traffic_seed=0)


def test_publication_grid_is_exact_stable_five_by_twenty() -> None:
    grid = hospital_publication_trial_grid()

    assert HOSPITAL_STORY_PROTOCOL_VERSION == "hospital_fixed_refuge_v4"
    assert len(HOSPITAL_STORIES) == len(HOSPITAL_STORY_IDS) == 5
    assert DEFAULT_HOSPITAL_TRAFFIC_SEEDS == tuple(range(20))
    assert len(grid) == 100
    assert tuple(trial.ordinal for trial in grid) == tuple(range(100))
    assert len({trial.case_id for trial in grid}) == 100
    assert len({trial.generator_seed for trial in grid}) == 100
    assert HOSPITAL_ACCEPTED_GENERATION_ATTEMPTS == (
        EXPECTED_ACCEPTED_GENERATION_ATTEMPTS
    )
    assert grid[0].case_id == "main_eastbound/seed-0"
    assert grid[-1].case_id == "south_eastbound/seed-19"
    for story_index, story_id in enumerate(HOSPITAL_STORY_IDS):
        trials = [trial for trial in grid if trial.story_id == story_id]
        assert len(trials) == 20
        assert {trial.traffic_seed for trial in trials} == set(range(20))
        assert {trial.story_index for trial in trials} == {story_index}
        assert tuple(trial.generation_attempt for trial in trials) == (
            EXPECTED_ACCEPTED_GENERATION_ATTEMPTS[story_index]
        )
        for trial in trials:
            assert get_hospital_publication_trial(
                story_id,
                trial.traffic_seed,
            ) == trial


def test_accepted_attempt_manifest_is_canonical_and_sha256_pinned() -> None:
    manifest = hospital_accepted_generation_attempt_manifest()
    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert manifest == {
        "schema": HOSPITAL_ACCEPTED_ATTEMPT_MANIFEST_SCHEMA,
        "story_order": list(HOSPITAL_STORY_IDS),
        "traffic_seeds": list(DEFAULT_HOSPITAL_TRAFFIC_SEEDS),
        "attempts": [
            list(attempts)
            for attempts in EXPECTED_ACCEPTED_GENERATION_ATTEMPTS
        ],
    }
    assert len(manifest["attempts"]) == 5
    assert all(len(row) == 20 for row in manifest["attempts"])
    assert all(
        0 <= attempt < MAX_DYNAMIC_FEASIBILITY_GENERATION_ATTEMPTS
        for row in manifest["attempts"]
        for attempt in row
    )
    actual_sha256 = hospital_accepted_generation_attempt_manifest_sha256()
    assert actual_sha256 == EXPECTED_MANIFEST_SHA256
    assert actual_sha256 == hashlib.sha256(encoded).hexdigest()


def test_strict_publication_build_generates_only_the_pinned_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = get_hospital_publication_trial("main_eastbound", 0)
    generated_seeds: list[int] = []
    original = hospital_scenarios.generate_hospital_crowd

    def record_generation(*args, **kwargs):
        generated_seeds.append(int(kwargs["seed"]))
        return original(*args, **kwargs)

    hospital_scenarios._cached_canonical_hospital_story_scenario.cache_clear()
    monkeypatch.setattr(
        hospital_scenarios,
        "generate_hospital_crowd",
        record_generation,
    )
    scenario = build_hospital_story_scenario(
        "main_eastbound",
        traffic_seed=0,
    )

    assert scenario.trial == expected
    assert generated_seeds == [expected.generator_seed]
    assert scenario.feasibility_audit.valid


def test_invalid_pinned_publication_world_fails_without_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = build_hospital_story_scenario(
        "main_eastbound",
        traffic_seed=0,
    )
    expected = reference.trial
    generated_seeds: list[int] = []
    original_generation = hospital_scenarios.generate_hospital_crowd

    def record_generation(*args, **kwargs):
        generated_seeds.append(int(kwargs["seed"]))
        return original_generation(*args, **kwargs)

    def reject_dense_world(*args, **kwargs):
        if kwargs["humans"]:
            return replace(
                reference.feasibility_audit,
                valid=False,
                failure_reason="forced pinned-world integrity failure",
            )
        return reference.blocker_only_feasibility_audit

    hospital_scenarios._cached_canonical_hospital_story_scenario.cache_clear()
    monkeypatch.setattr(
        hospital_scenarios,
        "generate_hospital_crowd",
        record_generation,
    )
    monkeypatch.setattr(
        hospital_scenarios,
        "validate_dynamic_refuge_feasibility",
        reject_dense_world,
    )

    with pytest.raises(RuntimeError, match="never re-searches"):
        build_hospital_story_scenario("main_eastbound", traffic_seed=0)
    assert generated_seeds == [expected.generator_seed]


def test_canonical_cache_is_bounded_and_respects_construction_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = build_hospital_story_scenario(
        "main_eastbound",
        traffic_seed=0,
    )
    before = (
        hospital_scenarios._cached_canonical_hospital_story_scenario.cache_info()
    )
    repeated = build_hospital_story_scenario(
        "main_eastbound",
        traffic_seed=0,
        config=replace(
            DEFAULT_CONFIG,
            policies=replace(DEFAULT_CONFIG.policies, cbf_alpha=0.91),
        ),
    )
    after = (
        hospital_scenarios._cached_canonical_hospital_story_scenario.cache_info()
    )

    assert before.maxsize == after.maxsize == 100
    assert after.hits == before.hits + 1
    assert repeated.world_sha256 == canonical.world_sha256
    assert repeated.config.policies.cbf_alpha == 0.91

    construction_config = replace(DEFAULT_CONFIG, dt=0.05)
    calls: list[object] = []
    original_uncached = hospital_scenarios._build_hospital_story_scenario_uncached

    def record_uncached(*args, **kwargs):
        calls.append(kwargs["config"])
        return replace(canonical, config=kwargs["config"])

    monkeypatch.setattr(
        hospital_scenarios,
        "_build_hospital_story_scenario_uncached",
        record_uncached,
    )
    custom = build_hospital_story_scenario(
        "main_eastbound",
        traffic_seed=0,
        config=construction_config,
    )
    monkeypatch.setattr(
        hospital_scenarios,
        "_build_hospital_story_scenario_uncached",
        original_uncached,
    )

    assert calls == [construction_config]
    assert custom.config == construction_config


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
    assert first["accepted_generation_attempt_manifest"] == (
        hospital_accepted_generation_attempt_manifest()
    )
    assert first["accepted_generation_attempt_manifest_sha256"] == (
        hospital_accepted_generation_attempt_manifest_sha256()
    )
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
    assert first["dynamic_feasibility_conditioning"] == {
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
        "minimum_operational_clearance_m": MINIMUM_OPERATIONAL_CLEARANCE_M,
        "dynamic_substeps_per_plant_step": WITNESS_DYNAMIC_SUBSTEPS,
        "maximum_witness_time_s": MAXIMUM_WITNESS_TIME_S,
        "goal_tolerance_m": GOAL_TOLERANCE_M,
        "witness_human_caution_clearance_m": (
            WITNESS_HUMAN_CAUTION_CLEARANCE_M
        ),
        "witness_preferred_clearance_m": WITNESS_PREFERRED_CLEARANCE_M,
        "nominal_jit_block_steps": WITNESS_NOMINAL_BLOCK_STEPS,
        "exact_synchronized_dynamic_replay": True,
        "clearance_approximation_used": False,
        "post_convoy_departure_delays_s": list(
            POST_CONVOY_DEPARTURE_DELAYS_S
        ),
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
    # This test isolates the traffic generator.  The strict 50-human
    # publication admission audit is covered separately below.
    first = build_hospital_story_scenario(
        story_id,
        traffic_seed=3,
        human_count=6,
    )
    repeated = build_hospital_story_scenario(
        story_id,
        traffic_seed=3,
        human_count=6,
    )
    changed = build_hospital_story_scenario(
        story_id,
        traffic_seed=4,
        human_count=6,
    )

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
    assert len(first.humans) == 6
    assert first.crowd_metadata.ordinary_stretcher_count == 0


@pytest.mark.parametrize("story_id", HOSPITAL_STORY_IDS)
def test_every_fixed_story_has_a_dynamically_feasible_blocker_only_witness(
    story_id: str,
) -> None:
    scenario = build_hospital_story_scenario(
        story_id,
        traffic_seed=0,
        human_count=0,
    )

    _assert_valid_dynamic_refuge_audit(
        scenario.blocker_only_feasibility_audit,
        blocker_only=True,
    )
    _assert_valid_dynamic_refuge_audit(
        scenario.feasibility_audit,
        blocker_only=True,
    )


def test_dense_publication_world_has_a_dynamically_feasible_refuge_witness(
) -> None:
    scenario = _seed_zero_publication_scenario("main_eastbound")

    assert len(scenario.humans) == PUBLICATION_HUMAN_COUNT
    assert 0 <= scenario.trial.generation_attempt < (
        MAX_DYNAMIC_FEASIBILITY_GENERATION_ATTEMPTS
    )
    _assert_valid_dynamic_refuge_audit(
        scenario.blocker_only_feasibility_audit,
        blocker_only=True,
    )
    _assert_valid_dynamic_refuge_audit(
        scenario.feasibility_audit,
        blocker_only=False,
    )


@pytest.mark.skipif(
    os.environ.get("PLCBF_RUN_FULL_HOSPITAL_PROTOCOL_AUDIT") != "1",
    reason=(
        "set PLCBF_RUN_FULL_HOSPITAL_PROTOCOL_AUDIT=1 to replay all 100 "
        "feasibility-conditioned publication worlds"
    ),
)
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
        _assert_valid_dynamic_refuge_audit(
            scenario.blocker_only_feasibility_audit,
            blocker_only=True,
        )
        _assert_valid_dynamic_refuge_audit(
            scenario.feasibility_audit,
            blocker_only=False,
        )
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


def test_diagnostic_refuge_witness_is_excluded_from_simulation_inputs(
) -> None:
    scenario = _seed_zero_publication_scenario("main_eastbound")
    diagnostic = scenario.diagnostic_metadata()
    benchmark_metadata = scenario.benchmark_metadata()
    simulation = scenario.to_simulation(DEFAULT_CONFIG)

    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["diagnostic_refuge_room"] == "Ward 44"
    assert diagnostic["used_as_controller_input"] is False
    assert diagnostic["used_for_policy_ranking"] is False
    assert diagnostic["used_for_success_classification"] is False
    assert (
        diagnostic["dynamic_witness_room"]
        == scenario.feasibility_audit.witness_room_label
    )
    assert diagnostic["dynamic_witness_used_as_controller_input"] is False
    assert "diagnostic_refuge_room" not in benchmark_metadata
    assert "Ward 44" not in json.dumps(benchmark_metadata, sort_keys=True)
    assert "dynamic_witness_room" not in benchmark_metadata
    assert "witness_room_label" not in benchmark_metadata
    assert "witness_control" not in benchmark_metadata
    assert "control_sequence" not in benchmark_metadata
    assert (
        scenario.feasibility_audit.witness_room_label
        not in json.dumps(benchmark_metadata, sort_keys=True)
    )
    assert (
        benchmark_metadata["dynamic_refuge_witness_is_controller_input"]
        is False
    )
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
    scenario = _seed_zero_publication_scenario(story_id)
    assert scenario.world_sha256 == expected_sha256
    assert hospital_story_world_sha256(
        scenario.initial_state,
        scenario.goal,
        scenario.obstacles,
    ) == expected_sha256


def test_dense_generation_attempt_and_world_hash_are_deterministic() -> None:
    first = _seed_zero_publication_scenario("main_eastbound")
    repeated = build_hospital_story_scenario(
        "main_eastbound",
        traffic_seed=0,
    )

    assert repeated.trial == first.trial
    assert repeated.trial.generation_attempt == first.trial.generation_attempt
    assert repeated.trial.generator_seed == first.trial.generator_seed
    assert repeated.world_sha256 == first.world_sha256
    assert (
        repeated.feasibility_audit.trajectory_sha256
        == first.feasibility_audit.trajectory_sha256
    )
    assert _obstacle_signature(repeated) == _obstacle_signature(first)


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
