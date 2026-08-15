from __future__ import annotations

from dataclasses import fields
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from examples.hospital import run
from examples.hospital import controller as controller_module
from examples.hospital.config import DEFAULT_CONFIG, RefugeConfig
from examples.hospital.controller import (
    ControllerResult,
    HospitalController,
    RoomPolicyProvider,
    sensed_obstacles,
    static_hocbf_constraints,
)
from examples.hospital.dynamics import (
    step_double_integrator,
    waypoint_control,
)
from examples.hospital.obstacles import Human
from examples.hospital.policies import rollout_policy
from examples.hospital.simulation import build_blocked_main_hall_scenario
from plcbf.policy_library import CBFHalfspace, PolicyCertificate


def test_controller_has_no_latched_refuge_executor() -> None:
    simulation = build_blocked_main_hall_scenario(3)
    controller = simulation.controller

    assert isinstance(controller.room_policy_provider, RoomPolicyProvider)
    assert not hasattr(controller, "refuge")
    source = inspect.getsource(HospitalController)
    assert "commit_room_policy" not in source
    assert "minimum_hold_time" not in source
    assert "guard_required_steps" not in source
    assert "RefugeState" not in source
    result_fields = {item.name for item in fields(ControllerResult)}
    assert "inside_refuge" in result_fields
    assert "refuge_state" not in result_fields

    policies = controller.candidate_policies(simulation.state)
    kinds = {policy.kind for policy in policies}
    assert {"nominal", "angle", "reverse", "stop", "room"} <= kinds
    # Like the playground, directional branches whose preview segment crosses
    # static geometry are not put into the runtime library.
    assert [policy.name for policy in policies if policy.kind == "angle"] == [
        f"angle_{index}" for index in range(3, 10)
    ]
    assert len(policies) == 13
    assert not hasattr(controller, "_cached_certificates")
    assert "certificate_update_period" not in source


def test_default_policy_library_uses_one_certificate_horizon() -> None:
    simulation = build_blocked_main_hall_scenario(3)
    policies = simulation.controller.candidate_policies(simulation.state)

    assert {policy.horizon for policy in policies} == {
        simulation.config.policies.room_horizon
    }


def test_room_rollout_cursor_reaches_and_holds_the_terminal_room() -> None:
    simulation = build_blocked_main_hall_scenario(3)
    room_policy = next(
        policy
        for policy in simulation.controller.candidate_policies(
            simulation.state
        )
        if policy.kind == "room" and policy.target_room.label == "Ward 44"
    )

    trajectory = rollout_policy(
        room_policy,
        simulation.state,
        simulation.config,
    )
    terminal = trajectory[-1]

    assert len(trajectory) == int(
        room_policy.horizon / room_policy.rollout_dt
    ) + 1
    assert room_policy.target_room.interior_margin(terminal[:2]) >= (
        simulation.config.refuge.terminal_interior_margin
    )
    assert np.linalg.norm(terminal[2:]) <= (
        simulation.config.refuge.terminal_speed_max
    )


def test_complete_library_is_never_gated_by_obstacle_or_phase() -> None:
    simulation = build_blocked_main_hall_scenario(2)
    controller = simulation.controller
    expected = controller.candidate_policies(simulation.state)
    active = controller._active_policy_candidates(
        simulation.state,
        simulation.obstacles,
    )
    assert [policy.name for policy in active] == [
        policy.name for policy in expected
    ]


def test_no_sensed_hazard_uses_unconstrained_nominal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = build_blocked_main_hall_scenario(2)

    def unexpected_selection(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("policy QP must be inactive without a sensed hazard")

    monkeypatch.setattr(
        controller_module,
        "select_policy",
        unexpected_selection,
    )
    result = simulation.controller.compute(simulation.state, [], 0.0)
    assert not result.hocbf_constraints
    assert result.selected_policy == "nominal"
    np.testing.assert_allclose(result.control, result.nominal_control)


def test_tiny_input_area_uses_max_value_backup_emergency_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = build_blocked_main_hall_scenario(2)
    state = simulation.state.copy()
    strip = (
        CBFHalfspace(np.array([1.0, 0.0]), 0.0, "lower"),
        CBFHalfspace(np.array([-1.0, 0.0]), -1e-5, "upper"),
    )
    first = PolicyCertificate(
        "candidate_a",
        value=0.5,
        halfspaces=strip,
        backup_control=np.array([0.3, -0.2]),
    )
    safest = PolicyCertificate(
        "candidate_b",
        value=0.8,
        halfspaces=strip,
        backup_control=np.array([-0.4, 0.2]),
    )

    monkeypatch.setattr(
        simulation.controller,
        "build_policy_certificates",
        lambda *_args, **_kwargs: ((first, safest), ()),
    )
    hocbf = controller_module.HocbfConstraint(
        a=np.array([1.0, 0.0]),
        b=0.2,
        label="hocbf:test",
        obstacle_id="test",
        proxy_index=0,
        h=1.0,
        h_dot=0.0,
        psi1=1.0,
        safe_distance=1.0,
    )
    monkeypatch.setattr(
        controller_module,
        "current_hocbf_constraints",
        lambda *_args, **_kwargs: [hocbf],
    )
    sensed_but_inactive = Human(
        "sensed-far",
        x=float(state[0] + 10.0),
        y=float(state[1]),
        vx=0.0,
        vy=0.0,
    )
    result = simulation.controller.compute(
        state,
        (sensed_but_inactive,),
        0.0,
    )

    assert result.decision.diagnostics.used_fallback is True
    assert (
        result.decision.diagnostics.fallback_reason
        == "no_positive_input_volume_policy"
    )
    assert result.selected_policy == "candidate_b"
    np.testing.assert_allclose(result.control, np.array([0.2, 0.2]))
    assert hocbf.margin(result.control) >= -1e-9


def test_playground_input_volume_ties_keep_policy_library_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = build_blocked_main_hall_scenario(2)
    state = simulation.state.copy()
    first = PolicyCertificate(
        "z_first_in_library",
        value=0.8,
        backup_control=np.array([0.1, 0.0]),
    )
    second = PolicyCertificate(
        "a_second_in_library",
        value=0.8,
        backup_control=np.array([-0.1, 0.0]),
    )
    monkeypatch.setattr(
        simulation.controller,
        "build_policy_certificates",
        lambda *_args, **_kwargs: ((first, second), ()),
    )
    monkeypatch.setattr(
        controller_module,
        "current_hocbf_constraints",
        lambda *_args, **_kwargs: [],
    )
    sensed = Human(
        "selection-trigger",
        x=float(state[0] + 3.0),
        y=float(state[1]),
        vx=0.0,
        vy=0.0,
    )

    result = simulation.controller.compute(
        state,
        (sensed,),
        0.0,
    )

    assert result.decision.diagnostics.used_fallback is False
    assert result.selected_policy == "z_first_in_library"
    assert result.decision.policy_id == "z_first_in_library"


def test_current_room_policy_is_omitted_without_changing_nominal_route() -> None:
    simulation = build_blocked_main_hall_scenario(2)
    nurse = next(
        room for room in simulation.environment.rooms if room.label == "Nurse"
    )
    _, _, _, terminal_center = simulation.environment.room_door_path(
        nurse,
        simulation.config.robot.radius,
        simulation.config.refuge.inside_door_offset,
        simulation.config.refuge.outside_door_offset,
    )
    state = np.r_[terminal_center, np.zeros(2)]
    controller = HospitalController(
        simulation.environment,
        simulation.planner,
        simulation.config,
        state,
        simulation.goal,
    )
    policies = controller.candidate_policies(state)
    assert policies[0].name == "nominal"
    assert all(policy.target_room is not nurse for policy in policies)
    assert {"reverse", "stop"} <= {
        policy.kind for policy in policies
    }


def test_nominal_route_from_room_uses_playground_door_centerline() -> None:
    simulation = build_blocked_main_hall_scenario(2)
    nurse = next(
        room for room in simulation.environment.rooms if room.label == "Nurse"
    )
    state = np.r_[nurse.center, np.zeros(2)]
    controller = HospitalController(
        simulation.environment,
        simulation.planner,
        simulation.config,
        state,
        simulation.goal,
    )
    center, inside, door, outside = controller._room_exit_route(nurse)

    np.testing.assert_allclose(controller.navigation_path[:4], [
        state[:2],
        inside,
        door,
        outside,
    ])
    np.testing.assert_allclose(center, nurse.center)
    route_radius = simulation.config.robot.radius + 0.08
    assert all(
        simulation.environment.segment_is_free(
            start,
            end,
            route_radius,
            step=0.55,
        )
        for start, end in zip(
            controller.navigation_path[:3],
            controller.navigation_path[1:4],
            strict=True,
        )
    )
    # The centerline points prevent a diagonal shortcut through the wall.
    assert nurse.contains(inside)
    assert np.linalg.norm(door - nurse.door.center) < 1e-9
    assert not nurse.contains(outside, margin=-0.05)


def test_unscheduled_room_entry_repairs_stale_nominal_route_once() -> None:
    simulation = build_blocked_main_hall_scenario(2)
    controller = simulation.controller
    nurse = next(
        room for room in simulation.environment.rooms if room.label == "Nurse"
    )
    diverted_position = nurse.center + np.array([0.8, 0.0])
    diverted_state = np.r_[diverted_position, np.zeros(2)]
    stale_path = [point.copy() for point in controller.navigation_path]
    stale_remaining = [
        point.copy()
        for point in stale_path[controller.navigation_index :]
    ]

    assert controller._ensure_room_exit_waypoints(diverted_state)
    assert not all(
        np.array_equal(left, right)
        for left, right in zip(
            controller.navigation_path,
            stale_path,
            strict=False,
        )
    )
    center, inside, door, outside = controller._room_exit_route(nurse)
    np.testing.assert_allclose(
        controller.navigation_path[:5],
        [diverted_state[:2], center, inside, door, outside],
    )
    # The geometric insertion rejoins and preserves the previously unvisited
    # nominal suffix rather than discarding it in a fresh goal replan.
    suffix_start = next(
        index
        for index, point in enumerate(controller.navigation_path)
        if np.linalg.norm(point - stale_remaining[0]) <= 0.35
    )
    np.testing.assert_allclose(
        controller.navigation_path[
            suffix_start : suffix_start + len(stale_remaining)
        ],
        stale_remaining,
    )
    path_after_repair = [point.copy() for point in controller.navigation_path]
    assert not controller._ensure_room_exit_waypoints(diverted_state)
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            controller.navigation_path,
            path_after_repair,
            strict=True,
        )
    )


def test_repaired_nominal_route_exits_room_and_resumes_preserved_suffix() -> None:
    simulation = build_blocked_main_hall_scenario(2)
    controller = simulation.controller
    nurse = next(
        room for room in simulation.environment.rooms if room.label == "Nurse"
    )
    state = np.r_[nurse.center + np.array([2.0, 3.0]), np.zeros(2)]
    stale_target = controller.navigation_path[controller.navigation_index].copy()

    assert controller._ensure_room_exit_waypoints(state)
    np.testing.assert_allclose(controller._navigation_target(state), nurse.center)

    exited = False
    for _ in range(250):
        target = controller._navigation_target(state)
        nominal = waypoint_control(
            state,
            target,
            simulation.config.robot,
            simulation.config.policies.nominal_target_speed,
        )
        state = step_double_integrator(
            state,
            nominal,
            simulation.config.dt,
            simulation.config.robot,
        )
        exited |= not nurse.contains(state[:2])

    assert exited
    assert any(
        np.linalg.norm(point - stale_target) <= 0.35
        for point in controller.navigation_path
    )
    assert controller.navigation_index >= 5


def test_goal_room_does_not_trigger_exit_route_repair() -> None:
    simulation = build_blocked_main_hall_scenario(2)
    nurse = next(
        room for room in simulation.environment.rooms if room.label == "Nurse"
    )
    state = np.r_[nurse.center, np.zeros(2)]
    controller = HospitalController(
        simulation.environment,
        simulation.planner,
        simulation.config,
        state,
        nurse.center,
    )
    original_path = [point.copy() for point in controller.navigation_path]

    assert not controller._ensure_room_exit_waypoints(state)
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            controller.navigation_path,
            original_path,
            strict=True,
        )
    )


def test_room_exit_repair_has_only_geometry_state_input() -> None:
    signature = inspect.signature(HospitalController._ensure_room_exit_waypoints)
    assert tuple(signature.parameters) == ("self", "state")
    source = inspect.getsource(HospitalController._ensure_room_exit_waypoints)
    for forbidden in (
        "obstacles",
        "blocker",
        "selected_policy",
        "time_seconds",
        "hold_time",
        "release_guard",
    ):
        assert forbidden not in source


def test_jax_warmup_preserves_navigation_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation = build_blocked_main_hall_scenario(2)
    controller = simulation.controller
    nurse = next(
        room for room in simulation.environment.rooms if room.label == "Nurse"
    )
    diverted_state = np.r_[nurse.center + np.array([2.0, 3.0]), np.zeros(2)]
    original_path = [point.copy() for point in controller.navigation_path]
    original_index = controller.navigation_index
    original_goal = controller.goal.copy()
    warmed_buckets: list[int] = []

    def record_warmup(
        _state,
        _nominal,
        policies,
        obstacles,
        _geometry,
        _parameters,
        *,
        include_diagnostics,
    ) -> None:
        assert policies.active_count > 0
        assert not include_diagnostics
        warmed_buckets.append(int(obstacles.active.shape[0]))

    monkeypatch.setattr(
        controller_module,
        "warmup_policy_groups",
        record_warmup,
    )
    controller.warmup_certificate_oracle(diverted_state)

    assert warmed_buckets == list(controller._jax_obstacle_buckets)
    assert controller.navigation_index == original_index
    np.testing.assert_allclose(controller.goal, original_goal)
    assert len(controller.navigation_path) == len(original_path)
    for actual, expected in zip(
        controller.navigation_path,
        original_path,
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected)


def test_dense_scene_controller_uses_only_nearest_sensed_limit() -> None:
    simulation = build_blocked_main_hall_scenario(2)
    state = simulation.state
    humans = [
        Human(
            f"human-{index}",
            x=float(state[0] + 0.8 + 0.15 * index),
            y=float(state[1]),
            vx=0.0,
            vy=0.0,
        )
        for index in range(30)
    ]
    active = sensed_obstacles(
        state,
        humans,
        simulation.config,
    )
    assert len(active) == DEFAULT_CONFIG.safety.max_obstacles == 12
    assert [item.identifier for item in active] == [
        f"human-{index}" for index in range(12)
    ]

    visible = sensed_obstacles(
        state,
        humans,
        simulation.config,
        environment=simulation.environment,
    )
    assert [item.identifier for item in visible] == ["human-0", "human-1"]


def test_room_policy_ends_at_playground_entry_not_room_center() -> None:
    simulation = build_blocked_main_hall_scenario(2)
    room_policy = next(
        policy
        for policy in simulation.controller.candidate_policies(simulation.state)
        if policy.kind == "room"
    )
    assert room_policy.target_room is not None
    _, _, entry, terminal_center = simulation.environment.room_door_path(
        room_policy.target_room,
        simulation.config.robot.radius,
        simulation.config.refuge.inside_door_offset,
        simulation.config.refuge.outside_door_offset,
    )
    np.testing.assert_allclose(room_policy.waypoints[-1], entry)
    assert np.linalg.norm(entry - terminal_center) > 1.0


def test_every_policy_certificate_includes_current_static_hocbfs() -> None:
    simulation = build_blocked_main_hall_scenario(2)
    state = np.array([57.0, 51.2, 0.0, 0.0])
    expected = {
        item.label
        for item in static_hocbf_constraints(
            state,
            simulation.environment,
            simulation.config,
        )
    }
    assert expected

    certificates, _ = simulation.controller.build_policy_certificates(
        state,
        [],
    )
    assert certificates
    for certificate in certificates:
        labels = {item.label for item in certificate.halfspaces}
        assert expected <= labels


def test_static_hocbf_matches_playground_near_wall_selection() -> None:
    simulation = build_blocked_main_hall_scenario(2)
    constraints = static_hocbf_constraints(
        np.array([57.0, 51.2, 0.0, 0.0]),
        simulation.environment,
        simulation.config,
    )
    assert constraints
    assert len(constraints) <= DEFAULT_CONFIG.safety.max_static_hocbf_constraints
    assert all(item.obstacle_id.startswith(("wall-", "floor-")) for item in constraints)


def test_static_hocbf_floor_rows_reference_true_union_boundary_segments() -> None:
    simulation = build_blocked_main_hall_scenario(2)
    # This room/corridor overlap previously emitted a fictitious row at the
    # internal edge [12, 15].
    position = np.array([12.0, 13.86956522])
    constraints = static_hocbf_constraints(
        np.r_[position, 0.0, 0.0],
        simulation.environment,
        simulation.config,
    )
    floor_rows = [
        item for item in constraints if item.obstacle_id.startswith("floor-")
    ]
    assert floor_rows
    for row in floor_rows:
        index = int(row.obstacle_id.removeprefix("floor-"))
        start = simulation.environment._floor_boundary_starts[index]
        end = simulation.environment._floor_boundary_ends[index]
        segment = end - start
        fraction = np.clip(
            np.dot(position - start, segment) / np.dot(segment, segment),
            0.0,
            1.0,
        )
        expected_closest = start + fraction * segment
        inferred_closest = position - 0.5 * row.a
        np.testing.assert_allclose(inferred_closest, expected_closest)
        assert np.linalg.norm(inferred_closest - np.array([12.0, 15.0])) > 1e-6


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("inside_door_offset", 0.0),
        ("outside_door_offset", float("nan")),
        ("terminal_interior_margin", -0.1),
        ("terminal_speed_max", float("inf")),
        ("waypoint_radius", 0.0),
    ),
)
def test_room_terminal_geometry_configuration_is_validated(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        RefugeConfig(**{field: value})


def test_cli_reports_policy_and_dense_obstacle_count(
    monkeypatch, capsys
) -> None:
    fake = SimpleNamespace(
        collision=True,
        reached_goal=False,
        time=0.06,
        state=np.array([1.0, 2.0, 0.0, 0.0]),
        last_controller=None,
        environment=SimpleNamespace(
            room_containing=lambda _point: None,
        ),
        obstacles=[object()] * 67,
        run=lambda steps: [],
    )
    monkeypatch.setattr(
        run,
        "build_benchmark_scenario",
        lambda *_args, **_kwargs: fake,
    )
    assert run.main(["--stretchers", "2", "--steps", "1"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["collision"] is True
    assert payload["inside_refuge"] is False
    assert payload["dynamic_obstacle_count"] == 67
