from __future__ import annotations

import numpy as np

from examples.hospital.simulation import build_blocked_main_hall_scenario


def test_room_labels_are_observational_not_controller_state() -> None:
    simulation = build_blocked_main_hall_scenario(2)
    result = simulation.controller.compute(
        simulation.state,
        simulation.obstacles,
        0.0,
    )
    assert result.inside_refuge is False
    assert result.decision.policy_id is not None
    assert len(result.policy_evaluations) == len(
        simulation.controller.candidate_policies(simulation.state)
    )

    # The commanded nominal remains the route tracker even if a room backup is
    # selected; it is not replaced by a phase-specific enter/hold/exit input.
    target = simulation.controller._navigation_target(simulation.state)
    assert np.linalg.norm(target - simulation.goal) < np.linalg.norm(
        simulation.state[:2] - simulation.goal
    )


def test_strict_blockage_naturally_selects_room_backup_without_mode_state() -> None:
    simulation = build_blocked_main_hall_scenario(3)
    onset_time = 0.18
    for obstacle in simulation.obstacles:
        obstacle.advance(onset_time, simulation.environment)

    result = simulation.controller.compute(
        simulation.state,
        simulation.obstacles,
        onset_time,
    )

    assert result.selected_policy.startswith("room_")
    assert result.inside_refuge is False
    assert result.decision.diagnostics.used_fallback is False
    assert result.decision.diagnostics.fallback_reason is None
    selected = next(
        evaluation
        for evaluation in result.decision.diagnostics.evaluations
        if evaluation.policy_id == result.decision.policy_id
    )
    assert selected.safe_value is True
    assert selected.feasible is True
    assert (
        selected.input_volume
        > simulation.config.policies.constraint_tolerance
    )
    safe_policy_ids = {
        evaluation.policy_id
        for evaluation in result.decision.diagnostics.evaluations
        if evaluation.safe_value
    }
    assert safe_policy_ids
    assert all(
        policy_id.startswith("room_")
        for policy_id in safe_policy_ids
    )
    state_names = set(vars(simulation.controller))
    assert {
        "environment",
        "planner",
        "config",
        "room_policy_provider",
        "goal",
        "navigation_path",
        "navigation_index",
        "_jax_geometry",
        "_jax_parameters",
        "_jax_obstacle_buckets",
    } == state_names
    for forbidden in (
        "refuge_state",
        "phase",
        "hold",
        "guard",
        "committed_room",
        "selected_room",
    ):
        assert all(forbidden not in name for name in state_names)
