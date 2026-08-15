from __future__ import annotations

from dataclasses import replace
from math import cos, sin
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.hospital import jax_rollout as backend
from examples.hospital.dynamics import step_double_integrator
from examples.hospital.jax_rollout import (
    HospitalJaxCapacities,
    HospitalJaxGroupedCapacities,
    HospitalJaxPolicyGroupSpec,
    capacities_for_obstacle_count,
    compiled_evaluator_cache_info,
    compiled_grouped_evaluator_cache_info,
    evaluate_policy_groups,
    evaluate_policy_batch,
    pack_obstacle_batch,
    pack_parameters,
    pack_policy_batch,
    pack_policy_groups,
    pack_static_geometry,
    select_obstacle_bucket,
)
from examples.hospital.obstacles import (
    Human,
    Stretcher,
    obstacle_clearance,
)
from examples.hospital.policies import (
    HospitalPolicy,
    rollout_policy,
    rollout_value,
)
from examples.hospital.simulation import build_blocked_main_hall_scenario


@pytest.fixture(scope="module")
def rollout_case():
    simulation = build_blocked_main_hall_scenario(2)
    candidates = simulation.controller.candidate_policies(simulation.state)
    policies = []
    for kind in ("nominal", "angle", "reverse", "stop", "room"):
        original = next(item for item in candidates if item.kind == kind)
        policies.append(replace(original, horizon=0.96))
    policies.append(
        HospitalPolicy(
            name="retrace",
            kind="retrace",
            horizon=0.96,
            rollout_dt=0.24,
            target_speed=1.2,
            waypoints=[
                np.array([55.0, 47.5]),
                np.array([52.0, 47.5]),
            ],
            feedback_gain=1.8,
        )
    )
    obstacles = (
        Human("stationary-human", 63.0, 47.5, 0.0, 0.0, 0.52),
        Stretcher(
            "stationary-stretcher",
            70.0,
            47.5,
            0.0,
            "x",
            4.0,
            136.0,
            4.1,
            1.45,
            False,
        ),
    )
    capacities = HospitalJaxCapacities(
        max_policies=8,
        max_obstacles=4,
        max_horizon_steps=4,
        max_swept_samples=3,
        human_prediction_steps=25,
    )
    packed_policies = pack_policy_batch(
        policies, simulation.config, capacities
    )
    packed_obstacles = pack_obstacle_batch(obstacles, capacities)
    geometry = pack_static_geometry(simulation.environment)
    parameters = pack_parameters(simulation.config)
    nominal = np.array([0.2, -0.1])
    evaluation = evaluate_policy_batch(
        simulation.state,
        nominal,
        packed_policies,
        packed_obstacles,
        geometry,
        parameters,
    )
    return {
        "simulation": simulation,
        "policies": tuple(policies),
        "obstacles": obstacles,
        "capacities": capacities,
        "packed_policies": packed_policies,
        "packed_obstacles": packed_obstacles,
        "geometry": geometry,
        "parameters": parameters,
        "nominal": nominal,
        "evaluation": evaluation,
    }


def test_pack_policy_batch_supports_complete_inventory(rollout_case) -> None:
    packed = rollout_case["packed_policies"]
    assert packed.names == (
        "nominal",
        "angle_3",
        "reverse_0",
        "stop",
        "room_0",
        "retrace",
    )
    np.testing.assert_array_equal(
        packed.batch.kinds[:6],
        np.array(
            [
                backend.POLICY_NOMINAL,
                backend.POLICY_ANGLE,
                backend.POLICY_REVERSE,
                backend.POLICY_STOP,
                backend.POLICY_ROOM,
                backend.POLICY_RETRACE,
            ]
        ),
    )
    assert packed.batch.active.tolist() == [True] * 6 + [False] * 2
    assert packed.batch.waypoints.shape == (8, 18, 2)
    assert packed.batch.swept_samples[:6].tolist() == [2, 2, 2, 2, 3, 2]


def test_batched_rollouts_and_values_match_scalar_all_policy_kinds(
    rollout_case,
) -> None:
    simulation = rollout_case["simulation"]
    evaluation = rollout_case["evaluation"]
    for index, policy in enumerate(rollout_case["policies"]):
        scalar = rollout_value(
            policy,
            simulation.state,
            rollout_case["obstacles"],
            simulation.environment,
            simulation.config,
            {},
        )
        mask = evaluation.trajectory_mask[index]
        np.testing.assert_allclose(
            evaluation.trajectories[index, mask],
            scalar.trajectory,
            rtol=2.0e-6,
            atol=8.0e-6,
        )
        assert evaluation.values[index] == pytest.approx(
            scalar.value, rel=3.0e-6, abs=8.0e-6
        )
    assert np.all(np.isfinite(evaluation.values))
    assert np.all(np.isfinite(evaluation.gradients))


def test_decision_only_evaluator_omits_host_rollouts_without_value_drift(
    rollout_case,
) -> None:
    case = rollout_case
    decision_only = evaluate_policy_batch(
        case["simulation"].state,
        case["nominal"],
        case["packed_policies"],
        case["packed_obstacles"],
        case["geometry"],
        case["parameters"],
        include_diagnostics=False,
    )

    assert decision_only.diagnostics_available is False
    assert decision_only.trajectories.shape == (
        len(case["policies"]),
        0,
        4,
    )
    assert decision_only.trajectory_mask.shape == (len(case["policies"]), 0)
    assert np.all(np.isnan(decision_only.nominal_prefix_values))
    assert np.all(np.isnan(decision_only.terminal_clearances))
    np.testing.assert_allclose(
        decision_only.values,
        case["evaluation"].values,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        decision_only.gradients,
        case["evaluation"].gradients,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        decision_only.time_derivatives,
        case["evaluation"].time_derivatives,
        atol=2.0e-6,
    )


def test_room_policy_certifies_the_complete_horizon_after_entry(
    rollout_case,
) -> None:
    case = rollout_case
    original = next(
        policy for policy in case["policies"] if policy.kind == "room"
    )
    assert original.target_room is not None
    state = np.r_[original.target_room.center, 0.0, 0.0]
    policy = replace(
        original,
        horizon=0.48,
        rollout_dt=0.24,
        waypoints=[original.target_room.center.copy()],
    )

    scalar = rollout_policy(policy, state, case["simulation"].config)
    packed = pack_policy_batch(
        (policy,), case["simulation"].config, case["capacities"]
    )
    evaluated = evaluate_policy_batch(
        state,
        case["nominal"],
        packed,
        case["packed_obstacles"],
        case["geometry"],
        case["parameters"],
    )

    assert scalar.shape == (3, 4)
    assert int(np.sum(evaluated.trajectory_mask[0])) == 3
    np.testing.assert_allclose(
        evaluated.trajectories[0, evaluated.trajectory_mask[0]],
        scalar,
        rtol=2.0e-6,
        atol=8.0e-6,
    )


def test_horizon_groups_match_unified_batch_in_original_order(
    rollout_case,
) -> None:
    case = rollout_case
    horizons = {
        "nominal": 0.48,
        "angle": 0.48,
        "room": 0.72,
        "reverse": 0.96,
        "stop": 0.96,
        "retrace": 0.96,
    }
    policies = tuple(
        replace(policy, horizon=horizons[policy.kind])
        for policy in case["policies"]
    )
    grouped_capacities = HospitalJaxGroupedCapacities(
        groups=(
            HospitalJaxPolicyGroupSpec(2, 2, 2),
            HospitalJaxPolicyGroupSpec(1, 3, 3),
            HospitalJaxPolicyGroupSpec(3, 4, 2),
        ),
        max_obstacles=4,
        human_prediction_steps=25,
    )
    grouped_policies = pack_policy_groups(
        policies, case["simulation"].config, grouped_capacities
    )
    grouped_obstacles = pack_obstacle_batch(
        case["obstacles"], grouped_capacities
    )
    grouped = evaluate_policy_groups(
        case["simulation"].state,
        case["nominal"],
        grouped_policies,
        grouped_obstacles,
        case["geometry"],
        case["parameters"],
    )
    unified_policies = pack_policy_batch(
        policies, case["simulation"].config, case["capacities"]
    )
    unified = evaluate_policy_batch(
        case["simulation"].state,
        case["nominal"],
        unified_policies,
        case["packed_obstacles"],
        case["geometry"],
        case["parameters"],
    )
    assert grouped.names == unified.names
    np.testing.assert_allclose(grouped.values, unified.values, atol=2.0e-6)
    np.testing.assert_allclose(grouped.gradients, unified.gradients, atol=2.0e-6)
    np.testing.assert_allclose(
        grouped.shifted_values, unified.shifted_values, atol=2.0e-6
    )
    np.testing.assert_allclose(
        grouped.nominal_prefix_values,
        unified.nominal_prefix_values,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        grouped.terminal_clearances,
        unified.terminal_clearances,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        grouped.trajectories, unified.trajectories, atol=2.0e-6
    )
    np.testing.assert_array_equal(
        grouped.trajectory_mask, unified.trajectory_mask
    )

    before = compiled_grouped_evaluator_cache_info()
    evaluate_policy_groups(
        case["simulation"].state,
        case["nominal"],
        grouped_policies,
        grouped_obstacles,
        case["geometry"],
        case["parameters"],
    )
    after = compiled_grouped_evaluator_cache_info()
    assert after.currsize == before.currsize
    assert after.misses == before.misses
    assert after.hits >= before.hits + 1


def test_shared_regular_and_room_time_grids_match_unified_evaluator(
    rollout_case,
) -> None:
    case = rollout_case
    policies = tuple(
        replace(
            policy,
            horizon=0.96,
            rollout_dt=(0.12 if policy.kind == "room" else 0.24),
        )
        for policy in case["policies"]
        if policy.kind != "retrace"
    )
    capacities = HospitalJaxCapacities(
        max_policies=len(policies),
        max_obstacles=4,
        max_horizon_steps=8,
        max_swept_samples=3,
        human_prediction_steps=25,
    )
    grouped_capacities = HospitalJaxGroupedCapacities(
        groups=(HospitalJaxPolicyGroupSpec(len(policies), 8, 3),),
        max_obstacles=4,
        human_prediction_steps=25,
    )
    unified = evaluate_policy_batch(
        case["simulation"].state,
        case["nominal"],
        pack_policy_batch(policies, case["simulation"].config, capacities),
        pack_obstacle_batch(case["obstacles"], capacities),
        case["geometry"],
        case["parameters"],
        include_diagnostics=False,
    )
    grouped = evaluate_policy_groups(
        case["simulation"].state,
        case["nominal"],
        pack_policy_groups(
            policies,
            case["simulation"].config,
            grouped_capacities,
        ),
        pack_obstacle_batch(case["obstacles"], grouped_capacities),
        case["geometry"],
        case["parameters"],
        include_diagnostics=False,
    )

    np.testing.assert_allclose(grouped.values, unified.values, atol=2.0e-6)
    np.testing.assert_allclose(
        grouped.gradients,
        unified.gradients,
        atol=3.0e-6,
    )
    np.testing.assert_allclose(
        grouped.time_derivatives,
        unified.time_derivatives,
        atol=6.0e-6,
    )


def test_shift_prefix_and_terminal_outputs_match_scalar(rollout_case) -> None:
    simulation = rollout_case["simulation"]
    config = simulation.config
    evaluation = rollout_case["evaluation"]
    prefix_state = step_double_integrator(
        simulation.state,
        rollout_case["nominal"],
        config.dt,
        config.robot,
    )
    for index, policy in enumerate(rollout_case["policies"]):
        shifted = rollout_value(
            policy,
            simulation.state,
            rollout_case["obstacles"],
            simulation.environment,
            config,
            {},
            time_offset=config.policies.time_derivative_step,
        )
        prefix = rollout_value(
            policy,
            prefix_state,
            rollout_case["obstacles"],
            simulation.environment,
            config,
            {},
            time_offset=config.dt,
        )
        assert evaluation.shifted_values[index] == pytest.approx(
            shifted.value, rel=3.0e-6, abs=8.0e-6
        )
        assert evaluation.nominal_prefix_values[index] == pytest.approx(
            prefix.value, rel=3.0e-6, abs=8.0e-6
        )
        expected_derivative = (
            shifted.value - evaluation.values[index]
        ) / config.policies.time_derivative_step
        assert evaluation.time_derivatives[index] == pytest.approx(
            expected_derivative, rel=2.0e-4, abs=3.0e-5
        )

        trajectory = rollout_value(
            policy,
            simulation.state,
            rollout_case["obstacles"],
            simulation.environment,
            config,
            {},
        ).trajectory
        terminal = trajectory[-1]
        terminal_time = (len(trajectory) - 1) * policy.rollout_dt
        clearances = [
            simulation.environment.static_clearance(
                terminal[:2], config.robot.radius + config.safety.static_margin
            )
        ]
        clearances.extend(
            obstacle_clearance(
                obstacle.predicted(terminal_time, simulation.environment),
                terminal[:2],
                config.robot.radius + config.safety.safety_margin,
                config.safety.human_margin,
                config.safety.stretcher_margin,
            )
            for obstacle in rollout_case["obstacles"]
        )
        assert evaluation.terminal_clearances[index] == pytest.approx(
            min(clearances), rel=3.0e-6, abs=8.0e-6
        )


def test_exact_autodiff_gradient_matches_value_finite_difference(
    rollout_case,
) -> None:
    case = rollout_case
    stop_index = next(
        index
        for index, policy in enumerate(case["policies"])
        if policy.kind == "stop"
    )
    state = case["simulation"].state.copy()
    step = 2.0e-3
    finite = np.zeros(4)
    for axis in range(4):
        plus, minus = state.copy(), state.copy()
        plus[axis] += step
        minus[axis] -= step
        plus_value = evaluate_policy_batch(
            plus,
            case["nominal"],
            case["packed_policies"],
            case["packed_obstacles"],
            case["geometry"],
            case["parameters"],
        ).values[stop_index]
        minus_value = evaluate_policy_batch(
            minus,
            case["nominal"],
            case["packed_policies"],
            case["packed_obstacles"],
            case["geometry"],
            case["parameters"],
        ).values[stop_index]
        finite[axis] = (plus_value - minus_value) / (2.0 * step)
    np.testing.assert_allclose(
        case["evaluation"].gradients[stop_index],
        finite,
        rtol=2.0e-2,
        atol=3.0e-3,
    )


def test_room_exact_autodiff_gradient_matches_value_finite_difference(
    rollout_case,
) -> None:
    """Cover the production AD path without changing it to finite differences."""

    case = rollout_case
    room_index = next(
        index
        for index, policy in enumerate(case["policies"])
        if policy.kind == "room"
    )
    state = case["simulation"].state.copy()
    step = 2.0e-3
    finite = np.zeros(4)
    for axis in range(4):
        plus, minus = state.copy(), state.copy()
        plus[axis] += step
        minus[axis] -= step
        plus_value = evaluate_policy_batch(
            plus,
            case["nominal"],
            case["packed_policies"],
            case["packed_obstacles"],
            case["geometry"],
            case["parameters"],
        ).values[room_index]
        minus_value = evaluate_policy_batch(
            minus,
            case["nominal"],
            case["packed_policies"],
            case["packed_obstacles"],
            case["geometry"],
            case["parameters"],
        ).values[room_index]
        finite[axis] = (plus_value - minus_value) / (2.0 * step)

    np.testing.assert_allclose(
        case["evaluation"].gradients[room_index],
        finite,
        rtol=3.0e-2,
        atol=5.0e-3,
    )


def test_static_floor_union_and_wall_clearance_match_environment(
    rollout_case,
) -> None:
    simulation = rollout_case["simulation"]
    radius = simulation.config.robot.radius + simulation.config.safety.static_margin
    points = np.array(
        [
            [36.0, 47.5],
            [50.0, 44.0],
            [50.0, 36.0],
            [59.0, 51.2],
            [5.2, 47.0],
            [69.66085247, 15.32107043],
            [69.76131569, 43.46573127],
            [12.0, 13.86956522],
        ],
        dtype=np.float32,
    )
    geometry = backend._device_tree(rollout_case["geometry"])
    actual = np.asarray(
        jax.vmap(
            lambda point: backend._static_clearance(
                point, jnp.asarray(radius), geometry
            )
        )(jnp.asarray(points))
    )
    expected = np.asarray(
        [
            simulation.environment.static_clearance(point, radius)
            for point in points
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=5.0e-6)
    jax_collision = np.asarray(
        backend._environment_collision(
            jnp.asarray(points),
            jnp.full((len(points),), radius),
            geometry,
        )
    )
    np.testing.assert_array_equal(
        jax_collision,
        simulation.environment.collisions(points, radius),
    )
    np.testing.assert_array_equal(jax_collision, actual <= 0.0)


def test_human_bounce_and_reflected_stretcher_predictions_match_scalar(
    rollout_case,
) -> None:
    simulation = rollout_case["simulation"]
    obstacles = (
        # This circle reaches the lower main-corridor boundary and exercises
        # the same axial-bounce branches as ``Human.predicted``.
        Human("bouncing", 20.0, 43.65, 0.8, -1.1, 0.52),
        Stretcher(
            "reflected",
            8.0,
            47.5,
            -3.0,
            "x",
            5.0,
            12.0,
            4.1,
            1.45,
            True,
        ),
    )
    packed = backend._device_tree(
        pack_obstacle_batch(obstacles, rollout_case["capacities"])
    )
    geometry = backend._device_tree(rollout_case["geometry"])
    checkpoints = backend._circle_checkpoints(
        packed,
        geometry,
        rollout_case["capacities"].human_prediction_steps,
    )
    times = np.array([0.0, 0.04, 0.05, 0.08, 0.24, 0.61, 1.1])
    actual = np.asarray(
        backend._obstacle_centers_at(
            jnp.asarray(times),
            packed,
            geometry,
            checkpoints[0],
            checkpoints[1],
        )
    )
    for time_index, elapsed in enumerate(times):
        for obstacle_index, obstacle in enumerate(obstacles):
            expected = obstacle.predicted(
                float(elapsed), simulation.environment
            ).center
            np.testing.assert_allclose(
                actual[time_index, obstacle_index],
                expected,
                rtol=2.0e-6,
                atol=6.0e-6,
            )


def test_oriented_stretcher_clearance_uses_rectangle_sdf(rollout_case) -> None:
    case = rollout_case
    packed = pack_obstacle_batch(
        (case["obstacles"][1],), case["capacities"]
    )
    angle = 0.63
    cosines, sines = packed.cosines.copy(), packed.sines.copy()
    cosines[0], sines[0] = cos(angle), sin(angle)
    packed = packed._replace(cosines=cosines, sines=sines)
    point = np.array([72.8, 48.9])
    values, mask = backend._dynamic_clearance(
        jnp.asarray(point),
        jnp.asarray(packed.centers),
        backend._device_tree(packed),
        backend._device_tree(case["parameters"]),
    )
    delta = point - packed.centers[0]
    local_x = cos(angle) * delta[0] + sin(angle) * delta[1]
    local_y = -sin(angle) * delta[0] + cos(angle) * delta[1]
    qx = abs(local_x) - packed.half_lengths[0]
    qy = abs(local_y) - packed.half_widths[0]
    expected = (
        np.hypot(max(qx, 0.0), max(qy, 0.0))
        + min(max(qx, qy), 0.0)
        - case["simulation"].config.robot.radius
        - case["simulation"].config.safety.safety_margin
        - case["simulation"].config.safety.stretcher_margin
    )
    assert bool(np.asarray(mask)[0])
    assert float(np.asarray(values)[0]) == pytest.approx(
        expected, rel=2.0e-6, abs=5.0e-6
    )


def test_inactive_padding_is_semantically_inert(rollout_case) -> None:
    case = rollout_case
    policies = case["packed_policies"]
    batch = policies.batch
    inactive = ~batch.active
    kinds = batch.kinds.copy()
    waypoints = batch.waypoints.copy()
    angles = batch.angles.copy()
    kinds[inactive] = backend.POLICY_ROOM
    waypoints[inactive] = 1.0e5
    angles[inactive] = -9.0e4
    adversarial_policy_batch = batch._replace(
        kinds=kinds, waypoints=waypoints, angles=angles
    )
    adversarial_policies = replace(
        policies, batch=adversarial_policy_batch
    )

    obstacles = case["packed_obstacles"]
    inactive_obstacles = ~obstacles.active
    obstacle_kinds = obstacles.kinds.copy()
    centers = obstacles.centers.copy()
    velocities = obstacles.velocities.copy()
    bounce = obstacles.bounce.copy()
    obstacle_kinds[inactive_obstacles] = backend.OBSTACLE_RECTANGLE
    centers[inactive_obstacles] = -8.0e4
    velocities[inactive_obstacles] = 7.0e4
    bounce[inactive_obstacles] = True
    adversarial_obstacles = obstacles._replace(
        kinds=obstacle_kinds,
        centers=centers,
        velocities=velocities,
        bounce=bounce,
    )
    actual = evaluate_policy_batch(
        case["simulation"].state,
        case["nominal"],
        adversarial_policies,
        adversarial_obstacles,
        case["geometry"],
        case["parameters"],
    )
    np.testing.assert_allclose(actual.values, case["evaluation"].values)
    np.testing.assert_allclose(actual.gradients, case["evaluation"].gradients)
    np.testing.assert_allclose(
        actual.nominal_prefix_values,
        case["evaluation"].nominal_prefix_values,
    )


def test_moving_obstacle_values_are_policy_order_independent(
    rollout_case,
) -> None:
    case = rollout_case
    moving = (
        Human("bouncing", 20.0, 43.65, 0.8, -1.1, 0.52),
        Stretcher(
            "reflected",
            8.0,
            47.5,
            -3.0,
            "x",
            5.0,
            12.0,
            4.1,
            1.45,
            True,
        ),
    )
    obstacles = pack_obstacle_batch(moving, case["capacities"])
    forward = evaluate_policy_batch(
        case["simulation"].state,
        case["nominal"],
        case["packed_policies"],
        obstacles,
        case["geometry"],
        case["parameters"],
    )
    reverse_policies = pack_policy_batch(
        tuple(reversed(case["policies"])),
        case["simulation"].config,
        case["capacities"],
    )
    reverse = evaluate_policy_batch(
        case["simulation"].state,
        case["nominal"],
        reverse_policies,
        obstacles,
        case["geometry"],
        case["parameters"],
    )
    forward_values = dict(zip(forward.names, forward.values, strict=True))
    reverse_values = dict(zip(reverse.names, reverse.values, strict=True))
    assert forward_values.keys() == reverse_values.keys()
    for name, value in forward_values.items():
        assert reverse_values[name] == pytest.approx(value, abs=2.0e-6)


def test_bucket_selection_and_process_wide_cache_reuse(rollout_case) -> None:
    assert select_obstacle_bucket(0) == 4
    assert select_obstacle_bucket(4) == 4
    assert select_obstacle_bucket(5) == 8
    assert select_obstacle_bucket(53) == 53
    with pytest.raises(ValueError, match="exceed"):
        select_obstacle_bucket(54)
    selected = capacities_for_obstacle_count(
        rollout_case["simulation"].config,
        7,
    )
    assert selected.max_obstacles == 8

    before = compiled_evaluator_cache_info()
    reduced_policies = pack_policy_batch(
        rollout_case["policies"][:2],
        rollout_case["simulation"].config,
        rollout_case["capacities"],
    )
    reduced_obstacles = pack_obstacle_batch(
        rollout_case["obstacles"][:1], rollout_case["capacities"]
    )
    evaluate_policy_batch(
        rollout_case["simulation"].state,
        rollout_case["nominal"],
        reduced_policies,
        reduced_obstacles,
        rollout_case["geometry"],
        rollout_case["parameters"],
    )
    after = compiled_evaluator_cache_info()
    assert after.currsize == before.currsize
    assert after.misses == before.misses
    assert after.hits >= before.hits + 1


def test_warm_fixed_shape_evaluation_is_millisecond_scale(rollout_case) -> None:
    case = rollout_case
    durations = []
    for _ in range(5):
        started = time.perf_counter()
        evaluate_policy_batch(
            case["simulation"].state,
            case["nominal"],
            case["packed_policies"],
            case["packed_obstacles"],
            case["geometry"],
            case["parameters"],
        )
        durations.append(time.perf_counter() - started)
    # Deliberately generous for loaded CI hosts; the scalar oracle takes
    # hundreds of milliseconds even for this compact inventory.
    assert float(np.median(durations)) < 0.1
