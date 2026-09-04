from __future__ import annotations

from io import BytesIO

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from examples.hospital.config import DEFAULT_CONFIG
from examples.hospital.dynamics import step_double_integrator
from examples.hospital.environment import build_hospital_environment
from examples.hospital.feasibility import (
    WITNESS_DYNAMIC_SUBSTEPS,
    _precompute_traffic,
)
from examples.hospital.obstacles import Human, Stretcher
from examples.hospital.planner import HospitalGridPlanner
from examples.hospital.simulation import (
    STRICT_WEST_JUNCTION_X,
    SweptTransitionSafety,
    build_blocked_main_hall_scenario,
    evaluate_swept_transition,
)
from examples.hospital.visualization import draw_simulation


def _planner():
    environment = build_hospital_environment()
    config = DEFAULT_CONFIG
    return environment, HospitalGridPlanner(
        environment,
        resolution=config.planner.resolution,
        clearance=config.robot.radius + config.planner.clearance_buffer,
        preferred_clearance=config.planner.preferred_clearance,
        clearance_weight=config.planner.clearance_weight,
    )


def test_floor_plan_doorway_and_planner_paths_are_robot_feasible() -> None:
    environment, planner = _planner()
    assert (environment.width, environment.height) == (140.0, 95.0)
    assert len(environment.corridor_rects) == 8
    assert len(environment.rooms) == 25

    nurse = next(room for room in environment.rooms if room.label == "Nurse")
    outside, door, inside, terminal_center = environment.room_door_path(
        nurse,
        DEFAULT_CONFIG.robot.radius,
        DEFAULT_CONFIG.refuge.inside_door_offset,
        DEFAULT_CONFIG.refuge.outside_door_offset,
    )
    assert nurse.door.side == "bottom"
    assert not nurse.contains(outside)
    assert nurse.contains(inside)
    assert (
        nurse.interior_margin(terminal_center)
        >= DEFAULT_CONFIG.refuge.terminal_interior_margin
    )
    for first, second in zip(
        (outside, door, inside),
        (door, inside, terminal_center),
    ):
        assert environment.segment_is_free(
            first,
            second,
            DEFAULT_CONFIG.robot.radius + 0.04,
        )

    path = planner.plan(np.array([67.0, 47.5]), np.array([130.0, 47.5]))
    assert len(path) >= 2
    for first, second in zip(path[:-1], path[1:]):
        assert environment.segment_is_free(
            first,
            second,
            DEFAULT_CONFIG.robot.radius
            + DEFAULT_CONFIG.planner.clearance_buffer,
        )


def test_vectorized_environment_queries_match_scalar_geometry_exactly() -> None:
    environment = build_hospital_environment()
    rng = np.random.default_rng(831)
    points = np.vstack(
        (
            rng.uniform([0.0, 0.0], [140.0, 95.0], size=(80, 2)),
            np.array(
                [
                    [0.0, 0.0],
                    [140.0, 95.0],
                    [67.0, 47.5],
                    [28.0, 47.5],
                ]
            ),
        )
    )
    radii = rng.uniform(0.0, 0.9, size=len(points))

    np.testing.assert_array_equal(
        environment.collisions(points, radii),
        np.asarray(
            [
                environment.is_collision(point, radius)
                for point, radius in zip(points, radii, strict=True)
            ]
        ),
    )
    for radius in (0.0, 1e-12):
        np.testing.assert_array_equal(
            environment.static_clearances(points, radius),
            [
                environment.static_clearance(point, radius)
                for point in points
            ],
        )
    for radius in (0.0, 0.55, 0.69):
        np.testing.assert_allclose(
            environment.static_clearances(points, radius),
            [
                environment.static_clearance(point, radius)
                for point in points
            ],
            rtol=0.0,
            atol=1e-12,
        )
    tiny_radii = np.resize(np.asarray([0.0, 1e-12]), len(points))
    np.testing.assert_array_equal(
        environment.collisions(points, tiny_radii),
        [
            environment.is_collision(point, radius)
            for point, radius in zip(
                points, tiny_radii, strict=True
            )
        ],
    )

    starts = points[:30]
    ends = starts + rng.uniform(-1.0, 1.0, size=starts.shape)
    np.testing.assert_array_equal(
        environment.segments_are_free(starts, ends, 0.69),
        np.asarray(
            [
                environment.segment_is_free(start, end, 0.69)
                for start, end in zip(starts, ends, strict=True)
            ]
        ),
    )


def test_x64_feasibility_human_replay_matches_3000_simulator_steps() -> None:
    """The offline witness must see the exact publication crowd trajectory."""

    environment = build_hospital_environment()
    humans = (
        Human("horizontal", 67.0, 47.5, 1.2, 0.0),
        Human("vertical", 24.0, 47.5, 0.0, 1.0),
    )
    prediction = _precompute_traffic(
        humans,
        (),
        environment,
        DEFAULT_CONFIG,
    )
    replay = [
        Human(
            human.identifier,
            human.x,
            human.y,
            human.vx,
            human.vy,
            human.radius,
        )
        for human in humans
    ]
    expected = np.empty_like(prediction.human_centers)
    expected[0] = np.asarray([human.center for human in replay])
    swept_check_steps = {0, 1, 10, 100, 500, 1000, 2000, 2999}
    expected_swept: dict[int, np.ndarray] = {}
    for step_index in range(1, 3001):
        source_index = step_index - 1
        if source_index in swept_check_steps:
            expected_swept[source_index] = np.asarray(
                [
                    [
                        human.predicted(
                            sample_index
                            / WITNESS_DYNAMIC_SUBSTEPS
                            * DEFAULT_CONFIG.dt,
                            environment,
                        ).center
                        for human in replay
                    ]
                    for sample_index in range(
                        WITNESS_DYNAMIC_SUBSTEPS + 1
                    )
                ]
            )
        for human in replay:
            human.advance(DEFAULT_CONFIG.dt, environment)
        expected[step_index] = np.asarray([human.center for human in replay])

    assert prediction.human_centers.shape == (3001, 2, 2)
    assert prediction.human_swept_centers.shape == (3000, 9, 2, 2)
    np.testing.assert_allclose(
        prediction.human_centers,
        expected,
        rtol=0.0,
        atol=1.0e-12,
    )
    for step_index, expected_samples in expected_swept.items():
        np.testing.assert_array_equal(
            prediction.human_swept_centers[step_index],
            expected_samples,
        )
    # At a reflection, Human.predicted(dt) may differ from the independent
    # production Human.advance(dt) carry.  The audit must retain both rather
    # than linearly interpolate between production endpoints.
    assert np.any(
        prediction.human_swept_centers[:, -1]
        != prediction.human_centers[1:]
    )
    # Both trajectories cross enough of the map to exercise repeated wall
    # reflections, rather than testing only unobstructed linear propagation.
    horizontal_direction = np.sign(np.diff(expected[:, 0, 0]))
    vertical_direction = np.sign(np.diff(expected[:, 1, 1]))
    assert np.count_nonzero(np.diff(horizontal_direction)) >= 2
    assert np.count_nonzero(np.diff(vertical_direction)) >= 2


def test_static_collision_and_signed_clearance_use_one_exact_geometry() -> None:
    environment = build_hospital_environment()
    rng = np.random.default_rng(2901)
    # Include wall corners and doorway/seam points that disagreed under the old
    # 16-sample collision and 12-sample clearance approximations.
    points = np.vstack(
        (
            rng.uniform([0.0, 0.0], [140.0, 95.0], size=(500, 2)),
            np.array(
                [
                    [69.66085247, 15.32107043],
                    [104.47959414, 23.96464175],
                    [69.76131569, 43.46573127],
                    [62.24284852, 23.59176186],
                    [12.0, 15.0],
                    [12.0, 13.86956522],
                ]
            ),
        )
    )
    radii = rng.uniform(0.0, 0.8, size=len(points))
    radii[-6:] = 0.55
    clearances = np.asarray(
        [
            environment.static_clearance(point, radius)
            for point, radius in zip(points, radii, strict=True)
        ]
    )
    np.testing.assert_array_equal(
        environment.collisions(points, radii),
        clearances <= 0.0,
    )


def test_cached_floor_segments_are_only_exterior_union_boundaries() -> None:
    environment = build_hospital_environment()
    assert len(environment._floor_boundary_starts) > 0
    for start, end in zip(
        environment._floor_boundary_starts,
        environment._floor_boundary_ends,
        strict=True,
    ):
        tangent = end - start
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        midpoint = 0.5 * (start + end)
        first = environment.is_on_floor(midpoint + 1.0e-5 * normal)
        second = environment.is_on_floor(midpoint - 1.0e-5 * normal)
        assert first is not second


def test_double_integrator_bounds_and_stretcher_geometry_prediction() -> None:
    robot = DEFAULT_CONFIG.robot
    state = np.array([1.0, 2.0, robot.v_max, 0.0])
    following = step_double_integrator(
        state,
        np.array([10.0, 10.0]),
        DEFAULT_CONFIG.dt,
        robot,
    )
    assert np.linalg.norm(following[2:]) <= robot.v_max + 1e-12
    assert np.all(np.isfinite(following))

    environment = build_hospital_environment()
    obstacle = Stretcher(
        identifier="vertical",
        coordinate=20.0,
        lateral=65.0,
        speed=-2.0,
        axis="y",
        route_min=10.0,
        route_max=30.0,
        length=5.0,
        width=2.0,
    )
    predicted = obstacle.predicted(8.0, environment)
    advanced = Stretcher(**obstacle.__dict__)
    advanced.advance(8.0, environment)
    np.testing.assert_allclose(predicted.center, advanced.center)
    np.testing.assert_allclose(predicted.velocity, advanced.velocity)
    assert predicted.route_min <= predicted.coordinate <= predicted.route_max
    assert obstacle.signed_clearance(obstacle.center, 0.5) < 0.0
    assert obstacle.signed_clearance(obstacle.center + np.array([4.0, 0.0]), 0.5) > 0.0


def test_one_way_stretcher_prediction_matches_advance_without_reflection() -> None:
    environment = build_hospital_environment()
    obstacle = Stretcher(
        identifier="departing",
        coordinate=12.0,
        lateral=47.5,
        speed=-4.2,
        axis="x",
        route_min=7.0,
        route_max=130.0,
        reflect_at_route_bounds=False,
    )
    elapsed = 3.0
    predicted = obstacle.predicted(elapsed, environment)
    advanced = Stretcher(**obstacle.__dict__)
    advanced.advance(elapsed, environment)
    assert predicted.coordinate == advanced.coordinate == pytest.approx(-0.6)
    assert predicted.speed == advanced.speed == -4.2
    assert not predicted.reflect_at_route_bounds


@pytest.mark.parametrize("strategy", ("stationary", "maximum_reverse"))
def test_no_room_corridor_strategy_is_swept_before_west_junction(
    strategy: str,
) -> None:
    simulation = build_blocked_main_hall_scenario(2)
    state = simulation.state.copy()
    violated = False
    for _ in range(300):
        control = (
            np.zeros(2)
            if strategy == "stationary"
            else np.array([-simulation.config.robot.a_max, 0.0])
        )
        following = step_double_integrator(
            state,
            control,
            simulation.config.dt,
            simulation.config.robot,
        )
        transition = evaluate_swept_transition(
            simulation.environment,
            simulation.obstacles,
            state,
            following,
            simulation.config.dt,
            simulation.config,
        )
        for obstacle in simulation.obstacles:
            obstacle.advance(
                simulation.config.dt,
                simulation.environment,
            )
        state = following
        assert simulation.environment.room_containing(state[:2]) is None
        if transition.minimum_safety_clearance < 0.0:
            violated = True
            break
    assert violated
    assert state[0] > STRICT_WEST_JUNCTION_X


def test_stretcher_corner_clearance_is_exact_circle_rectangle_distance() -> None:
    obstacle = Stretcher(
        identifier="corner",
        coordinate=0.0,
        lateral=0.0,
        speed=0.0,
        axis="x",
        route_min=-10.0,
        route_max=10.0,
        length=4.0,
        width=2.0,
    )
    # Relative to the box corner (2, 1), this point has offset (0.3, 0.4).
    # Its Euclidean distance is exactly the robot radius, so it is tangent.
    tangent = np.array([2.3, 1.4])
    assert np.isclose(obstacle.signed_clearance(tangent, 0.5), 0.0)
    assert obstacle.signed_clearance(tangent, 0.49) > 0.0
    assert obstacle.signed_clearance(tangent, 0.51) < 0.0


def test_swept_clearance_witness_attributes_synchronized_human_sample() -> None:
    environment = build_hospital_environment()
    human = Human(
        identifier="witness-human",
        x=51.4,
        y=47.5,
        vx=0.0,
        vy=0.0,
    )
    start = np.array([50.0, 47.5, 0.0, 0.0])
    end = np.array([50.2, 47.5, 0.0, 0.0])

    transition = evaluate_swept_transition(
        environment,
        (human,),
        start,
        end,
        DEFAULT_CONFIG.dt,
        DEFAULT_CONFIG,
    )

    physical = transition.minimum_clearance_witness
    operational = transition.minimum_safety_clearance_witness
    assert physical is not None and operational is not None
    assert physical.source_kind == operational.source_kind == "human"
    assert physical.obstacle_identifier == "witness-human"
    assert operational.obstacle_identifier == "witness-human"
    assert physical.sample_index == operational.sample_index == 8
    assert physical.sample_fraction == operational.sample_fraction == 1.0
    assert physical.elapsed_s == operational.elapsed_s == DEFAULT_CONFIG.dt
    assert physical.robot_position == operational.robot_position == (50.2, 47.5)
    assert physical.value == transition.minimum_clearance
    assert operational.value == transition.minimum_safety_clearance


def test_swept_clearance_witness_distinguishes_stretcher_and_static() -> None:
    environment = build_hospital_environment()
    state = np.array([50.0, 47.5, 0.0, 0.0])
    stretcher = Stretcher(
        identifier="witness-stretcher",
        coordinate=47.5,
        lateral=51.5,
        speed=0.0,
        axis="y",
        route_min=40.0,
        route_max=55.0,
    )
    dynamic = evaluate_swept_transition(
        environment,
        (stretcher,),
        state,
        state,
        0.0,
        DEFAULT_CONFIG,
    )
    static = evaluate_swept_transition(
        environment,
        (),
        state,
        state,
        0.0,
        DEFAULT_CONFIG,
    )

    assert dynamic.minimum_safety_clearance_witness is not None
    assert dynamic.minimum_safety_clearance_witness.source_kind == "stretcher"
    assert (
        dynamic.minimum_safety_clearance_witness.obstacle_identifier
        == "witness-stretcher"
    )
    assert static.minimum_safety_clearance_witness is not None
    assert static.minimum_safety_clearance_witness.source_kind == "static"
    assert static.minimum_safety_clearance_witness.obstacle_identifier is None


def test_swept_transition_result_remains_backward_constructible() -> None:
    transition = SweptTransitionSafety(False, 1.0, 0.5)
    assert transition.minimum_clearance_witness is None
    assert transition.minimum_safety_clearance_witness is None


def test_hospital_matplotlib_visualization_renders_headlessly() -> None:
    simulation = build_blocked_main_hall_scenario(3)
    figure, axes = draw_simulation(simulation)
    assert len(axes.patches) >= (
        len(simulation.environment.floor_rects)
        + len(simulation.environment.wall_rects)
        + len(simulation.obstacles)
        + 2
    )
    output = BytesIO()
    figure.savefig(output, format="png", dpi=60)
    plt.close(figure)
    assert output.tell() > 10_000
