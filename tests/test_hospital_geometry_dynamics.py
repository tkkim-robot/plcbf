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
from examples.hospital.obstacles import Stretcher
from examples.hospital.planner import HospitalGridPlanner
from examples.hospital.simulation import (
    STRICT_WEST_JUNCTION_X,
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
