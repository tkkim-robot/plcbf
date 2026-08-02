from __future__ import annotations

import numpy as np
import pytest

from examples.hospital.config import DEFAULT_CONFIG
from examples.hospital.environment import build_hospital_environment
from examples.hospital.obstacles import Human, Stretcher
from examples.hospital.scenario_generation import (
    DEFAULT_HUMAN_COUNT,
    DEFAULT_ORDINARY_STRETCHER_COUNT,
    DEFAULT_PROTECTED_CLEARANCE,
    GUARANTEED_BLOCKER_COUNTS,
    INITIAL_PAIRWISE_CLEARANCE,
    TOTAL_STRETCHER_COUNTS_WITH_BLOCKERS,
    generate_hospital_crowd,
    obstacle_pair_clearance,
)


EGO = np.array([67.0, 47.5])
GOAL = np.array([130.0, 47.5])


def _signature(crowd) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            obstacle.identifier,
            *obstacle.center,
            *obstacle.velocity,
        )
        for obstacle in crowd.obstacles
    )


def test_playground_scale_crowd_is_exact_and_seed_reproducible() -> None:
    environment = build_hospital_environment()
    first = generate_hospital_crowd(
        environment,
        seed=23,
        ego_position=EGO,
        goal_position=GOAL,
    )
    repeated = generate_hospital_crowd(
        environment,
        seed=23,
        ego_position=EGO,
        goal_position=GOAL,
    )
    changed = generate_hospital_crowd(
        environment,
        seed=24,
        ego_position=EGO,
        goal_position=GOAL,
    )

    assert len(first.humans) == DEFAULT_HUMAN_COUNT == 50
    assert (
        len(first.stretchers)
        == DEFAULT_ORDINARY_STRETCHER_COUNT
        == 15
    )
    assert first.metadata.obstacle_count == 65
    assert GUARANTEED_BLOCKER_COUNTS == (2, 3)
    assert TOTAL_STRETCHER_COUNTS_WITH_BLOCKERS == (17, 18)
    assert _signature(first) == _signature(repeated)
    assert _signature(first) != _signature(changed)


def test_generated_crowd_is_geometry_safe_and_bidirectional() -> None:
    environment = build_hospital_environment()
    crowd = generate_hospital_crowd(
        environment,
        DEFAULT_CONFIG,
        seed=991,
        ego_position=EGO,
        goal_position=GOAL,
    )

    assert crowd.metadata.minimum_pairwise_clearance is not None
    assert (
        crowd.metadata.minimum_pairwise_clearance
        >= INITIAL_PAIRWISE_CLEARANCE - 1e-12
    )
    assert crowd.metadata.minimum_protected_clearance is not None
    assert (
        crowd.metadata.minimum_protected_clearance
        >= DEFAULT_PROTECTED_CLEARANCE - 1e-12
    )
    for index, first in enumerate(crowd.obstacles):
        for second in crowd.obstacles[index + 1 :]:
            assert (
                obstacle_pair_clearance(first, second)
                >= INITIAL_PAIRWISE_CLEARANCE - 1e-12
            )

    for human in crowd.humans:
        assert isinstance(human, Human)
        assert not environment.is_collision(
            human.center,
            human.radius,
        )
        assert 0.0 < np.linalg.norm(human.velocity) <= 1.45 + 1e-12

    axes = {stretcher.axis for stretcher in crowd.stretchers}
    directions = {int(np.sign(stretcher.speed)) for stretcher in crowd.stretchers}
    magnitudes = {round(abs(stretcher.speed), 6) for stretcher in crowd.stretchers}
    assert axes == {"x", "y"}
    assert directions == {-1, 1}
    assert len(magnitudes) > 1
    for stretcher in crowd.stretchers:
        assert isinstance(stretcher, Stretcher)
        assert stretcher.reflect_at_route_bounds
        assert stretcher.route_min <= stretcher.coordinate <= stretcher.route_max
        predicted = stretcher.predicted(300.0, environment)
        assert predicted.route_min <= predicted.coordinate <= predicted.route_max
        assert np.count_nonzero(stretcher.velocity) == 1


def test_existing_convoy_blocker_is_respected_during_generation() -> None:
    blocker = Stretcher(
        identifier="blocking-stretcher-0",
        coordinate=76.0,
        lateral=47.5,
        speed=-4.2,
        axis="x",
        route_min=7.0,
        route_max=133.0,
        length=5.4,
        width=7.1,
        reflect_at_route_bounds=False,
    )
    crowd = generate_hospital_crowd(
        build_hospital_environment(),
        seed=77,
        ego_position=EGO,
        goal_position=GOAL,
        human_count=12,
        ordinary_stretcher_count=4,
        existing_obstacles=(blocker,),
    )
    assert all(
        obstacle_pair_clearance(obstacle, blocker)
        >= INITIAL_PAIRWISE_CLEARANCE - 1e-12
        for obstacle in crowd.obstacles
    )


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("human_count", -1),
        ("ordinary_stretcher_count", -1),
        ("protected_clearance", -0.1),
    ),
)
def test_invalid_generation_requests_fail_cleanly(
    keyword: str,
    value: float,
) -> None:
    arguments = {
        "seed": 1,
        "ego_position": EGO,
        "goal_position": GOAL,
        keyword: value,
    }
    with pytest.raises(ValueError):
        generate_hospital_crowd(
            build_hospital_environment(),
            **arguments,
        )
