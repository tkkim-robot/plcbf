"""Deterministic simulation harness shared by tests, CLI, and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .controller import PLCBF_NLQuad3D
from .dynamics import NLQuad3D
from .scenarios import (
    NLQuad3DScenario,
    advance_obstacles,
    minimum_clearance,
)

if TYPE_CHECKING:
    from .rerun_logger import NLQuad3DRerunLogger


@dataclass(frozen=True)
class NLQuad3DSimulationResult:
    scenario: str
    states: np.ndarray
    controls: np.ndarray
    obstacle_history: np.ndarray
    times: np.ndarray
    selected_policies: tuple[str, ...]
    controller_statuses: tuple[str, ...]
    minimum_clearance: float
    collision: bool
    reached_goal: bool
    completed_waypoints: int

    @property
    def steps(self) -> int:
        return self.controls.shape[0]

    @property
    def final_state(self) -> np.ndarray:
        return self.states[-1].copy()


def simulate(
    scenario: NLQuad3DScenario,
    *,
    model: NLQuad3D | None = None,
    controller: PLCBF_NLQuad3D | None = None,
    max_steps: int | None = None,
    use_plcbf: bool = True,
    stop_on_collision: bool = True,
    logger: "NLQuad3DRerunLogger | None" = None,
) -> NLQuad3DSimulationResult:
    """Run one scenario without any graphics or wall-clock pacing."""

    dynamics = model or (controller.model if controller is not None else NLQuad3D())
    if controller is None and use_plcbf:
        controller = PLCBF_NLQuad3D(dynamics, bounds=scenario.bounds)
    if controller is not None and not np.isclose(
        controller.config.dt,
        dynamics.dt,
    ):
        raise ValueError("controller dt and dynamics dt must match")
    limit = scenario.default_steps if max_steps is None else int(max_steps)
    if limit < 1:
        raise ValueError("max_steps must be positive")

    state = scenario.initial_state
    obstacles = scenario.obstacles.copy()
    waypoint_index = 1 if scenario.waypoints.shape[0] > 1 else 0
    states = [state.copy()]
    obstacle_history = [obstacles.copy()]
    controls: list[np.ndarray] = []
    selected_policies: list[str] = []
    statuses: list[str] = []
    min_clearance = minimum_clearance(
        dynamics.safety_point(state),
        obstacles,
        dynamics.config.robot_radius,
    )
    collision = min_clearance < 0.0
    reached = False

    if logger is not None:
        lower = np.asarray(scenario.bounds.lower) if scenario.bounds else None
        upper = np.asarray(scenario.bounds.upper) if scenario.bounds else None
        logger.log_world(
            goal=scenario.goal,
            bounds_lower=lower,
            bounds_upper=upper,
        )

    for step_index in range(limit):
        goal = scenario.waypoints[waypoint_index]
        if use_plcbf:
            assert controller is not None
            control = controller.solve_control_problem(state, goal, obstacles)
            evaluation = controller.last_evaluation
            status = controller.last_status
            selected = (
                evaluation.selected_name if evaluation is not None else "nominal"
            )
        else:
            control = dynamics.nominal_input(state, goal)
            evaluation = None
            status = "nominal"
            selected = "nominal"
        if logger is not None:
            logger.log_step(
                time_seconds=step_index * dynamics.dt,
                state=state,
                obstacles=obstacles,
                goal=goal,
                control=control,
                evaluation=evaluation,
            )
        state = dynamics.step(state, control)
        obstacles = advance_obstacles(obstacles, dynamics.dt, scenario.bounds)
        controls.append(np.asarray(control).copy())
        selected_policies.append(selected)
        statuses.append(status)
        states.append(state.copy())
        obstacle_history.append(obstacles.copy())

        clearance = minimum_clearance(
            dynamics.safety_point(state),
            obstacles,
            dynamics.config.robot_radius,
        )
        min_clearance = min(min_clearance, clearance)
        collision = collision or clearance < 0.0
        if collision and stop_on_collision:
            break

        if np.linalg.norm(state[:3] - goal) <= scenario.reach_threshold:
            if waypoint_index + 1 < scenario.waypoints.shape[0]:
                waypoint_index += 1
            else:
                reached = True
                break

    if logger is not None:
        logger.close()
    return NLQuad3DSimulationResult(
        scenario=scenario.name,
        states=np.asarray(states),
        controls=np.asarray(controls).reshape(-1, 4),
        obstacle_history=np.asarray(obstacle_history),
        times=np.arange(len(states), dtype=float) * dynamics.dt,
        selected_policies=tuple(selected_policies),
        controller_statuses=tuple(statuses),
        minimum_clearance=float(min_clearance),
        collision=collision,
        reached_goal=reached,
        completed_waypoints=waypoint_index + int(reached),
    )
