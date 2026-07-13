"""Regression tests for baseline-only Warehouse benchmark semantics."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from examples.warehouse import benchmark_additional_baselines_quad as benchmark


DISTINCTIVE_CONTROL = np.array([0.31, -0.27, 0.19, 4.25], dtype=float)


class _FakeEnvironment:
    def __init__(self, *, collide: bool = False):
        self.start_pos = np.zeros(2)
        self.goal_pos = np.array([1000.0, 1000.0])
        self.goal_radius = 0.5
        self.robot_pos = np.zeros(2)
        self.static_obstacles = (
            [{"x": 0.0, "y": 0.0, "radius": 1.0}] if collide else []
        )

    def step(self):
        return None

    def get_dynamic_obstacles(self):
        return []

    def get_static_obstacles(self):
        return list(self.static_obstacles)


class _FakeRobot:
    def __init__(self):
        self.controls = []

    def step(self, state, control):
        self.controls.append(np.asarray(control, dtype=float).reshape(-1).copy())
        return np.asarray(state, dtype=float).reshape(-1, 1)


class _FakeNominal:
    def __init__(self):
        self.waypoints = np.array([[1000.0, 1000.0]])
        self.wp_idx = 0

    def get_control(self, state):
        del state
        return np.zeros(4)


class _FakeShield:
    def __init__(self, *, event: str | None = None, returned=DISTINCTIVE_CONTROL):
        self.calls = 0
        self.event = event
        self.returned = returned
        self.runtime_error = False
        self.policy_configs = {f"policy_{index}": object() for index in range(66)}

    def update_obstacles(self, dynamic, static):
        del dynamic, static

    def solve_control_problem(self, state, control_ref):
        del state, control_ref
        self.calls += 1
        if isinstance(self.returned, BaseException):
            raise self.returned
        return self.returned

    def get_last_step_metrics(self):
        active_event = self.event if self.calls == 1 else None
        return {
            "selected_policy": None if active_event else "policy_0",
            "certificate_lost": active_event == "certificate_lost",
            "qp_infeasible": active_event == "qp_infeasible",
            "num_steps_with_no_certified_rollout": int(
                active_event == "certificate_lost"
            ),
            "num_steps_with_no_feasible_qp": int(
                active_event == "qp_infeasible"
            ),
            "fallback_used": False,
            "runtime_error": False,
        }


def _run(*, shield, max_steps=3, collide=False):
    env = _FakeEnvironment(collide=collide)
    robot = _FakeRobot()
    nominal = _FakeNominal()
    robot_spec = {
        "z_ref": 2.0,
        "radius": 0.5,
        "u_min": -10.0,
        "u_max": 10.0,
    }
    setup_result = (env, robot, nominal, shield, robot_spec, object())
    scenario = benchmark.TrialScenario(run_idx=7, seed=11, ghosts=tuple())
    with patch.object(benchmark.test_quad, "setup_test", return_value=setup_result):
        result = benchmark.run_trial(
            algo="library_pcbf_mi",
            scenario=scenario,
            level=7,
            safety_margin=1.3,
            alpha=6.0,
            max_steps=max_steps,
            jit_warmup_steps=0,
            tracking_tol=0.1,
            num_angle_policies=64,
        )
    return result, robot


def test_registry_contains_only_two_additional_baselines():
    assert benchmark.BASELINE_KEYS == (
        "multi_backup_cbf_mi",
        "library_pcbf_mi",
    )


def test_certificate_and_qp_events_are_disjoint_and_fallback_is_explicit():
    empty = SimpleNamespace(certificate_lost=True)
    assert benchmark._step_failure_events(
        {
            "num_steps_with_no_certified_rollout": 1,
            "num_steps_with_no_feasible_qp": 0,
        },
        empty,
    ) == (True, False, False)
    assert benchmark._step_failure_events(
        {
            "num_steps_with_no_certified_rollout": 0,
            "num_steps_with_no_feasible_qp": 1,
            "fallback_used": True,
        },
        empty,
    ) == (False, True, True)


@pytest.mark.parametrize("event", ["certificate_lost", "qp_infeasible"])
def test_diagnostic_event_applies_exact_returned_control_and_continues(event):
    result, robot = _run(shield=_FakeShield(event=event))

    assert len(robot.controls) == 3
    assert all(
        np.array_equal(control, DISTINCTIVE_CONTROL) for control in robot.controls
    )
    assert result.certificate_lost is (event == "certificate_lost")
    assert result.qp_infeasible is (event == "qp_infeasible")
    assert result.infeasible is False
    assert result.unrecoverable_infeasible is False
    assert result.historical_failure is False
    assert result.survived_horizon is True
    assert result.fallback_steps == 0


@pytest.mark.parametrize(
    "returned",
    [
        RuntimeError("solve failed"),
        None,
        np.zeros(3),
        np.array([np.nan, 0.0, 0.0, 0.0]),
        np.array([11.0, 0.0, 0.0, 0.0]),
    ],
)
def test_unrecoverable_control_failure_terminates_without_simulator_step(returned):
    result, robot = _run(shield=_FakeShield(returned=returned))

    assert robot.controls == []
    assert result.runtime_error is True
    assert result.infeasible is True
    assert result.unrecoverable_infeasible is True
    assert result.historical_failure is True
    assert result.total_steps == 0


def test_summary_main_failure_ignores_certificate_and_union_diagnostics():
    trial = benchmark.TrialResult(
        collision=False,
        infeasible=False,
        unrecoverable_infeasible=False,
        historical_failure=False,
        reached_goal=False,
        nominal_tracking_pct=100.0,
        solve_time_sum_sec=0.003,
        timed_steps=3,
        total_steps=3,
        certificate_lost=True,
        survived_horizon=True,
        completed_or_survived=True,
        filter_failure=True,
        union_failure=True,
        p_or_library_size=66,
    )
    summary = benchmark.summarize_trials(
        benchmark.AlgoSpec("library_pcbf_mi", "Lib-PCBF-MI"), [trial]
    )
    assert summary.fail_count == 0
    assert summary.union_failures == 1
    assert summary.certificate_losses == 1


def test_collision_causing_action_is_applied_and_sets_historical_failure():
    result, robot = _run(shield=_FakeShield(), collide=True)

    assert len(robot.controls) == 1
    assert np.array_equal(robot.controls[0], DISTINCTIVE_CONTROL)
    assert result.collision is True
    assert result.total_steps == 1
    assert result.historical_failure is True
