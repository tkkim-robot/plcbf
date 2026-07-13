"""Regression tests for apples-to-apples warehouse benchmark outcomes."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from examples.warehouse import benchmark_warehouse_randomized_quad as benchmark


class _FakeEnvironment:
    def __init__(self):
        self.start_pos = np.zeros(2)
        self.goal_pos = np.array([1000.0, 1000.0])
        self.goal_radius = 0.5
        self.robot_pos = np.zeros(2)
        self.static_obstacles = []

    def step(self):
        return None

    def get_dynamic_obstacles(self):
        return []

    def get_static_obstacles(self):
        return list(self.static_obstacles)


class _FakeRobot:
    def step(self, state, control):
        del control
        return np.asarray(state, dtype=float).reshape(-1, 1)


class _FakeNominal:
    def __init__(self):
        self.waypoints = np.array([[1000.0, 1000.0]])
        self.wp_idx = 0

    def get_control(self, state, update_state=False):
        del state, update_state
        return np.zeros(4)


class _FakeShield:
    def __init__(self):
        self.calls = 0
        self.policy_configs = {f"policy_{i}": object() for i in range(66)}

    def update_obstacles(self, dynamic, static):
        del dynamic, static

    def solve_control_problem(self, state, control_ref):
        del state, control_ref
        self.calls += 1
        return np.zeros(4)

    def get_last_step_metrics(self):
        certificate_lost = self.calls == 1
        return {
            "selected_policy": None if certificate_lost else "policy_0",
            "certificate_lost": certificate_lost,
            "num_steps_with_no_certified_rollout": int(certificate_lost),
            "num_steps_with_no_feasible_qp": 0,
            "fallback_used": certificate_lost,
        }


class WarehouseBenchmarkFailureSemanticsTests(unittest.TestCase):
    def test_shared_continuation_is_scoped_to_explicit_comparison(self):
        self.assertFalse(benchmark._comparison_control_enabled("plcbf", False))
        self.assertTrue(benchmark._comparison_control_enabled("plcbf", True))
        self.assertTrue(
            benchmark._comparison_control_enabled("multi_backup_cbf_mi", False)
        )

    def test_certificate_and_qp_events_are_disjoint_when_explicit(self):
        empty = SimpleNamespace(certificate_lost=True)
        self.assertEqual(
            benchmark._step_failure_events(
                {
                    "num_steps_with_no_certified_rollout": 1,
                    "num_steps_with_no_feasible_qp": 0,
                },
                empty,
            ),
            (True, False, True),
        )
        self.assertEqual(
            benchmark._step_failure_events(
                {
                    "num_steps_with_no_certified_rollout": 0,
                    "num_steps_with_no_feasible_qp": 1,
                },
                empty,
            ),
            (False, True, True),
        )
        self.assertEqual(
            benchmark._step_failure_events(
                {"certificate_lost": True, "qp_infeasible": True}, empty
            ),
            (True, False, True),
        )

    def test_certificate_loss_does_not_terminate_physical_episode(self):
        env = _FakeEnvironment()
        robot = _FakeRobot()
        nominal = _FakeNominal()
        shield = _FakeShield()
        robot_spec = {
            "z_ref": 2.0,
            "radius": 0.5,
            "u_min": -10.0,
            "u_max": 10.0,
        }
        setup_result = (env, robot, nominal, shield, robot_spec, object())
        scenario = benchmark.TrialScenario(run_idx=7, seed=11, ghosts=tuple())

        with patch.object(
            benchmark.test_quad, "setup_test", return_value=setup_result
        ), patch.object(
            benchmark, "_common_stop_control", return_value=np.zeros(4)
        ):
            result = benchmark.run_trial(
                algo="library_pcbf_mi",
                scenario=scenario,
                level=7,
                safety_margin=1.3,
                alpha=6.0,
                max_steps=3,
                jit_warmup_steps=0,
                tracking_tol=0.1,
                plcbf_num_angle_policies=64,
                mip_num_angle_policies=64,
            )

        self.assertEqual(shield.calls, 3)
        self.assertTrue(result.certificate_lost)
        self.assertFalse(result.collision)
        self.assertFalse(result.qp_infeasible)
        self.assertTrue(result.survived_horizon)
        self.assertFalse(result.task_completed)
        self.assertTrue(result.completed_or_survived)
        self.assertTrue(result.union_failure)
        self.assertEqual(result.certificate_loss_steps, 1)
        self.assertEqual(result.fallback_steps, 1)

    def test_summary_keeps_legacy_and_union_failures_separate(self):
        trial = benchmark.TrialResult(
            collision=False,
            infeasible=False,
            reached_goal=False,
            nominal_tracking_pct=100.0,
            solve_time_sum_sec=0.003,
            timed_steps=3,
            total_steps=3,
            certificate_lost=True,
            survived_horizon=True,
            task_completed=False,
            completed_or_survived=True,
            filter_failure=True,
            union_failure=True,
            p_or_library_size=66,
        )
        summary = benchmark.summarize_trials(
            benchmark.AlgoSpec("plcbf", "PL-CBF"), [trial]
        )
        self.assertEqual(summary.fail_count, 0)
        self.assertEqual(summary.union_failures, 1)
        self.assertEqual(summary.certificate_losses, 1)
        self.assertEqual(summary.task_completions, 0)
        self.assertEqual(summary.successful_outcomes, 1)
        self.assertEqual(summary.library_size, 66)

    def test_collision_causing_step_is_in_total_and_event_accounting(self):
        env = _FakeEnvironment()
        env.static_obstacles = [{"x": 0.0, "y": 0.0, "radius": 1.0}]
        robot = _FakeRobot()
        nominal = _FakeNominal()
        shield = _FakeShield()
        robot_spec = {
            "z_ref": 2.0,
            "radius": 0.5,
            "u_min": -10.0,
            "u_max": 10.0,
        }
        setup_result = (env, robot, nominal, shield, robot_spec, object())
        scenario = benchmark.TrialScenario(run_idx=8, seed=12, ghosts=tuple())

        with patch.object(
            benchmark.test_quad, "setup_test", return_value=setup_result
        ), patch.object(
            benchmark, "_common_stop_control", return_value=np.zeros(4)
        ):
            result = benchmark.run_trial(
                algo="library_pcbf_mi",
                scenario=scenario,
                level=7,
                safety_margin=1.3,
                alpha=6.0,
                max_steps=3,
                jit_warmup_steps=0,
                tracking_tol=0.1,
                plcbf_num_angle_policies=64,
                mip_num_angle_policies=64,
            )

        self.assertTrue(result.collision)
        self.assertEqual(result.total_steps, 1)
        self.assertEqual(result.certificate_loss_steps, 1)
        self.assertLessEqual(
            result.certificate_loss_steps + result.qp_infeasible_steps,
            result.total_steps,
        )


if __name__ == "__main__":
    unittest.main()
