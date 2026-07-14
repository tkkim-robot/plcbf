"""Focused fairness and failure tests for the additive Quad3D baselines."""

from collections import OrderedDict
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from safe_control.envs.warehouse_env import WarehouseEnv
from safe_control.position_control.backup_cbf_qp import BackupCBF
from examples.warehouse.algorithms.library_pcbf_mi_quad3d import (
    CandidatePCBFResult,
    LibraryPCBFMinInterventionQuad3D,
)
from examples.warehouse.algorithms.multi_backup_cbf_mi_quad3d import (
    CandidateCBFResult,
    MultiBackupCBFMinInterventionQuad3D,
    _FrozenGhostPredictor,
    _QuadPolicyAdapter,
)
from examples.warehouse.algorithms.plcbf_quad3d import PLCBF_Quad3D
from examples.warehouse.controllers.policies_quad3d_jax import (
    AnglePolicyJAX,
    StopPolicyJAX,
    WaypointPolicyJAX,
    WaypointPolicyParams,
)
from examples.warehouse.dynamics.quad3d import Quad3D


def robot_spec():
    return {
        "model": "Quad3D",
        "radius": 1.0,
        "mass": 3.0,
        "Ix": 0.5,
        "Iy": 0.5,
        "Iz": 0.5,
        "L": 0.3,
        "nu": 0.1,
        "g": 9.8,
        "u_max": 10.0,
        "u_min": -10.0,
        "v_max": 3.5,
        "v_ref": 3.0,
        "a_max_xy": 8.0,
        "z_ref": 0.0,
        "Kp_z": 4.0,
        "Kd_z": 3.0,
        "K_ang": 10.0,
        "Kd_ang": 4.0,
        "nominal_Kp_v": 7.0,
        "nominal_K_lat": 1.2,
        "nominal_v_lat_max": 2.5,
        "nominal_dist_threshold": 1.0,
        "angle_Kp_v": 7.0,
        "stop_Kp_v": 3.0,
    }


def initial_state():
    state = np.zeros(12, dtype=float)
    state[:2] = (10.0, 10.0)
    return state


def failed_constraint_audit():
    return SimpleNamespace(
        passed=False,
        constraint_count=4,
        max_violation=2e-5,
        max_tolerance=1e-5,
        max_violation_ratio=2.0,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )


class Quad3DAdditionalBaselineTests(unittest.TestCase):
    def setUp(self):
        self.spec = robot_spec()
        self.env = WarehouseEnv(level=7)
        self.robot = Quad3D(self.env.dt, self.spec)

    def test_library_equality_and_numpy_policy_adapters(self):
        reference = PLCBF_Quad3D(
            self.spec, backup_horizon=4.0, num_angle_policies=2
        )
        mb = MultiBackupCBFMinInterventionQuad3D(
            self.robot, self.spec, num_angle_policies=2
        )
        lib = LibraryPCBFMinInterventionQuad3D(
            self.spec, backup_horizon=4.0, num_angle_policies=2
        )
        mb.assert_library_equal_to(reference)
        lib.assert_policy_library_equal(reference)

        runtime = OrderedDict(mb.policy_configs)
        policy_type, params = runtime["nominal"]
        runtime["nominal"] = (
            policy_type,
            WaypointPolicyParams(
                waypoints=jnp.array([[10.0, 10.0], [30.0, 10.0]]),
                v_max=3.5,
                Kp=7.0,
                K_lat=1.2,
                v_lat_max=2.5,
                dist_threshold=1.0,
                current_wp_idx=1,
                ctrl=params.ctrl,
            ),
        )

        rng = np.random.default_rng(4)
        for _ in range(4):
            state = rng.normal(size=12)
            for name, (kind, policy_params) in runtime.items():
                actual = _QuadPolicyAdapter(name, kind, policy_params).compute_control(
                    state
                )
                policy_cls = {
                    "angle": AnglePolicyJAX,
                    "stop": StopPolicyJAX,
                    "waypoint": WaypointPolicyJAX,
                }[kind]
                expected = np.asarray(
                    policy_cls.compute(jnp.asarray(state), policy_params), dtype=float
                )
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_mb_angle_and_nominal_switch_to_exact_common_stop_tail(self):
        controller = MultiBackupCBFMinInterventionQuad3D(
            self.robot,
            self.spec,
            backup_horizon=0.2,
            maneuver_prefix_sec=0.1,
            num_angle_policies=2,
        )
        runtime = controller._runtime_policy_params(
            {
                "waypoints": np.array([[10.0, 10.0], [30.0, 10.0]]),
                "wp_idx": 1,
            }
        )
        stop_params = runtime["stop"][1]
        state = initial_state()
        state[3:12] = np.array(
            [0.03, -0.02, 0.01, 0.4, -0.2, 0.05, 0.02, -0.01, 0.01]
        )
        expected_stop = np.asarray(
            StopPolicyJAX.compute(jnp.asarray(state), stop_params), dtype=float
        )

        policy_classes = {"angle_0": AnglePolicyJAX, "nominal": WaypointPolicyJAX}
        for name, policy_cls in policy_classes.items():
            policy_type, params = runtime[name]
            adapter = controller._candidate_adapters[name]
            adapter.set_params(policy_type, params)
            adapter.prepare_rollout(state)

            adapter.set_rollout_step(controller.maneuver_prefix_steps - 1)
            expected_prefix = np.asarray(
                policy_cls.compute(jnp.asarray(state), params), dtype=float
            )
            np.testing.assert_allclose(
                adapter.compute_control(state), expected_prefix, atol=2e-5, rtol=2e-5
            )
            self.assertFalse(adapter.using_stop_tail)

            adapter.set_rollout_step(controller.maneuver_prefix_steps)
            np.testing.assert_allclose(
                adapter.compute_control(state), expected_stop, atol=2e-5, rtol=2e-5
            )
            np.testing.assert_allclose(
                adapter.compute_terminal_stop_control(state),
                expected_stop,
                atol=2e-5,
                rtol=2e-5,
            )
            self.assertTrue(adapter.using_stop_tail)

        stop_adapter = controller._candidate_adapters["stop"]
        stop_adapter.set_rollout_step(0)
        self.assertTrue(stop_adapter.using_stop_tail)
        np.testing.assert_allclose(
            stop_adapter.compute_control(state), expected_stop, atol=2e-5, rtol=2e-5
        )

        angle_adapter = controller._candidate_adapters["angle_0"]
        angle_candidate = controller._candidate_filters["angle_0"]
        angle_adapter.prepare_rollout(state)
        with patch.object(
            angle_adapter,
            "set_rollout_step",
            wraps=angle_adapter.set_rollout_step,
        ) as set_phase:
            angle_candidate._integrate_state_trajectory(state)
        self.assertEqual(
            [call.args[0] for call in set_phase.call_args_list],
            list(range(angle_candidate.N - 1)),
        )
        self.assertTrue(angle_adapter.using_stop_tail)

        config = controller.get_status()["terminal_config"]
        self.assertEqual(config["maneuver_prefix_steps"], 2)
        self.assertAlmostEqual(config["terminal_tail_sec"], 0.1)
        self.assertEqual(config["terminal_successor_steps"], 1)

    def test_lib_selector_is_order_independent_and_uses_fixed_index_tie_break(self):
        first = CandidatePCBFResult(
            "first", 0, True, 1.0, True, np.zeros(4), 0.2, np.sqrt(0.2),
            "optimal", 0.0,
        )
        lower_cost = CandidatePCBFResult(
            "lower", 1, True, 1.0, True, np.zeros(4), 0.1, np.sqrt(0.1),
            "optimal", 0.0,
        )
        tie_later = CandidatePCBFResult(
            "tie", 2, True, 1.0, True, np.zeros(4), 0.1 + 1e-8,
            np.sqrt(0.1 + 1e-8), "optimal", 0.0,
        )
        cls = LibraryPCBFMinInterventionQuad3D
        self.assertEqual(
            cls._select_minimum_intervention([first, tie_later, lower_cost]).policy_name,
            "lower",
        )
        self.assertEqual(
            cls._select_minimum_intervention([lower_cost, tie_later, first]).policy_name,
            "lower",
        )

    def test_lib_matches_plcbf_values_gradients_time_term_and_same_policy_qp(self):
        horizon = 0.2
        reference = PLCBF_Quad3D(
            self.spec, dt=self.env.dt, backup_horizon=horizon, num_angle_policies=2
        )
        controller = LibraryPCBFMinInterventionQuad3D(
            self.spec, dt=self.env.dt, backup_horizon=horizon, num_angle_policies=2
        )
        dynamic = [{"x": 30.0, "y": 10.0, "radius": 1.0, "vx": -0.2, "vy": 0.1}]
        static = [{"x": 40.0, "y": 40.0, "radius": 2.0}]
        reference.update_obstacles(dynamic, static)
        controller.update_obstacles(dynamic, static)
        state = initial_state()
        nominal = np.zeros(4)
        control_ref = {
            "u_ref": nominal,
            "waypoints": np.array([[10.0, 10.0], [30.0, 10.0]]),
            "wp_idx": 1,
        }

        expected_control = np.asarray(
            reference.solve_control_problem(state, control_ref), dtype=float
        ).reshape(-1)
        results, _, time_derivatives = controller._evaluate_policy_library(
            state, control_ref
        )
        for name in controller.policy_names:
            expected_value, expected_gradient, expected_trajectory = (
                reference._last_results[name]
            )
            value, gradient, trajectory = results[name]
            self.assertEqual(value, expected_value)
            np.testing.assert_array_equal(gradient, expected_gradient)
            np.testing.assert_array_equal(trajectory, expected_trajectory)

        selected = reference._last_best_name
        self.assertAlmostEqual(
            time_derivatives[selected], reference._last_time_derivative, places=12
        )
        value, gradient, _ = results[selected]
        candidate = controller._solve_candidate_qp(
            selected,
            controller.policy_names.index(selected),
            state,
            nominal,
            value,
            gradient,
            time_derivatives[selected],
        )
        self.assertTrue(candidate.feasible)
        self.assertIs(candidate.post_projection_constraints_satisfied, True)
        self.assertGreater(candidate.constraint_audit_count, 0)
        self.assertLessEqual(candidate.max_post_projection_violation_ratio, 1.0)
        np.testing.assert_allclose(
            candidate.u, expected_control, atol=2e-5, rtol=2e-5
        )

    def test_lib_rejects_failed_post_projection_constraint_audit(self):
        controller = LibraryPCBFMinInterventionQuad3D(
            self.spec, dt=self.env.dt, backup_horizon=0.2, num_angle_policies=2
        )
        with patch(
            "examples.warehouse.algorithms.library_pcbf_mi_quad3d."
            "audit_cvxpy_inequalities",
            return_value=failed_constraint_audit(),
        ):
            candidate = controller._solve_candidate_qp(
                policy_name="angle_0",
                policy_index=0,
                state=initial_state(),
                u_nom=np.zeros(4),
                value=1.0,
                gradient=np.zeros(12),
                time_derivative=0.0,
            )

        self.assertFalse(candidate.feasible)
        self.assertIsNone(candidate.u)
        self.assertIs(candidate.post_projection_constraints_satisfied, False)
        self.assertEqual(candidate.max_post_projection_violation_ratio, 2.0)
        self.assertEqual(candidate.error, "post-projection constraint audit failed")

    def test_mb_no_candidate_is_explicit_failure(self):
        controller = MultiBackupCBFMinInterventionQuad3D(
            self.robot, self.spec, num_angle_policies=1
        )
        controller.set_environment(self.env)
        controller.update_obstacles([], self.env.get_static_obstacles())

        def failed_candidate(candidate, state, u_nom):
            del state, u_nom
            return CandidateCBFResult(
                policy_name=candidate.backup_controller.policy_name,
                feasible=False,
                u=None,
                objective=float("inf"),
                solver_status="infeasible",
                rollout_safe=False,
                terminal_safe=False,
                solve_time_sec=0.0,
            )

        with patch.object(
            type(next(iter(controller._candidate_filters.values()))),
            "solve_candidate",
            failed_candidate,
        ):
            emergency = controller.solve_control_problem(
                initial_state(),
                {
                    "u_ref": np.zeros(4),
                    "waypoints": self.env.get_nominal_waypoints(),
                    "wp_idx": 1,
                },
            )
        self.assertFalse(controller.infeasible)
        self.assertTrue(controller.certificate_lost)
        self.assertFalse(controller.qp_infeasible)
        self.assertTrue(np.all(np.isfinite(emergency)))
        self.assertTrue(np.all(emergency <= self.spec["u_max"] + 1e-8))
        self.assertTrue(np.all(emergency >= self.spec["u_min"] - 1e-8))
        self.assertEqual(controller.get_last_step_metrics()["num_steps_with_no_safe_policy"], 1)

    def test_mb_candidate_exception_is_runtime_error_not_certificate_loss(self):
        controller = MultiBackupCBFMinInterventionQuad3D(
            self.robot, self.spec, num_angle_policies=1
        )
        controller.set_environment(self.env)
        controller.update_obstacles([], self.env.get_static_obstacles())

        def errored_candidate(candidate, state, u_nom):
            del state, u_nom
            return CandidateCBFResult(
                policy_name=candidate.backup_controller.policy_name,
                feasible=False,
                u=None,
                objective=float("inf"),
                solver_status="error",
                rollout_safe=None,
                terminal_safe=None,
                solve_time_sec=0.0,
                error="forced candidate evaluation error",
            )

        with patch.object(
            type(next(iter(controller._candidate_filters.values()))),
            "solve_candidate",
            errored_candidate,
        ), self.assertRaisesRegex(RuntimeError, "certificate status is unknown"):
            controller.solve_control_problem(
                initial_state(),
                {
                    "u_ref": np.zeros(4),
                    "waypoints": self.env.get_nominal_waypoints(),
                    "wp_idx": 1,
                },
            )
        self.assertTrue(controller.runtime_error)
        self.assertFalse(controller.certificate_lost)
        self.assertFalse(controller.qp_infeasible)
        metrics = controller.get_last_step_metrics()
        self.assertEqual(metrics["candidate_evaluation_error_count"], 3)
        self.assertEqual(metrics["num_steps_with_no_certified_rollout"], 0)

    def test_mb_selector_is_order_independent_and_uses_fixed_index_tie_break(self):
        controller = MultiBackupCBFMinInterventionQuad3D(
            self.robot, self.spec, num_angle_policies=1
        )
        first_name, second_name = controller.policy_names[:2]
        first = CandidateCBFResult(
            first_name, True, np.zeros(4), 0.1 + 1e-9, "optimal", True, True, 0.0
        )
        second = CandidateCBFResult(
            second_name, True, np.zeros(4), 0.1, "optimal", True, True, 0.0
        )
        self.assertEqual(
            controller._select_minimum_intervention([first, second]).policy_name,
            first_name,
        )
        self.assertEqual(
            controller._select_minimum_intervention([second, first]).policy_name,
            first_name,
        )

    def test_mb_single_candidate_reduces_to_backup_cbf_with_corrected_terminal(self):
        controller = MultiBackupCBFMinInterventionQuad3D(
            self.robot, self.spec, num_angle_policies=1
        )
        controller.set_environment(self.env)
        controller.update_obstacles([], self.env.get_static_obstacles())
        candidate = controller._candidate_filters["stop"]
        adapter = controller._candidate_adapters["stop"]
        candidate.set_environment(self.env)
        candidate.set_moving_obstacles(None)

        legacy = BackupCBF(
            self.robot,
            self.spec,
            dt=controller.dt,
            backup_horizon=controller.backup_horizon,
        )
        legacy.alpha = controller.cbf_alpha
        legacy.alpha_terminal = controller.terminal_alpha
        legacy.safety_margin = controller.safety_margin
        legacy.set_backup_controller(adapter)
        legacy.set_environment(self.env)
        legacy.set_moving_obstacles(None)

        # The repository's generic terminal helper reads Quad3D yaw as speed.
        # Give the standalone one-policy reference the additive warehouse
        # terminal definition so this comparison isolates the strict wrapper
        # and minimum-intervention orchestration, rather than that known base
        # class indexing bug.
        legacy._h_terminal = candidate._h_terminal
        legacy._grad_h_terminal = candidate._grad_h_terminal
        legacy._integrate_backup_trajectory = candidate._integrate_backup_trajectory

        state = initial_state()
        state[2:12] = np.array(
            [0.2, 0.04, -0.03, 0.02, 0.35, -0.25, 0.1, 0.03, -0.02, 0.01]
        )
        nominal = np.array([0.4, -0.3, 0.2, -0.1])
        nominal_plan = np.repeat(nominal.reshape(1, -1), 4, axis=0)
        candidate.set_nominal_trajectory(None, nominal_plan)
        legacy.set_nominal_trajectory(None, nominal_plan)
        strict_result = candidate.solve_candidate(state, nominal)
        legacy_result = legacy.solve_control_problem(state).reshape(-1)

        self.assertTrue(strict_result.feasible)
        self.assertTrue(strict_result.rollout_safe)
        self.assertTrue(strict_result.terminal_safe)
        self.assertIs(strict_result.post_projection_constraints_satisfied, True)
        self.assertGreater(strict_result.constraint_audit_count, 0)
        self.assertLessEqual(
            strict_result.max_post_projection_violation_ratio, 1.0
        )
        np.testing.assert_allclose(strict_result.u, legacy_result, atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(
            candidate.latest_backup_trajectory,
            legacy.latest_backup_trajectory,
            atol=1e-8,
            rtol=1e-8,
        )

    def test_mb_rejects_failed_post_projection_constraint_audit(self):
        controller = MultiBackupCBFMinInterventionQuad3D(
            self.robot, self.spec, num_angle_policies=1
        )
        controller.set_environment(self.env)
        controller.update_obstacles([], self.env.get_static_obstacles())
        candidate = controller._candidate_filters["stop"]
        candidate.set_moving_obstacles(None)
        state = initial_state()
        state[2:12] = np.array(
            [0.2, 0.04, -0.03, 0.02, 0.35, -0.25, 0.1, 0.03, -0.02, 0.01]
        )
        nominal = np.array([0.4, -0.3, 0.2, -0.1])
        candidate.set_nominal_trajectory(
            None, np.repeat(nominal.reshape(1, -1), 4, axis=0)
        )

        with patch(
            "examples.warehouse.algorithms.multi_backup_cbf_mi_quad3d."
            "audit_cvxpy_inequalities",
            return_value=failed_constraint_audit(),
        ):
            result = candidate.solve_candidate(state, nominal)

        self.assertTrue(result.qp_solved)
        self.assertFalse(result.feasible)
        self.assertIsNone(result.u)
        self.assertIs(result.post_projection_constraints_satisfied, False)
        self.assertEqual(result.max_post_projection_violation_ratio, 2.0)
        self.assertEqual(result.error, "post-projection constraint audit failed")

    def test_mb_terminal_proxy_uses_velocity_and_rejects_nonhover_yaw(self):
        controller = MultiBackupCBFMinInterventionQuad3D(
            self.robot,
            self.spec,
            backup_horizon=0.1,
            maneuver_prefix_sec=0.05,
            num_angle_policies=1,
        )
        controller.set_environment(self.env)
        candidate = controller._candidate_filters["stop"]
        candidate.set_moving_obstacles(None)

        state = initial_state()
        state[6:9] = np.array([0.2, 0.0, 0.0])
        components = candidate._terminal_envelope_components(state)
        self.assertAlmostEqual(
            components["linear_speed"],
            controller.terminal_linear_speed_tol - 0.2,
            places=12,
        )
        self.assertGreater(candidate._h_terminal(state), 0.0)

        yaw_changed = state.copy()
        yaw_changed[5] = 2.4
        # The generic BackupCBF check v_max - abs(x[5]) would accept this yaw
        # as "low velocity"; the Quad3D proxy must reject it as non-hover
        # attitude while keeping the true linear-speed margin unchanged.
        self.assertGreater(self.spec["v_max"] - abs(yaw_changed[5]), 0.0)
        yaw_components = candidate._terminal_envelope_components(yaw_changed)
        self.assertAlmostEqual(
            yaw_components["linear_speed"], components["linear_speed"], places=12
        )
        self.assertLess(yaw_components["attitude"], 0.0)
        self.assertLess(candidate._h_terminal(yaw_changed), 0.0)

        angle_equilibrium = initial_state()
        angle_equilibrium[6] = self.spec["v_ref"]
        self.assertLess(candidate._h_terminal(angle_equilibrium), 0.0)

        too_fast_xy = state.copy()
        too_fast_xy[6:9] = np.array(
            [controller.terminal_linear_speed_tol + 0.2, 0.0, 0.0]
        )
        self.assertLess(candidate._h_terminal(too_fast_xy), 0.0)

        too_fast_vertical = state.copy()
        too_fast_vertical[6:9] = np.array(
            [0.0, 0.0, controller.terminal_linear_speed_tol + 0.2]
        )
        self.assertLess(candidate._h_terminal(too_fast_vertical), 0.0)

        tilted = initial_state()
        tilted[3] = controller.terminal_attitude_tol + 0.1
        self.assertLess(candidate._h_terminal(tilted), 0.0)

        rotating = initial_state()
        rotating[9] = controller.terminal_angular_rate_tol + 0.1
        self.assertLess(candidate._h_terminal(rotating), 0.0)

        off_altitude = initial_state()
        off_altitude[2] = (
            self.spec["z_ref"] + controller.terminal_altitude_error_tol + 0.1
        )
        self.assertLess(candidate._h_terminal(off_altitude), 0.0)

    def test_mb_uncertified_candidates_are_skipped_before_qp_construction(self):
        controller = MultiBackupCBFMinInterventionQuad3D(
            self.robot,
            self.spec,
            backup_horizon=0.05,
            maneuver_prefix_sec=0.0,
            num_angle_policies=1,
        )
        controller.set_environment(self.env)
        candidate = controller._candidate_filters["stop"]
        candidate.set_moving_obstacles(None)

        unsafe_state = initial_state()
        unsafe_state[0] = 0.0
        with patch.object(
            candidate, "_compute_rollout_sensitivities"
        ) as sensitivity_fn, patch(
            "examples.warehouse.algorithms.multi_backup_cbf_mi_quad3d.cp.Problem"
        ) as problem_cls:
            unsafe_result = candidate.solve_candidate(unsafe_state, np.zeros(4))
        self.assertFalse(unsafe_result.feasible)
        self.assertFalse(unsafe_result.rollout_safe)
        self.assertFalse(unsafe_result.qp_solved)
        self.assertEqual(unsafe_result.solver_status, "uncertified_rollout")
        sensitivity_fn.assert_not_called()
        problem_cls.assert_not_called()

        # Isolate terminal certification from raw rollout safety by supplying a
        # safe-position rollout whose terminal translational speed exceeds the
        # common near-hover envelope.
        fast_terminal = initial_state()
        fast_terminal[6] = controller.terminal_linear_speed_tol + 0.5
        phi = np.repeat(fast_terminal.reshape(1, -1), candidate.N, axis=0)
        sensitivities = np.repeat(
            np.eye(candidate.n_states).reshape(1, candidate.n_states, candidate.n_states),
            candidate.N,
            axis=0,
        )
        with patch.object(
            candidate,
            "_integrate_state_trajectory",
            return_value=phi,
        ), patch.object(
            candidate, "_compute_rollout_sensitivities", return_value=sensitivities
        ) as sensitivity_fn, patch(
            "examples.warehouse.algorithms.multi_backup_cbf_mi_quad3d.cp.Problem"
        ) as problem_cls:
            terminal_result = candidate.solve_candidate(
                initial_state(), np.zeros(4)
            )
        self.assertFalse(terminal_result.feasible)
        self.assertTrue(terminal_result.rollout_safe)
        self.assertFalse(terminal_result.terminal_safe)
        self.assertFalse(terminal_result.qp_solved)
        self.assertEqual(terminal_result.solver_status, "uncertified_terminal")
        sensitivity_fn.assert_not_called()
        problem_cls.assert_not_called()

    def test_mb_vectorized_dynamic_barrier_matches_existing_backup_cbf(self):
        controller = MultiBackupCBFMinInterventionQuad3D(
            self.robot, self.spec, num_angle_policies=1
        )
        controller.set_environment(self.env)
        candidate = controller._candidate_filters["stop"]
        predictor = _FrozenGhostPredictor(
            [
                {"x": 30.0, "y": 15.0, "radius": 1.3, "vx": -0.4, "vy": 0.2},
                {"x": 80.0, "y": 70.0, "radius": 2.0, "vx": 0.1, "vy": -0.3},
            ]
        )
        candidate.set_moving_obstacles(predictor)

        legacy = BackupCBF(
            self.robot,
            self.spec,
            dt=controller.dt,
            backup_horizon=controller.backup_horizon,
        )
        legacy.safety_margin = controller.safety_margin
        legacy.set_environment(self.env)
        legacy.set_moving_obstacles(predictor)

        state = initial_state()
        state[:2] = (28.0, 14.0)
        for time_value in (0.0, 0.7, 3.9):
            self.assertAlmostEqual(
                candidate._h_safety(state, time_value),
                legacy._h_safety(state, time_value),
                places=12,
            )
            np.testing.assert_allclose(
                candidate._grad_h_safety(state, time_value),
                legacy._grad_h_safety(state, time_value),
                atol=1e-10,
                rtol=1e-10,
            )

    def test_mb_linear_rk4_step_matches_robot_including_angle_wrap(self):
        controller = MultiBackupCBFMinInterventionQuad3D(
            self.robot, self.spec, num_angle_policies=1
        )
        candidate = controller._candidate_filters["stop"]
        rng = np.random.default_rng(9)
        for index in range(5):
            state = rng.normal(scale=0.4, size=12)
            if index == 0:
                state[3:6] = np.array(
                    [np.pi - 1e-4, -np.pi + 1e-4, np.pi - 2e-4]
                )
                state[9:12] = np.array([2.0, -2.0, 2.5])
            control = rng.uniform(-2.0, 2.0, size=4)
            expected = np.asarray(
                self.robot.step(state.reshape(-1, 1), control.reshape(-1, 1))
            ).reshape(-1)
            actual = candidate._linear_rk4_step(state, control)
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_lib_no_certified_policy_returns_bounded_emergency_and_failure_flag(self):
        controller = LibraryPCBFMinInterventionQuad3D(
            self.spec, num_angle_policies=1
        )
        names = controller.policy_names
        fake_results = OrderedDict(
            (
                name,
                (-1.0, np.zeros(12), np.zeros((2, 12))),
            )
            for name in names
        )
        fake_time = {name: 0.0 for name in names}
        with patch.object(
            controller,
            "_evaluate_policy_library",
            return_value=(fake_results, {}, fake_time),
        ):
            control = controller.solve_control_problem(
                initial_state(), {"u_ref": np.zeros(4)}
            )
        self.assertFalse(controller.infeasible)
        self.assertTrue(controller.certificate_lost)
        self.assertFalse(controller.qp_infeasible)
        self.assertTrue(np.all(np.isfinite(control)))
        self.assertTrue(np.all(control <= self.spec["u_max"] + 1e-8))
        self.assertTrue(np.all(control >= self.spec["u_min"] - 1e-8))


if __name__ == "__main__":
    unittest.main()
