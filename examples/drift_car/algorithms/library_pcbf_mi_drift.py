"""Library PCBF with minimum-intervention selection for the drift car.

This is the controlled selector ablation described in the accompanying paper:
it reuses :class:`PLCBF`'s policy construction and exact value/gradient rollout
path, but solves the same PCBF-QP for every certified policy and selects the
realized minimum-intervention solution.  It is not presented as an unchanged
algorithm from a prior paper.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import cvxpy as cp
import jax.numpy as jnp
import numpy as np

from examples.additional_baseline_control_utils import (
    audit_cvxpy_inequalities,
    project_solver_control_with_diagnostics,
)
from examples.drift_car.algorithms.multi_policy_baseline_common_drift import (
    CandidateCBFResult,
    MultiPolicyMetrics,
    NOMINAL_POLICY_REPRESENTATION,
    assert_runtime_library_equal,
    select_minimum_intervention,
)
from examples.drift_car.algorithms.plcbf_drift import PLCBF, POLICY_ALPHA
from examples.drift_car.controllers.drift_policies_jax import StoppingControllerJAX


_SCS_AUDIT_ATOL = 1e-4
_SCS_AUDIT_RTOL = 1e-4


class LibraryPCBFMinInterventionDrift(PLCBF):
    """PL-CBF certificate library with realized minimum-intervention selection."""

    algorithm_key = "library_pcbf_mi"
    nominal_policy_representation = NOMINAL_POLICY_REPRESENTATION

    def __init__(
        self,
        robot,
        robot_spec: dict,
        dt: float = 0.05,
        backup_horizon: float = 10.0,
        cbf_alpha: float = 5.0,
        left_lane_y: float = 5.0,
        right_lane_y: float = -5.0,
        safety_margin: float = 0.0,
        debug: bool = False,
        ax=None,
        *,
        reference_plcbf: PLCBF,
    ):
        super().__init__(
            robot=robot,
            robot_spec=robot_spec,
            dt=dt,
            backup_horizon=backup_horizon,
            cbf_alpha=cbf_alpha,
            left_lane_y=left_lane_y,
            right_lane_y=right_lane_y,
            safety_margin=safety_margin,
            max_operator="input_space",
            debug=debug,
            ax=ax,
        )
        self.policy_names = tuple(self.policy_configs.keys()) + ("nominal",)
        self.metrics = MultiPolicyMetrics()
        self.infeasible = False
        self.qp_infeasible = False
        self.certificate_lost = False
        self.fallback_applied = False
        self.last_candidate_results: list[CandidateCBFResult] = []
        self.last_values: Dict[str, float] = {}
        self.last_gradients: Dict[str, np.ndarray] = {}
        self.last_trajectories: Dict[str, Optional[np.ndarray]] = {}
        self._using_backup = False
        self.assert_library_matches(reference_plcbf)

    def assert_library_matches(self, reference_plcbf: PLCBF) -> None:
        assert_runtime_library_equal(self, reference_plcbf)

    def _policy_alpha(self, policy_name: str) -> float:
        instance_alphas = getattr(self, "alphas", {})
        if policy_name in instance_alphas:
            return float(instance_alphas[policy_name])
        return float(POLICY_ALPHA.get(policy_name, self.cbf_alpha))

    def _intervention_objective(self, u: np.ndarray, u_nom: np.ndarray) -> float:
        # This is exactly PCBF._solve_cbf_qp's normalized weighted objective.
        weights = np.array([1.0, 10.0], dtype=float)
        difference_scaled = (np.asarray(u) - np.asarray(u_nom)) / self.u_max
        return float(np.sum(np.square(weights * difference_scaled)))

    def _solve_policy_candidate(
        self,
        policy_name: str,
        u_nom: np.ndarray,
        value: float,
        gradient: np.ndarray,
        f: np.ndarray,
        G: np.ndarray,
    ) -> CandidateCBFResult:
        started = time.perf_counter()
        gradient = np.asarray(gradient, dtype=float).copy()
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > 50.0:
            gradient *= 50.0 / gradient_norm

        try:
            self.cbf_alpha = self._policy_alpha(policy_name)
            u_scale = self.u_max
            u_nom_scaled = np.asarray(u_nom, dtype=float).reshape(-1) / u_scale
            u_scaled = cp.Variable(2)
            cbf_slack = cp.Variable(nonneg=True) if self.use_cbf_slack else None

            weights = np.array([1.0, 10.0])
            weighted_diff = cp.multiply(weights, u_scaled - u_nom_scaled)
            cost = cp.sum_squares(weighted_diff)
            constraints = [u_scaled >= -1.0, u_scaled <= 1.0]
            if cbf_slack is not None:
                cost += self.cbf_slack_weight * cp.square(cbf_slack)

            grad_v_g = gradient @ G
            grad_v_f = gradient @ f
            cbf_rhs = grad_v_f + self.cbf_alpha * value
            a_cbf = -grad_v_g * u_scale
            if cbf_slack is not None:
                constraints.insert(
                    0, a_cbf @ u_scaled <= cbf_rhs + 1e-4 + cbf_slack
                )
            else:
                constraints.insert(0, a_cbf @ u_scaled <= cbf_rhs + 1e-4)

            problem = cp.Problem(cp.Minimize(cost), constraints)
            problem.solve(
                solver=cp.SCS,
                verbose=False,
                max_iters=2000,
                eps=_SCS_AUDIT_ATOL,
            )
            if problem.status not in ("optimal", "optimal_inaccurate"):
                self.status = str(problem.status)
                raise ValueError(f"CBF-QP failed: {problem.status}")
            if u_scaled.value is None:
                self.status = "invalid_solution"
                raise ValueError("CBF-QP returned no control")

            slack_value = (
                None
                if cbf_slack is None or cbf_slack.value is None
                else float(np.asarray(cbf_slack.value).reshape(-1)[0])
            )
            if cbf_slack is not None and slack_value is None:
                self.status = "invalid_solution"
                raise ValueError("CBF-QP returned no slack value")
            self.status = (
                "optimal_with_slack"
                if slack_value is not None and slack_value > 1e-7
                else "optimal"
            )
            solver_status = str(self.status)
            solver_name = str(problem.solver_stats.solver_name)
            raw_control = np.asarray(u_scaled.value * u_scale, dtype=float).reshape(-1)
            projection = project_solver_control_with_diagnostics(
                raw_control,
                self.u_min,
                self.u_max,
                expected_shape=(2,),
            )
            if projection.control is None:
                return CandidateCBFResult(
                    policy_name=policy_name,
                    feasible=False,
                    u=None,
                    objective=float("inf"),
                    solver_status=solver_status,
                    rollout_safe=True,
                    terminal_safe=None,
                    solve_time_sec=time.perf_counter() - started,
                    qp_solved=True,
                    error="QP returned an invalid or out-of-bounds input",
                    raw_u=raw_control.copy(),
                    solver_name=solver_name,
                    cbf_slack=slack_value,
                )

            assignments = [(u_scaled, projection.control / u_scale)]
            if cbf_slack is not None:
                assignments.append((cbf_slack, slack_value))
            audit = audit_cvxpy_inequalities(
                constraints,
                assignments,
                absolute_tolerance=_SCS_AUDIT_ATOL,
                relative_tolerance=_SCS_AUDIT_RTOL,
            )
            feasible = audit.passed
            u = projection.control if feasible else None
            error = None if feasible else "post-projection constraint audit failed"
            objective = (
                self._intervention_objective(u, u_nom) if feasible else float("inf")
            )
        except Exception as exc:
            u = None
            feasible = False
            solver_status = str(self.status)
            objective = float("inf")
            error = str(exc)
            raw_control = None
            projection = None
            audit = None
            solver_name = None
            slack_value = None

        return CandidateCBFResult(
            policy_name=policy_name,
            feasible=feasible,
            u=u,
            objective=objective,
            solver_status=solver_status,
            rollout_safe=True,
            terminal_safe=None,
            solve_time_sec=time.perf_counter() - started,
            qp_solved=True,
            error=error,
            raw_u=None if raw_control is None else raw_control.copy(),
            projected_u=(
                None
                if projection is None or projection.control is None
                else projection.control.copy()
            ),
            projection_occurred=bool(
                projection is not None and projection.projection_applied
            ),
            projection_delta_inf=(
                0.0 if projection is None else projection.projection_delta_inf
            ),
            post_projection_constraints_satisfied=(
                None if audit is None else audit.passed
            ),
            max_post_projection_constraint_violation=(
                None if audit is None else audit.max_violation
            ),
            max_post_projection_violation_ratio=(
                None if audit is None else audit.max_violation_ratio
            ),
            constraint_audit_atol=(
                None if audit is None else audit.absolute_tolerance
            ),
            constraint_audit_rtol=(
                None if audit is None else audit.relative_tolerance
            ),
            constraint_audit_count=0 if audit is None else audit.constraint_count,
            solver_name=solver_name,
            cbf_slack=slack_value,
        )

    def _emergency_control(self, robot_state: np.ndarray) -> np.ndarray:
        params = self.policy_configs["stop"]["params"]
        u = np.asarray(
            StoppingControllerJAX.compute(jnp.asarray(robot_state), params),
            dtype=float,
        ).reshape(-1)
        return np.clip(u, self.u_min, self.u_max)

    def _record_trajectories(self, trajectories: Dict[str, Optional[np.ndarray]]) -> None:
        for name, trajectory in trajectories.items():
            if trajectory is not None and self.curr_step % self.save_every_N == 0:
                self.multi_backup_trajs.setdefault(name, []).append(trajectory.copy())

    def solve_control_problem(
        self,
        robot_state: np.ndarray,
        control_ref: Optional[dict] = None,
        friction: Optional[float] = None,
        nominal_trajectory: Optional[np.ndarray] = None,
        nominal_controls: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate the shared PCBF library and select minimum intervention."""

        robot_state = np.asarray(robot_state, dtype=float).reshape(-1)
        if friction is not None:
            self.set_friction(friction)
        if nominal_trajectory is not None:
            # The MPCC plan is copied once; candidate evaluation never advances MPCC.
            self.set_nominal_trajectory(
                np.array(nominal_trajectory, copy=True),
                None if nominal_controls is None else np.array(nominal_controls, copy=True),
            )
        u_nom = (
            np.asarray(control_ref["u_ref"], dtype=float).reshape(-1)
            if control_ref is not None and "u_ref" in control_ref
            else np.zeros(2, dtype=float)
        )
        self._update_obstacles()
        self.infeasible = False
        self.qp_infeasible = False
        self.certificate_lost = False
        self.fallback_applied = False
        self.best_policy_name = None
        self.last_candidate_results = []

        if not self.obstacles:
            applied_nominal = np.clip(u_nom, self.u_min, self.u_max)
            self.status = "optimal"
            self.best_policy_name = "nominal"
            self._using_backup = False
            self.metrics.feasible_candidates_per_step.append(1)
            self.metrics.rollout_safe_candidates_per_step.append(1)
            self.metrics.qp_feasible_candidates_per_step.append(1)
            self.metrics.record_selection(
                "nominal", float(np.linalg.norm(applied_nominal - u_nom))
            )
            return applied_nominal.reshape(-1, 1)

        x0_jax = jnp.asarray(robot_state)
        try:
            values, gradients, trajectories = self._compute_multi_value_and_grad(x0_jax)
        except Exception as exc:
            self.status = f"value_error: {exc}"
            self.infeasible = True
            self.certificate_lost = True
            self.fallback_applied = True
            self._using_backup = True
            self.metrics.num_steps_with_no_safe_policy += 1
            return self._emergency_control(robot_state).reshape(-1, 1)

        # These are direct copies of the values used by PL-CBF, exposed for
        # regression/fairness checks and benchmark diagnostics.
        self.last_values = dict(values)
        self.last_gradients = {name: np.array(value, copy=True) for name, value in gradients.items()}
        self.last_trajectories = {
            name: None if value is None else np.array(value, copy=True)
            for name, value in trajectories.items()
        }

        f = np.asarray(self.dynamics_jax.f_full(x0_jax, self.current_friction))
        G = np.asarray(self.dynamics_jax.g_full(x0_jax))
        safe_names = [name for name in self.policy_names if values[name] > 0.0]

        for name in safe_names:
            self.last_candidate_results.append(
                self._solve_policy_candidate(name, u_nom, values[name], gradients[name], f, G)
            )

        self.metrics.record_projection_audits(self.last_candidate_results)
        self.metrics.num_candidate_qps_solved += sum(
            result.qp_solved for result in self.last_candidate_results
        )
        feasible_count = sum(result.feasible for result in self.last_candidate_results)
        self.metrics.feasible_candidates_per_step.append(feasible_count)
        self.metrics.rollout_safe_candidates_per_step.append(len(safe_names))
        self.metrics.qp_feasible_candidates_per_step.append(feasible_count)
        best = select_minimum_intervention(self.last_candidate_results, self.policy_names)

        self._record_trajectories(trajectories)
        self.curr_step += 1

        if best is None:
            self.best_policy_name = None
            self.certificate_lost = len(safe_names) == 0
            self.qp_infeasible = len(safe_names) > 0
            self.infeasible = self.qp_infeasible
            self.fallback_applied = True
            self.status = (
                "certificate_lost_no_policy"
                if self.certificate_lost
                else "qp_infeasible_no_policy"
            )
            self._using_backup = True
            self.metrics.num_steps_with_no_safe_policy += 1
            self._update_multi_visualization(trajectories, "")
            return self._emergency_control(robot_state).reshape(-1, 1)

        self.best_policy_name = best.policy_name
        self.cbf_alpha = self._policy_alpha(best.policy_name)
        self.status = best.solver_status
        intervention_l2 = np.linalg.norm(np.asarray(best.u) - u_nom)
        normalized_delta = (np.asarray(best.u) - u_nom) / np.maximum(
            self.u_max, 1e-12
        )
        self._using_backup = bool(
            np.linalg.norm(np.array([1.0, 10.0]) * normalized_delta) > 0.1
        )
        self.metrics.record_selection(best.policy_name, intervention_l2)
        self._update_multi_visualization(trajectories, best.policy_name)
        return np.asarray(best.u).reshape(-1, 1)

    def get_metrics(self) -> Dict[str, object]:
        return self.metrics.as_dict()

    def is_using_backup(self) -> bool:
        return bool(self._using_backup)

    def get_status(self):
        status = super().get_status()
        status.update(
            {
                "algorithm": self.algorithm_key,
                "best_policy": self.best_policy_name,
                "infeasible": self.infeasible,
                "qp_infeasible": self.qp_infeasible,
                "certificate_lost": self.certificate_lost,
                "fallback_applied": self.fallback_applied,
                "num_candidate_qps_solved": sum(
                    result.qp_solved for result in self.last_candidate_results
                ),
            }
        )
        return status
