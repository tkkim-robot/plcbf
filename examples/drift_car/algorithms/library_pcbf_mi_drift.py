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

import jax.numpy as jnp
import numpy as np

from examples.drift_car.algorithms.multi_policy_baseline_common_drift import (
    CandidateCBFResult,
    MultiPolicyMetrics,
    NOMINAL_POLICY_REPRESENTATION,
    assert_runtime_library_equal,
    select_minimum_intervention,
    valid_bounded_control,
)
from examples.drift_car.algorithms.plcbf_drift import PLCBF, POLICY_ALPHA
from examples.drift_car.controllers.drift_policies_jax import StoppingControllerJAX


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
            u = np.asarray(
                self._solve_cbf_qp(u_nom, value, gradient, f, G),
                dtype=float,
            ).reshape(-1)
            solver_status = str(self.status)
            feasible = valid_bounded_control(u, self.u_min, self.u_max)
            error = None if feasible else "QP returned an invalid or out-of-bounds input"
            objective = self._intervention_objective(u, u_nom) if feasible else float("inf")
        except Exception as exc:
            u = None
            feasible = False
            solver_status = str(self.status)
            objective = float("inf")
            error = str(exc)

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
            self.status = "optimal"
            self.best_policy_name = "nominal"
            self._using_backup = False
            self.metrics.feasible_candidates_per_step.append(1)
            self.metrics.rollout_safe_candidates_per_step.append(1)
            self.metrics.qp_feasible_candidates_per_step.append(1)
            self.metrics.record_selection("nominal", 0.0)
            return u_nom.reshape(-1, 1)

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
