"""Library PCBF with minimum-intervention selection for Quad3D."""

from dataclasses import dataclass
import time
from typing import Optional, Sequence, Tuple

import cvxpy as cp
import jax.numpy as jnp
import numpy as np

from examples.additional_baseline_control_utils import audit_cvxpy_inequalities
from examples.warehouse.controllers.policies_quad3d_jax import (
    StopPolicyJAX,
    WaypointPolicyParams,
)
from .additional_baseline_control_quad3d import (
    SOLVER_INPUT_TOL,
    project_quad3d_solver_control_with_diagnostics,
)
from .plcbf_quad3d import PLCBF_Quad3D


_TIE_TOL = 1e-6
_INPUT_TOL = SOLVER_INPUT_TOL
_QP_AUDIT_ATOL = _INPUT_TOL
_QP_AUDIT_RTOL = _INPUT_TOL
_ACCEPTED_STATUSES = ("optimal", "optimal_inaccurate")


@dataclass(frozen=True)
class CandidatePCBFResult:
    policy_name: str
    policy_index: int
    certified: bool
    value: float
    feasible: bool
    u: Optional[np.ndarray]
    objective: float
    intervention_l2: float
    solver_status: str
    solve_time_sec: float
    error: Optional[str] = None
    raw_u: Optional[np.ndarray] = None
    projected_u: Optional[np.ndarray] = None
    projection_occurred: bool = False
    projection_delta_inf: float = 0.0
    post_projection_constraints_satisfied: Optional[bool] = None
    max_post_projection_constraint_violation: Optional[float] = None
    max_post_projection_violation_ratio: Optional[float] = None
    constraint_audit_atol: Optional[float] = None
    constraint_audit_rtol: Optional[float] = None
    constraint_audit_count: int = 0
    solver_name: Optional[str] = None


class LibraryPCBFMinInterventionQuad3D(PLCBF_Quad3D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.policy_names: Tuple[str, ...] = tuple(self.policy_configs.keys())
        if tuple(self.angle_names) + ("stop", "nominal") != self.policy_names:
            raise AssertionError(
                "Lib-PCBF-MI must inherit the exact ordered PL-CBF policy library"
            )

        self.infeasible = False
        self.qp_infeasible = False
        self.certificate_lost = False
        self.fallback_applied = False
        self.runtime_error = False

    def _evaluate_policy_library(self, state, control_ref=None):
        results = {}
        policy_params_used = {}
        time_derivatives = {}

        if self.dynamic_obstacles:
            obs_array = jnp.array([
                (o["x"], o["y"], o["radius"], o.get("vx", 0.0), o.get("vy", 0.0))
                for o in self.dynamic_obstacles
            ])
        else:
            obs_array = jnp.zeros((0, 5))

        if self.static_obstacles:
            stat_obs_array = jnp.array([
                (o["x"], o["y"], o["radius"]) for o in self.static_obstacles
            ])
        else:
            stat_obs_array = jnp.zeros((0, 3))

        val_grad_fn = self._get_jit_val_grad()
        grad_obs_fn = self._get_jit_val_grad_obs()
        state_jax = jnp.array(state)

        robot_radius = self.robot_spec.get("radius", 1.0) + self.safety_margin
        robot_radius_base = self.robot_spec.get("radius", 1.0)

        if self.angle_params_batch is not None and self.angle_names:
            batch_val_grad_fn, batch_grad_obs_fn = self._get_jit_angle_batch()
            (values_batch, trajectories_batch), gradients_batch = batch_val_grad_fn(
                state_jax,
                self.dynamics_params,
                self.angle_params_batch,
                obs_array,
                stat_obs_array,
                robot_radius,
                robot_radius_base,
            )
            if obs_array.shape[0] > 0:
                obstacle_gradients_batch = batch_grad_obs_fn(
                    state_jax,
                    self.dynamics_params,
                    self.angle_params_batch,
                    obs_array,
                    stat_obs_array,
                    robot_radius,
                    robot_radius_base,
                )
                obstacle_velocities = obs_array[:, 3:5]
                time_derivatives_batch = jnp.sum(
                    obstacle_gradients_batch[:, :, 0:2]
                    * obstacle_velocities[None, :, :],
                    axis=(1, 2),
                )
            else:
                time_derivatives_batch = jnp.zeros((len(self.angle_names),))

            for index, name in enumerate(self.angle_names):
                params = self.policy_configs[name][1]
                policy_params_used[name] = ("angle", params)
                results[name] = (
                    float(values_batch[index]),
                    np.array(gradients_batch[index]),
                    np.array(trajectories_batch[index]),
                )
                time_derivatives[name] = float(time_derivatives_batch[index])

        for name in ("stop", "nominal"):
            if name not in self.policy_configs:
                continue
            policy_type, params = self.policy_configs[name]
            if name == "nominal" and control_ref is not None and "waypoints" in control_ref:
                params = WaypointPolicyParams(
                    waypoints=jnp.array(np.array(control_ref["waypoints"], copy=True)),
                    v_max=float(self.robot_spec.get("v_max", 5.0)),
                    Kp=float(self.robot_spec.get("nominal_Kp_v", 6.0)),
                    K_lat=float(self.robot_spec.get("nominal_K_lat", 1.0)),
                    v_lat_max=float(
                        self.robot_spec.get(
                            "nominal_v_lat_max", self.robot_spec.get("v_ref", 4.0)
                        )
                    ),
                    dist_threshold=float(
                        self.robot_spec.get("nominal_dist_threshold", 0.8)
                    ),
                    current_wp_idx=int(control_ref.get("wp_idx", 0)),
                    ctrl=params.ctrl,
                )
            policy_params_used[name] = (policy_type, params)

            (value_jax, trajectory), gradient_jax = val_grad_fn(
                state_jax,
                self.dynamics_params,
                params,
                obs_array,
                stat_obs_array,
                policy_type,
                self.eval_horizon_steps,
                robot_radius,
                robot_radius_base,
                self.dt,
            )
            results[name] = (
                float(value_jax),
                np.array(gradient_jax),
                np.array(trajectory),
            )

            if obs_array.shape[0] > 0:
                obstacle_gradient = grad_obs_fn(
                    state_jax,
                    self.dynamics_params,
                    params,
                    obs_array,
                    stat_obs_array,
                    policy_type,
                    self.eval_horizon_steps,
                    robot_radius,
                    robot_radius_base,
                    self.dt,
                )
                obstacle_velocities = obs_array[:, 3:5]
                time_derivatives[name] = float(
                    jnp.sum(obstacle_gradient[:, 0:2] * obstacle_velocities)
                )
            else:
                time_derivatives[name] = 0.0

        if tuple(results.keys()) != self.policy_names:
            raise AssertionError("evaluated policy library differs from PL-CBF ordering")
        return results, policy_params_used, time_derivatives

    def _solve_candidate_qp(
        self,
        policy_name: str,
        policy_index: int,
        state: np.ndarray,
        u_nom: np.ndarray,
        value: float,
        gradient: np.ndarray,
        time_derivative: float,
    ) -> CandidatePCBFResult:
        if not (
            np.isfinite(value)
            and value > 0.0
            and np.all(np.isfinite(gradient))
            and np.isfinite(time_derivative)
        ):
            return CandidatePCBFResult(
                policy_name=policy_name,
                policy_index=policy_index,
                certified=bool(np.isfinite(value) and value > 0.0),
                value=value,
                feasible=False,
                u=None,
                objective=float("inf"),
                intervention_l2=float("inf"),
                solver_status="invalid_certificate",
                solve_time_sec=0.0,
            )

        u = cp.Variable(4)
        cost = cp.sum_squares(u - u_nom)
        constraints = []
        self._add_input_constraints(u, constraints)

        self._last_time_derivative = time_derivative
        self._add_cbf_constraints(u, constraints, state, value, gradient)
        problem = cp.Problem(cp.Minimize(cost), constraints)

        start = time.perf_counter()
        error = None
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            solver_status = str(problem.status)
        except Exception as exc:
            solver_status = "solver_error"
            error = f"{type(exc).__name__}: {exc}"
        solve_time_sec = time.perf_counter() - start

        solution = None if u.value is None else np.asarray(u.value, dtype=float).reshape(-1)
        lower = float(self.dynamics_params.u_min)
        upper = float(self.dynamics_params.u_max)
        solver_solution_valid = (
            solver_status in _ACCEPTED_STATUSES
            and solution is not None
            and solution.shape == (4,)
            and np.all(np.isfinite(solution))
            and np.all(solution >= lower - _INPUT_TOL)
            and np.all(solution <= upper + _INPUT_TOL)
        )

        if not solver_solution_valid:
            if error is None and solver_status in _ACCEPTED_STATUSES:
                error = "solver returned an invalid or out-of-bounds control"
            return CandidatePCBFResult(
                policy_name=policy_name,
                policy_index=policy_index,
                certified=True,
                value=value,
                feasible=False,
                u=None,
                objective=float("inf"),
                intervention_l2=float("inf"),
                solver_status=solver_status,
                solve_time_sec=solve_time_sec,
                error=error,
                raw_u=None if solution is None else solution.copy(),
                solver_name=(
                    None
                    if problem.solver_stats is None
                    else str(problem.solver_stats.solver_name)
                ),
            )

        projection = project_quad3d_solver_control_with_diagnostics(
            solution, lower, upper
        )
        if projection.control is None:
            raise AssertionError("validated Quad3D QP solution could not be projected")
        audit = audit_cvxpy_inequalities(
            constraints,
            [(u, projection.control)],
            absolute_tolerance=_QP_AUDIT_ATOL,
            relative_tolerance=_QP_AUDIT_RTOL,
        )
        solver_name = str(problem.solver_stats.solver_name)
        if not audit.passed:
            return CandidatePCBFResult(
                policy_name=policy_name,
                policy_index=policy_index,
                certified=True,
                value=value,
                feasible=False,
                u=None,
                objective=float("inf"),
                intervention_l2=float("inf"),
                solver_status=solver_status,
                solve_time_sec=solve_time_sec,
                error="post-projection constraint audit failed",
                raw_u=solution.copy(),
                projected_u=projection.control.copy(),
                projection_occurred=projection.projection_applied,
                projection_delta_inf=projection.projection_delta_inf,
                post_projection_constraints_satisfied=False,
                max_post_projection_constraint_violation=audit.max_violation,
                max_post_projection_violation_ratio=audit.max_violation_ratio,
                constraint_audit_atol=audit.absolute_tolerance,
                constraint_audit_rtol=audit.relative_tolerance,
                constraint_audit_count=audit.constraint_count,
                solver_name=solver_name,
            )
        projected_solution = projection.control
        delta = projected_solution - u_nom
        objective = float(np.dot(delta, delta))
        return CandidatePCBFResult(
            policy_name=policy_name,
            policy_index=policy_index,
            certified=True,
            value=value,
            feasible=True,
            u=projected_solution.copy(),
            objective=objective,
            intervention_l2=float(np.sqrt(max(objective, 0.0))),
            solver_status=solver_status,
            solve_time_sec=solve_time_sec,
            raw_u=solution.copy(),
            projected_u=projected_solution.copy(),
            projection_occurred=projection.projection_applied,
            projection_delta_inf=projection.projection_delta_inf,
            post_projection_constraints_satisfied=True,
            max_post_projection_constraint_violation=audit.max_violation,
            max_post_projection_violation_ratio=audit.max_violation_ratio,
            constraint_audit_atol=audit.absolute_tolerance,
            constraint_audit_rtol=audit.relative_tolerance,
            constraint_audit_count=audit.constraint_count,
            solver_name=solver_name,
        )

    @staticmethod
    def _select_minimum_intervention(
        candidates: Sequence[CandidatePCBFResult],
    ) -> Optional[CandidatePCBFResult]:
        best = None
        for candidate in candidates:
            if not candidate.feasible:
                continue
            if best is None:
                best = candidate
                continue
            if candidate.objective < best.objective - _TIE_TOL:
                best = candidate
            elif (
                abs(candidate.objective - best.objective) <= _TIE_TOL
                and candidate.policy_index < best.policy_index
            ):
                best = candidate
        return best

    def _emergency_control(self, state: np.ndarray) -> np.ndarray:
        stop_type, stop_params = self.policy_configs["stop"]
        if stop_type != "stop":
            raise AssertionError("the inherited PL-CBF stop policy was replaced")
        emergency = np.asarray(
            StopPolicyJAX.compute(jnp.array(state), stop_params), dtype=float
        ).reshape(-1)
        if emergency.shape != (4,) or not np.all(np.isfinite(emergency)):
            emergency = np.zeros(4, dtype=float)
        return np.clip(
            emergency,
            float(self.dynamics_params.u_min),
            float(self.dynamics_params.u_max),
        )

    def solve_control_problem(self, state, control_ref=None):
        self.runtime_error = False
        state = np.asarray(state, dtype=float).reshape(-1).copy()
        if control_ref and "u_ref" in control_ref:
            u_nom = np.asarray(control_ref["u_ref"], dtype=float).reshape(-1).copy()
        else:
            u_nom = np.zeros(4, dtype=float)
        if u_nom.shape != (4,) or not np.all(np.isfinite(u_nom)):
            raise ValueError("control_ref['u_ref'] must be a finite four-vector")

        results, _policy_params_used, time_derivatives = self._evaluate_policy_library(
            state, control_ref
        )

        candidate_results = []
        rollout_safe_count = 0

        for policy_index, name in enumerate(self.policy_names):
            value, gradient, _trajectory = results[name]
            certified = bool(np.isfinite(value) and value > 0.0)
            if not certified:
                candidate_results.append(
                    CandidatePCBFResult(
                        policy_name=name,
                        policy_index=policy_index,
                        certified=False,
                        value=float(value),
                        feasible=False,
                        u=None,
                        objective=float("inf"),
                        intervention_l2=float("inf"),
                        solver_status="not_certified",
                        solve_time_sec=0.0,
                    )
                )
                continue

            rollout_safe_count += 1
            candidate = self._solve_candidate_qp(
                policy_name=name,
                policy_index=policy_index,
                state=state,
                u_nom=u_nom,
                value=float(value),
                gradient=np.asarray(gradient),
                time_derivative=float(time_derivatives.get(name, 0.0)),
            )
            candidate_results.append(candidate)

        best = self._select_minimum_intervention(candidate_results)

        if best is None:
            self._last_best_name = None
            self.infeasible = rollout_safe_count > 0
            self.qp_infeasible = self.infeasible
            self.certificate_lost = rollout_safe_count == 0
            self.fallback_applied = True
            control = self._emergency_control(state)
        else:
            self._last_best_name = best.policy_name
            self._last_time_derivative = time_derivatives.get(best.policy_name, 0.0)
            self.infeasible = False
            self.qp_infeasible = False
            self.certificate_lost = False
            self.fallback_applied = False
            control = best.u.copy()

        self.curr_step += 1
        return control


__all__ = [
    "CandidatePCBFResult",
    "LibraryPCBFMinInterventionQuad3D",
]
