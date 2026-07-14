"""
Library PCBF with minimum-intervention selection for Quad3D.

This class is a controlled selector ablation of :class:`PLCBF_Quad3D`.
It intentionally reuses the parent's policy construction, JAX rollout/value/
gradient functions, dynamic-obstacle time derivative, input bounds, static
HOCBF constraints, CBF gain, and OSQP configuration.  The only algorithmic
difference is the final selection stage: every policy with ``V > 0`` gets the
same PCBF-QP that PL-CBF would solve for that policy, and the feasible result
with the smallest realized ``||u - u_nom||_2^2`` is applied.

This is not presented as an unchanged implementation of a prior algorithm.
It combines the repository's PCBF certificate with the multi-backup
minimum-intervention selection principle for a direct selector comparison.
"""

from dataclasses import dataclass
import time
from typing import Dict, Optional, Sequence, Tuple

import cvxpy as cp
import jax
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


# The existing PL-CBF selector already uses 1e-6 as its numerical comparison
# tolerance.  Reuse it here for deterministic objective ties.
_TIE_TOL = 1e-6
_INPUT_TOL = SOLVER_INPUT_TOL  # OSQP's default absolute feasibility tolerance.
_QP_AUDIT_ATOL = _INPUT_TOL
_QP_AUDIT_RTOL = _INPUT_TOL
_ACCEPTED_STATUSES = ("optimal", "optimal_inaccurate")


@dataclass(frozen=True)
class CandidatePCBFResult:
    """Result and accounting data for one library policy at one real step."""

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
    """PL-CBF certificate library with minimum-intervention QP selection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The inherited setup path is the policy-library source of truth.  Keep
        # its insertion order as the fixed deterministic policy index.
        self.policy_names: Tuple[str, ...] = tuple(self.policy_configs.keys())
        if tuple(self.angle_names) + ("stop", "nominal") != self.policy_names:
            raise AssertionError(
                "Lib-PCBF-MI must inherit the exact ordered PL-CBF policy library"
            )

        # Per-step status consumed by benchmark integration and regression tests.
        self.last_candidate_results: Tuple[CandidatePCBFResult, ...] = tuple()
        self.last_values: Dict[str, float] = {}
        self.last_gradients: Dict[str, np.ndarray] = {}
        self.last_trajectories: Dict[str, np.ndarray] = {}
        self.last_time_derivatives: Dict[str, float] = {}
        self.last_selected_policy_name: Optional[str] = None
        self.last_policy_switched = False
        self.last_selected_policy: Optional[str] = None
        self.last_selected_policy_idx = -1
        self.last_solver_status = "not_run"
        self.last_infeasible = False
        self.infeasible = False
        self.qp_infeasible = False
        self.certificate_lost = False
        self.last_certificate_lost = False
        self.last_emergency_action_used = False
        self.last_num_rollout_safe_candidates = 0
        self.last_num_qp_feasible_candidates = 0
        self.last_num_candidate_qps_solved = 0
        self.last_intervention_objective = float("nan")
        self.last_intervention_l2 = float("nan")
        self.last_policy_eval_time_sec = 0.0
        self.last_qp_solve_time_sec = 0.0
        self.last_selection_time_sec = 0.0
        self.last_total_time_sec = 0.0

        # Cumulative accounting supports the richer benchmark metrics without
        # changing any existing runner or controller implementation.
        self.total_candidate_qps_solved = 0
        self.num_steps_with_no_safe_policy = 0
        self.num_steps_with_no_feasible_qp = 0
        self.policy_switch_count = 0
        self.selected_policy_histogram: Dict[str, int] = {
            name: 0 for name in self.policy_names
        }
        self._previous_selected_policy: Optional[str] = None

    def assert_policy_library_equal(self, reference: PLCBF_Quad3D) -> None:
        """Fail loudly unless ``reference`` has the exact inherited library.

        This helper is intended for fairness/regression tests.  It compares the
        runtime dictionaries rather than reconstructing a second "similar"
        policy library.
        """

        if not isinstance(reference, PLCBF_Quad3D):
            raise TypeError("reference must be a PLCBF_Quad3D instance")

        if tuple(reference.policy_configs.keys()) != self.policy_names:
            raise AssertionError("policy names or ordering differ from PL-CBF")
        if tuple(reference.angle_names) != tuple(self.angle_names):
            raise AssertionError("angle-policy names or ordering differ from PL-CBF")

        for name in self.policy_names:
            self_type, self_params = self.policy_configs[name]
            ref_type, ref_params = reference.policy_configs[name]
            if self_type != ref_type:
                raise AssertionError(f"policy type differs for {name!r}")

            self_leaves, self_tree = jax.tree_util.tree_flatten(self_params)
            ref_leaves, ref_tree = jax.tree_util.tree_flatten(ref_params)
            if self_tree != ref_tree or len(self_leaves) != len(ref_leaves):
                raise AssertionError(f"policy parameter structure differs for {name!r}")
            for leaf_index, (self_leaf, ref_leaf) in enumerate(
                zip(self_leaves, ref_leaves)
            ):
                if not np.array_equal(np.asarray(self_leaf), np.asarray(ref_leaf)):
                    raise AssertionError(
                        f"policy parameter {leaf_index} differs for {name!r}"
                    )

        scalar_fields = (
            "num_angle_policies",
            "dt",
            "backup_horizon",
            "eval_horizon_steps",
            "cbf_alpha",
            "safety_margin",
        )
        for field in scalar_fields:
            if getattr(self, field) != getattr(reference, field):
                raise AssertionError(f"PL-CBF configuration differs for {field!r}")

        if self.dynamics_params.u_min != reference.dynamics_params.u_min:
            raise AssertionError("lower control limit differs from PL-CBF")
        if self.dynamics_params.u_max != reference.dynamics_params.u_max:
            raise AssertionError("upper control limit differs from PL-CBF")

    def _evaluate_policy_library(self, state, control_ref=None):
        """Evaluate the policy library through the exact inherited JAX paths."""

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

        # Match PLCBF_Quad3D exactly: all angle policies use its batched path.
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

            # Converting every JAX result to NumPy here also synchronizes all
            # candidate computations before the policy-evaluation timer stops.
            for index, name in enumerate(self.angle_names):
                params = self.policy_configs[name][1]
                policy_params_used[name] = ("angle", params)
                results[name] = (
                    float(values_batch[index]),
                    np.array(gradients_batch[index]),
                    np.array(trajectories_batch[index]),
                )
                time_derivatives[name] = float(time_derivatives_batch[index])

        # Match the parent's single-policy path for stop and the frozen nominal
        # waypoint context.  The runtime policy dictionary is never mutated.
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
        """Solve exactly the inherited PL-CBF QP for one certified policy."""

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

        # _add_cbf_constraints reads this inherited per-policy time term and
        # appends both the PCBF constraint and all common static HOCBFs.
        self._last_time_derivative = time_derivative
        self._add_cbf_constraints(u, constraints, state, value, gradient)
        problem = cp.Problem(cp.Minimize(cost), constraints)

        start = time.perf_counter()
        error = None
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            solver_status = str(problem.status)
        except Exception as exc:  # Preserve failure details; never use nominal silently.
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
        if projection.control is None:  # Kept explicit for static type checkers.
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
        """Select by objective, then fixed policy index for numerical ties."""

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
        """Use the shared library's stop policy after explicit certificate loss."""

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

    def get_last_step_metrics(self) -> dict:
        """Return JSON-safe status and accounting for the most recent step."""

        def finite_or_none(value):
            value = float(value)
            return value if np.isfinite(value) else None

        candidates = []
        for candidate in self.last_candidate_results:
            candidates.append({
                "policy_name": candidate.policy_name,
                "policy_index": candidate.policy_index,
                "certified": candidate.certified,
                "value": finite_or_none(candidate.value),
                "feasible": candidate.feasible,
                "u": None if candidate.u is None else candidate.u.tolist(),
                "objective": finite_or_none(candidate.objective),
                "intervention_l2": finite_or_none(candidate.intervention_l2),
                "solver_status": candidate.solver_status,
                "solve_time_sec": candidate.solve_time_sec,
                "error": candidate.error,
                "projection_occurred": candidate.projection_occurred,
                "projection_delta_inf": candidate.projection_delta_inf,
                "post_projection_constraints_satisfied": (
                    candidate.post_projection_constraints_satisfied
                ),
                "max_post_projection_constraint_violation": finite_or_none(
                    candidate.max_post_projection_constraint_violation
                )
                if candidate.max_post_projection_constraint_violation is not None
                else None,
                "max_post_projection_violation_ratio": finite_or_none(
                    candidate.max_post_projection_violation_ratio
                )
                if candidate.max_post_projection_violation_ratio is not None
                else None,
                "constraint_audit_atol": candidate.constraint_audit_atol,
                "constraint_audit_rtol": candidate.constraint_audit_rtol,
                "constraint_audit_count": candidate.constraint_audit_count,
                "solver_name": candidate.solver_name,
            })

        audited = [
            candidate
            for candidate in self.last_candidate_results
            if candidate.post_projection_constraints_satisfied is not None
        ]
        projection_events = [
            candidate for candidate in audited if candidate.projection_occurred
        ]
        rejected = [
            candidate
            for candidate in audited
            if candidate.post_projection_constraints_satisfied is False
        ]

        return {
            "selected_policy": self.last_selected_policy_name,
            "policy_switched": self.last_policy_switched,
            "selected_policy_idx": self.last_selected_policy_idx,
            "solver_status": self.last_solver_status,
            "infeasible": self.last_infeasible,
            "qp_infeasible": self.qp_infeasible,
            "certificate_lost": self.last_certificate_lost,
            "emergency_action_used": self.last_emergency_action_used,
            "num_rollout_safe_candidates": self.last_num_rollout_safe_candidates,
            "num_qp_feasible_candidates": self.last_num_qp_feasible_candidates,
            "num_candidate_qps_solved": self.last_num_candidate_qps_solved,
            "num_steps_with_no_safe_policy": int(
                self.last_num_rollout_safe_candidates == 0
            ),
            "num_steps_with_no_certified_rollout": int(
                self.last_num_rollout_safe_candidates == 0
            ),
            "num_steps_with_no_feasible_qp": int(
                self.last_infeasible and self.last_num_rollout_safe_candidates > 0
            ),
            "intervention_objective": finite_or_none(
                self.last_intervention_objective
            ),
            "intervention_l2": finite_or_none(self.last_intervention_l2),
            "policy_eval_time_sec": self.last_policy_eval_time_sec,
            "qp_solve_time_sec": self.last_qp_solve_time_sec,
            "selection_time_sec": self.last_selection_time_sec,
            "total_time_sec": self.last_total_time_sec,
            "num_post_projection_audits": len(audited),
            "projection_occurred": bool(projection_events),
            "projection_event_count": len(projection_events),
            "post_projection_rejection_count": len(rejected),
            "max_projection_delta_inf": max(
                (candidate.projection_delta_inf for candidate in audited),
                default=0.0,
            ),
            "max_post_projection_constraint_violation": max(
                (
                    candidate.max_post_projection_constraint_violation or 0.0
                    for candidate in audited
                ),
                default=0.0,
            ),
            "max_post_projection_violation_ratio": max(
                (
                    candidate.max_post_projection_violation_ratio or 0.0
                    for candidate in audited
                ),
                default=0.0,
            ),
            "candidate_results": candidates,
        }

    def solve_control_problem(self, state, control_ref=None):
        """Evaluate all certificates, solve all safe QPs, and select by intervention."""

        total_start = time.perf_counter()
        state = np.asarray(state, dtype=float).reshape(-1).copy()
        if control_ref and "u_ref" in control_ref:
            u_nom = np.asarray(control_ref["u_ref"], dtype=float).reshape(-1).copy()
        else:
            u_nom = np.zeros(4, dtype=float)
        if u_nom.shape != (4,) or not np.all(np.isfinite(u_nom)):
            raise ValueError("control_ref['u_ref'] must be a finite four-vector")

        eval_start = time.perf_counter()
        results, _policy_params_used, time_derivatives = self._evaluate_policy_library(
            state, control_ref
        )
        self.last_policy_eval_time_sec = time.perf_counter() - eval_start

        self._last_results = results
        self.last_values = {name: float(data[0]) for name, data in results.items()}
        self.last_gradients = {
            name: np.asarray(data[1]).copy() for name, data in results.items()
        }
        self.last_trajectories = {
            name: np.asarray(data[2]).copy() for name, data in results.items()
        }
        self.last_time_derivatives = dict(time_derivatives)

        candidate_results = []
        qps_solved = 0
        qp_time_sec = 0.0
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
            if candidate.solver_status != "invalid_certificate":
                qps_solved += 1
                qp_time_sec += candidate.solve_time_sec

        selection_start = time.perf_counter()
        best = self._select_minimum_intervention(candidate_results)
        self.last_selection_time_sec = time.perf_counter() - selection_start

        self.last_candidate_results = tuple(candidate_results)
        self.last_num_rollout_safe_candidates = rollout_safe_count
        self.last_num_qp_feasible_candidates = sum(
            int(candidate.feasible) for candidate in candidate_results
        )
        self.last_num_candidate_qps_solved = qps_solved
        self.total_candidate_qps_solved += qps_solved
        self.last_qp_solve_time_sec = qp_time_sec

        if best is None:
            self.last_selected_policy_name = None
            self.last_policy_switched = False
            self.last_selected_policy = None
            self.last_selected_policy_idx = -1
            self._last_best_name = None
            self.last_infeasible = rollout_safe_count > 0
            self.infeasible = self.last_infeasible
            self.qp_infeasible = self.last_infeasible
            self.certificate_lost = rollout_safe_count == 0
            self.last_certificate_lost = self.certificate_lost
            self.last_emergency_action_used = True
            self.last_intervention_objective = float("nan")
            self.last_intervention_l2 = float("nan")
            if rollout_safe_count == 0:
                self.last_solver_status = "no_certified_policy"
                self.num_steps_with_no_safe_policy += 1
            else:
                self.last_solver_status = "all_candidate_qps_failed"
                self.num_steps_with_no_feasible_qp += 1
            control = self._emergency_control(state)
        else:
            self.last_selected_policy_name = best.policy_name
            self.last_policy_switched = bool(
                self._previous_selected_policy is not None
                and self._previous_selected_policy != best.policy_name
            )
            self.last_selected_policy = best.policy_name
            self.last_selected_policy_idx = best.policy_index
            self._last_best_name = best.policy_name
            self._last_time_derivative = time_derivatives.get(best.policy_name, 0.0)
            self.last_solver_status = best.solver_status
            self.last_infeasible = False
            self.infeasible = False
            self.qp_infeasible = False
            self.certificate_lost = False
            self.last_certificate_lost = False
            self.last_emergency_action_used = False
            self.last_intervention_objective = best.objective
            self.last_intervention_l2 = best.intervention_l2
            self.selected_policy_histogram[best.policy_name] += 1
            if self.last_policy_switched:
                self.policy_switch_count += 1
            self._previous_selected_policy = best.policy_name
            control = best.u.copy()

        self.curr_step += 1
        self.last_total_time_sec = time.perf_counter() - total_start
        return control


__all__ = [
    "CandidatePCBFResult",
    "LibraryPCBFMinInterventionQuad3D",
]
