"""Shared, side-effect-free helpers for the drift multi-policy baselines."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from examples.additional_baseline_control_utils import (
    SOLVER_INPUT_TOL,
    project_solver_control,
)


POLICY_ORDER: Tuple[str, ...] = (
    "lane_change_left",
    "lane_change_right",
    "stop",
    "nominal",
)
NOMINAL_POLICY_REPRESENTATION = "frozen_mpcc_sequence"
INPUT_TOL = SOLVER_INPUT_TOL
TIE_TOL = 1e-5


@dataclass
class CandidateCBFResult:
    """Auditable result from one policy's candidate QP."""

    policy_name: str
    feasible: bool
    u: Optional[np.ndarray]
    objective: float
    solver_status: str
    rollout_safe: Optional[bool]
    terminal_safe: Optional[bool]
    solve_time_sec: float
    qp_solved: bool = False
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
    cbf_slack: Optional[float] = None


@dataclass
class MultiPolicyMetrics:
    """Per-controller counters consumed by the benchmark logger."""

    selected_policy_histogram: Counter = field(default_factory=Counter)
    intervention_l2: list[float] = field(default_factory=list)
    policy_switch_count: int = 0
    num_candidate_qps_solved: int = 0
    num_steps_with_no_safe_policy: int = 0
    feasible_candidates_per_step: list[int] = field(default_factory=list)
    certified_candidates_per_step: list[int] = field(default_factory=list)
    rollout_safe_candidates_per_step: list[int] = field(default_factory=list)
    qp_feasible_candidates_per_step: list[int] = field(default_factory=list)
    terminal_failure_count: int = 0
    certificate_loss_count: int = 0
    qp_infeasible_count: int = 0
    fallback_step_count: int = 0
    candidate_qp_failure_count: int = 0
    candidate_evaluation_error_count: int = 0
    num_post_projection_audits: int = 0
    projection_event_count: int = 0
    post_projection_rejection_count: int = 0
    max_projection_delta_inf: float = 0.0
    max_post_projection_constraint_violation: float = 0.0
    max_post_projection_violation_ratio: float = 0.0
    num_post_projection_audits_per_step: list[int] = field(default_factory=list)
    projection_event_count_per_step: list[int] = field(default_factory=list)
    post_projection_rejection_count_per_step: list[int] = field(default_factory=list)
    max_projection_delta_inf_per_step: list[float] = field(default_factory=list)
    max_post_projection_constraint_violation_per_step: list[float] = field(
        default_factory=list
    )
    max_post_projection_violation_ratio_per_step: list[float] = field(
        default_factory=list
    )
    _previous_policy: Optional[str] = None

    def record_selection(self, policy_name: str, intervention_l2: float) -> None:
        if self._previous_policy is not None and self._previous_policy != policy_name:
            self.policy_switch_count += 1
        self._previous_policy = policy_name
        self.selected_policy_histogram[policy_name] += 1
        self.intervention_l2.append(float(intervention_l2))

    def record_projection_audits(
        self, results: Sequence[CandidateCBFResult]
    ) -> None:
        """Accumulate one control step's post-projection candidate audits."""

        audited = [
            result
            for result in results
            if result.post_projection_constraints_satisfied is not None
        ]
        projection_events = [result for result in audited if result.projection_occurred]
        rejected = [
            result
            for result in audited
            if result.post_projection_constraints_satisfied is False
        ]

        delta_max = max(
            (float(result.projection_delta_inf) for result in audited),
            default=0.0,
        )
        violation_max = max(
            (
                float(result.max_post_projection_constraint_violation or 0.0)
                for result in audited
            ),
            default=0.0,
        )
        ratio_max = max(
            (
                float(result.max_post_projection_violation_ratio or 0.0)
                for result in audited
            ),
            default=0.0,
        )

        self.num_post_projection_audits += len(audited)
        self.projection_event_count += len(projection_events)
        self.post_projection_rejection_count += len(rejected)
        self.max_projection_delta_inf = max(self.max_projection_delta_inf, delta_max)
        self.max_post_projection_constraint_violation = max(
            self.max_post_projection_constraint_violation, violation_max
        )
        self.max_post_projection_violation_ratio = max(
            self.max_post_projection_violation_ratio, ratio_max
        )
        self.num_post_projection_audits_per_step.append(len(audited))
        self.projection_event_count_per_step.append(len(projection_events))
        self.post_projection_rejection_count_per_step.append(len(rejected))
        self.max_projection_delta_inf_per_step.append(delta_max)
        self.max_post_projection_constraint_violation_per_step.append(violation_max)
        self.max_post_projection_violation_ratio_per_step.append(ratio_max)

    def as_dict(self) -> Dict[str, Any]:
        values = np.asarray(self.intervention_l2, dtype=float)
        return {
            "mean_intervention_l2": float(np.mean(values)) if values.size else 0.0,
            "max_intervention_l2": float(np.max(values)) if values.size else 0.0,
            "policy_switch_count": int(self.policy_switch_count),
            "selected_policy_histogram": dict(self.selected_policy_histogram),
            "num_candidate_qps_solved": int(self.num_candidate_qps_solved),
            "num_steps_with_no_safe_policy": int(self.num_steps_with_no_safe_policy),
            "num_feasible_backup_candidates_per_step": list(self.feasible_candidates_per_step),
            "num_certified_backup_candidates_per_step": list(
                self.certified_candidates_per_step
            ),
            "num_rollout_safe_candidates_per_step": list(self.rollout_safe_candidates_per_step),
            "num_qp_feasible_candidates_per_step": list(self.qp_feasible_candidates_per_step),
            "terminal_failure_count": int(self.terminal_failure_count),
            "certificate_loss_count": int(self.certificate_loss_count),
            "qp_infeasible_count": int(self.qp_infeasible_count),
            "fallback_step_count": int(self.fallback_step_count),
            "candidate_qp_failure_count": int(self.candidate_qp_failure_count),
            "candidate_evaluation_error_count": int(
                self.candidate_evaluation_error_count
            ),
            "num_post_projection_audits": int(self.num_post_projection_audits),
            "projection_occurred": bool(self.projection_event_count > 0),
            "projection_event_count": int(self.projection_event_count),
            "post_projection_rejection_count": int(
                self.post_projection_rejection_count
            ),
            "max_projection_delta_inf": float(self.max_projection_delta_inf),
            "max_post_projection_constraint_violation": float(
                self.max_post_projection_constraint_violation
            ),
            "max_post_projection_violation_ratio": float(
                self.max_post_projection_violation_ratio
            ),
            "num_post_projection_audits_per_step": list(
                self.num_post_projection_audits_per_step
            ),
            "projection_event_count_per_step": list(
                self.projection_event_count_per_step
            ),
            "post_projection_rejection_count_per_step": list(
                self.post_projection_rejection_count_per_step
            ),
            "max_projection_delta_inf_per_step": list(
                self.max_projection_delta_inf_per_step
            ),
            "max_post_projection_constraint_violation_per_step": list(
                self.max_post_projection_constraint_violation_per_step
            ),
            "max_post_projection_violation_ratio_per_step": list(
                self.max_post_projection_violation_ratio_per_step
            ),
        }


def _params_signature(params: Any) -> Any:
    if hasattr(params, "_asdict"):
        return tuple((key, float(value)) for key, value in params._asdict().items())
    if is_dataclass(params):
        return tuple(sorted(asdict(params).items()))
    if isinstance(params, Mapping):
        return tuple(sorted((str(key), _params_signature(value)) for key, value in params.items()))
    if isinstance(params, np.ndarray):
        return (params.dtype.str, params.shape, params.tobytes())
    return params


def runtime_library_signature(controller: Any) -> Dict[str, Any]:
    """Return the fields whose equality defines a fair runtime library."""

    configs = controller.policy_configs
    names = tuple(getattr(controller, "policy_names", tuple(configs.keys()) + ("nominal",)))
    return {
        "names": names,
        "configs": tuple(
            (
                name,
                configs[name]["type"],
                int(configs[name]["horizon"]),
                _params_signature(configs[name]["params"]),
            )
            for name in names
            if name != "nominal"
        ),
        "left_lane_y": float(controller.left_lane_y),
        "right_lane_y": float(controller.right_lane_y),
        "u_min": tuple(np.asarray(controller.u_min, dtype=float)),
        "u_max": tuple(np.asarray(controller.u_max, dtype=float)),
        "dt": float(controller.dt),
        "backup_horizon": float(controller.backup_horizon),
        "nominal_policy_representation": getattr(
            controller,
            "nominal_policy_representation",
            NOMINAL_POLICY_REPRESENTATION,
        ),
    }


def assert_runtime_library_equal(candidate: Any, reference_plcbf: Any) -> None:
    """Fail loudly if a baseline is not using PL-CBF's exact runtime library."""

    candidate_signature = runtime_library_signature(candidate)
    reference_signature = runtime_library_signature(reference_plcbf)
    if candidate_signature != reference_signature:
        raise AssertionError(
            "Additional baseline policy library differs from runtime PL-CBF library:\n"
            f"candidate={candidate_signature!r}\nreference={reference_signature!r}"
        )


def select_minimum_intervention(
    results: Iterable[CandidateCBFResult],
    policy_names: Sequence[str],
    tie_tol: float = TIE_TOL,
) -> Optional[CandidateCBFResult]:
    """Select by objective, using only fixed policy order for numerical ties."""

    order = {name: index for index, name in enumerate(policy_names)}
    feasible = [result for result in results if result.feasible]
    if not feasible:
        return None
    feasible.sort(key=lambda result: order[result.policy_name])
    best = feasible[0]
    for result in feasible[1:]:
        if result.objective < best.objective - tie_tol:
            best = result
    return best


def valid_bounded_control(
    u: Optional[np.ndarray],
    u_min: np.ndarray,
    u_max: np.ndarray,
    tol: float = INPUT_TOL,
) -> bool:
    return project_bounded_control(u, u_min, u_max, tol=tol) is not None


def project_bounded_control(
    u: Optional[np.ndarray],
    u_min: np.ndarray,
    u_max: np.ndarray,
    tol: float = INPUT_TOL,
) -> Optional[np.ndarray]:
    """Project only a solver-tolerance-feasible input to exact bounds."""

    return project_solver_control(
        u,
        u_min,
        u_max,
        expected_shape=np.asarray(u_min).shape,
        tolerance=tol,
    )
