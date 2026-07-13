"""Shared, side-effect-free helpers for the drift multi-policy baselines."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


POLICY_ORDER: Tuple[str, ...] = (
    "lane_change_left",
    "lane_change_right",
    "stop",
    "nominal",
)
NOMINAL_POLICY_REPRESENTATION = "frozen_mpcc_sequence"
INPUT_TOL = 1e-5
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
    _previous_policy: Optional[str] = None

    def record_selection(self, policy_name: str, intervention_l2: float) -> None:
        if self._previous_policy is not None and self._previous_policy != policy_name:
            self.policy_switch_count += 1
        self._previous_policy = policy_name
        self.selected_policy_histogram[policy_name] += 1
        self.intervention_l2.append(float(intervention_l2))

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
    if u is None:
        return False
    value = np.asarray(u, dtype=float).reshape(-1)
    return bool(
        value.shape == np.asarray(u_min).shape
        and np.all(np.isfinite(value))
        and np.all(value >= np.asarray(u_min) - tol)
        and np.all(value <= np.asarray(u_max) + tol)
    )
