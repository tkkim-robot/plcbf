"""Common types and pointwise PCBF-family selection.

Only Policy-PCBF, PL-CBF, and Library-PCBF-MI can be decided from a collection
of pointwise policy certificates.  Backup-CBF, Multi-Backup-CBF-MI, MPS,
Gatekeeper, and MI-MPC require path-wise sensitivities, committed
trajectories, or continuous mixed-integer MPC variables.  They are therefore
implemented by the case-study baseline suites and cannot be silently reduced
to certificate selectors through this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import time
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from plcbf.policy_library import (
    PolicyCertificate,
    PolicyDecision,
    SelectionMode,
    select_policy,
)


FloatArray = NDArray[np.float64]


class BenchmarkMethod(str, Enum):
    """Algorithms included in both new case-study benchmarks."""

    POLICY_PCBF = "pcbf"
    PLCBF = "plcbf"
    MPS = "mps"
    GATEKEEPER = "gatekeeper"
    BACKUP_CBF = "backup_cbf"
    MI_MPC = "mi_mpc"
    MULTI_BACKUP_CBF_MI = "multi_backup_cbf_mi"
    LIBRARY_PCBF_MI = "library_pcbf_mi"


BENCHMARK_METHODS: tuple[str, ...] = tuple(method.value for method in BenchmarkMethod)


def _vector(value: ArrayLike, name: str, size: int | None = None) -> FloatArray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if size is not None and array.size != size:
        raise ValueError(f"{name} must contain {size} values")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return np.array(array, copy=True)


def _bounds(
    nominal_control: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    nominal = _vector(nominal_control, "nominal_control")
    low = np.broadcast_to(np.asarray(lower, dtype=float), nominal.shape).copy()
    high = np.broadcast_to(np.asarray(upper, dtype=float), nominal.shape).copy()
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise ValueError("control bounds must be finite")
    if np.any(low > high):
        raise ValueError("lower control bounds must not exceed upper bounds")
    return nominal, low, high


def _find_policy(
    certificates: Sequence[PolicyCertificate],
    preferred_policy_id: str,
) -> PolicyCertificate | None:
    """Return only the predeclared fixed policy.

    Fixed-backup baselines in the warehouse study never substitute a different
    policy when their configured backup is absent.  Returning a "best"
    nonnominal certificate here silently changed the algorithm and could give
    a single-backup baseline access to room or directional policies.
    """

    for certificate in certificates:
        if certificate.policy_id == preferred_policy_id:
            return certificate
    return None


@dataclass(frozen=True)
class BaselineDecision:
    """Uniform executable result returned by every comparison adapter."""

    method: str
    control: FloatArray
    policy_id: str | None
    feasible: bool
    status: str
    used_fallback: bool
    objective: float
    solve_time_s: float
    policy_decision: PolicyDecision | None = None

    def __post_init__(self) -> None:
        control = np.asarray(self.control, dtype=float).reshape(-1).copy()
        if control.size == 0 or not np.all(np.isfinite(control)):
            raise ValueError("decision control must be a non-empty finite vector")
        control.setflags(write=False)
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "method", str(self.method))
        object.__setattr__(self, "objective", float(self.objective))
        object.__setattr__(self, "solve_time_s", float(self.solve_time_s))


class TrajectoryBaselineRequiredError(RuntimeError):
    """Raised when a trajectory algorithm is requested from certificates."""


def _trajectory_algorithm_error(method: BenchmarkMethod | str) -> None:
    parsed = (
        method if isinstance(method, BenchmarkMethod) else BenchmarkMethod(method)
    )
    raise TrajectoryBaselineRequiredError(
        f"{parsed.value} cannot be solved from PolicyCertificate objects; "
        "use the case-study baseline suite so the full warehouse-style "
        "trajectory algorithm is constructed"
    )


def _from_policy_decision(
    method: str,
    decision: PolicyDecision,
    nominal: FloatArray,
    started_at: float,
    *,
    policy_id_override: str | None = None,
) -> BaselineDecision:
    diagnostics = decision.diagnostics
    objective = float(np.sum((decision.control - nominal) ** 2))
    return BaselineDecision(
        method=method,
        control=decision.control,
        policy_id=(
            decision.policy_id
            if policy_id_override is None
            else policy_id_override
        ),
        feasible=not diagnostics.used_fallback,
        status=(
            "fallback:" + str(diagnostics.fallback_reason)
            if diagnostics.used_fallback
            else "optimal"
        ),
        used_fallback=diagnostics.used_fallback,
        objective=objective,
        solve_time_s=time.perf_counter() - started_at,
        policy_decision=decision,
    )


def _replace_fallback_control(
    decision: PolicyDecision,
    control: FloatArray,
    *,
    source: str,
    clear_selected_policy: bool = False,
) -> PolicyDecision:
    """Keep selection diagnostics while applying an algorithm-specific fallback."""

    if not decision.diagnostics.used_fallback:
        return decision
    diagnostics = replace(
        decision.diagnostics,
        fallback_source=str(source),
        selected_policy_id=(
            None
            if clear_selected_policy
            else decision.diagnostics.selected_policy_id
        ),
    )
    return replace(
        decision,
        certificate=None if clear_selected_policy else decision.certificate,
        control=np.asarray(control, dtype=float).reshape(-1),
        diagnostics=diagnostics,
    )


def solve_policy_pcbf(
    certificates: Sequence[PolicyCertificate],
    nominal_control: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    backup_policy_id: str = "stop",
    safe_value_threshold: float = 0.0,
) -> BaselineDecision:
    """Policy-PCBF using one fixed backup policy.

    A fixed PCBF still solves its CBF-QP when the current rollout value is
    nonpositive.  The value changes the CBF right-hand side; it is not a
    license to swap policies or bypass the QP.
    """

    started_at = time.perf_counter()
    nominal, low, high = _bounds(nominal_control, lower, upper)
    certificate = _find_policy(certificates, backup_policy_id)
    selected = () if certificate is None else (certificate,)
    # A fixed-policy PCBF has no branch-eligibility threshold: its sole
    # predeclared certificate is always sent to the QP, even for V < 0.  The
    # shared selector requires a finite value floor, so use that certificate's
    # own value solely to disable selection gating.  This is not the MI-MPC
    # safety threshold and cannot make another branch eligible.
    del safe_value_threshold
    fixed_policy_value_floor = (
        0.0
        if certificate is None or not np.isfinite(certificate.value)
        else float(certificate.value)
    )
    decision = select_policy(
        selected,
        nominal,
        low,
        high,
        mode=SelectionMode.INTERVENTION,
        safe_value_threshold=fixed_policy_value_floor,
    )
    # PCBFBase._solve_qp returns nominal when its sole QP fails.  The fixed
    # backup trajectory defines the certificate, not an emergency action.
    decision = _replace_fallback_control(
        decision,
        np.clip(nominal, low, high),
        source="nominal_after_fixed_pcbf_qp_failure",
    )
    return _from_policy_decision(
        BenchmarkMethod.POLICY_PCBF.value, decision, nominal, started_at
    )


def solve_plcbf(
    certificates: Sequence[PolicyCertificate],
    nominal_control: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    safe_value_threshold: float = 0.0,
) -> BaselineDecision:
    """PL-CBF selection by feasible input-set volume."""

    started_at = time.perf_counter()
    nominal, low, high = _bounds(nominal_control, lower, upper)
    decision = select_policy(
        certificates,
        nominal,
        low,
        high,
        mode=SelectionMode.INPUT_VOLUME,
        safe_value_threshold=safe_value_threshold,
    )
    if decision.diagnostics.fallback_reason == "no_safe_policy":
        # Warehouse PL-CBF's max operator still chooses a policy when every
        # rollout value is nonpositive: first the least-negative V, then input
        # volume as a tie-break.  It solves that selected policy's QP and uses
        # the direct backup only if the QP itself is infeasible.  This
        # all-unsafe rule is specific to PL-CBF; MI-MPC eligibility remains a
        # hard, fixed threshold in plcbf.big_m_mpc.
        valid = [
            certificate
            for certificate in certificates
            if certificate.valid and np.isfinite(certificate.value)
        ]
        if valid:
            best_value = max(certificate.value for certificate in valid)
            least_unsafe = [
                certificate
                for certificate in valid
                if np.isclose(
                    certificate.value,
                    best_value,
                    rtol=0.0,
                    atol=1e-9,
                )
            ]
            decision = select_policy(
                least_unsafe,
                nominal,
                low,
                high,
                mode=SelectionMode.INPUT_VOLUME,
                safe_value_threshold=float(best_value),
            )
    return _from_policy_decision(
        BenchmarkMethod.PLCBF.value, decision, nominal, started_at
    )


def solve_library_pcbf_mi(
    certificates: Sequence[PolicyCertificate],
    nominal_control: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    safe_value_threshold: float = 0.0,
    emergency_policy_id: str = "stop",
    emergency_control: ArrayLike | None = None,
) -> BaselineDecision:
    """Solve one PCBF-QP per policy and choose minimum intervention."""

    started_at = time.perf_counter()
    nominal, low, high = _bounds(nominal_control, lower, upper)
    # The warehouse implementation certifies a branch only for finite
    # V > threshold (strictly; V == 0 is not sent to a candidate QP).
    threshold = float(safe_value_threshold)
    certified = tuple(
        certificate
        for certificate in certificates
        if (
            certificate.valid
            and np.isfinite(certificate.value)
            and certificate.value > threshold
        )
    )
    selector_threshold = (
        min(certificate.value for certificate in certified)
        if certified
        else threshold
    )
    decision = select_policy(
        certified,
        nominal,
        low,
        high,
        mode=SelectionMode.INTERVENTION,
        safe_value_threshold=selector_threshold,
    )
    executed_policy_id = None
    if decision.diagnostics.used_fallback:
        emergency_id = str(emergency_policy_id)
        if not emergency_id:
            raise ValueError("emergency_policy_id must not be empty")
        if emergency_control is None:
            emergency = _find_policy(certificates, emergency_id)
            if emergency is None or emergency.backup_control is None:
                raise ValueError(
                    "Library-PCBF-MI requires an explicit emergency control "
                    f"or backup_control for policy {emergency_id!r}"
                )
            emergency_array = _vector(
                emergency.backup_control,
                "emergency backup_control",
                nominal.size,
            )
        else:
            emergency_array = _vector(
                emergency_control,
                "emergency_control",
                nominal.size,
            )
        decision = _replace_fallback_control(
            decision,
            np.clip(emergency_array, low, high),
            source=f"emergency_policy:{emergency_id}",
            clear_selected_policy=True,
        )
        executed_policy_id = emergency_id
    return _from_policy_decision(
        BenchmarkMethod.LIBRARY_PCBF_MI.value,
        decision,
        nominal,
        started_at,
        policy_id_override=executed_policy_id,
    )


def solve_backup_cbf(
    certificates: Sequence[PolicyCertificate],
    nominal_control: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    backup_policy_id: str = "stop",
    safe_value_threshold: float = 0.0,
) -> BaselineDecision:
    """Reject the former one-certificate approximation of Backup-CBF."""

    del (
        certificates,
        nominal_control,
        lower,
        upper,
        backup_policy_id,
        safe_value_threshold,
    )
    _trajectory_algorithm_error(BenchmarkMethod.BACKUP_CBF)


def solve_multi_backup_cbf_mi(
    certificates: Sequence[PolicyCertificate],
    nominal_control: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    safe_value_threshold: float = 0.0,
) -> BaselineDecision:
    """Reject the former PCBF-selector approximation of MB-CBF-MI."""

    del certificates, nominal_control, lower, upper, safe_value_threshold
    _trajectory_algorithm_error(BenchmarkMethod.MULTI_BACKUP_CBF_MI)


def solve_mps(
    certificates: Sequence[PolicyCertificate],
    nominal_control: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    safe_value_threshold: float = 0.0,
) -> BaselineDecision:
    """Reject stateless certificate selection in place of committed MPS."""

    del certificates, nominal_control, lower, upper, safe_value_threshold
    _trajectory_algorithm_error(BenchmarkMethod.MPS)


def solve_gatekeeper(
    certificates: Sequence[PolicyCertificate],
    nominal_control: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    state: object | None = None,
    *,
    safe_value_threshold: float = 0.0,
) -> BaselineDecision:
    """Reject policy-ID latching in place of trajectory Gatekeeper."""

    del certificates, nominal_control, lower, upper, state, safe_value_threshold
    _trajectory_algorithm_error(BenchmarkMethod.GATEKEEPER)


def solve_mi_mpc(
    certificates: Sequence[PolicyCertificate],
    nominal_control: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    safe_value_threshold: float = 0.0,
    intervention_weight: float = 1.0,
    terminal_weight: float = 1.0,
) -> BaselineDecision:
    """Reject a branch-only MILP in place of trajectory MI-MPC."""

    del (
        certificates,
        nominal_control,
        lower,
        upper,
        safe_value_threshold,
        intervention_weight,
        terminal_weight,
    )
    _trajectory_algorithm_error(BenchmarkMethod.MI_MPC)


def solve_baseline(
    method: BenchmarkMethod | str,
    certificates: Sequence[PolicyCertificate],
    nominal_control: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    backup_policy_id: str = "stop",
    safe_value_threshold: float = 0.0,
) -> BaselineDecision:
    """Dispatch only algorithms that are valid pointwise certificate methods."""

    parsed = method if isinstance(method, BenchmarkMethod) else BenchmarkMethod(method)
    common = dict(
        certificates=certificates,
        nominal_control=nominal_control,
        lower=lower,
        upper=upper,
        safe_value_threshold=safe_value_threshold,
    )
    if parsed is BenchmarkMethod.POLICY_PCBF:
        return solve_policy_pcbf(
            **common, backup_policy_id=backup_policy_id
        )
    if parsed is BenchmarkMethod.PLCBF:
        return solve_plcbf(**common)
    if parsed is BenchmarkMethod.LIBRARY_PCBF_MI:
        return solve_library_pcbf_mi(**common)
    _trajectory_algorithm_error(parsed)


__all__ = [
    "BENCHMARK_METHODS",
    "BaselineDecision",
    "BenchmarkMethod",
    "TrajectoryBaselineRequiredError",
    "solve_backup_cbf",
    "solve_baseline",
    "solve_gatekeeper",
    "solve_library_pcbf_mi",
    "solve_mi_mpc",
    "solve_mps",
    "solve_multi_backup_cbf_mi",
    "solve_plcbf",
    "solve_policy_pcbf",
]
