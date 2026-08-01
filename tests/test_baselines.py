from __future__ import annotations

import numpy as np
import pytest

from plcbf.baselines import (
    BENCHMARK_METHODS,
    BenchmarkMethod,
    TrajectoryBaselineRequiredError,
    solve_baseline,
    solve_library_pcbf_mi,
    solve_mi_mpc,
    solve_plcbf,
    solve_policy_pcbf,
)
from plcbf.policy_library import CBFHalfspace, PolicyCertificate


def _certificate(
    policy_id: str,
    *,
    value: float = 1.0,
    control=(0.25, 0.0),
    offset: float = 0.2,
) -> PolicyCertificate:
    return PolicyCertificate(
        policy_id=policy_id,
        value=value,
        halfspaces=(CBFHalfspace(np.array([1.0, 0.0]), offset),),
        backup_control=np.asarray(control, dtype=float),
    )


def test_pointwise_dispatch_is_limited_to_the_pcbf_family() -> None:
    certificates = (
        _certificate("stop"),
        _certificate("radial"),
        _certificate("nominal"),
    )
    pointwise = {
        BenchmarkMethod.POLICY_PCBF.value,
        BenchmarkMethod.PLCBF.value,
        BenchmarkMethod.LIBRARY_PCBF_MI.value,
    }
    assert set(BENCHMARK_METHODS) - pointwise == {
        "backup_cbf",
        "mps",
        "gatekeeper",
        "mi_mpc",
        "multi_backup_cbf_mi",
    }
    for method in pointwise:
        decision = solve_baseline(
            method,
            certificates,
            nominal_control=np.array([0.5, 0.0]),
            lower=-np.ones(2),
            upper=np.ones(2),
        )
        assert decision.method == method
        assert np.all(np.isfinite(decision.control))


@pytest.mark.parametrize(
    "method",
    [
        BenchmarkMethod.BACKUP_CBF,
        BenchmarkMethod.MPS,
        BenchmarkMethod.GATEKEEPER,
        BenchmarkMethod.MI_MPC,
        BenchmarkMethod.MULTI_BACKUP_CBF_MI,
    ],
)
def test_trajectory_algorithms_cannot_be_reduced_to_certificates(
    method: BenchmarkMethod,
) -> None:
    with pytest.raises(TrajectoryBaselineRequiredError):
        solve_baseline(
            method,
            (_certificate("stop"),),
            nominal_control=np.array([0.5, 0.0]),
            lower=-np.ones(2),
            upper=np.ones(2),
        )


def test_direct_mi_mpc_api_rejects_a_branch_only_certificate_milp() -> None:
    with pytest.raises(
        TrajectoryBaselineRequiredError,
        match="full warehouse-style trajectory algorithm",
    ):
        solve_mi_mpc(
            (
                _certificate("safe-cheap", value=1.0, control=(0.1, 0.0)),
                _certificate("safe-expensive", value=2.0, control=(0.9, 0.0)),
            ),
            nominal_control=np.array([0.0, 0.0]),
            lower=-np.ones(2),
            upper=np.ones(2),
        )


def test_policy_pcbf_never_substitutes_a_missing_fixed_backup() -> None:
    decision = solve_policy_pcbf(
        (_certificate("room_0"), _certificate("radial_0")),
        nominal_control=np.array([0.5, 0.0]),
        lower=-np.ones(2),
        upper=np.ones(2),
        backup_policy_id="stop",
    )
    assert not decision.feasible
    assert decision.policy_id is None
    assert decision.used_fallback


def test_policy_pcbf_solves_its_qp_even_when_value_is_negative() -> None:
    certificate = _certificate("stop", value=-0.5, control=(-0.8, 0.0))
    decision = solve_policy_pcbf(
        (certificate,),
        nominal_control=np.array([-0.5, 0.0]),
        lower=-np.ones(2),
        upper=np.ones(2),
        backup_policy_id="stop",
    )
    assert decision.feasible
    assert not decision.used_fallback
    assert decision.control[0] >= 0.2 - 1e-8


def test_policy_pcbf_value_threshold_cannot_replace_its_fixed_policy() -> None:
    certificate = _certificate("stop", value=-0.5, control=(-0.8, 0.0))
    decision = solve_policy_pcbf(
        (certificate, _certificate("room", value=100.0)),
        nominal_control=np.array([-0.5, 0.0]),
        lower=-np.ones(2),
        upper=np.ones(2),
        backup_policy_id="stop",
        safe_value_threshold=50.0,
    )
    assert decision.feasible
    assert decision.policy_id == "stop"
    assert not decision.used_fallback
    assert decision.control[0] >= 0.2 - 1e-8


def test_plcbf_all_unsafe_rule_solves_least_negative_policy_qp() -> None:
    decision = solve_plcbf(
        (
            _certificate("more-negative", value=-2.0, control=(-0.9, 0.0)),
            _certificate("least-negative", value=-0.1, control=(-0.8, 0.0)),
        ),
        nominal_control=np.array([-0.5, 0.0]),
        lower=-np.ones(2),
        upper=np.ones(2),
    )
    assert decision.feasible
    assert decision.policy_id == "least-negative"
    assert not decision.used_fallback
    # The QP projection satisfies u_x >= 0.2; it is not the direct -0.8
    # backup action.
    np.testing.assert_allclose(decision.control, [0.2, 0.0], atol=1e-8)


def test_fixed_policy_pcbf_qp_failure_returns_nominal_not_backup() -> None:
    decision = solve_policy_pcbf(
        (
            _certificate(
                "stop",
                value=1.0,
                control=(-0.8, 0.0),
                offset=2.0,
            ),
        ),
        nominal_control=np.array([0.35, -0.1]),
        lower=-np.ones(2),
        upper=np.ones(2),
        backup_policy_id="stop",
    )

    assert not decision.feasible
    assert decision.used_fallback
    np.testing.assert_allclose(decision.control, [0.35, -0.1])
    assert (
        decision.policy_decision.diagnostics.fallback_source
        == "nominal_after_fixed_pcbf_qp_failure"
    )


def test_library_pcbf_failure_uses_stop_emergency_not_selected_backup() -> None:
    decision = solve_library_pcbf_mi(
        (
            _certificate(
                "stop",
                value=-2.0,
                control=(-0.25, 0.0),
            ),
            _certificate(
                "room",
                value=-0.1,
                control=(0.8, 0.0),
            ),
        ),
        nominal_control=np.array([0.5, 0.0]),
        lower=-np.ones(2),
        upper=np.ones(2),
    )

    assert not decision.feasible
    assert decision.used_fallback
    assert decision.policy_id == "stop"
    np.testing.assert_allclose(decision.control, [-0.25, 0.0])
    assert (
        decision.policy_decision.diagnostics.fallback_source
        == "emergency_policy:stop"
    )


def test_library_pcbf_requires_strictly_positive_certificate_value() -> None:
    decision = solve_library_pcbf_mi(
        (
            _certificate(
                "stop",
                value=0.0,
                control=(-0.25, 0.0),
            ),
        ),
        nominal_control=np.array([0.5, 0.0]),
        lower=-np.ones(2),
        upper=np.ones(2),
    )

    assert not decision.feasible
    assert decision.used_fallback
    assert decision.policy_id == "stop"
    np.testing.assert_allclose(decision.control, [-0.25, 0.0])


def test_library_pcbf_objective_tie_preserves_policy_library_order() -> None:
    decision = solve_library_pcbf_mi(
        (
            _certificate("z-first", value=0.1),
            _certificate("a-second", value=100.0),
            _certificate("stop", value=-1.0),
        ),
        nominal_control=np.array([0.5, 0.0]),
        lower=-np.ones(2),
        upper=np.ones(2),
    )

    assert decision.feasible
    assert not decision.used_fallback
    assert decision.policy_id == "z-first"


def test_plcbf_selected_qp_failure_still_uses_selected_policy_backup() -> None:
    decision = solve_plcbf(
        (
            _certificate(
                "selected",
                value=-0.1,
                control=(-0.7, 0.0),
                offset=2.0,
            ),
            _certificate(
                "more-negative",
                value=-1.0,
                control=(0.8, 0.0),
            ),
        ),
        nominal_control=np.array([0.4, 0.0]),
        lower=-np.ones(2),
        upper=np.ones(2),
    )

    assert not decision.feasible
    assert decision.used_fallback
    assert decision.policy_id == "selected"
    np.testing.assert_allclose(decision.control, [-0.7, 0.0])
