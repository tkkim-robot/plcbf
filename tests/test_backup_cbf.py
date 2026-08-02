from dataclasses import replace

import numpy as np
import pytest

from plcbf.backup_cbf import (
    evaluate_backup_cbf_candidate,
    solve_fixed_backup_cbf,
    solve_multi_backup_cbf_min_intervention,
)


def _candidate(
    policy_id,
    terminal_offset,
    *,
    formulation="strict_multi",
    path_constraint_start_index=0,
):
    def margins(state, time_offset):
        # Composed rollout margins with nonzero sensitivity to the controlled
        # velocity state.  Three path rows plus an explicit terminal row must
        # survive into the candidate QP.
        return (
            np.array(
                [
                    state[0] - time_offset,
                    state[0] + state[1] - time_offset,
                    state[0] + 2.0 * state[1] - time_offset,
                ]
            ),
            state[0] + state[1] + terminal_offset - time_offset,
        )

    return evaluate_backup_cbf_candidate(
        policy_id=policy_id,
        state=np.array([1.0, 0.0]),
        nominal_control=np.array([-1.0]),
        lower=np.array([-2.0]),
        upper=np.array([2.0]),
        drift=np.array([0.0, 0.0]),
        control_matrix=np.array([[1.0], [0.0]]),
        backup_closed_loop_drift=np.array([0.0, 0.0]),
        rollout_margins=margins,
        gradient_steps=np.array([1e-4, 1e-4]),
        time_derivative_step=1e-3,
        alpha=1.0,
        terminal_alpha=2.0,
        formulation=formulation,
        path_constraint_start_index=path_constraint_start_index,
    )


def test_backup_cbf_contains_every_path_row_and_terminal_row():
    candidate = _candidate("fixed", 1.0)
    assert len(candidate.path_values) == 3
    assert len(candidate.halfspaces) == 4
    assert candidate.halfspaces[-1].label == "fixed:terminal"
    assert candidate.rollout_safe
    assert candidate.terminal_safe


def test_multi_backup_uses_minimum_intervention_feasible_full_qp():
    first = _candidate("first", 1.0)
    second = _candidate("second", 2.0)
    decision = solve_multi_backup_cbf_min_intervention(
        (first, second),
        direct_backup_controls={
            "first": np.array([0.0]),
            "second": np.array([0.0]),
        },
        lower=np.array([-2.0]),
        upper=np.array([2.0]),
    )
    expected = min((first, second), key=lambda item: item.objective)
    assert decision.policy_id == expected.policy_id
    assert decision.feasible
    assert not decision.used_fallback


def test_multi_backup_objective_tie_preserves_policy_library_order():
    first = _candidate("z-later-lexically", 1.0)
    second = replace(
        first,
        policy_id="a-earlier-lexically",
        terminal_value=first.terminal_value + 100.0,
    )
    decision = solve_multi_backup_cbf_min_intervention(
        (first, second),
        direct_backup_controls={
            first.policy_id: np.array([0.0]),
            second.policy_id: np.array([0.0]),
        },
        lower=np.array([-2.0]),
        upper=np.array([2.0]),
    )

    assert decision.policy_id == first.policy_id


def test_path_rows_include_the_warehouse_backup_flow_correction():
    def margins(state, time_offset):
        return np.array([state[0] - time_offset]), state[0] + 10.0

    candidate = evaluate_backup_cbf_candidate(
        policy_id="flow",
        state=np.array([1.0]),
        nominal_control=np.array([0.0]),
        lower=np.array([-10.0]),
        upper=np.array([10.0]),
        drift=np.array([2.0]),
        control_matrix=np.array([[1.0]]),
        backup_closed_loop_drift=np.array([5.0]),
        rollout_margins=margins,
        gradient_steps=np.array([1e-4]),
        time_derivative_step=1e-3,
        alpha=1.0,
        terminal_alpha=1.0,
    )

    # -Qf + Qf_backup - partial_t h - alpha(h)
    # = -2 + 5 - (-1) - 1 = 3.
    assert candidate.halfspaces[0].offset == pytest.approx(3.0)


def test_explicit_path_flow_derivative_supports_time_indexed_backup_policy():
    def margins(state, time_offset):
        return np.array([state[0] - time_offset]), state[0] + 10.0

    candidate = evaluate_backup_cbf_candidate(
        policy_id="time-indexed",
        state=np.array([1.0]),
        nominal_control=np.array([0.0]),
        lower=np.array([-10.0]),
        upper=np.array([10.0]),
        drift=np.array([2.0]),
        control_matrix=np.array([[1.0]]),
        backup_closed_loop_drift=np.array([100.0]),
        path_flow_derivatives=np.array([7.0]),
        rollout_margins=margins,
        gradient_steps=np.array([1e-4]),
        time_derivative_step=1e-3,
        alpha=1.0,
        terminal_alpha=1.0,
    )

    # The explicit local grad_h(phi_i) @ f_policy_i = 7 overrides the
    # autonomous-flow approximation based on the unrelated initial drift 100.
    assert candidate.halfspaces[0].offset == pytest.approx(5.0)


def test_terminal_row_omits_time_derivative_like_warehouse_backup_cbf():
    def margins(state, time_offset):
        return np.array([state[0] + 10.0]), state[0] - 100.0 * time_offset

    candidate = evaluate_backup_cbf_candidate(
        policy_id="terminal",
        state=np.array([1.0]),
        nominal_control=np.array([0.0]),
        lower=np.array([-10.0]),
        upper=np.array([10.0]),
        drift=np.array([2.0]),
        control_matrix=np.array([[1.0]]),
        backup_closed_loop_drift=np.array([0.0]),
        rollout_margins=margins,
        gradient_steps=np.array([1e-4]),
        time_derivative_step=1e-3,
        alpha=1.0,
        terminal_alpha=3.0,
    )

    # -(grad_terminal @ f0 + alpha_terminal * h_terminal)
    # = -(1 * 2 + 3 * 1) = -5.  The -100 explicit time derivative is
    # intentionally absent from the warehouse terminal invariant-set row.
    assert candidate.halfspaces[-1].offset == pytest.approx(-5.0)


def test_multi_backup_failure_uses_fixed_stop_emergency():
    stop = _candidate("stop", -10.0)
    evasive = _candidate("evasive", -20.0)
    assert not stop.feasible
    assert not evasive.feasible

    decision = solve_multi_backup_cbf_min_intervention(
        (evasive, stop),
        direct_backup_controls={
            "evasive": np.array([1.5]),
            "stop": np.array([-0.25]),
        },
        lower=np.array([-2.0]),
        upper=np.array([2.0]),
    )

    assert not decision.feasible
    assert decision.used_fallback
    assert decision.policy_id == "stop"
    np.testing.assert_allclose(decision.control, [-0.25])


def test_fixed_backup_uses_qp_even_when_open_loop_certificate_is_unsafe():
    candidate = _candidate("fixed", -1.1, formulation="single")
    assert candidate.control is not None
    assert not candidate.feasible

    decision = solve_fixed_backup_cbf(
        candidate,
        direct_backup_control=np.array([-2.0]),
        lower=np.array([-2.0]),
        upper=np.array([2.0]),
    )

    assert decision.feasible
    assert not decision.safety_feasible
    assert not decision.used_fallback
    np.testing.assert_allclose(decision.control, candidate.control)


def test_fixed_backup_qp_failure_uses_nominal_while_margin_is_safe():
    candidate = _candidate("fixed", 1.0)
    assert candidate.rollout_safe and candidate.terminal_safe
    failed = replace(
        candidate,
        control=None,
        feasible=False,
        objective=float("inf"),
        status="infeasible",
    )

    decision = solve_fixed_backup_cbf(
        failed,
        direct_backup_control=np.array([-2.0]),
        lower=np.array([-2.0]),
        upper=np.array([2.0]),
    )

    assert not decision.feasible
    assert decision.safety_feasible
    assert decision.used_fallback
    assert decision.status == "nominal_after_qp_failure:infeasible"
    np.testing.assert_allclose(decision.control, failed.nominal_control)


def test_single_backup_skips_initial_path_row_but_retains_its_margin():
    candidate = _candidate(
        "fixed",
        1.0,
        formulation="single",
        path_constraint_start_index=1,
    )

    assert len(candidate.path_values) == 3
    assert [row.label for row in candidate.halfspaces] == [
        "fixed:path[1]",
        "fixed:path[2]",
        "fixed:terminal",
    ]


def test_single_and_strict_multi_match_warehouse_zero_control_rows():
    def margins(state, time_offset):
        return np.array([state[0] - time_offset]), state[0] + 10.0

    common = dict(
        policy_id="zero-row",
        state=np.array([1.0]),
        nominal_control=np.array([0.4]),
        lower=np.array([-1.0]),
        upper=np.array([1.0]),
        drift=np.array([0.0]),
        control_matrix=np.array([[0.0]]),
        backup_closed_loop_drift=np.array([2.0]),
        rollout_margins=margins,
        gradient_steps=np.array([1e-4]),
        time_derivative_step=1e-3,
        alpha=1.0,
        terminal_alpha=1.0,
    )
    single = evaluate_backup_cbf_candidate(
        **common,
        formulation="single",
    )
    strict = evaluate_backup_cbf_candidate(
        **common,
        formulation="strict_multi",
    )

    # The path row is 0*u >= 2: the legacy single controller skips it,
    # whereas strict multi rejects it before invoking the QP.
    assert single.control is not None
    np.testing.assert_allclose(single.control, [0.4])
    assert not strict.feasible
    assert strict.control is None
    assert strict.status == "infeasible_constant_path_constraint"


def test_strict_multi_rejects_unsafe_rollout_before_qp():
    strict = _candidate("strict", -1.1, formulation="strict_multi")
    single = _candidate("single", -1.1, formulation="single")

    assert strict.control is None
    assert strict.status == "uncertified_terminal"
    assert strict.solve_time_s == 0.0
    assert single.control is not None


def test_backup_qp_uses_warehouse_scaled_weighted_intervention_objective():
    def margins(state, _time_offset):
        return np.array([state[0]]), 1.0

    candidate = evaluate_backup_cbf_candidate(
        policy_id="scaled",
        state=np.array([0.0]),
        nominal_control=np.zeros(2),
        lower=np.array([-1.0, -2.0]),
        upper=np.array([1.0, 2.0]),
        drift=np.array([-1.0]),
        control_matrix=np.array([[1.0, 1.0]]),
        backup_closed_loop_drift=np.array([0.0]),
        rollout_margins=margins,
        gradient_steps=np.array([1e-4]),
        time_derivative_step=1e-3,
        alpha=0.0,
        terminal_alpha=1.0,
        formulation="strict_multi",
        control_scales=np.array([1.0, 2.0]),
        control_weights=np.ones(2),
    )

    assert candidate.feasible
    np.testing.assert_allclose(candidate.control, [0.2, 0.8], atol=2e-5)
    assert candidate.objective == pytest.approx(0.2, abs=2e-5)
