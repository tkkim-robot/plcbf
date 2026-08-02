from __future__ import annotations

import numpy as np
import pytest

import plcbf.big_m_mpc as big_m_mpc
from plcbf.big_m_mpc import (
    BigMTrajectoryMPCConfig,
    BigMTrajectoryMPCProblem,
    build_big_m_trajectory_milp,
    solve_big_m_trajectory_mpc,
)


def _structural_problem() -> BigMTrajectoryMPCProblem:
    branch_states = np.array(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.5, 0.0]],
            [[0.0, 0.0], [-0.5, 0.0], [-1.5, 0.0]],
        ]
    )
    branch_controls = np.array([[[0.5], [0.5]], [[-0.5], [-0.5]]])
    return BigMTrajectoryMPCProblem(
        x0=np.array([0.0, 0.0]),
        A=np.array(
            [
                [[1.0, 0.2], [0.0, 1.0]],
                [[1.0, 0.3], [0.0, 1.0]],
            ]
        ),
        B=np.array([[[1.0], [0.0]], [[2.0], [0.0]]]),
        c=np.array([[0.1, 0.0], [0.2, 0.0]]),
        branch_states=branch_states,
        branch_controls=branch_controls,
        branch_safety=np.array([0.25, -0.1]),
        state_lower=np.array([-5.0, -2.0]),
        state_upper=np.array([5.0, 2.0]),
        control_lower=np.array([-1.0]),
        control_upper=np.array([1.0]),
        position_indices=(0,),
        velocity_indices=(1,),
        tracking_target=np.array([1.0]),
        terminal_target=np.array([2.0]),
        nominal_control=np.array([0.2]),
    )


def test_model_contains_full_trajectory_one_hot_and_affine_dynamics() -> None:
    problem = _structural_problem()
    config = BigMTrajectoryMPCConfig(
        position_tube=0.5,
        early_control_tube=0.25,
        early_control_steps=2,
    )
    model = build_big_m_trajectory_milp(problem, config)
    layout = model.layout

    assert layout.state.shape == (3, 2)
    assert layout.control.shape == (2, 1)
    assert layout.selector.shape == (2,)
    assert np.all(model.integrality[layout.state] == 0)
    assert np.all(model.integrality[layout.control] == 0)
    assert np.all(model.integrality[layout.selector] == 1)
    np.testing.assert_array_equal(model.eligible_branches, [True, False])
    np.testing.assert_array_equal(model.admissible_branches, [True, False])
    assert not model.safety_threshold_relaxed
    assert model.effective_safety_threshold == 0.0
    assert model.promoted_branch is None
    assert model.bounds.ub[layout.selector[0]] == 1.0
    assert model.bounds.ub[layout.selector[1]] == 0.0
    np.testing.assert_allclose(
        model.bounds.lb[layout.state],
        np.broadcast_to([-5.0, -2.0], layout.state.shape),
    )
    np.testing.assert_allclose(
        model.bounds.ub[layout.control],
        np.ones(layout.control.shape),
    )

    dynamics = model.row_groups["dynamics"]
    assert dynamics.stop - dynamics.start == 4
    dynamics_matrix = model.constraints.A[dynamics].toarray()
    dynamics_lower = model.constraints.lb[dynamics]
    dynamics_upper = model.constraints.ub[dynamics]

    # First scalar row:
    # x[1,0] - x[0,0] - 0.2*x[0,1] - u[0,0] = 0.1.
    expected = np.zeros(model.objective.size)
    expected[layout.state[1, 0]] = 1.0
    expected[layout.state[0, 0]] = -1.0
    expected[layout.state[0, 1]] = -0.2
    expected[layout.control[0, 0]] = -1.0
    np.testing.assert_allclose(dynamics_matrix[0], expected)
    assert dynamics_lower[0] == dynamics_upper[0] == 0.1

    # The second step uses its own A[1], B[1], and c[1].
    expected_second_step = np.zeros(model.objective.size)
    expected_second_step[layout.state[2, 0]] = 1.0
    expected_second_step[layout.state[1, 0]] = -1.0
    expected_second_step[layout.state[1, 1]] = -0.3
    expected_second_step[layout.control[1, 0]] = -2.0
    np.testing.assert_allclose(dynamics_matrix[2], expected_second_step)
    assert dynamics_lower[2] == dynamics_upper[2] == 0.2

    one_hot = model.row_groups["one_hot"]
    one_hot_row = model.constraints.A[one_hot].toarray()[0]
    expected_one_hot = np.zeros(model.objective.size)
    expected_one_hot[layout.selector] = 1.0
    np.testing.assert_allclose(one_hot_row, expected_one_hot)
    assert model.constraints.lb[one_hot][0] == 1.0
    assert model.constraints.ub[one_hot][0] == 1.0

    safety_rows = model.row_groups["safety_admission"]
    assert safety_rows.stop - safety_rows.start == 2
    first_safety_row = model.constraints.A[safety_rows.start].toarray()[0]
    assert first_safety_row[layout.selector[0]] == 50.0
    assert model.constraints.ub[safety_rows.start] == 50.25
    np.testing.assert_allclose(model.safety_big_m, [50.0, 50.0])

    position_rows = model.row_groups["position_tubes"]
    control_rows = model.row_groups["control_tubes"]
    assert position_rows.stop - position_rows.start == 2 * 2 * 2 * 1
    assert control_rows.stop - control_rows.start == 2 * 2 * 2 * 1

    # The first active position-tube row is
    # x[1,0] + M*z[0] <= ref + tube + M.  M=5 is the tight value that
    # completely deactivates both signed rows over x in [-5, 5].
    assert model.position_big_m[0, 0, 0] == 5.0
    first_position_row = model.constraints.A[position_rows.start].toarray()[0]
    assert first_position_row[layout.state[1, 0]] == 1.0
    assert first_position_row[layout.selector[0]] == 5.0
    assert model.constraints.ub[position_rows.start] == 6.0

    # The control tube is also genuinely tied to z rather than constraining
    # every branch rollout simultaneously.
    assert model.control_big_m[0, 0, 0] == 1.25
    first_control_row = model.constraints.A[control_rows.start].toarray()[0]
    assert first_control_row[layout.control[0, 0]] == 1.0
    assert first_control_row[layout.selector[0]] == 1.25
    assert model.constraints.ub[control_rows.start] == 2.0


def test_numeric_milp_selects_safe_goal_directed_branch() -> None:
    # Two exact branch tubes for a scalar integrator.  Both are safe, and the
    # L1 goal objective must jointly select the positive branch and its u/x
    # trajectory rather than merely choosing a precomputed control one-hot.
    problem = BigMTrajectoryMPCProblem(
        x0=np.array([0.0]),
        A=np.array([[1.0]]),
        B=np.array([[1.0]]),
        c=np.array([0.0]),
        branch_states=np.array(
            [
                [[0.0], [1.0], [2.0]],
                [[0.0], [-1.0], [-2.0]],
            ]
        ),
        branch_controls=np.array([[[1.0], [1.0]], [[-1.0], [-1.0]]]),
        branch_safety=np.array([0.4, 0.5]),
        state_lower=np.array([-3.0]),
        state_upper=np.array([3.0]),
        control_lower=np.array([-1.0]),
        control_upper=np.array([1.0]),
        position_indices=(0,),
        tracking_target=np.array([2.0]),
        terminal_target=np.array([2.0]),
        nominal_control=np.array([0.0]),
    )
    result = solve_big_m_trajectory_mpc(
        problem,
        BigMTrajectoryMPCConfig(
            position_tube=0.0,
            early_control_tube=0.0,
            early_control_steps=2,
            tracking_weight=8.0,
            terminal_weight=16.0,
            velocity_weight=0.0,
            control_weight=0.02,
            nominal_weight=0.5,
            time_limit_s=5.0,
            mip_rel_gap=0.0,
        ),
    )

    assert result.feasible
    assert result.safety_feasible
    assert not result.used_fallback
    assert result.status == "optimal"
    assert not result.safety_threshold_relaxed
    assert result.effective_safety_threshold == 0.0
    assert result.selected_branch_meets_safety_threshold
    assert result.selected_branch == 0
    np.testing.assert_allclose(result.selector, [1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(result.control, [1.0], atol=1e-7)
    np.testing.assert_allclose(result.control_trajectory, [[1.0], [1.0]])
    np.testing.assert_allclose(result.state_trajectory[:, 0], [0.0, 1.0, 2.0])


def test_selector_only_chooses_disjunction_while_mpc_optimizes_controls() -> None:
    # The branch rollout has zero control, but its nonzero tubes permit the
    # trajectory MPC to optimize a different u sequence toward the goal.  A
    # fake z-only branch selector would return the stored zero branch action.
    problem = BigMTrajectoryMPCProblem(
        x0=np.array([0.0]),
        A=np.array([[1.0]]),
        B=np.array([[1.0]]),
        branch_states=np.array([[[0.0], [0.0], [0.0]]]),
        branch_controls=np.array([[[0.0], [0.0]]]),
        branch_safety=np.array([1.0]),
        state_lower=np.array([-3.0]),
        state_upper=np.array([3.0]),
        control_lower=np.array([-1.0]),
        control_upper=np.array([1.0]),
        position_indices=(0,),
        tracking_target=np.array([2.0]),
        terminal_target=np.array([2.0]),
        nominal_control=np.array([0.0]),
    )
    result = solve_big_m_trajectory_mpc(
        problem,
        BigMTrajectoryMPCConfig(
            position_tube=2.0,
            early_control_tube=1.0,
            early_control_steps=2,
            tracking_weight=8.0,
            terminal_weight=16.0,
            velocity_weight=0.0,
            control_weight=0.0,
            nominal_weight=0.0,
            safety_tiebreak_weight=0.0,
            time_limit_s=5.0,
            mip_rel_gap=0.0,
        ),
    )

    assert result.feasible
    assert result.safety_feasible
    assert result.selected_branch == 0
    np.testing.assert_allclose(result.selector, [1.0], atol=1e-8)
    np.testing.assert_allclose(result.control_trajectory, [[1.0], [1.0]])
    np.testing.assert_allclose(result.state_trajectory[:, 0], [0.0, 1.0, 2.0])
    assert not np.allclose(result.control, problem.branch_controls[0, 0])


def test_no_safe_branch_uses_warehouse_max_safety_emergency_admission() -> None:
    problem = _structural_problem()
    problem = BigMTrajectoryMPCProblem(
        **{
            **problem.__dict__,
            "branch_safety": np.array([-0.01, -2.0]),
            "fallback_control": np.array([0.4]),
        }
    )
    config = BigMTrajectoryMPCConfig(safety_threshold=0.0)
    model = build_big_m_trajectory_milp(problem, config)
    np.testing.assert_array_equal(model.eligible_branches, [False, False])
    np.testing.assert_array_equal(model.admissible_branches, [True, False])
    assert model.safety_threshold_relaxed
    assert model.promoted_branch == 0
    assert model.effective_safety_threshold == pytest.approx(-0.010001)
    assert model.bounds.ub[model.layout.selector[0]] == 1.0
    assert model.bounds.ub[model.layout.selector[1]] == 0.0

    result = solve_big_m_trajectory_mpc(
        problem,
        config,
    )

    assert result.feasible
    assert not result.safety_feasible
    assert not result.used_fallback
    assert result.status == "optimal_safety_threshold_relaxed"
    assert result.safety_threshold == 0.0
    assert result.effective_safety_threshold == pytest.approx(-0.010001)
    assert result.safety_threshold_relaxed
    assert result.eligible_branches == ()
    assert result.admissible_branches == (0,)
    assert result.promoted_branch == 0
    assert result.selected_branch == 0
    assert result.selected_branch_safety == -0.01
    assert not result.selected_branch_meets_safety_threshold
    assert "warehouse emergency admission promoted branch 0" in (
        result.solver_message
    )


def test_hard_exclusion_still_returns_explicit_fallback_without_solver(
    monkeypatch,
) -> None:
    problem = _structural_problem()
    problem = BigMTrajectoryMPCProblem(
        **{
            **problem.__dict__,
            "branch_safety": np.array([-0.01, -2.0]),
            "branch_eligible": np.array([False, False]),
            "fallback_control": np.array([0.4]),
        }
    )

    def fail_if_called(**_kwargs):
        raise AssertionError("MILP must not run when all branches are excluded")

    monkeypatch.setattr(big_m_mpc, "milp", fail_if_called)
    result = solve_big_m_trajectory_mpc(problem)

    assert not result.feasible
    assert not result.safety_feasible
    assert result.used_fallback
    assert result.status == "infeasible_no_admissible_branch"
    assert not result.safety_threshold_relaxed
    assert result.eligible_branches == ()
    assert result.admissible_branches == ()
    assert result.promoted_branch is None
    assert result.fallback_source == "caller_emergency_control"
    assert result.fallback_branch is None
    np.testing.assert_allclose(result.control, [0.4])


def test_safe_but_dynamically_infeasible_problem_reports_fallback() -> None:
    # The selected safe rollout demands x[1] == 1, but the zero-input affine
    # dynamics force x[1] == 0.  The emergency control is returned explicitly
    # while feasibility remains false.
    problem = BigMTrajectoryMPCProblem(
        x0=np.array([0.0]),
        A=np.array([[1.0]]),
        B=np.array([[0.0]]),
        branch_states=np.array([[[0.0], [1.0]]]),
        branch_controls=np.array([[[0.0]]]),
        branch_safety=np.array([1.0]),
        state_lower=np.array([-2.0]),
        state_upper=np.array([2.0]),
        control_lower=np.array([-1.0]),
        control_upper=np.array([1.0]),
        position_indices=(0,),
        nominal_control=np.array([0.8]),
        fallback_control=np.array([-0.3]),
    )
    result = solve_big_m_trajectory_mpc(
        problem,
        BigMTrajectoryMPCConfig(
            position_tube=0.0,
            early_control_tube=0.0,
            time_limit_s=5.0,
        ),
    )

    assert not result.feasible
    assert not result.safety_feasible
    assert result.used_fallback
    assert result.status == "infeasible"
    assert result.selected_branch is None
    assert result.fallback_source == "warehouse_max_safety_branch_blend"
    assert result.fallback_branch == 0
    # 0.75 * branch_u0(=0) + 0.25 * nominal(=0.8).
    np.testing.assert_allclose(result.control, [0.2])
    assert result.state_trajectory is None
    assert result.control_trajectory is None


def test_rejects_big_m_that_cannot_deactivate_unselected_branch() -> None:
    with np.testing.assert_raises_regex(ValueError, "big_m_position is too small"):
        build_big_m_trajectory_milp(
            _structural_problem(),
            BigMTrajectoryMPCConfig(
                position_tube=0.0,
                big_m_position=0.1,
            ),
        )
