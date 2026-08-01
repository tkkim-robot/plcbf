import numpy as np

from plcbf.trajectory_shielding import GatekeeperShield, ModelPredictiveShield


def _step(state, control):
    return np.asarray(state) + np.asarray(control)


def _nominal(_state):
    return np.array([1.0])


def _backup(_state):
    return np.array([-1.0])


def test_mps_continues_previous_committed_plan_when_candidate_is_invalid():
    safety_limit = {"maximum": 10.0}

    def safe(states):
        return bool(np.max(states[:, 0]) <= safety_limit["maximum"])

    shield = ModelPredictiveShield(
        step=_step,
        nominal_control=_nominal,
        backup_control=_backup,
        trajectory_is_safe=safe,
        backup_horizon_steps=3,
        backup_policy_id="fixed_backup",
    )
    first = shield.solve(np.array([0.0]))
    assert first.status == "committed_one_step_nominal"
    assert first.control == np.array([1.0])

    safety_limit["maximum"] = -100.0
    second = shield.solve(np.array([1.0]))
    assert second.status == "continued_committed_trajectory"
    # Index one of the previously committed plan is the first backup input.
    assert second.control == np.array([-1.0])
    assert second.committed_index == 1
    assert second.used_committed_backup


def test_gatekeeper_commits_longest_safe_prefix_without_release_counter():
    def safe(states):
        return bool(np.max(states[:, 0]) <= 2.0)

    shield = GatekeeperShield(
        step=_step,
        nominal_control=_nominal,
        backup_control=_backup,
        trajectory_is_safe=safe,
        backup_horizon_steps=2,
        backup_policy_id="fixed_backup",
        nominal_horizon_steps=4,
        horizon_discount_steps=1,
    )
    decision = shield.solve(np.array([0.0]))
    assert decision.status == "committed_prefix:2"
    assert decision.committed_trajectory.nominal_steps == 2
    assert decision.control == np.array([1.0])
    assert not hasattr(shield, "safe_release_steps")
    assert not hasattr(shield, "committed_policy_id")


def test_shields_use_one_fixed_backup_identity():
    shield = ModelPredictiveShield(
        step=_step,
        nominal_control=_nominal,
        backup_control=_backup,
        trajectory_is_safe=lambda states: True,
        backup_horizon_steps=2,
        backup_policy_id="retrace",
    )
    decision = shield.solve(np.array([0.0]))
    assert decision.committed_trajectory.backup_policy_id == "retrace"


def test_unsafe_initialized_backup_remains_reported_unsafe_while_committed():
    shield = ModelPredictiveShield(
        step=_step,
        nominal_control=_nominal,
        backup_control=_backup,
        trajectory_is_safe=lambda states: False,
        backup_horizon_steps=3,
        backup_policy_id="fixed_backup",
    )

    first = shield.solve(np.array([0.0]))
    second = shield.solve(np.array([-1.0]))

    assert not first.feasible
    assert not second.feasible
    assert first.status == "unsafe_initial_backup"
    assert second.status == "unsafe_initial_backup"
    assert not shield.committed_is_safe
