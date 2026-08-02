"""Trajectory-commitment shielding used by the MPS and Gatekeeper baselines.

The warehouse implementations of MPS and Gatekeeper are trajectory
algorithms, not policy-certificate selectors.  This module keeps that
distinction explicit while allowing a case study to supply its own dynamics,
nominal controller, fixed backup controller, and collision geometry.

Neither class selects among a policy library.  The fixed backup controller is
chosen once when the shield is constructed.  In particular, this module has
no notion of a hospital room, blockage event, refuge phase, hold timer, or
release guard.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
StateStep = Callable[[FloatArray, FloatArray], FloatArray]
FeedbackControl = Callable[[FloatArray], FloatArray]
TrajectoryValidator = Callable[[FloatArray], bool]


def _vector(value: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result.copy()


@dataclass(frozen=True)
class CommittedTrajectory:
    """A complete nominal-prefix plus fixed-backup plan."""

    states: FloatArray
    controls: FloatArray
    nominal_steps: int
    backup_policy_id: str

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=float)
        controls = np.asarray(self.controls, dtype=float)
        if states.ndim != 2 or controls.ndim != 2:
            raise ValueError("trajectory states and controls must be matrices")
        if states.shape[0] != controls.shape[0] + 1:
            raise ValueError("a trajectory must contain one more state than control")
        if states.shape[0] == 0 or not (
            np.all(np.isfinite(states)) and np.all(np.isfinite(controls))
        ):
            raise ValueError("trajectory entries must be finite")
        nominal_steps = int(self.nominal_steps)
        if nominal_steps < 0 or nominal_steps > controls.shape[0]:
            raise ValueError("nominal_steps is outside the control trajectory")
        states = states.copy()
        controls = controls.copy()
        states.setflags(write=False)
        controls.setflags(write=False)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "controls", controls)
        object.__setattr__(self, "nominal_steps", nominal_steps)
        object.__setattr__(self, "backup_policy_id", str(self.backup_policy_id))


@dataclass(frozen=True)
class ShieldDecision:
    """One receding-horizon control decision."""

    control: FloatArray
    feasible: bool
    status: str
    used_committed_backup: bool
    committed_index: int
    committed_trajectory: CommittedTrajectory
    solve_time_s: float

    def __post_init__(self) -> None:
        control = _vector(self.control, "control")
        control.setflags(write=False)
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "feasible", bool(self.feasible))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(
            self, "used_committed_backup", bool(self.used_committed_backup)
        )
        object.__setattr__(self, "committed_index", int(self.committed_index))
        object.__setattr__(self, "solve_time_s", float(self.solve_time_s))


class _TrajectoryCommitmentShield:
    """Common committed-trajectory machinery for MPS and Gatekeeper."""

    def __init__(
        self,
        *,
        step: StateStep,
        nominal_control: FeedbackControl,
        backup_control: FeedbackControl,
        trajectory_is_safe: TrajectoryValidator,
        backup_horizon_steps: int,
        backup_policy_id: str,
    ) -> None:
        if int(backup_horizon_steps) <= 0:
            raise ValueError("backup_horizon_steps must be positive")
        if not str(backup_policy_id):
            raise ValueError("backup_policy_id must not be empty")
        self._step = step
        self._nominal_control = nominal_control
        self._backup_control = backup_control
        self._trajectory_is_safe = trajectory_is_safe
        self.backup_horizon_steps = int(backup_horizon_steps)
        self.backup_policy_id = str(backup_policy_id)
        self.committed: CommittedTrajectory | None = None
        self.committed_is_safe = False
        self.current_index = 0

    def reset(self) -> None:
        self.committed = None
        self.committed_is_safe = False
        self.current_index = 0

    def _candidate(
        self,
        initial_state: ArrayLike,
        nominal_steps: int,
    ) -> CommittedTrajectory:
        state = _vector(initial_state, "initial_state")
        states = [state.copy()]
        controls: list[FloatArray] = []
        requested_nominal_steps = max(0, int(nominal_steps))

        for _ in range(requested_nominal_steps):
            control = _vector(
                self._nominal_control(state.copy()),
                "nominal control",
            )
            state = _vector(self._step(state.copy(), control.copy()), "next state")
            controls.append(control)
            states.append(state.copy())

        for _ in range(self.backup_horizon_steps):
            control = _vector(
                self._backup_control(state.copy()),
                "backup control",
            )
            state = _vector(self._step(state.copy(), control.copy()), "next state")
            controls.append(control)
            states.append(state.copy())

        return CommittedTrajectory(
            states=np.asarray(states),
            controls=np.asarray(controls),
            nominal_steps=requested_nominal_steps,
            backup_policy_id=self.backup_policy_id,
        )

    def _initialize(self, state: ArrayLike) -> bool:
        plan = self._candidate(state, 0)
        self.committed = plan
        self.current_index = 0
        self.committed_is_safe = bool(self._trajectory_is_safe(plan.states))
        return self.committed_is_safe

    def _commit(self, plan: CommittedTrajectory) -> None:
        self.committed = plan
        # Callers commit only after validating the complete candidate.
        self.committed_is_safe = True
        self.current_index = 0

    def _output(
        self,
        *,
        started_at: float,
        feasible: bool,
        status: str,
    ) -> ShieldDecision:
        assert self.committed is not None
        if self.current_index >= self.committed.controls.shape[0]:
            raise RuntimeError("committed trajectory is exhausted")
        output_index = self.current_index
        control = self.committed.controls[output_index].copy()
        used_backup = output_index >= self.committed.nominal_steps
        self.current_index += 1
        return ShieldDecision(
            control=control,
            feasible=feasible,
            status=status,
            used_committed_backup=used_backup,
            committed_index=output_index,
            committed_trajectory=self.committed,
            solve_time_s=time.perf_counter() - started_at,
        )

    def _ensure_unexhausted_backup(self, state: ArrayLike) -> bool:
        if (
            self.committed is not None
            and self.current_index < self.committed.controls.shape[0]
        ):
            return self.committed_is_safe
        return self._initialize(state)


class ModelPredictiveShield(_TrajectoryCommitmentShield):
    """Warehouse-style MPS: one nominal step followed by a fixed backup.

    At every physical step a new candidate is tested.  A valid candidate is
    committed in full.  If it is invalid, execution continues at the current
    index of the previously committed trajectory.
    """

    def solve(self, state: ArrayLike) -> ShieldDecision:
        started_at = time.perf_counter()
        initial_backup_safe = self._ensure_unexhausted_backup(state)
        candidate = self._candidate(state, 1)
        if self._trajectory_is_safe(candidate.states):
            self._commit(candidate)
            return self._output(
                started_at=started_at,
                feasible=True,
                status="committed_one_step_nominal",
            )
        return self._output(
            started_at=started_at,
            feasible=initial_backup_safe,
            status=(
                "continued_committed_trajectory"
                if initial_backup_safe
                else "unsafe_initial_backup"
            ),
        )


class GatekeeperShield(_TrajectoryCommitmentShield):
    """Warehouse-style Gatekeeper with longest-safe nominal-prefix search."""

    def __init__(
        self,
        *,
        nominal_horizon_steps: int,
        horizon_discount_steps: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if int(nominal_horizon_steps) < 0:
            raise ValueError("nominal_horizon_steps must be nonnegative")
        if int(horizon_discount_steps) <= 0:
            raise ValueError("horizon_discount_steps must be positive")
        self.nominal_horizon_steps = int(nominal_horizon_steps)
        self.horizon_discount_steps = int(horizon_discount_steps)

    def _prefix_lengths(self) -> tuple[int, ...]:
        values = list(
            range(
                self.nominal_horizon_steps,
                -1,
                -self.horizon_discount_steps,
            )
        )
        if not values or values[-1] != 0:
            values.append(0)
        return tuple(values)

    def solve(self, state: ArrayLike) -> ShieldDecision:
        started_at = time.perf_counter()
        initial_backup_safe = self._ensure_unexhausted_backup(state)
        for nominal_steps in self._prefix_lengths():
            candidate = self._candidate(state, nominal_steps)
            if self._trajectory_is_safe(candidate.states):
                self._commit(candidate)
                return self._output(
                    started_at=started_at,
                    feasible=True,
                    status=f"committed_prefix:{nominal_steps}",
                )
        return self._output(
            started_at=started_at,
            feasible=initial_backup_safe,
            status=(
                "continued_committed_trajectory"
                if initial_backup_safe
                else "unsafe_initial_backup"
            ),
        )


__all__ = [
    "CommittedTrajectory",
    "GatekeeperShield",
    "ModelPredictiveShield",
    "ShieldDecision",
]
