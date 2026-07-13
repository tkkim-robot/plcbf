"""Benchmark-adapted Chen et al. multi-backup CBF for the drift car.

Each candidate reuses the repository BackupCBF rollout, sensitivity, safety,
terminal-set, objective, and solver definitions.  Unlike the legacy class,
the strict candidate wrapper exposes failed QPs instead of silently applying a
fallback.  The outer controller selects the feasible candidate with minimum
realized intervention.  This is a benchmark adaptation, not a line-by-line
reproduction of Chen, Singletary, and Ames (2021).

For terminal compatibility, each non-stop strategy executes its exact runtime
library maneuver for a documented 1.0 s prefix and then uses the exact shared
stop policy for the remaining horizon; stop remains stop throughout.  Terminal
membership uses a named sampled stopping envelope at the terminal sample and
one successor under stop.  This finite check is intentionally not claimed to
be a formal invariant-set proof.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

import cvxpy as cp
import jax.numpy as jnp
import numpy as np

from safe_control.position_control.backup_cbf_qp import BackupCBF

from examples.drift_car.algorithms.multi_policy_baseline_common_drift import (
    CandidateCBFResult,
    MultiPolicyMetrics,
    NOMINAL_POLICY_REPRESENTATION,
    assert_runtime_library_equal,
    project_bounded_control,
    select_minimum_intervention,
)
from examples.drift_car.algorithms.plcbf_drift import PLCBF
from examples.drift_car.controllers.drift_policies_jax import (
    LaneChangeControllerJAX,
    StoppingControllerJAX,
)


@dataclass(frozen=True)
class DriftSampledStoppingTerminalEnvelope:
    """Sampled stop-tail terminal proxy used by every drift MB candidate.

    The speed tolerance is deliberately named a stopping-envelope tolerance,
    not "near rest": the exact shared stop policy cannot bring every 1.0 s
    maneuver-prefix candidate near zero speed within the fixed 3.0 s benchmark
    horizon, especially at the sensed 0.3 friction.  The 7.5 m/s bound includes
    those sampled prefix/tail rollouts while requiring a non-increasing absolute
    speed at one exact-stop successor.  This finite collection is auditable but
    is not a proof that the set is control invariant.
    """

    name: str = "drift_sampled_stop_tail_envelope_v1"
    longitudinal_speed_abs_max: float = 7.5
    yaw_rate_abs_max: float = 0.05
    sideslip_abs_max: float = 0.05
    steering_abs_max: float = float(np.deg2rad(5.0))
    stop_successor_abs_speed_increase_max: float = 1e-6
    safety_margin_min: float = 0.0


class _CompoundLibraryPolicyAdapter:
    """PL-CBF maneuver prefix followed by the exact shared stopping policy."""

    def __init__(self, config: dict, stop_config: dict, maneuver_prefix_steps: int):
        self.config = config
        self.stop_config = stop_config
        self.maneuver_prefix_steps = int(maneuver_prefix_steps)

    def _stop(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(
            StoppingControllerJAX.compute(jnp.asarray(state), self.stop_config["params"]),
            dtype=float,
        ).reshape(-1)

    def control_at_step(self, state: np.ndarray, step: int) -> np.ndarray:
        if self.config["type"] == "stop" or step >= self.maneuver_prefix_steps:
            return self._stop(state)
        state_jax = jnp.asarray(state)
        if self.config["type"] == "lane_change":
            value = LaneChangeControllerJAX.compute(state_jax, self.config["params"])
        else:
            raise ValueError(f"Unsupported drift policy type: {self.config['type']}")
        return np.asarray(value, dtype=float).reshape(-1)

    def compute_control(self, state, target=None):
        del target
        return self.control_at_step(np.asarray(state).reshape(-1), 0).reshape(-1, 1)


class _FrozenNominalPolicyAdapter:
    """Pure frozen-MPCC prefix followed by the common exact stopping tail.

    Candidate rollout never calls or advances MPCC.  At the documented common
    prefix boundary (or earlier if the frozen plan is shorter), it switches to
    exactly the same stopping controller used by every other compound strategy.
    """

    def __init__(self, stop_config: dict, maneuver_prefix_steps: int):
        self.stop_config = stop_config
        self.maneuver_prefix_steps = int(maneuver_prefix_steps)
        self.controls = np.empty((0, 2), dtype=float)

    def set_controls(self, controls: Optional[np.ndarray]) -> None:
        self.controls = (
            np.empty((0, 2), dtype=float)
            if controls is None
            else np.array(controls, dtype=float, copy=True).reshape(-1, 2)
        )

    def control_at_step(self, state: np.ndarray, step: int) -> np.ndarray:
        if step < self.maneuver_prefix_steps and step < len(self.controls):
            return self.controls[step].copy()
        stop = np.asarray(
            StoppingControllerJAX.compute(jnp.asarray(state), self.stop_config["params"]),
            dtype=float,
        ).reshape(-1)
        return stop

    def compute_control(self, state, target=None):
        del target
        return self.control_at_step(np.asarray(state).reshape(-1), 0).reshape(-1, 1)


class _StrictCandidateBackupCBF(BackupCBF):
    """BackupCBF candidate whose failed QP is reported, never hidden."""

    accepted_statuses = ("optimal", "optimal_inaccurate")

    def __init__(
        self,
        *args,
        policy_name: str,
        policy_adapter,
        stop_config: dict,
        terminal_envelope: DriftSampledStoppingTerminalEnvelope,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Keep the complete [0, T] rollout reproducible from the recorded
        # safe_control gitlink: include both the initial state and the terminal
        # sample locally instead of depending on an uncommitted submodule edit.
        self.N = int(np.ceil(self.backup_horizon / self.dt)) + 1
        self.policy_name = policy_name
        self.policy_adapter = policy_adapter
        self.stop_config = stop_config
        self.terminal_envelope = terminal_envelope
        self.last_terminal_envelope_status: Dict[str, object] = {}
        self.set_backup_controller(policy_adapter, target=None)

    def _control_at_step(self, state: np.ndarray, step: int) -> np.ndarray:
        return np.asarray(self.policy_adapter.control_at_step(state, step), dtype=float).reshape(-1)

    def _step_control(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        try:
            return np.asarray(
                self.robot.step(state.reshape(-1, 1), control.reshape(-1, 1))
            ).reshape(-1)
        except Exception:
            return state + self.dt * (
                self._dynamics_f(state) + self._dynamics_g(state) @ control
            )

    def _step_policy(self, state: np.ndarray, policy_step: int) -> np.ndarray:
        return self._step_control(state, self._control_at_step(state, policy_step))

    def _stop_control(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(
            StoppingControllerJAX.compute(
                jnp.asarray(state), self.stop_config["params"]
            ),
            dtype=float,
        ).reshape(-1)

    def _stop_successor(self, state: np.ndarray) -> np.ndarray:
        """One sampled successor under the exact common stopping controller."""

        return self._step_control(state, self._stop_control(state))

    def _stopping_envelope_margins(
        self, state: np.ndarray, prefix: str
    ) -> Dict[str, float]:
        envelope = self.terminal_envelope
        state = np.asarray(state, dtype=float).reshape(-1)
        return {
            f"{prefix}_longitudinal_speed": (
                envelope.longitudinal_speed_abs_max - abs(float(state[5]))
            ),
            f"{prefix}_yaw_rate": envelope.yaw_rate_abs_max - abs(float(state[3])),
            f"{prefix}_sideslip": envelope.sideslip_abs_max - abs(float(state[4])),
            f"{prefix}_steering": envelope.steering_abs_max - abs(float(state[6])),
        }

    def _evaluate_terminal_envelope(
        self,
        state: np.ndarray,
        *,
        record: bool = False,
    ) -> Dict[str, object]:
        """Evaluate the named sampled stop-tail terminal proxy.

        Both the terminal sample and its one-step successor under the exact stop
        policy must remain safe, satisfy the stopping-envelope bounds, and have
        non-increasing absolute speed. This check is auditable, but it is not a
        formal control-invariance proof.
        """

        state = np.asarray(state, dtype=float).reshape(-1)
        successor = self._stop_successor(state)
        components: Dict[str, float] = {
            "inherited_terminal": float(super()._h_terminal(state)),
            "terminal_safety": float(
                self._h_safety(state, self.backup_horizon)
                - self.terminal_envelope.safety_margin_min
            ),
            "successor_safety": float(
                self._h_safety(successor, self.backup_horizon + self.dt)
                - self.terminal_envelope.safety_margin_min
            ),
        }
        components.update(self._stopping_envelope_margins(state, "terminal"))
        components.update(self._stopping_envelope_margins(successor, "successor"))
        components["successor_abs_speed_nonincrease"] = float(
            abs(state[5])
            + self.terminal_envelope.stop_successor_abs_speed_increase_max
            - abs(successor[5])
        )
        value = float(min(components.values()))
        status: Dict[str, object] = {
            "name": self.terminal_envelope.name,
            "is_formal_invariant_proof": False,
            "description": (
                "sampled stopping-envelope proxy: terminal and one exact-stop "
                "successor satisfy safety, settled lateral-state bounds, a bounded "
                "longitudinal speed, and non-increasing absolute speed"
            ),
            "value": value,
            "satisfied": bool(np.isfinite(value) and value >= 0.0),
            "tolerances": asdict(self.terminal_envelope),
            "components": components,
        }
        if record:
            self.last_terminal_envelope_status = copy.deepcopy(status)
        return status

    def _h_terminal(self, x):
        """Scalar margin for the sampled stop-tail terminal envelope."""

        return float(self._evaluate_terminal_envelope(x)["value"])

    def get_terminal_envelope_status(self) -> Dict[str, object]:
        return copy.deepcopy(self.last_terminal_envelope_status)

    def _integrate_policy_trajectory(self, x0: np.ndarray) -> np.ndarray:
        """Roll out states only, before any sensitivity/constraint construction."""

        phi = np.zeros((self.N, self.n_states))
        state = np.asarray(x0, dtype=float).reshape(-1).copy()
        phi[0] = state

        for step in range(1, self.N):
            policy_step = step - 1
            state = self._step_policy(state, policy_step)
            phi[step] = state
        return phi

    def _compute_rollout_sensitivities(self, phi: np.ndarray) -> np.ndarray:
        """Construct sensitivities only for a rollout already certified safe."""

        sensitivities = np.zeros((self.N, self.n_states, self.n_states))
        sensitivity = np.eye(self.n_states)
        sensitivities[0] = sensitivity
        eps = 1e-5
        for step in range(1, self.N):
            policy_step = step - 1
            state = np.asarray(phi[step - 1], dtype=float)
            next_state = np.asarray(phi[step], dtype=float)
            jacobian = np.zeros((self.n_states, self.n_states))
            for state_index in range(self.n_states):
                perturbed = state.copy()
                perturbed[state_index] += eps
                perturbed_next = self._step_policy(perturbed, policy_step)
                jacobian[:, state_index] = (perturbed_next - next_state) / eps

            sensitivity = jacobian @ sensitivity
            sensitivities[step] = sensitivity
        return sensitivities

    def _integrate_backup_trajectory(self, x0):
        """Compatibility wrapper matching the parent integration interface."""

        phi = self._integrate_policy_trajectory(x0)
        return phi, self._compute_rollout_sensitivities(phi)

    def _input_scale(self) -> np.ndarray:
        return np.array(
            [
                self.robot_spec.get("delta_dot_max", np.deg2rad(15.0)),
                self.robot_spec.get("tau_dot_max", 8000.0),
            ],
            dtype=float,
        )

    def solve_candidate(self, robot_state: np.ndarray, u_nom: np.ndarray) -> CandidateCBFResult:
        """Certify the complete backup rollout, then solve its candidate QP.

        Selection is over a sampled certified-candidate set, analogous to the
        active set in Chen et al.  An unsafe rollout or terminal state is
        therefore rejected before system matrices, CBF constraints, or a QP
        are constructed.  The outer wrapper records the loss and applies the
        common bounded damage-mitigation action.
        """

        started = time.perf_counter()
        qp_solved = False
        try:
            robot_state = np.asarray(robot_state, dtype=float).reshape(-1)
            u_scale = self._input_scale()
            u_min, u_max = -u_scale, u_scale
            phi = self._integrate_policy_trajectory(robot_state)
            h_values = [self._h_safety(phi[index], index * self.dt) for index in range(len(phi))]
            h_safety_min = float(np.min(h_values))
            terminal_status = self._evaluate_terminal_envelope(phi[-1], record=True)
            h_terminal = float(terminal_status["value"])
            self._last_h_min = min(h_safety_min, h_terminal)
            self.global_min_h = min(self.global_min_h, self._last_h_min)
            self.latest_backup_trajectory = phi.copy()
            self.curr_step += 1

            rollout_safe = bool(
                np.all(np.isfinite(h_values)) and h_safety_min >= 0.0
            )
            terminal_safe = bool(
                np.isfinite(h_terminal) and h_terminal >= 0.0
            )
            if not (rollout_safe and terminal_safe):
                return CandidateCBFResult(
                    policy_name=self.policy_name,
                    feasible=False,
                    u=None,
                    objective=float("inf"),
                    solver_status="uncertified_rollout",
                    rollout_safe=rollout_safe,
                    terminal_safe=terminal_safe,
                    solve_time_sec=time.perf_counter() - started,
                    qp_solved=False,
                    error="Complete backup rollout or terminal constraint is unsafe",
                )

            sensitivities = self._compute_rollout_sensitivities(phi)
            f0 = self._dynamics_f(robot_state)
            g0 = self._dynamics_g(robot_state)
            G_list: list[np.ndarray] = []
            h_list: list[float] = []

            for index in range(self.N):
                state_i = phi[index]
                sensitivity_i = sensitivities[index]
                time_i = index * self.dt
                h_value = self._h_safety(state_i, time_i)
                gradient = self._grad_h_safety(state_i, time_i)
                if self.moving_obstacles is not None:
                    h_next_time = self._h_safety(state_i, time_i + self.dt)
                    dh_dt = (h_next_time - h_value) / self.dt
                else:
                    dh_dt = 0.0
                if index < self.N - 1:
                    policy_drift = (phi[index + 1] - phi[index]) / self.dt
                else:
                    policy_drift = (phi[index] - phi[index - 1]) / self.dt
                lhs = gradient @ sensitivity_i @ g0
                rhs = (
                    -(gradient @ sensitivity_i @ f0)
                    + gradient @ policy_drift
                    - dh_dt
                    - self._alpha(h_value)
                )
                if np.linalg.norm(lhs) > 1e-6:
                    G_list.append(lhs)
                    h_list.append(float(rhs))
                elif rhs > 1e-8:
                    return CandidateCBFResult(
                        policy_name=self.policy_name,
                        feasible=False,
                        u=None,
                        objective=float("inf"),
                        solver_status="infeasible_constant_path_constraint",
                        rollout_safe=True,
                        terminal_safe=True,
                        solve_time_sec=time.perf_counter() - started,
                        qp_solved=False,
                        error="A zero-control-coefficient path row is violated",
                    )

            terminal_gradient = self._grad_h_terminal(phi[-1])
            terminal_lhs = terminal_gradient @ sensitivities[-1] @ g0
            terminal_rhs = -(
                terminal_gradient @ sensitivities[-1] @ f0
                + self._alpha_terminal(h_terminal)
            )
            if np.linalg.norm(terminal_lhs) > 1e-6:
                G_list.append(terminal_lhs)
                h_list.append(float(terminal_rhs))
            elif terminal_rhs > 1e-8:
                return CandidateCBFResult(
                    policy_name=self.policy_name,
                    feasible=False,
                    u=None,
                    objective=float("inf"),
                    solver_status="infeasible_constant_terminal_constraint",
                    rollout_safe=True,
                    terminal_safe=True,
                    solve_time_sec=time.perf_counter() - started,
                    qp_solved=False,
                    error="A zero-control-coefficient terminal row is violated",
                )

            clipped_nominal = np.clip(np.asarray(u_nom, dtype=float).reshape(-1), u_min, u_max)
            if not G_list:
                intervention = self.Q_u * (
                    clipped_nominal / u_scale
                    - np.asarray(u_nom, dtype=float).reshape(-1) / u_scale
                )
                return CandidateCBFResult(
                    policy_name=self.policy_name,
                    feasible=True,
                    u=clipped_nominal,
                    objective=float(np.sum(np.square(intervention))),
                    solver_status="no_constraints",
                    rollout_safe=True,
                    terminal_safe=True,
                    solve_time_sec=time.perf_counter() - started,
                    qp_solved=False,
                    error=None,
                )

            constraint_matrix = np.asarray(G_list, dtype=float)
            constraint_rhs = np.asarray(h_list, dtype=float)
            if np.any(~np.isfinite(constraint_matrix)) or np.any(~np.isfinite(constraint_rhs)):
                raise FloatingPointError("non-finite Backup-CBF constraint")

            scaled_control = cp.Variable(self.n_controls)
            nominal_scaled = clipped_nominal / u_scale
            objective_expression = cp.multiply(self.Q_u, scaled_control - nominal_scaled)
            objective = cp.Minimize(cp.sum_squares(objective_expression))
            constraints = [
                (constraint_matrix @ np.diag(u_scale)) @ scaled_control >= constraint_rhs,
                scaled_control >= -1.0,
                scaled_control <= 1.0,
            ]
            problem = cp.Problem(objective, constraints)
            qp_solved = True
            try:
                problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            except Exception:
                problem.solve(solver=cp.SCS, verbose=False)

            solver_status = str(problem.status)
            if solver_status not in self.accepted_statuses or scaled_control.value is None:
                return CandidateCBFResult(
                    policy_name=self.policy_name,
                    feasible=False,
                    u=None,
                    objective=float("inf"),
                    solver_status=solver_status,
                    rollout_safe=True,
                    terminal_safe=True,
                    solve_time_sec=time.perf_counter() - started,
                    qp_solved=True,
                    error=f"Backup-CBF QP failed: {solver_status}",
                )

            raw_control = np.asarray(
                np.diag(u_scale) @ scaled_control.value, dtype=float
            ).reshape(-1)
            control = project_bounded_control(raw_control, u_min, u_max)
            if control is None:
                return CandidateCBFResult(
                    policy_name=self.policy_name,
                    feasible=False,
                    u=None,
                    objective=float("inf"),
                    solver_status="invalid_solution",
                    rollout_safe=rollout_safe,
                    terminal_safe=terminal_safe,
                    solve_time_sec=time.perf_counter() - started,
                    qp_solved=True,
                    error="QP returned a non-finite or out-of-bounds input",
                )
            intervention = self.Q_u * (control / u_scale - nominal_scaled)
            return CandidateCBFResult(
                policy_name=self.policy_name,
                feasible=True,
                u=control,
                objective=float(np.sum(np.square(intervention))),
                solver_status=solver_status,
                rollout_safe=rollout_safe,
                terminal_safe=terminal_safe,
                solve_time_sec=time.perf_counter() - started,
                qp_solved=True,
                error=None,
            )
        except Exception as exc:
            return CandidateCBFResult(
                policy_name=self.policy_name,
                feasible=False,
                u=None,
                objective=float("inf"),
                solver_status="error",
                rollout_safe=None,
                terminal_safe=None,
                solve_time_sec=time.perf_counter() - started,
                qp_solved=qp_solved,
                error=str(exc),
            )


class MultiBackupCBFMinInterventionDrift:
    """Evaluate the shared Backup-CBF library and select minimum intervention."""

    algorithm_key = "multi_backup_cbf_mi"
    nominal_policy_representation = NOMINAL_POLICY_REPRESENTATION

    def __init__(
        self,
        robot,
        robot_spec: dict,
        dt: float = 0.05,
        backup_horizon: float = 3.0,
        ax=None,
        *,
        reference_plcbf: PLCBF,
        maneuver_prefix_time: float = 1.0,
        terminal_envelope: Optional[DriftSampledStoppingTerminalEnvelope] = None,
    ):
        self.robot = robot
        self.robot_spec = robot_spec
        self.dt = float(dt)
        self.backup_horizon = float(backup_horizon)
        self.left_lane_y = float(reference_plcbf.left_lane_y)
        self.right_lane_y = float(reference_plcbf.right_lane_y)
        self.policy_configs = copy.deepcopy(reference_plcbf.policy_configs)
        self.policy_names = tuple(reference_plcbf.policy_configs.keys()) + ("nominal",)
        self.total_rollout_steps = int(np.ceil(self.backup_horizon / self.dt))
        requested_prefix_steps = max(0, int(round(float(maneuver_prefix_time) / self.dt)))
        self.maneuver_prefix_steps = min(
            requested_prefix_steps,
            max(self.total_rollout_steps - 1, 0),
        )
        self.maneuver_prefix_time = self.maneuver_prefix_steps * self.dt
        self.terminal_tail_steps = self.total_rollout_steps - self.maneuver_prefix_steps
        self.terminal_tail_time = self.terminal_tail_steps * self.dt
        self.terminal_envelope = (
            terminal_envelope or DriftSampledStoppingTerminalEnvelope()
        )
        self.u_min = np.array(reference_plcbf.u_min, dtype=float, copy=True)
        self.u_max = np.array(reference_plcbf.u_max, dtype=float, copy=True)
        self.env = None
        self.current_friction = float(robot_spec.get("mu", 1.0))
        self.nominal_trajectory: Optional[np.ndarray] = None
        self.nominal_controls: Optional[np.ndarray] = None
        self.metrics = MultiPolicyMetrics()
        self.infeasible = False
        self.qp_infeasible = False
        self.certificate_lost = False
        self.runtime_error = False
        self.fallback_applied = False
        self.status = "optimal"
        self.best_policy_name: Optional[str] = None
        self.last_candidate_results: list[CandidateCBFResult] = []
        self._using_backup = False

        stop_config = self.policy_configs["stop"]
        self._nominal_adapter = _FrozenNominalPolicyAdapter(
            stop_config,
            self.maneuver_prefix_steps,
        )
        self.filters: Dict[str, _StrictCandidateBackupCBF] = {}
        for name in self.policy_names:
            adapter = (
                self._nominal_adapter
                if name == "nominal"
                else _CompoundLibraryPolicyAdapter(
                    self.policy_configs[name],
                    stop_config,
                    self.maneuver_prefix_steps,
                )
            )
            candidate = _StrictCandidateBackupCBF(
                robot=robot,
                robot_spec=robot_spec,
                dt=dt,
                backup_horizon=backup_horizon,
                ax=ax,
                policy_name=name,
                policy_adapter=adapter,
                stop_config=stop_config,
                terminal_envelope=self.terminal_envelope,
            )
            if name in ("lane_change_left", "lane_change_right"):
                candidate.backup_target = float(self.policy_configs[name]["params"].target_y)
            self.filters[name] = candidate

        self.assert_library_matches(reference_plcbf)

    def assert_library_matches(self, reference_plcbf: PLCBF) -> None:
        assert_runtime_library_equal(self, reference_plcbf)

    def set_environment(self, env) -> None:
        self.env = env

    def set_friction(self, friction: float) -> None:
        self.current_friction = float(friction)
        # One shared model update before any candidate is evaluated.
        if abs(float(self.robot.get_friction()) - self.current_friction) > 1e-12:
            self.robot.set_friction(self.current_friction)

    @staticmethod
    def _row_major(array: Optional[np.ndarray], control_dim: int) -> Optional[np.ndarray]:
        if array is None:
            return None
        value = np.array(array, dtype=float, copy=True)
        if value.ndim != 2:
            raise ValueError("Frozen nominal plan must be a two-dimensional array")
        if value.shape[1] == control_dim:
            return value
        if value.shape[0] == control_dim:
            return value.T
        # State trajectories use 8 rather than control_dim; preserve existing
        # PLCBF convention of transposing when rows are fewer than columns.
        return value.T if value.shape[0] < value.shape[1] else value

    def set_nominal_trajectory(
        self,
        trajectory: Optional[np.ndarray],
        controls: Optional[np.ndarray] = None,
    ) -> None:
        if trajectory is None:
            self.nominal_trajectory = None
            self.nominal_controls = None
            self._nominal_adapter.set_controls(None)
            return
        trajectory_value = np.array(trajectory, dtype=float, copy=True)
        if trajectory_value.ndim != 2:
            raise ValueError("Frozen nominal trajectory must be two-dimensional")
        if trajectory_value.shape[1] != 8 and trajectory_value.shape[0] == 8:
            trajectory_value = trajectory_value.T
        self.nominal_trajectory = trajectory_value
        self.nominal_controls = self._row_major(controls, 2)
        self._nominal_adapter.set_controls(self.nominal_controls)

    def _frozen_environment(self):
        if self.env is None:
            return None
        frozen = copy.copy(self.env)
        if hasattr(self.env, "obstacles"):
            frozen.obstacles = copy.deepcopy(self.env.obstacles)
        return frozen

    def _emergency_control(self, robot_state: np.ndarray) -> np.ndarray:
        params = self.policy_configs["stop"]["params"]
        control = np.asarray(
            StoppingControllerJAX.compute(jnp.asarray(robot_state), params),
            dtype=float,
        ).reshape(-1)
        return np.clip(control, self.u_min, self.u_max)

    def evaluate_candidates(
        self,
        robot_state: np.ndarray,
        u_nom: np.ndarray,
        policy_order: Optional[Sequence[str]] = None,
    ) -> list[CandidateCBFResult]:
        """Evaluate an immutable physical-step context (sequential execution)."""

        order = tuple(policy_order) if policy_order is not None else self.policy_names
        if set(order) != set(self.policy_names) or len(order) != len(self.policy_names):
            raise ValueError("Candidate order must contain every runtime policy exactly once")
        frozen_environment = self._frozen_environment()
        results: list[CandidateCBFResult] = []
        frozen_state = np.array(robot_state, dtype=float, copy=True).reshape(-1)
        frozen_nominal = np.array(u_nom, dtype=float, copy=True).reshape(-1)
        for name in order:
            candidate = self.filters[name]
            candidate.set_environment(frozen_environment)
            results.append(candidate.solve_candidate(frozen_state.copy(), frozen_nominal.copy()))
        return results

    def solve_control_problem(
        self,
        robot_state: np.ndarray,
        control_ref: Optional[dict] = None,
        friction: Optional[float] = None,
        nominal_trajectory: Optional[np.ndarray] = None,
        nominal_controls: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        robot_state = np.asarray(robot_state, dtype=float).reshape(-1)
        if friction is not None:
            self.set_friction(friction)
        if nominal_trajectory is not None:
            self.set_nominal_trajectory(nominal_trajectory, nominal_controls)
        u_nom = (
            np.asarray(control_ref["u_ref"], dtype=float).reshape(-1)
            if control_ref is not None and "u_ref" in control_ref
            else np.zeros(2, dtype=float)
        )

        self.infeasible = False
        self.qp_infeasible = False
        self.certificate_lost = False
        self.runtime_error = False
        self.fallback_applied = False
        self.last_candidate_results = self.evaluate_candidates(robot_state, u_nom)
        self.metrics.num_candidate_qps_solved += sum(
            result.qp_solved for result in self.last_candidate_results
        )
        feasible_count = sum(result.feasible for result in self.last_candidate_results)
        rollout_safe_count = sum(result.rollout_safe is True for result in self.last_candidate_results)
        certified_count = sum(
            result.rollout_safe is True and result.terminal_safe is True
            for result in self.last_candidate_results
        )
        terminal_failure_count = sum(result.terminal_safe is False for result in self.last_candidate_results)
        candidate_qp_failure_count = sum(
            result.qp_solved and not result.feasible
            for result in self.last_candidate_results
        )
        candidate_evaluation_error_count = sum(
            result.solver_status == "error"
            for result in self.last_candidate_results
        )
        self.metrics.feasible_candidates_per_step.append(feasible_count)
        self.metrics.certified_candidates_per_step.append(certified_count)
        self.metrics.rollout_safe_candidates_per_step.append(rollout_safe_count)
        self.metrics.qp_feasible_candidates_per_step.append(feasible_count)
        self.metrics.terminal_failure_count += terminal_failure_count
        self.metrics.candidate_qp_failure_count += candidate_qp_failure_count
        self.metrics.candidate_evaluation_error_count += (
            candidate_evaluation_error_count
        )
        best = select_minimum_intervention(self.last_candidate_results, self.policy_names)

        if best is None:
            self.best_policy_name = None
            if candidate_evaluation_error_count:
                self.status = "candidate_evaluation_error"
                self.runtime_error = True
            elif certified_count == 0:
                self.status = "certificate_lost_no_backup_candidate"
                self.certificate_lost = True
                self.metrics.certificate_loss_count += 1
            else:
                self.status = "qp_infeasible_no_backup_candidate"
                self.infeasible = True
                self.qp_infeasible = True
                self.metrics.qp_infeasible_count += 1
            self.fallback_applied = True
            self._using_backup = True
            self.metrics.fallback_step_count += 1
            if self.runtime_error:
                raise RuntimeError(
                    "MB-CBF-MI candidate evaluation failed; certificate status "
                    "is unknown"
                )
            self.metrics.num_steps_with_no_safe_policy += 1
            return self._emergency_control(robot_state).reshape(-1, 1)

        self.status = best.solver_status
        self.best_policy_name = best.policy_name
        normalized_delta = (np.asarray(best.u) - u_nom) / np.maximum(self.u_max, 1e-12)
        self._using_backup = bool(
            np.linalg.norm(self.filters[best.policy_name].Q_u * normalized_delta) > 0.1
        )
        self.metrics.record_selection(best.policy_name, np.linalg.norm(np.asarray(best.u) - u_nom))
        return np.asarray(best.u).reshape(-1, 1)

    def is_using_backup(self) -> bool:
        return bool(self._using_backup)

    def get_metrics(self) -> Dict[str, object]:
        return self.metrics.as_dict()

    def get_status(self) -> Dict[str, object]:
        return {
            "algorithm": self.algorithm_key,
            "status": self.status,
            "using_backup": self._using_backup,
            "best_policy": self.best_policy_name,
            "infeasible": self.infeasible,
            "qp_infeasible": self.qp_infeasible,
            "certificate_lost": self.certificate_lost,
            "runtime_error": self.runtime_error,
            "fallback_applied": self.fallback_applied,
            "compound_strategy": {
                "maneuver_prefix_steps": self.maneuver_prefix_steps,
                "maneuver_prefix_time": self.maneuver_prefix_time,
                "terminal_tail_steps": self.terminal_tail_steps,
                "terminal_tail_time": self.terminal_tail_time,
                "tail_policy": "stop",
                "hard_switch_to_exact_stop": True,
            },
            "terminal_envelope": {
                **asdict(self.terminal_envelope),
                "is_formal_invariant_proof": False,
                "description": (
                    "sampled stopping-envelope proxy with one exact-stop successor; "
                    "not a formal invariance proof"
                ),
            },
            "candidate_terminal_status": {
                name: candidate.get_terminal_envelope_status()
                for name, candidate in self.filters.items()
            },
            "num_candidate_qps_solved": sum(
                result.qp_solved for result in self.last_candidate_results
            ),
        }

    def get_multi_backup_trajectories(self):
        return {
            name: [candidate.latest_backup_trajectory.copy()]
            if hasattr(candidate, "latest_backup_trajectory")
            else []
            for name, candidate in self.filters.items()
        }

    def clear_trajectories(self) -> None:
        for candidate in self.filters.values():
            candidate.clear_trajectories()
