"""Faithful baseline algorithms for the hospital case study.

The pointwise PCBF family consumes the same per-policy certificates as
Hospital PL-CBF.  Backup-CBF uses path-wise rollout and terminal constraints.
MPS and Gatekeeper own committed trajectories and use one fixed retrace backup.
No algorithm in this module receives a blockage flag or refuge phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
import time
from typing import Sequence

import numpy as np

from plcbf.backup_cbf import (
    BackupCbfDecision,
    evaluate_backup_cbf_candidate,
    solve_fixed_backup_cbf,
    solve_multi_backup_cbf_min_intervention,
)
from plcbf.big_m_mpc import (
    BigMTrajectoryMPCConfig,
    BigMTrajectoryMPCProblem,
    BigMTrajectoryMPCResult,
    solve_big_m_trajectory_mpc,
)
from plcbf.baselines import (
    BaselineDecision,
    BenchmarkMethod,
    solve_library_pcbf_mi,
    solve_plcbf,
    solve_policy_pcbf,
)
from plcbf.policy_library import PolicyCertificate
from plcbf.dynamics_linearization import linearize_discrete_trajectory
from plcbf.trajectory_shielding import (
    GatekeeperShield,
    ModelPredictiveShield,
    ShieldDecision,
)

from .config import HospitalConfig
from .controller import HospitalController
from .dynamics import step_double_integrator
from .environment import HospitalEnvironment
from .obstacles import DynamicObstacle, Human, Stretcher, obstacle_clearance
from .policies import HospitalPolicy


@dataclass(frozen=True)
class _PredictedObstacleGeometry:
    """Compact geometry for one decision-relative prediction time."""

    human_centers: np.ndarray
    human_radii: np.ndarray
    stretcher_centers: np.ndarray
    stretcher_cosines: np.ndarray
    stretcher_sines: np.ndarray
    stretcher_half_lengths: np.ndarray
    stretcher_half_widths: np.ndarray
    other_obstacles: tuple[DynamicObstacle, ...]


@dataclass(frozen=True)
class HospitalBaselineConfig:
    """Algorithm constants matching the warehouse comparison protocol."""

    backup_horizon_s: float = 4.0
    multi_backup_maneuver_s: float = 2.0
    gatekeeper_nominal_steps: int = 30
    gatekeeper_discount_steps: int = 1
    backup_cbf_alpha: float = 2.0
    backup_cbf_terminal_alpha: float = 2.0
    terminal_speed_mps: float = 0.25
    time_derivative_step_s: float = 0.06
    trajectory_swept_substeps: int = 4
    fixed_backup_policy_id: str = "retrace_waypoint"
    fixed_backup_target_speed_mps: float = 2.8
    mi_position_tube_m: float = 3.0
    mi_control_tube_mps2: float = 6.0
    mi_control_tube_steps: int = 2
    mi_directional_policy_count: int = 32
    mi_directional_target_speed_mps: float = 2.85
    mi_big_m_position: float = 400.0
    mi_big_m_control: float = 60.0
    mi_big_m_safety: float = 50.0
    mi_tracking_weight: float = 8.0
    mi_terminal_weight: float = 16.0
    mi_velocity_weight: float = 0.15
    mi_control_weight: float = 0.02
    mi_nominal_weight: float = 0.5
    mi_time_limit_s: float = 1.0
    mi_mip_rel_gap: float = 0.05

    def __post_init__(self) -> None:
        if self.backup_horizon_s <= 0.0:
            raise ValueError("backup_horizon_s must be positive")
        if not 0.0 <= self.multi_backup_maneuver_s < self.backup_horizon_s:
            raise ValueError(
                "multi_backup_maneuver_s must be shorter than the backup "
                "horizon"
            )
        if self.gatekeeper_nominal_steps < 0:
            raise ValueError("gatekeeper_nominal_steps must be nonnegative")
        if self.gatekeeper_discount_steps <= 0:
            raise ValueError("gatekeeper_discount_steps must be positive")
        if self.trajectory_swept_substeps < 1:
            raise ValueError("trajectory_swept_substeps must be positive")
        if self.fixed_backup_target_speed_mps <= 0.0:
            raise ValueError(
                "fixed_backup_target_speed_mps must be positive"
            )
        if self.mi_position_tube_m < 0.0:
            raise ValueError("mi_position_tube_m must be nonnegative")
        if self.mi_control_tube_mps2 < 0.0:
            raise ValueError("mi_control_tube_mps2 must be nonnegative")
        if self.mi_control_tube_steps < 0:
            raise ValueError("mi_control_tube_steps must be nonnegative")
        if self.mi_directional_policy_count <= 0:
            raise ValueError("mi_directional_policy_count must be positive")
        if self.mi_directional_target_speed_mps <= 0.0:
            raise ValueError(
                "mi_directional_target_speed_mps must be positive"
            )
        if self.mi_big_m_position <= 0.0:
            raise ValueError("mi_big_m_position must be positive")
        if self.mi_big_m_control <= 0.0:
            raise ValueError("mi_big_m_control must be positive")
        if self.mi_big_m_safety <= 0.0:
            raise ValueError("mi_big_m_safety must be positive")
        for name, value in (
            ("mi_tracking_weight", self.mi_tracking_weight),
            ("mi_terminal_weight", self.mi_terminal_weight),
            ("mi_velocity_weight", self.mi_velocity_weight),
            ("mi_control_weight", self.mi_control_weight),
            ("mi_nominal_weight", self.mi_nominal_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")


class HospitalBaselineSuite:
    """Stateful per-trial implementation of all hospital baselines."""

    def __init__(
        self,
        controller: HospitalController,
        environment: HospitalEnvironment,
        config: HospitalConfig,
        *,
        algorithm_config: HospitalBaselineConfig = HospitalBaselineConfig(),
    ) -> None:
        self.controller = controller
        self.environment = environment
        self.config = config
        self.algorithm_config = algorithm_config
        self._target = controller.navigation_path[controller.navigation_index].copy()
        self._obstacles: tuple[DynamicObstacle, ...] = ()
        self._predicted_obstacles: dict[
            tuple[int, float], DynamicObstacle
        ] = {}
        self._human_prediction_checkpoints: dict[int, list[Human]] = {}
        self._predicted_geometries: dict[
            float, _PredictedObstacleGeometry
        ] = {}
        self._batch_human_indices: tuple[int, ...] = ()
        self._batch_human_positions: list[np.ndarray] = []
        self._batch_human_velocities: list[np.ndarray] = []
        self._batch_human_radii = np.empty(0, dtype=float)
        self._backup_steps = max(
            1,
            int(algorithm_config.backup_horizon_s / config.dt),
        )
        self.last_mi_mpc_result: BigMTrajectoryMPCResult | None = None
        common = dict(
            step=self._plant_step,
            nominal_control=self._nominal_feedback,
            backup_control=self._fixed_backup_feedback,
            trajectory_is_safe=self._trajectory_is_safe,
            backup_horizon_steps=self._backup_steps,
            backup_policy_id=algorithm_config.fixed_backup_policy_id,
        )
        self.mps = ModelPredictiveShield(**common)
        self.gatekeeper = GatekeeperShield(
            **common,
            nominal_horizon_steps=algorithm_config.gatekeeper_nominal_steps,
            horizon_discount_steps=algorithm_config.gatekeeper_discount_steps,
        )

    def _plant_step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return step_double_integrator(
            state, control, self.config.dt, self.config.robot
        )

    def _nominal_feedback(self, state: np.ndarray) -> np.ndarray:
        from .dynamics import waypoint_control

        return waypoint_control(
            state,
            self._target,
            self.config.robot,
            self.config.policies.nominal_target_speed,
        )

    def _fixed_backup_feedback(self, state: np.ndarray) -> np.ndarray:
        return self.fixed_backup_policy(state).control(
            state,
            self.config,
        )

    def _terminal_stop_feedback(self, state: np.ndarray) -> np.ndarray:
        limit = self.config.robot.a_max
        return np.clip(
            -self.config.policies.stop_gain * np.asarray(state)[2:4],
            -limit,
            limit,
        )

    def fixed_backup_policy(
        self,
        state: Sequence[float] | None = None,
    ) -> HospitalPolicy:
        """Return the warehouse-style fixed retrace-waypoint backup.

        This policy is baseline-only and is deliberately absent from the
        playground PL-CBF library. Its target index is derived solely from the
        nominal route's monotone progress index; no obstacle, room, or refuge
        observation can change it.
        """

        del state
        path = self.controller.navigation_path
        active_index = max(
            0,
            min(len(path) - 1, self.controller.navigation_index - 1),
        )
        retrace_waypoints = [
            np.asarray(path[index], dtype=float).copy()
            for index in range(active_index, -1, -1)
        ]
        return HospitalPolicy(
            name=self.algorithm_config.fixed_backup_policy_id,
            kind="retrace",
            horizon=self.algorithm_config.backup_horizon_s,
            rollout_dt=self.config.dt,
            target_speed=min(
                self.algorithm_config.fixed_backup_target_speed_mps,
                self.config.robot.v_max,
            ),
            waypoints=retrace_waypoints,
        )

    def mi_mpc_policies(
        self,
        state: Sequence[float],
    ) -> tuple[HospitalPolicy, ...]:
        """Build MI-MPC's warehouse-style directional-only branch set."""

        count = self.algorithm_config.mi_directional_policy_count
        return tuple(
            HospitalPolicy(
                name=f"mi_angle_{index:02d}",
                kind="angle",
                horizon=self.algorithm_config.backup_horizon_s,
                rollout_dt=self.config.dt,
                target_speed=min(
                    self.algorithm_config.mi_directional_target_speed_mps,
                    self.config.robot.v_max,
                ),
                angle=float(angle),
            )
            for index, angle in enumerate(
                np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
            )
        )

    def _point_margin(
        self,
        point: np.ndarray,
        elapsed: float,
    ) -> float:
        return float(
            self._point_margins(
                np.asarray(point, dtype=float).reshape(1, 2),
                np.asarray([elapsed], dtype=float),
            )[0]
        )

    def _point_margins(
        self,
        points: np.ndarray,
        elapsed: np.ndarray,
    ) -> np.ndarray:
        """Evaluate aligned point/time margins with identical geometry."""

        point_array = np.asarray(points, dtype=float)
        if point_array.ndim == 1:
            point_array = point_array.reshape(1, 2)
        if point_array.ndim != 2 or point_array.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        elapsed_array = np.broadcast_to(
            np.asarray(elapsed, dtype=float),
            (point_array.shape[0],),
        )

        # Preserve test/instrumentation hooks which replace the scalar method
        # on an individual suite instance.
        scalar_override = self.__dict__.get("_point_margin")
        if scalar_override is not None:
            return np.asarray(
                [
                    scalar_override(point, time_value)
                    for point, time_value in zip(
                        point_array, elapsed_array, strict=True
                    )
                ],
                dtype=float,
            )

        margins = self.environment.static_clearances(
            point_array,
            self.config.robot.radius + self.config.safety.static_margin,
        )
        if not self._obstacles:
            return margins

        geometries = [
            self._predicted_geometry(time_value)
            for time_value in elapsed_array
        ]
        first = geometries[0]
        robot_radius = (
            self.config.robot.radius
            + self.config.safety.safety_margin
        )

        if first.human_centers.shape[0]:
            human_centers = np.stack(
                [geometry.human_centers for geometry in geometries],
                axis=0,
            )
            human_clearance = (
                np.linalg.norm(
                    point_array[:, None, :] - human_centers,
                    axis=2,
                )
                - first.human_radii[None, :]
                - robot_radius
                - self.config.safety.human_margin
            )
            margins = np.minimum(
                margins,
                np.min(human_clearance, axis=1),
            )

        if first.stretcher_centers.shape[0]:
            stretcher_centers = np.stack(
                [
                    geometry.stretcher_centers
                    for geometry in geometries
                ],
                axis=0,
            )
            delta = point_array[:, None, :] - stretcher_centers
            local_x = (
                first.stretcher_cosines[None, :] * delta[:, :, 0]
                + first.stretcher_sines[None, :] * delta[:, :, 1]
            )
            local_y = (
                -first.stretcher_sines[None, :] * delta[:, :, 0]
                + first.stretcher_cosines[None, :] * delta[:, :, 1]
            )
            qx = (
                np.abs(local_x)
                - first.stretcher_half_lengths[None, :]
            )
            qy = (
                np.abs(local_y)
                - first.stretcher_half_widths[None, :]
            )
            stretcher_clearance = (
                np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
                + np.minimum(np.maximum(qx, qy), 0.0)
                - robot_radius
                - self.config.safety.stretcher_margin
            )
            margins = np.minimum(
                margins,
                np.min(stretcher_clearance, axis=1),
            )

        if first.other_obstacles:
            for index, (point, geometry) in enumerate(
                zip(point_array, geometries, strict=True)
            ):
                for obstacle in geometry.other_obstacles:
                    margins[index] = min(
                        margins[index],
                        obstacle_clearance(
                            obstacle,
                            point,
                            robot_radius,
                            self.config.safety.human_margin,
                            self.config.safety.stretcher_margin,
                        ),
                    )
        # Vectorized transcendental kernels can differ from their scalar
        # counterparts by an ulp.  Resolve the only values where that could
        # alter a safety sign with the original scalar geometry.
        for index in np.flatnonzero(np.abs(margins) <= 1e-9):
            margins[index] = self._scalar_point_margin(
                point_array[index],
                elapsed_array[index],
            )
        return margins

    def _scalar_point_margin(
        self,
        point: np.ndarray,
        elapsed: float,
    ) -> float:
        """Reference implementation used for near-boundary sign fidelity."""

        values = [
            self.environment.static_clearance(
                point,
                self.config.robot.radius
                + self.config.safety.static_margin,
            )
        ]
        values.extend(
            obstacle_clearance(
                obstacle.predicted(elapsed, self.environment),
                point,
                robot_radius=(
                    self.config.robot.radius
                    + self.config.safety.safety_margin
                ),
                human_margin=self.config.safety.human_margin,
                stretcher_margin=(
                    self.config.safety.stretcher_margin
                ),
            )
            for obstacle in self._obstacles
        )
        return float(min(values))

    def _reset_prediction_cache(self) -> None:
        """Discard predictions whenever the live obstacle snapshot changes."""

        self._predicted_obstacles.clear()
        self._human_prediction_checkpoints.clear()
        self._predicted_geometries.clear()
        humans = [
            (index, obstacle)
            for index, obstacle in enumerate(self._obstacles)
            if isinstance(obstacle, Human)
        ]
        self._batch_human_indices = tuple(index for index, _ in humans)
        self._batch_human_positions = [
            np.asarray(
                [[obstacle.x, obstacle.y] for _, obstacle in humans],
                dtype=float,
            ).reshape(-1, 2)
        ]
        self._batch_human_velocities = [
            np.asarray(
                [[obstacle.vx, obstacle.vy] for _, obstacle in humans],
                dtype=float,
            ).reshape(-1, 2)
        ]
        self._batch_human_radii = np.asarray(
            [obstacle.radius for _, obstacle in humans],
            dtype=float,
        )

    def _advance_human_batch(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        elapsed: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply ``Human.advance`` exactly to an aligned human batch."""

        if positions.shape[0] == 0 or elapsed <= 0.0:
            return positions.copy(), velocities.copy()
        dt = float(elapsed)
        next_positions = positions + velocities * dt
        full_collision = self.environment.collisions(
            next_positions,
            self._batch_human_radii,
        )
        result_positions = positions.copy()
        result_velocities = velocities.copy()
        free_full = ~full_collision
        result_positions[free_full] = next_positions[free_full]

        collided = np.flatnonzero(full_collision)
        if collided.size:
            current = positions[collided]
            velocity = velocities[collided]
            next_x = np.column_stack(
                (
                    current[:, 0] + velocity[:, 0] * dt,
                    current[:, 1],
                )
            )
            next_y = np.column_stack(
                (
                    current[:, 0],
                    current[:, 1] + velocity[:, 1] * dt,
                )
            )
            radii = self._batch_human_radii[collided]
            free_x = ~self.environment.collisions(next_x, radii)
            free_y = ~self.environment.collisions(next_y, radii)
            neither = ~(free_x | free_y)

            result_positions[collided, 0] = np.where(
                free_x,
                next_x[:, 0],
                current[:, 0],
            )
            result_positions[collided, 1] = np.where(
                free_y,
                next_y[:, 1],
                current[:, 1],
            )
            result_velocities[collided, 0] = np.where(
                free_y | neither,
                -velocity[:, 0],
                velocity[:, 0],
            )
            result_velocities[collided, 1] = np.where(
                free_x | neither,
                -velocity[:, 1],
                velocity[:, 1],
            )
        return result_positions, result_velocities

    def _predicted_geometry(
        self,
        elapsed: float,
    ) -> _PredictedObstacleGeometry:
        """Return exact obstacle geometry without per-obstacle Python queries."""

        time_value = max(0.0, float(elapsed))
        cached = self._predicted_geometries.get(time_value)
        if cached is not None:
            return cached

        checkpoint_count = int(np.floor(time_value / 0.05))
        while len(self._batch_human_positions) <= checkpoint_count:
            positions, velocities = self._advance_human_batch(
                self._batch_human_positions[-1],
                self._batch_human_velocities[-1],
                0.05,
            )
            self._batch_human_positions.append(positions)
            self._batch_human_velocities.append(velocities)
        checkpoint_time = checkpoint_count * 0.05
        remainder = time_value - checkpoint_time
        # Human.predicted stops when its remaining duration is <= 1e-10.
        if remainder <= 1e-10:
            remainder = 0.0
        human_centers, _ = self._advance_human_batch(
            self._batch_human_positions[checkpoint_count],
            self._batch_human_velocities[checkpoint_count],
            remainder,
        )

        stretcher_centers = []
        stretcher_cosines = []
        stretcher_sines = []
        stretcher_half_lengths = []
        stretcher_half_widths = []
        other_obstacles: list[DynamicObstacle] = []
        for obstacle_index, obstacle in enumerate(self._obstacles):
            if isinstance(obstacle, Human):
                continue
            predicted = self._predicted_obstacle(
                obstacle_index,
                time_value,
            )
            if isinstance(predicted, Stretcher):
                stretcher_centers.append(predicted.center)
                stretcher_cosines.append(cos(predicted.theta))
                stretcher_sines.append(sin(predicted.theta))
                stretcher_half_lengths.append(0.5 * predicted.length)
                stretcher_half_widths.append(0.5 * predicted.width)
            else:
                other_obstacles.append(predicted)

        geometry = _PredictedObstacleGeometry(
            human_centers=human_centers,
            human_radii=self._batch_human_radii,
            stretcher_centers=np.asarray(
                stretcher_centers, dtype=float
            ).reshape(-1, 2),
            stretcher_cosines=np.asarray(stretcher_cosines, dtype=float),
            stretcher_sines=np.asarray(stretcher_sines, dtype=float),
            stretcher_half_lengths=np.asarray(
                stretcher_half_lengths, dtype=float
            ),
            stretcher_half_widths=np.asarray(
                stretcher_half_widths, dtype=float
            ),
            other_obstacles=tuple(other_obstacles),
        )
        self._predicted_geometries[time_value] = geometry
        return geometry

    def _predicted_obstacle(
        self,
        obstacle_index: int,
        elapsed: float,
    ) -> DynamicObstacle:
        """Return an exact, decision-local obstacle prediction.

        ``Human.predicted`` integrates in 0.05-second chunks from the current
        obstacle snapshot.  Repeating that integration from zero at every
        trajectory sample makes a horizon scan quadratic in its length.  The
        checkpoints below are those exact same 0.05-second states; only the
        final sub-0.05-second remainder is replayed for each requested time.
        Stretchers retain their existing analytic reflected-motion predictor.
        """

        time_value = max(0.0, float(elapsed))
        key = (int(obstacle_index), time_value)
        cached = self._predicted_obstacles.get(key)
        if cached is not None:
            return cached

        obstacle = self._obstacles[obstacle_index]
        if isinstance(obstacle, Human):
            checkpoints = self._human_prediction_checkpoints.setdefault(
                obstacle_index,
                [obstacle.predicted(0.0, self.environment)],
            )
            checkpoint_count = int(np.floor(time_value / 0.05))
            while len(checkpoints) <= checkpoint_count:
                checkpoints.append(
                    checkpoints[-1].predicted(0.05, self.environment)
                )
            checkpoint_time = checkpoint_count * 0.05
            remainder = time_value - checkpoint_time
            if remainder < 1e-10:
                remainder = 0.0
            predicted = checkpoints[checkpoint_count].predicted(
                remainder,
                self.environment,
            )
        else:
            predicted = obstacle.predicted(time_value, self.environment)

        self._predicted_obstacles[key] = predicted
        return predicted

    def _trajectory_is_safe(self, states: np.ndarray) -> bool:
        trajectory = np.asarray(states, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] != 4:
            return False
        if trajectory.shape[0] < 2:
            return True
        substeps = self.algorithm_config.trajectory_swept_substeps
        starts = trajectory[:-1, :2]
        ends = trajectory[1:, :2]
        if not np.all(
            self.environment.segments_are_free(
                starts,
                ends,
                self.config.robot.radius
                + self.config.safety.static_margin,
            )
        ):
            return False

        alphas = np.arange(substeps + 1, dtype=float) / substeps
        points = (
            starts[:, None, :]
            + alphas[None, :, None]
            * (ends - starts)[:, None, :]
        ).reshape(-1, 2)
        elapsed = (
            np.arange(starts.shape[0], dtype=float)[:, None]
            + alphas[None, :]
        ).reshape(-1) * self.config.dt
        return bool(np.all(self._point_margins(points, elapsed) >= 0.0))

    def _policy_rollout_margins(
        self,
        policy: HospitalPolicy,
        *,
        maneuver_steps: int,
        multi_backup: bool,
    ):
        def oracle(
            initial_state: np.ndarray,
            time_offset: float,
        ) -> tuple[np.ndarray, float]:
            trajectory = self._backup_rollout_states(
                policy,
                initial_state,
                maneuver_steps=maneuver_steps,
                multi_backup=multi_backup,
            )
            state_count = len(trajectory)
            state = trajectory[-1]
            path_values = self._point_margins(
                trajectory[:, :2],
                time_offset
                + np.arange(state_count, dtype=float) * self.config.dt,
            )

            terminal_values = [
                self._point_margin(
                    state[:2],
                    time_offset
                    + self.algorithm_config.backup_horizon_s,
                ),
                self.algorithm_config.terminal_speed_mps
                - float(np.linalg.norm(state[2:4])),
            ]
            if multi_backup:
                successor = self._plant_step(
                    state, self._terminal_stop_feedback(state)
                )
                terminal_values.extend(
                    (
                        self._point_margin(
                            successor[:2],
                            time_offset
                            + self.algorithm_config.backup_horizon_s
                            + self.config.dt,
                        ),
                        self.algorithm_config.terminal_speed_mps
                        - float(np.linalg.norm(successor[2:4])),
                    )
                )
            if policy.kind == "room" and policy.target_room is not None:
                terminal_values.append(
                    policy.target_room.interior_margin(state[:2])
                    - self.config.refuge.terminal_interior_margin
                )
                if multi_backup:
                    terminal_values.append(
                        policy.target_room.interior_margin(successor[:2])
                        - self.config.refuge.terminal_interior_margin
                    )
            return (
                np.asarray(path_values),
                float(min(terminal_values)),
            )

        return oracle

    def _backup_rollout_states(
        self,
        policy: HospitalPolicy,
        initial_state: np.ndarray,
        *,
        maneuver_steps: int,
        multi_backup: bool,
    ) -> np.ndarray:
        """Roll out the source algorithm's single/strict-MB conventions."""

        # The warehouse baselines intentionally use two conventions.
        # Single BackupCBF has N=int(T/dt) states and constrains phi[1:N].
        # Strict MB-CBF-MI has ceil(T/dt)+1 states and constrains phi[0:N].
        state_count = (
            int(
                np.ceil(
                    self.algorithm_config.backup_horizon_s
                    / self.config.dt
                )
            )
            + 1
            if multi_backup
            else self._backup_steps
        )
        state = np.asarray(initial_state, dtype=float).copy()
        states = [state.copy()]
        for step_index in range(1, state_count):
            control = (
                policy.control(state, self.config)
                if step_index <= maneuver_steps
                else self._terminal_stop_feedback(state)
            )
            state = self._plant_step(state, control)
            states.append(state.copy())
        return np.asarray(states)

    def _backup_path_flow_derivatives(
        self,
        policy: HospitalPolicy,
        state: np.ndarray,
        *,
        maneuver_steps: int,
        multi_backup: bool,
    ) -> np.ndarray:
        """Compute warehouse ``grad_h(phi_i) @ f_policy_i`` row terms."""

        trajectory = self._backup_rollout_states(
            policy,
            state,
            maneuver_steps=maneuver_steps,
            multi_backup=multi_backup,
        )
        epsilon = 1e-5
        elapsed = (
            np.arange(len(trajectory), dtype=float) * self.config.dt
        )
        base = self._point_margins(trajectory[:, :2], elapsed)
        gradients = np.zeros((len(trajectory), 4), dtype=float)
        for axis in (0, 1):
            perturbed = trajectory[:, :2].copy()
            perturbed[:, axis] += epsilon
            gradients[:, axis] = (
                self._point_margins(perturbed, elapsed) - base
            ) / epsilon
        policy_flow = np.empty_like(trajectory)
        policy_flow[:-1] = (
            trajectory[1:] - trajectory[:-1]
        ) / self.config.dt
        policy_flow[-1] = (
            trajectory[-1] - trajectory[-2]
        ) / self.config.dt
        return np.einsum("ij,ij->i", gradients, policy_flow)

    def _backup_candidate(
        self,
        policy: HospitalPolicy,
        state: np.ndarray,
        nominal: np.ndarray,
        *,
        maneuver_steps: int,
        multi_backup: bool = False,
    ):
        drift = np.array([state[2], state[3], 0.0, 0.0])
        control_matrix = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        initial_backup_control = (
            policy.control(state, self.config)
            if maneuver_steps > 0
            else self._terminal_stop_feedback(state)
        )
        return evaluate_backup_cbf_candidate(
            policy_id=policy.name,
            state=state,
            nominal_control=nominal,
            lower=-np.full(2, self.config.robot.a_max),
            upper=np.full(2, self.config.robot.a_max),
            drift=drift,
            control_matrix=control_matrix,
            backup_closed_loop_drift=(
                drift + control_matrix @ initial_backup_control
            ),
            rollout_margins=self._policy_rollout_margins(
                policy,
                maneuver_steps=maneuver_steps,
                multi_backup=multi_backup,
            ),
            gradient_steps=np.asarray(
                self.config.policies.gradient_steps,
                dtype=float,
            ),
            time_derivative_step=max(
                self.algorithm_config.time_derivative_step_s,
                self.config.dt,
            ),
            alpha=self.algorithm_config.backup_cbf_alpha,
            terminal_alpha=(
                self.algorithm_config.backup_cbf_terminal_alpha
            ),
            formulation=("strict_multi" if multi_backup else "single"),
            path_constraint_start_index=(0 if multi_backup else 1),
            path_flow_derivatives=self._backup_path_flow_derivatives(
                policy,
                state,
                maneuver_steps=maneuver_steps,
                multi_backup=multi_backup,
            ),
        )

    def _branch_rollout(
        self,
        state: np.ndarray,
        policy: HospitalPolicy,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        current = np.asarray(state, dtype=float).reshape(4).copy()
        states = [current.copy()]
        controls = []
        for step_index in range(self._backup_steps):
            control = np.asarray(
                policy.control(current, self.config), dtype=float
            ).reshape(2)
            current = self._plant_step(current, control)
            controls.append(control.copy())
            states.append(current.copy())
        state_array = np.asarray(states)
        minimum_margin = float(
            np.min(
                self._point_margins(
                    state_array[:, :2],
                    np.arange(len(state_array), dtype=float)
                    * self.config.dt,
                )
            )
        )
        return (
            state_array,
            np.asarray(controls),
            float(minimum_margin),
        )

    def _nominal_reference(
        self, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        current = np.asarray(state, dtype=float).reshape(4).copy()
        states = [current.copy()]
        controls = []
        for _ in range(self._backup_steps):
            control = self._nominal_feedback(current)
            current = self._plant_step(current, control)
            controls.append(control.copy())
            states.append(current.copy())
        return np.asarray(states), np.asarray(controls)

    def build_mi_mpc_problem(
        self,
        state: np.ndarray,
        nominal: np.ndarray,
        policies: Sequence[HospitalPolicy],
    ) -> tuple[BigMTrajectoryMPCProblem, int]:
        """Build the full continuous x/u MPC and directional tube branches."""

        policy_tuple = tuple(policies)
        if not policy_tuple:
            raise ValueError("Hospital MI-MPC requires policy branches")
        branch_data = [
            self._branch_rollout(state, policy)
            for policy in policy_tuple
        ]
        branch_states = np.asarray([item[0] for item in branch_data])
        branch_controls = np.asarray([item[1] for item in branch_data])
        branch_safety = np.asarray([item[2] for item in branch_data])

        nominal_states, nominal_controls = self._nominal_reference(state)
        matrices_a, matrices_b, affine = linearize_discrete_trajectory(
            self._plant_step,
            nominal_states,
            nominal_controls,
            state_steps=np.asarray(
                self.config.policies.gradient_steps, dtype=float
            ),
            control_steps=np.full(2, 1e-4),
        )
        state_lower = np.array(
            [
                self.config.robot.radius,
                self.config.robot.radius,
                -self.config.robot.v_max,
                -self.config.robot.v_max,
            ]
        )
        state_upper = np.array(
            [
                self.config.width - self.config.robot.radius,
                self.config.height - self.config.robot.radius,
                self.config.robot.v_max,
                self.config.robot.v_max,
            ]
        )
        fallback_index = int(np.argmax(branch_safety))
        fallback = policy_tuple[fallback_index].control(
            state,
            self.config,
        )
        return (
            BigMTrajectoryMPCProblem(
                x0=state,
                A=matrices_a,
                B=matrices_b,
                c=affine,
                branch_states=branch_states,
                branch_controls=branch_controls,
                branch_safety=branch_safety,
                state_lower=state_lower,
                state_upper=state_upper,
                control_lower=-np.full(2, self.config.robot.a_max),
                control_upper=np.full(2, self.config.robot.a_max),
                position_indices=(0, 1),
                velocity_indices=(2, 3),
                tracking_target=self._target,
                terminal_target=self.controller.goal,
                nominal_control=nominal,
                fallback_control=fallback,
            ),
            fallback_index,
        )

    def _mi_mpc_config(self) -> BigMTrajectoryMPCConfig:
        """Return the fixed warehouse publication MI-MPC constants."""

        return BigMTrajectoryMPCConfig(
            safety_threshold=self.config.policies.safe_value_threshold,
            position_tube=self.algorithm_config.mi_position_tube_m,
            early_control_tube=self.algorithm_config.mi_control_tube_mps2,
            early_control_steps=self.algorithm_config.mi_control_tube_steps,
            big_m_position=self.algorithm_config.mi_big_m_position,
            big_m_control=self.algorithm_config.mi_big_m_control,
            big_m_safety=self.algorithm_config.mi_big_m_safety,
            tracking_weight=self.algorithm_config.mi_tracking_weight,
            terminal_weight=self.algorithm_config.mi_terminal_weight,
            velocity_weight=self.algorithm_config.mi_velocity_weight,
            control_weight=self.algorithm_config.mi_control_weight,
            nominal_weight=self.algorithm_config.mi_nominal_weight,
            time_limit_s=self.algorithm_config.mi_time_limit_s,
            mip_rel_gap=self.algorithm_config.mi_mip_rel_gap,
        )

    def _solve_mi_mpc(
        self,
        state: np.ndarray,
        nominal: np.ndarray,
        policies: Sequence[HospitalPolicy],
    ) -> BaselineDecision:
        started_at = time.perf_counter()
        policy_tuple = tuple(policies)
        problem, fallback_index = self.build_mi_mpc_problem(
            state,
            nominal,
            policy_tuple,
        )
        fallback = np.asarray(problem.fallback_control, dtype=float)
        result = solve_big_m_trajectory_mpc(
            problem,
            self._mi_mpc_config(),
        )
        self.last_mi_mpc_result = result
        control = (
            fallback if result.control is None else np.asarray(result.control)
        )
        selected_policy = (
            policy_tuple[fallback_index].name
            if result.selected_branch is None
            else policy_tuple[result.selected_branch].name
        )
        objective = (
            float("inf")
            if result.objective is None
            else float(result.objective)
        )
        return BaselineDecision(
            method=BenchmarkMethod.MI_MPC.value,
            control=control,
            policy_id=selected_policy,
            feasible=result.feasible,
            status=result.status,
            used_fallback=result.used_fallback,
            objective=objective,
            solve_time_s=time.perf_counter() - started_at,
        )

    @staticmethod
    def _shield_baseline_decision(
        method: BenchmarkMethod,
        decision: ShieldDecision,
        nominal: np.ndarray,
    ) -> BaselineDecision:
        delta = decision.control - nominal
        return BaselineDecision(
            method=method.value,
            control=decision.control,
            policy_id=decision.committed_trajectory.backup_policy_id,
            feasible=decision.feasible,
            status=decision.status,
            used_fallback=decision.used_committed_backup,
            objective=float(delta @ delta),
            solve_time_s=decision.solve_time_s,
        )

    @staticmethod
    def _backup_baseline_decision(
        method: BenchmarkMethod,
        decision: BackupCbfDecision,
    ) -> BaselineDecision:
        return BaselineDecision(
            method=method.value,
            control=decision.control,
            policy_id=decision.policy_id,
            feasible=decision.feasible,
            status=decision.status,
            used_fallback=decision.used_fallback,
            objective=decision.objective,
            solve_time_s=decision.solve_time_s,
        )

    def solve(
        self,
        method: BenchmarkMethod | str,
        state: np.ndarray,
        obstacles: Sequence[DynamicObstacle],
        nominal: np.ndarray,
        certificates: Sequence[PolicyCertificate] = (),
        policies: Sequence[HospitalPolicy] | None = None,
        time_seconds: float = 0.0,
    ) -> BaselineDecision:
        """Solve one method without any scenario-triggered switching."""

        parsed = (
            method if isinstance(method, BenchmarkMethod) else BenchmarkMethod(method)
        )
        started_at = time.perf_counter()
        value = np.asarray(state, dtype=float).reshape(4)
        nominal_array = np.asarray(nominal, dtype=float).reshape(2)
        self._obstacles = tuple(obstacles)
        self._reset_prediction_cache()
        self._target = self.controller.navigation_path[
            self.controller.navigation_index
        ].copy()
        limit = self.config.robot.a_max
        lower = -np.full(2, limit)
        upper = np.full(2, limit)

        if parsed is BenchmarkMethod.POLICY_PCBF:
            return solve_policy_pcbf(
                certificates,
                nominal_array,
                lower,
                upper,
                backup_policy_id=self.algorithm_config.fixed_backup_policy_id,
                safe_value_threshold=self.config.policies.safe_value_threshold,
            )
        if parsed is BenchmarkMethod.PLCBF:
            result = self.controller.compute(
                value, self._obstacles, float(time_seconds)
            )
            delta = result.control - nominal_array
            used_fallback = result.decision.diagnostics.used_fallback
            return BaselineDecision(
                method=parsed.value,
                control=result.control,
                policy_id=result.selected_policy,
                feasible=bool(result.feasible and not used_fallback),
                status=(
                    "fallback:"
                    + str(
                        result.decision.diagnostics.fallback_reason
                    )
                    if used_fallback
                    else "optimal"
                ),
                used_fallback=used_fallback,
                objective=float(delta @ delta),
                solve_time_s=time.perf_counter() - started_at,
                policy_decision=result.decision,
            )
        if parsed is BenchmarkMethod.LIBRARY_PCBF_MI:
            return solve_library_pcbf_mi(
                certificates,
                nominal_array,
                lower,
                upper,
                safe_value_threshold=self.config.policies.safe_value_threshold,
                emergency_policy_id="stop",
                emergency_control=self._terminal_stop_feedback(value),
            )
        if parsed is BenchmarkMethod.MPS:
            return self._shield_baseline_decision(
                parsed, self.mps.solve(value), nominal_array
            )
        if parsed is BenchmarkMethod.GATEKEEPER:
            return self._shield_baseline_decision(
                parsed, self.gatekeeper.solve(value), nominal_array
            )

        if parsed is BenchmarkMethod.MI_MPC:
            policy_tuple = tuple(
                self.mi_mpc_policies(value)
                if policies is None
                else policies
            )
        else:
            policy_tuple = tuple(
                self.controller.candidate_policies(value)
                if policies is None
                else policies
            )
        if parsed is BenchmarkMethod.BACKUP_CBF:
            policy = self.fixed_backup_policy(value)
            candidate = self._backup_candidate(
                policy,
                value,
                nominal_array,
                maneuver_steps=self._backup_steps,
            )
            decision = solve_fixed_backup_cbf(
                candidate,
                direct_backup_control=policy.control(value, self.config),
                lower=lower,
                upper=upper,
            )
            return self._backup_baseline_decision(parsed, decision)
        if parsed is BenchmarkMethod.MULTI_BACKUP_CBF_MI:
            maneuver_steps = min(
                self._backup_steps,
                int(
                    round(
                        self.algorithm_config.multi_backup_maneuver_s
                        / self.config.dt
                    )
                ),
            )
            candidates = tuple(
                self._backup_candidate(
                    policy,
                    value,
                    nominal_array,
                    maneuver_steps=maneuver_steps,
                    multi_backup=True,
                )
                for policy in policy_tuple
            )
            decision = solve_multi_backup_cbf_min_intervention(
                candidates,
                direct_backup_controls={
                    policy.name: policy.control(value, self.config)
                    for policy in policy_tuple
                },
                lower=lower,
                upper=upper,
                emergency_policy_id="stop",
            )
            return self._backup_baseline_decision(parsed, decision)
        if parsed is BenchmarkMethod.MI_MPC:
            return self._solve_mi_mpc(
                value, nominal_array, policy_tuple
            )
        raise AssertionError(f"unhandled hospital method {parsed}")


__all__ = [
    "HospitalBaselineConfig",
    "HospitalBaselineSuite",
]
