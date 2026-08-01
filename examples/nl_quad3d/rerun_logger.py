"""Lazy Rerun visualization for the nonlinear-quadrotor case study.

Importing this module never imports ``rerun``. This keeps headless benchmark
workers independent of viewer availability; the dependency is resolved only
when :class:`NLQuad3DRerunLogger` is instantiated.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from .dynamics import NLQuad3D, rotation_matrix

if TYPE_CHECKING:
    from .controller import RolloutEvaluation


def _load_rerun() -> ModuleType:
    try:
        return importlib.import_module("rerun")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Rerun visualization requires the optional 'rerun-sdk' package. "
            "Install project dependencies or run the simulation without "
            "--visualize/--save-rrd."
        ) from exc


class NLQuad3DRerunLogger:
    """Log vehicle, spheres, and candidate rollouts to a viewer or ``.rrd``."""

    def __init__(
        self,
        model: NLQuad3D,
        *,
        application_id: str = "plcbf_nl_quad3d",
        spawn_viewer: bool = False,
        save_path: str | Path | None = None,
        rr_module: ModuleType | None = None,
    ):
        self.model = model
        self.rr = rr_module or _load_rerun()
        self.save_path = Path(save_path).expanduser().resolve() if save_path else None
        if self.save_path is not None:
            if self.save_path.suffix.lower() != ".rrd":
                raise ValueError("save_path must use the .rrd extension")
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.rr.init(application_id, spawn=False)
            if hasattr(self.rr, "set_sinks") and hasattr(self.rr, "FileSink"):
                self.rr.set_sinks(self.rr.FileSink(str(self.save_path)))
            elif hasattr(self.rr, "save"):
                self.rr.save(str(self.save_path))
            else:
                raise RuntimeError("installed rerun SDK cannot create an .rrd sink")
        else:
            self.rr.init(application_id, spawn=spawn_viewer)
        self._candidate_count = 0
        self._closed = False

    def _log(self, path: str, archetype: object, *, static: bool = False) -> None:
        self.rr.log(path, archetype, static=static)

    def log_world(
        self,
        *,
        goal: np.ndarray,
        bounds_lower: np.ndarray | None = None,
        bounds_upper: np.ndarray | None = None,
    ) -> None:
        self._log(
            "world/goal",
            self.rr.Points3D(
                [np.asarray(goal, dtype=float)],
                radii=[0.18],
                colors=[[80, 220, 120]],
                labels=["goal"],
            ),
            static=True,
        )
        if bounds_lower is not None and bounds_upper is not None:
            lower = np.asarray(bounds_lower, dtype=float)
            upper = np.asarray(bounds_upper, dtype=float)
            self._log(
                "world/bounds",
                self.rr.Boxes3D(
                    mins=[lower],
                    sizes=[upper - lower],
                    colors=[[90, 110, 140, 60]],
                    fill_mode="solid",
                ),
                static=True,
            )

    def _candidate_points(
        self,
        trajectory: np.ndarray,
    ) -> np.ndarray:
        points = []
        for state in trajectory:
            points.append(self.model.safety_point(state))
        return np.asarray(points)

    def log_step(
        self,
        *,
        time_seconds: float,
        state: np.ndarray,
        obstacles: np.ndarray,
        goal: np.ndarray,
        control: np.ndarray,
        evaluation: "RolloutEvaluation | None" = None,
    ) -> None:
        """Log a complete frame, including every PLCBF candidate trajectory."""

        self.rr.set_time("simulation", duration=float(time_seconds))
        x = np.asarray(state, dtype=float).reshape(12)
        position = x[:3]
        rotation = rotation_matrix(x[6], x[7], x[8])
        arm = self.model.config.arm_length
        x_arm = np.stack(
            [
                position + rotation @ np.array([-arm, 0.0, 0.0]),
                position + rotation @ np.array([arm, 0.0, 0.0]),
            ]
        )
        y_arm = np.stack(
            [
                position + rotation @ np.array([0.0, -arm, 0.0]),
                position + rotation @ np.array([0.0, arm, 0.0]),
            ]
        )
        self._log(
            "robot/arms",
            self.rr.LineStrips3D(
                [x_arm, y_arm],
                radii=[0.035],
                colors=[[70, 170, 255], [70, 170, 255]],
            ),
        )
        self._log(
            "robot/center",
            self.rr.Points3D(
                [position],
                radii=[self.model.config.robot_radius * 0.35],
                colors=[[50, 150, 255]],
            ),
        )
        safety_point = self.model.safety_point(x)
        self._log(
            "robot/safety_point",
            self.rr.Points3D(
                [safety_point],
                radii=[0.07],
                colors=[[255, 210, 70]],
            ),
        )
        velocity = x[3:6]
        self._log(
            "robot/velocity",
            self.rr.Arrows3D(
                origins=[position],
                vectors=[velocity],
                radii=[0.025],
                colors=[[100, 220, 255]],
            ),
        )
        thrust_fraction = np.asarray(control, dtype=float) / float(
            self.model.config.w_max
        )
        rotor_offsets = np.asarray(
            [
                [arm, 0.0, 0.0],
                [0.0, arm, 0.0],
                [-arm, 0.0, 0.0],
                [0.0, -arm, 0.0],
            ]
        )
        rotor_positions = position[None, :] + (rotation @ rotor_offsets.T).T
        self._log(
            "robot/rotors",
            self.rr.Points3D(
                rotor_positions,
                radii=0.055 + 0.035 * thrust_fraction,
                colors=[[120, 230, 255]] * 4,
            ),
        )

        obstacle_array = np.asarray(obstacles, dtype=float).reshape(-1, 7)
        if obstacle_array.shape[0]:
            self._log(
                "obstacles/spheres",
                self.rr.Ellipsoids3D(
                    centers=obstacle_array[:, :3],
                    radii=obstacle_array[:, 3],
                    colors=[[245, 95, 90, 170]] * obstacle_array.shape[0],
                    fill_mode="solid",
                ),
            )
            self._log(
                "obstacles/velocity",
                self.rr.Arrows3D(
                    origins=obstacle_array[:, :3],
                    vectors=obstacle_array[:, 4:7],
                    radii=[0.018],
                    colors=[[255, 170, 120]] * obstacle_array.shape[0],
                ),
            )
        else:
            self._log("obstacles", self.rr.Clear(recursive=True))

        self._log(
            "world/current_goal",
            self.rr.Points3D(
                [np.asarray(goal, dtype=float)],
                radii=[0.14],
                colors=[[80, 220, 120]],
            ),
        )
        if evaluation is None:
            self._log("rollouts", self.rr.Clear(recursive=True))
            self._candidate_count = 0
            return
        for index, (name, trajectory, value) in enumerate(
            zip(
                evaluation.names,
                evaluation.trajectories,
                evaluation.values,
                strict=True,
            )
        ):
            selected = index == evaluation.selected_index
            color = [45, 230, 120] if selected else [150, 165, 190, 105]
            radius = 0.035 if selected else 0.012
            line_options: dict[str, object] = {
                "radii": [radius],
                "colors": [color],
                # Rerun components are stateful across timeline frames.  An
                # empty label batch is therefore required to clear a label
                # left behind when this rollout was selected previously.
                "labels": (
                    [f"{name}: {float(value):.3f}"]
                    if selected
                    else []
                ),
            }
            # All candidate trajectories remain visible, but labeling every
            # rollout at their shared origin obscures both the vehicle and
            # obstacle geometry in screenshots. Keep the selected policy's
            # value as the single visual annotation.
            self._log(
                f"rollouts/{index:02d}_{name}",
                self.rr.LineStrips3D(
                    [self._candidate_points(trajectory)],
                    **line_options,
                ),
            )
        for index in range(len(evaluation.names), self._candidate_count):
            self._log(
                f"rollouts/{index:02d}",
                self.rr.Clear(recursive=True),
            )
        self._candidate_count = len(evaluation.names)

    def close(self) -> None:
        if not self._closed and hasattr(self.rr, "disconnect"):
            self.rr.disconnect()
        self._closed = True

    def __enter__(self) -> "NLQuad3DRerunLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
