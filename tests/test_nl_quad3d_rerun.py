from __future__ import annotations

from dataclasses import replace
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from examples.nl_quad3d.controller import RolloutEvaluation
from examples.nl_quad3d.dynamics import NLQuad3D, make_state


def test_importing_logger_module_does_not_import_rerun() -> None:
    sys.modules.pop("rerun", None)
    module = importlib.import_module("examples.nl_quad3d.rerun_logger")
    importlib.reload(module)
    assert "rerun" not in sys.modules


def test_missing_rerun_fails_only_when_logger_is_constructed(monkeypatch) -> None:
    module = importlib.import_module("examples.nl_quad3d.rerun_logger")
    real_import = module.importlib.import_module

    def import_without_rerun(name: str):
        if name == "rerun":
            raise ModuleNotFoundError("test")
        return real_import(name)

    monkeypatch.setattr(module.importlib, "import_module", import_without_rerun)
    with pytest.raises(RuntimeError, match="rerun-sdk"):
        module.NLQuad3DRerunLogger(NLQuad3D())


class _FakeRerun:
    def __init__(self) -> None:
        self.init_calls = []
        self.sinks = []
        self.logs = []
        self.times = []
        self.disconnected = False

    class FileSink:
        def __init__(self, path):
            self.path = path

    def init(self, application_id, *, spawn):
        self.init_calls.append((application_id, spawn))

    def set_sinks(self, *sinks):
        self.sinks.extend(sinks)

    def set_time(self, timeline, *, duration):
        self.times.append((timeline, duration))

    def log(self, path, archetype, *, static=False):
        self.logs.append((path, archetype, static))

    def disconnect(self):
        self.disconnected = True

    @staticmethod
    def _archetype(name, *args, **kwargs):
        return SimpleNamespace(name=name, args=args, kwargs=kwargs)

    def Points3D(self, *args, **kwargs):
        return self._archetype("Points3D", *args, **kwargs)

    def LineStrips3D(self, *args, **kwargs):
        return self._archetype("LineStrips3D", *args, **kwargs)

    def Arrows3D(self, *args, **kwargs):
        return self._archetype("Arrows3D", *args, **kwargs)

    def Ellipsoids3D(self, *args, **kwargs):
        return self._archetype("Ellipsoids3D", *args, **kwargs)

    def Boxes3D(self, *args, **kwargs):
        return self._archetype("Boxes3D", *args, **kwargs)

    def Clear(self, *args, **kwargs):
        return self._archetype("Clear", *args, **kwargs)


def test_headless_logger_uses_file_sink_and_logs_candidate_trajectories(
    tmp_path,
) -> None:
    module = importlib.import_module("examples.nl_quad3d.rerun_logger")
    fake = _FakeRerun()
    model = NLQuad3D()
    path = tmp_path / "case.rrd"
    logger = module.NLQuad3DRerunLogger(
        model,
        save_path=path,
        rr_module=fake,
    )
    state = make_state([0.0, 0.0, 1.0])
    trajectories = np.stack(
        [
            np.stack([state, state + np.array([0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]),
            np.stack([state, state + np.array([0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]),
        ]
    )
    evaluation = RolloutEvaluation(
        names=("radial_0", "nominal"),
        values=np.array([0.5, 0.4]),
        state_gradients=np.zeros((2, 12)),
        obstacle_gradients=np.zeros((2, 1, 7)),
        trajectories=trajectories,
        time_derivatives=np.zeros(2),
        control_directions=np.zeros((2, 4)),
        constraint_rhs=np.zeros(2),
        scores=np.ones(2),
        selected_index=0,
    )
    logger.log_world(
        goal=np.array([2.0, 0.0, 1.0]),
        bounds_lower=np.zeros(3),
        bounds_upper=np.ones(3) * 4.0,
    )
    logger.log_step(
        time_seconds=0.0,
        state=state,
        obstacles=np.array([[1.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0]]),
        goal=np.array([2.0, 0.0, 1.0]),
        control=model.hover_input,
        evaluation=evaluation,
    )
    assert fake.init_calls == [("plcbf_nl_quad3d", False)]
    assert len(fake.sinks) == 1
    assert fake.sinks[0].path == str(path.resolve())
    paths = [entry[0] for entry in fake.logs]
    assert "rollouts/00_radial_0" in paths
    assert "rollouts/01_nominal" in paths
    rollouts = {
        path: archetype
        for path, archetype, _ in fake.logs
        if path.startswith("rollouts/")
    }
    assert rollouts["rollouts/00_radial_0"].kwargs["labels"] == [
        "radial_0: 0.500"
    ]
    assert rollouts["rollouts/01_nominal"].kwargs["labels"] == []

    # Rerun retains component values across temporal frames.  When selection
    # changes, the old selected entity must receive an explicit empty label
    # batch or both labels remain visible in later screenshots.
    logger.log_step(
        time_seconds=0.05,
        state=state,
        obstacles=np.array([[1.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0]]),
        goal=np.array([2.0, 0.0, 1.0]),
        control=model.hover_input,
        evaluation=replace(evaluation, selected_index=1),
    )
    latest_rollouts = {
        path: archetype
        for path, archetype, _ in fake.logs
        if path.startswith("rollouts/")
    }
    assert latest_rollouts["rollouts/00_radial_0"].kwargs["labels"] == []
    assert latest_rollouts["rollouts/01_nominal"].kwargs["labels"] == [
        "nominal: 0.400"
    ]
    logger.close()
    assert not any("paraboloid" in path.lower() for path in paths)
    assert fake.disconnected
