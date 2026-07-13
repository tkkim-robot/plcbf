"""Unrecoverable-failure checks for the drift baseline-only benchmark."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from examples.drift_car import benchmark_additional_baselines as benchmark


class _Environment:
    track_length = 300.0

    @staticmethod
    def get_friction_at_position(position, default_friction):
        del position
        return default_friction


class _Car:
    def __init__(self):
        self.state = np.zeros((8, 1), dtype=float)
        self.state[5, 0] = 10.0
        self.friction = 1.0

    def get_state(self):
        return self.state.copy()

    def get_position(self):
        return self.state[:2, 0].copy()

    def get_friction(self):
        return self.friction

    def set_friction(self, value):
        self.friction = float(value)


class _Simulator:
    def __init__(self, car):
        self.car = car
        self.controls = []

    def step(self, control):
        self.controls.append(np.asarray(control).reshape(-1).copy())
        self.car.state[0, 0] += 0.1
        return {"collision": False}


class _Nominal:
    @staticmethod
    def solve_control_problem(state):
        del state
        return np.zeros((2, 1), dtype=float)

    @staticmethod
    def get_full_predictions():
        return None, None


class _Shield:
    def __init__(self, returned):
        self.returned = returned
        self.u_min = np.array([-1.0, -8000.0])
        self.u_max = np.array([1.0, 8000.0])

    def solve_control_problem(self, *args, **kwargs):
        del args, kwargs
        if isinstance(self.returned, BaseException):
            raise self.returned
        return self.returned

    @staticmethod
    def get_status():
        return {
            "certificate_lost": False,
            "qp_infeasible": False,
            "runtime_error": False,
            "fallback_applied": False,
        }

    @staticmethod
    def get_metrics():
        return {}


def _run(monkeypatch, returned):
    config = replace(benchmark.SimConfig(), tf=0.1, dt=0.05)
    env = _Environment()
    car = _Car()
    simulator = _Simulator(car)
    shield = _Shield(returned)
    monkeypatch.setattr(
        benchmark, "setup_env_and_lanes", lambda unused: (env, {"middle": 0.0})
    )
    monkeypatch.setattr(
        benchmark, "add_black_ice_and_obstacles", lambda *args: None
    )
    monkeypatch.setattr(benchmark, "DriftingCar", lambda *args, **kwargs: car)
    monkeypatch.setattr(
        benchmark,
        "DriftingCarSimulator",
        lambda *args, **kwargs: simulator,
    )
    monkeypatch.setattr(benchmark, "setup_mpcc", lambda *args: _Nominal())
    monkeypatch.setattr(benchmark, "setup_shielding", lambda *args: shield)
    scenario = benchmark.Scenario(0, 7, 1, ((80.0, "middle"),))
    result = benchmark.run_episode(benchmark.make_variants()[0], scenario, config)
    return result, simulator


@pytest.mark.parametrize(
    "returned",
    [
        RuntimeError("solve failed"),
        None,
        np.zeros(3),
        np.array([np.inf, 0.0]),
        np.array([1.0 + 5e-6, 0.0]),
        np.array([1.1, 0.0]),
    ],
)
def test_invalid_or_failed_control_is_unrecoverable(monkeypatch, returned):
    result, simulator = _run(monkeypatch, returned)

    assert simulator.controls == []
    assert result.runtime_error is True
    assert result.infeasible is True
    assert result.unrecoverable_infeasible is True
    assert result.historical_failure is True
    assert result.total_steps == 0
