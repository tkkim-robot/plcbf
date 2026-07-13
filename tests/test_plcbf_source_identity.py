"""Protect the approved historical PL-CBF implementation and execution paths."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import numpy as np

from examples.drift_car import benchmark_additional_baselines
from examples.drift_car import benchmark_black_ice as historical_benchmark


ROOT = Path(__file__).resolve().parents[1]
APPROVED_COMMIT = "34795fae8ab04846e312cdb399872e9a6deda7b5"
APPROVED_SHA256 = {
    "examples/drift_car/algorithms/plcbf_drift.py": (
        "ec8d01df926b6e804b9651fb24b70482ed4a45b1d871a830a8ae1c32124159b3"
    ),
    "examples/warehouse/algorithms/plcbf_quad3d.py": (
        "7910469ff23b6cd4554ee777cecaf0a2143959caf46206ae519ec388e8f5fe8a"
    ),
    "examples/drift_car/benchmark_black_ice.py": (
        "25990d82ecb5ddb5cc1ef11380cd8303510534852ffe99043330778a26638ab9"
    ),
    "examples/warehouse/benchmark_warehouse_randomized_quad.py": (
        "712aa44835aae226cc69bdee33f1fe6c639903c0d317cc88131d4830a7ea72d7"
    ),
    "examples/drift_car/test_drift_pcbf.py": (
        "2367b2b5bb179367ef353d7b64c625c20e4c6e383f3d02ef93910babc7ae6bae"
    ),
    "examples/warehouse/test_warehouse_quad.py": (
        "49fdf20e334a4b33448773a687c234921b5d28842c3153bba51365cbc9a050c2"
    ),
    "examples/drift_car/algorithms/__init__.py": (
        "94f5c5b32b935c396901a8ff6a7cdc9246b0cfefa1eb769c024ac36ded264ec3"
    ),
    "examples/warehouse/algorithms/__init__.py": (
        "276723fbd755778db576b8af9707163ee0dd7c7aa945da0bdecf8c0caa1c9e69"
    ),
}


def test_historical_sources_match_approved_commit_byte_for_byte():
    for relative_path, expected_hash in APPROVED_SHA256.items():
        current = (ROOT / relative_path).read_bytes()
        approved = subprocess.check_output(
            ["git", "show", f"{APPROVED_COMMIT}:{relative_path}"],
            cwd=ROOT,
        )
        assert current == approved, relative_path
        assert hashlib.sha256(current).hexdigest() == expected_hash


def test_new_registry_query_cannot_change_historical_control_or_trajectory(
    monkeypatch,
):
    config = historical_benchmark.SimConfig(tf=0.15, dt=0.05)
    returned_control = np.array([0.25, -1000.0])
    traces = []

    class Environment:
        track_length = 300.0

        @staticmethod
        def get_friction_at_position(position, default_friction):
            del position
            return default_friction

    class Car:
        def __init__(self):
            self.state = np.zeros((8, 1))
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

    class Simulator:
        def __init__(self, car):
            self.car = car
            self.controls = []
            self.states = []

        def step(self, control):
            vector = np.asarray(control).reshape(-1).copy()
            self.controls.append(vector)
            self.car.state[0, 0] += vector[0]
            self.car.state[1, 0] += vector[1] / 10000.0
            self.states.append(self.car.state.copy())
            return {"collision": False}

    class Nominal:
        @staticmethod
        def solve_control_problem(state):
            del state
            return np.zeros((2, 1))

        @staticmethod
        def get_full_predictions():
            return None, None

    class Shield:
        @staticmethod
        def solve_control_problem(*args, **kwargs):
            del args, kwargs
            return returned_control.reshape(-1, 1)

    current = {}
    environment = Environment()
    monkeypatch.setattr(
        historical_benchmark,
        "setup_env_and_lanes",
        lambda unused: (environment, {"middle": 0.0}),
    )
    monkeypatch.setattr(
        historical_benchmark, "add_black_ice_and_obstacles", lambda *args: None
    )

    def make_car(*args, **kwargs):
        del args, kwargs
        current["car"] = Car()
        return current["car"]

    def make_simulator(car, *args, **kwargs):
        del args, kwargs
        current["simulator"] = Simulator(car)
        return current["simulator"]

    monkeypatch.setattr(historical_benchmark, "DriftingCar", make_car)
    monkeypatch.setattr(
        historical_benchmark, "DriftingCarSimulator", make_simulator
    )
    monkeypatch.setattr(
        historical_benchmark, "setup_mpcc", lambda *args: Nominal()
    )
    monkeypatch.setattr(
        historical_benchmark, "setup_shielding", lambda *args: Shield()
    )

    variant = next(
        item for item in historical_benchmark.make_variants() if item.key == "plcbf"
    )
    scenario = historical_benchmark.Scenario(0, 7, 1, ((80.0, "middle"),))

    def run_trace():
        historical_benchmark.run_episode(variant, scenario, config)
        simulator = current["simulator"]
        return (
            np.asarray(simulator.controls),
            np.asarray(simulator.states),
        )

    traces.append(run_trace())
    assert {
        item.key for item in benchmark_additional_baselines.make_variants()
    } == {"multi_backup_cbf_mi", "library_pcbf_mi"}
    traces.append(run_trace())

    assert np.array_equal(traces[0][0], traces[1][0])
    assert np.array_equal(traces[0][1], traces[1][1])
