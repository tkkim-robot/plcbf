"""Seeded scenario parity with the approved historical benchmarks."""

from __future__ import annotations

import hashlib
import json

from examples.drift_car import benchmark_additional_baselines as drift
from examples.drift_car import benchmark_black_ice as historical_drift
from examples.warehouse import benchmark_additional_baselines_quad as warehouse
from examples.warehouse import (
    benchmark_warehouse_randomized_quad as historical_warehouse,
)


def _digest(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_drift_seeded_geometry_matches_historical_benchmark():
    scenarios = drift.generate_scenarios(50, 7)
    historical = historical_drift.generate_scenarios(50, 7)
    geometry = [
        {
            "run_idx": item.run_idx,
            "num_obstacles": item.num_obstacles,
            "obstacles": item.obstacles,
        }
        for item in scenarios
    ]
    historical_geometry = [
        {
            "run_idx": item.run_idx,
            "num_obstacles": item.num_obstacles,
            "obstacles": item.obstacles,
        }
        for item in historical
    ]
    assert geometry == historical_geometry
    assert _digest(geometry) == (
        "b1462f165fd0dcfa811d8a334d3d8c1106c83d063e66e92b62a318e20b617e15"
    )


def _warehouse_scenarios(module):
    return module.generate_random_scenarios(
        level=7,
        num_trials=100,
        seed=11,
        num_dynamic_obstacles=45,
        ghost_radius=2.4,
        speed_min=3.0,
        speed_max=4.5,
        start_exclusion_max_x=18.0,
        start_exclusion_max_y=18.0,
        start_clearance_radius=8.0,
        inter_ghost_clearance=0.2,
    )


def test_warehouse_seeded_geometry_matches_historical_benchmark():
    scenarios = _warehouse_scenarios(warehouse)
    historical = _warehouse_scenarios(historical_warehouse)
    geometry = [
        {"run_idx": item.run_idx, "ghosts": item.ghosts} for item in scenarios
    ]
    historical_geometry = [
        {"run_idx": item.run_idx, "ghosts": item.ghosts} for item in historical
    ]
    assert geometry == historical_geometry
    assert _digest(geometry) == (
        "125b631d315b3c9debdc141caa22a1d5105757bb252255568f8cec845e656604"
    )
