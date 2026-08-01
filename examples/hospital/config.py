"""Configuration for the hospital navigation case study.

The defaults intentionally mirror the browser playground where possible.  The
Python example keeps the configuration in dataclasses so benchmark scripts can
replace individual groups without mutating module globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class RobotConfig:
    radius: float = 0.55
    sensing_range: float = 14.0
    v_max: float = 2.85
    a_max: float = 1.7
    k_position: float = 0.95
    k_velocity: float = 2.1


@dataclass(frozen=True)
class PlannerConfig:
    resolution: float = 1.5
    clearance_buffer: float = 0.18
    preferred_clearance: float = 3.6
    clearance_weight: float = 4.5


@dataclass(frozen=True)
class PolicyConfig:
    rollout_dt: float = 0.24
    nominal_horizon: float = 3.0
    angle_horizon: float = 3.0
    reverse_horizon: float = 12.0
    stop_horizon: float = 12.0
    room_horizon: float = 7.2
    room_rollout_dt: float = 0.36
    num_angle_policies: int = 12
    angle_arc: float = 3.141592653589793
    angle_preview_distance: float = 5.2
    angle_target_speed: float = 1.75
    num_reverse_policies: int = 1
    reverse_policy_arc: float = 1.5707963267948966
    reverse_preview_distance: float = 3.2
    reverse_target_speed: float = 1.1
    body_reverse_policy: bool = True
    body_reverse_min_angle: float = 0.25
    nominal_target_speed: float = 2.85
    room_target_speed: float = 2.85
    stop_gain: float = 2.7
    room_policy_count: int = 7
    room_search_radius: float = 42.0
    component_temperature: float = 36.0
    time_temperature: float = 30.0
    cbf_alpha: float = 0.85
    cbf_value_buffer: float = 0.45
    gradient_steps: tuple[float, float, float, float] = (
        0.06,
        0.06,
        0.025,
        0.04,
    )
    time_derivative_step: float = 0.12
    max_gradient_norm: float = 70.0
    safe_value_threshold: float = 0.0
    constraint_tolerance: float = 1e-4


@dataclass(frozen=True)
class SafetyConfig:
    safety_margin: float = 0.45
    human_margin: float = 0.0
    stretcher_margin: float = 0.55
    static_margin: float = 0.14
    max_obstacles: int = 12
    enable_hocbf: bool = True
    hocbf_lambda1: float = 0.35
    hocbf_lambda2: float = 0.8
    hocbf_margin: float = 0.2
    hocbf_activation_margin: float = 3.2
    hocbf_wide_stretcher_width: float = 4.0
    hocbf_wide_stretcher_activation_margin: float = 1.0
    stretcher_proxy_count: int = 5
    enable_static_hocbf: bool = True
    static_hocbf_margin: float = 0.14
    static_hocbf_lambda1: float = 1.6
    static_hocbf_lambda2: float = 2.2
    static_hocbf_activation_margin: float = 1.4
    max_static_hocbf_constraints: int = 8
    occlusion_clearance: float = 0.1
    occlusion_floor_step: float = 0.55


@dataclass(frozen=True)
class RefugeConfig:
    """Geometry of the room backup terminal set.

    These are geometric/value-function conditions, not controller modes,
    timers, or transition guards.
    """

    inside_door_offset: float = 2.3
    outside_door_offset: float = 1.75
    terminal_interior_margin: float = 2.1
    terminal_speed_max: float = 2.2
    waypoint_radius: float = 0.75

    def __post_init__(self) -> None:
        positive = {
            "inside_door_offset": self.inside_door_offset,
            "outside_door_offset": self.outside_door_offset,
            "terminal_interior_margin": self.terminal_interior_margin,
            "terminal_speed_max": self.terminal_speed_max,
            "waypoint_radius": self.waypoint_radius,
        }
        for name, value in positive.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class HospitalConfig:
    width: float = 140.0
    height: float = 95.0
    dt: float = 0.06
    robot: RobotConfig = field(default_factory=RobotConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    policies: PolicyConfig = field(default_factory=PolicyConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    refuge: RefugeConfig = field(default_factory=RefugeConfig)


DEFAULT_CONFIG = HospitalConfig()


def _replace_group(
    group: Any,
    values: Mapping[str, Any],
    *,
    name: str,
) -> Any:
    allowed = {item.name for item in fields(group)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            f"unknown {name} configuration fields: "
            + ", ".join(sorted(unknown))
        )
    return replace(group, **dict(values))


def hospital_config_from_mapping(
    values: Mapping[str, Any],
    *,
    base: HospitalConfig = DEFAULT_CONFIG,
) -> HospitalConfig:
    """Build a configuration from an exact or partial JSON-compatible mapping."""

    allowed = {item.name for item in fields(HospitalConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            "unknown hospital configuration fields: "
            + ", ".join(sorted(unknown))
        )
    updates: dict[str, Any] = {}
    for name in ("width", "height", "dt"):
        if name in values:
            updates[name] = values[name]
    for name in ("robot", "planner", "policies", "safety", "refuge"):
        if name not in values:
            continue
        nested = values[name]
        if not isinstance(nested, Mapping):
            raise TypeError(f"hospital configuration {name!r} must be an object")
        updates[name] = _replace_group(
            getattr(base, name),
            nested,
            name=name,
        )
    return replace(base, **updates)


def load_hospital_config(
    path: str | Path,
    *,
    base: HospitalConfig = DEFAULT_CONFIG,
) -> HospitalConfig:
    """Load a direct configuration or a tuning summary's ``best_config``."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("hospital configuration JSON must contain an object")
    values: Any = payload.get("best_config", payload)
    if not isinstance(values, Mapping):
        raise TypeError("hospital best_config must contain an object")
    return hospital_config_from_mapping(values, base=base)


__all__ = [
    "DEFAULT_CONFIG",
    "HospitalConfig",
    "PlannerConfig",
    "PolicyConfig",
    "RefugeConfig",
    "RobotConfig",
    "SafetyConfig",
    "hospital_config_from_mapping",
    "load_hospital_config",
]
