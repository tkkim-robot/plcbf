"""Hospital navigation case study with QP-selected room-refuge fallbacks."""

from .config import (
    DEFAULT_CONFIG,
    HospitalConfig,
    hospital_config_from_mapping,
    load_hospital_config,
)
from .controller import (
    ControllerResult,
    HospitalController,
    HocbfConstraint,
    PolicyCbfEvaluation,
    RoomPolicyProvider,
    dynamic_hocbf_constraints,
    sensed_obstacles,
)
from .environment import (
    HospitalEnvironment,
    Rect,
    Room,
    build_hospital_environment,
)
from .scenario_generation import (
    DEFAULT_HUMAN_COUNT,
    DEFAULT_ORDINARY_STRETCHER_COUNT,
    GeneratedHospitalCrowd,
    generate_hospital_crowd,
)
from .simulation import (
    HospitalSimulation,
    STRICT_REFUGE_CONVOY_SPEED,
    STRICT_REFUGE_INITIAL_STATE,
    STRICT_REFUGE_POSITIONS,
    STRICT_WEST_JUNCTION_X,
    SweptTransitionSafety,
    build_blocked_main_hall_scenario,
    evaluate_swept_transition,
    strict_refuge_scenario_metadata,
    validate_strict_refuge_protocol,
)

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_HUMAN_COUNT",
    "DEFAULT_ORDINARY_STRETCHER_COUNT",
    "GeneratedHospitalCrowd",
    "HospitalConfig",
    "HospitalController",
    "HospitalEnvironment",
    "HospitalSimulation",
    "STRICT_REFUGE_CONVOY_SPEED",
    "STRICT_REFUGE_INITIAL_STATE",
    "STRICT_REFUGE_POSITIONS",
    "STRICT_WEST_JUNCTION_X",
    "SweptTransitionSafety",
    "ControllerResult",
    "HocbfConstraint",
    "PolicyCbfEvaluation",
    "Rect",
    "RoomPolicyProvider",
    "Room",
    "build_blocked_main_hall_scenario",
    "build_hospital_environment",
    "dynamic_hocbf_constraints",
    "sensed_obstacles",
    "evaluate_swept_transition",
    "generate_hospital_crowd",
    "hospital_config_from_mapping",
    "load_hospital_config",
    "strict_refuge_scenario_metadata",
    "validate_strict_refuge_protocol",
]
