"""Nonlinear 12-state quadrotor PLCBF case study.

The package is intentionally named ``nl_quad3d`` so the pre-existing linear
warehouse ``Quad3D`` example remains unambiguous.
"""

from .controller import (
    NLQuad3DControllerConfig,
    PLCBF_NLQuad3D,
    PolicyCandidate,
    RolloutEvaluation,
)
from .config_io import (
    DEFAULT_CONTROLLER_CONFIG_PATH,
    controller_config_from_mapping,
    load_controller_config_artifact,
    load_default_controller_config,
    write_controller_config_artifact,
)
from .dynamics import NLQuad3D, NLQuad3DConfig, make_state, rotate_to_input
from .scenarios import (
    NLQuad3DScenario,
    PLAYGROUND_CROWDED_SCENARIO,
    PLAYGROUND_STRESS_PROTOCOL_VERSION,
    PLAYGROUND_STRESS_SCENARIO,
    get_scenario,
    make_playground_crowded_scenario,
    make_playground_stress_scenario,
    scenario_names,
)

__all__ = [
    "NLQuad3D",
    "NLQuad3DConfig",
    "NLQuad3DControllerConfig",
    "NLQuad3DScenario",
    "DEFAULT_CONTROLLER_CONFIG_PATH",
    "PLAYGROUND_CROWDED_SCENARIO",
    "PLAYGROUND_STRESS_PROTOCOL_VERSION",
    "PLAYGROUND_STRESS_SCENARIO",
    "PLCBF_NLQuad3D",
    "PolicyCandidate",
    "RolloutEvaluation",
    "controller_config_from_mapping",
    "get_scenario",
    "load_controller_config_artifact",
    "load_default_controller_config",
    "make_state",
    "make_playground_crowded_scenario",
    "make_playground_stress_scenario",
    "rotate_to_input",
    "scenario_names",
    "write_controller_config_artifact",
]
