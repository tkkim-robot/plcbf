# mpcbf package
"""
Multi-Policy CBF (mpcbf) - Safety-critical control for autonomous vehicles.

This package provides:
- Drifting environment with racing track simulation
- Car dynamics based on kinematic bicycle model
- Integration with safe_control library
- Policy Control Barrier Function (PCBF) implementation in JAX
"""

__version__ = "0.1.0"

from mpcbf.pcbf import (
    PCBF, 
    DriftingCarDynamicsJAX,
    LaneChangeControllerJAX,
    StoppingControllerJAX,
)

__all__ = [
    'PCBF', 
    'DriftingCarDynamicsJAX',
    'LaneChangeControllerJAX',
    'StoppingControllerJAX',
]

