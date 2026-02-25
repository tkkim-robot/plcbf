"""PLCBF library for safe control using Control Barrier Functions."""

__version__ = "0.1.0"

# Base classes only - scenario-specific implementations are in examples/
from plcbf.pcbf import PCBFBase

__all__ = ['PCBFBase']
