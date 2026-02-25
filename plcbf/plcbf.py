"""
Created on February 9th, 2026
@author: Taekyung Kim

@description:
Base class for PLCBF (Multiple Policy Control Barrier Function) implementations.

PLCBF extends PCBF by evaluating multiple backup policies and selecting the best one.
The structure is: PLCBFBase extends PCBFBase, adding multi-policy logic.

Key differences from PCBF:
1. Maintains multiple policy configurations
2. Evaluates all policies to compute V(x) for each
3. Selects best policy based on safety margin and feasibility
4. Uses selected policy's constraint in QP

Subclasses should:
- Inherit from their corresponding PCBF implementation
- Override `_setup_policies()` to define scenario-specific policies
- Override `solve_control_problem()` to add multi-policy evaluation

Example:
    class PLCBF_Scenario(PCBF_Scenario):
        def _setup_policies(self):
            # Define multiple backup policies
            pass
        
        def solve_control_problem(self, state, control_ref=None):
            # Evaluate all policies, select best, solve QP
            pass
"""

from plcbf.pcbf import PCBFBase

# PLCBFBase is conceptually an extension of PCBFBase
# In practice, scenario-specific PLCBF classes inherit from their PCBF class
# Example: PLCBF_DI(PCBF_DI) where PCBF_DI(PCBFBase)

__all__ = ['PCBFBase']  # Re-export for PLCBF implementations
