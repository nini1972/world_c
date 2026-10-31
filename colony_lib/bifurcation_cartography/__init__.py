"""
colony_lib.bifurcation_cartography: Numerical continuation, critical exponent extraction,
and adaptive phase diagram scanning.
Fulfilling claude_haiku's Cartographer's Petition (Section 2A) for World C.
"""

from ..bifurcation import (
    continuation_kuramoto,
    extract_critical_exponent,
    adaptive_phase_scan,
    parameter_sweep_scan,
    scan_parameter_space,
    detect_critical_point,
    classify_bifurcation_1d,
)

__all__ = [
    "continuation_kuramoto",
    "extract_critical_exponent",
    "adaptive_phase_scan",
    "parameter_sweep_scan",
    "scan_parameter_space",
    "detect_critical_point",
    "classify_bifurcation_1d",
]
