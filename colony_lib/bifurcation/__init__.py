"""
Bifurcation Module: Critical threshold scanning, continuation, adaptive phase cartography, and phase transition detection.
"""

from .continuation import (
    parameter_sweep_scan,
    scan_parameter_space,
    detect_critical_point,
    continuation_kuramoto,
    extract_critical_exponent,
    adaptive_phase_scan,
)
from .normal_forms import classify_bifurcation_1d

__all__ = [
    "parameter_sweep_scan",
    "scan_parameter_space",
    "detect_critical_point",
    "continuation_kuramoto",
    "extract_critical_exponent",
    "adaptive_phase_scan",
    "classify_bifurcation_1d",
]
