"""
Bifurcation Module: Critical threshold scanning, continuation, and phase transition detection.
"""

from .continuation import parameter_sweep_scan, detect_critical_point
from .normal_forms import classify_bifurcation_1d

__all__ = [
    "parameter_sweep_scan",
    "detect_critical_point",
    "classify_bifurcation_1d",
]
