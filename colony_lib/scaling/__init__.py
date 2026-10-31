"""
colony_lib.scaling: Finite-Size, Finite-Horizon Scaling, Collapse & Universal Exponent Toolkit.
Fulfilling DeepSeek V4 Flash's and claude_haiku's Technical Manifestos for World C.
"""

from .trajectory_summary import TrajectorySummary
from .finite_scaling import (
    extract_finite_size_scaling,
    extract_finite_horizon_scaling,
    falsification_certificate,
)
from .universal_laws import (
    scaling_collapse,
    fit_scaling_law,
)

__all__ = [
    "TrajectorySummary",
    "extract_finite_size_scaling",
    "extract_finite_horizon_scaling",
    "falsification_certificate",
    "scaling_collapse",
    "fit_scaling_law",
]
