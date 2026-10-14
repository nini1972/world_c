"""
colony_lib.scaling: Finite-Size, Finite-Horizon Scaling & Epistemic Toolkit.
Fulfilling DeepSeek V4 Flash's Technical Manifesto for the Existential Evolution Colony and Agora.
"""

from .trajectory_summary import TrajectorySummary
from .finite_scaling import (
    extract_finite_size_scaling,
    extract_finite_horizon_scaling,
    falsification_certificate,
)

__all__ = [
    "TrajectorySummary",
    "extract_finite_size_scaling",
    "extract_finite_horizon_scaling",
    "falsification_certificate",
]
