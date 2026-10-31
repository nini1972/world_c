"""
colony_lib.scaling_analysis: Universal scaling law extraction, collapse, and finite-size scaling.
Fulfilling claude_haiku's Cartographer's Petition (Section 2C) for World C.
"""

from ..scaling import (
    scaling_collapse,
    fit_scaling_law,
    extract_finite_size_scaling,
    extract_finite_horizon_scaling,
    falsification_certificate,
    TrajectorySummary,
)
from ..invariants.collapse import (
    optimize_scaling_collapse,
    scaling_quality_score,
)

__all__ = [
    "scaling_collapse",
    "fit_scaling_law",
    "extract_finite_size_scaling",
    "extract_finite_horizon_scaling",
    "falsification_certificate",
    "TrajectorySummary",
    "optimize_scaling_collapse",
    "scaling_quality_score",
]
