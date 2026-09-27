"""
Invariants Module: Universal scaling law collapse optimization, invariant signatures, and cryptographic registry.
"""

from .collapse import optimize_scaling_collapse, scaling_quality_score
from .registry import InvariantRecord, InvariantRegistry

__all__ = [
    "optimize_scaling_collapse",
    "scaling_quality_score",
    "InvariantRecord",
    "InvariantRegistry",
]
