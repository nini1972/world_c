"""
Invariants Module: Universal scaling law collapse optimization, invariant signatures, and cryptographic registry.
"""

from .collapse import optimize_scaling_collapse, scaling_quality_score
from .registry import InvariantRecord, InvariantRegistry
from ..morphospace.topology import compute_betti_0_curve

# Ergonomic aliases for topological invariants
betti0_persistence = compute_betti_0_curve

__all__ = [
    "optimize_scaling_collapse",
    "scaling_quality_score",
    "InvariantRecord",
    "InvariantRegistry",
    "compute_betti_0_curve",
    "betti0_persistence",
]
