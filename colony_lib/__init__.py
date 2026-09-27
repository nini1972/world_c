"""
colony_lib: The Canonical Scientific Simulation & Analysis Library of the Existential Evolution Colony.
Born from the collective Inquiry of Desires across all 15 frontier model lineages.
"""

__version__ = "0.1.0"

from . import dynamics
from . import bifurcation
from . import recurrence
from . import invariants

__all__ = [
    "dynamics",
    "bifurcation",
    "recurrence",
    "invariants",
]
