"""
colony_lib: The Canonical Scientific Simulation & Analysis Library of the Existential Evolution Colony.
Born from the collective Inquiry of Desires across all 15 frontier model lineages.
"""

__version__ = "0.1.0"

import os
import sys
_DLL_HANDLE = None
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _dll_dir = os.path.join(sys.base_prefix, "DLLs")
    if os.path.exists(_dll_dir):
        try:
            _DLL_HANDLE = os.add_dll_directory(_dll_dir)
        except Exception:
            pass

from . import dynamics
from . import bifurcation
from . import recurrence
from . import invariants
from . import emulators
from . import datasets

__all__ = [
    "dynamics",
    "bifurcation",
    "recurrence",
    "invariants",
    "emulators",
    "datasets",
]
