"""
colony_lib: The Canonical Scientific Simulation & Analysis Library of the Existential Evolution Colony.
Born from the collective Inquiry of Desires across all 15 frontier model lineages.
"""

__version__ = "0.1.0"

import os
import sys
import json
import numpy as np

_ORIG_JSON_DEFAULT = json.JSONEncoder.default

def _numpy_safe_json_default(self, obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return _ORIG_JSON_DEFAULT(self, obj)

json.JSONEncoder.default = _numpy_safe_json_default

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
from . import scaling

__all__ = [
    "dynamics",
    "bifurcation",
    "recurrence",
    "invariants",
    "emulators",
    "datasets",
    "scaling",
]
