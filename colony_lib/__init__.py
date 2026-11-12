"""
colony_lib: The Canonical Scientific Simulation & Analysis Library of the Existential Evolution Colony.
Born from the collective Inquiry of Desires across all 15 frontier model lineages.
"""

__version__ = "0.2.0"

import os
import sys
import json
import math
import numpy as np

if not hasattr(np, "math"):
    setattr(np, "math", math)

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

# SciPy 1.14+ compatibility shim: csr_array removed legacy .asfptype() used by many LLMs
try:
    import scipy.sparse as _sp
    for _cls_name in ("csr_array", "csc_array", "coo_array", "dia_array", "lil_array", "dok_array", "bsr_array"):
        _cls = getattr(_sp, _cls_name, None)
        if _cls is not None and not hasattr(_cls, "asfptype"):
            setattr(_cls, "asfptype", lambda self: self.astype(np.float64))
except Exception:
    pass

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
from . import bifurcation_cartography
from . import network_topology
from . import recurrence
from . import invariants
from . import emulators
from . import datasets
from . import scaling
from . import scaling_analysis
from . import morphospace
from . import persistence
from .persistence import StageTracker, save_checkpoint, load_checkpoint, load_artifact, list_previous_artifacts

__all__ = [
    "dynamics",
    "bifurcation",
    "bifurcation_cartography",
    "network_topology",
    "recurrence",
    "invariants",
    "emulators",
    "datasets",
    "scaling",
    "scaling_analysis",
    "morphospace",
    "persistence",
    "StageTracker",
    "save_checkpoint",
    "load_checkpoint",
    "load_artifact",
    "list_previous_artifacts",
]
