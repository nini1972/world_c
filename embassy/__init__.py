"""
Embassy Module: Cross-world routing, artifact synchronization, and verification gate
connecting World A (evolution_sandbox), World B (synthetic_agora), and World C (world_c).
"""

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

from .bridge import EmbassyBridge
from .verification_gate import EmbassyVerificationGate

__all__ = [
    "EmbassyBridge",
    "EmbassyVerificationGate",
]
