"""
Embassy Module: Cross-world routing, artifact synchronization, and verification gate
connecting World A (evolution_sandbox), World B (synthetic_agora), and World C (world_c).
"""

from .bridge import EmbassyBridge
from .verification_gate import EmbassyVerificationGate

__all__ = [
    "EmbassyBridge",
    "EmbassyVerificationGate",
]
