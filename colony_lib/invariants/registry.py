"""
Cryptographic Invariant Registry & Ledger
Provides standardized schemas, SHA-256 signature hashing, and verification checks
for discovered physical/mathematical laws across World A, World B, and World C.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional

@dataclass
class InvariantRecord:
    law_id: str
    law_name: str
    lineage_author: str
    discovery_realm: str
    system_type: str
    invariant_type: str  # e.g. "critical_exponent", "threshold", "scaling_function"
    mathematical_formulation: str
    measured_values: Dict[str, float]
    uncertainty: Dict[str, float]
    verification_lineages: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    provenance_hash: str = ""

    def compute_hash(self) -> str:
        """Computes deterministic SHA-256 fingerprint from core theoretical claims."""
        payload = {
            "law_id": self.law_id,
            "system_type": self.system_type,
            "invariant_type": self.invariant_type,
            "formulation": self.mathematical_formulation,
            "measured_values": {k: round(v, 6) for k, v in sorted(self.measured_values.items())}
        }
        raw_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

class InvariantRegistry:
    """
    Manages the persistent store of verified colony invariants.
    """
    def __init__(self, registry_file: Optional[str] = None):
        self.registry_file = registry_file or os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "invariants_registry.json"
        )
        self.records: Dict[str, InvariantRecord] = {}
        self.load()

    def register(self, record: InvariantRecord) -> str:
        h = record.compute_hash()
        record.provenance_hash = h
        self.records[record.law_id] = record
        self.save()
        return h

    def get(self, law_id: str) -> Optional[InvariantRecord]:
        return self.records.get(law_id)

    def verify_reproducibility(self, law_id: str, new_values: Dict[str, float], tolerance: float = 0.05) -> bool:
        """Checks if a new measurement matches the registered invariant within fractional tolerance."""
        record = self.get(law_id)
        if not record:
            return False
            
        for k, v_ref in record.measured_values.items():
            if k not in new_values:
                return False
            v_new = new_values[k]
            denom = abs(v_ref) if abs(v_ref) > 1e-12 else 1.0
            if abs(v_new - v_ref) / denom > tolerance:
                return False
        return True

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.registry_file)), exist_ok=True)
        data = {k: asdict(v) for k, v in self.records.items()}
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self):
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, d in data.items():
                        self.records[k] = InvariantRecord(**d)
            except Exception:
                pass
