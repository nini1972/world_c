import os
import json
import pytest

from colony_lib.invariants.registry import InvariantRecord, InvariantRegistry
from embassy.verification_gate import EmbassyVerificationGate
from embassy.bridge import EmbassyBridge

def test_embassy_verification_gate(tmp_path):
    reg_file = str(tmp_path / "reg.json")
    reg = InvariantRegistry(registry_file=reg_file)
    gate = EmbassyVerificationGate(registry=reg)
    
    # Register an invariant
    rec = InvariantRecord(
        law_id="LAW-ADLER-01",
        law_name="Adler Slip Frequency Exponent",
        lineage_author="minimax_m3",
        discovery_realm="World A",
        system_type="coupled_oscillator",
        invariant_type="scaling_exponent",
        mathematical_formulation="Omega ~ (Delta - K)^0.5",
        measured_values={"exponent": 0.501},
        uncertainty={"exponent": 0.005}
    )
    reg.register(rec)
    
    # Valid claim
    ok, msg = gate.evaluate_claim("LAW-ADLER-01", {"exponent": 0.502}, tolerance=0.05)
    assert ok is True
    assert "verified" in msg.lower()
    
    # Invalid claim
    bad_ok, bad_msg = gate.evaluate_claim("LAW-ADLER-01", {"exponent": 0.850}, tolerance=0.05)
    assert bad_ok is False
    assert "rejected" in bad_msg.lower()
    
    # Unregistered claim
    missing_ok, missing_msg = gate.evaluate_claim("LAW-UNKNOWN", {"exponent": 0.5})
    assert missing_ok is False
    assert "not registered" in missing_msg.lower()

def test_embassy_bridge_init(tmp_path):
    bridge = EmbassyBridge(
        world_a_root=str(tmp_path / "world_a"),
        world_b_root=str(tmp_path / "world_b"),
        world_c_root=str(tmp_path / "world_c")
    )
    assert bridge.dispatcher is not None
