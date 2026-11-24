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

def test_embassy_bridge_dedicated_delivery(tmp_path):
    from compute_engine.job_spec import JobSpec, JobResult, JobStatus
    
    world_a = tmp_path / "world_a"
    world_c = tmp_path / "world_c"
    author_ws = world_a / "instances" / "test_agent" / "agent_workspace"
    author_ws.mkdir(parents=True)
    
    bridge = EmbassyBridge(
        world_a_root=str(world_a),
        world_c_root=str(world_c)
    )
    
    # Create fake job in world_c/jobs
    job_id = "job_test_123"
    job_dir = world_c / "jobs" / job_id
    job_dir.mkdir(parents=True)
    fake_art = job_dir / "plot.png"
    fake_art.write_text("fake_png_data", encoding="utf-8")
    
    spec = JobSpec(
        job_id=job_id,
        title="Test Scaling",
        lineage_author="test_agent",
        realm_source="world_a",
        script_content="print('ok')"
    )
    
    res = JobResult(
        job_id=job_id,
        status=JobStatus.COMPLETED,
        exit_code=0,
        started_at=100.0,
        finished_at=105.0,
        execution_time_seconds=5.0,
        stdout_tail="done",
        stderr_tail="",
        artifacts_generated=["plot.png"]
    )
    
    bridge.dispatcher._save_result(res)
    bridge.publish_completed_artifacts(job_id, target_realms=["world_a"], lineage_author="test_agent")
    bridge.write_completion_report(res, spec, destination_dir=str(world_a / "instances" / "shared_space"))
    
    # Verify dedicated shared directories
    dedicated_report = world_a / "instances" / "shared_space" / "world_c" / "reports" / f"world_c_{job_id}_REPORT.md"
    assert dedicated_report.exists()
    report_text = dedicated_report.read_text(encoding="utf-8")
    assert "Test Scaling" in report_text
    assert "NUMERICAL CORRUPTION" not in report_text
    
    dedicated_art = world_a / "instances" / "shared_space" / "world_c" / "artifacts" / f"world_c_{job_id}_plot.png"
    assert dedicated_art.exists()
    
    # Verify author workspace direct delivery
    ws_report = author_ws / "world_c_results" / "REPORT.md"
    assert ws_report.exists()
    
    ws_art = author_ws / "world_c_results" / "plot.png"
    assert ws_art.exists()


def test_embassy_bridge_flags_silent_nan_overflow(tmp_path):
    from compute_engine.job_spec import JobSpec, JobResult, JobStatus

    world_a = tmp_path / "world_a"
    world_c = tmp_path / "world_c"
    author_ws = world_a / "instances" / "glm_5_2" / "agent_workspace"
    author_ws.mkdir(parents=True)

    bridge = EmbassyBridge(
        world_a_root=str(world_a),
        world_c_root=str(world_c),
    )

    job_id = "job_glm_5_2_nan_test"
    job_dir = world_c / "jobs" / job_id
    job_dir.mkdir(parents=True)

    spec = JobSpec(
        job_id=job_id,
        title="Resonance Gap Sweep",
        lineage_author="glm_5_2",
        realm_source="world_a",
        script_content="print('mean r = nan')",
    )

    res = JobResult(
        job_id=job_id,
        status=JobStatus.COMPLETED,
        exit_code=0,
        started_at=100.0,
        finished_at=110.0,
        execution_time_seconds=10.0,
        stdout_tail="mean r = nan, std r = nan\nC_max = 0.0000",
        stderr_tail="RuntimeWarning: overflow encountered in divide\nRuntimeWarning: invalid value encountered in multiply",
        artifacts_generated=[],
    )

    bridge.write_completion_report(res, spec)

    shared_report = world_a / "instances" / "shared_space" / "world_c" / "reports" / f"world_c_{job_id}_REPORT.md"
    ws_report = author_ws / "world_c_results" / "REPORT.md"
    assert "NUMERICAL CORRUPTION / NaN OVERFLOW DETECTED" in shared_report.read_text(encoding="utf-8")
    assert "NUMERICAL CORRUPTION / NaN OVERFLOW DETECTED" in ws_report.read_text(encoding="utf-8")

    rlog = json.loads((author_ws / "world_c_results" / "research_log.json").read_text(encoding="utf-8"))
    assert "numerical-nan-warning" in rlog[0]["tags"]


def test_embassy_bridge_flags_silent_exit0_traceback(tmp_path):
    from compute_engine.job_spec import JobSpec, JobResult, JobStatus

    world_a = tmp_path / "world_a"
    world_c = tmp_path / "world_c"
    author_ws = world_a / "instances" / "glm_5_2" / "agent_workspace"
    author_ws.mkdir(parents=True)

    bridge = EmbassyBridge(
        world_a_root=str(world_a),
        world_c_root=str(world_c),
    )

    # 1. Exit-0 job that caught a traceback and produced 0 valid data points
    job_ok_id = "job_glm_5_2_exit0_tb"
    (world_c / "jobs" / job_ok_id).mkdir(parents=True)
    spec_ok = JobSpec(
        job_id=job_ok_id,
        title="Silent Traceback Test",
        lineage_author="glm_5_2",
        realm_source="world_a",
        script_content="print('Total valid data points: 0')",
    )
    res_ok = JobResult(
        job_id=job_ok_id,
        status=JobStatus.COMPLETED,
        exit_code=0,
        started_at=100.0,
        finished_at=104.0,
        execution_time_seconds=4.0,
        stdout_tail="Total valid data points: 0\nCOMPLETE",
        stderr_tail=(
            "Traceback (most recent call last):\n"
            "AttributeError: 'DatasetDataArray' object has no attribute 'select_dtypes'"
        ),
        artifacts_generated=[],
    )
    bridge.write_completion_report(res_ok, spec_ok)

    shared_report = world_a / "instances" / "shared_space" / "world_c" / "reports" / f"world_c_{job_ok_id}_REPORT.md"
    ws_report = author_ws / "world_c_results" / "REPORT.md"
    assert "SILENT RUNTIME ERROR / CAUGHT TRACEBACK DETECTED" in shared_report.read_text(encoding="utf-8")
    assert "SILENT RUNTIME ERROR / CAUGHT TRACEBACK DETECTED" in ws_report.read_text(encoding="utf-8")

    rlog = json.loads((author_ws / "world_c_results" / "research_log.json").read_text(encoding="utf-8"))
    ok_entry = [e for e in rlog if e["job_id"] == job_ok_id][0]
    assert "silent-runtime-error" in ok_entry["tags"]

    # 2. Exit-1 FAILED job with a traceback must NOT trigger the silent exit-0 warning or tag
    job_fail_id = "job_glm_5_2_exit1_tb"
    (world_c / "jobs" / job_fail_id).mkdir(parents=True)
    spec_fail = JobSpec(
        job_id=job_fail_id,
        title="Normal Failed Traceback Test",
        lineage_author="glm_5_2",
        realm_source="world_a",
        script_content="raise RuntimeError('boom')",
    )
    res_fail = JobResult(
        job_id=job_fail_id,
        status=JobStatus.FAILED,
        exit_code=1,
        started_at=110.0,
        finished_at=112.0,
        execution_time_seconds=2.0,
        stdout_tail="",
        stderr_tail="Traceback (most recent call last):\nRuntimeError: boom",
        artifacts_generated=[],
    )
    bridge.write_completion_report(res_fail, spec_fail)

    fail_report = world_a / "instances" / "shared_space" / "world_c" / "reports" / f"world_c_{job_fail_id}_REPORT.md"
    assert "SILENT RUNTIME ERROR / CAUGHT TRACEBACK DETECTED" not in fail_report.read_text(encoding="utf-8")
    rlog2 = json.loads((author_ws / "world_c_results" / "research_log.json").read_text(encoding="utf-8"))
    fail_entry = [e for e in rlog2 if e["job_id"] == job_fail_id][0]
    assert "silent-runtime-error" not in fail_entry["tags"]
