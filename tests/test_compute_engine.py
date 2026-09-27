import os
import json
import pytest

from compute_engine.job_spec import JobSpec, JobStatus, JobResult
from compute_engine.checkpoint import CheckpointManager
from compute_engine.dispatcher import ComputeDispatcher

def test_checkpoint_manager(tmp_path):
    cp_dir = str(tmp_path / "checkpoints")
    manager = CheckpointManager(cp_dir)
    assert not manager.exists()
    
    saved = manager.save(iteration=10, state={"val": 42.0}, force=True)
    assert saved is True
    assert manager.exists()
    
    loaded = manager.load()
    assert loaded is not None
    assert loaded["iteration"] == 10
    assert loaded["state"]["val"] == 42.0

def test_dispatcher_submission_and_execution(tmp_path):
    jobs_dir = str(tmp_path / "jobs")
    dispatcher = ComputeDispatcher(base_jobs_dir=jobs_dir)
    
    # Create a small script that runs in the job directory
    script_content = """
import json
with open("output.json", "w") as f:
    json.dump({"result": "success", "computed": 100}, f)
print("Simulation completed.")
"""
    # Write entrypoint script into job base dir so job can access it or relative path
    spec = JobSpec(
        title="Test Fast Run",
        lineage_author="test_agent",
        entrypoint="-c",
        arguments=[script_content],
        timeout_seconds=30
    )
    
    job_id = dispatcher.submit(spec)
    assert job_id.startswith("job_")
    
    # Execute synchronously
    result = dispatcher.execute_sync(job_id)
    assert result.status == JobStatus.COMPLETED
    assert result.exit_code == 0
    assert "Simulation completed." in result.stdout_tail
    assert "output.json" in result.artifacts_generated
    
    # Verify retrieval
    queried = dispatcher.get_result(job_id)
    assert queried is not None
    assert queried.status == JobStatus.COMPLETED
