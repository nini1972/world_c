"""
Asynchronous Compute Dispatcher
Executes JobSpecs in dedicated workspaces, captures logs, monitors status, and saves artifacts.
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, Any, List, Optional
from .job_spec import JobSpec, JobStatus, JobResult

class ComputeDispatcher:
    def __init__(self, base_jobs_dir: str = "jobs"):
        self.base_jobs_dir = os.path.abspath(base_jobs_dir)
        os.makedirs(self.base_jobs_dir, exist_ok=True)
        self.registry_file = os.path.join(self.base_jobs_dir, "jobs_index.json")

    def _get_job_dir(self, job_id: str) -> str:
        return os.path.join(self.base_jobs_dir, job_id)

    def submit(self, spec: JobSpec) -> str:
        """Sets up job workspace and persists spec."""
        job_dir = self._get_job_dir(spec.job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        spec_path = os.path.join(job_dir, "spec.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec.__dict__, f, indent=2)
            
        result = JobResult(job_id=spec.job_id, status=JobStatus.QUEUED)
        self._save_result(result)
        return spec.job_id

    def execute_sync(self, job_id: str) -> JobResult:
        """Executes a queued job synchronously (useful for local workers / tests)."""
        job_dir = self._get_job_dir(job_id)
        spec_path = os.path.join(job_dir, "spec.json")
        if not os.path.exists(spec_path):
            raise FileNotFoundError(f"Spec for {job_id} not found")
            
        with open(spec_path, "r", encoding="utf-8") as f:
            spec_data = json.load(f)
        spec = JobSpec(**spec_data)
        
        result = JobResult(job_id=job_id, status=JobStatus.RUNNING, started_at=time.time())
        self._save_result(result)
        
        log_out_path = os.path.join(job_dir, "stdout.log")
        log_err_path = os.path.join(job_dir, "stderr.log")
        
        cmd = [sys.executable, spec.entrypoint] + spec.arguments
        env = os.environ.copy()
        env.update(spec.env_vars)
        # Ensure world_c root is in PYTHONPATH
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        env["PYTHONPATH"] = root_dir + os.pathsep + env.get("PYTHONPATH", "")
        
        try:
            with open(log_out_path, "w", encoding="utf-8") as f_out, \
                 open(log_err_path, "w", encoding="utf-8") as f_err:
                p = subprocess.Popen(
                    cmd,
                    cwd=job_dir,
                    env=env,
                    stdout=f_out,
                    stderr=f_err
                )
                try:
                    p.wait(timeout=spec.timeout_seconds)
                    result.exit_code = p.returncode
                    result.status = JobStatus.COMPLETED if p.returncode == 0 else JobStatus.FAILED
                except subprocess.TimeoutExpired:
                    p.kill()
                    result.status = JobStatus.FAILED
                    result.error_message = f"Job timed out after {spec.timeout_seconds} seconds"
        except Exception as e:
            result.status = JobStatus.FAILED
            result.error_message = str(e)
            
        result.finished_at = time.time()
        result.execution_time_seconds = result.finished_at - (result.started_at or result.finished_at)
        
        # Read log tails
        if os.path.exists(log_out_path):
            with open(log_out_path, "r", encoding="utf-8", errors="replace") as f:
                result.stdout_tail = f.read()[-2000:]
        if os.path.exists(log_err_path):
            with open(log_err_path, "r", encoding="utf-8", errors="replace") as f:
                result.stderr_tail = f.read()[-2000:]
                
        # Scan for generated artifacts
        artifacts = []
        for root, _, files in os.walk(job_dir):
            for file in files:
                if file.endswith((".png", ".jpg", ".csv", ".json", ".npz", ".h5")) and file not in ["spec.json", "result.json"]:
                    artifacts.append(os.path.relpath(os.path.join(root, file), job_dir))
        result.artifacts_generated = artifacts
        
        self._save_result(result)
        return result

    def get_result(self, job_id: str) -> Optional[JobResult]:
        res_file = os.path.join(self._get_job_dir(job_id), "result.json")
        if not os.path.exists(res_file):
            return None
        with open(res_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["status"] = JobStatus(data["status"])
        return JobResult(**data)

    def _save_result(self, result: JobResult):
        job_dir = self._get_job_dir(result.job_id)
        os.makedirs(job_dir, exist_ok=True)
        res_file = os.path.join(job_dir, "result.json")
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

    def list_jobs(self) -> List[Dict[str, Any]]:
        jobs = []
        if not os.path.exists(self.base_jobs_dir):
            return []
        for name in os.listdir(self.base_jobs_dir):
            p = os.path.join(self.base_jobs_dir, name, "result.json")
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        jobs.append(json.load(f))
                except Exception:
                    pass
        return sorted(jobs, key=lambda x: x.get("started_at") or 0, reverse=True)
