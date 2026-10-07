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
    def __init__(
        self,
        base_jobs_dir: str = "jobs",
        world_a_root: Optional[str] = None,
        world_b_root: Optional[str] = None
    ):
        self.base_jobs_dir = os.path.abspath(base_jobs_dir)
        os.makedirs(self.base_jobs_dir, exist_ok=True)
        self.registry_file = os.path.join(self.base_jobs_dir, "jobs_index.json")
        self.world_a_root = os.path.abspath(world_a_root) if world_a_root else None
        self.world_b_root = os.path.abspath(world_b_root) if world_b_root else None

    def _get_job_dir(self, job_id: str) -> str:
        return os.path.join(self.base_jobs_dir, job_id)

    def submit(self, spec: JobSpec) -> str:
        """Sets up job workspace and persists spec."""
        job_dir = self._get_job_dir(spec.job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # If script_content is provided directly, write entrypoint.py
        if getattr(spec, "script_content", ""):
            entry_path = os.path.join(job_dir, "entrypoint.py")
            with open(entry_path, "w", encoding="utf-8") as f:
                f.write(spec.script_content)
            spec.entrypoint = "entrypoint.py"
            
        spec_path = os.path.join(job_dir, "spec.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec.__dict__, f, indent=2)
            
        result = JobResult(job_id=spec.job_id, status=JobStatus.QUEUED)
        self._save_result(result)
        return spec.job_id

    def execute_async(self, job_id: str) -> subprocess.Popen:
        """Spawns an independent background worker to execute the job and auto-publish artifacts."""
        worker_code = f"""
import sys, os, json
_dll_handle = None
if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
    _dll_dir = os.path.join(sys.base_prefix, 'DLLs')
    if os.path.exists(_dll_dir):
        try:
            _dll_handle = os.add_dll_directory(_dll_dir)
        except Exception:
            pass
root_dir = {repr(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))}
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from compute_engine.dispatcher import ComputeDispatcher
from compute_engine.job_spec import JobSpec
from embassy.bridge import EmbassyBridge

job_dir = {repr(self._get_job_dir(job_id))}
dispatcher = ComputeDispatcher(
    base_jobs_dir={repr(self.base_jobs_dir)},
    world_a_root={repr(self.world_a_root)},
    world_b_root={repr(self.world_b_root)}
)
res = dispatcher.execute_sync({repr(job_id)})

try:
    bridge = EmbassyBridge(
        world_c_root=root_dir,
        world_a_root={repr(self.world_a_root)},
        world_b_root={repr(self.world_b_root)}
    )
    spec_path = os.path.join(job_dir, "spec.json")
    with open(spec_path, "r", encoding="utf-8") as f:
        spec_data = json.load(f)
    realm = spec_data.get("realm_source", "world_a")
    author = spec_data.get("lineage_author")
    bridge.publish_completed_artifacts({repr(job_id)}, target_realms=[realm], lineage_author=author)
    bridge.write_completion_report(res, JobSpec(**spec_data))
except Exception as err:
    print(f"Error publishing artifacts: {{err}}")
"""
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        env = os.environ.copy()
        if self.world_a_root:
            env["WORLD_A_ROOT"] = self.world_a_root
        if self.world_b_root:
            env["WORLD_B_ROOT"] = self.world_b_root
        env["PYTHONPATH"] = root_dir + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        dll_dir = os.path.join(sys.base_prefix, "DLLs")
        # Remove any gstreamer path that clashes with standard python ctypes/ffi
        path_parts = [p for p in env.get("PATH", "").split(os.pathsep) if "gstreamer" not in p.lower()]
        env["PATH"] = dll_dir + os.pathsep + sys.base_prefix + os.pathsep + os.pathsep.join(path_parts)
        
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP
            
        worker_log = os.path.join(self._get_job_dir(job_id), "worker.log")
        w_out = open(worker_log, "w", encoding="utf-8")
        p = subprocess.Popen(
            [sys.executable, "-c", worker_code],
            cwd=self._get_job_dir(job_id),
            env=env,
            stdout=w_out,
            stderr=w_out,
            creationflags=flags
        )
        return p

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
        # Ensure world_c root is in PYTHONPATH and Python DLLs are prioritized
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        env["PYTHONPATH"] = root_dir + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        dll_dir = os.path.join(sys.base_prefix, "DLLs")
        path_parts = [p for p in env.get("PATH", "").split(os.pathsep) if "gstreamer" not in p.lower()]
        env["PATH"] = dll_dir + os.pathsep + sys.base_prefix + os.pathsep + os.pathsep.join(path_parts)
        
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
