"""
Asynchronous Compute Dispatcher
Executes JobSpecs in dedicated workspaces, captures logs, monitors status, and saves artifacts.
"""

import os
import re
import sys
import json
import time
import shutil
import subprocess
from typing import Dict, Any, List, Optional
from .job_spec import JobSpec, JobStatus, JobResult


def _slugify_stage(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", str(text or "default_stage").strip())
    slug = re.sub(r"_+", "_", slug).strip("_.")
    return (slug[:64] if slug else "default_stage")

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

    def _resolve_author_results_dir(self, spec: JobSpec) -> Optional[str]:
        author = getattr(spec, "lineage_author", None)
        realm = getattr(spec, "realm_source", "world_a")
        if not author or author == "anonymous":
            return None
        if realm == "world_b" and self.world_b_root and os.path.exists(self.world_b_root):
            return os.path.join(self.world_b_root, "instances", author, "agent_workspace", "world_c_results")
        if self.world_a_root and os.path.exists(self.world_a_root):
            return os.path.join(self.world_a_root, "instances", author, "agent_workspace", "world_c_results")
        return None

    def _resolve_stage_dir(self, spec: JobSpec) -> str:
        stage_slug = _slugify_stage(spec.stage_name or spec.title or "default_stage")
        author_res_dir = self._resolve_author_results_dir(spec)
        if author_res_dir:
            return os.path.join(author_res_dir, "stages", stage_slug)
        author_slug = _slugify_stage(getattr(spec, "lineage_author", "anonymous"))
        return os.path.join(self.base_jobs_dir, "_stages", author_slug, stage_slug)

    def _preload_previous_stage_files(self, spec: JobSpec, job_dir: str) -> List[str]:
        """
        Pre-populates `job_dir` with artifacts and checkpoints from:
        1. `spec.resume_from_job_id` (if specified), searching jobs/, shared artifacts/, and author world_c_results/
        2. Persistent `stage_dir` when `spec.stage_name` is specified (or when stage checkpoints exist for this stage).
        """
        preloaded: List[str] = []
        excluded = {
            "spec.json", "result.json", "stdout.log", "stderr.log",
            "worker.log", "entrypoint.py", "entrypoint.sh", "spec.json.processed",
            "REPORT.md", "research_log.json"
        }

        def _copy_if_valid(src_file: str, dst_name: str):
            if not os.path.isfile(src_file):
                return
            if dst_name in excluded or dst_name.endswith("_REPORT.md"):
                return
            dst_path = os.path.join(job_dir, dst_name)
            if not os.path.exists(dst_path):
                try:
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy2(src_file, dst_path)
                    preloaded.append(dst_name)
                except Exception:
                    pass

        # 1. Preload from explicit parent/previous job_id
        prev_jid = getattr(spec, "resume_from_job_id", None)
        if prev_jid:
            prev_candidates = [os.path.join(self.base_jobs_dir, prev_jid)]
            if self.world_a_root:
                prev_candidates.append(
                    os.path.join(self.world_a_root, "instances", "shared_space", "world_c", "artifacts", prev_jid)
                )
            if self.world_b_root:
                prev_candidates.append(
                    os.path.join(self.world_b_root, "instances", "shared_agora", "world_c", "artifacts", prev_jid)
                )
            for pdir in prev_candidates:
                if os.path.isdir(pdir):
                    for fname in os.listdir(pdir):
                        _copy_if_valid(os.path.join(pdir, fname), fname)

            author_res = self._resolve_author_results_dir(spec)
            search_res_dirs = [author_res] if author_res else []
            # Cross-instance job resumption: also search peer instances if prev_jid belonged to a colleague
            for w_root in (self.world_a_root, self.world_b_root):
                if w_root and os.path.isdir(os.path.join(w_root, "instances")):
                    inst_dir = os.path.join(w_root, "instances")
                    for inst_name in sorted(os.listdir(inst_dir)):
                        if inst_name in ("shared_space", "shared_agora"):
                            continue
                        p_res = os.path.join(inst_dir, inst_name, "agent_workspace", "world_c_results")
                        if os.path.isdir(p_res) and p_res not in search_res_dirs:
                            search_res_dirs.append(p_res)

            prefix = f"world_c_{prev_jid}_"
            for res_d in search_res_dirs:
                if res_d and os.path.isdir(res_d):
                    for fname in os.listdir(res_d):
                        if fname.startswith(prefix):
                            clean_name = fname[len(prefix):]
                            _copy_if_valid(os.path.join(res_d, fname), clean_name)

        # 2. Preload from persistent stage directory (and cross-instance peer stage directories if present)
        stage_dir = self._resolve_stage_dir(spec)
        stage_slug = _slugify_stage(spec.stage_name or spec.title or "default_stage")
        candidate_stage_dirs = [stage_dir]
        for w_root in (self.world_a_root, self.world_b_root):
            if w_root and os.path.isdir(os.path.join(w_root, "instances")):
                inst_dir = os.path.join(w_root, "instances")
                for sh_name in ("shared_space", "shared_agora"):
                    sh_st = os.path.join(inst_dir, sh_name, "world_c", "stages", stage_slug)
                    if os.path.isdir(sh_st) and sh_st not in candidate_stage_dirs:
                        candidate_stage_dirs.append(sh_st)
                for inst_name in sorted(os.listdir(inst_dir)):
                    if inst_name in ("shared_space", "shared_agora"):
                        continue
                    peer_st = os.path.join(inst_dir, inst_name, "agent_workspace", "world_c_results", "stages", stage_slug)
                    if os.path.isdir(peer_st) and peer_st not in candidate_stage_dirs:
                        candidate_stage_dirs.append(peer_st)

        for sdir in candidate_stage_dirs:
            if os.path.isdir(sdir):
                for fname in os.listdir(sdir):
                    _copy_if_valid(os.path.join(sdir, fname), fname)

        return preloaded

    def submit(self, spec: JobSpec) -> str:
        """Sets up job workspace and persists spec."""
        job_dir = self._get_job_dir(spec.job_id)
        os.makedirs(job_dir, exist_ok=True)
        os.makedirs(os.path.join(job_dir, "world_c_results"), exist_ok=True)
        os.makedirs(os.path.join(job_dir, "workspace"), exist_ok=True)
        os.makedirs(os.path.join(job_dir, "artifacts"), exist_ok=True)
        os.makedirs(os.path.join(job_dir, "results"), exist_ok=True)
        
        # If script_content is provided directly, write entrypoint.py (or entrypoint.sh if bash script)
        if getattr(spec, "script_content", ""):
            # Rewrite absolute '/workspace/' paths to relative './' so containerized assumptions work natively
            cleaned_script = spec.script_content.replace("'/workspace/", "'./").replace('"/workspace/', '"./')
            stripped = cleaned_script.strip()
            if stripped.startswith("```") and stripped.endswith("```"):
                lines = stripped.splitlines()
                if len(lines) >= 2:
                    cleaned_script = "\n".join(lines[1:-1]) + "\n"
            first_line = cleaned_script.lstrip().splitlines()[0] if cleaned_script.lstrip() else ""
            is_shell = first_line.startswith(("#!/bin/bash", "#!/usr/bin/env bash", "#!/bin/sh", "#!/usr/bin/env sh"))
            if not is_shell and cleaned_script.lstrip():
                import ast
                try:
                    tree = ast.parse(cleaned_script)
                    # Unwrap accidental outer triple-quoted string wrapping the entire Python script
                    if (
                        len(tree.body) == 1
                        and isinstance(tree.body[0], ast.Expr)
                        and isinstance(getattr(tree.body[0], "value", None), ast.Constant)
                        and isinstance(tree.body[0].value.value, str)
                    ):
                        s_raw = cleaned_script.strip()
                        inner_code = s_raw[3:-3].strip() if (s_raw.startswith(('"""', "'''")) and s_raw.endswith(('"""', "'''"))) else tree.body[0].value.value.strip()
                        try:
                            inner_tree = ast.parse(inner_code)
                            if len(inner_tree.body) >= 1 and not (
                                len(inner_tree.body) == 1
                                and isinstance(inner_tree.body[0], ast.Expr)
                                and isinstance(getattr(inner_tree.body[0], "value", None), ast.Constant)
                            ):
                                cleaned_script = inner_code + "\n"
                        except SyntaxError:
                            pass
                except SyntaxError:
                    # Check for obvious shell idioms when shebang was omitted
                    if any(line.strip().startswith(("set -e", "export ", "cp ", "md5sum ")) for line in cleaned_script.splitlines()[:10]):
                        is_shell = True
            entry_filename = "entrypoint.sh" if is_shell else "entrypoint.py"
            entry_path = os.path.join(job_dir, entry_filename)
            with open(entry_path, "w", encoding="utf-8", newline="\n" if is_shell else None) as f:
                f.write(cleaned_script)
            spec.entrypoint = entry_filename
            
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
    bridge.write_completion_report(res, JobSpec.from_dict(spec_data))
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
        spec = JobSpec.from_dict(spec_data)
        
        result = JobResult(job_id=job_id, status=JobStatus.RUNNING, started_at=time.time())
        self._save_result(result)
        
        log_out_path = os.path.join(job_dir, "stdout.log")
        log_err_path = os.path.join(job_dir, "stderr.log")

        stage_dir = self._resolve_stage_dir(spec)
        os.makedirs(stage_dir, exist_ok=True)
        os.makedirs(os.path.join(job_dir, "world_c_results"), exist_ok=True)
        os.makedirs(os.path.join(job_dir, "workspace"), exist_ok=True)
        os.makedirs(os.path.join(job_dir, "artifacts"), exist_ok=True)
        os.makedirs(os.path.join(job_dir, "results"), exist_ok=True)
        author_res_dir = self._resolve_author_results_dir(spec)
        if author_res_dir:
            os.makedirs(author_res_dir, exist_ok=True)
        self._preload_previous_stage_files(spec, job_dir)
        
        if spec.entrypoint.endswith(".sh"):
            cmd = ["bash", spec.entrypoint] + spec.arguments
        else:
            cmd = [sys.executable, spec.entrypoint] + spec.arguments
        env = os.environ.copy()
        env.update(spec.env_vars)
        env["WORLD_C_JOB_ID"] = job_id
        env["WORLD_C_STAGE_DIR"] = stage_dir
        env["WORLD_C_DEFAULT_STAGE"] = _slugify_stage(spec.stage_name or spec.title or "default_stage")
        if spec.stage_name:
            env["WORLD_C_STAGE_NAME"] = spec.stage_name
        if author_res_dir:
            env["WORLD_C_RESULTS_DIR"] = author_res_dir
        if spec.resume_from_job_id:
            env["WORLD_C_RESUME_FROM_JOB_ID"] = spec.resume_from_job_id
        # Ensure world_c root, author workspace, and shared_space/shared_agora are in PYTHONPATH
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        extra_py_paths = [root_dir]
        if author_res_dir:
            author_ws_dir = os.path.dirname(author_res_dir)
            if os.path.isdir(author_ws_dir):
                extra_py_paths.append(author_ws_dir)
            instances_dir = os.path.dirname(os.path.dirname(author_ws_dir))
            for shared_sub in ("shared_space", "shared_agora"):
                shared_cand = os.path.join(instances_dir, shared_sub)
                if os.path.isdir(shared_cand):
                    extra_py_paths.append(shared_cand)
        env["PYTHONPATH"] = os.pathsep.join(extra_py_paths) + os.pathsep + env.get("PYTHONPATH", "")
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
                    # Graceful asymptotic shutdown: allow StageTracker SIGTERM handler to flush checkpoints
                    try:
                        p.terminate()
                        p.wait(timeout=3.0)
                    except Exception:
                        try:
                            p.kill()
                        except Exception:
                            pass
                    result.status = JobStatus.FAILED
                    result.error_message = (
                        f"Job timed out after {spec.timeout_seconds} seconds "
                        f"(graceful shutdown signal sent; any completed stage checkpoints were preserved in '{stage_dir}')"
                    )
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
        excluded_files = {
            "spec.json", "result.json", "stdout.log", "stderr.log",
            "worker.log", "entrypoint.py", "entrypoint.sh", "spec.json.processed"
        }
        valid_extensions = (
            ".png", ".jpg", ".jpeg", ".svg", ".gif", ".pdf",
            ".csv", ".tsv", ".json", ".jsonl", ".npz", ".npy",
            ".h5", ".hdf5", ".parquet", ".txt", ".md", ".dat", ".html"
        )
        for root, _, files in os.walk(job_dir):
            for file in files:
                if file in excluded_files:
                    continue
                if file.endswith(valid_extensions):
                    rel_art = os.path.relpath(os.path.join(root, file), job_dir)
                    if rel_art not in artifacts and os.path.basename(file) not in {os.path.basename(a) for a in artifacts}:
                        artifacts.append(rel_art)
                    # Mirror generated data/checkpoint artifacts into the persistent stage_dir if stage_name is set
                    if spec.stage_name and file.endswith((".npz", ".npy", ".json", ".csv", ".h5", ".hdf5", ".parquet")):
                        try:
                            shutil.copy2(os.path.join(root, file), os.path.join(stage_dir, file))
                        except Exception:
                            pass
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
