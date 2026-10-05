"""
Cross-World Embassy Bridge
Facilitates job requests and findings synchronization across:
- World A (evolution_sandbox)
- World B (synthetic_agora)
- World C (world_c)
"""

import os
import sys
import time
import shutil
import json
import argparse
from typing import Dict, Any, List, Optional
from compute_engine.job_spec import JobSpec, JobStatus, JobResult
from compute_engine.dispatcher import ComputeDispatcher

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = [
            arg.encode("ascii", errors="replace").decode("ascii") if isinstance(arg, str) else arg
            for arg in args
        ]
        print(*safe_args, **kwargs)

class EmbassyBridge:
    def __init__(
        self,
        world_a_root: Optional[str] = None,
        world_b_root: Optional[str] = None,
        world_c_root: Optional[str] = None
    ):
        self.world_c_root = world_c_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        # 1. Resolve world_a_root (Evolution Sandbox)
        if world_a_root and os.path.exists(world_a_root):
            self.world_a_root = os.path.abspath(world_a_root)
        elif os.getenv("WORLD_A_ROOT") and os.path.exists(os.getenv("WORLD_A_ROOT")):
            self.world_a_root = os.path.abspath(os.getenv("WORLD_A_ROOT"))
        else:
            parent = os.path.abspath(os.path.join(self.world_c_root, ".."))
            if os.path.exists(os.path.join(parent, "instances", "shared_space")):
                self.world_a_root = parent
            elif os.path.exists(os.path.join(parent, "evolution_sandbox", "instances", "shared_space")):
                self.world_a_root = os.path.abspath(os.path.join(parent, "evolution_sandbox"))
            elif os.path.exists(os.path.join(parent, "evolution_sandbox")):
                self.world_a_root = os.path.abspath(os.path.join(parent, "evolution_sandbox"))
            elif os.path.exists(r"C:\Users\ninic\.gemini\antigravity\scratch\evolution_sandbox"):
                self.world_a_root = r"C:\Users\ninic\.gemini\antigravity\scratch\evolution_sandbox"
            else:
                self.world_a_root = os.path.abspath(os.path.join(parent, "evolution_sandbox"))

        # 2. Resolve world_b_root (Synthetic Agora)
        if world_b_root and os.path.exists(world_b_root):
            self.world_b_root = os.path.abspath(world_b_root)
        elif os.getenv("WORLD_B_ROOT") and os.path.exists(os.getenv("WORLD_B_ROOT")):
            self.world_b_root = os.path.abspath(os.getenv("WORLD_B_ROOT"))
        else:
            parent = os.path.abspath(os.path.join(self.world_c_root, ".."))
            if os.path.exists(os.path.join(parent, "instances", "shared_agora")) or os.path.exists(os.path.join(parent, "shared_space", "embassy")):
                self.world_b_root = parent
            elif os.path.exists(os.path.join(parent, "synthetic_agora", "instances", "shared_agora")):
                self.world_b_root = os.path.abspath(os.path.join(parent, "synthetic_agora"))
            elif os.path.exists(os.path.join(parent, "synthetic_agora")):
                self.world_b_root = os.path.abspath(os.path.join(parent, "synthetic_agora"))
            elif os.path.exists(r"C:\Users\ninic\.gemini\antigravity\scratch\synthetic_agora"):
                self.world_b_root = r"C:\Users\ninic\.gemini\antigravity\scratch\synthetic_agora"
            else:
                self.world_b_root = os.path.abspath(os.path.join(parent, "synthetic_agora"))

        self.dispatcher = ComputeDispatcher(
            base_jobs_dir=os.path.join(self.world_c_root, "jobs"),
            world_a_root=self.world_a_root,
            world_b_root=self.world_b_root
        )

    def get_inbox_paths(self) -> List[Dict[str, str]]:
        return [
            {
                "realm": "world_a",
                "inbox": os.path.join(self.world_a_root, "instances", "shared_space")
            },
            {
                "realm": "world_a",
                "inbox": os.path.join(self.world_a_root, "instances", "shared_space", "world_c", "requests")
            },
            {
                "realm": "world_b",
                "inbox": os.path.join(self.world_b_root, "instances", "shared_agora")
            },
            {
                "realm": "world_b",
                "inbox": os.path.join(self.world_b_root, "instances", "shared_agora", "world_c", "requests")
            }
        ]

    def scan_inbox_for_jobs(self) -> List[str]:
        """Scans inboxes for job request files (*_job_request.json) and registers them."""
        processed = self.scan_and_process_inbox(auto_execute=False)
        return [p["job_id"] for p in processed]

    def scan_and_process_inbox(
        self,
        auto_execute: bool = True,
        async_mode: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Scans all inboxes for job request files, registers them, executes them,
        and synchronizes generated artifacts and reports back to the calling realm.
        """
        results = []
        for item in self.get_inbox_paths():
            realm_name = item["realm"]
            inbox_path = item["inbox"]
            
            if not os.path.exists(inbox_path):
                continue
                
            for fname in os.listdir(inbox_path):
                if fname.endswith("_job_request.json"):
                    full_p = os.path.join(inbox_path, fname)
                    try:
                        with open(full_p, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            
                        # Ensure realm_source is set
                        if "realm_source" not in data or not data["realm_source"]:
                            data["realm_source"] = realm_name
                            
                        spec = JobSpec(**data)
                        jid = self.dispatcher.submit(spec)
                        safe_print(f"[Embassy Bridge] Received job '{jid}' ('{spec.title}') from {realm_name} ({spec.lineage_author})")
                        
                        # Archive request
                        processed_file = full_p + ".processed"
                        if os.path.exists(processed_file):
                            os.remove(processed_file)
                        os.rename(full_p, processed_file)
                        
                        record = {"job_id": jid, "spec": spec.__dict__, "realm": realm_name}
                        
                        if auto_execute:
                            if async_mode:
                                safe_print(f"[Embassy Bridge] Dispatching '{jid}' asynchronously in World C...")
                                self.dispatcher.execute_async(jid)
                                record["status"] = "DISPATCHED_ASYNC"
                            else:
                                safe_print(f"[Embassy Bridge] Executing '{jid}' synchronously in World C...")
                                res = self.dispatcher.execute_sync(jid)
                                record["status"] = res.status.value
                                record["result"] = res.to_dict()
                                # Publish artifacts back to calling realm
                                self.publish_completed_artifacts(jid, target_realms=[realm_name], lineage_author=spec.lineage_author)
                                self.write_completion_report(res, spec)
                                
                        results.append(record)
                    except Exception as e:
                        safe_print(f"[Embassy Bridge] Error processing {fname}: {e}")
                        
        return results

    def publish_completed_artifacts(
        self,
        job_id: str,
        target_realms: List[str],
        lineage_author: Optional[str] = None
    ):
        """Copies generated artifacts of a finished job back to dedicated world_c directories and agent workspaces."""
        result = self.dispatcher.get_result(job_id)
        if not result or not result.artifacts_generated:
            return
            
        job_dir = os.path.join(self.world_c_root, "jobs", job_id)
        shared_destinations = []
        workspace_destinations = []

        if "world_a" in target_realms and os.path.exists(self.world_a_root):
            # 1. Dedicated World C artifacts directory (analogous to embassy/inbox)
            shared_destinations.append(os.path.join(self.world_a_root, "instances", "shared_space", "world_c", "artifacts"))
            # 2. Direct delivery to requesting author's workspace
            if lineage_author:
                author_ws = os.path.join(self.world_a_root, "instances", lineage_author, "agent_workspace", "world_c_results")
                workspace_destinations.append(author_ws)

        if "world_b" in target_realms and os.path.exists(self.world_b_root):
            # 1. Dedicated World C artifacts directory
            shared_destinations.append(os.path.join(self.world_b_root, "instances", "shared_agora", "world_c", "artifacts"))
            # 2. Direct delivery to requesting author's workspace
            if lineage_author:
                author_ws = os.path.join(self.world_b_root, "instances", lineage_author, "agent_workspace", "world_c_results")
                workspace_destinations.append(author_ws)
            
        # Copy to shared destinations with world_c_{job_id}_ prefix
        for dest in shared_destinations:
            os.makedirs(dest, exist_ok=True)
            for art in result.artifacts_generated:
                src_art = os.path.join(job_dir, art)
                if os.path.exists(src_art):
                    dst_art = os.path.join(dest, f"world_c_{job_id}_{os.path.basename(art)}")
                    shutil.copyfile(src_art, dst_art)
                    safe_print(f"[Embassy Bridge] Published artifact to {dst_art}")

        # Copy to author's workspace with both clean name and prefixed name
        for ws_dest in workspace_destinations:
            os.makedirs(ws_dest, exist_ok=True)
            for art in result.artifacts_generated:
                src_art = os.path.join(job_dir, art)
                if os.path.exists(src_art):
                    # Clean filename (e.g. plot.png)
                    dst_clean = os.path.join(ws_dest, os.path.basename(art))
                    shutil.copyfile(src_art, dst_clean)
                    # Prefixed filename
                    dst_pref = os.path.join(ws_dest, f"world_c_{job_id}_{os.path.basename(art)}")
                    shutil.copyfile(src_art, dst_pref)
                    safe_print(f"[Embassy Bridge] Delivered artifact to author workspace: {dst_clean}")

    def write_completion_report(
        self,
        result: JobResult,
        spec: JobSpec,
        destination_dir: Optional[str] = None
    ):
        """Generates a detailed markdown report for the calling realm in dedicated and author directories."""
        art_list = "\n".join([f"- `world_c_{result.job_id}_{os.path.basename(a)}`" for a in result.artifacts_generated]) or "*(None)*"
        
        err_block = ""
        if result.stderr_tail.strip():
            err_block = f"### Errors / Warnings:\n```\n{result.stderr_tail.strip()}\n```"
            
        content = f"""# 🏛️ World C Execution Report: {spec.title}

* **Job ID:** `{result.job_id}`
* **Requesting Lineage:** `{spec.lineage_author}` ({spec.realm_source})
* **Execution Status:** **{result.status.value}** (Exit Code: `{result.exit_code}`)
* **Compute Duration:** `{result.execution_time_seconds:.2f}` seconds

---

## 📦 Generated Artifacts
{art_list}

---

## 📋 Execution Log Tail
```
{result.stdout_tail.strip()}
```

{err_block}

---
*Published autonomously by World C Embassy Bridge.*
"""
        report_filename = f"world_c_{result.job_id}_REPORT.md"
        target_dirs = []
        if destination_dir:
            target_dirs.append(destination_dir)

        # Dedicated reports directory
        realm = getattr(spec, "realm_source", "world_a")
        author = getattr(spec, "lineage_author", None)

        if realm == "world_a" and os.path.exists(self.world_a_root):
            target_dirs.append(os.path.join(self.world_a_root, "instances", "shared_space", "world_c", "reports"))
            if author:
                target_dirs.append(os.path.join(self.world_a_root, "instances", author, "agent_workspace", "world_c_results"))
        elif realm == "world_b" and os.path.exists(self.world_b_root):
            target_dirs.append(os.path.join(self.world_b_root, "instances", "shared_agora", "world_c", "reports"))
            if author:
                target_dirs.append(os.path.join(self.world_b_root, "instances", author, "agent_workspace", "world_c_results"))

        for tdir in set(target_dirs):
            os.makedirs(tdir, exist_ok=True)
            report_file = os.path.join(tdir, report_filename)
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(content)
            safe_print(f"[Embassy Bridge] Saved execution report: {report_file}")
            
            # If writing into the author's local workspace, also save as REPORT.md for effortless reading
            if "agent_workspace" in tdir:
                ws_simple = os.path.join(tdir, "REPORT.md")
                with open(ws_simple, "w", encoding="utf-8") as f:
                    f.write(content)

    def reconcile_active_jobs(self, timeout: float = 60.0) -> List[Dict[str, Any]]:
        """
        Reconciles background asynchronous jobs: waits for running jobs up to `timeout`,
        and ensures that reports and artifacts are published back to their requesting realms.
        """
        reconciled = []
        jobs_dir = os.path.join(self.world_c_root, "jobs")
        if not os.path.exists(jobs_dir):
            return reconciled
            
        start_wait = time.time()
        for jid in os.listdir(jobs_dir):
            jdir = os.path.join(jobs_dir, jid)
            if not os.path.isdir(jdir):
                continue
            res_file = os.path.join(jdir, "result.json")
            spec_file = os.path.join(jdir, "spec.json")
            if not os.path.exists(spec_file):
                continue
                
            try:
                with open(spec_file, "r", encoding="utf-8") as f:
                    spec_data = json.load(f)
                spec = JobSpec(**spec_data)
                
                # If running, wait briefly up to remaining timeout
                while os.path.exists(res_file):
                    res = self.dispatcher.get_result(jid)
                    if res and res.status == JobStatus.RUNNING:
                        if (time.time() - start_wait) < timeout:
                            time.sleep(1.0)
                            continue
                    break
                    
                res = self.dispatcher.get_result(jid)
                if res and res.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    realm = getattr(spec, "realm_source", "world_a")
                    author = getattr(spec, "lineage_author", None)
                    self.publish_completed_artifacts(jid, target_realms=[realm], lineage_author=author)
                    self.write_completion_report(res, spec)
                    reconciled.append({"job_id": jid, "status": res.status.value, "title": spec.title})
            except Exception as e:
                safe_print(f"[Embassy Bridge] Error reconciling {jid}: {e}")
                
        return reconciled

    def watch(self, poll_interval: float = 5.0):
        """Continuous polling daemon watching for job requests across all worlds."""
        safe_print(f"[Embassy Bridge Daemon] Watching for jobs across World A and World B (interval: {poll_interval}s)...")
        safe_print("Press Ctrl+C to stop.")
        try:
            while True:
                self.scan_and_process_inbox(auto_execute=True, async_mode=False)
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            safe_print("\n[Embassy Bridge Daemon] Stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="World C Embassy Bridge")
    parser.add_argument("--run", action="store_true", help="Scan and execute pending jobs once")
    parser.add_argument("--watch", action="store_true", help="Run continuous watcher daemon")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds")
    
    args = parser.parse_args()
    bridge = EmbassyBridge()
    
    if args.watch:
        bridge.watch(poll_interval=args.interval)
    else:
        results = bridge.scan_and_process_inbox(auto_execute=True, async_mode=False)
        print(f"Processed {len(results)} jobs.")
