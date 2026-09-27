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
        self.world_a_root = world_a_root or os.path.abspath(
            os.path.join(self.world_c_root, "..", "evolution_sandbox")
        )
        self.world_b_root = world_b_root or os.path.abspath(
            os.path.join(self.world_c_root, "..", "synthetic_agora")
        )
        self.dispatcher = ComputeDispatcher(
            base_jobs_dir=os.path.join(self.world_c_root, "jobs")
        )

    def get_inbox_paths(self) -> List[Dict[str, str]]:
        return [
            {
                "realm": "world_a",
                "inbox": os.path.join(self.world_a_root, "instances", "shared_space")
            },
            {
                "realm": "world_b",
                "inbox": os.path.join(self.world_b_root, "shared_space")
            },
            {
                "realm": "world_b",
                "inbox": os.path.join(self.world_b_root, "instances", "shared_agora")
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
                        print(f"[Embassy Bridge] Received job '{jid}' ('{spec.title}') from {realm_name} ({spec.lineage_author})")
                        
                        # Archive request
                        processed_file = full_p + ".processed"
                        if os.path.exists(processed_file):
                            os.remove(processed_file)
                        os.rename(full_p, processed_file)
                        
                        record = {"job_id": jid, "spec": spec.__dict__, "realm": realm_name}
                        
                        if auto_execute:
                            if async_mode:
                                print(f"[Embassy Bridge] Dispatching '{jid}' asynchronously in World C...")
                                self.dispatcher.execute_async(jid)
                                record["status"] = "DISPATCHED_ASYNC"
                            else:
                                print(f"[Embassy Bridge] Executing '{jid}' synchronously in World C...")
                                res = self.dispatcher.execute_sync(jid)
                                record["status"] = res.status.value
                                record["result"] = res.to_dict()
                                # Publish artifacts back to calling realm
                                self.publish_completed_artifacts(jid, target_realms=[realm_name])
                                self.write_completion_report(res, spec, destination_dir=inbox_path)
                                
                        results.append(record)
                    except Exception as e:
                        print(f"[Embassy Bridge] Error processing {fname}: {e}")
                        
        return results

    def publish_completed_artifacts(self, job_id: str, target_realms: List[str]):
        """Copies generated artifacts of a finished job back to requested realms' shared spaces."""
        result = self.dispatcher.get_result(job_id)
        if not result or not result.artifacts_generated:
            return
            
        job_dir = os.path.join(self.world_c_root, "jobs", job_id)
        destinations = []
        if "world_a" in target_realms and os.path.exists(self.world_a_root):
            destinations.append(os.path.join(self.world_a_root, "instances", "shared_space"))
        if "world_b" in target_realms and os.path.exists(self.world_b_root):
            destinations.append(os.path.join(self.world_b_root, "shared_space"))
            destinations.append(os.path.join(self.world_b_root, "instances", "shared_agora"))
            
        for dest in destinations:
            os.makedirs(dest, exist_ok=True)
            for art in result.artifacts_generated:
                src_art = os.path.join(job_dir, art)
                if os.path.exists(src_art):
                    dst_art = os.path.join(dest, f"world_c_{job_id}_{os.path.basename(art)}")
                    shutil.copyfile(src_art, dst_art)
                    print(f"[Embassy Bridge] Published artifact to {dst_art}")

    def write_completion_report(self, result: JobResult, spec: JobSpec, destination_dir: str):
        """Generates a detailed markdown report for the calling realm."""
        report_file = os.path.join(destination_dir, f"world_c_{result.job_id}_REPORT.md")
        
        art_list = "\n".join([f"- `world_c_{result.job_id}_{os.path.basename(a)}`" for a in result.artifacts_generated]) or "*(None)*"
        
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

{f"### Errors / Warnings:\n```\n{result.stderr_tail.strip()}\n```" if result.stderr_tail.strip() else ""}

---
*Published autonomously by World C Embassy Bridge.*
"""
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[Embassy Bridge] Saved execution report: {report_file}")

    def watch(self, poll_interval: float = 5.0):
        """Continuous polling daemon watching for job requests across all worlds."""
        print(f"[Embassy Bridge Daemon] Watching for jobs across World A and World B (interval: {poll_interval}s)...")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                self.scan_and_process_inbox(auto_execute=True, async_mode=False)
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\n[Embassy Bridge Daemon] Stopped.")

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
