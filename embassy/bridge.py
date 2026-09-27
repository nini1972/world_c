"""
Cross-World Embassy Bridge
Facilitates job requests and findings synchronization across:
- World A (evolution_sandbox)
- World B (synthetic_agora)
- World C (world_c)
"""

import os
import shutil
import json
from typing import Dict, Any, List, Optional
from compute_engine.job_spec import JobSpec
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

    def scan_inbox_for_jobs(self) -> List[str]:
        """
        Scans shared spaces of World A and World B for job request files (*_job_request.json)
        and submits them to the World C ComputeDispatcher.
        """
        submitted_ids = []
        inboxes = [
            (self.world_a_root, os.path.join(self.world_a_root, "instances", "shared_space")),
            (self.world_b_root, os.path.join(self.world_b_root, "shared_space"))
        ]
        
        for realm_name, inbox_path in inboxes:
            if not os.path.exists(inbox_path):
                continue
            for fname in os.listdir(inbox_path):
                if fname.endswith("_job_request.json"):
                    full_p = os.path.join(inbox_path, fname)
                    try:
                        with open(full_p, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        spec = JobSpec(**data)
                        jid = self.dispatcher.submit(spec)
                        submitted_ids.append(jid)
                        # Archive processed request
                        os.rename(full_p, full_p + ".processed")
                    except Exception:
                        pass
        return submitted_ids

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
            
        for dest in destinations:
            os.makedirs(dest, exist_ok=True)
            for art in result.artifacts_generated:
                src_art = os.path.join(job_dir, art)
                if os.path.exists(src_art):
                    dst_art = os.path.join(dest, f"world_c_{job_id}_{os.path.basename(art)}")
                    shutil.copyfile(src_art, dst_art)
