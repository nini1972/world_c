"""
Checkpoint Manager for Long-Running Parameter Sweeps and Multi-Stage Simulations.
Bridges `compute_engine.checkpoint.CheckpointManager` with `colony_lib.persistence.StageTracker`
so both JSON state and NumPy arrays persist across jobs and turns.
"""

import os
import json
import time
from typing import Dict, Any, Optional
import numpy as np
from colony_lib.persistence import StageTracker


class CheckpointManager:
    def __init__(self, job_dir: str = ".", stage_name: Optional[str] = None):
        self.job_dir = os.path.abspath(job_dir)
        self.checkpoint_file = os.path.join(self.job_dir, "checkpoint_state.json")
        self.arrays_file = os.path.join(self.job_dir, "checkpoint_arrays.npz")
        self._tracker = StageTracker(
            stage_name=stage_name or os.environ.get("WORLD_C_STAGE_NAME", "checkpoint"),
            checkpoint_dir=self.job_dir,
            register_signals=False,
        )

    def save(self, state: Dict[str, Any], step: Optional[int] = None) -> str:
        """Atomically saves simulation checkpoint state (including NumPy arrays) to disk and stage mirror."""
        os.makedirs(self.job_dir, exist_ok=True)
        # Also save via StageTracker so arrays and persistent stage mirrors are updated
        self._tracker.save_checkpoint(state=state, step=step)

        payload = {
            "timestamp": time.time(),
            "step": step,
            "state": state,
        }
        temp_file = self.checkpoint_file + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(temp_file, self.checkpoint_file)
        return self.checkpoint_file

    def load(self) -> Optional[Dict[str, Any]]:
        """Loads the latest checkpoint state if available (preferring array-preserving StageTracker state)."""
        ckpt = self._tracker.load_checkpoint(default=None)
        if ckpt is not None:
            return ckpt
        if not os.path.exists(self.checkpoint_file):
            return None
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("state")
        except Exception:
            return None

    def has_checkpoint(self) -> bool:
        return os.path.exists(self.checkpoint_file) or self._tracker.has_checkpoint()
