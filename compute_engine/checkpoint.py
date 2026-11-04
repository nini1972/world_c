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
        self.checkpoint_dir = self.job_dir
        self.checkpoint_file = os.path.join(self.job_dir, "checkpoint_state.json")
        self.state_file = self.checkpoint_file
        self.arrays_file = os.path.join(self.job_dir, "checkpoint_arrays.npz")
        self.last_saved_time = 0.0
        self._tracker = StageTracker(
            stage_name=stage_name or os.environ.get("WORLD_C_STAGE_NAME", "checkpoint"),
            checkpoint_dir=self.job_dir,
            register_signals=False,
        )

    def save(self, *args, **kwargs) -> bool:
        """Atomically saves simulation checkpoint state (including NumPy arrays) to disk and stage mirror."""
        # Handle flexible arguments: save(state, step=...) or save(iteration, state, force=...)
        state = None
        step = None
        force = kwargs.get("force", False)
        min_interval = kwargs.get("min_interval", 0.0)

        if "iteration" in kwargs:
            step = kwargs["iteration"]
        if "step" in kwargs:
            step = kwargs["step"]
        if "state" in kwargs:
            state = kwargs["state"]

        if args:
            if isinstance(args[0], (int, float)):
                step = int(args[0])
                if len(args) > 1:
                    state = args[1]
            elif isinstance(args[0], dict):
                state = args[0]
                if len(args) > 1:
                    step = args[1]

        if state is None:
            state = {}

        now = time.time()
        if not force and min_interval > 0.0 and (now - self.last_saved_time < min_interval):
            return False

        os.makedirs(self.job_dir, exist_ok=True)
        # Also save via StageTracker so arrays and persistent stage mirrors are updated
        try:
            self._tracker.save_checkpoint(state=state, step=step)
        except Exception:
            pass

        payload = {
            "iteration": step,
            "step": step,
            "timestamp": now,
            "state": state,
        }
        temp_file = self.checkpoint_file + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(temp_file, self.checkpoint_file)
        self.last_saved_time = now
        return True

    def load(self) -> Optional[Dict[str, Any]]:
        """Loads the latest checkpoint state if available."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data
            except Exception:
                pass
        ckpt = self._tracker.load_checkpoint(default=None)
        if ckpt is not None:
            if isinstance(ckpt, dict) and "state" in ckpt:
                return ckpt
            return {"state": ckpt, "step": None, "iteration": None}
        return None

    def exists(self) -> bool:
        return self.has_checkpoint()

    def has_checkpoint(self) -> bool:
        return os.path.exists(self.checkpoint_file) or self._tracker.has_checkpoint()
