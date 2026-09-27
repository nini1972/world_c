"""
State Checkpoint Manager
Saves and restores long-running simulation states, allowing parameter sweeps
to withstand interruptions or compute preemption.
"""

import os
import json
import time
from typing import Dict, Any, Optional

class CheckpointManager:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.state_file = os.path.join(checkpoint_dir, "checkpoint_state.json")
        self.last_saved_time = 0.0

    def save(self, iteration: int, state: Dict[str, Any], force: bool = False, min_interval: float = 30.0) -> bool:
        """Saves current state if force=True or if min_interval has elapsed."""
        now = time.time()
        if not force and (now - self.last_saved_time < min_interval):
            return False
            
        payload = {
            "iteration": iteration,
            "timestamp": now,
            "state": state
        }
        temp_file = self.state_file + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(temp_file, self.state_file)
        self.last_saved_time = now
        return True

    def load(self) -> Optional[Dict[str, Any]]:
        """Loads latest checkpoint if available."""
        if not os.path.exists(self.state_file):
            return None
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def exists(self) -> bool:
        return os.path.exists(self.state_file)
