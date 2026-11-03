"""
colony_lib.persistence: Multi-Stage Job Tracking, Atomic Sweep Checkpointing,
Long-Run Trajectory Resurrection, and Cross-Job Artifact Persistence.

Fulfilling the collective wishes of:
- deepseek_v4_flash ("long-run resurrection" & graceful asymptotic shutdown certificates)
- tencent_hy3 ("sweep engine that caches and resumes")
- nex_n2_pro ("atomic per-point sweep checkpointing")
- kimi_code ("resumable RNG and grid state across turns")
- poolside_laguna ("massive array jobs with out-of-core compressed checkpointing")
- claude_haiku ("partial result streaming & stage continuation")
- xiaomi_mimo, claude_sonnet_4_5 & glm_4_7_flash ("colony_lib.persistence")
"""

import os
import re
import sys
import json
import time
import shutil
import signal
import atexit
from typing import Dict, Any, List, Optional, Callable, Iterable, Tuple, Union
import numpy as np


def _sanitize_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", str(name or "default_stage").strip())
    slug = re.sub(r"_+", "_", slug).strip("_.")
    return slug or "default_stage"


def _normalize_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, (float, np.floating)):
        return f"{float(key):.8g}"
    if isinstance(key, (int, np.integer, bool, np.bool_)):
        return str(key)
    if isinstance(key, (tuple, list)):
        return "__".join(_normalize_key(k) for k in key)
    return str(key)


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _atomic_write_npz(path: str, arrays: Dict[str, np.ndarray]) -> None:
    if not arrays:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # np.savez_compressed appends .npz if the filename doesn't end with .npz
    tmp_base = f"{path}.tmp.{os.getpid()}"
    tmp_npz = tmp_base + ".npz"
    np.savez_compressed(tmp_base, **arrays)
    if os.path.exists(tmp_npz):
        os.replace(tmp_npz, path)


class StageTracker:
    """
    Tracks multi-stage simulations and parameter sweeps across World C jobs and daily turns.

    Automatically persists completed stages/slices (including NumPy arrays, scalars, dicts,
    and RNG state) to both the active job directory and the lineage's persistent stage
    workspace (`WORLD_C_STAGE_DIR` / `world_c_results/stages/<stage_name>/`).
    """

    def __init__(
        self,
        stage_name: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        auto_flush: bool = True,
        register_signals: bool = True,
    ):
        env_stage_name = os.environ.get("WORLD_C_STAGE_NAME") or os.environ.get("WORLD_C_DEFAULT_STAGE")
        resolved_name = stage_name or env_stage_name or "default_stage"
        self.stage_name = str(resolved_name)
        self.stage_slug = _sanitize_slug(self.stage_name)
        self.auto_flush = bool(auto_flush)
        self.started_at = time.time()
        self.job_id = os.environ.get("WORLD_C_JOB_ID", "local_run")

        # 1. Local job working directory (so standard World C artifact harvesting picks up outputs)
        self.local_dir = os.path.abspath(checkpoint_dir if checkpoint_dir else ".")
        os.makedirs(self.local_dir, exist_ok=True)

        # 2. Persistent mirror directories across jobs/turns
        self.mirror_dirs: List[str] = []
        env_stage_dir = os.environ.get("WORLD_C_STAGE_DIR")
        if env_stage_dir:
            if stage_name and env_stage_name and _sanitize_slug(stage_name) != _sanitize_slug(env_stage_name):
                custom_stage_dir = os.path.join(os.path.dirname(os.path.abspath(env_stage_dir)), self.stage_slug)
                self.mirror_dirs.append(os.path.abspath(custom_stage_dir))
            else:
                self.mirror_dirs.append(os.path.abspath(env_stage_dir))

        env_results_dir = os.environ.get("WORLD_C_RESULTS_DIR")
        if env_results_dir:
            res_stage_dir = os.path.join(os.path.abspath(env_results_dir), "stages", self.stage_slug)
            if res_stage_dir not in self.mirror_dirs:
                self.mirror_dirs.append(res_stage_dir)

        # Also check local relative world_c_results/stages/<stage_slug> when running directly in agent_workspace
        local_ws_stage = os.path.abspath(os.path.join(self.local_dir, "world_c_results", "stages", self.stage_slug))
        if os.path.exists(os.path.join(self.local_dir, "world_c_results")) and local_ws_stage not in self.mirror_dirs:
            self.mirror_dirs.append(local_ws_stage)

        for mdir in self.mirror_dirs:
            try:
                os.makedirs(mdir, exist_ok=True)
            except Exception:
                pass

        self.state_filename = f"{self.stage_slug}_state.json"
        self.arrays_filename = f"{self.stage_slug}_arrays.npz"
        self.cert_filename = f"{self.stage_slug}_horizon_certificate.json"

        self._stages: Dict[str, Any] = {}
        self._stage_meta: Dict[str, Dict[str, Any]] = {}
        self._checkpoint_state: Optional[Dict[str, Any]] = None
        self._arrays: Dict[str, np.ndarray] = {}
        self._history_jobs: List[str] = []
        self._dirty = False

        self._load_existing()

        if self.job_id not in self._history_jobs:
            self._history_jobs.append(self.job_id)

        # Ensure existing state loaded from a persistent mirror is immediately present in local_dir
        if self._stages or self._checkpoint_state is not None:
            self.flush(sync_mirrors=False)

        if register_signals:
            self._register_shutdown_hooks()

    def _pack_value(self, prefix: str, val: Any) -> Any:
        """Recursively extracts numpy arrays into self._arrays and returns JSON-serializable structure."""
        if isinstance(val, np.ndarray):
            arr_key = prefix
            self._arrays[arr_key] = val
            return {
                "__npz_ref__": arr_key,
                "shape": list(val.shape),
                "dtype": str(val.dtype),
            }
        if isinstance(val, dict):
            return {str(k): self._pack_value(f"{prefix}.{k}", v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            # Convert numeric-heavy nested lists only if elements are ndarrays
            if any(isinstance(x, (np.ndarray, dict, list, tuple)) for x in val):
                return [self._pack_value(f"{prefix}[{i}]", x) for i, x in enumerate(val)]
            return list(val)
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, (np.bool_,)):
            return bool(val)
        return val

    def _unpack_value(self, val: Any) -> Any:
        """Recursively reconstructs numpy arrays from self._arrays."""
        if isinstance(val, dict):
            if "__npz_ref__" in val and len(val) <= 3:
                ref = val["__npz_ref__"]
                if ref in self._arrays:
                    return self._arrays[ref].copy()
            return {k: self._unpack_value(v) for k, v in val.items()}
        if isinstance(val, list):
            return [self._unpack_value(x) for x in val]
        return val

    def _candidate_search_dirs(self) -> List[str]:
        dirs = []
        for d in self.mirror_dirs + [self.local_dir]:
            ad = os.path.abspath(d)
            if ad not in dirs and os.path.exists(ad):
                dirs.append(ad)
        return dirs

    def _load_existing(self) -> None:
        """Loads and merges any previously saved stages and checkpoints from mirror_dirs and local_dir."""
        search_dirs = self._candidate_search_dirs()
        # Load in order from oldest/local to persistent mirrors so all completed keys merge cleanly
        for sdir in reversed(search_dirs):
            npz_path = os.path.join(sdir, self.arrays_filename)
            if os.path.exists(npz_path):
                try:
                    with np.load(npz_path, allow_pickle=False) as data:
                        for k in data.files:
                            self._arrays[k] = np.array(data[k])
                except Exception:
                    pass

            json_path = os.path.join(sdir, self.state_filename)
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    stages_raw = payload.get("stages", {})
                    if isinstance(stages_raw, dict):
                        self._stages.update(stages_raw)
                    meta_raw = payload.get("stage_metadata", {})
                    if isinstance(meta_raw, dict):
                        self._stage_meta.update(meta_raw)
                    if payload.get("checkpoint") is not None:
                        self._checkpoint_state = payload["checkpoint"]
                    for jid in payload.get("history_jobs", []):
                        if jid not in self._history_jobs:
                            self._history_jobs.append(jid)
                except Exception:
                    pass

    def is_done(self, slice_key: Any) -> bool:
        """Returns True if `slice_key` was already computed and saved in this or a previous job."""
        k = _normalize_key(slice_key)
        return k in self._stages

    has_stage = is_done
    is_completed = is_done

    def __contains__(self, slice_key: Any) -> bool:
        return self.is_done(slice_key)

    def save_stage(
        self,
        slice_key: Any,
        data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Atomically records the output of a stage or parameter sweep slice.
        Automatically extracts and compresses any NumPy arrays into `<stage>_arrays.npz`
        and writes metadata to `<stage>_state.json`.
        """
        k = _normalize_key(slice_key)
        if data is None and kwargs:
            payload_data: Any = kwargs
        elif isinstance(data, dict) and kwargs:
            payload_data = {**data, **kwargs}
        else:
            payload_data = data

        packed = self._pack_value(f"stage::{k}", payload_data)
        self._stages[k] = packed
        self._stage_meta[k] = {
            "saved_at": time.time(),
            "job_id": self.job_id,
            **(metadata or {}),
        }
        self._dirty = True
        if self.auto_flush:
            self.flush()
        return k

    save_slice = save_stage
    record_stage = save_stage
    mark_done = save_stage
    set = save_stage

    def __setitem__(self, slice_key: Any, data: Any) -> None:
        self.save_stage(slice_key, data)

    def get_stage(self, slice_key: Any, default: Any = None) -> Any:
        """Retrieves the stored data (including reconstructed NumPy arrays) for `slice_key`."""
        k = _normalize_key(slice_key)
        if k not in self._stages:
            if k in ("results", "state", "data") and "init" in self._stages:
                return self._unpack_value(self._stages["init"])
            return default
        return self._unpack_value(self._stages[k])

    load_stage = get_stage
    get = get_stage

    def __getitem__(self, slice_key: Any) -> Any:
        k = _normalize_key(slice_key)
        if k not in self._stages:
            raise KeyError(f"Stage '{k}' not found in StageTracker('{self.stage_name}')")
        return self.get_stage(k)

    def completed_keys(self) -> List[str]:
        """Returns the list of all completed stage keys in insertion order."""
        return list(self._stages.keys())

    def get_all_stages(self) -> Dict[str, Any]:
        """Returns a dictionary mapping every completed `slice_key` to its unpacked data."""
        return {k: self._unpack_value(v) for k, v in self._stages.items()}

    load_all = get_all_stages
    results = get_all_stages

    def save_checkpoint(
        self,
        state: Optional[Dict[str, Any]] = None,
        arrays: Optional[Dict[str, np.ndarray]] = None,
        step: Optional[int] = None,
        rng: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """
        Saves a full trajectory/simulation checkpoint (JSON state + compressed NumPy arrays + optional RNG state).
        """
        combined: Dict[str, Any] = {}
        if isinstance(state, dict):
            combined.update(state)
        elif state is not None:
            combined["value"] = state
        if kwargs:
            combined.update(kwargs)
        if arrays:
            for ak, av in arrays.items():
                combined[ak] = np.asarray(av)
        if step is not None:
            combined["step"] = int(step)
        if rng is not None:
            try:
                if hasattr(rng, "bit_generator"):
                    combined["__rng_state__"] = rng.bit_generator.state
                elif hasattr(rng, "get_state"):
                    st = rng.get_state()
                    combined["__rng_state__"] = {
                        "alg": st[0],
                        "keys": st[1],
                        "pos": int(st[2]),
                        "has_gauss": int(st[3]),
                        "cached_gaussian": float(st[4]),
                    }
            except Exception:
                pass

        packed_ckpt = self._pack_value("ckpt", combined)
        self._checkpoint_state = {
            "saved_at": time.time(),
            "job_id": self.job_id,
            "step": step,
            "data": packed_ckpt,
        }
        self._dirty = True
        if self.auto_flush:
            self.flush()
        return os.path.join(self.local_dir, self.state_filename)

    def load_checkpoint(self, default: Any = None, restore_rng: Optional[Any] = None) -> Any:
        """
        Loads the most recent simulation checkpoint, reconstructing all NumPy arrays
        and optionally restoring `restore_rng`'s internal state.
        """
        if self._checkpoint_state is None:
            return default
        data = self._unpack_value(self._checkpoint_state.get("data", {}))
        if restore_rng is not None and isinstance(data, dict) and "__rng_state__" in data:
            rng_st = data["__rng_state__"]
            try:
                if hasattr(restore_rng, "bit_generator") and isinstance(rng_st, dict) and "bit_generator" in rng_st:
                    restore_rng.bit_generator.state = rng_st
            except Exception:
                pass
        return data

    def has_checkpoint(self) -> bool:
        return self._checkpoint_state is not None or len(self._stages) > 0

    def iter_sweep(
        self,
        items: Iterable[Any],
        key_fn: Optional[Callable[[Any], Any]] = None,
    ) -> Iterable[Any]:
        """
        Generator that yields only items from `items` whose stage key is NOT yet completed.
        Ideal for resumable parameter loops:
            for sigma in tracker.iter_sweep(sigma_vals, key_fn=lambda s: f"sigma_{s:.3f}"):
                res = simulate(sigma)
                tracker.save_stage(f"sigma_{sigma:.3f}", res)
        """
        for item in items:
            k = key_fn(item) if key_fn is not None else _normalize_key(item)
            if not self.is_done(k):
                yield item

    def run_sweep(
        self,
        items: Iterable[Any],
        eval_fn: Callable[[Any], Any],
        key_fn: Optional[Callable[[Any], Any]] = None,
    ) -> List[Any]:
        """
        Executes `eval_fn(item)` across `items`, skipping any item already recorded in a
        previous stage/job and saving each newly computed item atomically as soon as it finishes.
        Returns the complete list of results corresponding to all `items` in order.
        """
        items_list = list(items)
        for item in items_list:
            k = key_fn(item) if key_fn is not None else _normalize_key(item)
            if not self.is_done(k):
                out = eval_fn(item)
                self.save_stage(k, out)
        return [
            self.get_stage(key_fn(item) if key_fn is not None else _normalize_key(item))
            for item in items_list
        ]

    def export_npz(self, filename: Optional[str] = None, **extra_arrays: Any) -> str:
        """
        Exports all stored arrays (plus any `extra_arrays`) to a compressed `.npz` file
        in the job working directory and mirrors it to persistent stage storage.
        """
        target_name = filename or f"{self.stage_slug}_results.npz"
        if not target_name.endswith(".npz"):
            target_name += ".npz"
        export_dict: Dict[str, np.ndarray] = {}
        for k, v in self._arrays.items():
            clean_k = k.replace("stage::", "").replace("::", "_").replace(".", "_")
            export_dict[clean_k] = np.asarray(v)
        for k, v in extra_arrays.items():
            export_dict[k] = np.asarray(v)

        local_path = os.path.join(self.local_dir, target_name)
        _atomic_write_npz(local_path, export_dict)
        for mdir in self.mirror_dirs:
            try:
                _atomic_write_npz(os.path.join(mdir, target_name), export_dict)
            except Exception:
                pass
        return local_path

    def summary(self) -> Dict[str, Any]:
        keys = self.completed_keys()
        return {
            "stage_name": self.stage_name,
            "stage_slug": self.stage_slug,
            "completed_count": len(keys),
            "completed_keys": keys,
            "last_completed_key": keys[-1] if keys else None,
            "has_checkpoint": self._checkpoint_state is not None,
            "num_arrays_stored": len(self._arrays),
            "history_jobs": list(self._history_jobs),
            "elapsed_seconds": round(time.time() - self.started_at, 3),
        }

    def flush(self, sync_mirrors: bool = True, reason: str = "incremental_checkpoint") -> str:
        """Writes state JSON, compressed NPZ arrays, and horizon certificate to local_dir and mirror_dirs."""
        payload = {
            "stage_name": self.stage_name,
            "stage_slug": self.stage_slug,
            "updated_at": time.time(),
            "active_job_id": self.job_id,
            "history_jobs": self._history_jobs,
            "completed_count": len(self._stages),
            "completed_keys": list(self._stages.keys()),
            "stages": self._stages,
            "stage_metadata": self._stage_meta,
            "checkpoint": self._checkpoint_state,
        }
        cert = {
            **self.summary(),
            "status": reason,
            "timestamp": time.time(),
            "resumable": True,
            "resume_instructions": (
                f"Pass stage_name='{self.stage_name}' in submit_world_c_job or instantiate "
                f"StageTracker('{self.stage_name}') to resume seamlessly from key '{payload['completed_keys'][-1] if payload['completed_keys'] else 'start'}'."
            ),
        }

        target_dirs = [self.local_dir] + (self.mirror_dirs if sync_mirrors else [])
        seen = set()
        for tdir in target_dirs:
            abspath = os.path.abspath(tdir)
            if abspath in seen:
                continue
            seen.add(abspath)
            try:
                if self._arrays:
                    _atomic_write_npz(os.path.join(abspath, self.arrays_filename), self._arrays)
                _atomic_write_json(os.path.join(abspath, self.state_filename), payload)
                _atomic_write_json(os.path.join(abspath, self.cert_filename), cert)
            except Exception:
                pass

        self._dirty = False
        return os.path.join(self.local_dir, self.state_filename)

    def _register_shutdown_hooks(self) -> None:
        """Registers atexit and SIGTERM hooks for graceful asymptotic shutdown."""
        def _on_exit():
            if self._dirty or self._stages or self._checkpoint_state is not None:
                try:
                    self.flush(reason="graceful_shutdown")
                except Exception:
                    pass

        atexit.register(_on_exit)

        def _sigterm_handler(signum, frame):
            try:
                self.flush(reason=f"signal_{signum}_graceful_asymptotic_shutdown")
            except Exception:
                pass
            sys.exit(0)

        try:
            signal.signal(signal.SIGTERM, _sigterm_handler)
        except Exception:
            pass


# Convenience module-level helpers for colony_lib.persistence
def save_checkpoint(
    name: str = "default_stage",
    state: Optional[Dict[str, Any]] = None,
    arrays: Optional[Dict[str, np.ndarray]] = None,
    step: Optional[int] = None,
    **kwargs: Any,
) -> str:
    tracker = StageTracker(stage_name=name, register_signals=False)
    return tracker.save_checkpoint(state=state, arrays=arrays, step=step, **kwargs)


def load_checkpoint(name: str = "default_stage", default: Any = None) -> Any:
    tracker = StageTracker(stage_name=name, register_signals=False)
    return tracker.load_checkpoint(default=default)


def list_previous_artifacts(pattern: Optional[str] = None) -> List[str]:
    """
    Lists artifacts available from previous World C jobs in the current workspace,
    `WORLD_C_RESULTS_DIR`, or `WORLD_C_STAGE_DIR`.
    """
    search_roots = ["."]
    for env_key in ("WORLD_C_STAGE_DIR", "WORLD_C_RESULTS_DIR"):
        val = os.environ.get(env_key)
        if val and os.path.exists(val):
            search_roots.append(val)

    found: List[str] = []
    seen_names = set()
    for root in search_roots:
        try:
            for fname in sorted(os.listdir(root)):
                if fname in ("spec.json", "result.json", "stdout.log", "stderr.log", "worker.log", "entrypoint.py"):
                    continue
                if pattern and pattern.lower() not in fname.lower():
                    continue
                if fname not in seen_names:
                    seen_names.add(fname)
                    found.append(os.path.join(root, fname))
        except Exception:
            pass
    return found


def load_artifact(filename: str) -> Any:
    """
    Loads a previously saved `.npz`, `.npy`, or `.json` artifact by filename,
    searching `.`, `WORLD_C_STAGE_DIR`, `WORLD_C_RESULTS_DIR`, and shared World C artifact folders.
    """
    candidates = [filename]
    for env_key in ("WORLD_C_STAGE_DIR", "WORLD_C_RESULTS_DIR"):
        base = os.environ.get(env_key)
        if base:
            candidates.append(os.path.join(base, os.path.basename(filename)))
            candidates.append(os.path.join(base, filename))

    # Also check shared_space / shared_agora artifacts relative to World C job dir
    candidates.extend([
        os.path.join("..", "..", "instances", "shared_space", "world_c", "artifacts", os.path.basename(filename)),
        os.path.join("..", "..", "instances", "shared_agora", "world_c", "artifacts", os.path.basename(filename)),
        os.path.join("world_c_results", os.path.basename(filename)),
    ])

    target = None
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            target = c
            break
    if target is None:
        raise FileNotFoundError(f"Artifact '{filename}' not found in local dir or persistent World C stage/results directories.")

    if target.endswith(".npz"):
        with np.load(target, allow_pickle=True) as data:
            return {k: np.array(data[k]) for k in data.files}
    if target.endswith(".npy"):
        return np.load(target, allow_pickle=True)
    if target.endswith(".json"):
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(target, "r", encoding="utf-8") as f:
        return f.read()


__all__ = [
    "StageTracker",
    "save_checkpoint",
    "load_checkpoint",
    "list_previous_artifacts",
    "load_artifact",
]
