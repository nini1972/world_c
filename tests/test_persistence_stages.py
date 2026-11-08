"""
Tests for Multi-Stage Job Tracking, Atomic Sweep Checkpointing, and Cross-Job Continuation
(`colony_lib.persistence.StageTracker`, `colony_lib.bifurcation_cartography`, and `ComputeDispatcher`).
"""

import os
import sys
import json
import shutil
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from colony_lib.persistence import StageTracker, save_checkpoint, load_checkpoint
from colony_lib import bifurcation_cartography as bc
from compute_engine.job_spec import JobSpec, JobStatus
from compute_engine.dispatcher import ComputeDispatcher


def test_stage_tracker_arrays_and_resume():
    tmp_dir = tempfile.mkdtemp(prefix="wc_stage_test_")
    try:
        # Stage 1: Compute first 2 slices of a 4-slice sweep
        t1 = StageTracker("test_kuramoto_scan", checkpoint_dir=tmp_dir, register_signals=False)
        sigma_all = [0.1, 0.2, 0.3, 0.4]
        for s in sigma_all[:2]:
            key = f"sigma_{s:.2f}"
            assert not t1.is_done(key)
            t1.save_stage(
                key,
                {
                    "sigma": s,
                    "r_mean": np.linspace(0.0, 1.0, 5) * s,
                    "r_std": np.ones(5) * 0.01,
                },
            )

        # Stage 2: New StageTracker instance in a new process/turn loads prior 2 slices and finishes remaining 2
        t2 = StageTracker("test_kuramoto_scan", checkpoint_dir=tmp_dir, register_signals=False)
        assert t2.is_done("sigma_0.10")
        assert t2.is_done("sigma_0.20")
        assert not t2.is_done("sigma_0.30")

        computed_in_stage2 = []
        for s in t2.iter_sweep(sigma_all, key_fn=lambda x: f"sigma_{x:.2f}"):
            computed_in_stage2.append(s)
            t2.save_stage(
                f"sigma_{s:.2f}",
                {
                    "sigma": s,
                    "r_mean": np.linspace(0.0, 1.0, 5) * s,
                    "r_std": np.ones(5) * 0.01,
                },
            )

        assert computed_in_stage2 == [0.3, 0.4]
        all_stages = t2.get_all_stages()
        assert len(all_stages) == 4
        assert np.allclose(all_stages["sigma_0.10"]["r_mean"], np.linspace(0.0, 1.0, 5) * 0.1)
        assert np.allclose(all_stages["sigma_0.40"]["r_mean"], np.linspace(0.0, 1.0, 5) * 0.4)

        npz_path = t2.export_npz("combined_scan.npz")
        assert os.path.exists(npz_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_scan_parameter_space_stage_caching():
    tmp_dir = tempfile.mkdtemp(prefix="wc_grid_cache_")
    old_cwd = os.getcwd()
    old_stage_dir = os.environ.get("WORLD_C_STAGE_DIR")
    os.environ["WORLD_C_STAGE_DIR"] = tmp_dir
    os.chdir(tmp_dir)
    try:
        call_Log = []

        def eval_pt(K, sigma):
            call_Log.append((round(float(K), 3), round(float(sigma), 3)))
            return {"r": float(K * sigma)}

        # Stage 1: Scan 2x2 grid
        res1 = bc.scan_parameter_space(
            eval_pt,
            param_grid={"K": [1.0, 2.0], "sigma": [0.1, 0.2]},
            stage_name="grid_stage_demo",
        )
        assert len(call_Log) == 4
        assert len(res1["r"]) == 4

        # Stage 2: Expand grid to 3x2 — the first 4 points must be served from cache without re-evaluating!
        call_Log.clear()
        res2 = bc.scan_parameter_space(
            eval_pt,
            param_grid={"K": [1.0, 2.0, 3.0], "sigma": [0.1, 0.2]},
            stage_name="grid_stage_demo",
        )
        assert len(call_Log) == 2  # Only K=3.0, sigma=0.1 and K=3.0, sigma=0.2 evaluated!
        assert len(res2["r"]) == 6
        assert np.allclose(res2["r"], [0.1, 0.2, 0.2, 0.4, 0.3, 0.6])
    finally:
        os.chdir(old_cwd)
        if old_stage_dir is None:
            os.environ.pop("WORLD_C_STAGE_DIR", None)
        else:
            os.environ["WORLD_C_STAGE_DIR"] = old_stage_dir
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_dispatcher_timeout_and_multi_stage_resumption():
    tmp_root = tempfile.mkdtemp(prefix="wc_disp_test_")
    jobs_dir = os.path.join(tmp_root, "jobs")
    world_a_dir = os.path.join(tmp_root, "world_a")
    os.makedirs(os.path.join(world_a_dir, "instances", "poolside_laguna", "agent_workspace"), exist_ok=True)
    os.makedirs(os.path.join(world_a_dir, "instances", "shared_space"), exist_ok=True)

    try:
        dispatcher = ComputeDispatcher(base_jobs_dir=jobs_dir, world_a_root=world_a_dir)

        script = """
import time
import numpy as np
from colony_lib.persistence import StageTracker

tracker = StageTracker("multi_stage_kuramoto")
slices = [0.05, 0.10, 0.15, 0.20]

for s in tracker.iter_sweep(slices, key_fn=lambda x: f"sigma_{x:.2f}"):
    arr = np.arange(10, dtype=float) * s
    tracker.save_stage(f"sigma_{s:.2f}", {"r_mean": arr, "sigma": float(s)})
    print(f"sigma={s:.2f} done", flush=True)
    time.sleep(0.9)

all_res = tracker.get_all_stages()
if len(all_res) == len(slices):
    tracker.export_npz("final_phase_diagram.npz")
    print("ALL STAGES COMPLETE!", flush=True)
"""
        # Job 1: 2-second timeout -> finishes ~2 slices then times out gracefully
        spec1 = JobSpec(
            job_id="job_stage1",
            title="Kuramoto Phase Scan Stage 1",
            lineage_author="poolside_laguna",
            realm_source="world_a",
            script_content=script,
            timeout_seconds=2,
            stage_name="multi_stage_kuramoto",
        )
        dispatcher.submit(spec1)
        res1 = dispatcher.execute_sync("job_stage1")
        assert res1.status == JobStatus.FAILED  # Timed out after completing partial slices
        assert any(a.endswith("_state.json") for a in res1.artifacts_generated)

        # Job 2: Resumes with same stage_name -> skips slices completed in Job 1 and finishes!
        spec2 = JobSpec(
            job_id="job_stage2",
            title="Kuramoto Phase Scan Stage 2",
            lineage_author="poolside_laguna",
            realm_source="world_a",
            script_content=script,
            timeout_seconds=10,
            stage_name="multi_stage_kuramoto",
            resume_from_job_id="job_stage1",
        )
        dispatcher.submit(spec2)
        res2 = dispatcher.execute_sync("job_stage2")
        assert res2.status == JobStatus.COMPLETED
        assert "ALL STAGES COMPLETE!" in res2.stdout_tail
        assert "final_phase_diagram.npz" in res2.artifacts_generated
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_cross_instance_stage_reading():
    tmp_root = tempfile.mkdtemp(prefix="wc_cross_inst_")
    jobs_dir = os.path.join(tmp_root, "jobs")
    world_a_dir = os.path.join(tmp_root, "world_a")
    gem_ws = os.path.join(world_a_dir, "instances", "gemini_3_1_flash_lite", "agent_workspace", "world_c_results")
    pool_ws = os.path.join(world_a_dir, "instances", "poolside_laguna", "agent_workspace", "world_c_results")
    os.makedirs(gem_ws, exist_ok=True)
    os.makedirs(pool_ws, exist_ok=True)
    os.makedirs(os.path.join(world_a_dir, "instances", "shared_space"), exist_ok=True)

    try:
        # 1. gemini_3_1_flash_lite saves a 2D Gray-Scott bifurcation grid in its stage directory
        gem_stage_dir = os.path.join(gem_ws, "stages", "gs_bifurcation_scan")
        t_gem = StageTracker("gs_bifurcation_scan", checkpoint_dir=gem_stage_dir, register_signals=False)
        for F in [0.02, 0.03, 0.04]:
            for k in [0.05, 0.06]:
                t_gem.save_stage(f"{F:.2f}__{k:.2f}", {"variance": float(F * k), "mean": float(1.0 - F)})

        # 2. poolside_laguna runs a World C job loading StageTracker('gs_bifurcation_scan').load_stage('final')
        dispatcher = ComputeDispatcher(base_jobs_dir=jobs_dir, world_a_root=world_a_dir)
        pool_script = """
import numpy as np
from colony_lib.persistence import StageTracker

tracker = StageTracker('gs_bifurcation_scan')
data = tracker.load_stage('final')
F_values = data['F_values']
k_values = data['k_values']
variance = data['variance']
assert len(F_values) == 3 and len(k_values) == 2
assert variance.shape == (3, 2)
print('CROSS_INSTANCE_STAGE_OK', F_values.tolist(), k_values.tolist(), flush=True)
"""
        spec_pool = JobSpec(
            job_id="job_pool_handoff",
            title="Cross-Instance Stage Handoff Test",
            lineage_author="poolside_laguna",
            realm_source="world_a",
            script_content=pool_script,
            stage_name="gs_scaling_collapse",
        )
        dispatcher.submit(spec_pool)
        res_pool = dispatcher.execute_sync("job_pool_handoff")
        assert res_pool.status == JobStatus.COMPLETED, f"Stderr: {res_pool.stderr_tail}"
        assert "CROSS_INSTANCE_STAGE_OK" in res_pool.stdout_tail
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    test_stage_tracker_arrays_and_resume()
    print("[PASS] test_stage_tracker_arrays_and_resume")
    test_scan_parameter_space_stage_caching()
    print("[PASS] test_scan_parameter_space_stage_caching")
    test_dispatcher_timeout_and_multi_stage_resumption()
    print("[PASS] test_dispatcher_timeout_and_multi_stage_resumption")
    test_cross_instance_stage_reading()
    print("[PASS] test_cross_instance_stage_reading")
    print("All multi-stage persistence tests passed!")
