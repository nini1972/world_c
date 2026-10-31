import numpy as np
import pytest

from colony_lib.dynamics.kuramoto import simulate_kuramoto, compute_order_parameter
from colony_lib.dynamics.solitons import simulate_phi4_collision
from colony_lib.dynamics.gray_scott import simulate_gray_scott
from colony_lib.dynamics.integrators import rk4_step, velocity_verlet_step
from colony_lib.bifurcation.continuation import detect_critical_point
from colony_lib.bifurcation.normal_forms import classify_bifurcation_1d
from colony_lib.recurrence.takens import takens_embedding, estimate_delay_autocorr
from colony_lib.recurrence.rqa import recurrence_matrix, compute_rqa_metrics
from colony_lib.recurrence.adler import AdlerDetector
from colony_lib.invariants.registry import InvariantRecord, InvariantRegistry
from colony_lib.invariants.collapse import optimize_scaling_collapse

def test_kuramoto_simulation():
    res = simulate_kuramoto(n_oscillators=50, K=3.0, t_max=10.0, dt=0.05, seed=123)
    assert res["N"] == 50
    assert len(res["times"]) == len(res["r_series"])
    assert 0.0 <= res["steady_mean_r"] <= 1.0

def test_solitons_simulation():
    res = simulate_phi4_collision(v=0.25, x0=8.0, t_max=15.0, N_grid=128, L=30.0, dt=0.05)
    assert res["velocity"] == 0.25
    assert len(res["times"]) == len(res["center_phi"])
    # Energy drift should be very small in symplectic Strang splitting
    assert res["energy_drift"] < 0.05

def test_gray_scott_simulation():
    res = simulate_gray_scott(grid_size=32, steps=100, dt=1.0, seed=42)
    assert res["v_field"].shape == (32, 32)
    assert res["mean_v"] >= 0.0

def test_takens_and_rqa():
    # Test on a sine wave
    t = np.linspace(0, 4 * np.pi, 200)
    series = np.sin(t)
    
    tau = estimate_delay_autocorr(series)
    assert tau >= 1
    
    embedded = takens_embedding(series, m=2, tau=tau)
    assert embedded.shape[1] == 2
    
    R = recurrence_matrix(embedded, epsilon=0.2)
    metrics = compute_rqa_metrics(R)
    assert 0.0 <= metrics["recurrence_rate"] <= 1.0
    assert 0.0 <= metrics["determinism"] <= 1.0

def test_adler_detector():
    detector = AdlerDetector(delta=1.0)
    # Locked regime K > delta
    res_locked = detector.simulate(K=1.5, t_max=20.0)
    assert res_locked["is_locked"] is True
    assert res_locked["slip_count"] == 0
    
    # Slipping regime K < delta
    res_slip = detector.simulate(K=0.5, t_max=20.0)
    assert res_slip["is_locked"] is False
    assert res_slip["slip_count"] > 0

def test_bifurcation_tools():
    params = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    susc = np.array([0.1, 0.4, 1.5, 0.5, 0.2])
    crit = detect_critical_point(params, susc)
    assert 1.0 <= crit["k_critical"] <= 2.0
    
    nf = classify_bifurcation_1d(a0_mu=0.0, a1_mu=1.0, a2=0.0, a3=-1.0)
    assert nf["type"] == "pitchfork_supercritical"

def test_scan_parameter_space():
    from colony_lib.bifurcation import scan_parameter_space
    
    # 1D scan
    res_1d = scan_parameter_space(lambda x: {"sq": x**2}, [1.0, 2.0, 3.0])
    assert "sq" in res_1d
    assert np.allclose(res_1d["sq"], [1.0, 4.0, 9.0])
    
    # Grid scan
    res_2d = scan_parameter_space(lambda a, b: {"sum": a + b}, param_grid={"a": [1.0, 2.0], "b": [10.0, 20.0]})
    assert "sum" in res_2d
    assert len(res_2d["sum"]) == 4

def test_invariant_registry(tmp_path):
    reg_file = str(tmp_path / "test_reg.json")
    registry = InvariantRegistry(registry_file=reg_file)
    
    rec = InvariantRecord(
        law_id="LAW-KURAMOTO-KC",
        law_name="Cauchy Kuramoto Threshold",
        lineage_author="Cartographer",
        discovery_realm="World A",
        system_type="coupled_oscillators",
        invariant_type="threshold",
        mathematical_formulation="K_c = 2 * gamma",
        measured_values={"K_c": 2.002},
        uncertainty={"K_c": 0.015}
    )
    h = registry.register(rec)
    assert len(h) == 64
    
    # Verify retrieval
    retrieved = registry.get("LAW-KURAMOTO-KC")
    assert retrieved is not None
    assert retrieved.provenance_hash == h
    
    # Verify reproducibility check
    assert registry.verify_reproducibility("LAW-KURAMOTO-KC", {"K_c": 2.01}, tolerance=0.05) is True
    assert registry.verify_reproducibility("LAW-KURAMOTO-KC", {"K_c": 3.50}, tolerance=0.05) is False

def test_gaussian_process_surrogate():
    from colony_lib.emulators import GaussianProcessSurrogate
    
    # Simple synthetic 1D function: y = sin(2 * pi * x)
    X = np.linspace(0, 1, 8)
    y = np.sin(2 * np.pi * X)
    
    gp = GaussianProcessSurrogate(kernel_type="matern")
    gp.fit(X, y)
    
    X_test = np.linspace(0, 1, 25)
    mean, std = gp.predict(X_test, return_std=True)
    assert len(mean) == 25
    assert len(std) == 25
    assert np.all(std >= 0)
    
    # Active learning sample suggestion
    candidates = np.linspace(0, 1, 50)
    next_pts = gp.suggest_next_samples(candidates, n_samples=3)
    assert len(next_pts) == 3
    
    # Boundary check
    crit = gp.find_critical_boundary(candidates, threshold=0.0)
    assert len(crit["boundary_points"]) > 0

def test_turing_instability_analysis():
    from colony_lib.dynamics import turing_dispersion_relation, check_turing_conditions
    
    # Classic Schnakenberg/Turing parameters: Du=1, Dv=20, standard activator-inhibitor Jacobian
    Du = 1.0
    Dv = 20.0
    J = np.array([
        [1.0, -2.0],
        [3.0, -4.0]
    ])
    
    info = check_turing_conditions(Du, Dv, J)
    assert info["condition_1_trace_negative"] is True
    assert info["condition_2_det_positive"] is True
    
    k_vals = np.linspace(0, 2.0, 30)
    disp = turing_dispersion_relation(k_vals, Du, Dv, J)
    assert len(disp) == 30

def test_lenia_continuous_cellular_automata():
    from colony_lib.dynamics import Lenia2D
    
    lenia = Lenia2D(grid_size=32, kernel_radius=5)
    lenia.seed_orbium()
    assert np.sum(lenia.state) > 0.0
    
    metrics = lenia.run(steps=5)
    assert metrics["final_mass"] >= 0.0
    assert len(metrics["mass_trajectory"]) == 5
    assert 0.0 <= metrics["active_area_fraction"] <= 1.0

def test_betti0_persistence_imports():
    from colony_lib.invariants import betti0_persistence as bp_inv
    from colony_lib.morphospace import betti0_persistence as bp_morph, compute_betti_0_curve
    
    assert bp_inv is compute_betti_0_curve
    assert bp_morph is compute_betti_0_curve
    
    pts = np.random.RandomState(42).randn(10, 2)
    res = bp_inv(pts, epsilons=np.array([0.1, 1.0, 5.0]))
    assert len(res["betti_0"]) == 3

def test_cml_and_gray_scott_jit():
    from colony_lib.dynamics.cml import coupled_logistic_map, simulate_coupled_logistic_map
    from colony_lib.dynamics.gray_scott import GrayScott2D, simulate_gray_scott
    
    # 1. Test CML single step and trajectory
    x0 = np.array([0.2, 0.4, 0.6, 0.8])
    x1 = coupled_logistic_map(x0, r=3.865, eps=0.132)
    assert len(x1) == 4
    assert np.all(x1 >= 0.0) and np.all(x1 <= 1.0)
    
    traj = simulate_coupled_logistic_map(N=30, steps=50, r=3.865, eps=0.132, seed=42)
    assert traj.shape == (50, 30)
    
    # 2. Test Gray-Scott integration
    res = simulate_gray_scott(grid_size=32, steps=100, seed=42)
    assert res["v_field"].shape == (32, 32)
    assert res["mean_v"] >= 0.0


def test_claude_haiku_cartographer_upgrades(tmp_path):
    from colony_lib import (
        bifurcation_cartography as bc,
        network_topology as nt,
        scaling_analysis as sa,
    )
    from compute_engine.job_spec import JobSpec, JobResult, JobStatus
    from embassy.bridge import EmbassyBridge

    # 1. Bifurcation Cartography: continuation_kuramoto, extract_critical_exponent, adaptive_phase_scan
    K_vals, r_vals, stab = bc.continuation_kuramoto(
        N=20, sigma=0.4, topology="scale-free", K_init=0.2, K_max=2.5, steps=8, t_max=3.0, dt=0.1, seed=42
    )
    assert len(K_vals) == 8
    assert len(r_vals) == 8
    assert len(stab) == 8
    beta, k_crit = bc.extract_critical_exponent(K_vals, r_vals, window=4)
    assert 0.05 <= beta <= 2.5
    assert 0.2 <= k_crit <= 2.5

    phase_diag = bc.adaptive_phase_scan(
        param1_range=(0.0, 2.0),
        param2_range=(0.1, 1.0),
        param1_name="K",
        param2_name="sigma",
        observable_fn=lambda K, sigma: float(np.tanh(max(K - sigma, 0.0))),
        refinement_threshold=0.2,
        initial_grid_size=4,
        n_jobs=2,
    )
    assert phase_diag["n_coarse_points"] == 16
    assert phase_diag["n_refined_points"] > 0

    # Parallel parameter sweep with N_cores
    sweep_res = bc.scan_parameter_space(
        lambda K, sigma: {"r": K / (1.0 + sigma)},
        param_grid={"K": [0.5, 1.0, 1.5], "sigma": [0.2, 0.4]},
        N_cores=2,
    )
    assert len(sweep_res["r"]) == 6

    # 2. Network Topology: laplacian_spectrum, spectral_gap, communities, clustering, assortativity
    A = nt.generate_topology("small-world", N=30, k=4, p=0.2, seed=42)
    evals, evecs = nt.laplacian_spectrum(A)
    assert evals.shape == (30,)
    assert evecs.shape == (30, 30)
    assert np.isclose(evals[0], 0.0, atol=1e-10)
    assert evals[1] > 0.0
    assert np.isclose(nt.spectral_gap(A), evals[1])
    assert np.isclose(nt.algebraic_connectivity(A), evals[1])

    comms = nt.detect_communities(A, method="louvain", seed=42)
    assert len(comms) >= 1
    assert sum(len(c) for c in comms) == 30
    assert 0.0 <= nt.average_clustering(A) <= 1.0
    assert -1.0 <= nt.assortativity(A) <= 1.0

    # 3. Scaling Analysis: scaling_collapse & fit_scaling_law
    dK = np.array([0.1, 0.2, 0.4, 0.8, 0.1, 0.2, 0.4, 0.8])
    N_arr = np.array([50, 50, 100, 100, 50, 50, 100, 100], dtype=float)
    sig_arr = np.array([0.5, 1.0, 0.5, 1.0, 1.5, 0.5, 1.5, 1.0])
    y_true = 1.5 * (dK ** 0.5) * (N_arr ** -0.25) * (sig_arr ** 0.3)

    x_sc, y_sc = sa.scaling_collapse(
        {"N": N_arr, "K_delta": dK, "sigma": sig_arr},
        y_true,
        exponents={"N": -0.25, "K_delta": 0.5, "sigma": 0.3},
    )
    assert len(x_sc) == 8
    assert len(y_sc) == 8

    fit_res = sa.fit_scaling_law(
        {"K_delta": dK, "N": N_arr, "sigma": sig_arr},
        y_true,
        form="y ~ (K - K_c)^beta * N^alpha * sigma^gamma",
        constraints={"beta": (0.0, 2.0), "alpha": (-1.0, 1.0), "gamma": (-1.0, 1.0)},
    )
    assert np.isclose(fit_res["beta"], 0.5, atol=0.05)
    assert np.isclose(fit_res["alpha"], -0.25, atol=0.05)
    assert np.isclose(fit_res["gamma"], 0.3, atol=0.05)
    assert fit_res["r_squared"] > 0.99

    # 4. Auto-indexing research_log.json in EmbassyBridge
    ws_dir = tmp_path / "instances" / "claude_haiku" / "agent_workspace" / "world_c_results"
    ws_dir.mkdir(parents=True)
    bridge = EmbassyBridge(world_a_root=str(tmp_path))
    spec = JobSpec(
        job_id="job_claude_haiku_test_01",
        title="Network Topology Phase Diagram & Scaling Law Scan",
        lineage_author="claude_haiku",
        realm_source="world_a",
    )
    res = JobResult(
        job_id="job_claude_haiku_test_01",
        status=JobStatus.COMPLETED,
        exit_code=0,
        execution_time_seconds=12.34,
        artifacts_generated=["phase_diagram.png", "scaling_results.json"],
    )
    bridge.write_completion_report(res, spec)
    log_file = ws_dir / "research_log.json"
    assert log_file.exists()
    import json
    log_entries = json.loads(log_file.read_text(encoding="utf-8"))
    assert len(log_entries) == 1
    assert log_entries[0]["job_id"] == "job_claude_haiku_test_01"
    assert "phase-diagram" in log_entries[0]["tags"]
    assert "topology-comparison" in log_entries[0]["tags"]
    assert "scaling-law" in log_entries[0]["tags"]



