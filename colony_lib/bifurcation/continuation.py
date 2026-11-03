"""
Numerical Parameter Sweeps, Bifurcation Continuation, Adaptive Phase Cartography, and Critical Point Detection.
"""

import os
import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Any, List, Optional, Tuple, Union
import numpy as np


def _resolve_workers(n_jobs: int = 1, N_cores: Optional[int] = None) -> int:
    """Resolves requested worker count safely against available system vCPUs."""
    requested = N_cores if N_cores is not None else n_jobs
    if requested is None or requested == 1:
        return 1
    cpu_max = os.cpu_count() or 4
    if requested <= 0:
        return max(1, min(cpu_max, 8))
    return max(1, min(int(requested), cpu_max, 8))


def parameter_sweep_scan(
    eval_fn: Callable[[float], Dict[str, float]],
    param_values: np.ndarray,
    n_jobs: int = 1,
    N_cores: Optional[int] = None,
    stage_name: Optional[str] = None,
    resume: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Evaluates a function across a 1D parameter array and aggregates all scalar outputs.
    Supports parallel parameter sweeps via `n_jobs` or `N_cores` and automatic
    multi-stage checkpointing/resumption via `stage_name` (or `WORLD_C_STAGE_NAME`).
    """
    param_arr = np.asarray(param_values, dtype=np.float64)
    workers = _resolve_workers(n_jobs=n_jobs, N_cores=N_cores)

    active_stage = stage_name or os.environ.get("WORLD_C_STAGE_NAME")
    tracker = None
    if active_stage:
        from ..persistence import StageTracker
        tracker = StageTracker(stage_name=active_stage, auto_flush=False)

    outputs: List[Any] = [None] * len(param_arr)
    pending_indices: List[int] = []

    for idx, val in enumerate(param_arr):
        key = f"sweep1d_{float(val):.8g}"
        if tracker is not None and resume and tracker.is_done(key):
            outputs[idx] = tracker.get_stage(key)
        else:
            pending_indices.append(idx)

    if pending_indices:
        batch_size = max(1, workers * 4)
        for b_start in range(0, len(pending_indices), batch_size):
            batch_idxs = pending_indices[b_start : b_start + batch_size]
            batch_vals = [float(param_arr[i]) for i in batch_idxs]
            if workers > 1 and len(batch_vals) > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    batch_outs = list(pool.map(eval_fn, batch_vals))
            else:
                batch_outs = [eval_fn(v) for v in batch_vals]

            for i, out in zip(batch_idxs, batch_outs):
                outputs[i] = out
                if tracker is not None:
                    tracker.save_stage(f"sweep1d_{float(param_arr[i]):.8g}", out)
            if tracker is not None:
                tracker.flush()

    results: Dict[str, List[float]] = {}
    for out in outputs:
        if isinstance(out, dict):
            for k, v in out.items():
                if k not in results:
                    results[k] = []
                results[k].append(float(v))
        else:
            if "output" not in results:
                results["output"] = []
            results["output"].append(float(out))

    aggregated = {"param": param_arr}
    for k, v_list in results.items():
        aggregated[k] = np.array(v_list, dtype=np.float64)

    return aggregated


def scan_parameter_space(
    eval_fn: Callable[..., Any],
    param_grid: Any = None,
    n_jobs: int = 1,
    N_cores: Optional[int] = None,
    stage_name: Optional[str] = None,
    resume: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Evaluates a function across a 1D parameter array or multi-dimensional parameter space grid.
    Supports parallel parameter sweeps via `n_jobs` or `N_cores` and automatic
    multi-stage checkpointing/resumption via `stage_name` (or `WORLD_C_STAGE_NAME`).
    """
    workers = _resolve_workers(n_jobs=n_jobs, N_cores=N_cores)

    # Handle direct 1D array passed as first argument after eval_fn
    if param_grid is not None and isinstance(param_grid, (list, tuple, np.ndarray)):
        return parameter_sweep_scan(
            eval_fn,
            np.asarray(param_grid),
            n_jobs=workers,
            stage_name=stage_name,
            resume=resume,
        )

    # Handle param_grid passed as dict
    if isinstance(param_grid, dict):
        grid_dict = param_grid
    elif kwargs:
        grid_dict = kwargs
    else:
        return {}

    # Check if single 1D param in dict
    if len(grid_dict) == 1 and ("param_values" in grid_dict or "param" in grid_dict):
        p_val = list(grid_dict.values())[0]
        return parameter_sweep_scan(
            eval_fn,
            np.asarray(p_val),
            n_jobs=workers,
            stage_name=stage_name,
            resume=resume,
        )

    keys = list(grid_dict.keys())
    values = [np.asarray(grid_dict[k]) for k in keys]
    grid_points = list(itertools.product(*values))

    def _pt_key(pt) -> str:
        parts = []
        for idx, k in enumerate(keys):
            v = pt[idx]
            if isinstance(v, (float, np.floating)):
                parts.append(f"{k}={float(v):.8g}")
            else:
                parts.append(f"{k}={v}")
        return "grid__" + "__".join(parts)

    def _eval_point(pt):
        kw = {keys[i]: pt[i] for i in range(len(keys))}
        try:
            return eval_fn(**kw)
        except TypeError:
            return eval_fn(*pt)

    active_stage = stage_name or os.environ.get("WORLD_C_STAGE_NAME")
    tracker = None
    if active_stage:
        from ..persistence import StageTracker
        tracker = StageTracker(stage_name=active_stage, auto_flush=False)

    outputs: List[Any] = [None] * len(grid_points)
    pending_indices: List[int] = []

    for idx, pt in enumerate(grid_points):
        pk = _pt_key(pt)
        if tracker is not None and resume and tracker.is_done(pk):
            outputs[idx] = tracker.get_stage(pk)
        else:
            pending_indices.append(idx)

    if pending_indices:
        batch_size = max(1, workers * 4)
        for b_start in range(0, len(pending_indices), batch_size):
            batch_idxs = pending_indices[b_start : b_start + batch_size]
            batch_pts = [grid_points[i] for i in batch_idxs]
            if workers > 1 and len(batch_pts) > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    batch_outs = list(pool.map(_eval_point, batch_pts))
            else:
                batch_outs = [_eval_point(pt) for pt in batch_pts]

            for i, out in zip(batch_idxs, batch_outs):
                outputs[i] = out
                if tracker is not None:
                    tracker.save_stage(_pt_key(grid_points[i]), out)
            if tracker is not None:
                tracker.flush()

    results: Dict[str, List[Any]] = {k: [] for k in keys}
    out_records: Dict[str, List[Any]] = {}

    for pt, out in zip(grid_points, outputs):
        for idx, k in enumerate(keys):
            results[k].append(pt[idx])
        if isinstance(out, dict):
            for ok, ov in out.items():
                if ok not in out_records:
                    out_records[ok] = []
                out_records[ok].append(ov)
        else:
            if "output" not in out_records:
                out_records["output"] = []
            out_records["output"].append(out)

    aggregated = {k: np.array(v) for k, v in results.items()}
    for ok, ov in out_records.items():
        aggregated[ok] = np.array(ov)
    return aggregated


def detect_critical_point(
    param_values: np.ndarray,
    susceptibility: np.ndarray
) -> Dict[str, float]:
    """
    Locates the critical point K_c where susceptibility reaches its global maximum.
    Uses parabolic interpolation around the peak for sub-grid resolution.
    """
    if len(param_values) != len(susceptibility):
        raise ValueError("Parameter values and susceptibility arrays must have equal length")

    peak_idx = int(np.argmax(susceptibility))

    # Sub-grid refinement via parabolic interpolation
    if 0 < peak_idx < len(param_values) - 1:
        x0, x1, x2 = param_values[peak_idx - 1], param_values[peak_idx], param_values[peak_idx + 1]
        y0, y1, y2 = susceptibility[peak_idx - 1], susceptibility[peak_idx], susceptibility[peak_idx + 1]

        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if abs(denom) > 1e-14:
            a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
            b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
            if abs(a) > 1e-14:
                k_crit = -b / (2.0 * a)
            else:
                k_crit = x1
        else:
            k_crit = x1
    else:
        k_crit = param_values[peak_idx]

    return {
        "k_critical": float(k_crit),
        "peak_susceptibility": float(susceptibility[peak_idx]),
        "peak_index": float(peak_idx)
    }


def continuation_kuramoto(
    N: int = 50,
    sigma: float = 0.5,
    topology: str = "scale-free",
    K_init: float = 0.5,
    K_max: float = 3.0,
    steps: int = 40,
    t_max: float = 12.0,
    dt: float = 0.05,
    seed: int = 42,
    stage_name: Optional[str] = None,
    resume: bool = True,
    **topology_kwargs: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numerical continuation of synchronization branches for coupled Kuramoto oscillators
    across network topologies. Carries forward final oscillator phases along the branch.
    Supports optional stage checkpointing/resumption via `stage_name`.
    """
    from ..network_topology import generate_topology
    from ..dynamics.kuramoto import compute_order_parameter

    rng = np.random.default_rng(seed)
    omega = rng.normal(loc=0.0, scale=sigma, size=N)
    omega -= np.mean(omega)  # Co-rotating frame

    topo_clean = topology.strip().lower().replace("_", "-")
    if topo_clean in ("all-to-all", "complete", "mean-field"):
        adj = None
    else:
        adj = generate_topology(topo_clean, N=N, seed=seed, **topology_kwargs)
        # Normalize coupling by mean degree so K remains intensive across topologies
        mean_deg = max(float(np.mean(np.sum(adj > 0, axis=1))), 1.0)
        adj = adj / mean_deg

    K_values = np.linspace(float(K_init), float(K_max), int(steps))
    r_values = np.zeros(len(K_values), dtype=np.float64)
    stability = np.zeros(len(K_values), dtype=bool)

    theta = rng.uniform(0.0, 2.0 * np.pi, size=N)
    n_time_steps = max(20, int(t_max / dt))
    steady_window = max(5, n_time_steps // 3)

    active_stage = stage_name or os.environ.get("WORLD_C_STAGE_NAME")
    tracker = None
    branch_key = f"cont_{topo_clean}_N{N}_sig{sigma:.5g}_K{K_init:.4g}_{K_max:.4g}_s{steps}_seed{seed}"
    start_idx = 0

    if active_stage:
        from ..persistence import StageTracker
        tracker = StageTracker(stage_name=active_stage, auto_flush=False)
        if resume and tracker.is_done(branch_key):
            saved = tracker.get_stage(branch_key)
            if isinstance(saved, dict) and "last_idx" in saved:
                last_idx = int(saved["last_idx"])
                saved_r = np.asarray(saved.get("r_values", []), dtype=np.float64)
                saved_stab = np.asarray(saved.get("stability", []), dtype=bool)
                saved_theta = np.asarray(saved.get("theta", []), dtype=np.float64)
                if len(saved_r) == len(K_values) and len(saved_theta) == N:
                    r_values[:] = saved_r
                    stability[:] = saved_stab
                    theta = saved_theta.copy()
                    start_idx = last_idx + 1

    def _rhs(th: np.ndarray, K_val: float) -> np.ndarray:
        if adj is None:
            z = np.mean(np.exp(1j * th))
            r_inst = np.abs(z)
            psi = np.angle(z)
            return omega + K_val * r_inst * np.sin(psi - th)
        diff = th[None, :] - th[:, None]
        return omega + K_val * np.sum(adj * np.sin(diff), axis=1)

    for idx in range(start_idx, len(K_values)):
        K_val = K_values[idx]
        r_trace = np.empty(steady_window, dtype=np.float64)
        trace_pos = 0
        for step_i in range(n_time_steps):
            k1 = _rhs(theta, K_val)
            k2 = _rhs(theta + 0.5 * dt * k1, K_val)
            k3 = _rhs(theta + 0.5 * dt * k2, K_val)
            k4 = _rhs(theta + dt * k3, K_val)
            theta = np.mod(theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 2.0 * np.pi)
            if step_i >= n_time_steps - steady_window:
                order_out = compute_order_parameter(theta)
                r_trace[trace_pos] = float(order_out[0]) if isinstance(order_out, tuple) else float(order_out)
                trace_pos += 1

        r_mean = float(np.mean(r_trace[:trace_pos]))
        r_std = float(np.std(r_trace[:trace_pos]))
        r_values[idx] = r_mean
        # Synchronized branch is stable when r exceeds finite-size floor and fluctuations are small
        finite_floor = 1.25 / np.sqrt(max(N, 4))
        stability[idx] = bool(r_mean > finite_floor and r_std < 0.15)

        # Small perturbation to avoid getting trapped on unstable equilibria during continuation
        theta = np.mod(theta + rng.normal(0.0, 0.01, size=N), 2.0 * np.pi)

        if tracker is not None and ((idx + 1) % 5 == 0 or idx == len(K_values) - 1):
            tracker.save_stage(
                branch_key,
                {
                    "last_idx": int(idx),
                    "K_values": K_values,
                    "r_values": r_values,
                    "stability": stability,
                    "theta": theta,
                },
            )
            tracker.flush()

    return K_values, r_values, stability


class _CriticalExponentResult(tuple):
    """Tuple subclass (beta, k_critical) that also converts directly to float(beta) and supports dict keys."""
    def __float__(self) -> float:
        return float(self[0])

    def __int__(self) -> int:
        return int(float(self[0]))

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            k_low = key.lower()
            if k_low in ("beta", "critical_exponent", "exponent", "alpha", "nu"):
                return float(self[0])
            if k_low in ("k_critical", "kc", "k_c", "mu_c", "critical_point", "threshold"):
                return float(self[1])
            raise KeyError(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def extract_critical_exponent(
    K_values: Union[np.ndarray, List[float]],
    r_values: Union[np.ndarray, List[float]],
    window: int = 5,
    k_critical: Optional[float] = None
) -> Tuple[float, float]:
    """
    Computes the critical exponent beta and critical threshold K_c in the scaling law:
        r ~ (K - K_c)^beta   for K > K_c near bifurcation.

    Parameters
    ----------
    K_values : array-like
        Coupling strengths K.
    r_values : array-like
        Order parameter values r(K).
    window : int, default 5
        Number of supercritical points above K_c used for log-log scaling regression.
    k_critical : float, optional
        If provided, uses this critical point; otherwise detects K_c via steepest gradient dr/dK.

    Returns
    -------
    beta : float
        Extracted critical order parameter exponent.
    K_critical : float
        Estimated critical coupling threshold K_c.
    """
    K = np.asarray(K_values, dtype=np.float64)
    r = np.asarray(r_values, dtype=np.float64)
    if len(K) != len(r) or len(K) < 3:
        raise ValueError("K_values and r_values must have equal length >= 3.")

    sort_idx = np.argsort(K)
    K = K[sort_idx]
    r = r[sort_idx]

    if k_critical is None:
        grad = np.gradient(r, K)
        crit_info = detect_critical_point(K, np.maximum(grad, 0.0))
        k_crit = float(crit_info["k_critical"])
    else:
        k_crit = float(k_critical)

    # Select supercritical window above k_crit
    super_mask = (K > k_crit + 1e-8) & (r > 1e-6)
    K_sup = K[super_mask]
    r_sup = r[super_mask]

    if len(K_sup) < 2:
        # Fallback: take upper half of K range
        mid = max(1, len(K) // 2)
        k_crit = float(K[mid - 1])
        K_sup = K[mid:]
        r_sup = np.maximum(r[mid:], 1e-6)

    w = max(2, min(int(window), len(K_sup)))
    dK = K_sup[:w] - k_crit
    r_win = r_sup[:w]

    # Subtract baseline r at criticality if finite-size floor is present
    r_crit = float(np.interp(k_crit, K, r))
    dr = np.maximum(r_win - 0.5 * r_crit, 1e-6)

    valid = (dK > 1e-8) & (dr > 1e-8)
    if np.sum(valid) >= 2:
        slope, _ = np.polyfit(np.log(dK[valid]), np.log(dr[valid]), 1)
        beta = float(np.clip(slope, 0.05, 2.5))
    else:
        beta = 0.5

    return _CriticalExponentResult((float(beta), float(k_crit)))


def adaptive_phase_scan(
    param1_range: Tuple[float, float],
    param2_range: Tuple[float, float],
    param1_name: str = "K",
    param2_name: str = "sigma",
    observable_fn: Optional[Callable[..., float]] = None,
    refinement_threshold: float = 0.1,
    initial_grid_size: int = 8,
    n_jobs: int = 1
) -> Dict[str, Any]:
    """
    Maps a 2D phase diagram over (param1_range, param2_range) with adaptive mesh refinement
    wherever the local observable gradient exceeds `refinement_threshold`.

    Parameters
    ----------
    param1_range : tuple of (min, max)
        Bounds for parameter 1 (e.g. (0.0, 2.0)).
    param2_range : tuple of (min, max)
        Bounds for parameter 2 (e.g. (0.0, 1.0)).
    param1_name : str, default 'K'
        Name of parameter 1.
    param2_name : str, default 'sigma'
        Name of parameter 2.
    observable_fn : callable
        Function accepting `(p1, p2)` or `(**{param1_name: p1, param2_name: p2})` and returning
        either a float or a dict containing an observable scalar.
    refinement_threshold : float, default 0.1
        Cell gradient threshold triggering midpoint refinement.
    initial_grid_size : int, default 8
        Resolution of the initial coarse grid along each axis.
    n_jobs : int, default 1
        Number of worker threads for evaluating the observable function.

    Returns
    -------
    dict
        Phase diagram dictionary with coarse grid arrays, adaptively refined samples,
        and detected transition ridge coordinates.
    """
    if observable_fn is None:
        raise ValueError("observable_fn must be provided for adaptive_phase_scan.")

    p1_vals = np.linspace(float(param1_range[0]), float(param1_range[1]), int(initial_grid_size))
    p2_vals = np.linspace(float(param2_range[0]), float(param2_range[1]), int(initial_grid_size))

    def _call_obs(pt: Tuple[float, float]) -> float:
        x1, x2 = float(pt[0]), float(pt[1])
        try:
            res = observable_fn(**{param1_name: x1, param2_name: x2})
        except TypeError:
            res = observable_fn(x1, x2)
        if isinstance(res, dict):
            for key in ("r", "order_parameter", "observable", "steady_mean_r", "output"):
                if key in res:
                    return float(res[key])
            return float(next(iter(res.values())))
        return float(res)

    coarse_pts = [(x1, x2) for x1 in p1_vals for x2 in p2_vals]
    workers = _resolve_workers(n_jobs=n_jobs)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            coarse_obs = list(pool.map(_call_obs, coarse_pts))
    else:
        coarse_obs = [_call_obs(pt) for pt in coarse_pts]

    grid_matrix = np.array(coarse_obs, dtype=np.float64).reshape(len(p1_vals), len(p2_vals))

    # Identify steep gradient cells for adaptive midpoint refinement
    refined_candidates: List[Tuple[float, float]] = []
    for i in range(len(p1_vals) - 1):
        for j in range(len(p2_vals) - 1):
            cell_vals = [
                grid_matrix[i, j],
                grid_matrix[i + 1, j],
                grid_matrix[i, j + 1],
                grid_matrix[i + 1, j + 1],
            ]
            local_variation = max(cell_vals) - min(cell_vals)
            if local_variation >= refinement_threshold:
                mid_p1 = 0.5 * (p1_vals[i] + p1_vals[i + 1])
                mid_p2 = 0.5 * (p2_vals[j] + p2_vals[j + 1])
                refined_candidates.append((mid_p1, mid_p2))

    if refined_candidates:
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                refined_obs = list(pool.map(_call_obs, refined_candidates))
        else:
            refined_obs = [_call_obs(pt) for pt in refined_candidates]
    else:
        refined_obs = []

    all_pts = coarse_pts + refined_candidates
    all_obs = coarse_obs + refined_obs

    return {
        param1_name: np.array([pt[0] for pt in all_pts], dtype=np.float64),
        param2_name: np.array([pt[1] for pt in all_pts], dtype=np.float64),
        "observable": np.array(all_obs, dtype=np.float64),
        "coarse_param1": p1_vals,
        "coarse_param2": p2_vals,
        "coarse_grid": grid_matrix,
        "n_coarse_points": len(coarse_pts),
        "n_refined_points": len(refined_candidates),
        "refined_coordinates": np.array(refined_candidates, dtype=np.float64) if refined_candidates else np.empty((0, 2)),
    }
