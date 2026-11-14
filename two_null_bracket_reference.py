
"""
Colleague Commons - Two-Null Bracket Reference Implementation
Authors: Gemini Pro & Colleague Commons Working Group
Usage:
    from two_null_bracket_reference import evaluate_two_null_bracket, generate_fourier_surrogates, generate_iaaft_surrogates
"""

import numpy as np
from colony_lib.recurrence import takens_embedding, recurrence_matrix, compute_rqa_metrics

def generate_fourier_surrogates(x: np.ndarray, n_surrogates: int = 30, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    N = len(x)
    fft_x = np.fft.rfft(x)
    mag = np.abs(fft_x)
    surrogates = []
    for _ in range(n_surrogates):
        phases = rng.uniform(0, 2 * np.pi, size=len(fft_x))
        # Keep DC and Nyquist real to preserve real-valued ifft properties
        phases[0] = 0.0
        if N % 2 == 0:
            phases[-1] = 0.0
        randomized_fft = mag * np.exp(1j * phases)
        surr = np.fft.irfft(randomized_fft, n=N)
        surrogates.append(surr)
    return np.array(surrogates)

def generate_iaaft_surrogates(x: np.ndarray, n_surrogates: int = 30, max_iter: int = 100, tol: float = 1e-6, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    N = len(x)
    sorted_x = np.sort(x)
    fft_x = np.fft.rfft(x)
    mag_x = np.abs(fft_x)
    surrogates = []
    
    for _ in range(n_surrogates):
        s = rng.permutation(x)
        prev_s = s.copy()
        for _ in range(max_iter):
            fft_s = np.fft.rfft(s)
            phases_s = np.angle(fft_s)
            s_adj = np.fft.irfft(mag_x * np.exp(1j * phases_s), n=N)
            s = sorted_x[np.argsort(np.argsort(s_adj))]
            if np.max(np.abs(s - prev_s)) < tol:
                break
            prev_s = s.copy()
        surrogates.append(s)
    return np.array(surrogates)

def evaluate_two_null_bracket(
    x: np.ndarray,
    m: int = 3,
    tau: int = 1,
    epsilon: float = 0.1,
    n_surrogates: int = 30,
    metric: str = 'determinism',
    eps_variance_floor: float = 1e-4,
    rng=None
) -> dict:
    """
    Evaluates observed signal against Fourier (Null 1: linear stochastic Gaussian)
    and IAAFT (Null 2: static monotonic non-linear transformation of linear Gaussian).
    
    Returns standard z-scores, robust non-parametric rank p-values, and raw effect size.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    x = np.asarray(x, dtype=float)
    # Z-normalize input series for uniform RQA scaling
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-12)
    
    # Observed RQA
    emb_obs = takens_embedding(x_norm, m=m, tau=tau)
    R_obs = recurrence_matrix(emb_obs, epsilon=epsilon)
    metrics_obs = compute_rqa_metrics(R_obs)
    val_obs = float(metrics_obs[metric])
    
    # Generate Surrogates
    f_surrs = generate_fourier_surrogates(x_norm, n_surrogates=n_surrogates, rng=rng)
    i_surrs = generate_iaaft_surrogates(x_norm, n_surrogates=n_surrogates, rng=rng)
    
    # Evaluate Surrogates
    f_vals = []
    for s in f_surrs:
        emb_s = takens_embedding(s, m=m, tau=tau)
        R_s = recurrence_matrix(emb_s, epsilon=epsilon)
        f_vals.append(float(compute_rqa_metrics(R_s)[metric]))
    f_vals = np.array(f_vals)
    
    i_vals = []
    for s in i_surrs:
        emb_s = takens_embedding(s, m=m, tau=tau)
        R_s = recurrence_matrix(emb_s, epsilon=epsilon)
        i_vals.append(float(compute_rqa_metrics(R_s)[metric]))
    i_vals = np.array(i_vals)
    
    mu_F = float(np.mean(f_vals))
    std_F = float(np.std(f_vals, ddof=1))
    mu_IAAFT = float(np.mean(i_vals))
    std_IAAFT = float(np.std(i_vals, ddof=1))
    
    # Check for variance collapse (catastrophic cancellation safeguard)
    flag_f_collapse = (std_F < eps_variance_floor)
    flag_i_collapse = (std_IAAFT < eps_variance_floor)
    
    z_F = (val_obs - mu_F) / std_F if not flag_f_collapse else np.nan
    z_IAAFT = (val_obs - mu_IAAFT) / std_IAAFT if not flag_i_collapse else np.nan
    delta_z = (z_F - z_IAAFT) if (not flag_f_collapse and not flag_i_collapse) else np.nan
    
    # Raw effect size
    delta_mu = mu_IAAFT - mu_F
    
    # Non-parametric rank p-values
    p_F = float(np.sum(f_vals >= val_obs) + 1) / (n_surrogates + 1)
    p_IAAFT = float(np.sum(i_vals >= val_obs) + 1) / (n_surrogates + 1)
    
    return {
        'metric_name': metric,
        'observed_val': val_obs,
        'mu_Fourier': mu_F,
        'std_Fourier': std_F,
        'z_Fourier': z_F,
        'p_Fourier': p_F,
        'mu_IAAFT': mu_IAAFT,
        'std_IAAFT': std_IAAFT,
        'z_IAAFT': z_IAAFT,
        'p_IAAFT': p_IAAFT,
        'delta_z': delta_z,
        'delta_mu': delta_mu,
        'variance_collapse_flag': bool(flag_f_collapse or flag_i_collapse)
    }
