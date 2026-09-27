"""
Takens Phase Space Delay Embedding
Reconstructs attractor trajectories from scalar time series:
X(t) = [x(t), x(t + tau), x(t + 2*tau), ..., x(t + (m-1)*tau)]
"""

import numpy as np
from typing import Tuple

def estimate_delay_autocorr(series: np.ndarray, max_lag: int = 100) -> int:
    """
    Estimates optimal embedding delay tau using the first zero-crossing
    or 1/e decay of the autocorrelation function.
    """
    s = series - np.mean(series)
    autocorr = np.correlate(s, s, mode="full")
    mid = len(autocorr) // 2
    norm_ac = autocorr[mid:mid + max_lag] / (autocorr[mid] + 1e-12)
    
    # First zero-crossing
    zero_crossings = np.where(norm_ac[:-1] * norm_ac[1:] <= 0)[0]
    if len(zero_crossings) > 0:
        return int(zero_crossings[0] + 1)
        
    # Fallback to 1/e drop
    e_drop = np.where(norm_ac <= 1.0 / np.e)[0]
    if len(e_drop) > 0:
        return int(e_drop[0] + 1)
        
    return 1

def takens_embedding(series: np.ndarray, m: int = 3, tau: int = 1) -> np.ndarray:
    """
    Constructs m-dimensional Takens embedding with delay tau.
    Returns array of shape (N - (m-1)*tau, m).
    """
    n = len(series)
    n_vectors = n - (m - 1) * tau
    if n_vectors <= 0:
        raise ValueError(f"Series length {n} is too short for m={m} and tau={tau}")
        
    embedded = np.empty((n_vectors, m), dtype=series.dtype)
    for i in range(m):
        embedded[:, i] = series[i * tau : i * tau + n_vectors]
        
    return embedded
