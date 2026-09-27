"""
Numerical Parameter Sweeps and Critical Point Detection
Identifies critical points (e.g. K_c) using maximum susceptibility / variance peaks.
"""

import numpy as np
from typing import Callable, Dict, Any, List

def parameter_sweep_scan(
    eval_fn: Callable[[float], Dict[str, float]],
    param_values: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Evaluates a function across a 1D parameter array and aggregates all scalar outputs.
    """
    results: Dict[str, List[float]] = {}
    
    for val in param_values:
        out = eval_fn(float(val))
        for k, v in out.items():
            if k not in results:
                results[k] = []
            results[k].append(float(v))
            
    aggregated = {"param": param_values}
    for k, v_list in results.items():
        aggregated[k] = np.array(v_list, dtype=np.float64)
        
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
        
        # Vertex of parabola through (x0, y0), (x1, y1), (x2, y2)
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
