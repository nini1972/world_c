"""
Universal Scaling Collapse Optimizer
Optimizes critical exponents (beta, nu, x_c) to collapse finite-size curves onto a universal master curve:
y_tilde = y * L^(-beta / nu)
x_tilde = (x - x_c) * L^(1 / nu)
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Any, Tuple

def scaling_quality_score(
    curves: List[Tuple[np.ndarray, np.ndarray, float]],
    x_c: float,
    beta_over_nu: float,
    one_over_nu: float
) -> float:
    """
    Evaluates collapse residual variance using mean-squared difference between adjacent sorted points.
    curves: list of (x_array, y_array, L_system_size)
    """
    all_x_tilde = []
    all_y_tilde = []
    
    for x_arr, y_arr, L in curves:
        xt = (x_arr - x_c) * (L ** one_over_nu)
        yt = y_arr * (L ** (-beta_over_nu))
        all_x_tilde.append(xt)
        all_y_tilde.append(yt)
        
    x_cat = np.concatenate(all_x_tilde)
    y_cat = np.concatenate(all_y_tilde)
    
    # Sort by rescaled x
    sort_idx = np.argsort(x_cat)
    xs = x_cat[sort_idx]
    ys = y_cat[sort_idx]
    
    # Adjacent differences normalized by spacing
    dx = np.diff(xs)
    dy = np.diff(ys)
    
    # Avoid zero division
    valid = dx > 1e-9
    if np.sum(valid) < 5:
        return 1e6
        
    # Mean squared variation penalty
    score = np.mean((dy[valid] / (dx[valid] + 1e-4)) ** 2)
    return float(score)

def optimize_scaling_collapse(
    curves: List[Tuple[np.ndarray, np.ndarray, float]],
    initial_guess: Tuple[float, float, float] = (1.0, 0.5, 1.0),
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((0.0, 5.0), (-2.0, 2.0), (0.1, 5.0))
) -> Dict[str, Any]:
    """
    Finds optimal critical parameters (x_c, beta/nu, 1/nu) that minimize collapse scatter.
    """
    def objective(params):
        xc, b_nu, one_nu = params
        return scaling_quality_score(curves, xc, b_nu, one_nu)
        
    res = minimize(objective, initial_guess, bounds=bounds, method="Nelder-Mead")
    
    opt_xc, opt_b_nu, opt_one_nu = res.x
    nu = 1.0 / opt_one_nu if abs(opt_one_nu) > 1e-6 else np.nan
    beta = opt_b_nu * nu
    
    return {
        "success": bool(res.success),
        "x_critical": float(opt_xc),
        "beta_over_nu": float(opt_b_nu),
        "one_over_nu": float(opt_one_nu),
        "nu": float(nu),
        "beta": float(beta),
        "residual_quality": float(res.fun)
    }
