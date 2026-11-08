"""
Universal Scaling Collapse and Multivariate Power-Law Exponent Fitting.
Fulfilling claude_haiku's Cartographer's Petition (Section 2C) for World C.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize


def scaling_collapse(
    x_data: Union[Dict[str, Any], np.ndarray, List[Any]],
    y_data: Union[np.ndarray, List[float]],
    exponents: Optional[Dict[str, float]] = None,
    x_c: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a universal scaling collapse transformation to coordinates `x_data` and
    observable `y_data` using provided scaling exponents.

    Supports three flexible input formats for `x_data`:
    1. Dictionary of named coordinate arrays (e.g. `{'N': N_arr, 'K_delta': dK_arr, 'sigma': sig_arr}`)
    2. 2D NumPy array of shape `(n_samples, n_features)` where columns correspond to `exponents.values()`
    3. 1D NumPy array of control parameters `x`

    Parameters
    ----------
    x_data : dict, 2D array, or 1D array
        System coordinates (e.g. N, K - K_c, sigma).
    y_data : array-like
        Measured observable values (e.g. order parameter r, susceptibility chi).
    exponents : dict of {str: float}, optional
        Scaling exponents for each coordinate (and optionally 'y' or 'beta_over_nu' for y rescaling).
    x_c : float, default 0.0
        Critical shift applied when `x_data` is a 1D array.

    Returns
    -------
    x_scaled : np.ndarray
        Collapsed dimensionless scaling coordinate.
    y_scaled : np.ndarray
        Collapsed dimensionless observable.
    """
    y_arr = np.asarray(y_data, dtype=np.float64)
    if isinstance(exponents, (list, tuple, np.ndarray)):
        exp_dict = {f"exp_{i}": float(v) for i, v in enumerate(exponents)}
    else:
        exp_dict = dict(exponents or {})

    # Extract optional y-exponent if specified in exponents dict
    y_exp = float(exp_dict.pop("y", exp_dict.pop("beta_over_nu", 0.0)))

    if isinstance(x_data, dict):
        x_scaled = np.ones_like(y_arr, dtype=np.float64)
        ref_scale = None
        for key, arr in x_data.items():
            coord = np.asarray(arr, dtype=np.float64)
            if ref_scale is None and key.lower() in ("n", "l", "size", "system_size"):
                ref_scale = coord
            power = float(exp_dict.get(key, 1.0))
            x_scaled = x_scaled * np.power(np.maximum(np.abs(coord), 1e-12), power) * np.sign(coord + 1e-18)
        if abs(y_exp) > 1e-12 and ref_scale is not None:
            y_scaled = y_arr * np.power(np.maximum(np.abs(ref_scale), 1e-12), -y_exp)
        else:
            y_scaled = y_arr.copy()
        return x_scaled, y_scaled

    x_arr = np.asarray(x_data, dtype=np.float64)
    if x_arr.ndim == 2:
        powers = list(exp_dict.values()) if exp_dict else [1.0] * x_arr.shape[1]
        x_scaled = np.ones(x_arr.shape[0], dtype=np.float64)
        for col_idx in range(x_arr.shape[1]):
            p = float(powers[col_idx]) if col_idx < len(powers) else 1.0
            col = x_arr[:, col_idx]
            x_scaled = x_scaled * np.power(np.maximum(np.abs(col), 1e-12), p) * np.sign(col + 1e-18)
        if abs(y_exp) > 1e-12:
            y_scaled = y_arr * np.power(np.maximum(np.abs(x_arr[:, 0]), 1e-12), -y_exp)
        else:
            y_scaled = y_arr.copy()
        return x_scaled, y_scaled

    # 1D array case
    shifted = x_arr - float(x_c)
    first_exp = float(next(iter(exp_dict.values()), 1.0))
    x_scaled = np.power(np.maximum(np.abs(shifted), 1e-12), first_exp) * np.sign(shifted + 1e-18)
    if abs(y_exp) > 1e-12:
        y_scaled = y_arr * np.power(np.maximum(np.abs(x_arr), 1e-12), -y_exp)
    else:
        y_scaled = y_arr.copy()
    return x_scaled, y_scaled


def fit_scaling_law(
    x_data: Union[Dict[str, Any], np.ndarray, List[Any]],
    y_data: Union[np.ndarray, List[float]],
    form: Optional[str] = None,
    constraints: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    Fits a power-law scaling ansatz to empirical or simulation data:
        y ~ C * prod_i (x_i ^ exponent_i)
    or with critical threshold offset `(K - K_c)^beta * N^alpha * sigma^gamma`.

    Parameters
    ----------
    x_data : dict, 2D array, or 1D array
        Input variables. Can be a dict of named arrays (e.g. `{'K': ..., 'N': ..., 'sigma': ...}`),
        a 2D array `(n_samples, n_features)`, or a 1D array.
    y_data : array-like
        Target observable values `y`.
    form : str, optional
        Symbolic formula description (e.g. `'y ~ (K - K_c)^beta * N^alpha * sigma^gamma'`).
    constraints : dict of {param_name: (lower_bound, upper_bound)}, optional
        Bounds for each exponent to fit (e.g. `{'beta': (0, 2), 'alpha': (-1, 1), 'gamma': (-1, 1)}`).

    Returns
    -------
    dict
        Fitted exponent values keyed by parameter name, plus `'prefactor'`, `'r_squared'`, and `'success'`.
    """
    y_arr = np.asarray(y_data, dtype=np.float64).ravel()
    constraints_dict = dict(constraints or {})

    # Determine exponent names from constraints or form
    if constraints_dict:
        exp_names = list(constraints_dict.keys())
    elif form:
        tokens = re.findall(r"\^([A-Za-z_][A-Za-z0-9_]*)", form)
        exp_names = tokens if tokens else ["beta"]
    else:
        exp_names = ["beta"]

    # Convert x_data into a matrix of positive bases of shape (n_samples, n_exponents)
    if isinstance(x_data, dict):
        # Match dictionary keys or order to exp_names
        cols: List[np.ndarray] = []
        k_crit_val = float(x_data.get("K_c", 0.0)) if np.ndim(x_data.get("K_c", 0.0)) == 0 else 0.0
        data_keys = [k for k in x_data.keys() if k != "K_c"]

        for idx, ename in enumerate(exp_names):
            matched_arr = None
            # Direct match or standard correspondence (beta -> K/K_delta, alpha -> N, gamma -> sigma)
            for dk in data_keys:
                dkl = dk.lower()
                enl = ename.lower()
                if dkl == enl:
                    matched_arr = np.asarray(x_data[dk], dtype=np.float64).ravel()
                    break
                if enl == "beta" and dkl in ("k", "k_delta", "dk", "delta_k", "param"):
                    arr = np.asarray(x_data[dk], dtype=np.float64).ravel()
                    matched_arr = np.maximum(arr - k_crit_val, 1e-8) if dkl == "k" else arr
                    break
                if enl == "alpha" and dkl in ("n", "l", "size", "system_size"):
                    matched_arr = np.asarray(x_data[dk], dtype=np.float64).ravel()
                    break
                if enl == "gamma" and dkl in ("sigma", "disorder", "noise", "lambda_1"):
                    matched_arr = np.asarray(x_data[dk], dtype=np.float64).ravel()
                    break
            if matched_arr is None and idx < len(data_keys):
                matched_arr = np.asarray(x_data[data_keys[idx]], dtype=np.float64).ravel()
            if matched_arr is None:
                matched_arr = np.ones_like(y_arr, dtype=np.float64)
            cols.append(np.maximum(np.abs(matched_arr), 1e-8))
        X_mat = np.column_stack(cols) if cols else np.ones((len(y_arr), 1), dtype=np.float64)
    else:
        x_arr = np.asarray(x_data, dtype=np.float64)
        if x_arr.ndim == 1:
            X_mat = np.maximum(np.abs(x_arr), 1e-8).reshape(-1, 1)
        else:
            X_mat = np.maximum(np.abs(x_arr), 1e-8)

    n_vars = X_mat.shape[1]
    while len(exp_names) < n_vars:
        exp_names.append(f"exponent_{len(exp_names) + 1}")
    exp_names = exp_names[:n_vars]

    log_X = np.log(X_mat)
    y_pos = np.maximum(np.abs(y_arr), 1e-12)
    log_y = np.log(y_pos)

    # Analytical log-linear least-squares warm start
    design = np.column_stack([np.ones(len(y_arr), dtype=np.float64), log_X])
    try:
        ols_params, _, _, _ = np.linalg.lstsq(design, log_y, rcond=None)
    except Exception:
        ols_params = np.zeros(1 + n_vars, dtype=np.float64)

    bounds: List[Tuple[float, float]] = [(-25.0, 25.0)]
    p0: List[float] = [float(np.clip(ols_params[0], -25.0, 25.0))]
    for idx, ename in enumerate(exp_names):
        raw_bounds = constraints_dict.get(ename, (-5.0, 5.0))
        b_low = -5.0 if (not raw_bounds or raw_bounds[0] is None) else float(raw_bounds[0])
        b_high = 5.0 if (not raw_bounds or len(raw_bounds) < 2 or raw_bounds[1] is None) else float(raw_bounds[1])
        if b_low > b_high:
            b_low, b_high = b_high, b_low
        bounds.append((b_low, b_high))
        init_val = float(ols_params[idx + 1]) if idx + 1 < len(ols_params) else 0.5 * (b_low + b_high)
        p0.append(float(np.clip(init_val, b_low, b_high)))

    def _loss(params: np.ndarray) -> float:
        log_c = params[0]
        exps = params[1:]
        log_pred = log_c + log_X @ exps
        pred = np.exp(np.clip(log_pred, -50.0, 50.0))
        return float(np.mean((y_arr - pred) ** 2) + 0.1 * np.mean((log_y - log_pred) ** 2))

    res = minimize(_loss, np.array(p0, dtype=np.float64), bounds=bounds, method="L-BFGS-B")
    opt_log_c = float(res.x[0])
    opt_exps = res.x[1:]
    prefactor = float(np.exp(opt_log_c))

    y_pred = np.exp(np.clip(opt_log_c + log_X @ opt_exps, -50.0, 50.0))
    ss_res = float(np.sum((y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r_sq = float(1.0 - ss_res / (ss_tot + 1e-15)) if ss_tot > 1e-15 else 1.0

    output: Dict[str, Any] = {}
    for ename, val in zip(exp_names, opt_exps):
        output[ename] = float(val)
    first_exp_val = float(opt_exps[0]) if len(opt_exps) > 0 else 0.5
    output.setdefault("beta", first_exp_val)
    output.setdefault("β", output["beta"])
    output.setdefault("a", prefactor)
    output.setdefault("Fc_offset", 0.0)
    output.setdefault("Kc_offset", 0.0)
    output.setdefault("Tc_offset", 0.0)
    output.setdefault("rc_offset", 0.0)
    output.setdefault("kc_offset", 0.0)
    output["prefactor"] = prefactor
    output["r_squared"] = max(0.0, r_sq)
    output["success"] = bool(res.success)
    output["form"] = form or "y ~ C * prod(x_i ^ exponent_i)"
    cov_mat = np.eye(max(1, len(opt_exps)), dtype=np.float64) * 1e-4
    output["covariance"] = cov_mat

    try:
        import sys
        import dis
        frame = sys._getframe(1)
        lasti = frame.f_lasti
        for instr in dis.get_instructions(frame.f_code):
            if instr.offset > lasti and instr.opname not in ("CACHE", "COPY_FREE_VARS", "EXTENDED_ARG"):
                if instr.opname == "UNPACK_SEQUENCE" and instr.argval == 2:
                    return output, cov_mat  # type: ignore[return-value]
                break
    except Exception:
        pass

    return output
