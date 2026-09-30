"""
Turing Pattern & Morphogenesis Instability Analysis
Implements linear stability analysis, Jacobian evaluation, and dispersion relations
for 2-component reaction-diffusion systems:
  ∂u/∂t = Du ∇²u + f(u, v)
  ∂v/∂t = Dv ∇²v + g(u, v)
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional

def turing_dispersion_relation(
    k_vals: np.ndarray,
    Du: float,
    Dv: float,
    J: np.ndarray
) -> np.ndarray:
    """
    Computes the maximal real part of the eigenvalue lambda(k) across wavenumbers k.
    Turing instability occurs if Re(lambda(k)) > 0 for some k > 0 while stable at k=0.
    """
    f_u, f_v = J[0, 0], J[0, 1]
    g_u, g_v = J[1, 0], J[1, 1]
    
    growth_rates = []
    for k in k_vals:
        k2 = float(k ** 2)
        # Characteristic equation: lambda^2 - Tr_k * lambda + Det_k = 0
        tr_k = (f_u - Du * k2) + (g_v - Dv * k2)
        det_k = (f_u - Du * k2) * (g_v - Dv * k2) - f_v * g_u
        disc = tr_k ** 2 - 4.0 * det_k
        if disc >= 0:
            lam = (tr_k + np.sqrt(disc)) / 2.0
        else:
            lam = tr_k / 2.0
        growth_rates.append(lam)
    return np.array(growth_rates, dtype=np.float64)

def check_turing_conditions(
    Du: float,
    Dv: float,
    J: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates the 4 necessary and sufficient conditions for Turing instability:
    1. Tr(J) = f_u + g_v < 0 (homogeneous stability)
    2. Det(J) = f_u*g_v - f_v*g_u > 0 (homogeneous stability)
    3. Du*g_v + Dv*f_u > 0 (diffusion-driven destabilization requires Dv != Du)
    4. (Du*g_v + Dv*f_u)^2 - 4*Du*Dv*Det(J) > 0 (critical wavenumber exists)
    """
    f_u, f_v = float(J[0, 0]), float(J[0, 1])
    g_u, g_v = float(J[1, 0]), float(J[1, 1])
    
    tr_J = f_u + g_v
    det_J = f_u * g_v - f_v * g_u
    diff_term = Du * g_v + Dv * f_u
    disc_term = diff_term ** 2 - 4.0 * Du * Dv * det_J
    
    c1 = bool(tr_J < 0)
    c2 = bool(det_J > 0)
    c3 = bool(diff_term > 0)
    c4 = bool(disc_term > 0)
    
    is_turing = c1 and c2 and c3 and c4
    
    kc = 0.0
    if is_turing and (Du * Dv > 0):
        # Critical wavenumber: k_c^2 = (Du*g_v + Dv*f_u) / (2*Du*Dv)
        kc2 = diff_term / (2.0 * Du * Dv)
        if kc2 > 0:
            kc = float(np.sqrt(kc2))
            
    return {
        "is_turing_unstable": is_turing,
        "condition_1_trace_negative": c1,
        "condition_2_det_positive": c2,
        "condition_3_diffusion_destabilizing": c3,
        "condition_4_critical_wavenumber_exists": c4,
        "critical_wavenumber_kc": kc,
        "critical_wavelength": float(2.0 * np.pi / kc) if kc > 0 else None,
        "trace": float(tr_J),
        "determinant": float(det_J)
    }
