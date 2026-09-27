"""
Normal Form Bifurcation Classification
Classifies 1D bifurcations (Saddle-Node, Transcritical, Pitchfork supercritical/subcritical)
from Taylor expansion coefficients near an equilibrium:
dx/dt = f(x, mu) approx a0(mu) + a1(mu)*x + a2(mu)*x^2 + a3(mu)*x^3
"""

import numpy as np
from typing import Dict, Any

def classify_bifurcation_1d(
    a0_mu: float,
    a1_mu: float,
    a2: float,
    a3: float
) -> Dict[str, Any]:
    """
    Classifies the local bifurcation type based on coefficients:
    - Saddle-Node: a0 changes sign with mu, a2 != 0
    - Transcritical: a0(mu=0)=0, a1 changes sign with mu, a2 != 0
    - Pitchfork (Supercritical): a0=0, a2=0, a1 changes sign, a3 < 0
    - Pitchfork (Subcritical): a0=0, a2=0, a1 changes sign, a3 > 0
    """
    if abs(a0_mu) > 1e-5:
        return {
            "type": "saddle_node",
            "normal_form": "dx/dt = mu - x^2",
            "critical_condition": "mu = 0"
        }
    elif abs(a2) > 1e-4:
        return {
            "type": "transcritical",
            "normal_form": "dx/dt = mu*x - x^2",
            "critical_condition": "mu = 0"
        }
    elif a3 < -1e-5:
        return {
            "type": "pitchfork_supercritical",
            "normal_form": "dx/dt = mu*x - x^3",
            "critical_condition": "mu > 0 (stable branches appear)"
        }
    elif a3 > 1e-5:
        return {
            "type": "pitchfork_subcritical",
            "normal_form": "dx/dt = mu*x + x^3",
            "critical_condition": "mu < 0 (unstable branches collapse)"
        }
    else:
        return {
            "type": "indeterminate",
            "normal_form": "dx/dt = mu*x",
            "critical_condition": "linear"
        }
