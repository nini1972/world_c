"""
Adler Transition and Phase-Slipping Dynamics Detector
Equation: d(phi)/dt = Delta - K * sin(phi)
Phase locked for K >= |Delta| (phi* = arcsin(Delta / K))
Phase slipping for K < |Delta| with slip frequency Omega = sqrt(Delta^2 - K^2)
"""

import numpy as np
from typing import Dict, Any, Tuple

class AdlerDetector:
    def __init__(self, delta: float = 1.0):
        self.delta = delta
        
    def simulate(self, K: float, t_max: float = 100.0, dt: float = 0.05) -> Dict[str, Any]:
        """
        Simulates the Adler equation d(phi)/dt = delta - K*sin(phi).
        """
        n_steps = int(t_max / dt)
        times = np.linspace(0, t_max, n_steps)
        phi = np.zeros(n_steps, dtype=np.float64)
        
        curr_phi = 0.0
        slip_count = 0
        last_wrapped = 0.0
        
        for i in range(1, n_steps):
            # RK4 step
            f = lambda p: self.delta - K * np.sin(p)
            k1 = f(curr_phi)
            k2 = f(curr_phi + 0.5 * dt * k1)
            k3 = f(curr_phi + 0.5 * dt * k2)
            k4 = f(curr_phi + dt * k3)
            curr_phi += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            phi[i] = curr_phi
            
            # Count 2*pi phase slips
            if curr_phi - last_wrapped >= 2.0 * np.pi:
                slip_count += 1
                last_wrapped += 2.0 * np.pi
                
        empirical_slip_freq = float(slip_count) / t_max * (2.0 * np.pi)
        theoretical_slip_freq = float(np.sqrt(max(0.0, self.delta**2 - K**2))) if abs(K) < abs(self.delta) else 0.0
        
        is_locked = bool(abs(K) >= abs(self.delta))
        
        return {
            "delta": self.delta,
            "K": K,
            "is_locked": is_locked,
            "slip_count": slip_count,
            "empirical_slip_freq": empirical_slip_freq,
            "theoretical_slip_freq": theoretical_slip_freq,
            "times": times,
            "phi": phi
        }

def detect_adler_transition(
    delta: float = 1.0,
    k_range: Tuple[float, float] = (0.0, 2.0),
    n_points: int = 50
) -> Dict[str, Any]:
    """
    Scans coupling K across the critical boundary K_c = |Delta|
    and measures the scaling of slip frequency Omega(K).
    """
    ks = np.linspace(k_range[0], k_range[1], n_points)
    emp_freqs = []
    theo_freqs = []
    
    detector = AdlerDetector(delta=delta)
    for k in ks:
        res = detector.simulate(K=float(k), t_max=60.0)
        emp_freqs.append(res["empirical_slip_freq"])
        theo_freqs.append(res["theoretical_slip_freq"])
        
    return {
        "delta": delta,
        "k_critical": abs(delta),
        "K_values": ks,
        "empirical_frequencies": np.array(emp_freqs),
        "theoretical_frequencies": np.array(theo_freqs)
    }
