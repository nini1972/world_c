"""
High-Performance Vectorized Kuramoto Oscillator Simulation
Supports all-to-all and complex network topologies, order parameter dynamics, and susceptibility scaling.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

# Optional Numba JIT acceleration
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

@njit(fastmath=True)
def _kuramoto_rhs_all_to_all(theta: np.ndarray, omega: np.ndarray, K: float, N: int) -> np.ndarray:
    """Computes d(theta)/dt using mean-field order parameter representation in O(N)."""
    # r * e^(i psi) = (1/N) * sum_j e^(i theta_j)
    cos_sum = np.sum(np.cos(theta))
    sin_sum = np.sum(np.sin(theta))
    r = np.sqrt(cos_sum * cos_sum + sin_sum * sin_sum) / N
    psi = np.arctan2(sin_sum, cos_sum)
    return omega + K * r * np.sin(psi - theta)

@njit(fastmath=True)
def _kuramoto_rhs_matrix(theta: np.ndarray, omega: np.ndarray, A: np.ndarray, K: float, N: int) -> np.ndarray:
    """Computes d(theta)/dt for general network topology matrix A in O(N^2)."""
    dtheta = np.copy(omega)
    for i in range(N):
        s = 0.0
        for j in range(N):
            s += A[i, j] * np.sin(theta[j] - theta[i])
        dtheta[i] += (K / N) * s
    return dtheta

def compute_order_parameter(theta: np.ndarray) -> Tuple[float, float]:
    """
    Computes global Kuramoto order parameter r and mean phase psi:
    r * e^(i psi) = (1/N) * sum_{j=1}^N e^(i theta_j)
    """
    z = np.mean(np.exp(1j * theta))
    r = float(np.abs(z))
    psi = float(np.angle(z))
    return r, psi

class KuramotoEnsemble:
    """
    Represents an ensemble of coupled Kuramoto oscillators.
    """
    def __init__(
        self,
        n_oscillators: int = 100,
        natural_frequencies: Optional[np.ndarray] = None,
        coupling_matrix: Optional[np.ndarray] = None,
        freq_distribution: str = "cauchy",
        freq_scale: float = 1.0,
        seed: Optional[int] = None
    ):
        self.N = n_oscillators
        self.rng = np.random.default_rng(seed)
        
        if natural_frequencies is not None:
            if len(natural_frequencies) != self.N:
                raise ValueError(f"Frequencies length ({len(natural_frequencies)}) != N ({self.N})")
            self.omega = np.asarray(natural_frequencies, dtype=np.float64)
        else:
            if freq_distribution == "cauchy":
                # Standard Cauchy distribution with scale gamma
                u = self.rng.uniform(0.01, 0.99, size=self.N)
                self.omega = freq_scale * np.tan(np.pi * (u - 0.5))
            elif freq_distribution == "gaussian":
                self.omega = self.rng.normal(0.0, freq_scale, size=self.N)
            elif freq_distribution == "uniform":
                self.omega = self.rng.uniform(-freq_scale, freq_scale, size=self.N)
            else:
                raise ValueError(f"Unknown frequency distribution: {freq_distribution}")
                
        self.A = coupling_matrix.astype(np.float64) if coupling_matrix is not None else None
        self.theta = self.rng.uniform(-np.pi, np.pi, size=self.N).astype(np.float64)
        
    def step_rk4(self, K: float, dt: float):
        """Advances oscillator phases by dt using 4th-order Runge-Kutta."""
        N = self.N
        omega = self.omega
        A = self.A
        th = self.theta
        
        if A is None:
            # O(N) mean-field step
            k1 = _kuramoto_rhs_all_to_all(th, omega, K, N)
            k2 = _kuramoto_rhs_all_to_all(th + 0.5 * dt * k1, omega, K, N)
            k3 = _kuramoto_rhs_all_to_all(th + 0.5 * dt * k2, omega, K, N)
            k4 = _kuramoto_rhs_all_to_all(th + dt * k3, omega, K, N)
        else:
            # O(N^2) network step
            k1 = _kuramoto_rhs_matrix(th, omega, A, K, N)
            k2 = _kuramoto_rhs_matrix(th + 0.5 * dt * k1, omega, A, K, N)
            k3 = _kuramoto_rhs_matrix(th + 0.5 * dt * k2, omega, A, K, N)
            k4 = _kuramoto_rhs_matrix(th + dt * k3, omega, A, K, N)
            
        self.theta = (th + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)) % (2 * np.pi)

def simulate_kuramoto(
    n_oscillators: int = 200,
    K: float = 2.0,
    t_max: float = 50.0,
    dt: float = 0.05,
    transient_fraction: float = 0.5,
    coupling_matrix: Optional[np.ndarray] = None,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Runs a complete Kuramoto integration and returns time series of the order parameter,
    steady-state order parameter, and susceptibility.
    """
    ensemble = KuramotoEnsemble(
        n_oscillators=n_oscillators,
        coupling_matrix=coupling_matrix,
        seed=seed
    )
    
    n_steps = int(t_max / dt)
    n_transient = int(n_steps * transient_fraction)
    
    times = np.linspace(0, t_max, n_steps)
    r_series = np.zeros(n_steps, dtype=np.float64)
    
    for step in range(n_steps):
        r, _ = compute_order_parameter(ensemble.theta)
        r_series[step] = r
        ensemble.step_rk4(K, dt)
        
    steady_r = r_series[n_transient:]
    mean_r = float(np.mean(steady_r))
    std_r = float(np.std(steady_r))
    susceptibility = float(n_oscillators * (std_r ** 2))
    
    return {
        "N": n_oscillators,
        "K": K,
        "t_max": t_max,
        "dt": dt,
        "times": times,
        "r_series": r_series,
        "steady_mean_r": mean_r,
        "steady_std_r": std_r,
        "susceptibility": susceptibility
    }
