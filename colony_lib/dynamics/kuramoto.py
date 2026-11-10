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

@njit(fastmath=True)
def _simulate_kuramoto_all_to_all_jit(
    theta_init: np.ndarray,
    omega: np.ndarray,
    K: float,
    dt: float,
    steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compiled multi-step RK4 integration for all-to-all Kuramoto ensemble."""
    N = len(theta_init)
    r_series = np.empty(steps, dtype=np.float64)
    th = np.copy(theta_init)
    
    for s in range(steps):
        cos_sum = np.sum(np.cos(th))
        sin_sum = np.sum(np.sin(th))
        r = np.sqrt(cos_sum * cos_sum + sin_sum * sin_sum) / N
        r_series[s] = r
        
        k1 = _kuramoto_rhs_all_to_all(th, omega, K, N)
        k2 = _kuramoto_rhs_all_to_all(th + 0.5 * dt * k1, omega, K, N)
        k3 = _kuramoto_rhs_all_to_all(th + 0.5 * dt * k2, omega, K, N)
        k4 = _kuramoto_rhs_all_to_all(th + dt * k3, omega, K, N)
        th = (th + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        
    return th, r_series

@njit(fastmath=True)
def _simulate_kuramoto_matrix_jit(
    theta_init: np.ndarray,
    omega: np.ndarray,
    A: np.ndarray,
    K: float,
    dt: float,
    steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compiled multi-step RK4 integration for arbitrary topology network matrix."""
    N = len(theta_init)
    r_series = np.empty(steps, dtype=np.float64)
    th = np.copy(theta_init)
    
    for s in range(steps):
        cos_sum = np.sum(np.cos(th))
        sin_sum = np.sum(np.sin(th))
        r = np.sqrt(cos_sum * cos_sum + sin_sum * sin_sum) / N
        r_series[s] = r
        
        k1 = _kuramoto_rhs_matrix(th, omega, A, K, N)
        k2 = _kuramoto_rhs_matrix(th + 0.5 * dt * k1, omega, A, K, N)
        k3 = _kuramoto_rhs_matrix(th + 0.5 * dt * k2, omega, A, K, N)
        k4 = _kuramoto_rhs_matrix(th + dt * k3, omega, A, K, N)
        th = (th + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))
        
    return th, r_series

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
    Supports both keyword initialization:
        KuramotoEnsemble(n_oscillators=200, natural_frequencies=omega, coupling_matrix=A)
    and legacy positional initialization:
        KuramotoEnsemble(N, K, natural_frequencies, dt)
    """
    def __init__(
        self,
        n_oscillators: Optional[int] = None,
        natural_frequencies: Optional[Any] = None,
        coupling_matrix: Optional[Any] = None,
        freq_distribution: Any = "cauchy",
        freq_scale: float = 1.0,
        seed: Optional[int] = None,
        K: Optional[float] = None,
        dt: Optional[float] = None,
        N: Optional[int] = None,
        **kwargs
    ):
        if n_oscillators is None:
            n_oscillators = N if N is not None else kwargs.get("N", 100)
        self.N = int(n_oscillators)
        k_init = K if K is not None else kwargs.get("coupling", kwargs.get("coupling_strength", kwargs.get("k", 1.0)))
        self.K = float(k_init)
        self.coupling = self.K
        self.dt = float(dt if dt is not None else kwargs.get("dt", 0.05))
        rng_seed = seed if seed is not None else kwargs.get("seed", None)
        self.rng = np.random.default_rng(rng_seed)

        # Detect legacy positional call: KuramotoEnsemble(N, K, natural_frequencies, dt)
        if natural_frequencies is not None and np.ndim(natural_frequencies) == 0:
            self.K = float(natural_frequencies)
            if coupling_matrix is not None and np.ndim(coupling_matrix) == 1:
                natural_frequencies = coupling_matrix
                coupling_matrix = None
            else:
                natural_frequencies = None
            if isinstance(freq_distribution, (int, float, np.floating, np.integer)):
                self.dt = float(freq_distribution)
                freq_distribution = "cauchy"

        if natural_frequencies is not None:
            omega_arr = np.asarray(natural_frequencies, dtype=np.float64)
            if omega_arr.ndim == 0:
                self.omega = np.full(self.N, float(omega_arr), dtype=np.float64)
            else:
                if len(omega_arr) != self.N:
                    raise ValueError(f"Frequencies length ({len(omega_arr)}) != N ({self.N})")
                self.omega = omega_arr
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
                
        self.A = np.asarray(coupling_matrix, dtype=np.float64) if coupling_matrix is not None else None
        self.theta = self.rng.uniform(-np.pi, np.pi, size=self.N).astype(np.float64)
        self.phases: np.ndarray = np.empty((0, self.N), dtype=np.float64)
        self.r_series: np.ndarray = np.empty(0, dtype=np.float64)

    def run(
        self,
        t_max: float = 50.0,
        dt_or_K: Optional[float] = None,
        dt: Optional[float] = None,
        K: Optional[float] = None,
        T: Optional[float] = None,
        **kwargs
    ) -> Tuple[np.ndarray, list]:
        """
        Integrates the ensemble up to t_max, recording phase history in `self.phases`
        (shape: (n_steps, N)) and order parameter history in `self.r_series`.
        Returns `(times, order_parameters)` tuple so `times, r = ensemble.run(T, dt)` unpacks cleanly.
        """
        total_t = float(T) if T is not None else float(t_max)
        # Disambiguate 2nd positional arg: if dt is None and dt_or_K is small (<= 1.0), treat as dt
        if dt is not None:
            dt_val = float(dt)
            k_val = float(K) if K is not None else (float(dt_or_K) if dt_or_K is not None else self.K)
        elif dt_or_K is not None:
            if K is None and float(dt_or_K) <= 1.0:
                dt_val = float(dt_or_K)
                k_val = float(self.K)
            else:
                k_val = float(dt_or_K)
                dt_val = float(self.dt)
        else:
            dt_val = float(self.dt)
            k_val = float(K) if K is not None else float(self.K)

        # If caller initialized `ensemble.phases = rng.uniform(0, 2*pi, N)` as 1D initial state, adopt it
        if isinstance(self.phases, np.ndarray) and self.phases.ndim == 1 and len(self.phases) == self.N:
            self.theta = np.asarray(self.phases, dtype=np.float64).copy()
        elif len(self.theta) != self.N:
            self.theta = self.rng.uniform(-np.pi, np.pi, size=self.N).astype(np.float64)

        if len(self.omega) != self.N:
            self.omega = self.rng.normal(0.0, 1.0, size=self.N).astype(np.float64)

        n_steps = max(1, int(round(total_t / dt_val)))
        times = np.linspace(0.0, total_t, n_steps, endpoint=False)
        phases = np.empty((n_steps, self.N), dtype=np.float64)
        r_series = np.empty(n_steps, dtype=np.float64)
        for s in range(n_steps):
            phases[s] = self.theta
            r, _ = compute_order_parameter(self.theta)
            r_series[s] = r
            self.step_rk4(k_val, dt_val)
        self.phases = phases
        self.r_series = r_series
        self.times = times
        return times, r_series.tolist()

    def simulate(self, T: Any = 50.0, dt: Optional[float] = None, K: Optional[float] = None, **kwargs) -> Any:
        """Alias for run(T, dt) returning (times, order_parameters), or 2D phase trajectory
        `self.phases` (shape: (n_steps, N)) when a 1D initial phase vector `theta0` is passed
        as the first positional argument `ensemble.simulate(theta0, T, dt)`."""
        if isinstance(T, (np.ndarray, list, tuple)):
            self.theta = np.asarray(T, dtype=np.float64).copy()
            if "t_max" in kwargs or "T" in kwargs:
                t_max_val = float(kwargs.pop("t_max", kwargs.pop("T", 50.0)))
                dt_val = float(dt) if dt is not None else float(kwargs.pop("dt", self.dt))
                k_val = float(K) if K is not None else None
            elif dt is not None and float(dt) > 1.0:
                t_max_val = float(dt)
                dt_val = float(K) if (K is not None and float(K) <= 1.0) else float(kwargs.pop("dt", self.dt))
                k_val = float(K) if (K is not None and float(K) > 1.0) else None
            else:
                t_max_val = 50.0
                dt_val = float(dt) if dt is not None else float(self.dt)
                k_val = float(K) if K is not None else None
            self.run(t_max=t_max_val, dt=dt_val, K=k_val, **kwargs)
            return self.phases
        if "t_max" in kwargs:
            T = kwargs.pop("t_max")
        return self.run(t_max=T, dt=dt, K=K, **kwargs)

    def step(self, dt: Optional[float] = None, K: Optional[float] = None) -> float:
        """Advances oscillator phases by one step and returns current order parameter r."""
        dt_val = float(dt) if dt is not None else float(self.dt)
        k_val = float(K) if K is not None else float(self.K)
        if isinstance(self.phases, np.ndarray) and self.phases.ndim == 1 and len(self.phases) == self.N:
            self.theta = np.asarray(self.phases, dtype=np.float64).copy()
            self.phases = np.empty((0, self.N), dtype=np.float64)
        self.step_rk4(k_val, dt_val)
        r, _ = compute_order_parameter(self.theta)
        return float(r)

    def order_parameter(self) -> float:
        """Returns current scalar Kuramoto order parameter r."""
        th = self.phases if (isinstance(self.phases, np.ndarray) and self.phases.ndim == 1 and len(self.phases) == self.N) else self.theta
        r, _ = compute_order_parameter(th)
        return float(r)
        
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
    natural_frequencies: Optional[np.ndarray] = None,
    freq_distribution: str = "cauchy",
    freq_scale: float = 1.0,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Runs a complete Kuramoto integration and returns time series of the order parameter,
    steady-state order parameter, and susceptibility.
    """
    ensemble = KuramotoEnsemble(
        n_oscillators=n_oscillators,
        natural_frequencies=natural_frequencies,
        coupling_matrix=coupling_matrix,
        freq_distribution=freq_distribution,
        freq_scale=freq_scale,
        seed=seed,
        K=K,
        dt=dt
    )
    
    n_steps = int(t_max / dt)
    n_transient = int(n_steps * transient_fraction)
    
    times = np.linspace(0, t_max, n_steps)
    
    if _HAS_NUMBA:
        if ensemble.A is None:
            ensemble.theta, r_series = _simulate_kuramoto_all_to_all_jit(
                ensemble.theta, ensemble.omega, K, dt, n_steps
            )
        else:
            ensemble.theta, r_series = _simulate_kuramoto_matrix_jit(
                ensemble.theta, ensemble.omega, ensemble.A, K, dt, n_steps
            )
    else:
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
        "time": times,
        "r_series": r_series,
        "order_param_magnitude": r_series,
        "order_parameters": r_series,
        "r": r_series,
        "phases": ensemble.theta,
        "theta": ensemble.theta,
        "steady_mean_r": mean_r,
        "steady_state_order_parameter": mean_r,
        "order_parameter": mean_r,
        "mean_R": mean_r,
        "R": steady_r,
        "steady_std_r": std_r,
        "susceptibility": susceptibility
    }

def vectorized_kuramoto(
    N: Optional[int] = None,
    n_oscillators: Optional[int] = None,
    K: float = 2.0,
    omega: Optional[np.ndarray] = None,
    natural_frequencies: Optional[np.ndarray] = None,
    t_transient: float = 200.0,
    t_measure: float = 800.0,
    t_max: Optional[float] = None,
    dt: float = 0.05,
    coupling_matrix: Optional[np.ndarray] = None,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    High-performance vectorized Kuramoto simulation matching agent expectations.
    Returns dictionary with 'R' (steady-state time series), 'r_series' (full series),
    'mean_R', and 'var_R'.
    """
    n = N if N is not None else (n_oscillators or 100)
    freqs = omega if omega is not None else natural_frequencies
    total_time = t_max if t_max is not None else (t_transient + t_measure)
    
    ensemble = KuramotoEnsemble(
        n_oscillators=n,
        natural_frequencies=freqs,
        coupling_matrix=coupling_matrix,
        seed=seed
    )
    
    n_steps = max(1, int(total_time / dt))
    n_trans = int(t_transient / dt)
    r_series = np.zeros(n_steps, dtype=np.float64)
    
    for step in range(n_steps):
        r, _ = compute_order_parameter(ensemble.theta)
        r_series[step] = r
        ensemble.step_rk4(K, dt)
        
    steady_r = r_series[n_trans:] if n_trans < n_steps else r_series
    mean_r = float(np.mean(steady_r)) if len(steady_r) > 0 else 0.0
    var_r = float(np.var(steady_r)) if len(steady_r) > 0 else 0.0
    std_r = float(np.std(steady_r)) if len(steady_r) > 0 else 0.0
    susceptibility = float(n * var_r)
    
    return {
        "N": n,
        "K": K,
        "R": steady_r,
        "r_series": r_series,
        "mean_R": mean_r,
        "var_R": var_r,
        "steady_mean_r": mean_r,
        "steady_std_r": std_r,
        "susceptibility": susceptibility
    }

