"""
Strang-Splitting Spectral FFT Integrator for Classical Phi^4 Solitons
Simulates relativistic topological kink-antikink collisions, resonant bounce windows, and energy conservation.
Equation: d^2(phi)/dt^2 - d^2(phi)/dx^2 + phi*(phi^2 - 1) = 0
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

class Phi4SolitonSpectral:
    """
    Fourier pseudo-spectral integrator for 1D relativistic phi^4 field theory.
    Uses Strang splitting: exact kinetic spectral step + nonlinear potential RK4 step.
    """
    def __init__(
        self,
        N_grid: int = 512,
        L: float = 80.0,
        dt: float = 0.02
    ):
        self.N = N_grid
        self.L = L
        self.dx = 2.0 * L / N_grid
        self.dt = dt
        self.x = np.linspace(-L, L, N_grid, endpoint=False)
        
        # Spectral wavenumbers for spatial derivatives
        self.k = 2.0 * np.pi * np.fft.fftfreq(N_grid, d=self.dx)
        self.k2 = self.k ** 2

    def initialize_kink_antikink(self, v: float, x0: float = 12.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs relativistic kink at -x0 moving right (+v)
        and antikink at +x0 moving left (-v).
        Lorentz factor gamma = 1 / sqrt(1 - v^2).
        Single kink: phi_K(x, t) = tanh((x - v*t)*gamma / sqrt(2)).
        """
        if v >= 1.0 or v <= 0.0:
            raise ValueError(f"Velocity v must be strictly between 0 and 1, got {v}")
            
        gamma = 1.0 / np.sqrt(1.0 - v**2)
        scale = gamma / np.sqrt(2.0)
        
        # Initial field configuration
        phi = np.tanh(scale * (self.x + x0)) - np.tanh(scale * (self.x - x0)) - 1.0
        
        # Initial velocity field d(phi)/dt = pi_field
        # d/dt [tanh(scale*(x + x0 - v*t))] at t=0 = -v*scale * sech^2(scale*(x + x0))
        # d/dt [-tanh(scale*(x - x0 + v*t))] at t=0 = -v*scale * sech^2(scale*(x - x0))
        dphi_dt = -v * scale * (1.0 / np.cosh(scale * (self.x + x0)))**2 - \
                   v * scale * (1.0 / np.cosh(scale * (self.x - x0)))**2
                   
        return phi, dphi_dt

    def compute_energy(self, phi: np.ndarray, pi_field: np.ndarray) -> float:
        """
        Total Hamiltonian energy:
        E = integral [ 1/2*(pi)^2 + 1/2*(d_phi/dx)^2 + 1/4*(phi^2 - 1)^2 ] dx
        """
        # Spatial derivative via spectral FFT
        phi_k = np.fft.fft(phi)
        dphi_dx = np.real(np.fft.ifft(1j * self.k * phi_k))
        
        kinetic = 0.5 * (pi_field ** 2)
        gradient = 0.5 * (dphi_dx ** 2)
        potential = 0.25 * ((phi ** 2 - 1.0) ** 2)
        
        density = kinetic + gradient + potential
        return float(np.sum(density) * self.dx)

    def step(self, phi: np.ndarray, pi_field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Strang splitting step:
        1. Half-step nonlinear potential force on pi
        2. Full-step linear wave equation (d_tt phi - d_xx phi = 0) in Fourier domain
        3. Half-step nonlinear potential force on pi
        """
        dt = self.dt
        
        # Nonlinear force: F_nl(phi) = -d/dphi [1/4*(phi^2 - 1)^2] = -phi*(phi^2 - 1)
        force_nl = lambda p: -p * (p**2 - 1.0)
        
        # 1. Half step on momentum
        pi_half = pi_field + 0.5 * dt * force_nl(phi)
        
        # 2. Exact linear wave equation step in Fourier domain:
        # d/dt [phi_k, pi_k]^T = [pi_k, -k^2 * phi_k]^T
        phi_k = np.fft.fft(phi)
        pi_k = np.fft.fft(pi_half)
        
        omega = np.abs(self.k)
        # Avoid division by zero at k=0
        mask = omega > 1e-12
        cos_wt = np.cos(omega * dt)
        sin_wt = np.sin(omega * dt)
        
        phi_k_new = np.copy(phi_k)
        pi_k_new = np.copy(pi_k)
        
        # For k > 0
        phi_k_new[mask] = phi_k[mask] * cos_wt[mask] + (pi_k[mask] / omega[mask]) * sin_wt[mask]
        pi_k_new[mask] = -phi_k[mask] * omega[mask] * sin_wt[mask] + pi_k[mask] * cos_wt[mask]
        
        # For k == 0
        zero_k = ~mask
        phi_k_new[zero_k] = phi_k[zero_k] + dt * pi_k[zero_k]
        pi_k_new[zero_k] = pi_k[zero_k]
        
        phi_new = np.real(np.fft.ifft(phi_k_new))
        pi_mid = np.real(np.fft.ifft(pi_k_new))
        
        # 3. Second half step on momentum
        pi_new = pi_mid + 0.5 * dt * force_nl(phi_new)
        
        return phi_new, pi_new

def simulate_phi4_collision(
    v: float = 0.20,
    x0: float = 12.0,
    t_max: float = 80.0,
    N_grid: int = 512,
    L: float = 60.0,
    dt: float = 0.02
) -> Dict[str, Any]:
    """
    Simulates a complete kink-antikink collision and records center-of-mass trajectory phi(0, t),
    collision duration, and energy conservation ratio.
    """
    solver = Phi4SolitonSpectral(N_grid=N_grid, L=L, dt=dt)
    phi, pi_field = solver.initialize_kink_antikink(v=v, x0=x0)
    
    n_steps = int(t_max / dt)
    times = np.linspace(0, t_max, n_steps)
    center_phi = np.zeros(n_steps, dtype=np.float64)
    energy_series = np.zeros(n_steps, dtype=np.float64)
    
    initial_energy = solver.compute_energy(phi, pi_field)
    
    # Track center value phi(x=0, t)
    center_idx = N_grid // 2
    
    for step in range(n_steps):
        center_phi[step] = phi[center_idx]
        energy_series[step] = solver.compute_energy(phi, pi_field)
        phi, pi_field = solver.step(phi, pi_field)
        
    final_energy = energy_series[-1]
    energy_drift = abs(final_energy - initial_energy) / (initial_energy + 1e-12)
    
    # Detect bounces: number of times center_phi crosses 0 from below
    crossings = np.sum((center_phi[:-1] < 0.0) & (center_phi[1:] >= 0.0))
    
    return {
        "velocity": v,
        "initial_x0": x0,
        "t_max": t_max,
        "times": times,
        "center_phi": center_phi,
        "bounce_count": int(crossings),
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "energy_drift": float(energy_drift)
    }
