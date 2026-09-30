"""
Lenia: Continuous Cellular Automata with Gaussian Kernels
Implements continuous space, continuous time, and continuous state artificial life.
Uses concentric Gaussian ring kernels and Gaussian growth mappings to produce
self-organizing emergent solitons and gliders.
"""
import numpy as np
from scipy.signal import fftconvolve
from typing import Tuple, Dict, Any, Optional

class Lenia2D:
    """
    2D Lenia Continuous Cellular Automaton simulator.
    Convolution with normalized Gaussian ring kernel followed by Gaussian growth mapping.
    """
    def __init__(
        self,
        grid_size: int = 128,
        kernel_radius: int = 13,
        mu_k: float = 0.5,
        sigma_k: float = 0.15,
        mu_g: float = 0.135,
        sigma_g: float = 0.015,
        dt: float = 0.1
    ):
        self.size = grid_size
        self.R = kernel_radius
        self.mu_k = mu_k
        self.sigma_k = sigma_k
        self.mu_g = mu_g
        self.sigma_g = sigma_g
        self.dt = dt
        self.kernel = self._build_gaussian_kernel()
        self.state = np.zeros((grid_size, grid_size), dtype=np.float64)

    def _build_gaussian_kernel(self) -> np.ndarray:
        """Constructs the normalized annular Gaussian kernel K(r)."""
        rad = self.R
        y, x = np.ogrid[-rad:rad+1, -rad:rad+1]
        r = np.sqrt(x**2 + y**2) / rad
        # Gaussian ring centered at mu_k
        k = np.exp(-((r - self.mu_k) ** 2) / (2.0 * self.sigma_k ** 2))
        k[r > 1.0] = 0.0
        norm = np.sum(k)
        return k / norm if norm > 0 else k

    def growth_function(self, u: np.ndarray) -> np.ndarray:
        """Gaussian growth function G(u) mapping potential to state change rate [-1, 1]."""
        return 2.0 * np.exp(-((u - self.mu_g) ** 2) / (2.0 * self.sigma_g ** 2)) - 1.0

    def step(self) -> np.ndarray:
        """Executes a single continuous Euler integration step."""
        u = fftconvolve(self.state, self.kernel, mode='same')
        g = self.growth_function(u)
        self.state = np.clip(self.state + self.dt * g, 0.0, 1.0)
        return self.state

    def seed_orbium(self, center: Optional[Tuple[int, int]] = None):
        """Seeds a localized smooth circular Gaussian soliton blob."""
        cx = center[0] if center else self.size // 2
        cy = center[1] if center else self.size // 2
        y, x = np.ogrid[:self.size, :self.size]
        d = np.sqrt((x - cx)**2 + (y - cy)**2)
        blob = np.exp(-(d**2) / (2.0 * (self.R * 0.7)**2))
        self.state = np.clip(blob, 0.0, 1.0)

    def run(self, steps: int = 50) -> Dict[str, Any]:
        """Runs the simulation for N steps and returns trajectory metrics."""
        masses = []
        for _ in range(steps):
            self.step()
            masses.append(float(np.sum(self.state)))
        return {
            "final_mass": float(np.sum(self.state)),
            "mass_trajectory": masses,
            "max_amplitude": float(np.max(self.state)),
            "active_area_fraction": float(np.mean(self.state > 0.05))
        }
