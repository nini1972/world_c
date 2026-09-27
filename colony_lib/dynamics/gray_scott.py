"""
2D Reaction-Diffusion Gray-Scott Pattern Formation Engine
Simulates spot, stripe, and labyrinthine Turing patterns on periodic lattices.
Equations:
du/dt = Du * del^2(u) - u*v^2 + F*(1 - u)
dv/dt = Dv * del^2(v) + u*v^2 - (F + k)*v
"""

import numpy as np
from typing import Dict, Any, Tuple

def laplacian_2d(field: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """Computes discrete 2D Laplacian with periodic boundary conditions (5-point stencil)."""
    return (
        np.roll(field, 1, axis=0) +
        np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) +
        np.roll(field, -1, axis=1) -
        4.0 * field
    ) / (dx * dx)

class GrayScott2D:
    def __init__(
        self,
        grid_size: int = 128,
        Du: float = 0.16,
        Dv: float = 0.08,
        F: float = 0.035,
        k: float = 0.060,
        dt: float = 1.0,
        dx: float = 1.0
    ):
        self.N = grid_size
        self.Du = Du
        self.Dv = Dv
        self.F = F
        self.k = k
        self.dt = dt
        self.dx = dx
        
        # State fields
        self.u = np.ones((self.N, self.N), dtype=np.float64)
        self.v = np.zeros((self.N, self.N), dtype=np.float64)
        
    def seed_center(self, radius: int = 10, noise_amp: float = 0.05, seed: int = 42):
        """Seeds a square patch in the center with perturbing noise to seed pattern emergence."""
        rng = np.random.default_rng(seed)
        c = self.N // 2
        r = radius
        self.u = np.ones((self.N, self.N), dtype=np.float64)
        self.v = np.zeros((self.N, self.N), dtype=np.float64)
        
        self.u[c - r:c + r, c - r:c + r] = 0.50
        self.v[c - r:c + r, c - r:c + r] = 0.25
        
        self.u += rng.uniform(-noise_amp, noise_amp, size=(self.N, self.N))
        self.v += rng.uniform(-noise_amp, noise_amp, size=(self.N, self.N))
        self.u = np.clip(self.u, 0.0, 1.0)
        self.v = np.clip(self.v, 0.0, 1.0)

    def step(self):
        """Forward Euler step with periodic 2D Laplacian."""
        uv2 = self.u * (self.v ** 2)
        lap_u = laplacian_2d(self.u, self.dx)
        lap_v = laplacian_2d(self.v, self.dx)
        
        du = self.Du * lap_u - uv2 + self.F * (1.0 - self.u)
        dv = self.Dv * lap_v + uv2 - (self.F + self.k) * self.v
        
        self.u += self.dt * du
        self.v += self.dt * dv
        
        np.clip(self.u, 0.0, 1.0, out=self.u)
        np.clip(self.v, 0.0, 1.0, out=self.v)

def simulate_gray_scott(
    grid_size: int = 64,
    F: float = 0.035,
    k: float = 0.060,
    steps: int = 1000,
    Du: float = 0.16,
    Dv: float = 0.08,
    dt: float = 1.0,
    seed: int = 42
) -> Dict[str, Any]:
    """Runs a complete 2D Gray-Scott integration and extracts morphological statistics."""
    sim = GrayScott2D(grid_size=grid_size, Du=Du, Dv=Dv, F=F, k=k, dt=dt)
    sim.seed_center(seed=seed)
    
    for _ in range(steps):
        sim.step()
        
    v_field = sim.v
    mean_v = float(np.mean(v_field))
    std_v = float(np.std(v_field))
    spot_coverage = float(np.mean(v_field > 0.2))
    
    return {
        "grid_size": grid_size,
        "F": F,
        "k": k,
        "steps": steps,
        "mean_v": mean_v,
        "std_v": std_v,
        "spot_coverage": spot_coverage,
        "v_field": v_field
    }
