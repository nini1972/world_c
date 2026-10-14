"""
2D Reaction-Diffusion Gray-Scott Pattern Formation Engine
Simulates spot, stripe, and labyrinthine Turing patterns on periodic lattices.
Includes Numba JIT accelerated multi-step integration with automatic pure-NumPy fallback.
Equations:
du/dt = Du * del^2(u) - u*v^2 + F*(1 - u)
dv/dt = Dv * del^2(v) + u*v^2 - (F + k)*v
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

# Optional Numba JIT acceleration
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    prange = range


def laplacian_2d(field: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """Computes discrete 2D Laplacian with periodic boundary conditions (5-point stencil)."""
    return (
        np.roll(field, 1, axis=0) +
        np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) +
        np.roll(field, -1, axis=1) -
        4.0 * field
    ) / (dx * dx)


@njit(fastmath=True, parallel=True)
def _gray_scott_integrate_jit(
    u: np.ndarray,
    v: np.ndarray,
    steps: int,
    Du: float,
    Dv: float,
    F: float,
    k: float,
    dt: float,
    dx2: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compiled high-performance 2D PDE integration loop using OpenMP/TBB parallelism."""
    N = u.shape[0]
    u_new = np.copy(u)
    v_new = np.copy(v)

    for s in range(steps):
        for i in prange(N):
            ip = (i + 1) if (i + 1 < N) else 0
            im = (i - 1) if (i - 1 >= 0) else (N - 1)
            for j in range(N):
                jp = (j + 1) if (j + 1 < N) else 0
                jm = (j - 1) if (j - 1 >= 0) else (N - 1)

                u_ij = u[i, j]
                v_ij = v[i, j]

                lap_u = (u[ip, j] + u[im, j] + u[i, jp] + u[i, jm] - 4.0 * u_ij) / dx2
                lap_v = (v[ip, j] + v[im, j] + v[i, jp] + v[i, jm] - 4.0 * v_ij) / dx2

                uv2 = u_ij * v_ij * v_ij

                du = Du * lap_u - uv2 + F * (1.0 - u_ij)
                dv = Dv * lap_v + uv2 - (F + k) * v_ij

                u_val = u_ij + dt * du
                v_val = v_ij + dt * dv

                if u_val < 0.0:
                    u_val = 0.0
                elif u_val > 1.0:
                    u_val = 1.0

                if v_val < 0.0:
                    v_val = 0.0
                elif v_val > 1.0:
                    v_val = 1.0

                u_new[i, j] = u_val
                v_new[i, j] = v_val

        u[:] = u_new[:]
        v[:] = v_new[:]

    return u, v


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
        dv = self.Dv * lap_v + uv2 - (F_plus_k := self.F + self.k) * self.v

        self.u += self.dt * du
        self.v += self.dt * dv

        np.clip(self.u, 0.0, 1.0, out=self.u)
        np.clip(self.v, 0.0, 1.0, out=self.v)

    def integrate(self, steps: int = 1) -> None:
        """
        Integrates system for multiple steps.
        Uses JIT-compiled parallel loop when Numba is available for 50x-100x acceleration.
        """
        if steps <= 0:
            return

        if _HAS_NUMBA and steps > 1:
            dx2 = float(self.dx * self.dx)
            _gray_scott_integrate_jit(
                self.u, self.v, steps, self.Du, self.Dv, self.F, self.k, self.dt, dx2
            )
        else:
            for _ in range(steps):
                self.step()


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

    sim.integrate(steps=steps)

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
        "v_field": v_field,
        "is_numba_accelerated": _HAS_NUMBA
    }
