"""
Contact Process Module: 1D and 2D Stochastic Contact Process (Directed Percolation universality class).
"""

import numpy as np
from typing import Tuple, Optional

def contact_process(
    grid: np.ndarray,
    birth_rate: float,
    death_rate: float = 1.0,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, float]:
    """
    Executes one time step of the 2D contact process on a periodic lattice.

    Args:
        grid: 2D numpy array (0 = inactive, 1 = active).
        birth_rate: Probability of birth per active neighbor.
        death_rate: Probability of death per active site.
        rng: Optional NumPy random generator.

    Returns:
        Tuple of (updated_grid, active_site_density).
    """
    if rng is None:
        rng = np.random.default_rng()

    L = grid.shape[0]
    new_grid = grid.copy()

    # Death step
    death_mask = (rng.random(grid.shape) < death_rate) & (grid == 1)
    new_grid[death_mask] = 0

    # Neighbor convolution (4 nearest neighbors, periodic boundary)
    nbrs = (
        np.roll(grid, 1, axis=0) +
        np.roll(grid, -1, axis=0) +
        np.roll(grid, 1, axis=1) +
        np.roll(grid, -1, axis=1)
    )

    # Birth probability: 1 - (1 - birth_rate)^nbrs
    empty_mask = (new_grid == 0)
    birth_prob = 1.0 - (1.0 - birth_rate) ** nbrs
    birth_mask = empty_mask & (rng.random(grid.shape) < birth_prob)
    new_grid[birth_mask] = 1

    density = float(np.mean(new_grid))
    return new_grid, density

def simulate_contact_process(
    L: int = 50,
    birth_rate: float = 0.25,
    death_rate: float = 1.0,
    steps: int = 100,
    init_density: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates the 2D contact process over multiple time steps.

    Args:
        L: Lattice size (L x L).
        birth_rate: Birth rate per active neighbor.
        death_rate: Death rate per active site.
        steps: Number of simulation steps.
        init_density: Initial fraction of active sites.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (final_grid, densities_array).
    """
    rng = np.random.default_rng(seed)
    grid = (rng.random((L, L)) < init_density).astype(int)
    densities = np.zeros(steps)

    for t in range(steps):
        grid, rho = contact_process(grid, birth_rate, death_rate, rng=rng)
        densities[t] = rho
        if rho == 0.0:
            break

    return grid, densities
