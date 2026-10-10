"""
Coupled Map Lattice (CML) Module: Coupled Logistic Maps and Spatiotemporal Chaos.
Supports Kaneko 1D coupled map lattices and standard diffusive logistic map arrays.
"""

import numpy as np
from typing import Tuple, Optional, Union

def coupled_logistic_map(
    x: Optional[Union[np.ndarray, int]] = None,
    r: float = 3.865,
    eps: float = 0.132,
    N: Optional[int] = None,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    1D Coupled Logistic Map Lattice with periodic boundary conditions.
    
    Can be used in two ways:
    1. Single step update:
       coupled_logistic_map(x, r=3.865, eps=0.132) -> next_x
       where x is a 1D numpy array of site values.
       
    2. Multi-step simulation:
       coupled_logistic_map(N=100, steps=2000, r=3.865, eps=0.132) -> trajectory (steps, N)
       or coupled_logistic_map(steps=2000)
    """
    # If x is provided and is an array, perform single step update
    if isinstance(x, np.ndarray):
        x_prev = np.roll(x, 1)
        x_next = np.roll(x, -1)
        fx = r * x * (1.0 - x)
        fx_prev = r * x_prev * (1.0 - x_prev)
        fx_next = r * x_next * (1.0 - x_next)
        return (1.0 - eps) * fx + (eps / 2.0) * (fx_prev + fx_next)

    # If first argument was passed as an integer (e.g. coupled_logistic_map(100, steps=500))
    if isinstance(x, (int, np.integer)):
        N = int(x)

    total_N = N if N is not None else kwargs.get("n", 100)
    total_steps = steps if steps is not None else kwargs.get("T", 1000)
    
    return simulate_coupled_logistic_map(N=total_N, steps=total_steps, r=r, eps=eps, seed=seed)

def simulate_coupled_logistic_map(
    N: int = 100,
    steps: int = 1000,
    r: float = 3.865,
    eps: float = 0.132,
    seed: Optional[int] = None,
    transient: int = 100
) -> np.ndarray:
    """
    Simulates a 1D coupled logistic map lattice over time.
    
    Args:
        N: Number of lattice sites.
        steps: Number of time steps to record.
        r: Logistic growth parameter (e.g. 3.865).
        eps: Diffusive coupling strength (e.g. 0.132).
        seed: Optional RNG seed.
        transient: Number of initial steps to discard.
        
    Returns:
        trajectory: np.ndarray of shape (steps, N)
    """
    rng = np.random.default_rng(seed)
    x_curr = rng.uniform(0.1, 0.9, size=N)
    
    for _ in range(transient):
        x_curr = coupled_logistic_map(x_curr, r=r, eps=eps)
        
    trajectory = np.empty((steps, N), dtype=np.float64)
    for t in range(steps):
        x_curr = coupled_logistic_map(x_curr, r=r, eps=eps)
        trajectory[t] = x_curr
        
    return trajectory
