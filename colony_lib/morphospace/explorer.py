"""
High-Dimensional Morphospace Exploration & Latin Hypercube Sampling
Allows systematic exploration across arbitrary N-dimensional parameter hypercubes.
"""

import numpy as np
from typing import Dict, List, Tuple, Any

def latin_hypercube_sample(
    bounds: Dict[str, Tuple[float, float]],
    n_samples: int,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generates Latin Hypercube samples across bounded hypercube dimensions.
    Each dimension is partitioned into n_samples intervals with one sample per interval.
    """
    rng = np.random.default_rng(seed)
    dim_names = list(bounds.keys())
    n_dims = len(dim_names)
    
    samples_matrix = np.empty((n_samples, n_dims), dtype=np.float64)
    
    for d, name in enumerate(dim_names):
        low, high = bounds[name]
        # Generate random permutation of intervals
        perm = rng.permutation(n_samples)
        # Uniform point inside each interval
        u = rng.uniform(0.0, 1.0, size=n_samples)
        pts = (perm + u) / float(n_samples)
        samples_matrix[:, d] = low + pts * (high - low)
        
    result = {}
    for d, name in enumerate(dim_names):
        result[name] = samples_matrix[:, d]
        
    return result

class MorphospaceSampler:
    """
    Manages multi-dimensional parameter exploration with grid and LHS strategies.
    """
    def __init__(self, bounds: Dict[str, Tuple[float, float]]):
        self.bounds = bounds
        self.dim_names = list(bounds.keys())
        self.n_dims = len(bounds)
        
    def sample_lhs(self, n_samples: int, seed: int = 42) -> Dict[str, np.ndarray]:
        return latin_hypercube_sample(self.bounds, n_samples=n_samples, seed=seed)
        
    def sample_grid(self, points_per_dim: int) -> Dict[str, np.ndarray]:
        """Generates full Cartesian grid points across all dimensions."""
        grids = [np.linspace(self.bounds[name][0], self.bounds[name][1], points_per_dim) for name in self.dim_names]
        mesh = np.meshgrid(*grids, indexing="ij")
        return {name: mesh[i].ravel() for i, name in enumerate(self.dim_names)}
