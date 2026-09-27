"""
Topological Morphospace Profiling
Computes connected components (Betti-0) as a function of filtration scale epsilon.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from typing import Dict, Any

def compute_betti_0_curve(
    points: np.ndarray,
    epsilons: np.ndarray
) -> Dict[str, Any]:
    """
    Computes Betti-0 (number of connected components) across a filtration of distances.
    At eps -> 0, Betti_0 = N.
    At eps -> max_dist, Betti_0 = 1.
    The decay rate characterizes point cloud clustering and fractal geometry.
    """
    N = len(points)
    dists = squareform(pdist(points, metric="euclidean"))
    
    betti_0_vals = []
    for eps in epsilons:
        adj = (dists <= eps).astype(np.int32)
        n_components, _ = connected_components(adj, directed=False)
        betti_0_vals.append(int(n_components))
        
    return {
        "N_points": N,
        "epsilons": epsilons,
        "betti_0": np.array(betti_0_vals, dtype=np.int32)
    }
