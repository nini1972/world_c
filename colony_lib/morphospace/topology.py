"""
Topological Morphospace Profiling
Computes connected components (Betti-0) as a function of filtration scale epsilon.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from typing import Dict, Any, Optional

def compute_betti_0_curve(
    points: np.ndarray,
    epsilons: Optional[np.ndarray] = None,
    n_steps: int = 50,
    metric: str = "euclidean",
    **kwargs
) -> Dict[str, Any]:
    """
    Computes Betti-0 (number of connected components) across a filtration of distances.
    At eps -> 0, Betti_0 = N.
    At eps -> max_dist, Betti_0 = 1.
    The decay rate characterizes point cloud clustering and fractal geometry.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts[:, None]
    N = len(pts)
    dists = squareform(pdist(pts, metric=metric))
    
    if epsilons is None:
        max_d = float(np.max(dists)) if dists.size > 0 else 1.0
        epsilons = np.linspace(0.0, max_d if max_d > 0 else 1.0, int(n_steps))
    else:
        epsilons = np.asarray(epsilons, dtype=float)
    
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
