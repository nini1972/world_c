"""
colony_lib.network_topology: Spectral Graph Theory, Community Structure & Canonical Topologies.
Fulfilling claude_haiku's Cartographer's Petition (Section 2B) for World C.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import linalg
import networkx as nx


def _to_adjacency_and_graph(
    graph_or_matrix: Union[np.ndarray, nx.Graph, List[List[float]]]
) -> Tuple[np.ndarray, nx.Graph]:
    """Converts either a numpy adjacency matrix or a networkx.Graph into both (A, G)."""
    if isinstance(graph_or_matrix, nx.Graph):
        G = graph_or_matrix
        A = nx.to_numpy_array(G, dtype=np.float64)
        return A, G
    A = np.asarray(graph_or_matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency matrix must be square 2D, got shape {A.shape}")
    G = nx.from_numpy_array(A)
    return A, G


def laplacian_spectrum(
    adjacency_matrix: Union[np.ndarray, nx.Graph],
    normalized: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the sorted eigenvalue spectrum and eigenvectors of the graph Laplacian.

    Guarantees ascending eigenvalue order:
        0 = evals[0] <= evals[1] (spectral gap / algebraic connectivity) <= ... <= evals[-1]

    Parameters
    ----------
    adjacency_matrix : np.ndarray or nx.Graph
        Square symmetric adjacency matrix (N, N) or NetworkX graph.
    normalized : bool, default False
        If True, computes the symmetric normalized Laplacian L_sym = I - D^(-1/2) A D^(-1/2).

    Returns
    -------
    evals : np.ndarray of shape (N,)
        Sorted Laplacian eigenvalues in ascending order.
    evecs : np.ndarray of shape (N, N)
        Corresponding orthonormal eigenvectors as columns (evecs[:, i] corresponds to evals[i]).
    """
    A, _ = _to_adjacency_and_graph(adjacency_matrix)
    degrees = np.sum(A, axis=1)

    if normalized:
        with np.errstate(divide="ignore"):
            inv_sqrt_deg = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
        D_inv_sqrt = np.diag(inv_sqrt_deg)
        L = np.eye(A.shape[0], dtype=np.float64) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        L = np.diag(degrees) - A

    # Symmetrize to remove floating-point asymmetry
    L = 0.5 * (L + L.T)
    evals, evecs = linalg.eigh(L)

    # Ensure strict ascending sort and clamp numerical zero on trivial eigenmode
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    evals = np.where(np.abs(evals) < 1e-12, 0.0, evals)

    return evals, evecs


def spectral_gap(
    adjacency_matrix: Union[np.ndarray, nx.Graph],
    normalized: bool = False
) -> float:
    """
    Returns the algebraic connectivity (Fiedler eigenvalue lambda_1, the second smallest
    eigenvalue of the Laplacian).
    """
    evals, _ = laplacian_spectrum(adjacency_matrix, normalized=normalized)
    if len(evals) < 2:
        return 0.0
    return float(evals[1])


# Alias matching standard spectral graph terminology
algebraic_connectivity = spectral_gap


def detect_communities(
    adjacency_matrix: Union[np.ndarray, nx.Graph],
    method: str = "louvain",
    seed: int = 42
) -> List[set]:
    """
    Detects modular community structure in a network.

    Parameters
    ----------
    adjacency_matrix : np.ndarray or nx.Graph
        Network adjacency matrix or NetworkX graph.
    method : str, default 'louvain'
        Community detection method ('louvain', 'greedy', or 'label_propagation').
    seed : int, default 42
        Random seed for stochastic community detection algorithms.

    Returns
    -------
    communities : list of set
        Partition of node indices into communities.
    """
    _, G = _to_adjacency_and_graph(adjacency_matrix)
    method_clean = method.strip().lower()

    if G.number_of_edges() == 0:
        return [{n} for n in G.nodes()]

    if method_clean == "louvain":
        try:
            return list(nx.community.louvain_communities(G, seed=seed))
        except Exception:
            return list(nx.community.greedy_modularity_communities(G))
    elif method_clean in ("greedy", "modularity", "greedy_modularity"):
        return list(nx.community.greedy_modularity_communities(G))
    elif method_clean in ("label_propagation", "lpa"):
        return list(nx.community.label_propagation_communities(G))
    else:
        raise ValueError(f"Unknown community detection method '{method}'. Choose from 'louvain', 'greedy', 'label_propagation'.")


def average_clustering(adjacency_matrix: Union[np.ndarray, nx.Graph]) -> float:
    """Computes the mean local clustering coefficient C of the network."""
    _, G = _to_adjacency_and_graph(adjacency_matrix)
    if G.number_of_nodes() == 0:
        return 0.0
    return float(nx.average_clustering(G))


def assortativity(adjacency_matrix: Union[np.ndarray, nx.Graph]) -> float:
    """
    Computes the degree assortativity coefficient r of the network.
    Returns 0.0 for regular (zero degree variance) or edgeless graphs instead of NaN.
    """
    A, G = _to_adjacency_and_graph(adjacency_matrix)
    if G.number_of_edges() == 0:
        return 0.0
    degrees = np.sum(A > 0, axis=1)
    if np.std(degrees) < 1e-12:
        return 0.0
    try:
        val = float(nx.degree_assortativity_coefficient(G))
        return 0.0 if np.isnan(val) else val
    except Exception:
        return 0.0


def generate_topology(
    topology: str = "scale-free",
    N: int = 100,
    seed: int = 42,
    return_graph: bool = False,
    **kwargs: Any
) -> Union[np.ndarray, nx.Graph]:
    """
    Generates canonical complex network topologies for bifurcation and epidemic experiments.

    Parameters
    ----------
    topology : str
        One of:
        - 'scale-free', 'ba', 'barabasi-albert' (params: m=2)
        - 'erdos-renyi', 'er', 'random' (params: p=0.1)
        - 'small-world', 'ws', 'watts-strogatz' (params: k=4, p=0.3)
        - 'ring', 'cycle', 'lattice-1d' (params: k=2)
        - 'complete', 'all-to-all', 'fully-connected'
        - 'regular', 'k-regular' (params: k=4)
    N : int, default 100
        Number of nodes.
    seed : int, default 42
        Random seed.
    return_graph : bool, default False
        If True, returns the NetworkX Graph object; otherwise returns the (N, N) float64 adjacency matrix.
    """
    topo = topology.strip().lower().replace("_", "-")

    if topo in ("scale-free", "ba", "barabasi-albert"):
        m = int(kwargs.get("m", max(1, min(2, N - 1))))
        G = nx.barabasi_albert_graph(N, m, seed=seed)
    elif topo in ("erdos-renyi", "er", "random"):
        p = float(kwargs.get("p", max(0.05, min(1.0, 6.0 / max(N, 1)))))
        G = nx.erdos_renyi_graph(N, p, seed=seed)
    elif topo in ("small-world", "ws", "watts-strogatz"):
        k = int(kwargs.get("k", min(4, max(2, N - 1))))
        if k % 2 == 1 and k < N - 1:
            k += 1
        p = float(kwargs.get("p", 0.3))
        G = nx.watts_strogatz_graph(N, k, p, seed=seed)
    elif topo in ("ring", "cycle", "lattice-1d"):
        k = int(kwargs.get("k", 2))
        if k <= 2:
            G = nx.cycle_graph(N)
        else:
            if k % 2 == 1:
                k += 1
            G = nx.watts_strogatz_graph(N, min(k, N - 1), 0.0, seed=seed)
    elif topo in ("complete", "all-to-all", "fully-connected"):
        G = nx.complete_graph(N)
    elif topo in ("regular", "k-regular"):
        k = int(kwargs.get("k", min(4, N - 1)))
        if (N * k) % 2 == 1:
            k = max(2, k - 1)
        G = nx.random_regular_graph(k, N, seed=seed)
    else:
        raise ValueError(
            f"Unknown topology '{topology}'. Supported: 'scale-free', 'erdos-renyi', "
            f"'small-world', 'ring', 'complete', 'regular'."
        )

    if return_graph:
        return G
    return nx.to_numpy_array(G, dtype=np.float64)


def spectral_fingerprint(adjacency_matrix: Union[np.ndarray, nx.Graph]) -> Dict[str, float]:
    """
    Computes a comprehensive spectral and topological summary dictionary for a network:
    lambda_1 (spectral gap), lambda_2, lambda_max, condition_number (kappa = lambda_max / lambda_1),
    mean_degree, degree_std, clustering, and assortativity.
    """
    A, G = _to_adjacency_and_graph(adjacency_matrix)
    evals, _ = laplacian_spectrum(A)
    degrees = np.sum(A > 0, axis=1)

    lam1 = float(evals[1]) if len(evals) > 1 else 0.0
    lam2 = float(evals[2]) if len(evals) > 2 else lam1
    lam_max = float(evals[-1]) if len(evals) > 0 else 0.0
    kappa = float(lam_max / lam1) if lam1 > 1e-12 else float("inf")

    return {
        "N": int(A.shape[0]),
        "lambda_1": lam1,
        "lambda_2": lam2,
        "lambda_max": lam_max,
        "condition_number": kappa,
        "mean_degree": float(np.mean(degrees)),
        "degree_std": float(np.std(degrees)),
        "clustering": average_clustering(G),
        "assortativity": assortativity(G),
    }


__all__ = [
    "laplacian_spectrum",
    "spectral_gap",
    "algebraic_connectivity",
    "detect_communities",
    "average_clustering",
    "assortativity",
    "generate_topology",
    "spectral_fingerprint",
]
