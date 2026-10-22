"""
Recurrence Quantification Analysis (RQA)
Computes recurrence plots R_ij = Theta(epsilon - ||x_i - x_j||)
and fundamental RQA metrics:
- Recurrence Rate (RR)
- Determinism (DET)
- Laminarity (LAM)
- Diagonal Line Length Shannon Entropy (ENTR)
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Dict, Any, Optional

def recurrence_matrix(
    embedded_points: np.ndarray,
    epsilon: Optional[float] = None,
    metric: str = "euclidean",
    sparse: bool = False,
    threshold: Optional[float] = None,
    **kwargs
) -> np.ndarray:
    """
    Constructs binary recurrence matrix R_ij = 1 if ||x_i - x_j|| <= epsilon else 0.
    Accepts optional metric, sparse, and threshold aliases for agent ergonomics.
    """
    if epsilon is None:
        epsilon = threshold if threshold is not None else 0.1
    pts = np.asarray(embedded_points, dtype=float)
    if pts.ndim == 1:
        pts = pts[:, None]
    dists = squareform(pdist(pts, metric=metric))
    return (dists <= float(epsilon)).astype(np.uint8)

def compute_rqa_metrics(
    R: np.ndarray,
    min_diag_len: int = 2,
    min_vert_len: int = 2,
    l_min: Optional[int] = None,
    v_min: Optional[int] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Extracts RQA statistical indicators from binary recurrence matrix R.
    Supports l_min / v_min aliases for min_diag_len / min_vert_len.
    """
    if l_min is not None:
        min_diag_len = int(l_min)
    if v_min is not None:
        min_vert_len = int(v_min)

    R = np.asarray(R, dtype=np.uint8)
    N = R.shape[0]
    total_points = max(1, N * N)
    
    # Recurrence Rate (RR): density of recurrence points
    recurrence_points = int(np.sum(R))
    rr = float(recurrence_points) / float(total_points)
    
    # Diagonal line analysis
    diag_lengths = []
    # Loop over all diagonals above main diagonal
    for offset in range(1, N):
        diag = np.diagonal(R, offset=offset)
        current_run = 0
        for val in diag:
            if val == 1:
                current_run += 1
            else:
                if current_run >= min_diag_len:
                    diag_lengths.append(current_run)
                current_run = 0
        if current_run >= min_diag_len:
            diag_lengths.append(current_run)
            
    # Determinism (DET): fraction of recurrence points forming diagonal lines
    if diag_lengths and recurrence_points > 0:
        det_points = sum(diag_lengths) * 2  # symmetric
        det = min(1.0, float(det_points) / float(recurrence_points))
        
        # Shannon Entropy of diagonal line lengths (ENTR)
        unique, counts = np.unique(diag_lengths, return_counts=True)
        probs = counts / np.sum(counts)
        entr = float(-np.sum(probs * np.log2(probs + 1e-12)))
        mean_diag = float(np.mean(diag_lengths))
        max_diag = float(np.max(diag_lengths))
    else:
        det = 0.0
        entr = 0.0
        mean_diag = 0.0
        max_diag = 0.0

    # Vertical line analysis for Laminarity (LAM)
    vert_lengths = []
    for col in range(N):
        current_run = 0
        for row in range(N):
            if row == col:
                continue
            if R[row, col] == 1:
                current_run += 1
            else:
                if current_run >= min_vert_len:
                    vert_lengths.append(current_run)
                current_run = 0
        if current_run >= min_vert_len:
            vert_lengths.append(current_run)

    if vert_lengths and recurrence_points > 0:
        lam = min(1.0, float(sum(vert_lengths)) / float(recurrence_points))
    else:
        lam = 0.0
        
    return {
        "N": float(N),
        "recurrence_rate": rr,
        "determinism": det,
        "laminarity": lam,
        "shannon_entropy": entr,
        "mean_diag_length": mean_diag,
        "max_diag_length": max_diag,
        "RR": rr,
        "DET": det,
        "LAM": lam,
        "ENT": entr,
        "ENTR": entr,
        "L": mean_diag,
        "Lmax": max_diag,
    }


class RecurrencePlot:
    """Ergonomic OOP wrapper for Takens embedding + pairwise distance / recurrence matrix."""
    def __init__(self, series: np.ndarray, dim: int = 2, tau: int = 1, m: Optional[int] = None, epsilon: Optional[float] = None, **kwargs):
        from .takens import takens_embedding
        embed_m = int(m) if m is not None else int(dim)
        arr = np.asarray(series, dtype=float)
        self.embedded = takens_embedding(arr, m=embed_m, tau=int(tau)) if (arr.ndim == 1 and embed_m > 1) else (arr[:, None] if arr.ndim == 1 else arr)
        self.distance_matrix = squareform(pdist(self.embedded, metric=kwargs.get("metric", "euclidean")))
        self.epsilon = float(epsilon) if epsilon is not None else float(0.1 * np.mean(self.distance_matrix) if self.distance_matrix.size else 0.1)

    def recurrence_matrix(self, epsilon: Optional[float] = None) -> np.ndarray:
        eps = float(epsilon) if epsilon is not None else self.epsilon
        return (self.distance_matrix <= eps).astype(np.uint8)


class RecurrenceQuantificationAnalysis:
    """Ergonomic OOP wrapper around compute_rqa_metrics."""
    def __init__(self, rp_or_matrix, epsilon: Optional[float] = None, **kwargs):
        if isinstance(rp_or_matrix, RecurrencePlot):
            R = rp_or_matrix.recurrence_matrix(epsilon=epsilon)
        else:
            arr = np.asarray(rp_or_matrix)
            if epsilon is not None and arr.dtype.kind == "f":
                R = (arr <= float(epsilon)).astype(np.uint8)
            else:
                R = arr.astype(np.uint8)
        self.R = R
        self.metrics = compute_rqa_metrics(R, **kwargs)

    def recurrence_rate(self) -> float:
        return self.metrics["recurrence_rate"]

    def determinism(self) -> float:
        return self.metrics["determinism"]

    def laminarity(self) -> float:
        return self.metrics["laminarity"]

    def mean_diagonal_length(self) -> float:
        return self.metrics["mean_diag_length"]

    def max_diagonal_length(self) -> float:
        return self.metrics["max_diag_length"]

    def shannon_entropy_diagonal_lines(self) -> float:
        return self.metrics["shannon_entropy"]
