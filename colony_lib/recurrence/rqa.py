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
        "max_diag_length": max_diag
    }
