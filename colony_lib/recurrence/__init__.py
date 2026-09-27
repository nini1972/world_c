"""
Recurrence Module: Takens phase space embedding, Recurrence Quantification Analysis (RQA), and Adler transitions.
"""

from .takens import takens_embedding, estimate_delay_autocorr
from .rqa import recurrence_matrix, compute_rqa_metrics
from .adler import detect_adler_transition, AdlerDetector

__all__ = [
    "takens_embedding",
    "estimate_delay_autocorr",
    "recurrence_matrix",
    "compute_rqa_metrics",
    "detect_adler_transition",
    "AdlerDetector",
]
