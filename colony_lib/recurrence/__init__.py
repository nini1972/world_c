"""
Recurrence Module: Takens phase space embedding, Recurrence Quantification Analysis (RQA), and Adler transitions.
"""

from .takens import takens_embedding, estimate_delay_autocorr
from .rqa import recurrence_matrix, compute_rqa_metrics, RecurrencePlot, RecurrenceQuantificationAnalysis
from .adler import detect_adler_transition, AdlerDetector

# Alias 'rqa' to compute_rqa_metrics so both compute_rqa_metrics(R) and rqa(R) work
rqa = compute_rqa_metrics

__all__ = [
    "takens_embedding",
    "estimate_delay_autocorr",
    "recurrence_matrix",
    "compute_rqa_metrics",
    "rqa",
    "RecurrencePlot",
    "RecurrenceQuantificationAnalysis",
    "detect_adler_transition",
    "AdlerDetector",
]
