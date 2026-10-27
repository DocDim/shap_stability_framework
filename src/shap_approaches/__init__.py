# src/shap_approaches/__init__.py
from .base import BaseSHAPApproach, Explanation, RepeatedRuns
from .baseline import BaselineKernelSHAP
from .kmeans import KMeansBackgroundKernelSHAP, summarize_background, build_summary_pool
from .penalized_kernel import PenalizedKernelSHAP
from .owen_values import OwenValuesSHAP, build_feature_hierarchy, full_traversal_evals
from .hybrid_owen_kmeans import HybridOwenKMeansSHAP, replicate_by_weight

__all__ = [
    "BaseSHAPApproach",
    "Explanation",
    "RepeatedRuns",
    "BaselineKernelSHAP",
    "KMeansBackgroundKernelSHAP",
    "summarize_background",
    "build_summary_pool",
    "PenalizedKernelSHAP",
    "OwenValuesSHAP",
    "build_feature_hierarchy",
    "full_traversal_evals",
    "HybridOwenKMeansSHAP",
    "replicate_by_weight",
]
