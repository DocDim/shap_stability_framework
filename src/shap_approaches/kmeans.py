# src/shap_approaches/kmeans.py
import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from src.shap_approaches.base import BaseSHAPApproach, Explanation, weighted_background


def summarize_background(data: pd.DataFrame, n_clusters=10, method="medoids", random_state=42):
    """
    Summarizes the background as n_clusters rows weighted by cluster size.

    "centroids" averages one-hot columns into fractional values the model never saw in
    training; "medoids" uses the real record closest to each centroid instead.
    """
    X = data.to_numpy(dtype=float)
    n_clusters = min(n_clusters, len(X))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(X)
    weights = np.bincount(kmeans.labels_, minlength=n_clusters) / len(X)

    if method == "centroids":
        rows = kmeans.cluster_centers_
    elif method == "medoids":
        dist = kmeans.transform(X)  # (n_rows, n_clusters) distance to each centroid
        rows = np.empty_like(kmeans.cluster_centers_)
        for c in range(n_clusters):
            members = np.flatnonzero(kmeans.labels_ == c)
            rows[c] = X[members[np.argmin(dist[members, c])]]
    else:
        raise ValueError(f"Unknown background summary method: {method!r}")

    return pd.DataFrame(rows, columns=data.columns), weights


def build_summary_pool(data: pd.DataFrame, n_summaries, n_clusters=10, method="medoids",
                       bootstrap=True, random_state=42):
    """
    A list of (rows, weights) background summaries, one per repeat.

    With bootstrap=True each summary is fitted on its own bootstrap sample of `data` with its
    own KMeans seed, so the pool carries the sampling variability of the summary itself (what a
    practitioner would see after a data refresh or a reseed). Baseline-style approaches pay for
    that variability by resampling their background; KMeans approaches must too, or their
    stability is overstated. With bootstrap=False the pool holds one fixed summary.
    """
    if not bootstrap:
        return [summarize_background(data, n_clusters, method, random_state)]

    rng = np.random.default_rng(random_state)
    pool = []
    for i in range(n_summaries):
        idx = rng.choice(len(data), size=len(data), replace=True)
        pool.append(summarize_background(data.iloc[idx], n_clusters, method, random_state + i))
    return pool


class KMeansBackgroundKernelSHAP(BaseSHAPApproach):
    """
    KernelSHAP against a cluster-weighted KMeans background summary. Repeat i uses
    summary_pool[i] (cycling if the pool is shorter); without a pool, one summary of
    `background_data` is computed and reused for every repeat.
    """
    def __init__(self, model, background_data, n_clusters=10, summary="medoids", random_state=None,
                 summary_pool=None):
        super().__init__(model, background_data, random_state=random_state)
        if summary_pool is None:
            summary_pool = [summarize_background(background_data, n_clusters, summary)]
        self._bg_summaries = [weighted_background(rows, weights) for rows, weights in summary_pool]

    def run_iteration(self, target_instance, n_samples=200) -> Explanation:
        bg_summary = self._bg_summaries[self.iteration % len(self._bg_summaries)]
        explainer = shap.KernelExplainer(self.model.predict_proba, bg_summary)
        return self._explain(explainer, target_instance, n_samples)
