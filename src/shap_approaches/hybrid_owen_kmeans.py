# src/shap_approaches/hybrid_owen_kmeans.py
import numpy as np
from src.shap_approaches.owen_values import OwenValuesSHAP
from src.shap_approaches.kmeans import summarize_background


def replicate_by_weight(rows: np.ndarray, weights: np.ndarray, n_rows=100) -> np.ndarray:
    """
    Repeats each row in proportion to its weight (largest-remainder rounding to n_rows),
    because the Partition masker averages its background uniformly. Clusters holding less
    than ~1/(2 n_rows) of the data may get no copies.
    """
    exact = weights / weights.sum() * n_rows
    counts = np.floor(exact).astype(int)
    counts[np.argsort(exact - counts)[::-1][: n_rows - counts.sum()]] += 1
    return np.repeat(rows, counts, axis=0)


class HybridOwenKMeansSHAP(OwenValuesSHAP):
    """
    Owen values against a cluster-weighted KMeans background summary. Repeat i uses
    summary_pool[i] (cycling if the pool is shorter). Owen values are deterministic for a
    given background, so all run-to-run variation comes from the summary pool; with a single
    summary the approach is deterministic.
    """
    def __init__(self, model, background_data, n_clusters=10, summary="medoids",
                 random_state=None, n_background_rows=100, summary_pool=None):
        super().__init__(model, background_data, random_state=random_state)
        if summary_pool is None:
            summary_pool = [summarize_background(background_data, n_clusters, summary)]
        self._bg_mats = [replicate_by_weight(rows.to_numpy(dtype=float), weights, n_background_rows)
                         for rows, weights in summary_pool]

    def _background(self):
        return self._bg_mats[self.iteration % len(self._bg_mats)]
