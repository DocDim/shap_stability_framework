# src/shap_approaches/owen_values.py
import numpy as np
import pandas as pd
import shap
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from src.shap_approaches.base import BaseSHAPApproach, Explanation


def build_feature_hierarchy(data: pd.DataFrame) -> np.ndarray:
    """
    Average-linkage clustering of features on 1 - |corr|, returned as a scipy linkage matrix
    (the binary tree PartitionExplainer uses as its coalition structure). Constant features
    get distance 1 to everything, so they join the tree last.
    """
    n_features = data.shape[1]
    valid_mask = data.std(axis=0).to_numpy() > 1e-7

    dist_matrix = np.ones((n_features, n_features), dtype=float)
    if np.sum(valid_mask) > 1:
        corr = data.loc[:, valid_mask].corr().to_numpy()
        valid_dist = np.clip(1.0 - np.abs(corr), 0.0, 1.0)
        dist_matrix[np.ix_(valid_mask, valid_mask)] = (valid_dist + valid_dist.T) / 2.0
    np.fill_diagonal(dist_matrix, 0.0)

    return linkage(squareform(dist_matrix, checks=False), method="average")


class OwenValuesSHAP(BaseSHAPApproach):
    """
    Owen values over the feature correlation hierarchy, via shap.PartitionExplainer.

    Credit is assigned top-down through the tree: correlated features are switched between
    background and target together at coarse levels and separated only near the leaves.
    `n_samples` is the evaluation budget (max_evals). Subtrees the budget does not reach split
    their credit across members in proportion to size, so efficiency always holds. With a
    budget large enough to explore the whole tree the result is the exact Owen value for the
    given background.

    Pass `clustering` to reuse a hierarchy built elsewhere (e.g. the reference explainer, whose
    small background should not define the coalition structure).
    """
    estimand = "owen"

    def __init__(self, model, background_data, random_state=None, background_size=50, clustering=None):
        super().__init__(model, background_data, random_state=random_state, background_size=background_size)
        self.clustering = build_feature_hierarchy(background_data) if clustering is None else clustering

    def _background(self) -> np.ndarray:
        """Background rows for one iteration (the masker averages over them uniformly)."""
        return self._sample_background().to_numpy(dtype=float)

    def _predict_positive(self, X):
        return self.model.predict_proba(X)[:, 1]

    def run_iteration(self, target_instance, n_samples=200) -> Explanation:
        bg = self._background()
        masker = shap.maskers.Partition(bg, max_samples=len(bg), clustering=self.clustering)
        explainer = shap.PartitionExplainer(self._predict_positive, masker)
        result = explainer(np.asarray(target_instance, dtype=float), max_evals=n_samples, silent=True)

        # +2 for the all-background and all-target evaluations outside the Owen recursion
        return Explanation(
            phi=np.asarray(result.values).reshape(-1),
            base_value=float(np.ravel(result.base_values)[0]),
            n_coalitions=int(explainer.last_eval_count) + 2,
            n_background=len(bg),
        )
