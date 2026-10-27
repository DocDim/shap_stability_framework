# src/shap_approaches/owen_values.py
import numpy as np
import pandas as pd
import shap
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from src.shap_approaches.base import BaseSHAPApproach, Explanation

# Average-linkage distance (1 - |corr|) below which features form one correlated group
GROUP_DISTANCE = 0.5


def build_feature_hierarchy(data: pd.DataFrame, group_distance: float = GROUP_DISTANCE) -> np.ndarray:
    """
    Coalition structure for PartitionExplainer, as a scipy-style linkage matrix.

    Features are grouped by average-linkage clustering on 1 - |corr|, cut at `group_distance`
    (constant features get distance 1, so each is its own group). The groups, and the features
    within each group (in dendrogram order), are then joined in a balanced binary tree.

    Balancing matters for cost: PartitionExplainer reaches a node at depth d through 2^d
    context paths, so the raw dendrogram, which chains weakly correlated features into a tree
    dozens of levels deep, would need ~10^14 evaluations to explore. A balanced tree needs
    O(M^2) (see full_traversal_evals), so the exact Owen value is computable.
    """
    n_features = data.shape[1]
    dendrogram = _correlation_linkage(data)
    order = leaves_list(dendrogram)
    labels = fcluster(dendrogram, t=group_distance, criterion="distance")[order]
    # Flat clusters of a monotone linkage are subtrees, so each group is contiguous in leaf order
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    groups = np.split(order, starts[1:])
    return _balanced_linkage(groups, n_features)


def _balanced_linkage(groups, n_features) -> np.ndarray:
    """Balanced binary tree over the groups (split near half the features, never inside a group
    while several remain), then over each group's features. Merge heights are tree levels."""
    rows = []

    def build(segments):
        if len(segments) == 1 and len(segments[0]) == 1:
            return int(segments[0][0]), 1, 0
        if len(segments) == 1:
            half = len(segments[0]) // 2
            left, right = [segments[0][:half]], [segments[0][half:]]
        else:
            sizes = np.cumsum([len(s) for s in segments])
            cut = int(np.argmin(np.abs(sizes[:-1] - sizes[-1] / 2))) + 1
            left, right = segments[:cut], segments[cut:]
        a, size_a, height_a = build(left)
        b, size_b, height_b = build(right)
        height = max(height_a, height_b) + 1
        rows.append([a, b, height, size_a + size_b])
        return n_features + len(rows) - 1, size_a + size_b, height

    build(list(groups))
    return np.array(rows, dtype=float)


def full_traversal_evals(clustering: np.ndarray) -> int:
    """
    Model evaluations PartitionExplainer (fixed_context=None) needs to explore the whole tree:
    each internal node at depth d is visited once per context path (2^d) and each visit
    evaluates two coalitions.
    """
    n_features = len(clustering) + 1
    total, stack = 0, [(2 * n_features - 2, 0)]
    while stack:
        node, depth = stack.pop()
        if node < n_features:
            continue
        total += 2 * 2 ** depth
        left, right = clustering[node - n_features, :2].astype(int)
        stack += [(left, depth + 1), (right, depth + 1)]
    return total


def _correlation_linkage(data: pd.DataFrame) -> np.ndarray:
    """Average-linkage clustering of features on 1 - |corr|."""
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

    Credit is assigned top-down through the tree: correlated groups are switched between
    background and target together at coarse levels and separated only near the leaves.
    `n_samples` is the evaluation budget (max_evals). Subtrees the budget does not reach split
    their credit across members in proportion to size, so efficiency always holds. With
    full_traversal_evals(clustering) + 3 evaluations the whole tree is explored and the result
    is the exact Owen value for the given background.

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
