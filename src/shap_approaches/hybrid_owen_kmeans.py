# src/shap_approaches/hybrid_owen_kmeans.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from src.shap_approaches.base import BaseSHAPApproach
import shap


class HybridOwenKMeansKernelSHAP(BaseSHAPApproach):
    def __init__(self, model, background_data, correlation_threshold=0.5, n_clusters=10):
        super().__init__(model, background_data)
        self.correlation_threshold = correlation_threshold
        self.n_clusters = min(n_clusters, len(background_data))
        
        # 1. Deterministic KMeans background summary
        self.bg_summary = self._compute_kmeans_background(background_data)
        
        # 2. Hierarchical feature coalitions
        self.coalition_labels = self._build_feature_coalitions(background_data)
        self.unique_coalitions = np.unique(self.coalition_labels)
        self.n_coalitions = len(self.unique_coalitions)

    def _compute_kmeans_background(self, data):
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(df.values)
        return pd.DataFrame(kmeans.cluster_centers_, columns=df.columns)

    def _build_feature_coalitions(self, data):
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        n_features = df.shape[1]
        
        stds = df.std(axis=0).to_numpy()
        valid_mask = stds > 1e-7
        
        dist_matrix = np.ones((n_features, n_features), dtype=float)
        np.fill_diagonal(dist_matrix, 0.0)

        if np.sum(valid_mask) > 1:
            valid_df = df.iloc[:, valid_mask]
            corr = valid_df.corr().to_numpy()
            valid_dist = np.clip(1.0 - np.abs(corr), 0.0, 1.0)
            np.fill_diagonal(valid_dist, 0.0)
            valid_dist = (valid_dist + valid_dist.T) / 2.0

            valid_indices = np.where(valid_mask)[0]
            for i_idx, i_orig in enumerate(valid_indices):
                for j_idx, j_orig in enumerate(valid_indices):
                    dist_matrix[i_orig, j_orig] = valid_dist[i_idx, j_idx]

        condensed_dist = squareform(dist_matrix, checks=False)
        Z = linkage(condensed_dist, method="average")
        dist_cutoff = max(1.0 - self.correlation_threshold, 1e-4)
        return fcluster(Z, t=dist_cutoff, criterion="distance")

    def _make_coalition_predict(self, target_row):
        bg_mat = self.bg_summary.to_numpy()
        target_vec = target_row.flatten()
        n_bg = bg_mat.shape[0]

        def coalition_predict(z_coalitions):
            n_evals = z_coalitions.shape[0]
            X_eval = np.tile(bg_mat, (n_evals, 1))
            
            for c_idx, c_id in enumerate(self.unique_coalitions):
                feat_indices = np.where(self.coalition_labels == c_id)[0]
                active_mask = (z_coalitions[:, c_idx] == 1)
                if np.any(active_mask):
                    active_eval_indices = np.repeat(active_mask, n_bg)
                    X_eval[active_eval_indices, :][:, feat_indices] = target_vec[feat_indices]

            probs = self.model.predict_proba(X_eval)
            p1 = probs[:, 1] if probs.ndim == 2 else probs
            return p1.reshape(n_evals, n_bg).mean(axis=1)

        return coalition_predict

    def run_iteration(self, target_instance, n_samples=200) -> np.ndarray:
        target_row = target_instance.to_numpy() if isinstance(target_instance, pd.DataFrame) else target_instance
        coalition_pred_fn = self._make_coalition_predict(target_row)

        z_bg = np.zeros((1, self.n_coalitions))
        z_target = np.ones((1, self.n_coalitions))

        explainer = shap.KernelExplainer(coalition_pred_fn, z_bg)
        coalition_shap = explainer.shap_values(z_target, nsamples=n_samples, l1_reg=False, silent=True)

        if isinstance(coalition_shap, list):
            c_vals = np.array(coalition_shap[1]).flatten()
        else:
            c_vals = np.array(coalition_shap).flatten()

        n_features = target_instance.shape[1]
        final_shap_values = np.zeros(n_features)
        for c_idx, c_id in enumerate(self.unique_coalitions):
            feat_indices = np.where(self.coalition_labels == c_id)[0]
            final_shap_values[feat_indices] = c_vals[c_idx] / len(feat_indices)

        return final_shap_values