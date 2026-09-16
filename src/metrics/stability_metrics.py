import numpy as np
from scipy.stats import wasserstein_distance

class StabilityEvaluator:
    @staticmethod
    def calculate_acv(shap_runs):
        """V1: Feature Association Coefficient of Variation (ACV)[cite: 1]."""
        std_per_feature = np.std(shap_runs, axis=0)
        mean_per_feature = np.abs(np.mean(shap_runs, axis=0))
        acv = np.where(mean_per_feature > 1e-6, std_per_feature / mean_per_feature, 0.0)
        return np.mean(acv)

    @staticmethod
    def calculate_jaccard_rank_agreement(shap_runs, k=5):
        """V2: Jaccard Index Rank Agreement[cite: 1]."""
        n_iterations = shap_runs.shape[0]
        jaccard_scores = []
        top_k_indices = [set(np.argsort(np.abs(run))[-k:]) for run in shap_runs]
        
        for i in range(n_iterations):
            for j in range(i + 1, n_iterations):
                set_i = top_k_indices[i]
                set_j = top_k_indices[j]
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                jaccard_scores.append(intersection / union)
                
        return np.mean(jaccard_scores)

    @staticmethod
    def calculate_wasserstein_distance(baseline_runs, approach_runs):
        """V3: Distance Metrics (Wasserstein Distance relative to Baseline)[cite: 1]."""
        n_features = baseline_runs.shape[1]
        distances = []
        for f in range(n_features):
            dist = wasserstein_distance(baseline_runs[:, f], approach_runs[:, f])
            distances.append(dist)
        return np.mean(distances)