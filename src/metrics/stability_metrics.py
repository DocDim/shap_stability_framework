import numpy as np


class StabilityEvaluator:
    @staticmethod
    def calculate_acv(shap_runs):
        """
        V1: Attribution Coefficient of Variation, importance-weighted:
        sum_j std(phi_j) / sum_j |mean(phi_j)|.

        Equivalent to averaging per-feature CVs weighted by |mean(phi_j)|, so features with
        near-zero attributions cannot dominate the score.
        """
        std_per_feature = np.std(shap_runs, axis=0)
        mean_per_feature = np.abs(np.mean(shap_runs, axis=0))
        total = mean_per_feature.sum()
        return float(std_per_feature.sum() / total) if total > 1e-12 else float("nan")

    @staticmethod
    def calculate_mean_feature_cv(shap_runs):
        """Unweighted mean of per-feature CVs (the original ACV). Kept for comparison with
        earlier results; dominated by features whose mean attribution is near zero."""
        std_per_feature = np.std(shap_runs, axis=0)
        mean_per_feature = np.abs(np.mean(shap_runs, axis=0))
        cv = np.where(mean_per_feature > 1e-6, std_per_feature / np.maximum(mean_per_feature, 1e-6), 0.0)
        return float(np.mean(cv))

    @staticmethod
    def calculate_jaccard_rank_agreement(shap_runs, k=5):
        """
        V2: Mean pairwise Jaccard index of the top-k |phi| sets across runs.

        Returns NaN when there are no more than k features to rank (agreement would be
        trivially 1) or fewer than two runs.
        """
        n_iterations, n_features = shap_runs.shape
        if n_features <= k or n_iterations < 2:
            return float("nan")

        top_k_indices = [set(np.argsort(np.abs(run))[-k:]) for run in shap_runs]
        jaccard_scores = []
        for i in range(n_iterations):
            for j in range(i + 1, n_iterations):
                set_i = top_k_indices[i]
                set_j = top_k_indices[j]
                jaccard_scores.append(len(set_i & set_j) / len(set_i | set_j))

        return float(np.mean(jaccard_scores))

    @staticmethod
    def calculate_attribution_sum(shap_runs):
        """Mean over runs of sum(phi)."""
        return float(np.mean(np.sum(shap_runs, axis=1)))

    @staticmethod
    def calculate_efficiency_gap(shap_runs, f_x, base_values):
        """
        Efficiency check: mean |sum(phi) - (f(x) - E_bg[f])| using each run's own background
        expectation. Comparing against a different background's E[f] mixes bias in the
        background summary with error in the attributions.
        """
        return float(np.mean(np.abs(np.sum(shap_runs, axis=1) - (f_x - np.asarray(base_values)))))

    @staticmethod
    def calculate_reference_error(shap_runs, reference_phi):
        """
        V3: Fidelity — mean over runs of the RMSE between the run's attributions and a
        high-budget reference explanation of the same instance.
        """
        return float(np.mean(np.sqrt(np.mean((shap_runs - reference_phi) ** 2, axis=1))))

    @staticmethod
    def is_degenerate(shap_runs, tol=1e-8):
        """True when every attribution is ~0 (ACV/Jaccard become trivially perfect)."""
        return bool(np.all(np.abs(shap_runs) < tol))
