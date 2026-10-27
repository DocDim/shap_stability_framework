# src/shap_approaches/penalized_kernel.py
import shap
from src.shap_approaches.base import BaseSHAPApproach, Explanation


class PenalizedKernelSHAP(BaseSHAPApproach):
    def run_iteration(self, target_instance, n_samples=200) -> Explanation:
        bg_sample = self._sample_background()
        explainer = shap.KernelExplainer(self.model.predict_proba, bg_sample)
        # L1 (LassoLarsIC / AIC) feature selection before the weighted least squares fit.
        # Same as the deprecated l1_reg="auto" whenever < 20% of coalitions are sampled.
        return self._explain(explainer, target_instance, n_samples, l1_reg="aic")
