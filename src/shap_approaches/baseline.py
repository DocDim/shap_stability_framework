import shap
from src.shap_approaches.base import BaseSHAPApproach, Explanation


class BaselineKernelSHAP(BaseSHAPApproach):
    def run_iteration(self, target_instance, n_samples=200) -> Explanation:
        # shap.sample() defaults to random_state=0, which froze the background across
        # iterations; resample each run so background noise is part of the baseline.
        bg_sample = self._sample_background()
        explainer = shap.KernelExplainer(self.model.predict_proba, bg_sample)
        return self._explain(explainer, target_instance, n_samples)
