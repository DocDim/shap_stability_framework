import shap
from src.shap_approaches.base import BaseSHAPApproach

class KMeansBackgroundKernelSHAP(BaseSHAPApproach):
    def run_iteration(self, target_instance, n_samples=200):
        # shap.kmeans is deterministic, so compute the summary once and reuse it
        if not hasattr(self, "_bg_summary"):
            self._bg_summary = shap.kmeans(self.background_data, 10)
        explainer = shap.KernelExplainer(self.model.predict_proba, self._bg_summary)
        shap_values = explainer.shap_values(target_instance, nsamples=n_samples, l1_reg=False, silent=True)
        
        if isinstance(shap_values, list):
            return shap_values[1].flatten()
        return shap_values[:, :, 1].flatten() if shap_values.ndim == 3 else shap_values.flatten()