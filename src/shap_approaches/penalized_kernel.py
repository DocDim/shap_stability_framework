# src/shap_approaches/penalized_kernel.py
import shap
import numpy as np
from src.shap_approaches.base import BaseSHAPApproach


class PenalizedKernelSHAP(BaseSHAPApproach):
    def run_iteration(self, target_instance, n_samples=200) -> np.ndarray:
        bg_sample = self.background_data.sample(n=min(50, len(self.background_data)), replace=False)
        explainer = shap.KernelExplainer(self.model.predict_proba, bg_sample)
        
        shap_values = explainer.shap_values(
            target_instance,
            nsamples=n_samples,
            l1_reg="auto",
            silent=True
        )
        
        if isinstance(shap_values, list):
            # Class 1 attributions
            return np.array(shap_values[1]).flatten()
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                return shap_values[:, :, 1].flatten()
            elif shap_values.ndim == 2 and shap_values.shape[1] == target_instance.shape[1] * 2:
                # Sliced to positive class half
                return shap_values[0, target_instance.shape[1]:]
        return np.array(shap_values).flatten()