# src/shap_approaches/base.py
from abc import ABC, abstractmethod
import numpy as np


class BaseSHAPApproach(ABC):
    """
    Abstract Base Class for repeated SHAP stability experiments.
    """
    def __init__(self, model, background_data, random_state=None):
        self.model = model
        self.background_data = background_data
        self.rng = np.random.default_rng(random_state)

    def _sample_background(self, n=50):
        """Draws a fresh background subsample each call (no fixed seed per iteration)."""
        n = min(n, len(self.background_data))
        idx = self.rng.choice(len(self.background_data), size=n, replace=False)
        return self.background_data.iloc[idx]

    @abstractmethod
    def run_iteration(self, target_instance, n_samples=200) -> np.ndarray:
        """
        Executes a single explanation run for the given target instance.
        Must return a 1D numpy array of SHAP values.
        """
        pass

    def run_repeated_experiments(self, target_instance, n_iterations=30, n_samples=200) -> np.ndarray:
        """
        Runs repeated iterations to evaluate post-hoc explanation variance.
        Returns a 2D array of shape (n_iterations, n_features).
        """
        all_runs = []
        for i in range(n_iterations):
            shap_vals = self.run_iteration(target_instance, n_samples=n_samples)
            all_runs.append(shap_vals)
        return np.array(all_runs)
