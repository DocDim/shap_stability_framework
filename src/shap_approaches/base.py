# src/shap_approaches/base.py
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from shap.utils._legacy import DenseData


@dataclass
class Explanation:
    """One explanation run: per-feature attributions plus what is needed to audit them."""
    phi: np.ndarray        # (n_features,) positive-class attributions
    base_value: float      # E_bg[f] for the background this run used; sum(phi) ≈ f(x) - base_value
    n_coalitions: int      # Coalitions actually evaluated (capped at 2^M - 2 by KernelExplainer)
    n_background: int      # Background rows each coalition is averaged over

    @property
    def model_evals(self) -> int:
        """Nominal compute cost: rows passed to the model (coalitions x background rows)."""
        return self.n_coalitions * self.n_background


@dataclass
class RepeatedRuns:
    phi: np.ndarray           # (n_iterations, n_features)
    base_values: np.ndarray   # (n_iterations,)
    n_coalitions: np.ndarray  # (n_iterations,)
    n_background: np.ndarray  # (n_iterations,)
    seconds: np.ndarray       # (n_iterations,) wall-clock time per explanation

    @property
    def model_evals(self) -> np.ndarray:
        return self.n_coalitions * self.n_background


class BaseSHAPApproach(ABC):
    """
    Abstract Base Class for repeated SHAP stability experiments.
    """
    # Which attribution the approach estimates; selects the matching fidelity reference
    estimand = "shapley"

    def __init__(self, model, background_data, random_state=None, background_size=50):
        self.model = model
        self.background_data = background_data
        self.background_size = background_size
        self.rng = np.random.default_rng(random_state)
        # Index of the current repeat; approaches with a pool of background summaries use it
        # so that repeat i sees the same summary for every instance, model and budget
        self.iteration = 0

    def reseed(self, seed):
        """Resets background resampling, KernelExplainer's coalition sampling (global NumPy RNG)
        and the repeat counter."""
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)
        self.iteration = 0

    def _sample_background(self, n=None):
        """Draws a fresh background subsample each call (no fixed seed per iteration)."""
        n = min(n or self.background_size, len(self.background_data))
        idx = self.rng.choice(len(self.background_data), size=n, replace=False)
        return self.background_data.iloc[idx]

    @staticmethod
    def _explain(explainer, target_instance, n_samples, l1_reg=False) -> Explanation:
        """Runs a KernelExplainer and keeps the positive-class attributions and base value."""
        shap_values = explainer.shap_values(target_instance, nsamples=n_samples, l1_reg=l1_reg, silent=True)
        if isinstance(shap_values, list):  # Older shap: one array per class
            shap_values = shap_values[1]
        values = np.asarray(shap_values)
        if values.ndim == 3:  # (n_rows, n_features, n_classes)
            values = values[..., 1]
        base_value = np.atleast_1d(explainer.expected_value)[-1]
        # nsamples is only set when more than one feature varies; otherwise the result is exact
        return Explanation(values.reshape(-1), float(base_value), int(getattr(explainer, "nsamples", 0)),
                           int(explainer.N))

    @abstractmethod
    def run_iteration(self, target_instance, n_samples=200) -> Explanation:
        """
        Executes a single explanation run for the given target instance.
        """

    def run_repeated_experiments(self, target_instance, n_iterations=30, n_samples=200) -> RepeatedRuns:
        """
        Runs repeated iterations to evaluate post-hoc explanation variance.
        """
        runs, seconds = [], []
        for i in range(n_iterations):
            self.iteration = i
            start = time.perf_counter()
            runs.append(self.run_iteration(target_instance, n_samples=n_samples))
            seconds.append(time.perf_counter() - start)
        return RepeatedRuns(
            phi=np.array([r.phi for r in runs]),
            base_values=np.array([r.base_value for r in runs]),
            n_coalitions=np.array([r.n_coalitions for r in runs]),
            n_background=np.array([r.n_background for r in runs]),
            seconds=np.array(seconds),
        )


def weighted_background(rows, weights):
    """Background with per-row weights (e.g. cluster sizes), as shap.kmeans builds it."""
    return DenseData(np.asarray(rows, dtype=float), list(rows.columns), None, np.array(weights, dtype=float))
