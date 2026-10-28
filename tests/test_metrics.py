import numpy as np
import pytest

from src.metrics.stability_metrics import StabilityEvaluator as ev
from src.metrics.comparative_analysis import bootstrap_ci


def test_bootstrap_ci_stays_in_range_and_brackets_mean():
    # Skewed, non-negative metric with most values near 0: a t-interval would go below 0
    values = np.array([0.0] * 17 + [0.5, 0.8, 1.2])
    low, high = bootstrap_ci(values)
    assert 0.0 <= low <= values.mean() <= high
    assert bootstrap_ci(values) == (low, high)  # seeded: reproducible
    assert all(np.isnan(bootstrap_ci([0.3])))


def test_acv_is_not_dominated_by_near_zero_features():
    rng = np.random.default_rng(0)
    important = 1.0 + 0.01 * rng.standard_normal(30)   # CV ≈ 0.01
    negligible = 1e-5 + 1e-3 * np.tile([1.0, -1.0], 15)  # mean 1e-5, std 1e-3: CV = 100, tiny attribution
    runs = np.column_stack([important, negligible])

    assert ev.calculate_mean_feature_cv(runs) > 10
    assert ev.calculate_acv(runs) < 0.05


def test_acv_zero_for_identical_runs_and_nan_when_all_zero():
    runs = np.tile([0.3, -0.2, 0.1], (10, 1))
    assert ev.calculate_acv(runs) == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(ev.calculate_acv(np.zeros((10, 3))))


def test_jaccard_identical_and_disjoint():
    same = np.tile(np.arange(10.0), (5, 1))
    assert ev.calculate_jaccard_rank_agreement(same, k=3) == 1.0

    a = np.array([[5, 4, 3, 0, 0, 0], [0, 0, 0, 5, 4, 3]], dtype=float)
    assert ev.calculate_jaccard_rank_agreement(a, k=3) == 0.0


def test_jaccard_is_nan_when_too_few_features_or_runs():
    runs = np.tile([0.5, 0.1, 0.3, 0.2], (4, 1))
    assert np.isnan(ev.calculate_jaccard_rank_agreement(runs, k=4))
    assert np.isnan(ev.calculate_jaccard_rank_agreement(runs[:1], k=2))


def test_efficiency_gap_and_reference_error():
    runs = np.array([[0.1, 0.2], [0.2, 0.1]])
    assert ev.calculate_efficiency_gap(runs, f_x=0.8, base_values=[0.5, 0.5]) == pytest.approx(0.0)
    assert ev.calculate_reference_error(runs, np.array([0.1, 0.2])) == pytest.approx(0.05)


def test_is_degenerate():
    assert ev.is_degenerate(np.zeros((3, 4)))
    assert not ev.is_degenerate(np.eye(3))


def test_floating_point_noise_is_not_a_significant_difference():
    import pandas as pd
    from src.metrics.comparative_analysis import ComparativeAnalyzer
    base = np.linspace(0.01, 0.02, 20)
    rows = [{"domain": "D", "model": "M", "budget": 10, "instance_id": i, "approach": a,
             "acv": v, "jaccard": 0.5, "reference_rmse": v, "base_value_bias": v}
            for a, vals in [("Baseline", base), ("Other", base + 1e-17)] for i, v in enumerate(vals)]
    comparison = ComparativeAnalyzer.compare_to_baseline(pd.DataFrame(rows))
    assert not comparison["significant"].any()
