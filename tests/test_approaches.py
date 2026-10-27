import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.shap_approaches import (
    BaselineKernelSHAP,
    KMeansBackgroundKernelSHAP,
    PenalizedKernelSHAP,
    OwenValuesSHAP,
    HybridOwenKMeansSHAP,
    summarize_background,
    build_summary_pool,
    build_feature_hierarchy,
    replicate_by_weight,
)


@pytest.fixture(scope="module")
def synthetic():
    """Six features; x0 and x1 strongly correlated, the rest independent."""
    rng = np.random.default_rng(0)
    n = 400
    x0 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x0": x0,
        "x1": x0 + 0.1 * rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "x3": rng.standard_normal(n),
        "x4": rng.standard_normal(n),
        "x5": rng.integers(0, 2, n).astype(float),
    })
    logits = 1.5 * X["x0"] - 1.0 * X["x2"] + 0.5 * X["x5"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    model = LogisticRegression().fit(X, y)
    return model, X, X.iloc[[0]]


class AdditiveModel:
    """p(x) = 0.5 + x @ w: Shapley and Owen values are both w_j * (x_j - mean_bg(x_j))."""
    def __init__(self, w):
        self.w = np.asarray(w, dtype=float)

    def predict_proba(self, X):
        p = 0.5 + np.asarray(X, dtype=float) @ self.w
        return np.column_stack([1 - p, p])


def make_approaches(model, X):
    return {
        "baseline": BaselineKernelSHAP(model, X, random_state=0, background_size=50),
        "kmeans": KMeansBackgroundKernelSHAP(model, X, n_clusters=5),
        "l1": PenalizedKernelSHAP(model, X, random_state=0, background_size=50),
        "owen": OwenValuesSHAP(model, X, random_state=0, background_size=50),
        "hybrid": HybridOwenKMeansSHAP(model, X, n_clusters=5, n_background_rows=50),
    }


@pytest.mark.parametrize("name", ["baseline", "kmeans", "l1", "owen", "hybrid"])
def test_attributions_satisfy_efficiency(synthetic, name):
    model, X, target = synthetic
    approach = make_approaches(model, X)[name]
    approach.reseed(0)
    runs = approach.run_repeated_experiments(target, n_iterations=3, n_samples=100)

    assert runs.phi.shape == (3, X.shape[1])
    assert not np.allclose(runs.phi, 0), "all-zero attributions"
    assert (runs.n_coalitions > 0).all()
    f_x = model.predict_proba(target)[0, 1]
    np.testing.assert_allclose(runs.phi.sum(axis=1), f_x - runs.base_values, atol=1e-6)


@pytest.mark.parametrize("name, n_background", [("baseline", 50), ("kmeans", 5), ("l1", 50), ("owen", 50), ("hybrid", 50)])
def test_cost_is_recorded(synthetic, name, n_background):
    model, X, target = synthetic
    runs = make_approaches(model, X)[name].run_repeated_experiments(target, n_iterations=2, n_samples=40)
    assert (runs.n_background == n_background).all()
    np.testing.assert_array_equal(runs.model_evals, runs.n_coalitions * runs.n_background)
    assert (runs.seconds > 0).all()


def test_owen_reference_explores_full_tree_with_shared_hierarchy(synthetic):
    model, X, target = synthetic
    clustering = build_feature_hierarchy(X)
    small_bg = X.iloc[:20]
    ref = OwenValuesSHAP(model, small_bg, background_size=20, clustering=clustering)
    assert ref.clustering is clustering
    budget = 4 * X.shape[1] ** 2
    expl = ref.run_iteration(target, n_samples=budget)
    assert expl.n_coalitions < budget  # queue emptied before the budget ran out: exact
    # Deterministic for a fixed background (order of rows does not matter)
    np.testing.assert_allclose(ref.run_iteration(target, n_samples=budget).phi, expl.phi, atol=1e-12)


def test_hierarchy_merges_correlated_features_first(synthetic):
    _, X, _ = synthetic
    Z = build_feature_hierarchy(X)
    assert Z.shape == (X.shape[1] - 1, 4)
    assert sorted(Z[0, :2].astype(int)) == [0, 1]


def test_owen_values_give_correlated_features_their_own_credit(synthetic):
    _, X, _ = synthetic
    w = np.array([0.03, -0.01, 0.02, 0.0, 0.01, 0.05])  # x0 and x1 correlated but weighted differently
    model = AdditiveModel(w)
    approach = OwenValuesSHAP(model, X, random_state=0, background_size=50)
    approach.reseed(0)
    bg_mean = X.iloc[approach.rng.choice(len(X), size=50, replace=False)].mean().to_numpy()

    approach.reseed(0)
    target = X.iloc[[0]]
    phi = approach.run_iteration(target, n_samples=500).phi
    np.testing.assert_allclose(phi, w * (target.to_numpy()[0] - bg_mean), atol=1e-9)


def test_replicate_by_weight_matches_proportions():
    rows = np.arange(3).reshape(3, 1).astype(float)
    out = replicate_by_weight(rows, np.array([0.5, 0.3, 0.2]), n_rows=10)
    assert len(out) == 10
    assert np.bincount(out[:, 0].astype(int)).tolist() == [5, 3, 2]


def test_medoids_are_real_records_and_weights_sum_to_one(synthetic):
    _, X, _ = synthetic
    rows, weights = summarize_background(X, n_clusters=5, method="medoids")
    assert weights.sum() == pytest.approx(1.0)
    data = {tuple(r) for r in X.to_numpy()}
    assert all(tuple(r) in data for r in rows.to_numpy())


def test_bootstrap_pool_varies_and_keeps_real_records(synthetic):
    _, X, _ = synthetic
    pool = build_summary_pool(X, n_summaries=4, n_clusters=5, random_state=0)
    assert len(pool) == 4
    data = {tuple(r) for r in X.to_numpy()}
    for rows, weights in pool:
        assert weights.sum() == pytest.approx(1.0)
        assert all(tuple(r) in data for r in rows.to_numpy())
    assert not np.array_equal(pool[0][0].to_numpy(), pool[1][0].to_numpy())

    fixed = build_summary_pool(X, n_summaries=4, n_clusters=5, bootstrap=False, random_state=0)
    assert len(fixed) == 1


def test_hybrid_varies_only_through_the_summary_pool(synthetic):
    model, X, target = synthetic
    pool = build_summary_pool(X, n_summaries=3, n_clusters=5, random_state=0)
    hybrid = HybridOwenKMeansSHAP(model, X, n_background_rows=50, summary_pool=pool)

    hybrid.reseed(0)
    first = hybrid.run_repeated_experiments(target, n_iterations=3, n_samples=60).phi
    assert not np.allclose(first[0], first[1])  # different summaries -> different explanations
    hybrid.reseed(0)
    np.testing.assert_allclose(hybrid.run_repeated_experiments(target, n_iterations=3, n_samples=60).phi, first)

    fixed = HybridOwenKMeansSHAP(model, X, n_clusters=5, n_background_rows=50)
    runs = fixed.run_repeated_experiments(target, n_iterations=3, n_samples=60).phi
    np.testing.assert_allclose(runs, np.tile(runs[0], (3, 1)))  # single summary: deterministic


def test_kmeans_approach_uses_pool_entry_per_iteration(synthetic):
    model, X, target = synthetic
    pool = build_summary_pool(X, n_summaries=2, n_clusters=5, random_state=0)
    approach = KMeansBackgroundKernelSHAP(model, X, summary_pool=pool)
    approach.reseed(0)
    runs = approach.run_repeated_experiments(target, n_iterations=4, n_samples=100)
    # Repeats 0 and 2 share summary 0 and, with full enumeration (2^6 - 2 <= 100), are identical
    np.testing.assert_allclose(runs.phi[0], runs.phi[2])
    assert not np.allclose(runs.base_values[0], runs.base_values[1])


@pytest.mark.parametrize("cls", [BaselineKernelSHAP, OwenValuesSHAP])
def test_reseed_makes_runs_reproducible(synthetic, cls):
    model, X, target = synthetic
    approach = cls(model, X, background_size=20)
    approach.reseed(7)
    first = approach.run_repeated_experiments(target, n_iterations=2, n_samples=30).phi
    approach.reseed(7)
    second = approach.run_repeated_experiments(target, n_iterations=2, n_samples=30).phi
    np.testing.assert_array_equal(first, second)
