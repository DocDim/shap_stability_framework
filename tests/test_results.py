import numpy as np
import pandas as pd
import pytest

from src.metrics.results import reference_table, per_instance_metrics


def _rows(approach, estimand, phis, base=0.3, f_x=0.5, budget=10):
    return pd.DataFrame([{
        "domain": "D", "model": "M", "model_test_auc": 0.7, "budget": budget, "approach": approach,
        "estimand": estimand, "instance_id": 1, "f_x": f_x, "iteration": i, "base_value": base,
        "n_coalitions": 10, "n_background": 5, "model_evals": 50, "seconds": 0.1, "status": "ok",
        "phi__a": p[0], "phi__b": p[1],
    } for i, p in enumerate(phis)])


@pytest.fixture
def reference():
    return pd.concat([
        _rows("Reference", "shapley", [[0.1, 0.1], [0.1, 0.1], [0.2, 0.0]]),  # noisy repeats
        _rows("Reference", "owen", [[0.15, 0.05]]),
    ], ignore_index=True)


def test_reference_table_means_repeats_and_measures_noise(reference):
    refs = reference_table(reference)
    shapley = refs.loc[("M", 1, "shapley")]
    np.testing.assert_allclose(shapley["phi"], [0.4 / 3, 0.2 / 3])
    assert shapley["noise"] > 0
    assert refs.loc[("M", 1, "owen"), "noise"] == 0


def test_each_approach_is_scored_against_its_own_estimand(reference):
    runs = pd.concat([
        _rows("Baseline", "shapley", [[0.4 / 3, 0.2 / 3]] * 3),
        _rows("Owen", "owen", [[0.15, 0.05]] * 3),
    ], ignore_index=True)
    m = per_instance_metrics(runs, reference, k=1).set_index("approach")

    assert m.loc["Baseline", "reference_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert m.loc["Owen", "reference_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert m.loc["Owen", "reference_rmse_shapley"] > 0
    assert m.loc["Owen", "reference_estimand_gap"] == pytest.approx(m.loc["Owen", "reference_rmse_shapley"])
    assert m.loc["Baseline", "mean_model_evals"] == 50
