import numpy as np
import pandas as pd
import pytest

import config
from data.common import calculate_psi
from data import financial_preprocessing as fin
from data import clinical_preprocessing as cli


def test_credit_history_handles_pre_1969_credit_lines():
    df = pd.DataFrame({"issue_d": ["Dec-11", "Dec-11"], "earliest_cr_line": ["Jan-65", "Jan-01"]})
    out = fin.encode(df)
    # 'Jan-65' must be 1965, not 2065
    assert out["credit_history_months"].tolist() == [(2011 - 1965) * 12 + 11, (2011 - 2001) * 12 + 11]


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.standard_normal(5000))
    assert calculate_psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert calculate_psi(a, a + 2.0) > 0.25


def test_not_measured_lab_results_become_a_category():
    df = pd.DataFrame({"A1Cresult": [np.nan, ">8"], "max_glu_serum": [np.nan, "Norm"]})
    out = cli.fill_not_measured(df)
    assert out["A1Cresult"].tolist() == ["None", ">8"]
    assert out["max_glu_serum"].tolist() == ["None", "Norm"]


def test_expired_and_hospice_discharges_are_excluded():
    df = pd.DataFrame({"discharge_disposition_id": [1, 11, 13, 3, 21]})
    out = cli.exclude_ineligible_discharges(df)
    assert out["discharge_disposition_id"].tolist() == [1, 3]


def test_clinical_split_keeps_each_patient_on_one_side(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS_TABLES_DIR", str(tmp_path))
    rng = np.random.default_rng(0)
    patients = np.repeat(np.arange(200), 3)  # three encounters per patient
    df = pd.DataFrame({
        "feat": rng.standard_normal(len(patients)),
        cli.GROUP_COL: patients,
        cli.TARGET: rng.integers(0, 2, len(patients)),
    })
    X_train, X_test, _, _ = cli.split_clinical_data(df, test_size=0.2, random_state=0)

    train_patients = set(df.loc[X_train.index, cli.GROUP_COL])
    test_patients = set(df.loc[X_test.index, cli.GROUP_COL])
    assert not train_patients & test_patients
    assert cli.GROUP_COL not in X_train.columns
    assert 0.15 < len(X_test) / len(df) < 0.25
