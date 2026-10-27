"""
common.py — Preprocessing steps shared by the financial and clinical pipelines:
            missingness audit, PSI drift audit, and train-only imputation/scaling.
"""

import os
import logging
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import config

log = logging.getLogger(__name__)


def calculate_psi(train_col: pd.Series, test_col: pd.Series, num_buckets: int = 10) -> float:
    """Population Stability Index between train and test, using train-quantile buckets."""
    train_vals = train_col.dropna()
    test_vals = test_col.dropna()
    if train_vals.empty or test_vals.empty:
        return float("nan")

    buckets = np.unique(np.percentile(train_vals, np.linspace(0, 100, num_buckets + 1)))
    if len(buckets) < 2:
        return 0.0  # Constant on train: no buckets to compare
    buckets[0] = -np.inf
    buckets[-1] = np.inf

    train_counts, _ = np.histogram(train_vals, bins=buckets)
    test_counts, _ = np.histogram(test_vals, bins=buckets)

    # Proportions of non-null values; empty buckets get a small floor to keep log() finite
    train_pct = np.maximum(train_counts / len(train_vals), 1e-4)
    test_pct = np.maximum(test_counts / len(test_vals), 1e-4)
    return float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))


def write_audit(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Writes a per-column missingness / dtype / cardinality audit to outputs/tables."""
    audit = pd.DataFrame({
        "null_rate":   df.isnull().mean(),
        "dtype":       df.dtypes.astype(str),
        "n_unique":    df.nunique(),
        "n_nonnull":   df.notnull().sum(),
    })
    audit.index.name = "column"
    audit = audit.sort_values("null_rate", ascending=False)
    audit["exceeds_threshold"] = audit["null_rate"] > config.MISSING_THRESHOLD

    os.makedirs(config.OUTPUTS_TABLES_DIR, exist_ok=True)
    audit_path = os.path.join(config.OUTPUTS_TABLES_DIR, filename)
    audit.to_csv(audit_path)
    log.info("  Audit table written to: %s", audit_path)
    return audit


def impute_and_scale(
    X_train: pd.DataFrame, X_test: pd.DataFrame, artifact_prefix: str = ""
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fits a median imputer and standard scaler on the training split only (0/1 dummies are
    left untouched), audits PSI drift for each scaled column, and persists the artifacts.
    """
    num_cols = X_train.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns.tolist()
    dummy_cols = [c for c in num_cols if X_train[c].dropna().isin([0.0, 1.0]).all()]
    scale_cols = [c for c in num_cols if c not in dummy_cols]

    # Covariate drift assessment (PSI)
    psi = pd.Series({col: calculate_psi(X_train[col], X_test[col]) for col in scale_cols}, name="psi")
    for col, psi_val in psi[psi > 0.25].items():
        log.warning("  High covariate drift detected in '%s' (PSI = %.4f)", col, psi_val)
    os.makedirs(config.OUTPUTS_TABLES_DIR, exist_ok=True)
    psi.rename_axis("feature").to_csv(os.path.join(config.OUTPUTS_TABLES_DIR, f"{artifact_prefix}psi.csv"))

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imp = imputer.fit_transform(X_train[scale_cols])
    test_imp = imputer.transform(X_test[scale_cols])

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[scale_cols] = scaler.fit_transform(train_imp)
    X_test_scaled[scale_cols] = scaler.transform(test_imp)

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    joblib.dump(imputer, os.path.join(config.DATA_PROCESSED_DIR, f"{artifact_prefix}imputer.joblib"))
    joblib.dump(scaler, os.path.join(config.DATA_PROCESSED_DIR, f"{artifact_prefix}scaler.joblib"))

    return X_train_scaled, X_test_scaled, scaler
