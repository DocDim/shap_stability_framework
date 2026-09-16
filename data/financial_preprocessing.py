"""
preprocessing.py — Data ingestion, quality audit, target construction,
                   feature selection, encoding, temporal splitting, and scaling.
"""

import os
import logging
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def calculate_psi(train_col: pd.Series, test_col: pd.Series, num_buckets: int = 10) -> float:
    """Calculates Population Stability Index (PSI) between train and test distributions."""
    try:
        percentiles = np.linspace(0, 100, num_buckets + 1)
        buckets = np.percentile(train_col.dropna(), percentiles)
        buckets[0] -= 1e-5
        buckets[-1] += 1e-5
        
        train_counts, _ = np.histogram(train_col.dropna(), bins=buckets)
        test_counts, _ = np.histogram(test_col.dropna(), bins=buckets)
        
        train_pct = np.where(train_counts == 0, 1e-4, train_counts) / len(train_col)
        test_pct = np.where(test_counts == 0, 1e-4, test_counts) / len(test_col)
        
        psi = np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))
        return float(psi)
    except Exception:
        return 0.0


def load_and_audit(filepath: str = config.FINANCIAL_RAW_DATA_FILE) -> pd.DataFrame:
    log.info("Step 1 — Loading raw data from: %s", filepath)
    df = pd.read_csv(filepath, low_memory=False)
    log.info("  Raw shape: %s rows × %s columns", *df.shape)

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
    audit_path = os.path.join(config.OUTPUTS_TABLES_DIR, "audit.csv")
    audit.to_csv(audit_path)
    log.info("  Audit table written to: %s", audit_path)

    return df


def define_target(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 2 — Defining binary target (filter to resolved loans only)")
    keep_statuses = {config.POSITIVE_CLASS, config.NEGATIVE_CLASS}
    before = len(df)
    df = df[df[config.TARGET_COL].isin(keep_statuses)].copy()
    after = len(df)
    log.info("  Rows before filter: %d  |  after: %d  |  removed: %d", before, after, before - after)

    df["default"] = (df[config.TARGET_COL] == config.POSITIVE_CLASS).astype(int)
    df = df.drop(columns=[config.TARGET_COL])
    return df


FEATURE_SPEC: List[dict] = [
    {"feature": "loan_amnt",          "encoding": "NUM", "keep": True},
    {"feature": "term",               "encoding": "OHE", "keep": True},
    {"feature": "purpose",            "encoding": "OHE", "keep": True},
    {"feature": "application_type",   "encoding": "OHE", "keep": True},
    {"feature": "dti",                "encoding": "NUM", "keep": True},
    {"feature": "annual_inc",         "encoding": "NUM", "keep": True},
    {"feature": "verification_status","encoding": "OHE", "keep": True},
    {"feature": "emp_length",         "encoding": "ORD", "keep": True},
    {"feature": "home_ownership",     "encoding": "OHE", "keep": True},
    {"feature": "earliest_cr_line",   "encoding": "DATE", "keep": True},
    {"feature": "open_acc",           "encoding": "NUM", "keep": True},
    {"feature": "pub_rec",            "encoding": "NUM", "keep": True},
    {"feature": "revol_bal",          "encoding": "NUM", "keep": True},
    {"feature": "revol_util",         "encoding": "NUM", "keep": True},
    {"feature": "total_acc",          "encoding": "NUM", "keep": True},
    {"feature": "inq_last_6mths",     "encoding": "NUM", "keep": True},
    {"feature": "delinq_2yrs",        "encoding": "NUM", "keep": True},
    {"feature": "addr_state",         "encoding": "OHE", "keep": True},
    {"feature": "issue_d",            "encoding": "DATE", "keep": True},
]

FEATURES_TO_KEEP: List[str] = [spec["feature"] for spec in FEATURE_SPEC if spec["keep"]]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 3 — Feature selection")
    decision_df = pd.DataFrame(FEATURE_SPEC)
    os.makedirs(config.OUTPUTS_TABLES_DIR, exist_ok=True)
    decision_df.to_csv(os.path.join(config.OUTPUTS_TABLES_DIR, "feature_decisions.csv"), index=False)

    missing_rate = df.isnull().mean()
    high_missing = missing_rate[missing_rate > config.MISSING_THRESHOLD].index.tolist()

    available = set(df.columns) - set(high_missing) - {"default"}
    keep = [f for f in FEATURES_TO_KEEP if f in available]
    keep_with_target = keep + ["default"]

    return df[keep_with_target].copy()


EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4,  "5 years": 5, "6 years": 6, "7 years": 7,
    "8 years": 8,  "9 years": 9, "10+ years": 10,
}


def encode(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 4 — Encoding features")
    df = df.copy()

    if "issue_d" in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%y", errors="coerce")

    if "earliest_cr_line" in df.columns and "issue_d" in df.columns:
        ecl = pd.to_datetime(df["earliest_cr_line"], format="%b-%y", errors="coerce")
        df["credit_history_months"] = (
            (df["issue_d"].dt.year - ecl.dt.year) * 12 + (df["issue_d"].dt.month - ecl.dt.month)
        )
        df = df.drop(columns=["earliest_cr_line"])

    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].map(EMP_LENGTH_MAP)

    if "term" in df.columns:
        df["term"] = df["term"].astype(str).str.strip().str.replace(" months", "", regex=False)

    ohe_cols = [
        c for c in ["home_ownership", "purpose", "verification_status",
                     "application_type", "term", "addr_state"]
        if c in df.columns
    ]
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=float)

    return df


def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    log.info("Step 5 — Temporal train/test split at %s", config.SPLIT_DATE)
    df = df.sort_values("issue_d").reset_index(drop=True)
    split_dt = pd.to_datetime(config.SPLIT_DATE)

    train_mask = df["issue_d"] < split_dt
    test_mask  = df["issue_d"] >= split_dt

    drop_cols = ["issue_d", "default"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_train = df.loc[train_mask, feature_cols].copy()
    X_test  = df.loc[test_mask,  feature_cols].copy()
    y_train = df.loc[train_mask, "default"].copy()
    y_test  = df.loc[test_mask,  "default"].copy()

    log.info("  Train: %d rows | Test: %d rows", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def scale(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    log.info("Step 6 — Scaling (fit on train only) & PSI Covariate Drift Audit")

    num_cols = X_train.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns.tolist()
    dummy_cols  = [c for c in num_cols if X_train[c].dropna().isin([0.0, 1.0]).all()]
    scale_cols  = [c for c in num_cols if c not in dummy_cols]

    # Covariate drift assessment (PSI)
    psi_results = {}
    for col in scale_cols:
        psi_val = calculate_psi(X_train[col], X_test[col])
        psi_results[col] = psi_val
        if psi_val > 0.25:
            log.warning("  High covariate drift detected in '%s' (PSI = %.4f)", col, psi_val)

    # Imputation & Scaling
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train[scale_cols])

    X_train_imp = X_train.copy()
    X_test_imp  = X_test.copy()
    X_train_imp[scale_cols] = imputer.transform(X_train[scale_cols])
    X_test_imp[scale_cols]  = imputer.transform(X_test[scale_cols])

    scaler = StandardScaler()
    scaler.fit(X_train_imp[scale_cols])

    X_train_scaled = X_train_imp.copy()
    X_test_scaled  = X_test_imp.copy()
    X_train_scaled[scale_cols] = scaler.transform(X_train_imp[scale_cols])
    X_test_scaled[scale_cols]  = scaler.transform(X_test_imp[scale_cols])

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    joblib.dump(imputer, os.path.join(config.DATA_PROCESSED_DIR, "imputer.joblib"))
    joblib.dump(scaler,  os.path.join(config.DATA_PROCESSED_DIR, "scaler.joblib"))

    return X_train_scaled, X_test_scaled, scaler


def run_preprocessing_pipeline(filepath: str = config.FINANCIAL_RAW_DATA_FILE):
    df = load_and_audit(filepath)
    df = define_target(df)
    df = select_features(df)
    df = encode(df)
    X_train, X_test, y_train, y_test = temporal_split(df)
    X_train_sc, X_test_sc, scaler    = scale(X_train, X_test)

    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    X_train_sc.to_parquet(os.path.join(config.DATA_PROCESSED_DIR, "X_train_financial.parquet"))
    X_test_sc.to_parquet( os.path.join(config.DATA_PROCESSED_DIR, "X_test_financial.parquet"))
    y_train.to_csv(        os.path.join(config.DATA_PROCESSED_DIR, "y_train_financial.csv"), header=True, index=False)
    y_test.to_csv(         os.path.join(config.DATA_PROCESSED_DIR, "y_test_financial.csv"),  header=True, index=False)

    return X_train_sc, X_test_sc, y_train, y_test, scaler