"""
clinical_preprocessing.py — Data ingestion, quality audit, readmission target
                            construction, clinical feature selection, encoding,
                            splitting, and scaling for UCI Diabetes 130-US Hospitals.
"""

import os
import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config
from data.common import write_audit, impute_and_scale

log = logging.getLogger(__name__)

GROUP_COL = "patient_nbr"
TARGET = "readmitted_30d"

# UCI encodes "test not performed" as the string "None", which pandas reads as NaN.
# It is an informative category, not missing data.
NOT_MEASURED_COLS = ["max_glu_serum", "A1Cresult"]


# ---------------------------------------------------------------------------
# Step 1 — load_and_audit
# ---------------------------------------------------------------------------

def load_and_audit(
    features_path: str = config.CLINICAL_FEATURES_FILE,
    targets_path: str = config.CLINICAL_TARGETS_FILE,
    ids_path: str = config.CLINICAL_IDS_FILE,
) -> pd.DataFrame:
    """
    Loads features, targets and patient IDs from UCI Diabetes 130-US Hospitals dataset,
    merges them, treats '?' as NaN, and generates a data quality audit.
    """
    log.info("Step 1 — Loading raw clinical data from: %s, %s and %s", features_path, targets_path, ids_path)

    df_features = pd.read_csv(features_path, low_memory=False)
    df_targets = pd.read_csv(targets_path, low_memory=False)
    df_ids = pd.read_csv(ids_path, low_memory=False)

    # Standardize '?' missing value indicators in clinical data to np.nan
    df = pd.concat([df_features, df_targets, df_ids], axis=1).replace("?", np.nan)
    log.info("  Raw clinical shape: %s rows × %s columns", *df.shape)

    df = fill_not_measured(df)
    write_audit(df, "clinical_audit.csv")
    return df


def fill_not_measured(df: pd.DataFrame) -> pd.DataFrame:
    """Restores the 'None' (test not performed) category for lab result columns."""
    df = df.copy()
    for col in NOT_MEASURED_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    return df


# ---------------------------------------------------------------------------
# Step 2 — define_target
# ---------------------------------------------------------------------------

def define_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the binary 30-day readmission risk target (Y in {0, 1}).

    Encoding Schema:
    ----------------
    '<30'  -> 1 (Readmitted within 30 days - High Risk)
    '>30'  -> 0 (Readmitted after 30 days)
    'NO'   -> 0 (No recorded readmission)
    """
    log.info("Step 2 — Defining binary clinical readmission target (<30 days vs >=30/NO)")
    target_col = "readmitted"

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    df = exclude_ineligible_discharges(df)

    # Binary transformation
    df[TARGET] = (df[target_col].astype(str).str.strip() == "<30").astype(int)

    # Drop raw multiclass target
    df = df.drop(columns=[target_col])

    pos_count = int(df[TARGET].sum())
    total_count = len(df)
    pos_rate = (pos_count / total_count) * 100

    log.info("  Total observations: %d", total_count)
    log.info("  Class 1 (<30 days): %d (%.2f%%)", pos_count, pos_rate)
    log.info("  Class 0 (>=30 / NO): %d (%.2f%%)", total_count - pos_count, 100 - pos_rate)

    return df


def exclude_ineligible_discharges(df: pd.DataFrame) -> pd.DataFrame:
    """Drops encounters discharged as expired or to hospice: they cannot be readmitted."""
    disposition = pd.to_numeric(df["discharge_disposition_id"], errors="coerce")
    ineligible = disposition.isin(config.CLINICAL_EXCLUDED_DISCHARGE_IDS)
    log.info("  Excluding %d expired/hospice encounters", int(ineligible.sum()))
    return df.loc[~ineligible].copy()


# ---------------------------------------------------------------------------
# Step 3 — select_features
# ---------------------------------------------------------------------------

CLINICAL_FEATURE_SPEC: List[dict] = [
    # ── Patient Demographics ──────────────────────────────────────────────
    {"feature": "race",                     "encoding": "OHE", "keep": True,  "reason": "Demographic baseline"},
    {"feature": "gender",                   "encoding": "OHE", "keep": True,  "reason": "Demographic baseline"},
    {"feature": "age",                      "encoding": "ORD", "keep": True,  "reason": "Decade bins [0-10) to [90-100)"},
    {"feature": "weight",                   "encoding": "—",   "keep": False, "reason": ">90% missingness (audit-derived)"},

    # ── Admission / Encounter Structural Factors ──────────────────────────
    {"feature": "admission_type_id",        "encoding": "OHE", "keep": True,  "reason": "Emergency, Urgent, Elective, etc."},
    {"feature": "discharge_disposition_id", "encoding": "OHE", "keep": True,  "reason": "Discharge destination proxy"},
    {"feature": "admission_source_id",      "encoding": "OHE", "keep": True,  "reason": "Referral source"},
    {"feature": "time_in_hospital",         "encoding": "NUM", "keep": True,  "reason": "Length of stay in days"},
    {"feature": "payer_code",               "encoding": "—",   "keep": False, "reason": "High missingness and administrative noise"},
    {"feature": "medical_specialty",        "encoding": "—",   "keep": False, "reason": ">50% missingness"},

    # ── Clinical Utilization & Laboratory Metrics ─────────────────────────
    {"feature": "num_lab_procedures",       "encoding": "NUM", "keep": True,  "reason": "Lab testing diagnostic intensity"},
    {"feature": "num_procedures",           "encoding": "NUM", "keep": True,  "reason": "Non-lab procedural count"},
    {"feature": "num_medications",          "encoding": "NUM", "keep": True,  "reason": "Medication count / regimen complexity"},
    {"feature": "number_outpatient",        "encoding": "NUM", "keep": True,  "reason": "Prior outpatient visits in past year"},
    {"feature": "number_emergency",         "encoding": "NUM", "keep": True,  "reason": "Prior emergency visits in past year"},
    {"feature": "number_inpatient",         "encoding": "NUM", "keep": True,  "reason": "Prior inpatient admissions in past year"},
    {"feature": "number_diagnoses",         "encoding": "NUM", "keep": True,  "reason": "Comorbidity count entered into system"},

    # ── Test Results & Medication Adjustments ─────────────────────────────
    {"feature": "max_glu_serum",            "encoding": "OHE", "keep": True,  "reason": "Blood glucose test result ('None' = not measured)"},
    {"feature": "A1Cresult",                "encoding": "OHE", "keep": True,  "reason": "HbA1c test result ('None' = not measured)"},
    {"feature": "metformin",                "encoding": "OHE", "keep": True,  "reason": "Diabetes medication adjustment"},
    {"feature": "insulin",                  "encoding": "OHE", "keep": True,  "reason": "Insulin medication adjustment"},
    {"feature": "change",                   "encoding": "OHE", "keep": True,  "reason": "Medication dosage change flag"},
    {"feature": "diabetesMed",              "encoding": "OHE", "keep": True,  "reason": "Prescribed diabetic medication flag"},

    # ── Identifiers (Drop) ────────────────────────────────────────────────
    {"feature": "encounter_id",             "encoding": "—",   "keep": False, "reason": "Unique encounter ID — no predictive signal"},
    {"feature": "patient_nbr",              "encoding": "—",   "keep": False, "reason": "Patient ID — used only to group the train/test split"},
    {"feature": "diag_1",                   "encoding": "—",   "keep": False, "reason": "High cardinality raw ICD-9 codes"},
    {"feature": "diag_2",                   "encoding": "—",   "keep": False, "reason": "High cardinality raw ICD-9 codes"},
    {"feature": "diag_3",                   "encoding": "—",   "keep": False, "reason": "High cardinality raw ICD-9 codes"},
]

CLINICAL_FEATURES_TO_KEEP: List[str] = [spec["feature"] for spec in CLINICAL_FEATURE_SPEC if spec["keep"]]


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 3 — Clinical Feature Selection")

    decision_df = pd.DataFrame(CLINICAL_FEATURE_SPEC)
    os.makedirs(config.OUTPUTS_TABLES_DIR, exist_ok=True)
    decision_df.to_csv(os.path.join(config.OUTPUTS_TABLES_DIR, "clinical_feature_decisions.csv"), index=False)

    missing_rate = df.isnull().mean()
    high_missing = missing_rate[missing_rate > config.MISSING_THRESHOLD].index.tolist()

    available = set(df.columns) - set(high_missing) - {TARGET}
    keep = [f for f in CLINICAL_FEATURES_TO_KEEP if f in available]
    dropped = [f for f in CLINICAL_FEATURES_TO_KEEP if f not in available]
    if dropped:
        log.warning("  Features marked keep=True but dropped for missingness: %s", dropped)

    # patient_nbr is carried through only as the split grouping key
    df = df[keep + [GROUP_COL, TARGET]].copy()
    log.info("  Retained %d clinical features (+ target)", len(keep))
    return df


# ---------------------------------------------------------------------------
# Step 4 — encode
# ---------------------------------------------------------------------------

AGE_MAP = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
    "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9
}


def encode(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 4 — Encoding clinical features")
    df = df.copy()

    # Ordinal encoding for decade age bins
    if "age" in df.columns:
        df["age"] = df["age"].map(AGE_MAP)

    # Convert ID-based nominal codes to categorical strings before OHE
    id_cols = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
    for col in id_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # One-hot encode nominal clinical categories (drop_first=True to avoid collinearity)
    ohe_candidates = [
        "race", "gender", "admission_type_id", "discharge_disposition_id",
        "admission_source_id", "max_glu_serum", "A1Cresult", "metformin",
        "insulin", "change", "diabetesMed"
    ]
    ohe_cols = [c for c in ohe_candidates if c in df.columns]

    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=float)

    log.info("  Shape after encoding: %d rows × %d columns", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Step 5 — split (Train / Test)
# ---------------------------------------------------------------------------

def split_clinical_data(
    df: pd.DataFrame, test_size: float = config.CLINICAL_TEST_SIZE, random_state: int = config.SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits clinical data into train/test sets, stratified on the readmission target and
    grouped by patient so that no patient's encounters appear on both sides.
    """
    log.info("Step 5 — Splitting clinical data (stratified, grouped by %s)", GROUP_COL)

    X = df.drop(columns=[TARGET, GROUP_COL])
    y = df[TARGET]
    groups = df[GROUP_COL]

    splitter = StratifiedGroupKFold(n_splits=round(1 / test_size), shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    if overlap:
        raise RuntimeError(f"{len(overlap)} patients appear in both train and test")

    log.info("  Train: %d instances (Readmission rate: %.2f%%)", len(X_train), y_train.mean() * 100)
    log.info("  Test:  %d instances (Readmission rate: %.2f%%)", len(X_test), y_test.mean() * 100)

    # Persist clinical split statistics table
    stats = pd.DataFrame({
        "split": ["train", "test"],
        "n_obs": [len(X_train), len(X_test)],
        "n_patients": [groups.iloc[train_idx].nunique(), groups.iloc[test_idx].nunique()],
        "n_readmitted": [int(y_train.sum()), int(y_test.sum())],
        "readmission_rate": [y_train.mean(), y_test.mean()]
    })
    os.makedirs(config.OUTPUTS_TABLES_DIR, exist_ok=True)
    stats_path = os.path.join(config.OUTPUTS_TABLES_DIR, "clinical_split_statistics.csv")
    stats.to_csv(stats_path, index=False)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_clinical_preprocessing_pipeline(
    features_path: str = config.CLINICAL_FEATURES_FILE,
    targets_path: str = config.CLINICAL_TARGETS_FILE,
    ids_path: str = config.CLINICAL_IDS_FILE,
):
    df = load_and_audit(features_path, targets_path, ids_path)
    df = define_target(df)
    df = select_features(df)
    df = encode(df)
    X_train, X_test, y_train, y_test = split_clinical_data(df)
    log.info("Step 6 — Clinical scaling (fit on train only) & PSI Covariate Drift Audit")
    X_train_sc, X_test_sc, scaler = impute_and_scale(X_train, X_test, artifact_prefix="clinical_")

    # Persist processed Parquet/CSV splits
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    X_train_sc.to_parquet(os.path.join(config.DATA_PROCESSED_DIR, "X_train_clinical.parquet"))
    X_test_sc.to_parquet(os.path.join(config.DATA_PROCESSED_DIR, "X_test_clinical.parquet"))
    y_train.to_csv(os.path.join(config.DATA_PROCESSED_DIR, "y_train_clinical.csv"), header=True, index=False)
    y_test.to_csv(os.path.join(config.DATA_PROCESSED_DIR, "y_test_clinical.csv"), header=True, index=False)
    log.info("Processed clinical splits saved to: %s", config.DATA_PROCESSED_DIR)

    return X_train_sc, X_test_sc, y_train, y_test, scaler
