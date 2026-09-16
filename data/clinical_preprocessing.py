"""
clinical_preprocessing.py — Data ingestion, quality audit, readmission target
                            construction, clinical feature selection, encoding,
                            splitting, and scaling for UCI Diabetes 130-US Hospitals.
"""

import os
import logging
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility — PSI Covariate Drift Audit
# ---------------------------------------------------------------------------

def calculate_psi(train_col: pd.Series, test_col: pd.Series, num_buckets: int = 10) -> float:
    """Calculates Population Stability Index (PSI) between train and test clinical distributions."""
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


# ---------------------------------------------------------------------------
# Step 1 — load_and_audit
# ---------------------------------------------------------------------------

def load_and_audit(
    features_path: str = config.CLINICAL_FEATURES_FILE,
    targets_path: str = config.CLINICAL_TARGETS_FILE
) -> pd.DataFrame:
    """
    Loads features and targets from UCI Diabetes 130-US Hospitals dataset,
    merges them, treats '?' as NaN, and generates a data quality audit.
    """
    log.info("Step 1 — Loading raw clinical data from: %s and %s", features_path, targets_path)    
   
    df_features = pd.read_csv(features_path, low_memory=False)
    df_targets = pd.read_csv(targets_path, low_memory=False)

    # Standardize '?' missing value indicators in clinical data to np.nan
    df = pd.concat([df_features, df_targets], axis=1).replace("?", np.nan)
    log.info("  Raw clinical shape: %s rows × %s columns", *df.shape)

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
    audit_path = os.path.join(config.OUTPUTS_TABLES_DIR, "clinical_audit.csv")
    audit.to_csv(audit_path)
    log.info("  Clinical audit table written to: %s", audit_path)

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

    # Binary transformation
    df["readmitted_30d"] = (df[target_col].astype(str).str.strip() == "<30").astype(int)
    
    # Drop raw multiclass target
    df = df.drop(columns=[target_col])
    
    pos_count = int(df["readmitted_30d"].sum())
    total_count = len(df)
    pos_rate = (pos_count / total_count) * 100
    
    log.info("  Total observations: %d", total_count)
    log.info("  Class 1 (<30 days): %d (%.2f%%)", pos_count, pos_rate)
    log.info("  Class 0 (>=30 / NO): %d (%.2f%%)", total_count - pos_count, 100 - pos_rate)
    
    return df


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
    {"feature": "max_glu_serum",            "encoding": "OHE", "keep": True,  "reason": "Blood glucose test result flag"},
    {"feature": "A1Cresult",                "encoding": "OHE", "keep": True,  "reason": "HbA1c test result flag"},
    {"feature": "metformin",                "encoding": "OHE", "keep": True,  "reason": "Diabetes medication adjustment"},
    {"feature": "insulin",                  "encoding": "OHE", "keep": True,  "reason": "Insulin medication adjustment"},
    {"feature": "change",                   "encoding": "OHE", "keep": True,  "reason": "Medication dosage change flag"},
    {"feature": "diabetesMed",              "encoding": "OHE", "keep": True,  "reason": "Prescribed diabetic medication flag"},

    # ── Identifiers (Drop) ────────────────────────────────────────────────
    {"feature": "encounter_id",             "encoding": "—",   "keep": False, "reason": "Unique encounter ID — no predictive signal"},
    {"feature": "patient_nbr",              "encoding": "—",   "keep": False, "reason": "Unique patient ID — no predictive signal"},
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

    available = set(df.columns) - set(high_missing) - {"readmitted_30d"}
    keep = [f for f in CLINICAL_FEATURES_TO_KEEP if f in available]
    keep_with_target = keep + ["readmitted_30d"]

    df = df[keep_with_target].copy()
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
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits clinical data into train/test sets using stratification on the readmission target.
    """
    log.info("Step 5 — Splitting clinical data (80/20 Stratified Split)")
    
    X = df.drop(columns=["readmitted_30d"])
    y = df["readmitted_30d"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    log.info("  Train: %d instances (Readmission rate: %.2f%%)", len(X_train), y_train.mean() * 100)
    log.info("  Test:  %d instances (Readmission rate: %.2f%%)", len(X_test), y_test.mean() * 100)

    # Persist clinical split statistics table
    stats = pd.DataFrame({
        "split": ["train", "test"],
        "n_obs": [len(X_train), len(X_test)],
        "n_readmitted": [int(y_train.sum()), int(y_test.sum())],
        "readmission_rate": [y_train.mean(), y_test.mean()]
    })
    stats_path = os.path.join(config.OUTPUTS_TABLES_DIR, "clinical_split_statistics.csv")
    stats.to_csv(stats_path, index=False)
    
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 6 — scale
# ---------------------------------------------------------------------------

def scale(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fits median imputer and standard scaler on training split only,
    computes PSI covariate drift metrics, and persists artifacts.
    """
    log.info("Step 6 — Clinical scaling (fit on train only) & PSI Covariate Drift Audit")

    num_cols = X_train.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns.tolist()
    dummy_cols = [c for c in num_cols if X_train[c].dropna().isin([0.0, 1.0]).all()]
    scale_cols = [c for c in num_cols if c not in dummy_cols]

    # Covariate drift assessment (PSI)
    for col in scale_cols:
        psi_val = calculate_psi(X_train[col], X_test[col])
        if psi_val > 0.25:
            log.warning("  High covariate drift detected in clinical feature '%s' (PSI = %.4f)", col, psi_val)

    # Imputation (Median for clinical lab measurements and count variables)
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train[scale_cols])

    X_train_imp = X_train.copy()
    X_test_imp = X_test.copy()
    X_train_imp[scale_cols] = imputer.transform(X_train[scale_cols])
    X_test_imp[scale_cols] = imputer.transform(X_test[scale_cols])

    # Scaling
    scaler = StandardScaler()
    scaler.fit(X_train_imp[scale_cols])

    X_train_scaled = X_train_imp.copy()
    X_test_scaled = X_test_imp.copy()
    X_train_scaled[scale_cols] = scaler.transform(X_train_imp[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test_imp[scale_cols])

    # Persist artifacts
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    joblib.dump(imputer, os.path.join(config.DATA_PROCESSED_DIR, "clinical_imputer.joblib"))
    joblib.dump(scaler, os.path.join(config.DATA_PROCESSED_DIR, "clinical_scaler.joblib"))

    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_clinical_preprocessing_pipeline(
    features_path: str = config.CLINICAL_FEATURES_FILE,
    targets_path: str = config.CLINICAL_TARGETS_FILE
):
    df = load_and_audit(features_path, targets_path)
    df = define_target(df)
    df = select_features(df)
    df = encode(df)
    X_train, X_test, y_train, y_test = split_clinical_data(df)
    X_train_sc, X_test_sc, scaler = scale(X_train, X_test)

    # Persist processed Parquet/CSV splits
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    X_train_sc.to_parquet(os.path.join(config.DATA_PROCESSED_DIR, "X_train_clinical.parquet"))
    X_test_sc.to_parquet(os.path.join(config.DATA_PROCESSED_DIR, "X_test_clinical.parquet"))
    y_train.to_csv(os.path.join(config.DATA_PROCESSED_DIR, "y_train_clinical.csv"), header=True, index=False)
    y_test.to_csv(os.path.join(config.DATA_PROCESSED_DIR, "y_test_clinical.csv"), header=True, index=False)
    log.info("Processed clinical splits saved to: %s", config.DATA_PROCESSED_DIR)

    return X_train_sc, X_test_sc, y_train, y_test, scaler