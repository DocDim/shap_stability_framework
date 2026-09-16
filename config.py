import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUTS_TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
MODEL_SAVED_DIR = os.path.join(BASE_DIR, "src", "models", "saved")

# Financial Paths
FINANCIAL_RAW_DATA_FILE = os.path.join(DATA_DIR, "lendingclub_raw.csv")

# Clinical Paths (UCI Diabetes 130-US Hospitals)
CLINICAL_FEATURES_FILE = os.path.join(DATA_DIR, "diabetes_130_features.csv")
CLINICAL_TARGETS_FILE = os.path.join(DATA_DIR, "diabetes_130_targets.csv")

# Pipeline Parameters
MISSING_THRESHOLD = 0.50  # 50% missingness threshold (Belcastro et al., 2023)
TARGET_COL = "loan_status"
POSITIVE_CLASS = "Charged Off"
NEGATIVE_CLASS = "Fully Paid"
SPLIT_DATE = "2014-01-01"  # Out-of-time train/test boundary