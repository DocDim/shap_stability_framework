import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
OUTPUTS_TABLES_DIR = os.path.join(OUTPUTS_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
EXPERIMENTS_DIR = os.path.join(OUTPUTS_DIR, "experiments")
LOG_FILE = os.path.join(OUTPUTS_DIR, "experiment_log.txt")
MODEL_SAVED_DIR = os.path.join(BASE_DIR, "src", "models", "saved")

# Financial Paths
FINANCIAL_RAW_DATA_FILE = os.path.join(DATA_DIR, "lendingclub_raw.csv")

# Clinical Paths (UCI Diabetes 130-US Hospitals)
CLINICAL_FEATURES_FILE = os.path.join(DATA_DIR, "diabetes_130_features.csv")
CLINICAL_TARGETS_FILE = os.path.join(DATA_DIR, "diabetes_130_targets.csv")
CLINICAL_IDS_FILE = os.path.join(DATA_DIR, "diabetes_130_ids.csv")  # encounter_id, patient_nbr

# Pipeline Parameters
SEED = 42
MISSING_THRESHOLD = 0.50  # 50% missingness threshold (Belcastro et al., 2023)
TARGET_COL = "loan_status"
POSITIVE_CLASS = "Charged Off"
NEGATIVE_CLASS = "Fully Paid"
SPLIT_DATE = "2014-01-01"  # Out-of-time train/test boundary
CLINICAL_TEST_SIZE = 0.2   # Patient-grouped, stratified holdout
# Expired (11, 19, 20, 21) or hospice (13, 14) discharges cannot be readmitted (Strack et al., 2014)
CLINICAL_EXCLUDED_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]

# Experiment grid
DOMAINS = ["financial", "clinical"]
EXPERIMENT_VERSION = 6          # Bump when approach code changes so old results are not resumed
N_ITERATIONS = 30               # Repeated explanations per instance
N_INSTANCES = 20                # Test instances per model, spread across the predicted-risk range
# KernelSHAP nsamples = multiplier x n_features. Starts at 2x: at 1x the KernelSHAP regression
# has about as many coalitions as unknowns and is singular, so attributions can explode.
BUDGET_MULTIPLIERS = [2, 4, 8]
BACKGROUND_SIZE = 50            # Rows resampled per iteration (Baseline, L1, Owen)
N_BG_CLUSTERS = 10              # KMeans background size (Approaches 1 and 4)
BACKGROUND_SUMMARY = "medoids"  # "medoids" (real records) or "centroids"
# Approaches 1 and 4: refit the KMeans summary on a fresh bootstrap sample of the training data
# for each repeat (one summary per iteration, shared across instances, models and budgets), so
# the summary's own sampling variability counts toward stability just as Baseline's background
# resampling does. False = one fixed summary (Approach 4 is then deterministic).
BOOTSTRAP_SUMMARIES = True
# Approach 4: the Partition masker averages its background uniformly, so the weighted
# KMeans summary is replicated into this many rows in proportion to cluster size
OWEN_KMEANS_BACKGROUND_ROWS = 100
# Approaches 3 and 4: features whose average-linkage distance 1 - |corr| is below this form
# one coalition group; groups and their members are joined in a balanced binary tree
OWEN_GROUP_DISTANCE = 0.5
# Fidelity references: one per estimand, both on the same fixed background so they differ
# only in what they estimate
REFERENCE_BACKGROUND_SIZE = 200  # Fixed background rows shared by both references
REFERENCE_REPEATS = 3            # Shapley reference repeats (different seeds) to measure its own noise
TOP_K = 5                     # Jaccard top-k
MIN_MODEL_AUC = 0.55            # Models below this are flagged in the analysis
