import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
import config
from data.financial_loader import load_preprocessed_finance_dataset
from data.clinical_loader import load_clinical_dataset
from src.models.trainer import ModelManager
from src.shap_approaches import (
    BaselineKernelSHAP,
    KMeansBackgroundKernelSHAP,
    PenalizedKernelSHAP,
    OwenValuesKernelSHAP,
    HybridOwenKMeansKernelSHAP
)
from src.metrics.stability_metrics import StabilityEvaluator


class DualLogger:
    """Redirects all print statements to both stdout console and a text file."""
    def __init__(self, filepath="outputs/experiment_log.txt"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.terminal = sys.stdout
        # Append so resumed runs keep the earlier log
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger("outputs/experiment_log.txt")


# Experiment grid
BUDGETS = [20, 50, 100, 150]   # KernelSHAP nsamples per explanation
N_ITERATIONS = 30
SEED = 42
RESULTS_CSV = os.path.join(config.BASE_DIR, "outputs", "experiments_all.csv")


def _completed_keys(csv_path):
    """(domain, model, budget) combinations already written, so an interrupted run can resume."""
    if not os.path.exists(csv_path):
        return set()
    done = pd.read_csv(csv_path, usecols=["domain", "model", "budget"]).drop_duplicates()
    return set(map(tuple, done.itertuples(index=False, name=None)))


def _append_rows(csv_path, rows):
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)


def run_domain_experiment(
    domain_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
    corr_threshold: float,
    force_retrain: bool = False
):
    print("\n" + "#"*85)
    print(f"   STARTING EXPERIMENTAL PIPELINE FOR: {domain_name.upper()}")
    print("#"*85)

    domain_tag = domain_name.split()[0].lower()

    # 1. Fit every predefined model (no selection step)
    manager = ModelManager(model_dir=config.MODEL_SAVED_DIR)
    models = manager.train_all_models(X_train, X_test, y_train, y_test, domain_tag, force_retrain=force_retrain)

    target_instance = X_test.iloc[0:1]
    feature_names = list(X_train.columns)
    ref_bg = X_train.sample(n=min(1000, len(X_train)), random_state=SEED)

    # Coalitions and KMeans summaries depend only on X_train, so build each approach
    # once per domain and swap in the model being explained.
    approaches = {
        "Baseline": BaselineKernelSHAP(None, X_train),
        "Approach 1 (KMeans BG)": KMeansBackgroundKernelSHAP(None, X_train),
        "Approach 2 (L1 Lasso)": PenalizedKernelSHAP(None, X_train),
        "Approach 3 (Owen Values)": OwenValuesKernelSHAP(None, X_train, correlation_threshold=corr_threshold),
        "Approach 4 (Hybrid Owen + KMeans)": HybridOwenKMeansKernelSHAP(None, X_train, correlation_threshold=corr_threshold, n_clusters=10),
    }

    evaluator = StabilityEvaluator()
    done = _completed_keys(RESULTS_CSV)

    # 2. SHAP experiments for every model x budget
    for model_name, (model, auc) in models.items():
        f_x = float(model.predict_proba(target_instance)[0, 1])
        expected_f = float(model.predict_proba(ref_bg)[:, 1].mean())

        for budget in BUDGETS:
            if (domain_name, model_name, budget) in done:
                print(f"[{domain_tag}] {model_name} | budget={budget}: already in CSV, skipping")
                continue

            print(f"\n[{domain_tag}] {model_name} | budget={budget} | {N_ITERATIONS} iterations")
            np.random.seed(SEED)  # KernelExplainer draws coalitions from the global NumPy RNG
            rows = []

            for app_name, approach in approaches.items():
                approach.model = model
                approach.rng = np.random.default_rng(SEED)
                try:
                    runs = approach.run_repeated_experiments(target_instance, n_iterations=N_ITERATIONS, n_samples=budget)
                    status = "ok"
                except Exception as e:
                    runs = np.full((N_ITERATIONS, len(feature_names)), np.nan)
                    status = f"failed: {type(e).__name__}"
                    print(f"    {app_name:<34} FAILED ({type(e).__name__}: {str(e)[:90]})")

                if status == "ok":
                    if evaluator.is_degenerate(runs):
                        print(f"    WARNING: '{app_name}' returned all-zero attributions")
                    print(f"    {app_name:<34} ACV={evaluator.calculate_acv(runs):8.4f}  "
                          f"Jaccard@5={evaluator.calculate_jaccard_rank_agreement(runs, k=5):.4f}  "
                          f"Sum(phi)={evaluator.calculate_attribution_sum(runs):+.4f}  (ref {f_x - expected_f:+.4f})")

                for it in range(runs.shape[0]):
                    for f_idx, feat in enumerate(feature_names):
                        rows.append({
                            "domain": domain_name,
                            "model": model_name,
                            "model_test_auc": round(auc, 6),
                            "budget": budget,
                            "approach": app_name,
                            "iteration": it,
                            "feature": feat,
                            "shap_value": runs[it, f_idx],
                            "f_x": f_x,
                            "expected_f": expected_f,
                            "status": status,
                        })

            _append_rows(RESULTS_CSV, rows)

    print(f"\n[{domain_tag}] All results appended to: {RESULTS_CSV}")


def main():
    # Financial Domain D1
    X_train_fin, X_test_fin, y_train_fin, y_test_fin, _ = load_preprocessed_finance_dataset(config.FINANCIAL_RAW_DATA_FILE)
    run_domain_experiment(
        domain_name="Financial (Loan Default)",
        X_train=X_train_fin,
        X_test=X_test_fin,
        y_train=y_train_fin,
        y_test=y_test_fin,
        corr_threshold=0.5,
    )

    # Clinical Domain D2
    X_train_cli, X_test_cli, y_train_cli, y_test_cli, _ = load_clinical_dataset(
        config.CLINICAL_FEATURES_FILE, config.CLINICAL_TARGETS_FILE
    )
    run_domain_experiment(
        domain_name="Clinical (Readmission Risk)",
        X_train=X_train_cli,
        X_test=X_test_cli,
        y_train=y_train_cli,
        y_test=y_test_cli,
        corr_threshold=0.4,
    )


if __name__ == "__main__":
    main()
