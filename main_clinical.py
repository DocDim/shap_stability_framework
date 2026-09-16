import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pandas as pd
import config
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
from src.metrics.comparative_analysis import ComparativeAnalyzer
from src.metrics.visualizer import SHAPVisualizer


class DualLogger:
    """Redirects all print statements to both stdout console and a text file."""
    def __init__(self, filepath="outputs/clinical_experiment_log.txt"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger("outputs/clinical_experiment_log.txt")


def run_clinical_experiment_pipeline(force_retrain: bool = False):
    print("==========================================================")
    print("--- DATASET D2: CLINICAL DATA & RISK STRATIFICATION ---")
    print("==========================================================\n")
    
    # 1. Load Clinical Dataset (D2)
    X_train, X_test, y_train, y_test, _ = load_clinical_dataset(
        config.CLINICAL_FEATURES_FILE, config.CLINICAL_TARGETS_FILE
    )
    
    # 2. Multi-Model CV Evaluation / Cached Model Ingestion
    manager = ModelManager(model_dir=config.MODEL_SAVED_DIR, model_name="best_model_clinical.joblib")
    model_path, best_name = manager.train_and_select_best_model(
        X_train, X_test, y_train, y_test, scoring="roc_auc", cv_splits=5, force_retrain=force_retrain
    )
    
    model = manager.load_model()
    target_patient_instance = X_test.iloc[0:1]
    feature_names = list(X_train.columns)
    
    N_ITERATIONS = 30
    N_SAMPLES = 200
    CORR_THRESHOLD = 0.4
    
    print(f"\n--- Part 2: Running Parallel SHAP Experiments on Clinical Model ('{best_name}') ---")
    
    # Baseline
    print(f"Executing Baseline Experiment ({N_ITERATIONS} iterations)...")
    baseline_exp = BaselineKernelSHAP(model, X_train)
    baseline_runs = baseline_exp.run_repeated_experiments(target_patient_instance, n_iterations=N_ITERATIONS, n_samples=N_SAMPLES)
    
    # Approach 1: KMeans Background Optimization
    print("Executing Approach 1: KMeans Background Selection...")
    app1_exp = KMeansBackgroundKernelSHAP(model, X_train)
    app1_runs = app1_exp.run_repeated_experiments(target_patient_instance, n_iterations=N_ITERATIONS, n_samples=N_SAMPLES)
    
    # Approach 2: Penalized Kernel Regression
    print("Executing Approach 2: Penalized/Regularized Kernel Regression...")
    app2_exp = PenalizedKernelSHAP(model, X_train)
    app2_runs = app2_exp.run_repeated_experiments(target_patient_instance, n_iterations=N_ITERATIONS, n_samples=N_SAMPLES)
    
    # Approach 3: Owen Values
    print(f"Executing Approach 3: Owen Values (Correlation Threshold = {CORR_THRESHOLD})...")
    app3_exp = OwenValuesKernelSHAP(model, X_train, correlation_threshold=CORR_THRESHOLD)
    app3_runs = app3_exp.run_repeated_experiments(target_patient_instance, n_iterations=N_ITERATIONS, n_samples=N_SAMPLES)
    
    # Approach 4: Hybrid (Owen + KMeans)
    print("Executing Approach 4: Hybrid (Owen Values + KMeans BG)...")
    app4_exp = HybridOwenKMeansKernelSHAP(model, X_train, correlation_threshold=CORR_THRESHOLD, n_clusters=10)
    app4_runs = app4_exp.run_repeated_experiments(target_instance=target_patient_instance, n_iterations=N_ITERATIONS, n_samples=N_SAMPLES)
    
    print("\n--- Part 3: Quantitative Stability Analysis (Clinical Context) ---")
    evaluator = StabilityEvaluator()
    
    approaches = {
        "Baseline": baseline_runs,
        "Approach 1 (KMeans BG)": app1_runs,
        "Approach 2 (Penalized L2)": app2_runs,
        "Approach 3 (Owen Values)": app3_runs,
        "Approach 4 (Hybrid Owen + KMeans)": app4_runs
    }
    
    raw_results = []
    for name, runs in approaches.items():
        acv = evaluator.calculate_acv(runs)
        jaccard = evaluator.calculate_jaccard_rank_agreement(runs, k=5)
        wasserstein = evaluator.calculate_wasserstein_distance(baseline_runs, runs)
        
        raw_results.append({
            "Approach": name,
            "ACV (Volatility) [Lower = Better]": round(acv, 4),
            "Top-5 Jaccard Agreement [Higher = Better]": round(jaccard, 4),
            "Wasserstein Dist vs Baseline": round(wasserstein, 4)
        })
        
    df_raw = pd.DataFrame(raw_results)
    
    # Comparative Report
    analyzer = ComparativeAnalyzer()
    analysis_report = analyzer.generate_comparative_report(df_raw, domain_label="Clinical (Readmission Risk)")
    analyzer.print_formatted_report(analysis_report)
    
    # Save SHAP Runs & Generate Plots
    visualizer = SHAPVisualizer()
    visualizer.save_shap_runs(approaches, feature_names=feature_names, domain_name="clinical")
    visualizer.plot_attribution_variance_boxplot(approaches, feature_names=feature_names, domain_name="clinical", top_n=6)
    visualizer.plot_metric_comparison_bars(df_raw, domain_name="clinical")


if __name__ == "__main__":
    run_clinical_experiment_pipeline(force_retrain=False)