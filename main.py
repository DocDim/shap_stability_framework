import warnings
warnings.filterwarnings("ignore")

import os
import sys
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
from src.metrics.comparative_analysis import ComparativeAnalyzer
from src.metrics.visualizer import SHAPVisualizer


class DualLogger:
    """Redirects all print statements to both stdout console and a text file."""
    def __init__(self, filepath="outputs/experiment_log.txt"):
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

sys.stdout = DualLogger("outputs/experiment_log.txt")


def run_domain_experiment(
    domain_name: str, 
    X_train, 
    X_test, 
    y_train, 
    y_test, 
    model_filename: str, 
    corr_threshold: float,
    force_retrain: bool = False
):
    print("\n" + "#"*85)
    print(f"   STARTING EXPERIMENTAL PIPELINE FOR: {domain_name.upper()}")
    print("#"*85)
    
    # 1. Multi-Model CV Selection / Cached Model Ingestion
    manager = ModelManager(model_dir=config.MODEL_SAVED_DIR, model_name=model_filename)
    model_path, best_name = manager.train_and_select_best_model(
        X_train, X_test, y_train, y_test, scoring="roc_auc", cv_splits=5, force_retrain=force_retrain
    )
    
    model = manager.load_model()
    target_instance = X_test.iloc[0:1]
    feature_names = list(X_train.columns)
    
    N_ITERATIONS = 30
    N_SAMPLES = 200
    
    print(f"\n--- Part 2: Parallel SHAP Experiments on Model ('{best_name}') ---")
    
    # Baseline
    print(f"Running Baseline Experiment ({N_ITERATIONS} iterations)...")
    baseline_exp = BaselineKernelSHAP(model, X_train)
    baseline_runs = baseline_exp.run_repeated_experiments(target_instance, n_iterations=N_ITERATIONS, n_samples=N_SAMPLES)
    
    # Approach 1: KMeans Background
    print("Running Approach 1: KMeans Background Selection...")
    app1_exp = KMeansBackgroundKernelSHAP(model, X_train)
    app1_runs = app1_exp.run_repeated_experiments(target_instance, n_iterations=N_ITERATIONS, n_samples=N_SAMPLES)
    
    # Approach 2: Penalized L2 Regression
    print("Running Approach 2: Penalized/Regularized Kernel Regression...")
    app2_exp = PenalizedKernelSHAP(model, X_train)
    app2_runs = app2_exp.run_repeated_experiments(target_instance, n_iterations=N_ITERATIONS, n_samples=N_SAMPLES)
    
    # Approach 3: Owen Values
    print(f"Running Approach 3: Owen Values (Correlation Threshold = {corr_threshold})...")
    app3_exp = OwenValuesKernelSHAP(model, X_train, correlation_threshold=corr_threshold)
    app3_runs = app3_exp.run_repeated_experiments(target_instance, n_iterations=N_ITERATIONS, n_samples=N_SAMPLES)
    
    # Approach 4: Hybrid (Owen + KMeans)
    print("Running Approach 4: Hybrid (Owen Values + KMeans BG)...")
    app4_exp = HybridOwenKMeansKernelSHAP(model, X_train, correlation_threshold=corr_threshold, n_clusters=10)
    app4_runs = app4_exp.run_repeated_experiments(target_instance, n_iterations=N_ITERATIONS, n_samples=N_SAMPLES)
    
    # 2. Metric Evaluations
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
    
    # 3. Comparative Analysis
    analyzer = ComparativeAnalyzer()
    analysis_report = analyzer.generate_comparative_report(df_raw, domain_label=domain_name)
    analyzer.print_formatted_report(analysis_report)
    
    # 4. Save Raw SHAP Arrays & Generate Figures
    clean_domain_tag = domain_name.split()[0].lower()
    visualizer = SHAPVisualizer()
    visualizer.save_shap_runs(approaches, feature_names=feature_names, domain_name=clean_domain_tag)
    visualizer.plot_attribution_variance_boxplot(approaches, feature_names=feature_names, domain_name=clean_domain_tag, top_n=6)
    visualizer.plot_metric_comparison_bars(df_raw, domain_name=clean_domain_tag)


def main():
    # Financial Domain D1 (Set force_retrain=True to fit model to full 82 real features)
    X_train_fin, X_test_fin, y_train_fin, y_test_fin, _ = load_preprocessed_finance_dataset(config.FINANCIAL_RAW_DATA_FILE)
    run_domain_experiment(
        domain_name="Financial (Loan Default)",
        X_train=X_train_fin, 
        X_test=X_test_fin,
        y_train=y_train_fin, 
        y_test=y_test_fin,
        model_filename="best_model_finance.joblib",
        corr_threshold=0.5,
        force_retrain=False
    )
    
    # Clinical Domain D2 (Set force_retrain=True to fit model to real Diabetes-130 features)
    X_train_cli, X_test_cli, y_train_cli, y_test_cli, _ = load_clinical_dataset(
        config.CLINICAL_FEATURES_FILE, config.CLINICAL_TARGETS_FILE
    )
    run_domain_experiment(
        domain_name="Clinical (Readmission Risk)",
        X_train=X_train_cli, 
        X_test=X_test_cli,
        y_train=y_train_cli, 
        y_test=y_test_cli,
        model_filename="best_model_clinical.joblib",
        corr_threshold=0.4,
        force_retrain=False
    )


if __name__ == "__main__":
    main()