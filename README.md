# KernelSHAP Stability & Regulatory Compliance Framework

A quantitative research and benchmarking framework designed to evaluate, diagnose, and resolve algorithmic instability in KernelSHAP explanations across high-stakes regulated domains.

---

## 📌 Overview

Post-hoc explanation methods like KernelSHAP are widely deployed to meet adverse action and legal interpretability requirements (such as **GDPR Article 22** and the **Equal Credit Opportunity Act / ECOA**). However, standard KernelSHAP suffers from severe attribution volatility across identical queries due to reference background sampling noise and independent coalition assumptions over correlated features.

This framework implements:
1. **End-to-End Leakage-Free Preprocessing Pipelines** for Financial and Clinical domains (including Population Stability Index audits and temporal/stratified split integrity).
2. **Multi-Model Benchmark**: every predefined model (Random Forest, Gradient Boosting, Logistic Regression, Naive Bayes, MLP) is trained and explained — no model selection step.
3. **Four Distinct Post-Hoc Stability Interventions**:
   * **Approach 1:** Background Data Summarization (\$K\$-Means Centroids)
   * **Approach 2:** Regularized Kernel Regression (L1 / Lasso-AIC feature selection via `l1_reg="auto"`)
   * **Approach 3:** Coalitional Feature Grouping (Owen Values via Hierarchical Correlation Clustering)
   * **Approach 4:** Hybrid Optimization (Owen Values + \$K\$-Means Background Summarization)
4. **Three-Dimensional Quantitative Evaluation**:
   * **\$V1\$:** Feature Association Coefficient of Variation (ACV Volatility)
   * **\$V2\$:** Top- \$k\$ Jaccard Rank Agreement Consistency
   * **\$V3\$:** Attribution Distribution Fidelity (Wasserstein Distance vs. Baseline)

---

## 🗂️ Project Directory Structure

```text
shap_stability_framework/
├── config.py                          # Global paths, thresholds, and configuration constants
├── main.py                            # Unified execution driver for Financial and Clinical domains
├── data/
│   ├── __init__.py                    # Ingestion and pipeline API exports
│   ├── downloaders.py                 # Automated downloaders for LendingClub & UCI Diabetes 130
│   ├── financial_preprocessing.py     # 6-step temporal pipeline & PSI audit for LendingClub (D1)
│   ├── financial_loader.py            # High-level loader for Financial data
│   ├── clinical_preprocessing.py      # 6-step stratified pipeline for UCI Diabetes 130 (D2)
│   ├── clinical_loader.py             # High-level loader for Clinical data
│   └── processed/                     # Serialized Parquet splits, scalers, and imputers
├── models/
│   ├── trainer.py                     # Multi-model selection via TimeSeriesSplit CV
│   └── saved/                         # Persisted winner model binaries (.joblib)
├── shap_approaches/
│   ├── __init__.py                    # SHAP strategy exports
│   ├── base.py                        # Abstract base class for repeated SHAP experimentation
│   ├── baseline.py                    # Standard KernelSHAP implementation
│   ├── background_sampling.py         # Approach 1: KMeans Background Centroids
│   ├── penalized_kernel.py            # Approach 2: L1 (Lasso) Regularized Regression
│   ├── owen_values.py                 # Approach 3: Hierarchical Owen Values
│   └── hybrid_owen_kmeans.py          # Approach 4: Hybrid Owen + KMeans
├── metrics/
│   ├── stability_metrics.py           # Core implementations of ACV, Jaccard, and Wasserstein
│   └── comparative_analysis.py        # Relative improvement calculators and pairwise matrices
└── outputs/
    └── tables/                        # Output CSV audits, feature decisions, and split stats
```

---

## ⚙️ Installation & Environment Setup

### 1. Prerequisites
Ensure you are using Python 3.10–3.11 with `numpy <= 2.3` (required by Numba/SHAP):

```bash
# Create conda environment
conda create -n shap_env python=3.11 -y
conda activate shap_env

# Install required dependencies
pip install \"numpy<2.4\" pandas scikit-learn scipy shap joblib ucimlrepo kagglehub pyarrow
```

### 2. Configure Kaggle Credentials (for Dataset \$D1\$)
Dataset \$D1\$ requires `kagglehub` to fetch the LendingClub dataset. Ensure your `kaggle.json` API token is located at `~/.kaggle/kaggle.json` or configured in your environment variables.

---

## 🚀 Running the Framework

Run the end-to-end benchmarking pipeline across both Financial and Clinical domains:

```bash
python main.py
```

### Logging Console Output to File

* **Windows Command Prompt:**
  ```cmd
  python main.py > outputs/experiment_results.txt
  ```

* **PowerShell (Live display + save to file):**
  ```powershell
  python main.py | Tee-Object -FilePath \"outputs/experiment_results.txt\"
  ```

---

## 📊 Evaluation Metrics Overview

| Metric | Scope | Target | Legal & Regulatory Context |
| :--- | :--- | :---: | :--- |
| **ACV (Volatility)** | Measures numerical variance of feature scores across repeated runs on the exact same instance. | **Lower** (\$\rightarrow 0.0\$) | Guarantees numerical repeatability under regulatory audits (GDPR Art. 22). |
| **Top- \$k\$ Jaccard Agreement** | Quantifies the overlap and rank consistency of the top \$k\$ key decision factors. | **Higher** (\$\rightarrow 1.0\$) | Ensures Adverse Action Notices consistently cite the same primary denial/risk factors. |
| **Wasserstein Distance** | Measures the earth mover's distance between stabilized attributions and baseline distributions. | **Fidelity Check** | Verifies that noise reduction did not distort the model's true underlying prediction profile. |

---

## 🔬 Benchmark Methodology

1. **Step 1 — Ingestion & Missingness Audit:** Removes features with >50% missing values and logs audit tables to disk.
2. **Step 2 — Target Construction:** Filters resolved credit outcomes (`Charged Off` vs. `Fully Paid`) for Finance, and 30-day readmissions (`<30`) for Clinical.
3. **Step 3 — Domain Feature Selection:** Enforces strict domain feature sets while removing identifier and leakage columns.
4. **Step 4 — Domain-Specific Encoding:** Ordinal mapping for bin categories and one-hot encoding for nominal categories.
5. **Step 5 — Leakage-Free Splitting:** Enforces Out-of-Time temporal splitting (`SPLIT_DATE = 2014-01-01`) for Finance and stratified splitting for Clinical.
6. **Step 6 — Scaler/Imputer Fit on Train Only:** Imputes missing values, scales continuous variables, and audits Population Stability Index (PSI) drift.
7. **Model Training:** Fits Random Forest, Gradient Boosting, Logistic Regression, Naive Bayes, and MLP Classifiers on the training split and caches each as `src/models/saved/<domain>_<model>.joblib` (holdout AUC is logged; no selection).
8. **Parallel SHAP Interventions:** Explains each model with all five approaches across 30 repeated runs at budgets (`nsamples`) of 20, 50, 100 and 150. Every attribution is appended to `outputs/experiments_all.csv` (long format: `domain, model, model_test_auc, budget, approach, iteration, feature, shap_value, f_x, expected_f, status`); re-running `main.py` resumes and skips completed model × budget combinations.
9. **Comparative Analysis:** Generates pairwise difference matrices and percentage gain summaries across all strategies.
