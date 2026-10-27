# KernelSHAP Stability & Regulatory Compliance Framework

A quantitative research and benchmarking framework for evaluating algorithmic instability in KernelSHAP explanations, and the interventions proposed to reduce it, across high-stakes regulated domains.

---

## 📌 Overview

Post-hoc explanation methods like KernelSHAP are widely deployed to meet adverse action and legal interpretability requirements (such as **GDPR Article 22** and the **Equal Credit Opportunity Act / ECOA**). However, KernelSHAP attributions vary across repeated queries for the same instance, because of background sampling noise, coalition sampling noise, and the independence assumption over correlated features.

This framework implements:
1. **Leakage-Free Preprocessing Pipelines** for a Financial (LendingClub, out-of-time split) and a Clinical (UCI Diabetes 130, patient-grouped split) domain, with missingness and Population Stability Index (PSI) audits.
2. **Multi-Model Benchmark**: every predefined model (Random Forest, Gradient Boosting, Logistic Regression, Naive Bayes, MLP) is trained and explained — no model selection step.
3. **Four Post-Hoc Stability Interventions** compared against a baseline:
   * **Baseline:** KernelSHAP with a fresh 50-row background sample per run
   * **Approach 1:** Background summarization (K-Means, cluster-size weighted; medoids by default), refitted on a bootstrap sample of the training data for each repeat
   * **Approach 2:** L1-regularized kernel regression (LassoLarsIC / AIC, `l1_reg="aic"`)
   * **Approach 3:** Owen values over a feature correlation hierarchy (`shap.PartitionExplainer`; correlated groups from average linkage on 1 − |ρ|, joined in a balanced binary tree), fresh 50-row background per run
   * **Approach 4:** Hybrid — Owen values against the same bootstrapped K-Means backgrounds (replicated by cluster weight, since the Partition masker averages its background uniformly)

   Every approach is exposed to the sampling variability of its own background: Baseline, L1 and Owen redraw 50 rows per repeat, and the K-Means approaches use a summary refitted on a fresh bootstrap sample per repeat (`BOOTSTRAP_SUMMARIES`). Without this, a fixed summary would hide that variability and overstate the K-Means approaches' stability.
4. **Evaluation across many instances and budgets**, with 95% confidence intervals and paired significance tests against the baseline.

---

## 🗂️ Project Directory Structure

```text
shap_stability_framework/
├── config.py                          # Paths, pipeline parameters and the experiment grid
├── main.py                            # Runs every domain × model × budget × approach; writes Parquet results
├── analyze.py                         # Metrics, baseline comparisons, tables and figures from a run
├── data/
│   ├── downloaders.py                 # Downloaders for LendingClub (kagglehub) & UCI Diabetes 130 (ucimlrepo)
│   ├── common.py                      # Shared audit, PSI and train-only impute/scale steps
│   ├── financial_preprocessing.py     # 6-step temporal pipeline for LendingClub (D1)
│   ├── financial_loader.py            # High-level loader for Financial data
│   ├── clinical_preprocessing.py      # 6-step patient-grouped pipeline for UCI Diabetes 130 (D2)
│   ├── clinical_loader.py             # High-level loader for Clinical data
│   └── processed/                     # (generated) Parquet splits, scalers, imputers
├── src/
│   ├── models/
│   │   ├── trainer.py                 # Trains/caches all models; retrains caches fitted on other features
│   │   └── saved/                     # (generated) <domain>_<model>.joblib
│   ├── shap_approaches/
│   │   ├── base.py                    # Abstract base class; seeding, positive-class extraction, base values
│   │   ├── baseline.py                # Baseline KernelSHAP
│   │   ├── kmeans.py                  # Approach 1 + shared K-Means medoid/centroid background summary
│   │   ├── penalized_kernel.py        # Approach 2: L1 (Lasso/AIC) regularized regression
│   │   ├── owen_values.py             # Approach 3: Owen values (PartitionExplainer) + feature hierarchy
│   │   └── hybrid_owen_kmeans.py      # Approach 4: Owen values + K-Means background
│   └── metrics/
│       ├── stability_metrics.py       # ACV, Jaccard, reference error, efficiency check
│       ├── results.py                 # Loads runs and computes per-instance metrics
│       ├── comparative_analysis.py    # Summaries with CIs; Wilcoxon tests vs baseline (BH-corrected)
│       └── visualizer.py              # Metric-vs-budget curves and attribution boxplots
├── tests/                             # pytest unit tests (metrics, approaches, preprocessing)
├── docs/                              # Historical reports
└── outputs/                           # (generated) experiments/, tables/, figures/, experiment_log.txt
```

---

## ⚙️ Installation & Environment Setup

### 1. Prerequisites
Requires **Python 3.10–3.13**; tested with Python 3.13 and the pinned versions in `requirements.txt`. Python 3.14 is not supported: numba 0.62.1 (required by shap) refuses to build on it, and pip fails with `Cannot install on Python version 3.14`.

```bash
conda create -n shap_env python=3.13 -y
conda activate shap_env
pip install -r requirements.txt
```

Without conda, [uv](https://docs.astral.sh/uv/) can download Python 3.13 for you:

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

To use the environment from Jupyter, register it as a kernel and select it in the notebook:

```bash
pip install ipykernel
python -m ipykernel install --user --name shap_env --display-name "Python 3.13 (shap_env)"
```

### 2. Configure Kaggle Credentials (for Dataset D1)
Dataset D1 requires `kagglehub` to fetch the LendingClub dataset. Ensure your `kaggle.json` API token is located at `~/.kaggle/kaggle.json` or configured in your environment variables. Dataset D2 is downloaded automatically from the UCI repository (features, targets and patient IDs).

### 3. Run the tests

```bash
python -m pytest
```

---

## 🚀 Running the Framework

```bash
python main.py      # run the experiments (resumable)
python analyze.py   # metrics, comparisons and figures for the latest run of each domain
```

All experiment settings live in `config.py` (`N_INSTANCES`, `N_ITERATIONS`, `BUDGET_MULTIPLIERS`, `BACKGROUND_SIZE`, `BACKGROUND_SUMMARY`, …). The settings are hashed into a **run id**; results go to `outputs/experiments/<domain>/<run_id>/`, one Parquet file per model × budget plus a high-budget reference per model. Re-running `main.py` with the same settings skips files that already exist; changing any setting starts a new run. Bump `EXPERIMENT_VERSION` after changing approach code so that old results are not resumed.

Console output is also appended to `outputs/experiment_log.txt`. `analyze.py` can be pointed at specific runs: `python analyze.py outputs/experiments/financial/<run_id>`.

> Cost scales with `N_INSTANCES × N_ITERATIONS × budgets × 5 approaches × 5 models`. Lower `N_INSTANCES` or drop the largest budget multiplier for a quick run.

---

## 📊 Evaluation Metrics

Each metric is computed per (model, budget, approach, instance) over the `N_ITERATIONS` repeated explanations, then summarized across instances (mean with a 95% percentile-bootstrap CI, which stays within each metric's range).

| Metric | Definition | Target | Regulatory Context |
| :--- | :--- | :---: | :--- |
| **ACV (Volatility)** | Importance-weighted coefficient of variation: Σⱼ std(φⱼ) / Σⱼ \|mean(φⱼ)\|. Unlike an unweighted mean of per-feature CVs, it is not dominated by features whose attribution is ~0. | **Lower** | Numerical repeatability of explanations under audit (GDPR Art. 22). |
| **Top-k Jaccard Agreement** | Mean pairwise Jaccard index of the top-k \|φ\| sets across runs. | **Higher** | Adverse Action Notices consistently cite the same primary factors. |
| **Reference RMSE (Fidelity)** | RMSE between each run's attributions and the high-budget reference **for the same estimand**: Shapley approaches vs the Shapley reference, Owen approaches vs the Owen reference (see below). | **Lower** | Stabilization must not move the explanation away from what the model does. |
| **E[f] bias** | \|E_bg[f] − reference E[f]\|: how far a background's expected prediction is from the reference. A biased background shifts every attribution sum. | **Lower** | Explanations should be relative to a representative baseline. |
| **Compute cost** | Model evaluations per explanation (coalitions × background rows) and wall-clock seconds. | — | Equal budgets are not equal cost: backgrounds are 50 (Baseline, L1, Owen), 10 (KMeans) and 100 rows (Hybrid Owen). |
| *Efficiency gap* (sanity check) | \|Σφ − (f(x) − E_bg[f])\|, ~0 by construction; non-zero values indicate a bug. | ≈ 0 | — |

### Fidelity references

Owen values and Shapley values are different quantities, so each instance gets two references, computed on the **same fixed 200-row background** so that they differ only in estimand:

* **Shapley reference:** KernelSHAP at shap's default budget (2M + 2048 coalitions), repeated 3 times with different seeds. `reference_noise` (the mean RMSE of the repeats from their mean) is the floor below which differences in reference RMSE are not meaningful.
* **Owen reference:** `PartitionExplainer` with exactly enough evaluations to explore the whole hierarchy (`full_traversal_evals`, about 0.75 × M² for the balanced tree), which makes it exact for that background. A warning is logged if any reference runs out of budget.

The per-instance table also reports error against both references for every approach (`reference_rmse_shapley`, `reference_rmse_owen`) and `reference_estimand_gap`, the distance between the two references themselves, i.e. how much choosing Owen over Shapley changes the explanation regardless of estimation noise.

### Equal-compute comparison

`*_vs_cost.png` plots every metric against model evaluations per explanation (log scale), one line per approach through its budgets. Use it, not the per-budget tables, to compare approaches at equal compute.

Each approach is compared with the baseline on the same instances with a two-sided Wilcoxon signed-rank test, Benjamini–Hochberg corrected within each metric. The number of coalitions actually evaluated is recorded for every run. For the Owen approaches the budget is PartitionExplainer's `max_evals`, which plays the same role as KernelSHAP's `nsamples`.

---

## 🔬 Benchmark Methodology

1. **Step 1 — Ingestion & Missingness Audit:** Writes per-column audit tables. For the clinical data, the UCI value `"None"` (test not performed) in `A1Cresult` / `max_glu_serum` is restored as a category before the >50% missingness filter.
2. **Step 2 — Target Construction:** Resolved credit outcomes (`Charged Off` vs. `Fully Paid`) for Finance; 30-day readmission (`<30`) for Clinical, excluding encounters discharged as expired or to hospice.
3. **Step 3 — Domain Feature Selection:** Strict domain feature sets; identifier and leakage columns removed.
4. **Step 4 — Domain-Specific Encoding:** Ordinal mapping for binned categories, one-hot encoding for nominal ones. Credit history length is computed from `earliest_cr_line` with two-digit years before 1969 corrected to the 1900s.
5. **Step 5 — Leakage-Free Splitting:** Out-of-time split (`SPLIT_DATE = 2014-01-01`) for Finance; stratified split grouped by `patient_nbr` for Clinical, so no patient appears in both train and test.
6. **Step 6 — Imputer/Scaler Fit on Train Only:** Median imputation and standard scaling of continuous variables; PSI drift per feature written to `outputs/tables/`.
7. **Model Training:** All five classifiers are fitted on the training split and cached; caches fitted on a different feature set are retrained. Holdout AUC is logged, and models below `MIN_MODEL_AUC` are flagged.
8. **Instance Selection:** For each model, `N_INSTANCES` test rows at evenly spaced quantiles of its predicted risk.
9. **SHAP Interventions:** Each approach explains each instance `N_ITERATIONS` times at budgets of `BUDGET_MULTIPLIERS × n_features` coalitions. Every approach is reseeded identically per instance, so resampling approaches share background draws (paired design).
10. **References:** For each model and instance, a Shapley and an Owen reference on a shared fixed background (see *Fidelity references*).
11. **Analysis (`analyze.py`):** Per-instance metrics, summaries with CIs, baseline comparisons, metric-vs-budget and metric-vs-compute figures, and attribution boxplots with both references marked.

---

## ⚠️ Limitations

* **The references are tied to their background.** Both are computed on one fixed 200-row background. The Shapley reference is still a sampled estimate (its noise is reported); the Owen reference is exact for that background but not for the full data distribution.
* **Owen values are deterministic for a given background.** PartitionExplainer walks the hierarchy deterministically instead of sampling coalitions, so the Owen approaches vary only through their background: row resampling (Approach 3) or the bootstrapped K-Means summary (Approach 4). With `BOOTSTRAP_SUMMARIES = False`, Approach 4 has ACV = 0 and Jaccard = 1 *by construction*, which reflects determinism, not accuracy.
* **Summary pool cost.** One K-Means summary per repeat is fitted on the full training set (about 2 minutes each at LendingClub size, so roughly an hour per domain for 30 repeats). The pool is cached in the run directory and reused on resume.
* **Owen values depend on the hierarchy.** They are defined relative to the coalition tree, which is built from the training data: features whose average-linkage distance 1 − |ρ| is below `OWEN_GROUP_DISTANCE` (0.5) form a group, and the groups and their members are joined in a balanced binary tree. A different threshold or linkage gives different values. The tree is balanced because `PartitionExplainer` reaches a node at depth d through 2^d context paths: the raw dendrogram chains weakly correlated features 40+ levels deep and would need ~10¹⁴ evaluations to explore, while the balanced tree (depth 7) needs about 0.75 × M². When the budget runs out before the tree is fully explored, the remaining subtrees split their credit across members in proportion to size.
* **Budgets start at 2 × M.** At 1 × M the KernelSHAP regression has about as many coalitions as unknowns; it is singular and attributions can explode (|φ| > 1 on a probability scale), so that budget is not used.
* **Results so far.** Earlier single-instance findings are in [docs/fix_and_rerun_report.md](docs/fix_and_rerun_report.md) and are superseded by the multi-instance design above. Every change since then, and the reason for it, is documented in [docs/changes_report.md](docs/changes_report.md).
