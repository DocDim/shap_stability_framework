# Fix & Rerun Report — KernelSHAP Stability Framework

**Date:** 2026-10-09 · **Branch:** `dev` (changes uncommitted)

## Summary

The Owen-value approaches (3 and 4) had a bug that made every attribution zero. That made their ACV exactly 0 and their Jaccard exactly 1, so the earlier "100% ACV reduction" result was a side effect of the bug. With the bug fixed and the baseline made comparable to the other approaches:

- **Owen-value approaches are *less* stable than the baseline** in both domains.
- **KMeans-background approaches look stable but are inaccurate.** Their attributions sum to about 6% (finance) or about 2× (clinical) of the expected total, f(x) − E[f(X)].
- **Approach 2 (L1/Lasso) is the only method that is both more stable than the baseline and accurate**, but most of the volatility remains.

---

## Changes made

| # | Change | Files |
|---|---|---|
| 1 | **Owen indexing bug fixed.** `X_eval[rows, :][:, cols] = v` wrote to a temporary copy, so the model saw unchanged inputs and every attribution was 0. Replaced with `X_eval[np.ix_(rows, cols)] = v`. | [owen_values.py](../src/shap_approaches/owen_values.py), [hybrid_owen_kmeans.py](../src/shap_approaches/hybrid_owen_kmeans.py) |
| 2 | **Fair baseline.** `shap.sample` defaults to `random_state=0`, so the baseline used the same 50 background rows in every iteration while Approaches 2 and 3 resampled theirs. Baseline, L1 and Owen now share `_sample_background()`, which draws a fresh 50 rows each iteration from one seeded generator. | [base.py](../src/shap_approaches/base.py), [baseline.py](../src/shap_approaches/baseline.py), [penalized_kernel.py](../src/shap_approaches/penalized_kernel.py) |
| 3 | **Seeding.** `np.random.seed(42)` for KernelSHAP's coalition sampling, plus `random_state=42` passed to the approaches that resample backgrounds. | [main.py](../main.py) |
| 4 | **Sanity checks.** New `calculate_attribution_sum` (efficiency check) and `is_degenerate` (all-zero check). The results table now has a `Mean Sum(phi)` column, the run prints the f(x) − E[f(X)] reference, and a warning is printed for any all-zero approach. | [stability_metrics.py](../src/metrics/stability_metrics.py), [main.py](../main.py) |
| 5 | **Weighted hybrid centroids.** Hybrid KMeans centroids are now weighted by cluster size, as `shap.kmeans` does. Before, all centroids counted equally. | [hybrid_owen_kmeans.py](../src/shap_approaches/hybrid_owen_kmeans.py) |
| 6 | **Approach 2 relabeled.** `l1_reg="auto"` is L1 (Lasso/AIC) feature selection, not an L2 ridge penalty. The label is now "Approach 2 (L1 Lasso)". | [main.py](../main.py), [main_clinical.py](../main_clinical.py), [main_financial.py](../main_financial.py), [README.md](../README.md) |
| 7 | **Broken export fixed.** `data/__init__.py` exported `load_preprocessed_clinical_dataset`, which doesn't exist; it now exports `load_clinical_dataset`. | [data/\_\_init\_\_.py](../data/__init__.py) |

Before the full run, a synthetic test (logistic regression on 6 features, two of them correlated) confirmed that Owen and Hybrid now return non-zero attributions that sum to about f(x) − E[f(X)].

---

## Results

Setup: one test instance (`X_test.iloc[0]`), 30 iterations, `nsamples=200`, background of 50 rows (10 centroids for the KMeans methods), Top-5 Jaccard.

| Approach | Finance ACV ↓ | Finance Jaccard ↑ | Finance Σφ | Clinical ACV ↓ | Clinical Jaccard ↑ | Clinical Σφ |
|---|---|---|---|---|---|---|
| *Reference f(x) − E[f(X)]* | | | **−0.0565** | | | **0.0133** |
| Baseline | 7.31 | 0.68 | −0.058 | 3.20 | 0.43 | 0.014 |
| 1. KMeans BG | 0.23 | 1.00 | **−0.004** | 0.31 | 1.00 | **0.030** |
| 2. L1 Lasso | 4.40 | 0.71 | −0.058 | 2.50 | 0.44 | 0.014 |
| 3. Owen | **9.56** | 0.66 | −0.058 | **5.40** | 0.43 | 0.014 |
| 4. Hybrid Owen + KMeans | **11.24** | 1.00 | **−0.003** | **5.74** | 1.00 | **0.030** |

Change in ACV relative to the baseline (negative means less volatile):

| Approach | Finance | Clinical |
|---|---|---|
| 1. KMeans BG | −96.8% | −90.4% |
| 2. L1 Lasso | −39.8% | −21.8% |
| 3. Owen | **+30.8%** | **+68.7%** |
| 4. Hybrid | **+53.7%** | **+79.3%** |

### Interpretation

- **Owen values (as implemented) don't stabilize anything.** Both Owen variants are more volatile than the baseline in both domains.
- **The KMeans approaches are stable but biased.** Their Σφ misses the reference by a wide margin. A likely cause is that the centroids average one-hot columns into fractional values, so the model is evaluated on points unlike any real record. Their perfect Jaccard scores come from an almost fixed 10-point background, not from accurate explanations. The Wasserstein-vs-baseline metric did not catch this.
- **L1 Lasso** is the only intervention that is both more stable than the baseline and accurate, but the gain is modest.

---

## Run notes

- **Python environment:** the `TM` conda env (shap 0.49.1, scikit-learn 1.7.2, numpy 2.3.5). `shap_env` has no packages installed.
- **Stale cached models.** `best_model_finance.joblib` expects 28 features and `best_model_clinical.joblib` expects 20, while the current pipelines produce 82 and 72. Plain `python main.py` crashes for this reason.
  - **Finance** used `best_model.joblib` (Gradient Boosting, 82 features, from the August 20 run).
  - **Clinical** was retrained. Gradient Boosting won with CV AUC 0.666 (TimeSeriesSplit) and holdout AUC 0.682, and the new model overwrote `best_model_clinical.joblib`.
  - Both domains were run from scratch scripts, so `main.py` still points at the stale finance model file.
- **First clinical attempt timed out.** It hit a 30-minute limit during the MLP cross-validation step. The rerun completed.
- **Logs:** `outputs/experiment_log_financial.txt` and `outputs/experiment_log_clinical.txt`.
- **Extra diff noise:** committed `__pycache__/*.pyc` files show as modified, and some lines in `main.py` lost trailing whitespace.

---

## Limitations and next steps

1. **One instance only:** the results aren't publishable yet. Explain 50–100 test instances, ideally stratified by predicted risk, and report mean ± CI with significance tests.
2. **Fidelity reference:** replace Wasserstein-vs-baseline with comparison to a high-budget reference, e.g. KernelSHAP with large `nsamples`, or interventional TreeSHAP, since both winning models are Gradient Boosting.
3. **Matched budgets:** report stability as a function of `nsamples`, because grouped methods search a much smaller coalition space.
4. **Naming:** the "Owen values" are really group Shapley values with each group's credit split evenly. Either rename them or implement Shapley values within each group.
5. **KMeans bias:** test a background built from cluster medoids (real records) instead of centroids.
6. **Clinical data:** split by `patient_nbr` to prevent patient leakage, use `StratifiedKFold` instead of `TimeSeriesSplit`, and consider excluding deceased/hospice discharges.
7. **Housekeeping:** point `main.py` at a valid finance model or retrain it; fix `.gitignore` (`outputs/`, `src/models/saved/`, `__pycache__/`); bring the README directory tree up to date; add unit tests for the metrics and the attribution-sum check.
