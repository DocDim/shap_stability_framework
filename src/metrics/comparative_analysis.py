# src/metrics/comparative_analysis.py
import numpy as np
import pandas as pd
from scipy import stats

CELL = ["domain", "model", "budget"]

# metric -> True when lower is better. reference_rmse is against each approach's own estimand
# (Owen approaches vs the Owen reference, the rest vs the Shapley reference).
METRICS = {
    "acv": True,
    "jaccard": False,
    "reference_rmse": True,
    "base_value_bias": True,
}
# Summarized but not compared:
#   efficiency_gap            ~0 for every approach unless something is broken
#   mean_feature_cv           the original unweighted ACV, kept for continuity with earlier results
#   reference_rmse_shapley/_owen  error against each reference regardless of estimand
#   reference_noise           noise floor of the matched reference
#   reference_estimand_gap    how far the Shapley and Owen references are from each other
#   mean_model_evals, mean_seconds  compute cost per explanation
SUMMARY_ONLY = ["efficiency_gap", "mean_feature_cv", "reference_rmse_shapley", "reference_rmse_owen",
                "reference_noise", "reference_estimand_gap", "mean_model_evals", "mean_seconds"]

N_BOOTSTRAP = 2000


def bootstrap_ci(values, level=0.95, n_boot=N_BOOTSTRAP, seed=0):
    """
    Percentile bootstrap interval for the mean. Unlike a t-interval it stays inside the
    metric's range (no negative RMSE, no Jaccard above 1) and does not assume symmetry,
    which matters for skewed, bounded metrics over ~20 instances.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan, np.nan
    means = np.random.default_rng(seed).choice(values, size=(n_boot, len(values))).mean(axis=1)
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


class ComparativeAnalyzer:
    @staticmethod
    def summarize(per_instance: pd.DataFrame) -> pd.DataFrame:
        """Mean, standard deviation and 95% percentile-bootstrap interval of each metric across instances."""
        rows = []
        for keys, grp in per_instance.groupby(CELL + ["approach"]):
            row = dict(zip(CELL + ["approach"], keys))
            row["model_test_auc"] = grp["model_test_auc"].iloc[0]
            row["n_instances"] = grp["instance_id"].nunique()
            row["mean_n_coalitions"] = grp["mean_n_coalitions"].mean()
            for metric in [*METRICS, *SUMMARY_ONLY]:
                vals = grp[metric].dropna()
                row[f"{metric}_mean"] = vals.mean() if len(vals) else np.nan
                row[f"{metric}_sd"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = bootstrap_ci(vals)
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def compare_to_baseline(per_instance: pd.DataFrame, baseline="Baseline") -> pd.DataFrame:
        """
        Paired comparison of every approach against the baseline on the same instances:
        relative change of the mean and a two-sided Wilcoxon signed-rank test, with
        Benjamini-Hochberg correction within each metric.
        """
        rows = []
        for keys, cell in per_instance.groupby(CELL):
            base = cell[cell["approach"] == baseline].set_index("instance_id")
            for approach, grp in cell[cell["approach"] != baseline].groupby("approach"):
                other = grp.set_index("instance_id")
                for metric, lower_is_better in METRICS.items():
                    paired = pd.concat([base[metric], other[metric]], axis=1, keys=["base", "other"]).dropna()
                    if paired.empty:
                        continue
                    diff = paired["other"] - paired["base"]
                    base_mean = paired["base"].mean()
                    p_value = (
                        stats.wilcoxon(paired["other"], paired["base"]).pvalue
                        if len(paired) > 1 and np.any(diff != 0) else np.nan
                    )
                    rows.append({
                        **dict(zip(CELL, keys)),
                        "approach": approach,
                        "metric": metric,
                        "n_pairs": len(paired),
                        "baseline_mean": base_mean,
                        "approach_mean": paired["other"].mean(),
                        "change_pct": 100 * diff.mean() / abs(base_mean) if base_mean else np.nan,
                        "improved": bool(diff.mean() < 0) if lower_is_better else bool(diff.mean() > 0),
                        "p_value": p_value,
                    })

        result = pd.DataFrame(rows)
        if result.empty:
            return result
        result["p_adjusted"] = np.nan
        for metric, idx in result.groupby("metric").groups.items():
            pvals = result.loc[idx, "p_value"]
            valid = pvals.notna()
            if valid.any():
                result.loc[pvals[valid].index, "p_adjusted"] = stats.false_discovery_control(pvals[valid])
        result["significant"] = result["p_adjusted"] < 0.05
        return result

    @staticmethod
    def print_formatted_report(summary: pd.DataFrame, comparison: pd.DataFrame, domain: str):
        dom_summary = summary[summary["domain"] == domain]
        dom_comp = comparison[comparison["domain"] == domain] if not comparison.empty else comparison

        print("\n" + "="*85)
        print(f"   STABILITY SUMMARY [{domain.upper()}]  (mean [95% bootstrap CI] across instances)")
        print("="*85)
        for (model, budget), grp in dom_summary.groupby(["model", "budget"]):
            print(f"\n{model} (AUC {grp['model_test_auc'].iloc[0]:.3f}) | budget = {budget}")
            for _, r in grp.iterrows():
                print(f"  {r['approach']:<34} "
                      f"ACV={r['acv_mean']:7.3f} [{r['acv_ci_low']:.3f}, {r['acv_ci_high']:.3f}]  "
                      f"Jaccard@k={r['jaccard_mean']:.3f}  "
                      f"RefRMSE={r['reference_rmse_mean']:.4f} (floor {r['reference_noise_mean']:.4f})  "
                      f"E[f]bias={r['base_value_bias_mean']:.4f}  "
                      f"EffGap={r['efficiency_gap_mean']:.1e}  "
                      f"evals={r['mean_model_evals_mean']:.0f}  {r['mean_seconds_mean']:.2f}s")

        if dom_comp.empty:
            return
        sig = dom_comp[dom_comp["significant"]]
        print("\n" + "-"*85)
        print(f"SIGNIFICANT CHANGES VS BASELINE ({domain}, BH-adjusted p < 0.05)")
        print("-"*85)
        if sig.empty:
            print("  none")
        for _, r in sig.sort_values(["metric", "model", "budget"]).iterrows():
            direction = "better" if r["improved"] else "worse"
            print(f"  {r['metric']:<15} {r['model']:<20} b={r['budget']:<5} {r['approach']:<34} "
                  f"{r['change_pct']:+7.1f}% ({direction}, p_adj={r['p_adjusted']:.3g})")
        print("="*85 + "\n")
