"""
analyze.py — Computes stability / fidelity metrics from the explanations written by main.py,
             compares every approach to the baseline, and draws the figures.

Usage:
    python analyze.py                 # latest run of each domain
    python analyze.py <run_dir> ...   # specific run directories
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")  # figures are only written to disk; set here, not in the plotting module

import config
from src.metrics.comparative_analysis import ComparativeAnalyzer
from src.metrics.results import latest_run_dir, load_run, per_instance_metrics, phi_columns, reference_table
from src.metrics.visualizer import SHAPVisualizer


def analyze_run(run_dir: str, analyzer: ComparativeAnalyzer, visualizer: SHAPVisualizer):
    domain_tag = os.path.basename(os.path.dirname(os.path.normpath(run_dir)))
    run_id = os.path.basename(os.path.normpath(run_dir))
    runs, reference = load_run(run_dir)

    failed = runs[runs["status"] != "ok"]
    if not failed.empty:
        print(f"[{domain_tag}] WARNING: {failed.groupby(['model', 'approach', 'instance_id']).ngroups} "
              f"failed explanations excluded:\n{failed['status'].value_counts().to_string()}")

    per_instance = per_instance_metrics(runs, reference)
    per_instance["low_auc"] = per_instance["model_test_auc"] < config.MIN_MODEL_AUC
    summary = analyzer.summarize(per_instance)
    comparison = analyzer.compare_to_baseline(per_instance)

    os.makedirs(config.OUTPUTS_TABLES_DIR, exist_ok=True)
    stem = os.path.join(config.OUTPUTS_TABLES_DIR, f"stability_{domain_tag}_{run_id}")
    per_instance.to_csv(f"{stem}_per_instance.csv", index=False)
    summary.to_csv(f"{stem}_summary.csv", index=False)
    comparison.to_csv(f"{stem}_vs_baseline.csv", index=False)
    print(f"[{domain_tag}] Tables written to: {stem}_*.csv")

    for domain in summary["domain"].unique():
        analyzer.print_formatted_report(summary, comparison, domain)
    low = summary.loc[summary["model_test_auc"] < config.MIN_MODEL_AUC, "model"].unique()
    if len(low):
        print(f"[{domain_tag}] Models with AUC < {config.MIN_MODEL_AUC} (interpret with care): {', '.join(low)}")

    phi_cols = phi_columns(runs)
    visualizer.plot_metrics_vs_budget(summary, domain_tag, n_features=len(phi_cols))
    visualizer.plot_metrics_vs_cost(summary, domain_tag)

    # Attribution spread for the median-risk instance at the largest budget, per model
    refs = reference_table(reference)
    max_budget = runs["budget"].max()
    for model, model_runs in runs[(runs["budget"] == max_budget) & (runs["status"] == "ok")].groupby("model"):
        risks = model_runs.groupby("instance_id")["f_x"].first().sort_values()
        iid = risks.index[len(risks) // 2]
        ref_phis = {estimand: refs.loc[(model, iid, estimand), "phi"] for estimand in ("shapley", "owen")}
        slug = model.lower().replace(" ", "_")
        visualizer.plot_attribution_variance_boxplot(
            model_runs[model_runs["instance_id"] == iid], phi_cols, ref_phis,
            title=f"{domain_tag} — {model}, instance {iid} (f(x)={risks[iid]:.3f}), budget={max_budget}",
            file_stem=f"{domain_tag}_{slug}",
        )


def main(argv):
    run_dirs = argv or [d for d in (latest_run_dir(t) for t in config.DOMAINS) if d]
    if not run_dirs:
        print(f"No runs found under {config.EXPERIMENTS_DIR}. Run main.py first.")
        return
    analyzer = ComparativeAnalyzer()
    visualizer = SHAPVisualizer(output_fig_dir=config.FIGURES_DIR)
    for run_dir in run_dirs:
        analyze_run(run_dir, analyzer, visualizer)


if __name__ == "__main__":
    main(sys.argv[1:])
