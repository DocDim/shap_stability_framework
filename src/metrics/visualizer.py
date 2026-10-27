# src/metrics/visualizer.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config

# metric -> (axis label, figure title name)
METRIC_LABELS = {
    "acv": ("ACV (weighted) — lower is better", "Attribution volatility (ACV)"),
    "jaccard": ("Top-k Jaccard — higher is better", "Top-k rank agreement"),
    "reference_rmse": ("RMSE vs reference — lower is better", "Fidelity (RMSE vs reference)"),
    "base_value_bias": ("|E_bg[f] − reference E[f]| — lower is better", "Background bias"),
}

# One colour and marker per approach in every figure. Colourblind-safe palette plus distinct
# markers so figures still read in greyscale; Baseline is neutral grey.
APPROACH_ORDER = [
    "Baseline",
    "Approach 1 (KMeans BG)",
    "Approach 2 (L1 Lasso)",
    "Approach 3 (Owen Values)",
    "Approach 4 (Hybrid Owen + KMeans)",
]
_PALETTE = sns.color_palette("colorblind")
APPROACH_STYLE = {
    "Baseline": ("0.4", "o"),
    "Approach 1 (KMeans BG)": (_PALETTE[0], "s"),
    "Approach 2 (L1 Lasso)": (_PALETTE[1], "^"),
    "Approach 3 (Owen Values)": (_PALETTE[2], "D"),
    "Approach 4 (Hybrid Owen + KMeans)": (_PALETTE[3], "v"),
}
_FALLBACK_STYLE = ("0.7", "x")

# Multiplicative x-offset between approaches on log axes, so points at equal x stay visible
_DODGE = 1.04


def ordered_approaches(names) -> list:
    """Known approaches in APPROACH_ORDER, then any others alphabetically."""
    names = set(names)
    return [a for a in APPROACH_ORDER if a in names] + sorted(names - set(APPROACH_ORDER))


def approach_style(name):
    return APPROACH_STYLE.get(name, _FALLBACK_STYLE)


class SHAPVisualizer:
    def __init__(self, output_fig_dir="outputs/figures"):
        self.fig_dir = output_fig_dir
        os.makedirs(self.fig_dir, exist_ok=True)

        sns.set_theme(style="whitegrid", font="sans-serif")
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })

    def _save(self, fig, stem: str):
        """PNG for quick viewing, PDF (vector) for the paper."""
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(self.fig_dir, f"{stem}.{ext}"), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved to: {os.path.join(self.fig_dir, stem)}.png/.pdf")

    def _metric_panels(self, summary: pd.DataFrame, metric: str, x_of, xlabel: str, log_base: int,
                       xticks=None, xticklabels=None, title: str = "", stem: str = ""):
        """
        One panel per model: metric mean with its 95% bootstrap CI against x, one line per approach.
        """
        approaches = ordered_approaches(summary["approach"].unique())
        models = sorted(summary["model"].unique())
        n_cols = min(3, len(models))
        n_rows = int(np.ceil(len(models) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.8 * n_rows), squeeze=False)

        for ax, model in zip(axes.flat, models):
            m = summary[summary["model"] == model]
            for i, approach in enumerate(approaches):
                d = m[m["approach"] == approach].sort_values("budget")
                if d.empty:
                    continue
                x = x_of(d) * _DODGE ** (i - (len(approaches) - 1) / 2)
                y = d[f"{metric}_mean"].to_numpy()
                yerr = np.clip(np.vstack([y - d[f"{metric}_ci_low"], d[f"{metric}_ci_high"] - y]), 0, None)
                color, marker = approach_style(approach)
                ax.errorbar(x, y, yerr=yerr, color=color, marker=marker, markersize=5, capsize=3, label=approach)

            if metric == "reference_rmse":
                # Below the Shapley reference's own noise, RMSE differences are not meaningful.
                # The Owen reference is exact, so its floor is 0.
                floor = m.loc[m["approach"] == "Baseline", "reference_noise_mean"].mean()
                if np.isfinite(floor):
                    ax.axhline(floor, color="0.5", linestyle="--", linewidth=1, label="Shapley reference noise")
            if metric == "jaccard":
                ax.set_ylim(0, 1.05)

            auc = m["model_test_auc"].iloc[0]
            low = "  (low AUC)" if auc < config.MIN_MODEL_AUC else ""
            ax.set_title(f"{model} — AUC {auc:.3f}{low}")
            ax.set_xscale("log", base=log_base)
            if xticks is not None:
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
                ax.minorticks_off()
            ax.set_xlabel(xlabel)

        for ax in axes.flat[len(models):]:
            ax.set_visible(False)
        for row in axes:
            row[0].set_ylabel(METRIC_LABELS[metric][0])
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Strategy", loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.suptitle(title)
        fig.tight_layout()
        self._save(fig, stem)

    def plot_metrics_vs_budget(self, summary: pd.DataFrame, domain_tag: str, n_features: int):
        """
        Metric vs evaluation budget, shown as multiples of the feature count so the two domains
        line up. The budget is KernelSHAP's nsamples or PartitionExplainer's max_evals.
        """
        multipliers = sorted((summary["budget"] / n_features).round(6).unique())
        for metric, (_, name) in METRIC_LABELS.items():
            self._metric_panels(
                summary, metric, x_of=lambda d: d["budget"].to_numpy() / n_features,
                xlabel=f"Evaluation budget (× M, M = {n_features} features)", log_base=2,
                xticks=multipliers, xticklabels=[f"{v:g}×" for v in multipliers],
                title=f"{name} vs budget — {domain_tag}", stem=f"{domain_tag}_{metric}_vs_budget",
            )

    def plot_metrics_vs_cost(self, summary: pd.DataFrame, domain_tag: str):
        """
        Metric vs compute cost (model evaluations per explanation, log scale). Approaches use
        different background sizes, so equal budgets are not equal cost; this is the like-for-like view.
        """
        for metric, (_, name) in METRIC_LABELS.items():
            self._metric_panels(
                summary, metric, x_of=lambda d: d["mean_model_evals_mean"].to_numpy(),
                xlabel="Model evaluations per explanation", log_base=10,
                title=f"{name} vs compute — {domain_tag}", stem=f"{domain_tag}_{metric}_vs_cost",
            )

    def plot_attribution_variance_boxplot(self, runs: pd.DataFrame, phi_cols: list, reference_phis: dict,
                                          title: str, file_stem: str, top_n: int = 6):
        """
        Spread of attributions across iterations for the top_n features of the Shapley reference
        (one instance, one model, one budget). Outlying runs are shown: occasional extreme
        explanations are part of the instability being measured. References are marked in black.
        """
        markers = {"shapley": "D", "owen": "X"}
        top_idx = np.argsort(np.abs(reference_phis["shapley"]))[-top_n:][::-1]
        top_cols = [phi_cols[i] for i in top_idx]
        df_plot = runs.melt(id_vars="approach", value_vars=top_cols, var_name="Feature", value_name="Attribution")
        df_plot["Feature"] = df_plot["Feature"].str.removeprefix("phi__")

        order = ordered_approaches(df_plot["approach"].unique())
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=df_plot, x="Feature", y="Attribution", hue="approach", hue_order=order,
                    palette={a: approach_style(a)[0] for a in order}, fliersize=2.5, ax=ax)
        for estimand, ref_phi in reference_phis.items():
            ax.scatter(range(top_n), ref_phi[top_idx], color="black", marker=markers[estimand], s=50,
                       zorder=5, label=f"{estimand.capitalize()} reference")
        ax.set_title(title)
        ax.set_xlabel("Top features of the Shapley reference")
        ax.set_ylabel("Attribution (Δ predicted probability)")
        ax.tick_params(axis="x", labelrotation=25)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        ax.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
        self._save(fig, f"{file_stem}_attribution_boxplot")
