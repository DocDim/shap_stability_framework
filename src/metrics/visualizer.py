# src/metrics/visualizer.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class SHAPVisualizer:
    def __init__(self, output_fig_dir="outputs/figures", output_data_dir="outputs/shap_runs"):
        self.fig_dir = output_fig_dir
        self.data_dir = output_data_dir
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        sns.set_theme(style="whitegrid", font="sans-serif")
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.autolayout": True
        })

    def _align_shape(self, runs: np.ndarray, n_features: int) -> np.ndarray:
        """Guarantees runs array has shape (N_iterations, n_features)."""
        runs = np.array(runs)
        if runs.shape[1] == 2 * n_features:
            return runs[:, n_features:]
        elif runs.shape[1] > n_features:
            return runs[:, :n_features]
        return runs

    def save_shap_runs(self, approaches_dict: dict, feature_names: list, domain_name: str):
        n_features = len(feature_names)
        for name, runs in approaches_dict.items():
            runs_aligned = self._align_shape(runs, n_features)
            sanitized_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "and")
            df_runs = pd.DataFrame(runs_aligned, columns=feature_names)
            df_runs.index.name = "iteration"
            
            out_file = os.path.join(self.data_dir, f"{domain_name}_{sanitized_name}_shap_runs.csv")
            df_runs.to_csv(out_file)
        print(f"[{domain_name}] Raw SHAP iteration arrays saved to: {self.data_dir}/")

    def plot_attribution_variance_boxplot(self, approaches_dict: dict, feature_names: list, domain_name: str, top_n: int = 6):
        n_features = len(feature_names)
        base_runs = self._align_shape(approaches_dict["Baseline"], n_features)
        
        mean_abs_importance = np.mean(np.abs(base_runs), axis=0)
        top_indices = np.argsort(mean_abs_importance)[-top_n:][::-1]
        top_feature_names = [feature_names[i] for i in top_indices]

        plot_records = []
        for app_name, runs in approaches_dict.items():
            runs_aligned = self._align_shape(runs, n_features)
            for i, feat_idx in enumerate(top_indices):
                feat_vals = runs_aligned[:, feat_idx]
                for val in feat_vals:
                    plot_records.append({
                        "Approach": app_name,
                        "Feature": top_feature_names[i],
                        "SHAP Value": val
                    })
        
        df_plot = pd.DataFrame(plot_records)
        
        plt.figure(figsize=(14, 6))
        sns.boxplot(
            data=df_plot,
            x="Feature",
            y="SHAP Value",
            hue="Approach",
            palette="Set2",
            showfliers=False
        )
        plt.title(f"Attribution Volatility Across Iterations — {domain_name.upper()} (Top {top_n} Features)")
        plt.xlabel("Key Decision Features")
        plt.ylabel("SHAP Attribution Score")
        plt.xticks(rotation=25, ha="right")
        plt.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
        
        fig_path = os.path.join(self.fig_dir, f"{domain_name}_attribution_volatility_boxplot.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[{domain_name}] Volatility boxplot saved to: {fig_path}")

    def plot_metric_comparison_bars(self, summary_df: pd.DataFrame, domain_name: str):
        df = summary_df.copy()
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        
        x = np.arange(len(df))
        width = 0.35
        
        rects1 = ax1.bar(x - width/2, df["ACV (Volatility) [Lower = Better]"], width, label="ACV (Volatility)", color="#4C72B0")
        rects2 = ax2.bar(x + width/2, df["Top-5 Jaccard Agreement [Higher = Better]"], width, label="Jaccard Agreement", color="#55A868")
        
        ax1.set_ylabel("ACV Volatility (Lower is Better)", color="#4C72B0", fontweight="bold")
        ax2.set_ylabel("Top-5 Jaccard Agreement (Higher is Better)", color="#55A868", fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(df["Approach"], rotation=20, ha="right")
        
        plt.title(f"Intervention Performance: Volatility vs. Rank Agreement ({domain_name})")
        ax1.grid(False)
        ax2.grid(True, linestyle="--", alpha=0.5)
        
        fig_path = os.path.join(self.fig_dir, f"{domain_name}_benchmark_metrics_barchart.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[{domain_name}] Comparative bar chart saved to: {fig_path}")