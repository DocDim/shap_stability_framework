# src/metrics/comparative_analysis.py
import pandas as pd
import numpy as np

class ComparativeAnalyzer:
    @staticmethod
    def generate_comparative_report(results_df: pd.DataFrame, domain_label: str = "Domain") -> dict:
        """
        Computes relative improvements vs. Baseline and pairwise cross-approach comparisons.
        """
        df = results_df.copy()
        
        # Identify baseline metrics
        baseline_row = df[df["Approach"] == "Baseline"].iloc[0]
        base_acv = baseline_row["ACV (Volatility) [Lower = Better]"]
        base_jaccard = baseline_row["Top-5 Jaccard Agreement [Higher = Better]"]
        
        # 1. Percentage improvements relative to Baseline
        df["ACV Reduction vs Baseline (%)"] = df["ACV (Volatility) [Lower = Better]"].apply(
            lambda x: round(((base_acv - x) / base_acv) * 100, 2) if base_acv > 0 else 0.0
        )
        
        df["Jaccard Gain vs Baseline (%)"] = df["Top-5 Jaccard Agreement [Higher = Better]"].apply(
            lambda x: round(((x - base_jaccard) / (1.0 - base_jaccard + 1e-6)) * 100, 2) if base_jaccard < 1.0 else 0.0
        )
        
        # 2. Pairwise Difference Matrices
        approaches = df["Approach"].tolist()
        acv_matrix = pd.DataFrame(index=approaches, columns=approaches)
        jaccard_matrix = pd.DataFrame(index=approaches, columns=approaches)
        
        for app1 in approaches:
            acv1 = df[df["Approach"] == app1]["ACV (Volatility) [Lower = Better]"].values[0]
            jac1 = df[df["Approach"] == app1]["Top-5 Jaccard Agreement [Higher = Better]"].values[0]
            
            for app2 in approaches:
                acv2 = df[df["Approach"] == app2]["ACV (Volatility) [Lower = Better]"].values[0]
                jac2 = df[df["Approach"] == app2]["Top-5 Jaccard Agreement [Higher = Better]"].values[0]
                
                # Positive ACV diff = app1 is less volatile than app2
                acv_matrix.loc[app1, app2] = round(acv2 - acv1, 4)
                # Positive Jaccard diff = app1 has higher rank agreement than app2
                jaccard_matrix.loc[app1, app2] = round(jac1 - jac2, 4)

        return {
            "domain": domain_label,
            "summary_table": df,
            "acv_pairwise_diff": acv_matrix,
            "jaccard_pairwise_diff": jaccard_matrix
        }

    @staticmethod
    def print_formatted_report(analysis_dict: dict):
        domain = analysis_dict["domain"]
        summary_df = analysis_dict["summary_table"]
        acv_matrix = analysis_dict["acv_pairwise_diff"]
        
        print("\n" + "="*85)
        print(f"   PART 3A: BENCHMARK VS BASELINE COMPARATIVE ANALYSIS [{domain.upper()}]")
        print("="*85)
        print(summary_df.to_string(index=False))
        
        print("\n" + "="*85)
        print(f"   PART 3B: PAIRWISE ACV VOLATILITY DIFFERENCE MATRIX [{domain.upper()}]")
        print("   (Row vs Column: Positive Value = Row Approach is More Stable than Column)")
        print("="*85)
        print(acv_matrix.to_string())
        
        # Executive Summary Highlights
        non_baseline_df = summary_df[summary_df["Approach"] != "Baseline"]
        best_acv_row = non_baseline_df.sort_values("ACV (Volatility) [Lower = Better]").iloc[0]
        best_jaccard_row = non_baseline_df.sort_values("Top-5 Jaccard Agreement [Higher = Better]", ascending=False).iloc[0]
        
        print("\n" + "-"*85)
        print(f"EXECUTIVE SUMMARY HIGHLIGHTS ({domain})")
        print("-"*85)
        print(f"• Top Volatility Stabilizer: '{best_acv_row['Approach']}'")
        print(f"  └─ Achieved {best_acv_row['ACV Reduction vs Baseline (%)']}% ACV volatility reduction vs. Baseline")
        print(f"• Top Ranking Consistency:   '{best_jaccard_row['Approach']}'")
        print(f"  └─ Achieved {best_jaccard_row['Top-5 Jaccard Agreement [Higher = Better]']} Top-5 Jaccard Agreement ({best_jaccard_row['Jaccard Gain vs Baseline (%)']}% gain vs. Baseline)")
        print("="*85 + "\n")