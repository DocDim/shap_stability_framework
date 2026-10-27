# src/metrics/results.py
import glob
import os

import pandas as pd

import config
from src.metrics.stability_metrics import StabilityEvaluator

PHI_PREFIX = "phi__"
REFERENCE = "Reference"
CELL_KEYS = ["domain", "model", "model_test_auc", "budget", "approach", "instance_id"]


def phi_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith(PHI_PREFIX)]


def latest_run_dir(domain_tag: str):
    """Most recently modified run directory for a domain, or None."""
    runs = glob.glob(os.path.join(config.EXPERIMENTS_DIR, domain_tag, "*", "run_config.json"))
    return os.path.dirname(max(runs, key=os.path.getmtime)) if runs else None


def load_run(run_dir: str):
    """Returns (explanation runs, reference explanations)."""
    # "_b<digits>": a bare "_b*" would also match e.g. gradient_boosting_reference.parquet
    runs = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(run_dir, "*_b[0-9]*.parquet")))],
                     ignore_index=True)
    reference = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(run_dir, "*_reference.parquet")))],
                          ignore_index=True)
    return runs, reference


def reference_table(reference: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (model, instance, estimand): the reference attribution (mean over repeats), its
    base value, and its own noise (mean RMSE of each repeat from that mean; 0 for one repeat).
    """
    ev = StabilityEvaluator
    phi_cols = phi_columns(reference)
    rows = []
    for (model, iid, estimand), grp in reference.groupby(["model", "instance_id", "estimand"]):
        phi = grp[phi_cols].to_numpy()
        ref_phi = phi.mean(axis=0)
        rows.append({
            "model": model, "instance_id": iid, "estimand": estimand,
            "base_value": grp["base_value"].mean(),
            "noise": ev.calculate_reference_error(phi, ref_phi),
            "phi": ref_phi,
        })
    return pd.DataFrame(rows).set_index(["model", "instance_id", "estimand"])


def per_instance_metrics(runs: pd.DataFrame, reference: pd.DataFrame, k: int = config.TOP_K) -> pd.DataFrame:
    """One row per (domain, model, budget, approach, instance) with every stability / fidelity / cost metric."""
    ev = StabilityEvaluator
    phi_cols = phi_columns(runs)
    refs = reference_table(reference)

    rows = []
    for keys, grp in runs[runs["status"] == "ok"].groupby(CELL_KEYS):
        row = dict(zip(CELL_KEYS, keys))
        phi = grp[phi_cols].to_numpy()
        model, iid = row["model"], row["instance_id"]
        shapley_ref, owen_ref = refs.loc[(model, iid, "shapley")], refs.loc[(model, iid, "owen")]
        # Score each approach against the reference for the quantity it estimates
        matched = owen_ref if grp["estimand"].iloc[0] == "owen" else shapley_ref

        row.update(
            estimand=grp["estimand"].iloc[0],
            acv=ev.calculate_acv(phi),
            mean_feature_cv=ev.calculate_mean_feature_cv(phi),
            jaccard=ev.calculate_jaccard_rank_agreement(phi, k=k),
            # Sanity check: ~0 by construction for KernelSHAP and Owen values; non-zero signals a bug
            efficiency_gap=ev.calculate_efficiency_gap(phi, grp["f_x"].iloc[0], grp["base_value"]),
            # How far the background's E[f] is from the reference background's: a biased summary
            # shifts every attribution sum even when efficiency holds
            base_value_bias=abs(grp["base_value"].mean() - shapley_ref["base_value"]),
            reference_rmse=ev.calculate_reference_error(phi, matched["phi"]),
            reference_rmse_shapley=ev.calculate_reference_error(phi, shapley_ref["phi"]),
            reference_rmse_owen=ev.calculate_reference_error(phi, owen_ref["phi"]),
            # Floor below which reference_rmse differences are not meaningful
            reference_noise=matched["noise"],
            # How much the two estimands differ for this instance, independent of any approach
            reference_estimand_gap=ev.calculate_reference_error(shapley_ref["phi"][None, :], owen_ref["phi"]),
            mean_n_coalitions=grp["n_coalitions"].mean(),
            mean_model_evals=grp["model_evals"].mean(),
            mean_seconds=grp["seconds"].mean(),
            degenerate=ev.is_degenerate(phi),
        )
        rows.append(row)
    return pd.DataFrame(rows)
