"""
main.py — Runs the KernelSHAP stability benchmark for both domains and writes every
          explanation to outputs/experiments/<domain>/<run_id>/. Analyse with analyze.py.

Re-running resumes: (model, budget) files that already exist for the current run
configuration are skipped. Changing any setting in config.py starts a new run_id.
"""
import hashlib
import json
import logging
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

import config
from data.financial_loader import load_preprocessed_finance_dataset
from data.clinical_loader import load_clinical_dataset
from src.models.trainer import ModelManager
from src.shap_approaches import (
    BaselineKernelSHAP,
    KMeansBackgroundKernelSHAP,
    PenalizedKernelSHAP,
    OwenValuesSHAP,
    HybridOwenKMeansSHAP,
    RepeatedRuns,
    build_feature_hierarchy,
    build_summary_pool,
    full_traversal_evals,
)
from src.metrics.results import PHI_PREFIX, REFERENCE, per_instance_metrics

log = logging.getLogger("experiment")


def setup_logging():
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Append so resumed runs keep the earlier log
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(config.LOG_FILE, mode="a", encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    # Route warnings (e.g. ConvergenceWarning) to the log instead of hiding them
    logging.captureWarnings(True)
    # KernelExplainer passes NumPy arrays to models fitted on DataFrames
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    # Expected when a sampled coalition matrix is rank-deficient (likelier at low budgets) and when
    # LARS drops collinear coalitions. Logged per call, they would flood the log.
    warnings.filterwarnings("ignore", message="Linear regression equation is singular")
    warnings.filterwarnings("ignore", message="Regressors in active set degenerate")
    warnings.filterwarnings("ignore", message="Early stopping the lars path")
    # shap logs internal solver values at INFO, and warns on every reference call that a
    # >100-row background is slow (the reference background is that large on purpose)
    shap_log = logging.getLogger("shap")
    shap_log.setLevel(logging.WARNING)
    shap_log.addFilter(lambda record: "background data samples could cause slower" not in record.getMessage())


def run_config(domain_tag: str) -> dict:
    return {
        "version": config.EXPERIMENT_VERSION,
        "seed": config.SEED,
        "n_iterations": config.N_ITERATIONS,
        "n_instances": config.N_INSTANCES,
        "budget_multipliers": config.BUDGET_MULTIPLIERS,
        "background_size": config.BACKGROUND_SIZE,
        "n_bg_clusters": config.N_BG_CLUSTERS,
        "background_summary": config.BACKGROUND_SUMMARY,
        "bootstrap_summaries": config.BOOTSTRAP_SUMMARIES,
        "owen_kmeans_background_rows": config.OWEN_KMEANS_BACKGROUND_ROWS,
        "reference_background_size": config.REFERENCE_BACKGROUND_SIZE,
        "reference_repeats": config.REFERENCE_REPEATS,
        "owen_group_distance": config.OWEN_GROUP_DISTANCE,
        "domain": domain_tag,
    }


def load_summary_pool(X_train, run_dir):
    """KMeans background summaries for Approaches 1 and 4, cached in the run directory
    because refitting one per iteration on the full training set is slow."""
    path = os.path.join(run_dir, "background_summaries.joblib")
    if os.path.exists(path):
        log.info("Loaded cached background summaries from: %s", path)
        return joblib.load(path)

    n_summaries = config.N_ITERATIONS if config.BOOTSTRAP_SUMMARIES else 1
    log.info("Fitting %d KMeans background summar%s (%s, k=%d)%s...", n_summaries,
             "ies" if n_summaries > 1 else "y", config.BACKGROUND_SUMMARY, config.N_BG_CLUSTERS,
             " on bootstrap samples" if config.BOOTSTRAP_SUMMARIES else "")
    start = time.perf_counter()
    pool = build_summary_pool(X_train, n_summaries, n_clusters=config.N_BG_CLUSTERS, method=config.BACKGROUND_SUMMARY,
                              bootstrap=config.BOOTSTRAP_SUMMARIES, random_state=config.SEED)
    log.info("  done in %.0fs", time.perf_counter() - start)
    joblib.dump(pool, path)
    return pool


def build_approaches(X_train, summary_pool, clustering):
    # The feature hierarchy and KMeans summaries depend only on X_train, so build each approach
    # once per domain and swap in the model being explained.
    return {
        "Baseline": BaselineKernelSHAP(None, X_train, background_size=config.BACKGROUND_SIZE),
        "Approach 1 (KMeans BG)": KMeansBackgroundKernelSHAP(None, X_train, summary_pool=summary_pool),
        "Approach 2 (L1 Lasso)": PenalizedKernelSHAP(None, X_train, background_size=config.BACKGROUND_SIZE),
        "Approach 3 (Owen Values)": OwenValuesSHAP(None, X_train, background_size=config.BACKGROUND_SIZE,
                                                   clustering=clustering),
        "Approach 4 (Hybrid Owen + KMeans)": HybridOwenKMeansSHAP(
            None, X_train, n_background_rows=config.OWEN_KMEANS_BACKGROUND_ROWS, summary_pool=summary_pool,
            clustering=clustering),
    }


def select_instances(model, X_test, n, seed=config.SEED, pool_size=5000):
    """Test rows at evenly spaced quantiles of the model's predicted risk."""
    pool = X_test.sample(n=min(pool_size, len(X_test)), random_state=seed)
    order = np.argsort(model.predict_proba(pool)[:, 1])
    picks = order[np.linspace(0, len(order) - 1, min(n, len(order))).round().astype(int)]
    return list(pool.index[picks])


def to_frame(meta: dict, runs: RepeatedRuns, feature_names, status: str) -> pd.DataFrame:
    info = pd.DataFrame({
        **meta,
        "iteration": np.arange(len(runs.phi)),
        "base_value": runs.base_values,
        "n_coalitions": runs.n_coalitions,
        "n_background": runs.n_background,
        "model_evals": runs.model_evals,
        "seconds": runs.seconds,
        "status": status,
    })
    phi = pd.DataFrame(runs.phi, columns=[PHI_PREFIX + f for f in feature_names])
    return pd.concat([info, phi], axis=1)


def compute_references(model, ref_bg, clustering, X_test, instance_ids, f_x, base_meta, feature_names):
    """
    Fidelity references for each instance, on one fixed background so the two differ only in estimand:
      - Shapley: KernelSHAP at shap's "auto" budget (2M + 2048), repeated with different seeds so
        the reference's own noise can be measured
      - Owen: PartitionExplainer with enough evaluations to explore the whole hierarchy (exact
        for this background, hence deterministic and run once)
    """
    n_features = len(feature_names)
    shapley_budget = 2 * n_features + 2048
    # +3: shap reserves 2 evaluations for the empty and full coalitions and stops strictly below max_evals
    owen_budget = full_traversal_evals(clustering) + 3
    shapley_ref = BaselineKernelSHAP(model, ref_bg, background_size=len(ref_bg))
    owen_ref = OwenValuesSHAP(model, ref_bg, background_size=len(ref_bg), clustering=clustering)

    frames = []
    for iid in instance_ids:
        target = X_test.loc[[iid]]
        meta = {**base_meta, "approach": REFERENCE, "instance_id": iid, "f_x": f_x[iid]}

        shapley_ref.reseed(config.SEED)
        runs = shapley_ref.run_repeated_experiments(target, n_iterations=config.REFERENCE_REPEATS, n_samples=shapley_budget)
        frames.append(to_frame({**meta, "budget": shapley_budget, "estimand": "shapley"}, runs, feature_names, "ok"))

        owen_ref.reseed(config.SEED)
        runs = owen_ref.run_repeated_experiments(target, n_iterations=1, n_samples=owen_budget)
        if runs.n_coalitions[0] >= owen_budget:
            log.warning("    Owen reference for instance %s ran out of budget (%d); it is not exact",
                        iid, owen_budget)
        frames.append(to_frame({**meta, "budget": owen_budget, "estimand": "owen"}, runs, feature_names, "ok"))
    return pd.concat(frames, ignore_index=True)


def write_parquet(df: pd.DataFrame, path: str):
    # Write then rename, so an interrupted run never leaves a partial file that resume would skip
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def run_domain_experiment(domain_name: str, domain_tag: str, X_train, X_test, y_train, y_test,
                          force_retrain: bool = False):
    log.info("#" * 85)
    log.info("   STARTING EXPERIMENTAL PIPELINE FOR: %s", domain_name.upper())
    log.info("#" * 85)

    cfg = run_config(domain_tag)
    run_id = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    run_dir = os.path.join(config.EXPERIMENTS_DIR, domain_tag, run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    log.info("Run %s -> %s", run_id, run_dir)

    # 1. Fit every predefined model (no selection step)
    manager = ModelManager(model_dir=config.MODEL_SAVED_DIR)
    models = manager.train_all_models(X_train, X_test, y_train, y_test, domain_tag, force_retrain=force_retrain)

    feature_names = list(X_train.columns)
    n_features = len(feature_names)
    budgets = [m * n_features for m in config.BUDGET_MULTIPLIERS]
    ref_bg = X_train.sample(n=min(config.REFERENCE_BACKGROUND_SIZE, len(X_train)), random_state=config.SEED)
    log.info("%d features | budgets %s", n_features, budgets)

    # One feature hierarchy shared by both Owen approaches and the Owen reference
    clustering = build_feature_hierarchy(X_train, config.OWEN_GROUP_DISTANCE)
    log.info("Owen hierarchy: depth %d | exact Owen values need %d evaluations",
             int(clustering[-1, 2]), full_traversal_evals(clustering))
    approaches = build_approaches(X_train, load_summary_pool(X_train, run_dir), clustering)

    # 2. SHAP experiments for every model x budget
    for model_name, (model, auc) in models.items():
        slug = model_name.lower().replace(" ", "_")
        instance_ids = select_instances(model, X_test, config.N_INSTANCES)
        f_x = dict(zip(instance_ids, model.predict_proba(X_test.loc[instance_ids])[:, 1]))
        base_meta = {"domain": domain_name, "model": model_name, "model_test_auc": round(auc, 6)}

        ref_path = os.path.join(run_dir, f"{slug}_reference.parquet")
        if not os.path.exists(ref_path):
            log.info("[%s] %s | Shapley and Owen references for %d instances", domain_tag, model_name, len(instance_ids))
            references = compute_references(model, ref_bg, clustering, X_test, instance_ids, f_x, base_meta, feature_names)
            write_parquet(references, ref_path)
        reference = pd.read_parquet(ref_path)

        for budget in budgets:
            out_path = os.path.join(run_dir, f"{slug}_b{budget}.parquet")
            if os.path.exists(out_path):
                log.info("[%s] %s | budget=%d: already done, skipping", domain_tag, model_name, budget)
                continue

            log.info("[%s] %s | budget=%d | %d instances x %d iterations",
                     domain_tag, model_name, budget, len(instance_ids), config.N_ITERATIONS)
            frames = []
            for app_name, approach in approaches.items():
                approach.model = model
                for iid in instance_ids:
                    # Same seed per approach and instance: results don't depend on run order,
                    # and resampling approaches share background draws (paired comparison)
                    approach.reseed(config.SEED)
                    try:
                        runs = approach.run_repeated_experiments(
                            X_test.loc[[iid]], n_iterations=config.N_ITERATIONS, n_samples=budget)
                        status = "ok"
                    except Exception as e:
                        nan = np.full(config.N_ITERATIONS, np.nan)
                        runs = RepeatedRuns(np.full((config.N_ITERATIONS, n_features), np.nan), nan, nan, nan, nan)
                        status = f"failed: {type(e).__name__}"
                        log.error("    %s | instance %s FAILED (%s: %s)", app_name, iid, type(e).__name__, str(e)[:90])
                    meta = {**base_meta, "budget": budget, "approach": app_name, "estimand": approach.estimand,
                            "instance_id": iid, "f_x": f_x[iid]}
                    frames.append(to_frame(meta, runs, feature_names, status))

            results = pd.concat(frames, ignore_index=True)
            write_parquet(results, out_path)

            metrics = per_instance_metrics(results, reference)
            for app_name, m in metrics.groupby("approach", sort=False):
                if m["degenerate"].any():
                    log.warning("    '%s' returned all-zero attributions for %d instances", app_name, int(m["degenerate"].sum()))
                if m["efficiency_gap"].max() > 1e-6:
                    log.warning("    '%s' attributions do not sum to f(x) - E_bg[f] (max gap %.2e)",
                                app_name, m["efficiency_gap"].max())
                log.info("    %-34s ACV=%7.3f  Jaccard@%d=%.3f  RefRMSE=%.4f  E[f]bias=%.4f  evals=%.0f  %.2fs/expl",
                         app_name, m["acv"].median(), config.TOP_K, m["jaccard"].mean(),
                         m["reference_rmse"].mean(), m["base_value_bias"].mean(),
                         m["mean_model_evals"].mean(), m["mean_seconds"].mean())

    log.info("[%s] All results written to: %s", domain_tag, run_dir)


def main():
    setup_logging()

    # Financial Domain D1
    X_train_fin, X_test_fin, y_train_fin, y_test_fin, _ = load_preprocessed_finance_dataset(config.FINANCIAL_RAW_DATA_FILE)
    run_domain_experiment("Financial (Loan Default)", "financial", X_train_fin, X_test_fin, y_train_fin, y_test_fin)

    # Clinical Domain D2
    X_train_cli, X_test_cli, y_train_cli, y_test_cli, _ = load_clinical_dataset()
    run_domain_experiment("Clinical (Readmission Risk)", "clinical", X_train_cli, X_test_cli, y_train_cli, y_test_cli)


if __name__ == "__main__":
    main()
