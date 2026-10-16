from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .config import DATASETS, DEFAULTS


METRICS = [
    "roc_auc", "pr_auc", "balanced_accuracy", "macro_f1",
    "dp_mean", "dp_max", "eo_mean", "eo_max",
    "aod_mean", "aod_max", "worst_group_ba",
    "dp_mean_s5", "dp_max_s5", "eo_mean_s5", "eo_max_s5",
    "aod_mean_s5", "aod_max_s5", "worst_group_ba_s5",
    "dp_mean_s10", "dp_max_s10", "eo_mean_s10", "eo_max_s10",
    "aod_mean_s10", "aod_max_s10", "worst_group_ba_s10",
    "dp_mean_s20", "dp_max_s20", "eo_mean_s20", "eo_max_s20",
    "aod_mean_s20", "aod_max_s20", "worst_group_ba_s20",
    "detection_distance", "numeric_ks_mean",
    "categorical_tv_mean", "correlation_distance",
]


def _holm(pvals):
    """Holm step-down adjusted p-values in original order."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid) == 0:
        return out
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p[idx])
        running = max(running, adj)
        out[idx] = running
    return out


def _bootstrap_mean_ci(x, seed=20260922, B=10000):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.empty(B, dtype=float)
    for b in range(B):
        vals[b] = np.mean(rng.choice(x, size=len(x), replace=True))
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def _collect(runs: Path, filename: str):
    out = []
    for d in sorted(runs.iterdir()) if runs.exists() else []:
        p = d / filename
        if p.exists():
            try:
                z = pd.read_csv(p)
                if len(z):
                    out.append(z)
            except Exception:
                pass
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def aggregate(root: Path):
    base = root / "results" / "journal2026_v21"
    runs = base / "runs"
    out = base / "aggregate"
    out.mkdir(parents=True, exist_ok=True)

    statuses = []
    if runs.exists():
        for d in sorted(runs.iterdir()):
            if not d.is_dir():
                continue
            sp = d / "status.json"
            if sp.exists():
                try:
                    statuses.append(json.loads(sp.read_text()))
                except Exception:
                    pass
    pd.DataFrame(statuses).to_csv(out / "job_status.csv", index=False)

    m = _collect(runs, "metrics.csv")
    r = _collect(runs, "runtime.csv")
    b = _collect(runs, "support_buckets.csv")

    # Smoke tests are engineering checks, never scientific observations.
    if len(m) and "job_id" in m.columns:
        m = m[~m["job_id"].astype(str).str.startswith("smoke_")].copy()
    if len(r) and "job_id" in r.columns:
        r = r[~r["job_id"].astype(str).str.startswith("smoke_")].copy()
    if len(b) and "job_id" in b.columns:
        b = b[~b["job_id"].astype(str).str.startswith("smoke_")].copy()

    # Backward-compatible default for any old pilot row, although v2.1 jobs
    # write eval_split explicitly.
    if len(m) and "eval_split" not in m.columns:
        m["eval_split"] = "test"
    if len(b) and "eval_split" not in b.columns:
        b["eval_split"] = "test"

    if len(r):
        r.to_csv(out / "all_runtime.csv", index=False)
        rt = (
            r.groupby(["family", "dataset", "method", "runtime_n", "train_rows"], dropna=False)
            [["fit_seconds", "generation_seconds", "samples_per_second", "job_seconds"]]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        rt.columns = [
            "__".join(str(x) for x in c if str(x) != "") if isinstance(c, tuple) else c
            for c in rt.columns
        ]
        rt.to_csv(out / "runtime_summary.csv", index=False)

    if len(b):
        b.to_csv(out / "all_support_buckets.csv", index=False)
        bs = (
            b.groupby(
                ["family", "dataset", "method", "ablation", "classifier", "k", "support_bucket"],
                dropna=False,
            )[["n_groups", "eo_max_mean_within_bucket", "aod_mean_within_bucket"]]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        bs.columns = [
            "__".join(str(x) for x in c if str(x) != "") if isinstance(c, tuple) else c
            for c in bs.columns
        ]
        bs.to_csv(out / "support_bucket_summary.csv", index=False)

    if not len(m):
        print("No metrics found.")
        return

    m.to_csv(out / "all_metrics.csv", index=False)
    keep = [c for c in METRICS if c in m.columns]

    grp = ["eval_split", "family", "dataset", "method", "k", "ablation", "classifier"]
    summary = (
        m.groupby(grp, dropna=False)[keep]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "__".join([str(x) for x in col if str(x) != ""])
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary.to_csv(out / "summary_by_dataset.csv", index=False)

    # Core k-scaling.
    core = m[(m["eval_split"] == "test") & m["family"].astype(str).str.contains("core", regex=False)].copy()
    core_methods = core[core.method.isin(["gdt", "singlefair", "multifair"])]
    ks = (
        core_methods.groupby(["dataset", "method", "k", "classifier"], dropna=False)[keep]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    ks.columns = [
        "__".join(str(x) for x in c if str(x) != "") if isinstance(c, tuple) else c
        for c in ks.columns
    ]
    ks.to_csv(out / "k_scaling_summary.csv", index=False)

    # Main maximal-intersection table: all generators that have a row at kmax.
    max_rows = []
    for ds, spec in DATASETS.items():
        kmax = len(spec["groups"])
        z = m[(m["eval_split"] == "test") & (m.dataset == ds) & (m.k.astype(int) == kmax)].copy()
        max_rows.append(z)
    maxm = pd.concat(max_rows, ignore_index=True) if max_rows else pd.DataFrame()
    if len(maxm):
        main = (
            maxm.groupby(["dataset", "method", "classifier", "ablation"], dropna=False)[keep]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        main.columns = [
            "__".join(str(x) for x in c if str(x) != "") if isinstance(c, tuple) else c
            for c in main.columns
        ]
        main.to_csv(out / "main_maxk_summary.csv", index=False)

    # Dedicated fair baseline table at k=1.
    fair1 = m[
        (m["eval_split"] == "test")
        & (m.k.astype(int) == 1)
        & (
            m.method.isin(["tabfairgan", "decaf", "singlefair", "multifair", "gdt"])
        )
    ]
    if len(fair1):
        fsum = (
            fair1.groupby(["dataset", "method", "classifier"], dropna=False)[keep]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        fsum.columns = [
            "__".join(str(x) for x in c if str(x) != "") if isinstance(c, tuple) else c
            for c in fsum.columns
        ]
        fsum.to_csv(out / "fair_baseline_k1_summary.csv", index=False)

    # Ablation summary at kmax.
    abl = m[(m["eval_split"] == "test") & m["family"].astype(str).str.contains("ablation", regex=False)].copy()
    if len(abl):
        asum = (
            abl.groupby(["dataset", "ablation", "classifier"], dropna=False)[keep]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        asum.columns = [
            "__".join(str(x) for x in c if str(x) != "") if isinstance(c, tuple) else c
            for c in asum.columns
        ]
        asum.to_csv(out / "ablation_summary.csv", index=False)

    # Sweep summary retains parameter values, making every OFAT curve recoverable.
    sw = m[(m["eval_split"] == "validation") & m["family"].astype(str).str.contains("sweep", regex=False)].copy()
    if len(sw):
        pars = [
            "lambda_fair", "lambda_u", "alpha", "eta", "epsilon",
            "update_batch", "candidate_count", "constraint_mode", "adapt",
            "group_mode", "penalty_mode", "feature_correction"
        ]
        ssum = (
            sw.groupby(pars + ["dataset", "classifier"], dropna=False)[keep]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        ssum.columns = [
            "__".join(str(x) for x in c if str(x) != "") if isinstance(c, tuple) else c
            for c in ssum.columns
        ]
        ssum.to_csv(out / "sweep_summary.csv", index=False)

    # Validation-only recommendation for the primary fairness strength.
    # Other design parameters remain fixed at their declared defaults and are
    # treated as sensitivity analyses. We select lambda_fair by minimising the
    # average of DP_mean and EO_mean subject to no more than 0.01 absolute
    # ROC-AUC loss versus the default GDT validation reference when available.
    if len(sw):
        numeric_cols = ["lambda_fair", "lambda_u", "alpha", "eta", "epsilon", "update_batch", "candidate_count"]
        for c in numeric_cols:
            if c in sw.columns:
                sw[c] = pd.to_numeric(sw[c], errors="coerce")

        lam = sw[
            np.isclose(sw["lambda_u"], float(DEFAULTS["lambda_u"]))
            & np.isclose(sw["alpha"], float(DEFAULTS["alpha"]))
            & np.isclose(sw["eta"], float(DEFAULTS["eta"]))
            & np.isclose(sw["epsilon"], float(DEFAULTS["epsilon"]))
            & (sw["update_batch"] == int(DEFAULTS["update_batch"]))
            & (sw["candidate_count"] == int(DEFAULTS["candidate_count"]))
            & (sw["constraint_mode"].astype(str) == str(DEFAULTS["constraint_mode"]))
            & (sw["group_mode"].astype(str) == str(DEFAULTS["group_mode"]))
            & (sw["penalty_mode"].astype(str) == str(DEFAULTS["penalty_mode"]))
            & (sw["feature_correction"].astype(str).str.lower().isin(["true", "1"]))
        ].copy()

        # If GDT validation references have not been run, use lambda=0 as the
        # within-sweep utility reference; this keeps selection validation-only.
        if len(lam):
            ref = (
                lam[np.isclose(lam["lambda_fair"], 0.0)]
                .groupby(["dataset", "seed"])["roc_auc"].mean()
                .rename("reference_auc")
            )
            joined = lam.merge(ref, on=["dataset", "seed"], how="left")
            joined["auc_loss"] = joined["reference_auc"] - joined["roc_auc"]
            joined["fairness_score"] = 0.5 * (joined["dp_mean"] + joined["eo_mean"])
            cal = (
                joined.groupby("lambda_fair", as_index=False)
                .agg(
                    mean_auc=("roc_auc", "mean"),
                    mean_auc_loss=("auc_loss", "mean"),
                    mean_dp=("dp_mean", "mean"),
                    mean_eo=("eo_mean", "mean"),
                    mean_fairness_score=("fairness_score", "mean"),
                    n=("roc_auc", "count"),
                )
                .sort_values(["mean_fairness_score", "mean_auc_loss"])
            )
            eligible = cal[cal["mean_auc_loss"] <= 0.01]
            cal["eligible_auc_loss_le_0p01"] = cal["mean_auc_loss"] <= 0.01
            if len(eligible):
                chosen = float(eligible.iloc[0]["lambda_fair"])
            else:
                chosen = float(cal.sort_values("mean_auc_loss").iloc[0]["lambda_fair"])
            cal["recommended_lambda_fair"] = chosen
            cal.to_csv(out / "lambda_fair_calibration.csv", index=False)

    # Paired dataset-level statistics at maximal intersection, RF only.
    # Seeds are averaged within dataset first: the 9 datasets are the paired units.
    stat_rows = []
    rf = maxm[maxm.classifier == "rf"].copy() if len(maxm) else pd.DataFrame()
    proposed = rf[
        (rf.method == "multifair")
        & (rf["family"].astype(str).str.contains("core", regex=False))
        & (rf["ablation"].fillna("").astype(str) == "")
    ].copy()

    baselines = [
        "gdt", "singlefair", "bootstrap", "gaussian_copula",
        "ctgan", "tvae", "tabddpm"
    ]
    stat_metrics = [x for x in ["roc_auc", "pr_auc", "dp_max", "eo_max", "aod_max", "worst_group_ba"] if x in m]

    for metric in stat_metrics:
        pa = proposed.groupby("dataset")[metric].mean()
        for baseline in baselines:
            pb = rf[rf.method == baseline].groupby("dataset")[metric].mean()
            joined = pd.concat([pa.rename("proposed"), pb.rename("baseline")], axis=1).dropna()
            if len(joined) < 2:
                continue
            diff = (joined.proposed - joined.baseline).to_numpy(dtype=float)
            lo, hi = _bootstrap_mean_ci(diff)
            try:
                pval = float(wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue) if np.any(diff != 0) else 1.0
            except Exception:
                pval = np.nan
            stat_rows.append({
                "metric": metric,
                "baseline": baseline,
                "n_datasets": int(len(diff)),
                "mean_difference_proposed_minus_baseline": float(np.mean(diff)),
                "bootstrap95_low": lo,
                "bootstrap95_high": hi,
                "wilcoxon_p": pval,
            })

    stats = pd.DataFrame(stat_rows)
    if len(stats):
        stats["holm_p"] = _holm(stats["wilcoxon_p"].to_numpy())
        stats.to_csv(out / "statistical_tests.csv", index=False)

    print(f"Wrote aggregate files to {out}")
