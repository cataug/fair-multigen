from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import DATASETS, DEFAULTS


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_paper_outputs(root: Path):
    base = root / "results" / "journal2026_v21"
    agg = base / "aggregate"
    figs = base / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    metrics_path = agg / "all_metrics.csv"
    runtime_path = agg / "all_runtime.csv"
    support_path = root / "manifests" / "intersection_support.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} does not exist. Run `aggregate` first."
        )

    m = pd.read_csv(metrics_path)

    # ------------------------------------------------------------------
    # Figure: EO_max versus number of sensitive/group attributes.
    # ------------------------------------------------------------------
    core = m[
        (m["eval_split"].astype(str) == "test")
        & m["family"].astype(str).str.contains("core", regex=False)
        & m["method"].isin(["gdt", "singlefair", "multifair"])
        & (m["classifier"] == "rf")
    ].copy()

    ksum = (
        core.groupby(["method", "k"], as_index=False)["eo_max"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in ["gdt", "singlefair", "multifair"]:
        z = ksum[ksum["method"] == method].sort_values("k")
        if len(z):
            ax.plot(z["k"], z["mean"], marker="o", label=method)
    ax.set_xlabel("Number of grouping attributes, k", fontsize=13)
    ax.set_ylabel("EO max difference", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    _save(fig, figs / "01_k_scaling_eo_max.pdf")

    # ------------------------------------------------------------------
    # Figure: support collapse measured independently of model outputs.
    # ------------------------------------------------------------------
    if support_path.exists():
        s = pd.read_csv(support_path)
        z = s.groupby("k", as_index=False).agg(
            median_support=("median_support", "median"),
            fraction_groups_lt_10=("fraction_groups_lt_10", "mean"),
            n_datasets=("dataset_id", "nunique"),
        )

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(z["k"], z["median_support"], marker="o")
        ax.set_yscale("log")
        ax.set_xlabel("Number of grouping attributes, k", fontsize=13)
        ax.set_ylabel("Median group support (log scale)", fontsize=13)
        ax.tick_params(labelsize=11)
        _save(fig, figs / "02_support_median_vs_k.pdf")

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(z["k"], z["fraction_groups_lt_10"], marker="o")
        ax.set_xlabel("Number of grouping attributes, k", fontsize=13)
        ax.set_ylabel("Fraction of groups with support < 10", fontsize=13)
        ax.tick_params(labelsize=11)
        _save(fig, figs / "03_sparse_fraction_vs_k.pdf")

    # ------------------------------------------------------------------
    # Figure: fairness/utility curve for lambda_fair OFAT sweep.
    # Only configurations where every other parameter equals the default.
    # ------------------------------------------------------------------
    sw = m[
        (m["eval_split"].astype(str) == "validation")
        & m["family"].astype(str).str.contains("sweep", regex=False)
        & (m["method"] == "multifair")
        & (m["classifier"] == "rf")
    ].copy()

    if len(sw):
        def close(col, value):
            return np.isclose(pd.to_numeric(sw[col], errors="coerce"), float(value))

        mask = (
            close("lambda_u", DEFAULTS["lambda_u"])
            & close("alpha", DEFAULTS["alpha"])
            & close("eta", DEFAULTS["eta"])
            & close("epsilon", DEFAULTS["epsilon"])
            & (pd.to_numeric(sw["update_batch"], errors="coerce") == DEFAULTS["update_batch"])
            & (pd.to_numeric(sw["candidate_count"], errors="coerce") == DEFAULTS["candidate_count"])
            & (sw["feature_correction"].astype(str).str.lower() == str(DEFAULTS["feature_correction"]).lower())
            & (sw["constraint_mode"].astype(str) == DEFAULTS["constraint_mode"])
            & (sw["group_mode"].astype(str) == DEFAULTS["group_mode"])
            & (sw["penalty_mode"].astype(str) == DEFAULTS["penalty_mode"])
        )
        lam = sw[mask].groupby("lambda_fair", as_index=False).agg(
            roc_auc=("roc_auc", "mean"),
            eo_max=("eo_max", "mean"),
            dp_max=("dp_max", "mean"),
        ).sort_values("lambda_fair")

        if len(lam):
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            ax.plot(lam["eo_max"], lam["roc_auc"], marker="o")
            for _, r in lam.iterrows():
                ax.annotate(
                    f"{r['lambda_fair']:g}",
                    (r["eo_max"], r["roc_auc"]),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=9,
                )
            ax.set_xlabel("EO max difference", fontsize=13)
            ax.set_ylabel("ROC-AUC", fontsize=13)
            ax.tick_params(labelsize=11)
            _save(fig, figs / "04_lambda_fair_tradeoff.pdf")

    # ------------------------------------------------------------------
    # Figure: sparse-support effect from actual test groups.
    # Main comparison at each job's maximum k.
    # ------------------------------------------------------------------
    bucket_path = agg / "all_support_buckets.csv"
    if bucket_path.exists():
        b = pd.read_csv(bucket_path)
        wanted = b[
            (b["classifier"] == "rf")
            & b["method"].isin(["gdt", "singlefair", "multifair"])
        ].copy()

        # Full MultiFair core and no-smoothing ablation are separately labelled.
        def variant(row):
            if row["method"] == "gdt":
                return "GDT"
            if row["method"] == "singlefair":
                return "SingleFair"
            abl = str(row.get("ablation", ""))
            if abl == "no_smooth":
                return "MultiFair-no-smooth"
            if "core" in str(row.get("family", "")) and abl in ("", "nan"):
                return "MultiFair"
            return None

        wanted["variant"] = wanted.apply(variant, axis=1)
        wanted = wanted[wanted["variant"].notna()]

        # Keep only kmax for each dataset.
        wanted["kmax"] = wanted["dataset"].map(
            lambda ds: len(DATASETS[ds]["groups"])
        )
        wanted = wanted[wanted["k"].astype(int) == wanted["kmax"].astype(int)]

        order = ["1-4", "5-9", "10-19", "20-49", "50-99", "100+"]
        summ = wanted.groupby(["variant", "support_bucket"], as_index=False).agg(
            eo=("eo_max_mean_within_bucket", "mean")
        )
        if len(summ):
            fig, ax = plt.subplots(figsize=(7.8, 4.8))
            for variant in ["GDT", "SingleFair", "MultiFair-no-smooth", "MultiFair"]:
                z = summ[summ["variant"] == variant].copy()
                if not len(z):
                    continue
                z["ord"] = z["support_bucket"].map({v: i for i, v in enumerate(order)})
                z = z.sort_values("ord")
                ax.plot(z["support_bucket"], z["eo"], marker="o", label=variant)
            ax.set_xlabel("Intersection-group support", fontsize=13)
            ax.set_ylabel("Mean group EO deviation", fontsize=13)
            ax.tick_params(labelsize=10)
            ax.legend(fontsize=10)
            _save(fig, figs / "05_eo_by_group_support.pdf")

    # ------------------------------------------------------------------
    # Runtime: one figure per dataset, no subplots.
    # ------------------------------------------------------------------
    if runtime_path.exists():
        r = pd.read_csv(runtime_path)
        rr = r[
            r["family"].astype(str).str.contains("runtime", regex=False)
            & r["method"].isin(["gdt", "multifair"])
        ].copy()
        for dataset in ["adult", "diabetes130", "census_income_kdd"]:
            z = rr[rr["dataset"] == dataset]
            if not len(z):
                continue
            s = z.groupby(["method", "train_rows"], as_index=False).agg(
                generation_seconds=("generation_seconds", "mean")
            )
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            for method in ["gdt", "multifair"]:
                q = s[s["method"] == method].sort_values("train_rows")
                if len(q):
                    ax.plot(q["train_rows"], q["generation_seconds"], marker="o", label=method)
            ax.set_xlabel("Training rows", fontsize=13)
            ax.set_ylabel("Generation time (s)", fontsize=13)
            ax.tick_params(labelsize=11)
            ax.legend(fontsize=11)
            _save(fig, figs / f"runtime_{dataset}.pdf")

    print(f"Wrote paper figures to {figs}")
