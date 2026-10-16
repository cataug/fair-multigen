from pathlib import Path
from collections import defaultdict
import math
import re
import json

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path("/home/tahiti/MultiFairGDT")
STATS = ROOT / "STATS"
OUT = ROOT / "FIGURES"

DIR_OVERVIEW = OUT / "00_overview"
DIR_DATASET = OUT / "10_dataset_method"
DIR_TRADEOFF = OUT / "20_tradeoffs"
DIR_SUPPORT = OUT / "30_support_levels"
DIR_MULTIFAIR = OUT / "40_multifair"
DIR_RUNTIME = OUT / "50_runtime"
DIR_DATA = OUT / "_data"

for p in [
    OUT,
    DIR_OVERVIEW,
    DIR_DATASET,
    DIR_TRADEOFF,
    DIR_SUPPORT,
    DIR_MULTIFAIR,
    DIR_RUNTIME,
    DIR_DATA,
]:
    p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# STYLE — Fractional Unlearn family
# =============================================================================

DPI = 300

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10.5,
    "figure.titlesize": 19,

    "figure.facecolor": "white",
    "savefig.facecolor": "white",

    "axes.facecolor": "#F1F1F1",
    "axes.edgecolor": "#9A9A9A",
    "axes.linewidth": 0.9,

    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.25,
    "grid.alpha": 1.0,

    "axes.spines.top": False,
    "axes.spines.right": False,

    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.facecolor": "white",
    "legend.edgecolor": "#BBBBBB",

    "lines.linewidth": 2.2,
    "lines.markersize": 7,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# High-contrast paper palette.
METHOD_STYLE = {
    "multifair": {
        "color": "#E57C23",
        "marker": "o",
        "label": "MultiFair",
    },
    "bootstrap": {
        "color": "#3776B6",
        "marker": "s",
        "label": "Bootstrap",
    },
    "ctgan": {
        "color": "#4D9A74",
        "marker": "^",
        "label": "CTGAN",
    },
    "gaussian_copula": {
        "color": "#7B7B7B",
        "marker": "D",
        "label": "Gaussian Copula",
    },
    "tabddpm": {
        "color": "#8B63A8",
        "marker": "P",
        "label": "TabDDPM",
    },
    "tvae": {
        "color": "#2A9D9A",
        "marker": "v",
        "label": "TVAE",
    },
    "decaf": {
        "color": "#C94F5D",
        "marker": "X",
        "label": "DECAF",
    },
    "real": {
        "color": "#222222",
        "marker": "*",
        "label": "Real",
    },
    "gdt": {
        "color": "#A06D2E",
        "marker": "<",
        "label": "GDT",
    },
    "singlefair": {
        "color": "#D2A730",
        "marker": ">",
        "label": "SingleFair",
    },
    "tabfairgan": {
        "color": "#CC6EA8",
        "marker": "h",
        "label": "TabFairGAN",
    },
}

METHOD_ORDER = [
    "bootstrap",
    "multifair",
    "singlefair",
    "gdt",
    "real",
    "gaussian_copula",
    "ctgan",
    "tvae",
    "tabddpm",
    "decaf",
    "tabfairgan",
]

METRIC_LABEL = {
    "macro_f1": "Macro-F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",

    "dp_max": "DP disparity",
    "eo_mean": "EO disparity",
    "aod_mean": "AOD disparity",

    "dp_max_s5": "DP, support ≥5",
    "dp_max_s10": "DP, support ≥10",

    "eo_mean_s5": "EO, support ≥5",
    "eo_mean_s10": "EO, support ≥10",
    "eo_max_s5": "EO max, support ≥5",

    "aod_mean_s5": "AOD, support ≥5",
    "aod_mean_s10": "AOD, support ≥10",

    "worst_group_ba": "Worst-group BA",
    "worst_group_ba_s5": "Worst-group BA, support ≥5",
    "worst_group_ba_s10": "Worst-group BA, support ≥10",

    "worst_group_fpr": "Worst-group FPR",
    "worst_group_tpr": "Worst-group TPR",
}

UTILITY_METRICS = [
    "macro_f1",
    "roc_auc",
    "pr_auc",
]

FAIRNESS_METRICS = [
    "dp_max",
    "eo_mean",
    "aod_mean",
]

SUPPORT_EO = [
    ("eo_mean_s5", "support ≥5"),
    ("eo_mean_s10", "support ≥10"),
    ("eo_mean", "all evaluable groups"),
]

SUPPORT_DP = [
    ("dp_max_s5", "support ≥5"),
    ("dp_max_s10", "support ≥10"),
    ("dp_max", "all evaluable groups"),
]

SUPPORT_WGBA = [
    ("worst_group_ba_s5", "support ≥5"),
    ("worst_group_ba_s10", "support ≥10"),
    ("worst_group_ba", "all evaluable groups"),
]

CONFIG_CANDIDATES = [
    "lambda_fair",
    "lambda_u",
    "k",
    "epsilon",
    "alpha",
    "eta",
    "adapt",
    "feature_correction",
    "update_batch",
    "min_eval_group_support",
]

SENSITIVITY_PRIORITY = [
    "lambda_fair",
    "k",
    "epsilon",
    "alpha",
    "lambda_u",
    "eta",
]

MAX_SENS_PARAMS_PER_DATASET = 3


# =============================================================================
# UTILITIES
# =============================================================================

FIGURE_RECORDS = []


def safe_name(x):
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    return x.strip("_")


def method_label(method):
    return METHOD_STYLE.get(method, {}).get(
        "label",
        str(method).replace("_", " ").title()
    )


def method_color(method):
    return METHOD_STYLE.get(method, {}).get(
        "color",
        "#555555"
    )


def method_marker(method):
    return METHOD_STYLE.get(method, {}).get(
        "marker",
        "o"
    )


def present_method_order(values):
    vals = list(dict.fromkeys(values))
    ordered = [x for x in METHOD_ORDER if x in vals]
    ordered += sorted(x for x in vals if x not in ordered)
    return ordered


def metric_label(metric):
    return METRIC_LABEL.get(
        metric,
        metric.replace("_", " ").title()
    )


def clean_axis(ax):
    ax.set_axisbelow(True)
    ax.tick_params(colors="#202020")
    ax.xaxis.label.set_color("#202020")
    ax.yaxis.label.set_color("#202020")
    ax.title.set_color("#111111")


def add_zero_lines(ax, x=False, y=False):
    if x:
        ax.axvline(
            0,
            color="#333333",
            linewidth=1.1,
            linestyle="--",
            alpha=0.8,
            zorder=1,
        )
    if y:
        ax.axhline(
            0,
            color="#333333",
            linewidth=1.1,
            linestyle="--",
            alpha=0.8,
            zorder=1,
        )


def save_figure(fig, directory, stem, title, kind):
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)

    FIGURE_RECORDS.append({
        "stem": stem,
        "title": title,
        "kind": kind,
        "png": str(png.relative_to(ROOT)),
        "pdf": str(pdf.relative_to(ROOT)),
    })

    print(f"[figure {len(FIGURE_RECORDS):03d}] {stem}")


def get_metric_table(df, dataset, metric):
    if metric not in df.columns:
        return pd.DataFrame()

    q = df[
        (df["dataset"] == dataset)
        & df[metric].notna()
    ].copy()

    return q


def seed_balanced_metric_stats(seed_df, dataset, metrics):
    q = seed_df[seed_df.dataset == dataset].copy()
    methods = present_method_order(q.method.unique())

    rows = []
    for method in methods:
        qm = q[q.method == method]

        for metric in metrics:
            if metric not in qm.columns:
                continue

            vals = pd.to_numeric(
                qm[metric],
                errors="coerce"
            ).dropna()

            if not len(vals):
                continue

            rows.append({
                "dataset": dataset,
                "method": method,
                "metric": metric,
                "n_seed": len(vals),
                "mean": vals.mean(),
                "std": vals.std(ddof=1) if len(vals) > 1 else np.nan,
                "median": vals.median(),
                "q25": vals.quantile(.25),
                "q75": vals.quantile(.75),
            })

    return pd.DataFrame(rows)


def pareto_front(df, x, y):
    """
    Minimize x, maximize y.
    """
    q = df[
        df[x].notna()
        & df[y].notna()
    ].copy()

    if q.empty:
        return q

    q = q.sort_values(
        [x, y],
        ascending=[True, False]
    )

    keep = []
    best_y = -np.inf

    for idx, r in q.iterrows():
        yy = float(r[y])
        if yy > best_y + 1e-12:
            keep.append(idx)
            best_y = yy

    return q.loc[keep].sort_values(x)


def bubble_sizes(values, lo=55, hi=900):
    v = pd.to_numeric(
        values,
        errors="coerce"
    ).astype(float)

    mask = np.isfinite(v)

    if not mask.any():
        return np.full(len(v), lo)

    vv = v.copy()
    vv[~mask] = np.nanmedian(v[mask])

    z = np.log10(np.clip(vv, 1e-3, None))

    if np.nanmax(z) <= np.nanmin(z) + 1e-12:
        return np.full(len(z), (lo + hi) / 2)

    zz = (
        (z - np.nanmin(z))
        / (np.nanmax(z) - np.nanmin(z))
    )

    return lo + zz * (hi - lo)


def annotate_points(ax, df, x, y, method_col="method"):
    for _, r in df.iterrows():
        ax.annotate(
            method_label(r[method_col]),
            (r[x], r[y]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
            color="#202020",
            bbox=dict(
                boxstyle="round,pad=0.16",
                facecolor="white",
                edgecolor="none",
                alpha=0.78,
            ),
        )


# =============================================================================
# LOAD + HARMONIZE
# =============================================================================

jobs_path = STATS / "00_jobs_harmonized.csv"
metrics_path = STATS / "16_metrics_long.csv"
coverage_path = STATS / "04_method_dataset_summary.csv"

for p in [jobs_path, metrics_path, coverage_path]:
    if not p.exists():
        raise SystemExit(f"Missing required file: {p}")

jobs = pd.read_csv(
    jobs_path,
    keep_default_na=False,
)

metrics = pd.read_csv(
    metrics_path,
    keep_default_na=False,
)

coverage = pd.read_csv(
    coverage_path,
    keep_default_na=False,
)

jobs["job_id"] = jobs["job_id"].astype(str)
metrics["job_id"] = metrics["job_id"].astype(str)

metrics["metric_clean"] = (
    metrics["metric"]
    .astype(str)
    .str.replace(
        "metrics.csv::",
        "",
        regex=False,
    )
)

bad_name = (
    metrics["metric_clean"].eq("")
    | metrics["metric_clean"].str.startswith("Unnamed")
)

metrics.loc[bad_name, "metric_clean"] = (
    metrics.loc[bad_name, "metric_id"]
    .astype(str)
    .str.replace(
        "metrics.csv::",
        "",
        regex=False,
    )
)

# Fields used as structural identifiers must never also become
# pivoted metric columns; otherwise reset_index() collides with them.
_RESERVED_METRIC_NAMES = {
    "job_id",
    "method",
    "dataset",
    "seed",
    "status",
}

_collisions = sorted(
    set(metrics["metric_clean"].astype(str))
    & _RESERVED_METRIC_NAMES
)

if _collisions:
    print(
        "[metrics] dropping structural fields from metric values:",
        ", ".join(_collisions),
    )

metrics = metrics[
    ~metrics["metric_clean"].astype(str).isin(_RESERVED_METRIC_NAMES)
].copy()

successful_ids = set(
    jobs.loc[
        jobs.status == "success",
        "job_id",
    ].astype(str)
)

metrics_success = metrics[
    metrics.job_id.isin(successful_ids)
].copy()

# Job × metric wide table.
wide = (
    metrics_success
    .pivot_table(
        index=[
            "job_id",
            "method",
            "dataset",
            "seed",
        ],
        columns="metric_clean",
        values="value",
        aggfunc="last",
    )
    .reset_index()
)

wide.columns.name = None

job_extra_cols = [
    c for c in [
        "job_id",
        "elapsed_sec",
        "elapsed_min",
        "elapsed_hours",
        "started_time",
        "finished_time",
    ]
    if c in jobs.columns
]

job_extra = (
    jobs[job_extra_cols]
    .drop_duplicates("job_id")
)

wide = wide.merge(
    job_extra,
    on="job_id",
    how="left",
)

# Convert relevant metric/config fields to numeric.
for c in wide.columns:
    if c in {
        "job_id",
        "method",
        "dataset",
        "started_time",
        "finished_time",
    }:
        continue

    try:
        wide[c] = pd.to_numeric(
            wide[c],
            errors="ignore",
        )
    except Exception:
        pass


# =============================================================================
# AGGREGATION LEVEL 1:
# seed-balanced summaries WITHIN EACH DATASET
#
# Important:
# - dataset dimension is NEVER collapsed.
# - for MultiFair, multiple configs belonging to the same seed are summarized
#   by the within-seed median here.
# - raw/config-level plots are generated separately below.
# =============================================================================

numeric_metrics = [
    c for c in [
        "macro_f1",
        "roc_auc",
        "pr_auc",
        "dp_max",
        "eo_mean",
        "aod_mean",
        "worst_group_ba",
        "worst_group_fpr",
        "worst_group_tpr",

        "dp_max_s5",
        "dp_max_s10",
        "eo_mean_s5",
        "eo_mean_s10",
        "eo_max_s5",
        "aod_mean_s5",
        "aod_mean_s10",
        "worst_group_ba_s5",
        "worst_group_ba_s10",

        "elapsed_min",
        "elapsed_hours",
    ]
    if c in wide.columns
]

seed_balanced = (
    wide.groupby(
        ["dataset", "method", "seed"],
        dropna=False,
    )[numeric_metrics]
    .median(numeric_only=True)
    .reset_index()
)

seed_balanced.to_csv(
    DIR_DATA / "seed_balanced_metrics.csv",
    index=False,
)

method_dataset_rows = []

for ds in sorted(seed_balanced.dataset.unique()):
    z = seed_balanced_metric_stats(
        seed_balanced,
        ds,
        numeric_metrics,
    )
    if len(z):
        method_dataset_rows.append(z)

if method_dataset_rows:
    method_dataset_stats = pd.concat(
        method_dataset_rows,
        ignore_index=True,
    )
else:
    method_dataset_stats = pd.DataFrame()

method_dataset_stats.to_csv(
    DIR_DATA / "method_dataset_seedbalanced_stats.csv",
    index=False,
)


# =============================================================================
# AGGREGATION LEVEL 2:
# MultiFair configuration means over seeds, still separately by dataset.
# =============================================================================

config_cols = []

for c in CONFIG_CANDIDATES:
    if c not in wide.columns:
        continue

    q = wide.loc[
        wide.method == "multifair",
        c,
    ]

    if q.notna().sum() == 0:
        continue

    config_cols.append(c)

mf = wide[
    wide.method == "multifair"
].copy()

mf_cfg = pd.DataFrame()

if len(mf) and config_cols:
    agg_cols = [
        c for c in [
            "macro_f1",
            "roc_auc",
            "pr_auc",
            "dp_max",
            "eo_mean",
            "aod_mean",
            "worst_group_ba",
            "worst_group_fpr",
            "worst_group_tpr",
            "elapsed_min",
        ]
        if c in mf.columns
    ]

    group_cols = [
        "dataset",
        *config_cols,
    ]

    grouped = mf.groupby(
        group_cols,
        dropna=False,
    )

    frames = []

    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {
            c: v
            for c, v in zip(
                group_cols,
                keys,
            )
        }

        row["n_runs"] = len(g)
        row["n_seeds"] = g["seed"].nunique()

        for metric in agg_cols:
            vals = pd.to_numeric(
                g[metric],
                errors="coerce",
            ).dropna()

            if len(vals):
                row[f"{metric}_mean"] = vals.mean()
                row[f"{metric}_std"] = (
                    vals.std(ddof=1)
                    if len(vals) > 1
                    else np.nan
                )
                row[f"{metric}_median"] = vals.median()

        frames.append(row)

    mf_cfg = pd.DataFrame(frames)

mf_cfg.to_csv(
    DIR_DATA / "multifair_configuration_means.csv",
    index=False,
)


# =============================================================================
# DATASET LIST
# =============================================================================

datasets = sorted(
    set(jobs.dataset.astype(str))
    | set(wide.dataset.astype(str))
)

print()
print("=" * 88)
print("BUILDING FIGURES")
print("=" * 88)
print("datasets:", ", ".join(datasets))
print("raw successful metric jobs:", len(wide))
print("MultiFair config columns:", config_cols)
print()


# =============================================================================
# OVERVIEW FIGURES
#
# These are MATRICES with dataset as a retained axis.
# No dataset averaging.
# =============================================================================

def plot_matrix(
    matrix,
    title,
    stem,
    directory,
    fmt=".3f",
    cmap="viridis",
    log_norm=False,
    footer=None,
):
    if matrix.empty:
        return

    mat = matrix.copy()

    # ordering
    methods = present_method_order(mat.index)
    mat = mat.reindex(methods)

    ds_order = [
        d for d in datasets
        if d in mat.columns
    ]
    mat = mat.reindex(columns=ds_order)

    arr = mat.to_numpy(dtype=float)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return

    masked = np.ma.masked_invalid(arr)

    fig_w = max(
        11,
        1.35 * len(mat.columns) + 4.5,
    )
    fig_h = max(
        5,
        0.58 * len(mat.index) + 2.8,
    )

    fig, ax = plt.subplots(
        figsize=(fig_w, fig_h),
        constrained_layout=True,
    )

    if log_norm:
        positives = finite[finite > 0]
        if positives.size:
            norm = LogNorm(
                vmin=max(
                    positives.min(),
                    1e-6,
                ),
                vmax=positives.max(),
            )
        else:
            norm = Normalize(
                finite.min(),
                finite.max(),
            )
    else:
        lo = finite.min()
        hi = finite.max()

        if hi <= lo + 1e-12:
            hi = lo + 1e-6

        norm = Normalize(
            vmin=lo,
            vmax=hi,
        )

    im = ax.imshow(
        masked,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax.set_xticks(
        np.arange(len(mat.columns))
    )
    ax.set_xticklabels(
        [
            x.replace("_", "\n")
            for x in mat.columns
        ],
        rotation=0,
        ha="center",
    )

    ax.set_yticks(
        np.arange(len(mat.index))
    )
    ax.set_yticklabels(
        [
            method_label(x)
            for x in mat.index
        ]
    )

    ax.grid(False)

    # outlined cells
    ax.set_xticks(
        np.arange(-0.5, len(mat.columns), 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, len(mat.index), 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=2.0,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # Always black text; light bbox preserves readability.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isfinite(v):
                continue

            txt = format(v, fmt)

            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=9.2,
                color="#111111",
                bbox=dict(
                    boxstyle="round,pad=0.13",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.70,
                ),
            )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )
    cbar.ax.tick_params(
        labelsize=10,
        colors="#202020",
    )

    ax.set_title(
        title,
        pad=14,
        weight="bold",
    )

    if footer:
        fig.text(
            0.5,
            0.002,
            footer,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#444444",
        )

    clean_axis(ax)

    save_figure(
        fig,
        directory,
        stem,
        title,
        "overview_matrix",
    )


# 1. Success fraction.
cov = coverage.copy()

for c in [
    "success",
    "total",
    "failed",
    "running",
    "left",
]:
    if c in cov.columns:
        cov[c] = pd.to_numeric(
            cov[c],
            errors="coerce",
        )

cov["success_fraction"] = (
    cov["success"]
    / cov["total"].replace(0, np.nan)
)

m = cov.pivot(
    index="method",
    columns="dataset",
    values="success_fraction",
)

plot_matrix(
    m,
    "Experiment completion by method and dataset",
    "fig_001_success_fraction",
    DIR_OVERVIEW,
    fmt=".2f",
    cmap="viridis",
    footer="Each cell is success / planned jobs for that method–dataset pair.",
)


# 2. Failure counts.
m = cov.pivot(
    index="method",
    columns="dataset",
    values="failed",
)

plot_matrix(
    m,
    "Failed jobs by method and dataset",
    "fig_002_failure_counts",
    DIR_OVERVIEW,
    fmt=".0f",
    cmap="magma",
)


# 3. Metric-job coverage.
metric_cov = (
    wide.groupby(
        ["method", "dataset"]
    )["job_id"]
    .nunique()
    .reset_index(name="metric_jobs")
)

m = metric_cov.pivot(
    index="method",
    columns="dataset",
    values="metric_jobs",
)

plot_matrix(
    m,
    "Successful jobs with harmonized metrics",
    "fig_003_metric_coverage",
    DIR_OVERVIEW,
    fmt=".0f",
    cmap="viridis",
)


# 4. Runtime median matrix — from successful jobs, per method × dataset.
runtime_success = jobs[
    jobs.status == "success"
].copy()

runtime_success["elapsed_min"] = pd.to_numeric(
    runtime_success["elapsed_min"],
    errors="coerce",
)

runtime_med = (
    runtime_success
    .groupby(
        ["method", "dataset"]
    )["elapsed_min"]
    .median()
    .reset_index()
)

m = runtime_med.pivot(
    index="method",
    columns="dataset",
    values="elapsed_min",
)

plot_matrix(
    m,
    "Median successful runtime within each dataset",
    "fig_004_runtime_median",
    DIR_OVERVIEW,
    fmt=".1f",
    cmap="viridis",
    log_norm=True,
    footer="Minutes; logarithmic color scale. No averaging across datasets.",
)


# 5. Runtime p90 matrix.
runtime_p90 = (
    runtime_success
    .groupby(
        ["method", "dataset"]
    )["elapsed_min"]
    .quantile(.90)
    .reset_index()
)

m = runtime_p90.pivot(
    index="method",
    columns="dataset",
    values="elapsed_min",
)

plot_matrix(
    m,
    "90th-percentile successful runtime within each dataset",
    "fig_005_runtime_p90",
    DIR_OVERVIEW,
    fmt=".1f",
    cmap="viridis",
    log_norm=True,
    footer="Minutes; logarithmic color scale.",
)


def seedbalanced_matrix(metric):
    if metric not in seed_balanced.columns:
        return pd.DataFrame()

    q = (
        seed_balanced
        .groupby(
            ["method", "dataset"]
        )[metric]
        .mean()
        .reset_index()
    )

    return q.pivot(
        index="method",
        columns="dataset",
        values=metric,
    )


# 6. Macro-F1 matrix.
plot_matrix(
    seedbalanced_matrix("macro_f1"),
    "Macro-F1 by method and dataset",
    "fig_006_macro_f1_matrix",
    DIR_OVERVIEW,
    fmt=".3f",
    cmap="viridis",
    footer="Within each dataset: equal weight per seed. Dataset dimension is not collapsed.",
)


# 7. EO matrix.
plot_matrix(
    seedbalanced_matrix("eo_mean"),
    "Equalized-odds disparity by method and dataset",
    "fig_007_eo_matrix",
    DIR_OVERVIEW,
    fmt=".3f",
    cmap="viridis_r",
    footer="Lower is better; each column remains a separate dataset.",
)


# 8. Worst-group BA matrix.
plot_matrix(
    seedbalanced_matrix("worst_group_ba"),
    "Worst-group balanced accuracy by method and dataset",
    "fig_008_worst_group_ba_matrix",
    DIR_OVERVIEW,
    fmt=".3f",
    cmap="viridis",
    footer="Higher is better.",
)


# 9. DP matrix.
plot_matrix(
    seedbalanced_matrix("dp_max"),
    "Demographic-parity disparity by method and dataset",
    "fig_009_dp_matrix",
    DIR_OVERVIEW,
    fmt=".3f",
    cmap="viridis_r",
    footer="Lower is better.",
)


# 10. Utility retention relative to Bootstrap, per cell.
if (
    "macro_f1" in seed_balanced.columns
    and "bootstrap" in set(seed_balanced.method)
):
    tmp = (
        seed_balanced
        .groupby(
            ["method", "dataset"]
        )["macro_f1"]
        .mean()
        .reset_index()
    )

    boot = (
        tmp[tmp.method == "bootstrap"]
        [["dataset", "macro_f1"]]
        .rename(
            columns={
                "macro_f1": "bootstrap_macro_f1"
            }
        )
    )

    tmp = tmp.merge(
        boot,
        on="dataset",
        how="left",
    )

    tmp["delta_macro_f1"] = (
        tmp["macro_f1"]
        - tmp["bootstrap_macro_f1"]
    )

    m = tmp.pivot(
        index="method",
        columns="dataset",
        values="delta_macro_f1",
    )

    plot_matrix(
        m,
        "Macro-F1 difference relative to Bootstrap",
        "fig_010_delta_f1_vs_bootstrap_matrix",
        DIR_OVERVIEW,
        fmt="+.3f",
        cmap="coolwarm",
        footer="Difference is computed separately within every dataset.",
    )


# =============================================================================
# DATASET-SPECIFIC FIGURES
# =============================================================================

def plot_seedbalanced_bars(
    dataset,
    metrics_list,
    title,
    stem,
    ylabel=None,
    directory=DIR_DATASET,
):
    q = seed_balanced[
        seed_balanced.dataset == dataset
    ].copy()

    metrics_here = [
        m for m in metrics_list
        if m in q.columns
        and q[m].notna().any()
    ]

    if not metrics_here:
        return

    methods = present_method_order(
        q.method.unique()
    )

    if not methods:
        return

    n_metrics = len(metrics_here)
    x = np.arange(len(methods))
    width = min(
        0.24,
        0.82 / max(n_metrics, 1),
    )

    cmap = plt.get_cmap("viridis")
    metric_colors = [
        cmap(v)
        for v in np.linspace(
            0.18,
            0.82,
            n_metrics,
        )
    ]

    fig_w = max(
        11,
        1.22 * len(methods) + 4.5,
    )

    fig, ax = plt.subplots(
        figsize=(fig_w, 6.3),
        constrained_layout=True,
    )

    offsets = (
        np.arange(n_metrics)
        - (n_metrics - 1) / 2
    ) * width

    for j, metric in enumerate(metrics_here):
        means = []
        stds = []

        for method in methods:
            vals = pd.to_numeric(
                q.loc[
                    q.method == method,
                    metric,
                ],
                errors="coerce",
            ).dropna()

            means.append(
                vals.mean()
                if len(vals)
                else np.nan
            )

            stds.append(
                vals.std(ddof=1)
                if len(vals) > 1
                else 0.0
            )

        means = np.array(
            means,
            dtype=float,
        )
        stds = np.array(
            stds,
            dtype=float,
        )

        xpos = x + offsets[j]

        bars = ax.bar(
            xpos,
            means,
            width=width * 0.92,
            yerr=stds,
            capsize=3.5,
            color=metric_colors[j],
            edgecolor="#252525",
            linewidth=0.7,
            alpha=0.92,
            label=metric_label(metric),
            zorder=3,
        )

        # seed points
        for i, method in enumerate(methods):
            vals = pd.to_numeric(
                q.loc[
                    q.method == method,
                    metric,
                ],
                errors="coerce",
            ).dropna()

            if not len(vals):
                continue

            jitter = np.linspace(
                -width * 0.18,
                width * 0.18,
                len(vals),
            )

            ax.scatter(
                np.full(len(vals), xpos[i]) + jitter,
                vals,
                s=18,
                facecolor="white",
                edgecolor="#202020",
                linewidth=0.6,
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            method_label(m)
            for m in methods
        ],
        rotation=22,
        ha="right",
    )

    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_title(
        f"{title}\n{dataset}",
        weight="bold",
    )

    ax.legend(
        ncol=min(
            len(metrics_here),
            3,
        ),
        loc="best",
    )

    clean_axis(ax)

    save_figure(
        fig,
        directory,
        stem,
        f"{title}: {dataset}",
        "seed_balanced_bars",
    )


def plot_support_paths(dataset, stem):
    q = seed_balanced[
        seed_balanced.dataset == dataset
    ].copy()

    available_blocks = []

    for block_name, block in [
        ("EO disparity", SUPPORT_EO),
        ("DP disparity", SUPPORT_DP),
        ("Worst-group BA", SUPPORT_WGBA),
    ]:
        cols = [
            c for c, _ in block
            if c in q.columns
            and q[c].notna().any()
        ]

        if len(cols) >= 2:
            available_blocks.append(
                (block_name, block)
            )

    if not available_blocks:
        return

    fig, axes = plt.subplots(
        1,
        len(available_blocks),
        figsize=(
            6.2 * len(available_blocks),
            5.6,
        ),
        constrained_layout=True,
        squeeze=False,
    )

    axes = axes[0]

    methods = present_method_order(
        q.method.unique()
    )

    for ax, (block_name, block) in zip(
        axes,
        available_blocks,
    ):
        xlabels = [
            label
            for col, label in block
            if col in q.columns
            and q[col].notna().any()
        ]

        cols = [
            col
            for col, label in block
            if col in q.columns
            and q[col].notna().any()
        ]

        x = np.arange(len(cols))

        for method in methods:
            qm = q[
                q.method == method
            ]

            means = []
            valid = True

            for col in cols:
                vals = pd.to_numeric(
                    qm[col],
                    errors="coerce",
                ).dropna()

                if not len(vals):
                    valid = False
                    break

                means.append(vals.mean())

            if not valid:
                continue

            ax.plot(
                x,
                means,
                marker=method_marker(method),
                color=method_color(method),
                linewidth=2.2,
                markersize=7,
                label=method_label(method),
                alpha=0.92,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            xlabels,
            rotation=14,
            ha="right",
        )

        ax.set_title(block_name)
        clean_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=min(
                5,
                len(handles),
            ),
        )

    fig.suptitle(
        f"Group-support sensitivity — {dataset}",
        weight="bold",
        y=1.10,
    )

    save_figure(
        fig,
        DIR_SUPPORT,
        stem,
        f"Group-support sensitivity: {dataset}",
        "support_paths",
    )


def plot_raw_tradeoff(
    dataset,
    fairness_metric,
    stem,
    title,
):
    if (
        "macro_f1" not in wide.columns
        or fairness_metric not in wide.columns
    ):
        return

    q = wide[
        (wide.dataset == dataset)
        & wide["macro_f1"].notna()
        & wide[fairness_metric].notna()
    ].copy()

    if q.empty:
        return

    methods = present_method_order(
        q.method.unique()
    )

    fig, ax = plt.subplots(
        figsize=(8.4, 6.7),
        constrained_layout=True,
    )

    # raw runs
    for method in methods:
        qm = q[
            q.method == method
        ]

        alpha = (
            0.20
            if method == "multifair"
            else 0.55
        )

        size = (
            26
            if method == "multifair"
            else 42
        )

        ax.scatter(
            qm[fairness_metric],
            qm["macro_f1"],
            s=size,
            color=method_color(method),
            marker=method_marker(method),
            alpha=alpha,
            linewidth=0.35,
            edgecolor="#202020",
            label=method_label(method),
            zorder=2,
        )

    # seed-balanced method centroids
    sb = seed_balanced[
        seed_balanced.dataset == dataset
    ].copy()

    centres = []

    for method in methods:
        qm = sb[
            sb.method == method
        ]

        if (
            qm.empty
            or fairness_metric not in qm.columns
        ):
            continue

        xvals = pd.to_numeric(
            qm[fairness_metric],
            errors="coerce",
        ).dropna()

        yvals = pd.to_numeric(
            qm["macro_f1"],
            errors="coerce",
        ).dropna()

        if not len(xvals) or not len(yvals):
            continue

        row = {
            "method": method,
            fairness_metric: xvals.mean(),
            "macro_f1": yvals.mean(),
        }
        centres.append(row)

        ax.scatter(
            row[fairness_metric],
            row["macro_f1"],
            s=190,
            color=method_color(method),
            marker=method_marker(method),
            edgecolor="#111111",
            linewidth=1.3,
            zorder=7,
        )

    if centres:
        centres = pd.DataFrame(centres)
        annotate_points(
            ax,
            centres,
            fairness_metric,
            "macro_f1",
        )

    ax.set_xlabel(
        f"{metric_label(fairness_metric)} ↓"
    )
    ax.set_ylabel("Macro-F1 ↑")

    ax.set_title(
        f"{title}\n{dataset}",
        weight="bold",
    )

    # Custom legend without duplicating raw/centroid
    handles = [
        Line2D(
            [0],
            [0],
            marker=method_marker(m),
            linestyle="none",
            markerfacecolor=method_color(m),
            markeredgecolor="#222222",
            markersize=8,
            label=method_label(m),
        )
        for m in methods
    ]

    ax.legend(
        handles=handles,
        loc="best",
        ncol=2,
    )

    clean_axis(ax)

    save_figure(
        fig,
        DIR_TRADEOFF,
        stem,
        f"{title}: {dataset}",
        "raw_tradeoff",
    )


def plot_runtime_utility(dataset, stem):
    if (
        "macro_f1" not in wide.columns
        or "elapsed_min" not in wide.columns
    ):
        return

    q = wide[
        (wide.dataset == dataset)
        & wide["macro_f1"].notna()
        & wide["elapsed_min"].notna()
    ].copy()

    q["elapsed_min"] = pd.to_numeric(
        q["elapsed_min"],
        errors="coerce",
    )

    q = q[
        q.elapsed_min > 0
    ]

    if q.empty:
        return

    methods = present_method_order(
        q.method.unique()
    )

    fig, ax = plt.subplots(
        figsize=(8.6, 6.7),
        constrained_layout=True,
    )

    for method in methods:
        qm = q[
            q.method == method
        ]

        ax.scatter(
            qm["elapsed_min"],
            qm["macro_f1"],
            s=(
                27
                if method == "multifair"
                else 45
            ),
            color=method_color(method),
            marker=method_marker(method),
            edgecolor="#202020",
            linewidth=0.4,
            alpha=(
                0.22
                if method == "multifair"
                else 0.62
            ),
            label=method_label(method),
        )

        # within-dataset method median runtime and seed-balanced F1
        x = np.median(
            qm["elapsed_min"]
        )

        sb = seed_balanced[
            (seed_balanced.dataset == dataset)
            & (seed_balanced.method == method)
        ]

        if (
            len(sb)
            and sb["macro_f1"].notna().any()
        ):
            y = sb["macro_f1"].mean()

            ax.scatter(
                x,
                y,
                s=200,
                color=method_color(method),
                marker=method_marker(method),
                edgecolor="#111111",
                linewidth=1.3,
                zorder=8,
            )

            ax.annotate(
                method_label(method),
                (x, y),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=9,
                color="#222222",
                bbox=dict(
                    boxstyle="round,pad=0.14",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
            )

    ax.set_xscale("log")
    ax.set_xlabel(
        "Runtime per successful job [min, log scale] ↓"
    )
    ax.set_ylabel("Macro-F1 ↑")

    ax.set_title(
        f"Compute–utility trade-off\n{dataset}",
        weight="bold",
    )

    clean_axis(ax)

    save_figure(
        fig,
        DIR_RUNTIME,
        stem,
        f"Compute–utility trade-off: {dataset}",
        "runtime_tradeoff",
    )


def plot_bubble_tradeoff(dataset, stem):
    needed = [
        "macro_f1",
        "eo_mean",
        "elapsed_min",
    ]

    if not all(
        x in wide.columns
        for x in needed
    ):
        return

    q = wide[
        (wide.dataset == dataset)
        & wide["macro_f1"].notna()
        & wide["eo_mean"].notna()
        & wide["elapsed_min"].notna()
    ].copy()

    if q.empty:
        return

    methods = present_method_order(
        q.method.unique()
    )

    fig, ax = plt.subplots(
        figsize=(8.7, 6.8),
        constrained_layout=True,
    )

    for method in methods:
        qm = q[
            q.method == method
        ].copy()

        sizes = bubble_sizes(
            qm["elapsed_min"],
            lo=35,
            hi=500,
        )

        ax.scatter(
            qm["eo_mean"],
            qm["macro_f1"],
            s=sizes,
            color=method_color(method),
            marker=method_marker(method),
            edgecolor="#1F1F1F",
            linewidth=0.45,
            alpha=(
                0.19
                if method == "multifair"
                else 0.50
            ),
        )

    ax.set_xlabel("EO disparity ↓")
    ax.set_ylabel("Macro-F1 ↑")

    ax.set_title(
        f"Fairness–utility–compute landscape\n{dataset}",
        weight="bold",
    )

    ax.text(
        0.99,
        0.02,
        "Bubble area ∝ log runtime",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#333333",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.76,
            boxstyle="round,pad=0.22",
        ),
    )

    clean_axis(ax)

    save_figure(
        fig,
        DIR_TRADEOFF,
        stem,
        f"Fairness–utility–compute landscape: {dataset}",
        "bubble_tradeoff",
    )


def plot_delta_vs_bootstrap(dataset, stem):
    required = [
        "macro_f1",
        "eo_mean",
    ]

    if not all(
        x in seed_balanced.columns
        for x in required
    ):
        return

    q = seed_balanced[
        seed_balanced.dataset == dataset
    ]

    b = q[
        q.method == "bootstrap"
    ]

    if b.empty:
        return

    b_f1 = pd.to_numeric(
        b["macro_f1"],
        errors="coerce",
    ).mean()

    b_eo = pd.to_numeric(
        b["eo_mean"],
        errors="coerce",
    ).mean()

    rows = []

    for method in present_method_order(
        q.method.unique()
    ):
        qm = q[
            q.method == method
        ]

        if qm.empty:
            continue

        f1 = pd.to_numeric(
            qm["macro_f1"],
            errors="coerce",
        ).mean()

        eo = pd.to_numeric(
            qm["eo_mean"],
            errors="coerce",
        ).mean()

        if not (
            np.isfinite(f1)
            and np.isfinite(eo)
        ):
            continue

        rows.append({
            "method": method,
            "delta_eo": eo - b_eo,
            "delta_f1": f1 - b_f1,
        })

    d = pd.DataFrame(rows)

    if d.empty:
        return

    fig, ax = plt.subplots(
        figsize=(8.2, 6.5),
        constrained_layout=True,
    )

    add_zero_lines(
        ax,
        x=True,
        y=True,
    )

    for _, r in d.iterrows():
        method = r["method"]

        ax.scatter(
            r["delta_eo"],
            r["delta_f1"],
            s=190,
            color=method_color(method),
            marker=method_marker(method),
            edgecolor="#111111",
            linewidth=1.2,
            zorder=5,
        )

    annotate_points(
        ax,
        d,
        "delta_eo",
        "delta_f1",
    )

    ax.set_xlabel(
        "Δ EO vs Bootstrap  ← fairer | less fair →"
    )
    ax.set_ylabel(
        "Δ Macro-F1 vs Bootstrap  ↓ lower | higher ↑"
    )

    ax.set_title(
        f"Utility–fairness movement relative to Bootstrap\n{dataset}",
        weight="bold",
    )

    # desirability hint, not a ranking
    ax.text(
        0.015,
        0.985,
        "upper-left: higher utility + lower EO",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#333333",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.78,
            boxstyle="round,pad=0.22",
        ),
    )

    clean_axis(ax)

    save_figure(
        fig,
        DIR_TRADEOFF,
        stem,
        f"Movement relative to Bootstrap: {dataset}",
        "delta_vs_bootstrap",
    )


# =============================================================================
# MULTIFAIR CONFIGURATION-LEVEL FIGURES
# =============================================================================

def get_cfg_col(metric, stat="mean"):
    c = f"{metric}_{stat}"
    return (
        c
        if c in mf_cfg.columns
        else None
    )


def plot_multifair_config_pareto(
    dataset,
    fairness_metric,
    stem,
):
    if mf_cfg.empty:
        return

    x = get_cfg_col(
        fairness_metric,
        "mean",
    )
    y = get_cfg_col(
        "macro_f1",
        "mean",
    )

    if not x or not y:
        return

    q = mf_cfg[
        mf_cfg.dataset == dataset
    ].copy()

    q = q[
        q[x].notna()
        & q[y].notna()
    ]

    if len(q) < 2:
        return

    runtime_col = get_cfg_col(
        "elapsed_min",
        "median",
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 6.7),
        constrained_layout=True,
    )

    if (
        runtime_col
        and q[runtime_col].notna().any()
    ):
        rv = np.log10(
            np.clip(
                pd.to_numeric(
                    q[runtime_col],
                    errors="coerce",
                ),
                1e-3,
                None,
            )
        )

        norm = Normalize(
            np.nanmin(rv),
            np.nanmax(rv)
            if np.nanmax(rv) > np.nanmin(rv)
            else np.nanmin(rv) + 1e-6,
        )

        sc = ax.scatter(
            q[x],
            q[y],
            c=rv,
            cmap="viridis",
            norm=norm,
            s=58,
            edgecolor="#222222",
            linewidth=0.5,
            alpha=0.74,
            zorder=3,
        )

        cb = fig.colorbar(
            sc,
            ax=ax,
            pad=0.02,
        )
        cb.set_label(
            "log10 median runtime [min]"
        )

    else:
        ax.scatter(
            q[x],
            q[y],
            s=58,
            color=method_color("multifair"),
            edgecolor="#222222",
            linewidth=0.5,
            alpha=0.70,
            zorder=3,
        )

    front = pareto_front(
        q,
        x,
        y,
    )

    if len(front):
        ax.plot(
            front[x],
            front[y],
            color=method_color("multifair"),
            linewidth=3.1,
            marker="o",
            markersize=7,
            markeredgecolor="#111111",
            markerfacecolor=method_color("multifair"),
            zorder=7,
            label="MultiFair Pareto envelope",
        )

    # Baseline seed-balanced centroids in same dataset.
    sb = seed_balanced[
        seed_balanced.dataset == dataset
    ]

    for method in present_method_order(
        sb.method.unique()
    ):
        if method == "multifair":
            continue

        qm = sb[
            sb.method == method
        ]

        if (
            fairness_metric not in qm.columns
            or "macro_f1" not in qm.columns
        ):
            continue

        xx = pd.to_numeric(
            qm[fairness_metric],
            errors="coerce",
        ).mean()

        yy = pd.to_numeric(
            qm["macro_f1"],
            errors="coerce",
        ).mean()

        if not (
            np.isfinite(xx)
            and np.isfinite(yy)
        ):
            continue

        ax.scatter(
            xx,
            yy,
            s=210,
            color=method_color(method),
            marker=method_marker(method),
            edgecolor="#111111",
            linewidth=1.2,
            zorder=8,
        )

        ax.annotate(
            method_label(method),
            (xx, yy),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
            color="#202020",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.77,
                boxstyle="round,pad=0.16",
            ),
        )

    ax.set_xlabel(
        f"{metric_label(fairness_metric)} ↓"
    )
    ax.set_ylabel("Macro-F1 ↑")

    ax.set_title(
        f"MultiFair configuration Pareto landscape\n{dataset}",
        weight="bold",
    )

    if len(front):
        ax.legend(loc="best")

    clean_axis(ax)

    save_figure(
        fig,
        DIR_MULTIFAIR,
        stem,
        f"MultiFair config Pareto: {dataset}",
        "multifair_config_pareto",
    )


def plot_multifair_sensitivity(
    dataset,
    param,
    stem,
):
    if mf_cfg.empty or param not in mf_cfg.columns:
        return

    q = mf_cfg[
        mf_cfg.dataset == dataset
    ].copy()

    if q.empty:
        return

    q[param] = pd.to_numeric(
        q[param],
        errors="coerce",
    )

    q = q[
        q[param].notna()
    ]

    unique = sorted(
        q[param].unique()
    )

    if len(unique) < 2:
        return

    metrics_to_plot = []

    for metric, label, direction in [
        ("macro_f1", "Macro-F1 ↑", "higher"),
        ("eo_mean", "EO disparity ↓", "lower"),
        ("worst_group_ba", "Worst-group BA ↑", "higher"),
        ("elapsed_min", "Runtime [min] ↓", "lower"),
    ]:
        c = get_cfg_col(
            metric,
            "mean",
        )

        if metric == "elapsed_min":
            c = (
                get_cfg_col(
                    metric,
                    "median",
                )
                or c
            )

        if (
            c
            and c in q.columns
            and q[c].notna().any()
        ):
            metrics_to_plot.append(
                (metric, c, label)
            )

    if len(metrics_to_plot) < 2:
        return

    # up to four panels
    n = len(metrics_to_plot)
    ncols = 2
    nrows = math.ceil(n / 2)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12.4, 4.8 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    axes_flat = axes.ravel()

    cmap = plt.get_cmap("viridis")
    panel_colors = [
        cmap(v)
        for v in np.linspace(
            0.18,
            0.82,
            len(metrics_to_plot),
        )
    ]

    for k, (metric, col, label) in enumerate(
        metrics_to_plot
    ):
        ax = axes_flat[k]

        rows = []

        for val in unique:
            z = pd.to_numeric(
                q.loc[
                    q[param] == val,
                    col,
                ],
                errors="coerce",
            ).dropna()

            if not len(z):
                continue

            rows.append({
                "x": val,
                "median": z.median(),
                "q25": z.quantile(.25),
                "q75": z.quantile(.75),
                "n": len(z),
            })

        a = pd.DataFrame(rows)

        if a.empty:
            ax.axis("off")
            continue

        ax.plot(
            a["x"],
            a["median"],
            marker="o",
            color=panel_colors[k],
            linewidth=2.7,
            markersize=7,
        )

        ax.fill_between(
            a["x"],
            a["q25"],
            a["q75"],
            color=panel_colors[k],
            alpha=0.18,
            linewidth=0,
        )

        # Show underlying config means.
        ax.scatter(
            q[param],
            q[col],
            s=20,
            color=panel_colors[k],
            alpha=0.18,
            edgecolor="none",
        )

        ax.set_xlabel(
            param.replace("_", " ")
        )
        ax.set_ylabel(label)
        ax.set_title(
            metric_label(metric)
            if metric != "elapsed_min"
            else "Runtime"
        )

        if metric == "elapsed_min":
            positive = pd.to_numeric(
                q[col],
                errors="coerce",
            )
            if (
                (positive > 0).sum()
                == positive.notna().sum()
                and positive.notna().sum()
            ):
                ax.set_yscale("log")

        clean_axis(ax)

    for k in range(
        len(metrics_to_plot),
        len(axes_flat),
    ):
        axes_flat[k].axis("off")

    fig.suptitle(
        f"MultiFair sensitivity to {param.replace('_', ' ')} — {dataset}\n"
        f"median and IQR over other configuration dimensions",
        weight="bold",
        y=1.03,
    )

    save_figure(
        fig,
        DIR_MULTIFAIR,
        stem,
        f"MultiFair {param} sensitivity: {dataset}",
        "multifair_sensitivity",
    )


# =============================================================================
# GENERATE DATASET FIGURES
# =============================================================================

for ds in datasets:
    ds_safe = safe_name(ds)

    print()
    print("-" * 88)
    print("DATASET:", ds)
    print("-" * 88)

    # A. Utility — within dataset only.
    plot_seedbalanced_bars(
        ds,
        UTILITY_METRICS,
        "Predictive utility — seed-balanced",
        f"{ds_safe}__utility_seedbalanced",
        ylabel="Score ↑",
    )

    # B. Fairness disparities.
    plot_seedbalanced_bars(
        ds,
        FAIRNESS_METRICS,
        "Fairness disparities — seed-balanced",
        f"{ds_safe}__fairness_seedbalanced",
        ylabel="Disparity ↓",
    )

    # C. Worst-group performance at several support thresholds.
    plot_seedbalanced_bars(
        ds,
        [
            "worst_group_ba",
            "worst_group_ba_s5",
            "worst_group_ba_s10",
        ],
        "Worst-group balanced accuracy",
        f"{ds_safe}__worst_group_ba",
        ylabel="Balanced accuracy ↑",
    )

    # D. Support-level trajectories.
    plot_support_paths(
        ds,
        f"{ds_safe}__support_trajectories",
    )

    # E. Raw job-level F1 vs EO.
    plot_raw_tradeoff(
        ds,
        "eo_mean",
        f"{ds_safe}__raw_f1_vs_eo",
        "Run-level utility–EO landscape",
    )

    # F. Raw job-level F1 vs DP.
    plot_raw_tradeoff(
        ds,
        "dp_max",
        f"{ds_safe}__raw_f1_vs_dp",
        "Run-level utility–DP landscape",
    )

    # G. Raw job-level F1 vs AOD.
    plot_raw_tradeoff(
        ds,
        "aod_mean",
        f"{ds_safe}__raw_f1_vs_aod",
        "Run-level utility–AOD landscape",
    )

    # H. Runtime–utility.
    plot_runtime_utility(
        ds,
        f"{ds_safe}__runtime_vs_f1",
    )

    # I. Three-way bubble.
    plot_bubble_tradeoff(
        ds,
        f"{ds_safe}__bubble_f1_eo_runtime",
    )

    # J. Difference from bootstrap inside same dataset.
    plot_delta_vs_bootstrap(
        ds,
        f"{ds_safe}__delta_vs_bootstrap",
    )

    # K. MultiFair config-level Pareto, EO.
    plot_multifair_config_pareto(
        ds,
        "eo_mean",
        f"{ds_safe}__multifair_config_pareto_eo",
    )

    # L. MultiFair config-level Pareto, DP.
    plot_multifair_config_pareto(
        ds,
        "dp_max",
        f"{ds_safe}__multifair_config_pareto_dp",
    )

    # M. MultiFair configuration sensitivity.
    if not mf_cfg.empty:
        params_here = []

        for p in SENSITIVITY_PRIORITY:
            if p not in mf_cfg.columns:
                continue

            vals = pd.to_numeric(
                mf_cfg.loc[
                    mf_cfg.dataset == ds,
                    p,
                ],
                errors="coerce",
            ).dropna().unique()

            if len(vals) >= 2:
                params_here.append(p)

            if (
                len(params_here)
                >= MAX_SENS_PARAMS_PER_DATASET
            ):
                break

        for p in params_here:
            plot_multifair_sensitivity(
                ds,
                p,
                f"{ds_safe}__multifair_sensitivity_{safe_name(p)}",
            )


# =============================================================================
# ADDITIONAL NONTRIVIAL WITHIN-DATASET COMBINATIONS
# =============================================================================

def plot_fairness_pair(dataset, stem):
    needed = [
        "eo_mean",
        "dp_max",
        "macro_f1",
    ]

    if not all(
        c in wide.columns
        for c in needed
    ):
        return

    q = wide[
        (wide.dataset == dataset)
        & wide.eo_mean.notna()
        & wide.dp_max.notna()
        & wide.macro_f1.notna()
    ].copy()

    if q.empty:
        return

    fig, ax = plt.subplots(
        figsize=(8.4, 6.6),
        constrained_layout=True,
    )

    norm = Normalize(
        q["macro_f1"].min(),
        q["macro_f1"].max()
        if q["macro_f1"].max() > q["macro_f1"].min()
        else q["macro_f1"].min() + 1e-6,
    )

    # Different marker by method, color by F1.
    for method in present_method_order(
        q.method.unique()
    ):
        z = q[
            q.method == method
        ]

        ax.scatter(
            z["dp_max"],
            z["eo_mean"],
            c=z["macro_f1"],
            cmap="viridis",
            norm=norm,
            s=(
                28
                if method == "multifair"
                else 48
            ),
            marker=method_marker(method),
            edgecolor="#222222",
            linewidth=0.35,
            alpha=(
                0.22
                if method == "multifair"
                else 0.62
            ),
        )

    sm = ScalarMappable(
        cmap="viridis",
        norm=norm,
    )
    sm.set_array([])

    cb = fig.colorbar(
        sm,
        ax=ax,
        pad=0.02,
    )
    cb.set_label("Macro-F1 ↑")

    ax.set_xlabel("DP disparity ↓")
    ax.set_ylabel("EO disparity ↓")
    ax.set_title(
        f"Fairness geometry colored by predictive utility\n{dataset}",
        weight="bold",
    )

    clean_axis(ax)

    save_figure(
        fig,
        DIR_TRADEOFF,
        stem,
        f"Fairness geometry: {dataset}",
        "fairness_pair",
    )


def plot_worstgroup_tradeoff(dataset, stem):
    needed = [
        "worst_group_ba",
        "macro_f1",
        "eo_mean",
    ]

    if not all(
        c in wide.columns
        for c in needed
    ):
        return

    q = wide[
        (wide.dataset == dataset)
        & wide.worst_group_ba.notna()
        & wide.macro_f1.notna()
        & wide.eo_mean.notna()
    ].copy()

    if q.empty:
        return

    fig, ax = plt.subplots(
        figsize=(8.4, 6.6),
        constrained_layout=True,
    )

    norm = Normalize(
        q["eo_mean"].min(),
        q["eo_mean"].max()
        if q["eo_mean"].max() > q["eo_mean"].min()
        else q["eo_mean"].min() + 1e-6,
    )

    for method in present_method_order(
        q.method.unique()
    ):
        z = q[
            q.method == method
        ]

        ax.scatter(
            z["macro_f1"],
            z["worst_group_ba"],
            c=z["eo_mean"],
            cmap="viridis_r",
            norm=norm,
            marker=method_marker(method),
            s=(
                28
                if method == "multifair"
                else 50
            ),
            edgecolor="#222222",
            linewidth=0.4,
            alpha=(
                0.22
                if method == "multifair"
                else 0.62
            ),
        )

    sm = ScalarMappable(
        cmap="viridis_r",
        norm=norm,
    )
    sm.set_array([])

    cb = fig.colorbar(
        sm,
        ax=ax,
        pad=0.02,
    )
    cb.set_label("EO disparity ↓")

    ax.set_xlabel("Macro-F1 ↑")
    ax.set_ylabel("Worst-group BA ↑")

    ax.set_title(
        f"Overall utility vs worst-group utility\n{dataset}",
        weight="bold",
    )

    clean_axis(ax)

    save_figure(
        fig,
        DIR_TRADEOFF,
        stem,
        f"Overall vs worst-group utility: {dataset}",
        "worstgroup_tradeoff",
    )


def plot_runtime_fairness(dataset, stem):
    needed = [
        "elapsed_min",
        "eo_mean",
        "macro_f1",
    ]

    if not all(
        c in wide.columns
        for c in needed
    ):
        return

    q = wide[
        (wide.dataset == dataset)
        & wide.elapsed_min.notna()
        & wide.eo_mean.notna()
        & wide.macro_f1.notna()
    ].copy()

    q = q[
        pd.to_numeric(
            q.elapsed_min,
            errors="coerce",
        ) > 0
    ]

    if q.empty:
        return

    fig, ax = plt.subplots(
        figsize=(8.5, 6.6),
        constrained_layout=True,
    )

    norm = Normalize(
        q["macro_f1"].min(),
        q["macro_f1"].max()
        if q["macro_f1"].max() > q["macro_f1"].min()
        else q["macro_f1"].min() + 1e-6,
    )

    for method in present_method_order(
        q.method.unique()
    ):
        z = q[
            q.method == method
        ]

        ax.scatter(
            z["elapsed_min"],
            z["eo_mean"],
            c=z["macro_f1"],
            cmap="viridis",
            norm=norm,
            marker=method_marker(method),
            s=(
                28
                if method == "multifair"
                else 50
            ),
            edgecolor="#222222",
            linewidth=0.4,
            alpha=(
                0.22
                if method == "multifair"
                else 0.62
            ),
        )

    ax.set_xscale("log")

    sm = ScalarMappable(
        cmap="viridis",
        norm=norm,
    )
    sm.set_array([])

    cb = fig.colorbar(
        sm,
        ax=ax,
        pad=0.02,
    )
    cb.set_label("Macro-F1 ↑")

    ax.set_xlabel(
        "Runtime [min, log scale] ↓"
    )
    ax.set_ylabel("EO disparity ↓")

    ax.set_title(
        f"Compute–fairness landscape colored by utility\n{dataset}",
        weight="bold",
    )

    clean_axis(ax)

    save_figure(
        fig,
        DIR_RUNTIME,
        stem,
        f"Compute–fairness landscape: {dataset}",
        "runtime_fairness",
    )


for ds in datasets:
    ds_safe = safe_name(ds)

    plot_fairness_pair(
        ds,
        f"{ds_safe}__dp_vs_eo_colored_f1",
    )

    plot_worstgroup_tradeoff(
        ds,
        f"{ds_safe}__f1_vs_worstgroup_colored_eo",
    )

    plot_runtime_fairness(
        ds,
        f"{ds_safe}__runtime_vs_eo_colored_f1",
    )


# =============================================================================
# FIGURE INDEX + README
# =============================================================================

fig_index = pd.DataFrame(
    FIGURE_RECORDS
)

fig_index.to_csv(
    OUT / "FIGURE_INDEX.csv",
    index=False,
)

kind_counts = (
    fig_index["kind"]
    .value_counts()
    .to_dict()
    if len(fig_index)
    else {}
)

readme = []

readme.append(
    "# MultiFairGDT — FIGURES"
)
readme.append("")
readme.append(
    "All figures preserve dataset identity. "
    "There is no averaging across datasets anywhere in this figure suite."
)
readme.append("")
readme.append(
    "Aggregation levels represented:"
)
readme.append("")
readme.append(
    "1. Raw successful job-level observations."
)
readme.append(
    "2. Seed-balanced summaries computed separately inside each dataset."
)
readme.append(
    "3. Method × dataset matrices retaining every dataset as a separate column."
)
readme.append(
    "4. MultiFair configuration-level summaries across seeds within each dataset."
)
readme.append(
    "5. MultiFair Pareto envelopes within each dataset."
)
readme.append(
    "6. Hyperparameter marginal sensitivity within each dataset."
)
readme.append(
    "7. Group-support sensitivity at s=5, s=10, and all evaluable groups."
)
readme.append("")
readme.append(
    f"Unique figure concepts generated: **{len(fig_index)}**"
)
readme.append("")
readme.append(
    "Every figure is exported as both 300-dpi PNG and vector PDF."
)
readme.append("")
readme.append(
    "## Figure categories"
)
readme.append("")

for k, v in sorted(
    kind_counts.items()
):
    readme.append(
        f"- `{k}`: {v}"
    )

readme.append("")
readme.append(
    "## Derived data"
)
readme.append("")
readme.append(
    "- `_data/seed_balanced_metrics.csv`"
)
readme.append(
    "- `_data/method_dataset_seedbalanced_stats.csv`"
)
readme.append(
    "- `_data/multifair_configuration_means.csv`"
)
readme.append("")
readme.append(
    "See `FIGURE_INDEX.csv` for the title and paths of every figure."
)

(OUT / "README.md").write_text(
    "\n".join(readme) + "\n",
    encoding="utf-8",
)


# =============================================================================
# FINAL PRINT
# =============================================================================

print()
print("=" * 88)
print("FIGURE BUILD COMPLETE")
print("=" * 88)
print("folder          :", OUT)
print("figure concepts :", len(fig_index))
print("PNG files       :", len(list(OUT.rglob("*.png"))))
print("PDF files       :", len(list(OUT.rglob("*.pdf"))))
print()

print("=== BY TYPE ===")
for k, v in sorted(
    kind_counts.items()
):
    print(f"{k:30s} {v:4d}")

print()
print("=== BY DIRECTORY ===")
for d in [
    DIR_OVERVIEW,
    DIR_DATASET,
    DIR_TRADEOFF,
    DIR_SUPPORT,
    DIR_MULTIFAIR,
    DIR_RUNTIME,
]:
    print(
        f"{d.name:24s} "
        f"{len(list(d.glob('*.png'))):4d} figures"
    )

print()
print("INDEX :", OUT / "FIGURE_INDEX.csv")
print("README:", OUT / "README.md")
print("=" * 88)
