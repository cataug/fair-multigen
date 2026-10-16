from pathlib import Path
import math
import re

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import (
    Normalize,
    LogNorm,
    LinearSegmentedColormap,
    to_rgb,
)
from matplotlib.lines import Line2D


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path("/home/tahiti/MultiFairGDT")
STATS = ROOT / "STATS"
OUT = ROOT / "FIGURES_MAIN"

OUT.mkdir(parents=True, exist_ok=True)

JOBS_PATH = STATS / "00_jobs_harmonized.csv"
METRICS_PATH = STATS / "16_metrics_long.csv"
COVERAGE_PATH = STATS / "04_method_dataset_summary.csv"

for p in [JOBS_PATH, METRICS_PATH, COVERAGE_PATH]:
    if not p.exists():
        raise SystemExit(f"Missing required file: {p}")


# =============================================================================
# STYLE — Fractional Unlearn family
# =============================================================================

DPI = 300

mpl.rcParams.update({
    "font.family": "DejaVu Sans",

    "font.size": 16,
    "axes.titlesize": 21,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 24,

    "figure.facecolor": "white",
    "savefig.facecolor": "white",

    "axes.facecolor": "#F2F3F5",
    "axes.edgecolor": "#909090",
    "axes.linewidth": 1.0,

    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.5,
    "grid.alpha": 1.0,

    "axes.spines.top": False,
    "axes.spines.right": False,

    "lines.linewidth": 3.0,
    "lines.markersize": 9,

    "legend.frameon": True,
    "legend.framealpha": 0.94,
    "legend.facecolor": "white",
    "legend.edgecolor": "#D0D0D0",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# Pastel palette used in the earlier figure family.
PALETTE = {
    "blue":   "#5F87C8",
    "cyan":   "#5AAFBE",
    "orange": "#DCA05A",
    "green":  "#6EAA78",
    "violet": "#967DB9",
    "rose":   "#CD8791",
    "yellow": "#E1C36E",
    "gray":   "#969696",
    "dark":   "#444444",
}


METHOD_STYLE = {
    "bootstrap": {
        "color": PALETTE["blue"],
        "marker": "s",
        "label": "Bootstrap",
    },
    "multifair": {
        "color": PALETTE["orange"],
        "marker": "o",
        "label": "MultiFair",
    },
    "singlefair": {
        "color": PALETTE["yellow"],
        "marker": ">",
        "label": "SingleFair",
    },
    "gdt": {
        "color": "#B98D59",
        "marker": "<",
        "label": "GDT",
    },
    "real": {
        "color": "#555555",
        "marker": "*",
        "label": "Real",
    },
    "gaussian_copula": {
        "color": PALETTE["gray"],
        "marker": "D",
        "label": "Gaussian Copula",
    },
    "ctgan": {
        "color": PALETTE["green"],
        "marker": "^",
        "label": "CTGAN",
    },
    "tvae": {
        "color": PALETTE["cyan"],
        "marker": "v",
        "label": "TVAE",
    },
    "tabddpm": {
        "color": PALETTE["violet"],
        "marker": "P",
        "label": "TabDDPM",
    },
    "decaf": {
        "color": PALETTE["rose"],
        "marker": "X",
        "label": "DECAF",
    },
    "tabfairgan": {
        "color": "#D999BA",
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


DATASET_COLORS = [
    PALETTE["blue"],
    PALETTE["orange"],
    PALETTE["green"],
    PALETTE["violet"],
    PALETTE["rose"],
    PALETTE["cyan"],
    PALETTE["yellow"],
    "#A58F75",
    "#7E9A9A",
]


METRIC_LABEL = {
    "macro_f1": "Macro-F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",

    "dp_max": "DP disparity",
    "eo_mean": "EO disparity",
    "aod_mean": "AOD disparity",

    "worst_group_ba": "Worst-group BA",
    "worst_group_fpr": "Worst-group FPR",
    "worst_group_tpr": "Worst-group TPR",

    "dp_max_s5": "DP (support ≥5)",
    "dp_max_s10": "DP (support ≥10)",

    "eo_mean_s5": "EO (support ≥5)",
    "eo_mean_s10": "EO (support ≥10)",

    "aod_mean_s5": "AOD (support ≥5)",
    "aod_mean_s10": "AOD (support ≥10)",

    "worst_group_ba_s5": "Worst-group BA (support ≥5)",
    "worst_group_ba_s10": "Worst-group BA (support ≥10)",
}


# =============================================================================
# HELPERS
# =============================================================================

FIGURES = []


def method_label(m):
    return METHOD_STYLE.get(m, {}).get(
        "label",
        str(m).replace("_", " ").title(),
    )


def method_color(m):
    return METHOD_STYLE.get(m, {}).get(
        "color",
        "#666666",
    )


def method_marker(m):
    return METHOD_STYLE.get(m, {}).get(
        "marker",
        "o",
    )


def metric_label(x):
    return METRIC_LABEL.get(
        x,
        str(x).replace("_", " ").title(),
    )


def ordered_methods(values):
    values = list(dict.fromkeys(values))

    result = [
        x for x in METHOD_ORDER
        if x in values
    ]

    result.extend(
        sorted(
            x for x in values
            if x not in result
        )
    )

    return result


def safe_name(x):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(x).lower(),
    ).strip("_")


def clean_axis(ax):
    ax.set_axisbelow(True)

    ax.tick_params(
        colors="#252525",
        width=1.0,
    )

    ax.xaxis.label.set_color("#222222")
    ax.yaxis.label.set_color("#222222")
    ax.title.set_color("#151515")


def bbox_kw(alpha=0.88):
    return dict(
        boxstyle="round,pad=0.28",
        facecolor="white",
        edgecolor="#D0D0D0",
        linewidth=0.7,
        alpha=alpha,
    )


def pastel_cmap(color):
    rgb = np.array(to_rgb(color))

    white = np.array([1.0, 1.0, 1.0])

    very_light = white * 0.85 + rgb * 0.15
    light = white * 0.58 + rgb * 0.42

    return LinearSegmentedColormap.from_list(
        "pastel",
        [
            tuple(very_light),
            tuple(light),
            tuple(rgb),
        ],
    )


def diverging_cmap():
    return LinearSegmentedColormap.from_list(
        "pastel_div",
        [
            PALETTE["blue"],
            "#EEF0F1",
            PALETTE["orange"],
        ],
    )


def gradient_band(
    ax,
    x,
    low,
    high,
    color,
    layers=10,
    max_alpha=0.14,
):
    """
    Smooth-looking nested transparent interval band.
    Highest opacity is near the center.
    """
    x = np.asarray(x, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)

    mid = (low + high) / 2.0

    for i in range(layers, 0, -1):
        f = i / layers

        lo = mid - (mid - low) * f
        hi = mid + (high - mid) * f

        alpha = (
            max_alpha
            * (1.0 - f * 0.55)
            / layers
            * 4.5
        )

        ax.fill_between(
            x,
            lo,
            hi,
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=1,
        )


def save(fig, num, stem, title):
    stem = f"fig_{num:02d}_{stem}"

    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"

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

    FIGURES.append({
        "figure": num,
        "stem": stem,
        "title": title,
        "png": png.name,
        "pdf": pdf.name,
    })

    print(f"[{num:02d}] {stem}")


def annotation(
    ax,
    x,
    y,
    text,
    dx=6,
    dy=5,
    fontsize=11,
):
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=fontsize,
        color="#222222",
        bbox=bbox_kw(0.82),
        zorder=15,
    )


def pareto_front(df, x, y):
    """
    x lower better, y higher better.
    """
    q = df[
        df[x].notna()
        & df[y].notna()
    ].copy()

    if q.empty:
        return q

    q = q.sort_values(
        [x, y],
        ascending=[True, False],
    )

    keep = []
    best_y = -np.inf

    for idx, r in q.iterrows():
        val = float(r[y])

        if val > best_y + 1e-12:
            keep.append(idx)
            best_y = val

    return q.loc[keep].sort_values(x)


# =============================================================================
# LOAD
# =============================================================================

jobs = pd.read_csv(
    JOBS_PATH,
    keep_default_na=False,
)

metrics = pd.read_csv(
    METRICS_PATH,
    keep_default_na=False,
)

coverage = pd.read_csv(
    COVERAGE_PATH,
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


bad = (
    metrics.metric_clean.eq("")
    | metrics.metric_clean.str.startswith(
        "Unnamed"
    )
)

metrics.loc[
    bad,
    "metric_clean"
] = (
    metrics.loc[
        bad,
        "metric_id"
    ]
    .astype(str)
    .str.replace(
        "metrics.csv::",
        "",
        regex=False,
    )
)


# Avoid exactly the collision that killed the previous script.
RESERVED = {
    "job_id",
    "method",
    "dataset",
    "seed",
    "status",
    "source",
}

collisions = sorted(
    set(metrics.metric_clean)
    & RESERVED
)

if collisions:
    print(
        "[metrics] structural pseudo-metrics removed:",
        ", ".join(collisions),
    )

metrics = metrics[
    ~metrics.metric_clean.isin(RESERVED)
].copy()


success_ids = set(
    jobs.loc[
        jobs.status == "success",
        "job_id",
    ]
)

metrics = metrics[
    metrics.job_id.isin(success_ids)
].copy()


# =============================================================================
# WIDE JOB TABLE
# =============================================================================

wide = (
    metrics
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


# Runtime from status.
job_runtime = jobs[
    [
        c for c in [
            "job_id",
            "elapsed_min",
            "elapsed_hours",
        ]
        if c in jobs.columns
    ]
].drop_duplicates("job_id")

wide = wide.merge(
    job_runtime,
    on="job_id",
    how="left",
)


# Force numeric where appropriate.
for col in wide.columns:
    if col in {
        "job_id",
        "method",
        "dataset",
    }:
        continue

    try:
        wide[col] = pd.to_numeric(
            wide[col],
            errors="ignore",
        )
    except Exception:
        pass


datasets = sorted(
    wide.dataset.astype(str).unique()
)

dataset_color = {
    ds: DATASET_COLORS[i % len(DATASET_COLORS)]
    for i, ds in enumerate(datasets)
}


# =============================================================================
# DIFFERENT AGGREGATION LEVELS
# =============================================================================

CORE_METRICS = [
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

        "aod_mean_s5",
        "aod_mean_s10",

        "worst_group_ba_s5",
        "worst_group_ba_s10",

        "elapsed_min",
    ]
    if c in wide.columns
]


# -------------------------------------------------------------------------
# Level A:
# equal weight per seed WITHIN each dataset.
#
# MultiFair may contain many configurations in one seed. For these
# high-level seed-balanced summaries we first take within-seed median.
# Raw configuration-level clouds remain available separately.
# -------------------------------------------------------------------------

seed_balanced = (
    wide.groupby(
        [
            "dataset",
            "method",
            "seed",
        ],
        dropna=False,
    )[CORE_METRICS]
    .median(numeric_only=True)
    .reset_index()
)


# -------------------------------------------------------------------------
# Level B:
# method means over seed-balanced results, still WITHIN DATASET.
# -------------------------------------------------------------------------

method_dataset = (
    seed_balanced
    .groupby(
        [
            "dataset",
            "method",
        ],
        dropna=False,
    )[CORE_METRICS]
    .agg(
        ["mean", "std", "median"]
    )
)

method_dataset.columns = [
    f"{a}_{b}"
    for a, b in method_dataset.columns
]

method_dataset = (
    method_dataset
    .reset_index()
)


# -------------------------------------------------------------------------
# Level C:
# MultiFair configuration means over seeds WITHIN DATASET.
# -------------------------------------------------------------------------

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
]

config_cols = []

for col in CONFIG_CANDIDATES:
    if col not in wide.columns:
        continue

    z = pd.to_numeric(
        wide.loc[
            wide.method == "multifair",
            col,
        ],
        errors="coerce",
    )

    if z.notna().any():
        wide[col] = pd.to_numeric(
            wide[col],
            errors="coerce",
        )

        config_cols.append(col)


mf_raw = wide[
    wide.method == "multifair"
].copy()


mf_cfg_rows = []

if len(mf_raw) and config_cols:
    grouped = mf_raw.groupby(
        [
            "dataset",
            *config_cols,
        ],
        dropna=False,
    )

    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        row = dict(
            zip(
                [
                    "dataset",
                    *config_cols,
                ],
                key,
            )
        )

        row["n_seeds"] = g.seed.nunique()

        for metric in CORE_METRICS:
            if metric not in g.columns:
                continue

            vals = pd.to_numeric(
                g[metric],
                errors="coerce",
            ).dropna()

            if not len(vals):
                continue

            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_median"] = vals.median()
            row[f"{metric}_std"] = (
                vals.std(ddof=1)
                if len(vals) > 1
                else np.nan
            )

        mf_cfg_rows.append(row)


mf_cfg = pd.DataFrame(
    mf_cfg_rows
)


# Save derived tables in same folder, no subfolders.
seed_balanced.to_csv(
    OUT / "DATA_seed_balanced.csv",
    index=False,
)

method_dataset.to_csv(
    OUT / "DATA_method_dataset.csv",
    index=False,
)

mf_cfg.to_csv(
    OUT / "DATA_multifair_configs.csv",
    index=False,
)


# =============================================================================
# MATRIX HELPERS
# =============================================================================

def matrix_from_seed(metric):
    if metric not in seed_balanced.columns:
        return pd.DataFrame()

    q = (
        seed_balanced
        .groupby(
            [
                "method",
                "dataset",
            ]
        )[metric]
        .mean()
        .reset_index()
    )

    return q.pivot(
        index="method",
        columns="dataset",
        values=metric,
    )


def draw_heatmap(
    matrix,
    title,
    num,
    stem,
    value_format=".3f",
    cmap=None,
    log=False,
    center_zero=False,
    cbar_label=None,
):
    if matrix.empty:
        return

    methods = ordered_methods(
        matrix.index
    )

    matrix = matrix.reindex(
        methods
    )

    matrix = matrix.reindex(
        [
            ds for ds in datasets
            if ds in matrix.columns
        ],
        axis=1,
    )

    arr = matrix.to_numpy(
        dtype=float
    )

    finite = arr[
        np.isfinite(arr)
    ]

    if not len(finite):
        return

    masked = np.ma.masked_invalid(
        arr
    )

    if cmap is None:
        cmap = pastel_cmap(
            PALETTE["blue"]
        )

    if center_zero:
        vmax = max(
            abs(np.nanmin(finite)),
            abs(np.nanmax(finite)),
            1e-8,
        )

        norm = Normalize(
            -vmax,
            vmax,
        )

    elif log:
        pos = finite[
            finite > 0
        ]

        if len(pos):
            norm = LogNorm(
                vmin=max(
                    pos.min(),
                    1e-4,
                ),
                vmax=pos.max(),
            )
        else:
            norm = Normalize(
                finite.min(),
                finite.max() + 1e-8,
            )

    else:
        lo = finite.min()
        hi = finite.max()

        if hi <= lo:
            hi = lo + 1e-8

        norm = Normalize(
            lo,
            hi,
        )

    fig, ax = plt.subplots(
        figsize=(16, 7.4),
        constrained_layout=True,
    )

    im = ax.imshow(
        masked,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(
        np.arange(
            len(matrix.columns)
        )
    )

    ax.set_xticklabels(
        [
            x.replace(
                "_",
                "\n",
            )
            for x in matrix.columns
        ],
        fontsize=13,
    )

    ax.set_yticks(
        np.arange(
            len(matrix.index)
        )
    )

    ax.set_yticklabels(
        [
            method_label(x)
            for x in matrix.index
        ],
        fontsize=14,
    )

    ax.grid(False)

    # Clean white separation between cells.
    ax.set_xticks(
        np.arange(
            -0.5,
            len(matrix.columns),
            1,
        ),
        minor=True,
    )

    ax.set_yticks(
        np.arange(
            -0.5,
            len(matrix.index),
            1,
        ),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=3.0,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # boxed values
    for i in range(
        arr.shape[0]
    ):
        for j in range(
            arr.shape[1]
        ):
            val = arr[i, j]

            if not np.isfinite(val):
                continue

            ax.text(
                j,
                i,
                format(
                    val,
                    value_format,
                ),
                ha="center",
                va="center",
                fontsize=11.5,
                color="#202020",
                bbox=dict(
                    boxstyle="round,pad=0.20",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.72,
                ),
            )

    cb = fig.colorbar(
        im,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )

    if cbar_label:
        cb.set_label(
            cbar_label,
            fontsize=15,
        )

    cb.ax.tick_params(
        labelsize=12,
    )

    ax.set_title(
        title,
        weight="bold",
        pad=17,
    )

    clean_axis(ax)

    save(
        fig,
        num,
        stem,
        title,
    )


# =============================================================================
# FIGURES 1–9: MAIN MATRICES
# =============================================================================

coverage_numeric = coverage.copy()

for c in [
    "total",
    "success",
    "failed",
    "running",
]:
    if c in coverage_numeric.columns:
        coverage_numeric[c] = pd.to_numeric(
            coverage_numeric[c],
            errors="coerce",
        )


coverage_numeric[
    "success_fraction"
] = (
    coverage_numeric["success"]
    / coverage_numeric[
        "total"
    ].replace(
        0,
        np.nan,
    )
)


# 01
draw_heatmap(
    coverage_numeric.pivot(
        index="method",
        columns="dataset",
        values="success_fraction",
    ),
    "Experiment completion by method and dataset",
    1,
    "completion",
    value_format=".2f",
    cmap=pastel_cmap(
        PALETTE["green"]
    ),
    cbar_label="Success fraction",
)


# Runtime from successful status.
runtime_jobs = jobs[
    jobs.status == "success"
].copy()

runtime_jobs[
    "elapsed_min"
] = pd.to_numeric(
    runtime_jobs.elapsed_min,
    errors="coerce",
)


# 02
r = (
    runtime_jobs
    .groupby(
        [
            "method",
            "dataset",
        ]
    )["elapsed_min"]
    .median()
    .reset_index()
)

draw_heatmap(
    r.pivot(
        index="method",
        columns="dataset",
        values="elapsed_min",
    ),
    "Median runtime per successful job",
    2,
    "runtime_median",
    value_format=".1f",
    cmap=pastel_cmap(
        PALETTE["violet"]
    ),
    log=True,
    cbar_label="Minutes, logarithmic scale",
)


# 03
r = (
    runtime_jobs
    .groupby(
        [
            "method",
            "dataset",
        ]
    )["elapsed_min"]
    .quantile(
        0.90
    )
    .reset_index()
)

draw_heatmap(
    r.pivot(
        index="method",
        columns="dataset",
        values="elapsed_min",
    ),
    "90th-percentile runtime per successful job",
    3,
    "runtime_p90",
    value_format=".1f",
    cmap=pastel_cmap(
        PALETTE["rose"]
    ),
    log=True,
    cbar_label="Minutes, logarithmic scale",
)


# 04
metric_coverage = (
    wide.groupby(
        [
            "method",
            "dataset",
        ]
    )
    .job_id
    .nunique()
    .reset_index(
        name="jobs"
    )
)

draw_heatmap(
    metric_coverage.pivot(
        index="method",
        columns="dataset",
        values="jobs",
    ),
    "Successful jobs represented in harmonized metrics",
    4,
    "metric_coverage",
    value_format=".0f",
    cmap=pastel_cmap(
        PALETTE["cyan"]
    ),
    cbar_label="Jobs",
)


# 05
draw_heatmap(
    matrix_from_seed(
        "macro_f1"
    ),
    "Macro-F1 — seed-balanced within each dataset",
    5,
    "macro_f1",
    value_format=".3f",
    cmap=pastel_cmap(
        PALETTE["blue"]
    ),
    cbar_label="Macro-F1 ↑",
)


# 06
draw_heatmap(
    matrix_from_seed(
        "eo_mean"
    ),
    "Equalized-odds disparity — seed-balanced within each dataset",
    6,
    "eo_disparity",
    value_format=".3f",
    cmap=pastel_cmap(
        PALETTE["rose"]
    ),
    cbar_label="EO disparity ↓",
)


# 07
draw_heatmap(
    matrix_from_seed(
        "worst_group_ba"
    ),
    "Worst-group balanced accuracy — seed-balanced within each dataset",
    7,
    "worst_group_ba",
    value_format=".3f",
    cmap=pastel_cmap(
        PALETTE["green"]
    ),
    cbar_label="Worst-group BA ↑",
)


# Bootstrap-relative matrices, always per dataset.
if (
    "macro_f1"
    in seed_balanced.columns
):
    mm = (
        seed_balanced
        .groupby(
            [
                "method",
                "dataset",
            ]
        )["macro_f1"]
        .mean()
        .reset_index()
    )

    bb = (
        mm[
            mm.method
            == "bootstrap"
        ][
            [
                "dataset",
                "macro_f1",
            ]
        ]
        .rename(
            columns={
                "macro_f1":
                "bootstrap"
            }
        )
    )

    mm = mm.merge(
        bb,
        on="dataset",
        how="left",
    )

    mm["delta"] = (
        mm.macro_f1
        - mm.bootstrap
    )

    # 08
    draw_heatmap(
        mm.pivot(
            index="method",
            columns="dataset",
            values="delta",
        ),
        "Macro-F1 difference relative to Bootstrap",
        8,
        "delta_f1_vs_bootstrap",
        value_format="+.3f",
        cmap=diverging_cmap(),
        center_zero=True,
        cbar_label="Δ Macro-F1",
    )


if (
    "eo_mean"
    in seed_balanced.columns
):
    mm = (
        seed_balanced
        .groupby(
            [
                "method",
                "dataset",
            ]
        )["eo_mean"]
        .mean()
        .reset_index()
    )

    bb = (
        mm[
            mm.method
            == "bootstrap"
        ][
            [
                "dataset",
                "eo_mean",
            ]
        ]
        .rename(
            columns={
                "eo_mean":
                "bootstrap"
            }
        )
    )

    mm = mm.merge(
        bb,
        on="dataset",
        how="left",
    )

    mm["delta"] = (
        mm.eo_mean
        - mm.bootstrap
    )

    # 09
    draw_heatmap(
        mm.pivot(
            index="method",
            columns="dataset",
            values="delta",
        ),
        "EO-disparity difference relative to Bootstrap",
        9,
        "delta_eo_vs_bootstrap",
        value_format="+.3f",
        cmap=diverging_cmap(),
        center_zero=True,
        cbar_label="Δ EO; negative is lower disparity",
    )


# =============================================================================
# DATASET SUMMARY FIGURES 10–18
# =============================================================================

def grouped_bars(
    ax,
    dataset,
    metrics_list,
    ylabel,
    colors,
):
    q = seed_balanced[
        seed_balanced.dataset
        == dataset
    ]

    methods = ordered_methods(
        q.method.unique()
    )

    metrics_here = [
        m for m in metrics_list
        if (
            m in q.columns
            and q[m].notna().any()
        )
    ]

    if not methods or not metrics_here:
        ax.axis("off")
        return

    x = np.arange(
        len(methods)
    )

    width = min(
        0.22,
        0.80
        / len(metrics_here),
    )

    offsets = (
        np.arange(
            len(metrics_here)
        )
        - (
            len(metrics_here)
            - 1
        ) / 2
    ) * width

    for j, metric in enumerate(
        metrics_here
    ):
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

        bars = ax.bar(
            x + offsets[j],
            means,
            width=width * 0.90,
            yerr=stds,
            capsize=3.0,
            color=colors[j],
            alpha=0.82,
            edgecolor="#333333",
            linewidth=0.7,
            label=metric_label(
                metric
            ),
            zorder=3,
        )

        # translucent shadow/base
        for b in bars:
            b.set_path_effects([])

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            method_label(m)
            for m in methods
        ],
        rotation=24,
        ha="right",
        fontsize=11,
    )

    ax.set_ylabel(
        ylabel
    )

    ax.legend(
        ncol=1,
        loc="best",
        fontsize=10.5,
    )

    clean_axis(ax)


def tradeoff_panel(
    ax,
    dataset,
):
    needed = [
        "macro_f1",
        "eo_mean",
    ]

    if not all(
        x in wide.columns
        for x in needed
    ):
        ax.axis("off")
        return

    q = wide[
        (wide.dataset == dataset)
        & wide.macro_f1.notna()
        & wide.eo_mean.notna()
    ].copy()

    if q.empty:
        ax.axis("off")
        return

    # Raw configuration/job clouds.
    for method in ordered_methods(
        q.method.unique()
    ):
        z = q[
            q.method == method
        ]

        alpha = (
            0.13
            if method == "multifair"
            else 0.30
        )

        size = (
            30
            if method == "multifair"
            else 45
        )

        ax.scatter(
            z.eo_mean,
            z.macro_f1,
            s=size,
            marker=method_marker(
                method
            ),
            color=method_color(
                method
            ),
            alpha=alpha,
            edgecolor="none",
            zorder=2,
        )

    # MultiFair configuration Pareto line.
    if (
        not mf_cfg.empty
        and "eo_mean_mean"
        in mf_cfg.columns
        and "macro_f1_mean"
        in mf_cfg.columns
    ):
        z = mf_cfg[
            mf_cfg.dataset
            == dataset
        ].copy()

        front = pareto_front(
            z,
            "eo_mean_mean",
            "macro_f1_mean",
        )

        if len(front):
            ax.plot(
                front.eo_mean_mean,
                front.macro_f1_mean,
                color=PALETTE[
                    "orange"
                ],
                linewidth=3.2,
                marker="o",
                markersize=7,
                alpha=0.95,
                zorder=5,
            )

    # Seed-balanced method centroids.
    q2 = seed_balanced[
        seed_balanced.dataset
        == dataset
    ]

    for method in ordered_methods(
        q2.method.unique()
    ):
        z = q2[
            q2.method == method
        ]

        xx = pd.to_numeric(
            z.eo_mean,
            errors="coerce",
        ).mean()

        yy = pd.to_numeric(
            z.macro_f1,
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
            s=170,
            color=method_color(
                method
            ),
            marker=method_marker(
                method
            ),
            edgecolor="#222222",
            linewidth=1.1,
            zorder=8,
        )

        annotation(
            ax,
            xx,
            yy,
            method_label(
                method
            ),
            fontsize=9.5,
        )

    ax.set_xlabel(
        "EO disparity ↓"
    )

    ax.set_ylabel(
        "Macro-F1 ↑"
    )

    ax.set_title(
        "Utility–fairness landscape"
    )

    clean_axis(ax)


def support_panel(
    ax,
    dataset,
):
    needed = [
        "eo_mean_s5",
        "eo_mean_s10",
        "eo_mean",
    ]

    if not all(
        x in seed_balanced.columns
        for x in needed
    ):
        ax.axis("off")
        return

    q = seed_balanced[
        seed_balanced.dataset
        == dataset
    ]

    x = np.arange(3)

    labels = [
        "support ≥5",
        "support ≥10",
        "all groups",
    ]

    for method in ordered_methods(
        q.method.unique()
    ):
        z = q[
            q.method == method
        ]

        vals = []

        for col in needed:
            a = pd.to_numeric(
                z[col],
                errors="coerce",
            ).dropna()

            vals.append(
                a.mean()
                if len(a)
                else np.nan
            )

        if not np.isfinite(
            vals
        ).any():
            continue

        ax.plot(
            x,
            vals,
            color=method_color(
                method
            ),
            marker=method_marker(
                method
            ),
            linewidth=2.5,
            alpha=0.90,
            label=method_label(
                method
            ),
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=15,
        ha="right",
        fontsize=11,
    )

    ax.set_ylabel(
        "EO disparity ↓"
    )

    ax.set_title(
        "Aggregation/support sensitivity"
    )

    clean_axis(ax)


def dataset_summary(
    dataset,
    number,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(17, 12),
        constrained_layout=True,
    )

    grouped_bars(
        axes[0, 0],
        dataset,
        [
            "macro_f1",
            "roc_auc",
            "pr_auc",
        ],
        "Predictive score ↑",
        [
            PALETTE["blue"],
            PALETTE["cyan"],
            PALETTE["green"],
        ],
    )

    axes[0, 0].set_title(
        "Predictive utility"
    )

    grouped_bars(
        axes[0, 1],
        dataset,
        [
            "dp_max",
            "eo_mean",
            "aod_mean",
        ],
        "Disparity ↓",
        [
            PALETTE["rose"],
            PALETTE["violet"],
            PALETTE["yellow"],
        ],
    )

    axes[0, 1].set_title(
        "Fairness disparities"
    )

    tradeoff_panel(
        axes[1, 0],
        dataset,
    )

    support_panel(
        axes[1, 1],
        dataset,
    )

    fig.suptitle(
        dataset.replace(
            "_",
            " "
        ).title(),
        weight="bold",
        y=1.02,
    )

    save(
        fig,
        number,
        f"dataset_{safe_name(dataset)}",
        f"Dataset summary: {dataset}",
    )


for i, ds in enumerate(
    datasets,
    start=10,
):
    dataset_summary(
        ds,
        i,
    )


# =============================================================================
# FIGURES 19–21: MULTIFAIR HYPERPARAMETER SENSITIVITY
# =============================================================================

def multifair_sensitivity(
    param,
    number,
):
    if (
        param not in mf_raw.columns
        or mf_raw.empty
    ):
        print(
            f"[skip] sensitivity {param}"
        )
        return

    mf_raw[param] = pd.to_numeric(
        mf_raw[param],
        errors="coerce",
    )

    if (
        mf_raw[param]
        .dropna()
        .nunique()
        < 2
    ):
        print(
            f"[skip] sensitivity {param}: <2 levels"
        )
        return

    panels = [
        (
            "macro_f1",
            "Macro-F1 ↑",
        ),
        (
            "eo_mean",
            "EO disparity ↓",
        ),
        (
            "worst_group_ba",
            "Worst-group BA ↑",
        ),
        (
            "elapsed_min",
            "Runtime [min] ↓",
        ),
    ]

    panels = [
        x for x in panels
        if x[0] in mf_raw.columns
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(17, 11),
        constrained_layout=True,
    )

    axes = axes.ravel()

    for k, (
        metric,
        ylabel,
    ) in enumerate(panels):
        ax = axes[k]

        for ds in datasets:
            q = mf_raw[
                (mf_raw.dataset == ds)
                & mf_raw[param].notna()
                & mf_raw[metric].notna()
            ].copy()

            if q.empty:
                continue

            rows = []

            for val, g in q.groupby(
                param
            ):
                z = pd.to_numeric(
                    g[metric],
                    errors="coerce",
                ).dropna()

                if not len(z):
                    continue

                rows.append({
                    "x": float(val),
                    "median": z.median(),
                    "q25": z.quantile(
                        0.25
                    ),
                    "q75": z.quantile(
                        0.75
                    ),
                })

            a = pd.DataFrame(
                rows
            ).sort_values(
                "x"
            )

            if len(a) < 2:
                continue

            color = dataset_color[
                ds
            ]

            gradient_band(
                ax,
                a.x,
                a.q25,
                a.q75,
                color,
            )

            ax.plot(
                a.x,
                a["median"],
                color=color,
                marker="o",
                linewidth=2.8,
                markersize=7,
                label=ds.replace(
                    "_",
                    " "
                ),
                zorder=5,
            )

        ax.set_xlabel(
            param.replace(
                "_",
                " "
            )
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            metric_label(
                metric
            )
            if metric
            != "elapsed_min"
            else "Runtime"
        )

        if metric == "elapsed_min":
            vals = pd.to_numeric(
                mf_raw.elapsed_min,
                errors="coerce",
            )

            if (
                vals.notna().any()
                and (
                    vals.dropna()
                    > 0
                ).all()
            ):
                ax.set_yscale(
                    "log"
                )

        clean_axis(ax)

    for k in range(
        len(panels),
        len(axes),
    ):
        axes[k].axis(
            "off"
        )

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                1.045,
            ),
            ncol=3,
            fontsize=11,
        )

    fig.suptitle(
        f"MultiFair sensitivity to {param.replace('_', ' ')}",
        weight="bold",
        y=1.09,
    )

    save(
        fig,
        number,
        f"multifair_sensitivity_{safe_name(param)}",
        f"MultiFair sensitivity: {param}",
    )


multifair_sensitivity(
    "lambda_fair",
    19,
)

multifair_sensitivity(
    "k",
    20,
)

multifair_sensitivity(
    "epsilon",
    21,
)


# =============================================================================
# FIGURE 22: MULTIFAIR PARETO ATLAS — DATASETS REMAIN SEPARATE PANELS
# =============================================================================

def pareto_atlas(number=22):
    if (
        mf_cfg.empty
        or "macro_f1_mean"
        not in mf_cfg.columns
        or "eo_mean_mean"
        not in mf_cfg.columns
    ):
        return

    n = len(datasets)

    ncols = 3
    nrows = math.ceil(
        n / ncols
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            17,
            5.3 * nrows,
        ),
        constrained_layout=True,
        squeeze=False,
    )

    axes = axes.ravel()

    for i, ds in enumerate(
        datasets
    ):
        ax = axes[i]

        q = mf_cfg[
            mf_cfg.dataset
            == ds
        ].copy()

        if q.empty:
            ax.axis("off")
            continue

        # raw configuration means
        ax.scatter(
            q.eo_mean_mean,
            q.macro_f1_mean,
            s=52,
            color=PALETTE[
                "orange"
            ],
            alpha=0.20,
            edgecolor="none",
            zorder=2,
        )

        front = pareto_front(
            q,
            "eo_mean_mean",
            "macro_f1_mean",
        )

        if len(front):
            ax.plot(
                front.eo_mean_mean,
                front.macro_f1_mean,
                color=PALETTE[
                    "orange"
                ],
                linewidth=3.2,
                marker="o",
                markersize=7,
                alpha=0.95,
                zorder=6,
            )

        # baseline centroids
        b = seed_balanced[
            seed_balanced.dataset
            == ds
        ]

        for method in ordered_methods(
            b.method.unique()
        ):
            if method == "multifair":
                continue

            z = b[
                b.method == method
            ]

            xx = pd.to_numeric(
                z.eo_mean,
                errors="coerce",
            ).mean()

            yy = pd.to_numeric(
                z.macro_f1,
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
                s=130,
                marker=method_marker(
                    method
                ),
                color=method_color(
                    method
                ),
                edgecolor="#222222",
                linewidth=0.9,
                zorder=7,
            )

            annotation(
                ax,
                xx,
                yy,
                method_label(
                    method
                ),
                fontsize=8.5,
                dx=4,
                dy=4,
            )

        ax.set_title(
            ds.replace(
                "_",
                " "
            ).title(),
            fontsize=17,
        )

        ax.set_xlabel(
            "EO disparity ↓",
            fontsize=14,
        )

        ax.set_ylabel(
            "Macro-F1 ↑",
            fontsize=14,
        )

        clean_axis(ax)

    for i in range(
        len(datasets),
        len(axes),
    ):
        axes[i].axis(
            "off"
        )

    fig.suptitle(
        "MultiFair configuration Pareto landscapes",
        weight="bold",
        y=1.015,
    )

    save(
        fig,
        number,
        "multifair_pareto_atlas",
        "MultiFair Pareto atlas",
    )


pareto_atlas()


# =============================================================================
# FIGURE 23: COMPUTE–UTILITY ATLAS
# =============================================================================

def compute_atlas(number=23):
    n = len(datasets)

    ncols = 3
    nrows = math.ceil(
        n / ncols
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            17,
            5.2 * nrows,
        ),
        constrained_layout=True,
        squeeze=False,
    )

    axes = axes.ravel()

    for i, ds in enumerate(
        datasets
    ):
        ax = axes[i]

        q = seed_balanced[
            seed_balanced.dataset
            == ds
        ]

        runtime_ds = runtime_jobs[
            runtime_jobs.dataset
            == ds
        ]

        rows = []

        for method in ordered_methods(
            q.method.unique()
        ):
            z = q[
                q.method == method
            ]

            f1 = pd.to_numeric(
                z.macro_f1,
                errors="coerce",
            ).mean()

            rr = pd.to_numeric(
                runtime_ds.loc[
                    runtime_ds.method
                    == method,
                    "elapsed_min",
                ],
                errors="coerce",
            ).dropna()

            if (
                not np.isfinite(f1)
                or not len(rr)
            ):
                continue

            runtime = rr.median()

            if runtime <= 0:
                continue

            rows.append({
                "method": method,
                "runtime": runtime,
                "f1": f1,
            })

        d = pd.DataFrame(
            rows
        )

        for _, r in d.iterrows():
            m = r.method

            ax.scatter(
                r.runtime,
                r.f1,
                s=165,
                marker=method_marker(
                    m
                ),
                color=method_color(
                    m
                ),
                edgecolor="#222222",
                linewidth=1.0,
                zorder=5,
            )

            annotation(
                ax,
                r.runtime,
                r.f1,
                method_label(
                    m
                ),
                fontsize=8.5,
                dx=4,
                dy=4,
            )

        ax.set_xscale(
            "log"
        )

        ax.set_xlabel(
            "Median runtime [min] ↓",
            fontsize=14,
        )

        ax.set_ylabel(
            "Macro-F1 ↑",
            fontsize=14,
        )

        ax.set_title(
            ds.replace(
                "_",
                " "
            ).title(),
            fontsize=17,
        )

        clean_axis(ax)

    for i in range(
        len(datasets),
        len(axes),
    ):
        axes[i].axis(
            "off"
        )

    fig.suptitle(
        "Compute–utility landscape by dataset",
        weight="bold",
        y=1.015,
    )

    save(
        fig,
        number,
        "compute_utility_atlas",
        "Compute–utility atlas",
    )


compute_atlas()


# =============================================================================
# FIGURE 24: SUPPORT-LEVEL SENSITIVITY ATLAS
# =============================================================================

def support_atlas(number=24):
    needed = [
        "eo_mean_s5",
        "eo_mean_s10",
        "eo_mean",
    ]

    if not all(
        x in seed_balanced.columns
        for x in needed
    ):
        return

    ncols = 3
    nrows = math.ceil(
        len(datasets)
        / ncols
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            17,
            5.15 * nrows,
        ),
        constrained_layout=True,
        squeeze=False,
    )

    axes = axes.ravel()

    x = np.arange(3)

    xlabels = [
        "support ≥5",
        "support ≥10",
        "all groups",
    ]

    for i, ds in enumerate(
        datasets
    ):
        ax = axes[i]

        q = seed_balanced[
            seed_balanced.dataset
            == ds
        ]

        for method in ordered_methods(
            q.method.unique()
        ):
            z = q[
                q.method == method
            ]

            means = []
            lows = []
            highs = []

            for metric in needed:
                vals = pd.to_numeric(
                    z[metric],
                    errors="coerce",
                ).dropna()

                if not len(vals):
                    means.append(
                        np.nan
                    )
                    lows.append(
                        np.nan
                    )
                    highs.append(
                        np.nan
                    )
                    continue

                means.append(
                    vals.mean()
                )

                lows.append(
                    vals.quantile(
                        0.25
                    )
                )

                highs.append(
                    vals.quantile(
                        0.75
                    )
                )

            means = np.array(
                means,
                dtype=float,
            )

            lows = np.array(
                lows,
                dtype=float,
            )

            highs = np.array(
                highs,
                dtype=float,
            )

            if not np.isfinite(
                means
            ).any():
                continue

            gradient_band(
                ax,
                x,
                lows,
                highs,
                method_color(
                    method
                ),
                layers=8,
                max_alpha=0.11,
            )

            ax.plot(
                x,
                means,
                color=method_color(
                    method
                ),
                marker=method_marker(
                    method
                ),
                linewidth=2.5,
                markersize=7,
                alpha=0.92,
                label=method_label(
                    method
                ),
            )

        ax.set_xticks(x)

        ax.set_xticklabels(
            xlabels,
            rotation=16,
            ha="right",
            fontsize=11,
        )

        ax.set_ylabel(
            "EO disparity ↓",
            fontsize=14,
        )

        ax.set_title(
            ds.replace(
                "_",
                " "
            ).title(),
            fontsize=17,
        )

        clean_axis(ax)

    for i in range(
        len(datasets),
        len(axes),
    ):
        axes[i].axis(
            "off"
        )

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                1.015,
            ),
            ncol=4,
            fontsize=11,
        )

    fig.suptitle(
        "Fairness sensitivity to group-support aggregation",
        weight="bold",
        y=1.06,
    )

    save(
        fig,
        number,
        "support_sensitivity_atlas",
        "Support sensitivity atlas",
    )


support_atlas()


# =============================================================================
# FIGURE 25: BOOTSTRAP-RELATIVE FAIRNESS–UTILITY MOVEMENT
# =============================================================================

def delta_atlas(number=25):
    if not all(
        x in seed_balanced.columns
        for x in [
            "macro_f1",
            "eo_mean",
        ]
    ):
        return

    ncols = 3
    nrows = math.ceil(
        len(datasets)
        / ncols
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            17,
            5.2 * nrows,
        ),
        constrained_layout=True,
        squeeze=False,
    )

    axes = axes.ravel()

    for i, ds in enumerate(
        datasets
    ):
        ax = axes[i]

        q = seed_balanced[
            seed_balanced.dataset
            == ds
        ]

        b = q[
            q.method
            == "bootstrap"
        ]

        if b.empty:
            ax.axis("off")
            continue

        b_f1 = pd.to_numeric(
            b.macro_f1,
            errors="coerce",
        ).mean()

        b_eo = pd.to_numeric(
            b.eo_mean,
            errors="coerce",
        ).mean()

        rows = []

        for method in ordered_methods(
            q.method.unique()
        ):
            z = q[
                q.method == method
            ]

            f1 = pd.to_numeric(
                z.macro_f1,
                errors="coerce",
            ).mean()

            eo = pd.to_numeric(
                z.eo_mean,
                errors="coerce",
            ).mean()

            if not (
                np.isfinite(f1)
                and np.isfinite(eo)
            ):
                continue

            rows.append({
                "method": method,
                "dx": eo - b_eo,
                "dy": f1 - b_f1,
            })

        d = pd.DataFrame(
            rows
        )

        ax.axvline(
            0,
            color="#777777",
            linestyle="--",
            linewidth=1.3,
            zorder=1,
        )

        ax.axhline(
            0,
            color="#777777",
            linestyle="--",
            linewidth=1.3,
            zorder=1,
        )

        # pastel desirable quadrant
        ymin, ymax = (
            ax.get_ylim()
        )

        for _, r in d.iterrows():
            method = r.method

            ax.scatter(
                r.dx,
                r.dy,
                s=160,
                color=method_color(
                    method
                ),
                marker=method_marker(
                    method
                ),
                edgecolor="#222222",
                linewidth=1.0,
                zorder=5,
            )

            annotation(
                ax,
                r.dx,
                r.dy,
                method_label(
                    method
                ),
                fontsize=8.5,
                dx=4,
                dy=4,
            )

        ax.set_xlabel(
            "Δ EO vs Bootstrap\n← lower disparity",
            fontsize=13,
        )

        ax.set_ylabel(
            "Δ Macro-F1 vs Bootstrap ↑",
            fontsize=13,
        )

        ax.set_title(
            ds.replace(
                "_",
                " "
            ).title(),
            fontsize=17,
        )

        ax.text(
            0.02,
            0.97,
            "upper-left:\nhigher utility,\nlower disparity",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#333333",
            bbox=bbox_kw(
                0.80
            ),
        )

        clean_axis(ax)

    for i in range(
        len(datasets),
        len(axes),
    ):
        axes[i].axis(
            "off"
        )

    fig.suptitle(
        "Movement relative to Bootstrap within each dataset",
        weight="bold",
        y=1.015,
    )

    save(
        fig,
        number,
        "delta_vs_bootstrap_atlas",
        "Bootstrap-relative atlas",
    )


delta_atlas()


# =============================================================================
# INDEX
# =============================================================================

index = pd.DataFrame(
    FIGURES
).sort_values(
    "figure"
)

index.to_csv(
    OUT / "FIGURE_INDEX.csv",
    index=False,
)


print()
print("=" * 90)
print("FIGURES_MAIN COMPLETE")
print("=" * 90)
print("folder :", OUT)
print("figures:", len(index))
print("PNGs   :", len(list(OUT.glob('*.png'))))
print("PDFs   :", len(list(OUT.glob('*.pdf'))))
print()

print(
    index[
        [
            "figure",
            "stem",
            "title",
        ]
    ].to_string(
        index=False
    )
)

print("=" * 90)
