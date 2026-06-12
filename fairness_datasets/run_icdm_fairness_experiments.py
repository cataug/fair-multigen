# ============================================================
# ICDM Fairness-Aware Synthetic Data Experiments
# CPU-only, parallel, resumable
#
# Main:
#   6 datasets × 4 sensitive settings × 6 methods × 5 seeds = 720 runs
#
# Ablation:
#   3 datasets × 4 sensitive settings × 5 variants × 5 seeds = 300 runs
#
# Total:
#   1020 runs
# ============================================================

import os

# ------------------------------------------------------------
# CPU-only / avoid oversubscription
# ------------------------------------------------------------

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time
import math
import shutil
import traceback
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics import pairwise_distances
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 0. Global paths and experiment config
# ============================================================

BASE = Path(os.environ.get("A100", "/home/tahiti/DataGenaration")).expanduser().resolve()
DATA_DIR = BASE / "fairness_datasets"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = BASE / "RESULTS" / "icdm_fairness_final_cpu"
RUNS_DIR = RESULTS_DIR / "runs_json"
MAIN_RUNS_DIR = RUNS_DIR / "main"
ABL_RUNS_DIR = RUNS_DIR / "ablation"

for d in [RESULTS_DIR, RUNS_DIR, MAIN_RUNS_DIR, ABL_RUNS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

N_SEEDS = 5
SEEDS = list(range(N_SEEDS))

# Parallel CPU workers.
# Keep moderate because each process trains sklearn models.
MAX_WORKERS = max(1, min(12, (os.cpu_count() or 4) // 2))

# Synthetic train size. 1.0 = same size as real train.
SYNTH_SIZE_RATIO = 1.0

# Classifier features:
# True = remove current sensitive columns from downstream classifier.
# This is common in fairness evaluation to test proxy bias.
DROP_CURRENT_SENSITIVE_FROM_MODEL = True

# Minimum group size for fairness metrics.
MIN_GROUP_COUNT = 20

# Main datasets for final paper.
MAIN_DATASETS = [
    "adult",
    "compas",
    "german_credit",
    "bank_marketing",
    "default_credit_card",
    "communities_crime",
]

# Optional supplementary datasets.
SUPPLEMENTARY_DATASETS = [
    "student_performance",
    "heart_disease",
]

# Ablations only on representative larger datasets.
ABLATION_DATASETS = [
    "adult",
    "compas",
    "default_credit_card",
]

# Main methods: exactly 6 methods for 720 runs.
MAIN_METHODS = [
    "real_train",
    "bootstrap",
    "gaussian_copula",
    "vanilla_gdt",
    "single_fair_gdt",
    "our_multi_fair_gdt",
]

# Five ablation variants for 300 runs.
ABLATION_METHODS = [
    "abl_no_fairness",
    "abl_single_sensitive_only",
    "abl_separate_no_intersection",
    "abl_no_smoothing",
    "abl_weak_fairness",
]


# ============================================================
# 1. Dataset registry
# ============================================================

DATASET_FILES = {
    "adult": "adult_processed.csv",
    "compas": "compas_processed.csv",
    "german_credit": "german_credit_processed.csv",
    "bank_marketing": "bank_marketing_processed.csv",
    "default_credit_card": "default_credit_card_processed.csv",
    "communities_crime": "communities_crime_processed.csv",
    "student_performance": "student_performance_processed.csv",
    "heart_disease": "heart_disease_processed.csv",
}


# Four sensitive settings per dataset.
# Each dataset must have:
#   binary
#   nonbinary
#   age
#   intersectional
#
# Some "binary" settings use derived columns created in add_derived_columns().
SENSITIVE_SETTINGS = {
    "adult": {
        "single_binary": ["sex"],
        "single_nonbinary": ["race"],
        "age_binned": ["age_group"],
        "intersectional": ["sex", "race", "age_group"],
    },
    "compas": {
        "single_binary": ["sex"],
        "single_nonbinary": ["race"],
        "age_binned": ["age_cat"],
        "intersectional": ["sex", "race", "age_cat"],
    },
    "german_credit": {
        "single_binary": ["foreign_worker"],
        "single_nonbinary": ["personal_status_sex"],
        "age_binned": ["age_group"],
        "intersectional": ["personal_status_sex", "foreign_worker", "age_group"],
    },
    "bank_marketing": {
        "single_binary": ["marital_binary"],
        "single_nonbinary": ["education"],
        "age_binned": ["age_group"],
        "intersectional": ["marital", "education", "age_group"],
    },
    "default_credit_card": {
        "single_binary": ["SEX"],
        "single_nonbinary": ["EDUCATION"],
        "age_binned": ["age_group"],
        "intersectional": ["SEX", "EDUCATION", "MARRIAGE", "age_group"],
    },
    "communities_crime": {
        "single_binary": ["black_high"],
        "single_nonbinary": ["black_group"],
        "age_binned": ["young_group"],
        "intersectional": ["black_group", "white_group", "hisp_group"],
    },
    "student_performance": {
        "single_binary": ["sex"],
        "single_nonbinary": ["address"],
        "age_binned": ["age_group"],
        "intersectional": ["sex", "address", "famsize", "age_group"],
    },
    "heart_disease": {
        "single_binary": ["sex"],
        "single_nonbinary": ["age_group"],
        "age_binned": ["age_group"],
        "intersectional": ["sex", "age_group"],
    },
}


# ============================================================
# 2. Utilities
# ============================================================

def log(msg):
    print(msg, flush=True)


def safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    except Exception:
        return None


def qcut_safe(s, q=4):
    s = pd.to_numeric(s, errors="coerce")
    try:
        return pd.qcut(s, q=q, labels=False, duplicates="drop").astype("float")
    except Exception:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype="float")


def make_group_key(df, sensitive_cols):
    if len(sensitive_cols) == 0:
        return pd.Series(["ALL"] * len(df), index=df.index)

    parts = []
    for c in sensitive_cols:
        if c not in df.columns:
            parts.append(pd.Series(["MISSING"] * len(df), index=df.index))
        else:
            parts.append(df[c].astype(str).fillna("NA"))

    key = parts[0]
    for p in parts[1:]:
        key = key + "||" + p
    return key


def clean_dataframe(df):
    df = df.copy()

    # Normalize column names minimally.
    df.columns = [str(c).strip() for c in df.columns]

    # Drop fully empty columns.
    df = df.dropna(axis=1, how="all")

    # Replace inf.
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop rows without target.
    if "target" not in df.columns:
        raise ValueError("Dataset has no target column")

    df = df.dropna(subset=["target"]).copy()

    # Force binary target if possible.
    y = pd.to_numeric(df["target"], errors="coerce")
    if y.notna().mean() > 0.95:
        df["target"] = y.astype(int)
    else:
        vals = sorted(df["target"].dropna().unique().tolist())
        mapping = {v: i for i, v in enumerate(vals)}
        df["target"] = df["target"].map(mapping).astype(int)

    # If target has more than two classes, binarize by > 0.
    if df["target"].nunique() > 2:
        df["target"] = (df["target"] > 0).astype(int)

    return df


def add_derived_columns(df, dataset_name):
    df = df.copy()

    # General age group if age-like column exists.
    if "age_group" not in df.columns:
        for age_col in ["age", "AGE", "AGEP"]:
            if age_col in df.columns:
                df["age_group"] = qcut_safe(df[age_col], q=4)
                break

    if dataset_name == "bank_marketing":
        if "marital" in df.columns:
            df["marital_binary"] = np.where(
                df["marital"].astype(str).str.lower().eq("married"),
                "married",
                "not_married"
            )

    if dataset_name == "communities_crime":
        if "racepctblack" in df.columns:
            x = pd.to_numeric(df["racepctblack"], errors="coerce")
            df["black_high"] = np.where(x > x.median(), "high_black_share", "low_black_share")
            if "black_group" not in df.columns:
                df["black_group"] = qcut_safe(x, q=4)

        if "racePctWhite" in df.columns and "white_group" not in df.columns:
            df["white_group"] = qcut_safe(df["racePctWhite"], q=4)

        if "racePctHisp" in df.columns and "hisp_group" not in df.columns:
            df["hisp_group"] = qcut_safe(df["racePctHisp"], q=4)

        if "racePctAsian" in df.columns and "asian_group" not in df.columns:
            df["asian_group"] = qcut_safe(df["racePctAsian"], q=4)

        # Young-age composition as age sensitive proxy.
        if "agePct12t29" in df.columns:
            df["young_group"] = qcut_safe(df["agePct12t29"], q=4)
        elif "agePct16t24" in df.columns:
            df["young_group"] = qcut_safe(df["agePct16t24"], q=4)

    return df


def load_dataset(dataset_name):
    path = PROCESSED_DIR / DATASET_FILES[dataset_name]
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df = clean_dataframe(df)
    df = add_derived_columns(df, dataset_name)

    return df


def get_feature_columns(df, sensitive_cols):
    cols = [c for c in df.columns if c != "target"]

    if DROP_CURRENT_SENSITIVE_FROM_MODEL:
        cols = [c for c in cols if c not in sensitive_cols]

    # Avoid leakage original regression target in Communities.
    if "ViolentCrimesPerPop" in cols:
        cols.remove("ViolentCrimesPerPop")

    # Avoid grade leakage for Student Performance, target was derived from G3.
    for leak in ["G3"]:
        if leak in cols:
            cols.remove(leak)

    return cols


def split_columns(df, feature_cols):
    numeric_cols = []
    categorical_cols = []

    for c in feature_cols:
        if c not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[c]):
            # Low-cardinality numeric columns can still be treated numeric.
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    return numeric_cols, categorical_cols


def make_preprocessor(df, feature_cols):
    numeric_cols, categorical_cols = split_columns(df, feature_cols)

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=True)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot),
    ])

    transformers = []

    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))

    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )

    return pre


def make_downstream_models(seed):
    models = {
        "logreg": LogisticRegression(
            max_iter=500,
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
        ),
        "rf": RandomForestClassifier(
            n_estimators=120,
            max_depth=None,
            min_samples_leaf=3,
            n_jobs=1,
            class_weight="balanced_subsample",
            random_state=seed,
        ),
    }
    return models


# ============================================================
# 3. Fairness metrics
# ============================================================

def positive_rate(y_pred):
    y_pred = np.asarray(y_pred)
    if len(y_pred) == 0:
        return np.nan
    return float(np.mean(y_pred == 1))


def safe_rate(mask, values):
    mask = np.asarray(mask)
    values = np.asarray(values)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(values[mask] == 1))


def compute_fairness_metrics(df_eval, y_true, y_pred, sensitive_cols, min_group_count=20):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    groups = make_group_key(df_eval, sensitive_cols).astype(str)
    group_counts = groups.value_counts()
    valid_groups = group_counts[group_counts >= min_group_count].index.tolist()

    out = {
        "n_groups_total": int(group_counts.shape[0]),
        "n_groups_valid": int(len(valid_groups)),
        "min_group_count": int(group_counts.min()) if len(group_counts) else 0,
        "max_group_count": int(group_counts.max()) if len(group_counts) else 0,
    }

    if len(valid_groups) <= 1:
        out.update({
            "dp_diff": None,
            "dp_max_gap": None,
            "equal_opp_diff": None,
            "equalized_odds_diff": None,
            "avg_odds_diff": None,
            "worst_group_positive_rate": None,
            "best_group_positive_rate": None,
        })
        return out

    global_pr = positive_rate(y_pred)

    group_prs = []
    tpr_gaps = []
    fpr_gaps = []
    eq_odds_gaps = []

    global_tpr = safe_rate(y_true == 1, y_pred)
    global_fpr = safe_rate(y_true == 0, y_pred)

    for g in valid_groups:
        m = (groups.values == g)

        pr_g = positive_rate(y_pred[m])
        group_prs.append(pr_g)

        tpr_g = safe_rate(m & (y_true == 1), y_pred)
        fpr_g = safe_rate(m & (y_true == 0), y_pred)

        if not np.isnan(tpr_g) and not np.isnan(global_tpr):
            tpr_gaps.append(abs(tpr_g - global_tpr))

        if not np.isnan(fpr_g) and not np.isnan(global_fpr):
            fpr_gaps.append(abs(fpr_g - global_fpr))

        local = []
        if not np.isnan(tpr_g) and not np.isnan(global_tpr):
            local.append(abs(tpr_g - global_tpr))
        if not np.isnan(fpr_g) and not np.isnan(global_fpr):
            local.append(abs(fpr_g - global_fpr))
        if local:
            eq_odds_gaps.append(max(local))

    group_prs = [x for x in group_prs if not np.isnan(x)]

    dp_abs = [abs(x - global_pr) for x in group_prs]
    dp_diff = max(dp_abs) if dp_abs else None
    dp_max_gap = (max(group_prs) - min(group_prs)) if group_prs else None

    equal_opp_diff = max(tpr_gaps) if tpr_gaps else None
    fpr_diff = max(fpr_gaps) if fpr_gaps else None
    equalized_odds_diff = max(eq_odds_gaps) if eq_odds_gaps else None

    # Average odds: average of max TPR and max FPR deviation.
    if equal_opp_diff is not None and fpr_diff is not None:
        avg_odds_diff = 0.5 * (equal_opp_diff + fpr_diff)
    else:
        avg_odds_diff = None

    out.update({
        "dp_diff": safe_float(dp_diff),
        "dp_max_gap": safe_float(dp_max_gap),
        "equal_opp_diff": safe_float(equal_opp_diff),
        "equalized_odds_diff": safe_float(equalized_odds_diff),
        "avg_odds_diff": safe_float(avg_odds_diff),
        "worst_group_positive_rate": safe_float(max(group_prs) if group_prs else None),
        "best_group_positive_rate": safe_float(min(group_prs) if group_prs else None),
    })

    return out


# ============================================================
# 4. Synthetic data quality metrics
# ============================================================

def categorical_tvd(real_s, synth_s):
    r = real_s.astype(str).fillna("NA")
    s = synth_s.astype(str).fillna("NA")

    cats = sorted(set(r.unique()).union(set(s.unique())))
    if len(cats) == 0:
        return None

    rp = r.value_counts(normalize=True).reindex(cats).fillna(0.0)
    sp = s.value_counts(normalize=True).reindex(cats).fillna(0.0)

    return float(0.5 * np.abs(rp.values - sp.values).sum())


def numeric_ks_like(real_s, synth_s):
    # Lightweight KS approximation without scipy.
    r = pd.to_numeric(real_s, errors="coerce").dropna().values
    s = pd.to_numeric(synth_s, errors="coerce").dropna().values

    if len(r) < 2 or len(s) < 2:
        return None

    values = np.sort(np.unique(np.concatenate([r, s])))
    if len(values) > 1000:
        rng = np.random.default_rng(123)
        values = np.sort(rng.choice(values, size=1000, replace=False))

    r_sorted = np.sort(r)
    s_sorted = np.sort(s)

    r_cdf = np.searchsorted(r_sorted, values, side="right") / len(r_sorted)
    s_cdf = np.searchsorted(s_sorted, values, side="right") / len(s_sorted)

    return float(np.max(np.abs(r_cdf - s_cdf)))


def correlation_distance(real_df, synth_df, numeric_cols):
    cols = [c for c in numeric_cols if c in real_df.columns and c in synth_df.columns]
    if len(cols) < 2:
        return None

    r = real_df[cols].apply(pd.to_numeric, errors="coerce").fillna(real_df[cols].median(numeric_only=True))
    s = synth_df[cols].apply(pd.to_numeric, errors="coerce").fillna(synth_df[cols].median(numeric_only=True))

    try:
        rc = r.corr().fillna(0.0).values
        sc = s.corr().fillna(0.0).values
        return float(np.mean(np.abs(rc - sc)))
    except Exception:
        return None


def detection_auc(real_df, synth_df, feature_cols, seed):
    n = min(len(real_df), len(synth_df), 5000)
    if n < 50:
        return None

    real_sample = real_df.sample(n=n, random_state=seed)
    synth_sample = synth_df.sample(n=n, random_state=seed)

    a = real_sample[feature_cols].copy()
    b = synth_sample[feature_cols].copy()

    a["_is_synth"] = 0
    b["_is_synth"] = 1

    combined = pd.concat([a, b], axis=0, ignore_index=True)
    y = combined["_is_synth"].astype(int).values
    X = combined.drop(columns=["_is_synth"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.35,
        random_state=seed,
        stratify=y,
    )

    pre = make_preprocessor(X_train, list(X_train.columns))

    clf = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=300, solver="liblinear", random_state=seed)),
    ])

    try:
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        return float(roc_auc_score(y_test, proba))
    except Exception:
        return None


def compute_quality_metrics(real_train, synth_train, feature_cols, seed):
    numeric_cols, categorical_cols = split_columns(real_train, feature_cols)

    ks_vals = []
    for c in numeric_cols:
        if c in synth_train.columns:
            v = numeric_ks_like(real_train[c], synth_train[c])
            if v is not None:
                ks_vals.append(v)

    tvd_vals = []
    for c in categorical_cols:
        if c in synth_train.columns:
            v = categorical_tvd(real_train[c], synth_train[c])
            if v is not None:
                tvd_vals.append(v)

    corr_dist = correlation_distance(real_train, synth_train, numeric_cols)
    det_auc = detection_auc(real_train, synth_train, feature_cols, seed)

    return {
        "quality_numeric_ks_mean": safe_float(np.mean(ks_vals) if ks_vals else None),
        "quality_categorical_tvd_mean": safe_float(np.mean(tvd_vals) if tvd_vals else None),
        "quality_corr_distance": safe_float(corr_dist),
        "quality_detection_auc": safe_float(det_auc),
    }


# ============================================================
# 5. Generators
# ============================================================

def repair_synthetic_dtypes(real_df, synth_df):
    synth = synth_df.copy()

    for c in real_df.columns:
        if c not in synth.columns:
            synth[c] = real_df[c].sample(n=len(synth), replace=True, random_state=0).values

    synth = synth[real_df.columns]

    for c in real_df.columns:
        if c == "target":
            synth[c] = pd.to_numeric(synth[c], errors="coerce").round().fillna(0).astype(int)
            synth[c] = np.where(synth[c] > 0, 1, 0)
            continue

        if pd.api.types.is_numeric_dtype(real_df[c]):
            synth[c] = pd.to_numeric(synth[c], errors="coerce")

            med = pd.to_numeric(real_df[c], errors="coerce").median()
            if pd.isna(med):
                med = 0.0

            synth[c] = synth[c].fillna(med)

            # If original looked integer-like, round.
            real_nonnull = pd.to_numeric(real_df[c], errors="coerce").dropna()
            if len(real_nonnull) > 0:
                is_int_like = np.allclose(real_nonnull.values, np.round(real_nonnull.values), atol=1e-8)
                if is_int_like:
                    synth[c] = np.round(synth[c]).astype(int)
        else:
            mode = real_df[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "NA"
            synth[c] = synth[c].astype(object).where(~synth[c].isna(), fill).astype(str)

    return synth


def generate_real_train(train_df, n_samples, seed, **kwargs):
    if n_samples <= len(train_df):
        return train_df.sample(n=n_samples, replace=False, random_state=seed).reset_index(drop=True)
    return train_df.sample(n=n_samples, replace=True, random_state=seed).reset_index(drop=True)


def generate_bootstrap(train_df, n_samples, seed, **kwargs):
    return train_df.sample(n=n_samples, replace=True, random_state=seed).reset_index(drop=True)


def generate_gaussian_copula_light(train_df, n_samples, seed, **kwargs):
    rng = np.random.default_rng(seed)
    train = train_df.copy()

    feature_cols = [c for c in train.columns]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    out = pd.DataFrame(index=np.arange(n_samples))

    if numeric_cols:
        X = train[numeric_cols].apply(pd.to_numeric, errors="coerce")
        med = X.median(numeric_only=True)
        X = X.fillna(med).fillna(0.0)

        mu = X.mean().values
        cov = np.cov(X.values, rowvar=False)

        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

        # Stabilize covariance.
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        cov = cov + np.eye(len(numeric_cols)) * 1e-6

        try:
            samples = rng.multivariate_normal(mu, cov, size=n_samples)
        except Exception:
            std = X.std().replace(0, 1).fillna(1).values
            samples = rng.normal(mu, std, size=(n_samples, len(numeric_cols)))

        for j, c in enumerate(numeric_cols):
            real_vals = pd.to_numeric(train[c], errors="coerce").dropna()
            vals = samples[:, j]

            if len(real_vals):
                vals = np.clip(vals, real_vals.min(), real_vals.max())

                is_int_like = np.allclose(real_vals.values, np.round(real_vals.values), atol=1e-8)
                if is_int_like:
                    vals = np.round(vals).astype(int)

            out[c] = vals

    for c in categorical_cols:
        s = train[c].astype(str).fillna("NA")
        probs = s.value_counts(normalize=True)
        out[c] = rng.choice(probs.index.values, size=n_samples, replace=True, p=probs.values)

    out = repair_synthetic_dtypes(train_df, out)
    return out.reset_index(drop=True)


def build_tree_leaf_buckets(train_df, seed, n_estimators=48, min_samples_leaf=8):
    feature_cols = [c for c in train_df.columns if c != "target"]
    X = train_df[feature_cols]
    y = train_df["target"].astype(int).values

    pre = make_preprocessor(train_df, feature_cols)
    forest = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        n_jobs=1,
        random_state=seed,
        class_weight=None,
    )

    pipe = Pipeline([
        ("pre", pre),
        ("forest", forest),
    ])

    pipe.fit(X, y)

    X_enc = pipe.named_steps["pre"].transform(X)
    leaves = pipe.named_steps["forest"].apply(X_enc)

    # Buckets: (tree_id, leaf_id) -> row indices.
    buckets = {}
    n_trees = leaves.shape[1]

    for t in range(n_trees):
        for leaf_id in np.unique(leaves[:, t]):
            idx = np.where(leaves[:, t] == leaf_id)[0]
            if len(idx) > 0:
                buckets[(t, int(leaf_id))] = idx

    keys = list(buckets.keys())
    sizes = np.array([len(buckets[k]) for k in keys], dtype=float)
    sizes = sizes / sizes.sum()

    return {
        "keys": keys,
        "sizes": sizes,
        "buckets": buckets,
    }


def compute_row_fair_weights(
    train_df,
    sensitive_cols,
    mode="multi_intersection",
    alpha=2.0,
    lambda_fair=1.0,
    cap=20.0,
):
    """
    Row weights for fairness-aware resampling.

    Intuition:
    For each protected group A=a, make P(Y=1 | A=a) closer to global P(Y=1)
    by reweighting rows with target y inside each group.

    Modes:
      none
      single_only
      separate_only
      multi_intersection
    """
    n = len(train_df)
    y = train_df["target"].astype(int).values

    weights = np.ones(n, dtype=float)

    if mode == "none" or not sensitive_cols:
        return weights

    global_counts = pd.Series(y).value_counts().reindex([0, 1]).fillna(0.0).values + alpha
    global_probs = global_counts / global_counts.sum()

    sensitive_sets = []

    if mode == "single_only":
        sensitive_sets = [[sensitive_cols[0]]]

    elif mode == "separate_only":
        sensitive_sets = [[c] for c in sensitive_cols]

    elif mode == "multi_intersection":
        sensitive_sets = [[c] for c in sensitive_cols]
        if len(sensitive_cols) > 1:
            sensitive_sets.append(list(sensitive_cols))

    else:
        sensitive_sets = [list(sensitive_cols)]

    for sens_set in sensitive_sets:
        group_key = make_group_key(train_df, sens_set)

        tmp = pd.DataFrame({
            "group": group_key.values,
            "target": y,
        })

        group_target_counts = (
            tmp.groupby(["group", "target"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=[0, 1], fill_value=0)
        )

        for target_value in [0, 1]:
            if target_value not in group_target_counts.columns:
                group_target_counts[target_value] = 0

        group_target_counts = group_target_counts[[0, 1]].astype(float) + alpha
        group_probs = group_target_counts.div(group_target_counts.sum(axis=1), axis=0)

        # Weight factor = desired global target prob / group target prob.
        ratio = {}
        for g in group_probs.index:
            for target_value in [0, 1]:
                denom = float(group_probs.loc[g, target_value])
                numer = float(global_probs[target_value])
                ratio[(g, target_value)] = numer / max(denom, 1e-9)

        factors = np.array([
            ratio.get((group_key.iloc[i], int(y[i])), 1.0)
            for i in range(n)
        ])

        # Smooth strength.
        factors = np.power(factors, lambda_fair)
        weights *= factors

    weights = np.nan_to_num(weights, nan=1.0, posinf=cap, neginf=1.0)
    weights = np.clip(weights, 1.0 / cap, cap)

    if weights.sum() <= 0:
        weights = np.ones(n, dtype=float)

    weights = weights / weights.sum()

    return weights


def generate_gdt_resampler(
    train_df,
    n_samples,
    seed,
    sensitive_cols=None,
    fair_mode="none",
    alpha=2.0,
    lambda_fair=1.0,
    jitter_numeric=True,
):
    rng = np.random.default_rng(seed)

    train = train_df.reset_index(drop=True).copy()
    sensitive_cols = sensitive_cols or []

    leaf_model = build_tree_leaf_buckets(train, seed=seed)

    row_weights = compute_row_fair_weights(
        train,
        sensitive_cols=sensitive_cols,
        mode=fair_mode,
        alpha=alpha,
        lambda_fair=lambda_fair,
        cap=30.0,
    )

    keys = leaf_model["keys"]
    sizes = leaf_model["sizes"]
    buckets = leaf_model["buckets"]

    selected_indices = []

    for _ in range(n_samples):
        k_idx = rng.choice(len(keys), p=sizes)
        key = keys[k_idx]
        idx = buckets[key]

        if len(idx) == 1:
            selected_indices.append(idx[0])
        else:
            local_weights = row_weights[idx].astype(float)
            if local_weights.sum() <= 0 or np.isnan(local_weights.sum()):
                local_weights = np.ones(len(idx)) / len(idx)
            else:
                local_weights = local_weights / local_weights.sum()

            selected_indices.append(int(rng.choice(idx, p=local_weights)))

    synth = train.iloc[selected_indices].copy().reset_index(drop=True)

    if jitter_numeric:
        numeric_cols = [
            c for c in synth.columns
            if c != "target" and pd.api.types.is_numeric_dtype(train[c])
        ]

        for c in numeric_cols:
            real_vals = pd.to_numeric(train[c], errors="coerce").dropna()
            if len(real_vals) < 10:
                continue

            std = float(real_vals.std())
            if std <= 1e-12 or np.isnan(std):
                continue

            noise = rng.normal(0, 0.01 * std, size=len(synth))
            vals = pd.to_numeric(synth[c], errors="coerce").values.astype(float) + noise

            vals = np.clip(vals, real_vals.min(), real_vals.max())

            is_int_like = np.allclose(real_vals.values, np.round(real_vals.values), atol=1e-8)
            if is_int_like:
                vals = np.round(vals).astype(int)

            synth[c] = vals

    synth = repair_synthetic_dtypes(train_df, synth)
    return synth.reset_index(drop=True)


def generate_synthetic(train_df, method, n_samples, seed, sensitive_cols):
    start = time.time()

    if method == "real_train":
        synth = generate_real_train(train_df, n_samples, seed)

    elif method == "bootstrap":
        synth = generate_bootstrap(train_df, n_samples, seed)

    elif method == "gaussian_copula":
        synth = generate_gaussian_copula_light(train_df, n_samples, seed)

    elif method == "vanilla_gdt":
        synth = generate_gdt_resampler(
            train_df,
            n_samples=n_samples,
            seed=seed,
            sensitive_cols=sensitive_cols,
            fair_mode="none",
            alpha=2.0,
            lambda_fair=0.0,
        )

    elif method == "single_fair_gdt":
        synth = generate_gdt_resampler(
            train_df,
            n_samples=n_samples,
            seed=seed,
            sensitive_cols=sensitive_cols,
            fair_mode="single_only",
            alpha=2.0,
            lambda_fair=1.0,
        )

    elif method == "our_multi_fair_gdt":
        synth = generate_gdt_resampler(
            train_df,
            n_samples=n_samples,
            seed=seed,
            sensitive_cols=sensitive_cols,
            fair_mode="multi_intersection",
            alpha=2.0,
            lambda_fair=1.0,
        )

    # Ablations.
    elif method == "abl_no_fairness":
        synth = generate_gdt_resampler(
            train_df,
            n_samples=n_samples,
            seed=seed,
            sensitive_cols=sensitive_cols,
            fair_mode="none",
            alpha=2.0,
            lambda_fair=0.0,
        )

    elif method == "abl_single_sensitive_only":
        synth = generate_gdt_resampler(
            train_df,
            n_samples=n_samples,
            seed=seed,
            sensitive_cols=sensitive_cols,
            fair_mode="single_only",
            alpha=2.0,
            lambda_fair=1.0,
        )

    elif method == "abl_separate_no_intersection":
        synth = generate_gdt_resampler(
            train_df,
            n_samples=n_samples,
            seed=seed,
            sensitive_cols=sensitive_cols,
            fair_mode="separate_only",
            alpha=2.0,
            lambda_fair=1.0,
        )

    elif method == "abl_no_smoothing":
        synth = generate_gdt_resampler(
            train_df,
            n_samples=n_samples,
            seed=seed,
            sensitive_cols=sensitive_cols,
            fair_mode="multi_intersection",
            alpha=0.0,
            lambda_fair=1.0,
        )

    elif method == "abl_weak_fairness":
        synth = generate_gdt_resampler(
            train_df,
            n_samples=n_samples,
            seed=seed,
            sensitive_cols=sensitive_cols,
            fair_mode="multi_intersection",
            alpha=2.0,
            lambda_fair=0.35,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    gen_time = time.time() - start
    return synth, gen_time


# ============================================================
# 6. Evaluation
# ============================================================

def evaluate_downstream(
    train_synth,
    train_real,
    test_real,
    sensitive_cols,
    seed,
):
    feature_cols = get_feature_columns(train_real, sensitive_cols)

    # Ensure columns available.
    feature_cols = [c for c in feature_cols if c in train_synth.columns and c in test_real.columns]

    X_train = train_synth[feature_cols].copy()
    y_train = train_synth["target"].astype(int).values

    X_test = test_real[feature_cols].copy()
    y_test = test_real["target"].astype(int).values

    models = make_downstream_models(seed)

    all_metrics = {}

    for model_name, model in models.items():
        pre = make_preprocessor(train_real, feature_cols)

        pipe = Pipeline([
            ("pre", pre),
            ("clf", model),
        ])

        try:
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test).astype(int)

            if hasattr(pipe, "predict_proba"):
                y_prob = pipe.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred

            metrics = {
                f"{model_name}_accuracy": safe_float(accuracy_score(y_test, y_pred)),
                f"{model_name}_balanced_accuracy": safe_float(balanced_accuracy_score(y_test, y_pred)),
                f"{model_name}_f1": safe_float(f1_score(y_test, y_pred, zero_division=0)),
            }

            try:
                metrics[f"{model_name}_roc_auc"] = safe_float(roc_auc_score(y_test, y_prob))
            except Exception:
                metrics[f"{model_name}_roc_auc"] = None

            fair = compute_fairness_metrics(
                df_eval=test_real,
                y_true=y_test,
                y_pred=y_pred,
                sensitive_cols=sensitive_cols,
                min_group_count=MIN_GROUP_COUNT,
            )

            for k, v in fair.items():
                metrics[f"{model_name}_{k}"] = v

            all_metrics.update(metrics)

        except Exception as e:
            all_metrics[f"{model_name}_error"] = str(e)

    # Convenience aggregate metrics from RF as default.
    for key in [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "dp_diff",
        "dp_max_gap",
        "equal_opp_diff",
        "equalized_odds_diff",
        "avg_odds_diff",
    ]:
        rf_key = f"rf_{key}"
        lr_key = f"logreg_{key}"

        if rf_key in all_metrics:
            all_metrics[f"main_{key}"] = all_metrics[rf_key]
        elif lr_key in all_metrics:
            all_metrics[f"main_{key}"] = all_metrics[lr_key]
        else:
            all_metrics[f"main_{key}"] = None

    return all_metrics


def run_one_task(task):
    """
    One independent experimental run.
    """
    run_type = task["run_type"]
    dataset_name = task["dataset"]
    setting_name = task["setting"]
    method = task["method"]
    seed = int(task["seed"])
    out_path = Path(task["out_path"])

    if out_path.exists():
        return {
            "status": "skipped_exists",
            "out_path": str(out_path),
        }

    start_total = time.time()

    try:
        df = load_dataset(dataset_name)

        sensitive_cols = SENSITIVE_SETTINGS[dataset_name][setting_name]
        missing = [c for c in sensitive_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing sensitive columns for {dataset_name}/{setting_name}: {missing}")

        # Keep rows with valid sensitive columns.
        df = df.dropna(subset=sensitive_cols + ["target"]).copy()

        # Basic safety.
        if df["target"].nunique() < 2:
            raise ValueError("Target has less than 2 classes")

        y = df["target"].astype(int)

        stratify = y if y.value_counts().min() >= 2 else None

        train_real, test_real = train_test_split(
            df,
            test_size=0.30,
            random_state=seed,
            stratify=stratify,
        )

        train_real = train_real.reset_index(drop=True)
        test_real = test_real.reset_index(drop=True)

        n_synth = int(round(len(train_real) * SYNTH_SIZE_RATIO))
        n_synth = max(50, n_synth)

        synth_train, gen_time = generate_synthetic(
            train_df=train_real,
            method=method,
            n_samples=n_synth,
            seed=seed,
            sensitive_cols=sensitive_cols,
        )

        synth_train = clean_dataframe(synth_train)
        synth_train = add_derived_columns(synth_train, dataset_name)

        # Ensure target has both classes. If generator collapsed, repair by bootstrap.
        if synth_train["target"].nunique() < 2:
            extra = train_real.sample(n=len(synth_train), replace=True, random_state=seed)
            synth_train["target"] = extra["target"].values

        eval_start = time.time()

        downstream = evaluate_downstream(
            train_synth=synth_train,
            train_real=train_real,
            test_real=test_real,
            sensitive_cols=sensitive_cols,
            seed=seed,
        )

        feature_cols = get_feature_columns(train_real, sensitive_cols)
        feature_cols = [c for c in feature_cols if c in synth_train.columns]

        quality = compute_quality_metrics(
            real_train=train_real,
            synth_train=synth_train,
            feature_cols=feature_cols,
            seed=seed,
        )

        eval_time = time.time() - eval_start
        total_time = time.time() - start_total

        result = {
            "status": "ok",
            "run_type": run_type,
            "dataset": dataset_name,
            "setting": setting_name,
            "method": method,
            "seed": seed,
            "sensitive_cols": sensitive_cols,
            "n_rows_total": int(len(df)),
            "n_train_real": int(len(train_real)),
            "n_test_real": int(len(test_real)),
            "n_synth": int(len(synth_train)),
            "n_features_model": int(len(feature_cols)),
            "drop_current_sensitive_from_model": bool(DROP_CURRENT_SENSITIVE_FROM_MODEL),
            "generation_time_sec": safe_float(gen_time),
            "evaluation_time_sec": safe_float(eval_time),
            "total_time_sec": safe_float(total_time),
        }

        result.update(downstream)
        result.update(quality)

    except Exception as e:
        result = {
            "status": "error",
            "run_type": run_type,
            "dataset": dataset_name,
            "setting": setting_name,
            "method": method,
            "seed": seed,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    tmp_path.replace(out_path)

    return {
        "status": result.get("status", "unknown"),
        "out_path": str(out_path),
    }


# ============================================================
# 7. Task creation
# ============================================================

def make_run_id(run_type, dataset, setting, method, seed):
    return f"{run_type}__{dataset}__{setting}__{method}__seed{seed}"


def build_tasks():
    tasks = []

    # Main 720 runs.
    for dataset in MAIN_DATASETS:
        for setting in ["single_binary", "single_nonbinary", "age_binned", "intersectional"]:
            for method in MAIN_METHODS:
                for seed in SEEDS:
                    run_id = make_run_id("main", dataset, setting, method, seed)
                    out_path = MAIN_RUNS_DIR / f"{run_id}.json"
                    tasks.append({
                        "run_type": "main",
                        "dataset": dataset,
                        "setting": setting,
                        "method": method,
                        "seed": seed,
                        "out_path": str(out_path),
                    })

    # Ablation 300 runs.
    for dataset in ABLATION_DATASETS:
        for setting in ["single_binary", "single_nonbinary", "age_binned", "intersectional"]:
            for method in ABLATION_METHODS:
                for seed in SEEDS:
                    run_id = make_run_id("ablation", dataset, setting, method, seed)
                    out_path = ABL_RUNS_DIR / f"{run_id}.json"
                    tasks.append({
                        "run_type": "ablation",
                        "dataset": dataset,
                        "setting": setting,
                        "method": method,
                        "seed": seed,
                        "out_path": str(out_path),
                    })

    return tasks


# ============================================================
# 8. Aggregation
# ============================================================

def read_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {
            "status": "bad_json",
            "path": str(path),
            "error": str(e),
        }


def aggregate_results():
    all_json = sorted(RUNS_DIR.rglob("*.json"))

    rows = []
    for p in all_json:
        rows.append(read_json_safe(p))

    df = pd.DataFrame(rows)

    all_path = RESULTS_DIR / "all_results.csv"
    df.to_csv(all_path, index=False)

    if len(df) > 0:
        if "run_type" in df.columns:
            main = df[df["run_type"] == "main"].copy()
            abl = df[df["run_type"] == "ablation"].copy()

            main.to_csv(RESULTS_DIR / "main_results.csv", index=False)
            abl.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)

        # Summary only for successful runs.
        ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()

        group_cols = ["run_type", "dataset", "setting", "method"]
        metric_cols = [
            "main_roc_auc",
            "main_f1",
            "main_balanced_accuracy",
            "main_dp_diff",
            "main_equal_opp_diff",
            "main_equalized_odds_diff",
            "main_avg_odds_diff",
            "quality_numeric_ks_mean",
            "quality_categorical_tvd_mean",
            "quality_corr_distance",
            "quality_detection_auc",
            "generation_time_sec",
            "total_time_sec",
        ]

        existing_metrics = [c for c in metric_cols if c in ok.columns]
        existing_group = [c for c in group_cols if c in ok.columns]

        if existing_group and existing_metrics:
            summary = (
                ok.groupby(existing_group)[existing_metrics]
                .agg(["mean", "std", "count"])
                .reset_index()
            )

            # Flatten columns.
            summary.columns = [
                "_".join([str(x) for x in col if str(x) != ""])
                if isinstance(col, tuple) else str(col)
                for col in summary.columns
            ]

            summary.to_csv(RESULTS_DIR / "summary_mean_std.csv", index=False)

        # Status table.
        status_cols = ["run_type", "dataset", "setting", "method", "status"]
        status_cols = [c for c in status_cols if c in df.columns]
        if status_cols:
            status = df.groupby(status_cols).size().reset_index(name="count")
            status.to_csv(RESULTS_DIR / "run_status.csv", index=False)

    return df


# ============================================================
# 9. Main execution
# ============================================================

def main():
    log("=" * 80)
    log("ICDM FAIRNESS EXPERIMENTS CPU FINAL")
    log("=" * 80)
    log(f"BASE: {BASE}")
    log(f"PROCESSED_DIR: {PROCESSED_DIR}")
    log(f"RESULTS_DIR: {RESULTS_DIR}")
    log(f"MAX_WORKERS: {MAX_WORKERS}")
    log(f"SEEDS: {SEEDS}")
    log(f"DROP_CURRENT_SENSITIVE_FROM_MODEL: {DROP_CURRENT_SENSITIVE_FROM_MODEL}")
    log("=" * 80)

    # Check datasets.
    log("\nDataset availability:")
    for name in MAIN_DATASETS + SUPPLEMENTARY_DATASETS:
        p = PROCESSED_DIR / DATASET_FILES[name]
        log(f"  {name:<25} {'OK' if p.exists() else 'MISSING'} {p}")

    tasks = build_tasks()

    total_tasks = len(tasks)
    expected_main = len(MAIN_DATASETS) * 4 * len(MAIN_METHODS) * len(SEEDS)
    expected_abl = len(ABLATION_DATASETS) * 4 * len(ABLATION_METHODS) * len(SEEDS)

    log("\nTask count:")
    log(f"  main expected:     {expected_main}")
    log(f"  ablation expected: {expected_abl}")
    log(f"  total expected:    {expected_main + expected_abl}")
    log(f"  built total:       {total_tasks}")

    pending = [t for t in tasks if not Path(t["out_path"]).exists()]
    done = total_tasks - len(pending)

    log("\nResume status:")
    log(f"  already done: {done}")
    log(f"  pending:      {len(pending)}")

    task_manifest_path = RESULTS_DIR / "task_manifest.json"
    with open(task_manifest_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    log(f"\nSaved task manifest: {task_manifest_path}")

    if not pending:
        log("\nNo pending tasks. Aggregating only.")
        df = aggregate_results()
        log(f"Aggregated rows: {len(df)}")
        log(f"Results saved to: {RESULTS_DIR}")
        return

    log("\nStarting parallel runs...")

    start = time.time()
    completed = 0
    ok_count = 0
    err_count = 0
    skipped_count = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(run_one_task, task) for task in pending]

        for fut in as_completed(futures):
            completed += 1

            try:
                res = fut.result()
                status = res.get("status", "unknown")
            except Exception as e:
                status = "executor_error"
                res = {"error": str(e)}

            if status == "ok":
                ok_count += 1
            elif status == "skipped_exists":
                skipped_count += 1
            else:
                err_count += 1

            if completed % 10 == 0 or completed == len(pending):
                elapsed = time.time() - start
                rate = completed / max(elapsed, 1e-9)
                log(
                    f"[{completed:>5}/{len(pending)}] "
                    f"ok={ok_count} err={err_count} skipped={skipped_count} "
                    f"elapsed={elapsed/60:.1f} min rate={rate:.3f} runs/sec"
                )

                # Periodic aggregation.
                if completed % 50 == 0:
                    aggregate_results()

    log("\nFinished pending runs. Aggregating final results...")
    df = aggregate_results()

    log("=" * 80)
    log("DONE")
    log("=" * 80)
    log(f"Total result rows: {len(df)}")
    log(f"All results:       {RESULTS_DIR / 'all_results.csv'}")
    log(f"Main results:      {RESULTS_DIR / 'main_results.csv'}")
    log(f"Ablation results:  {RESULTS_DIR / 'ablation_results.csv'}")
    log(f"Summary:           {RESULTS_DIR / 'summary_mean_std.csv'}")
    log(f"Status:            {RESULTS_DIR / 'run_status.csv'}")
    log("=" * 80)


if __name__ == "__main__":
    main()