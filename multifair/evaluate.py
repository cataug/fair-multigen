from __future__ import annotations

from typing import Dict, List, Any, Tuple
import os
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from .data import model_columns
from .config import DATASETS

THREADS = max(1, int(os.environ.get("MULTIFAIR_THREADS", "8")))


def _build_preprocessor(train_x: pd.DataFrame):
    num = [c for c in train_x.columns if pd.api.types.is_numeric_dtype(train_x[c])]
    cat = [c for c in train_x.columns if c not in num]
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    return ColumnTransformer([
        ("num", numeric, num),
        ("cat", categorical, cat),
    ]), num, cat


def _classifier(name: str, seed: int):
    if name == "lr":
        return LogisticRegression(
            max_iter=1200,
            tol=1e-3,
            solver="saga",
            C=1.0,
            random_state=seed,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=THREADS,
            random_state=seed,
            class_weight="balanced_subsample",
        )
    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise RuntimeError("xgboost not installed") from e
        return XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            n_jobs=THREADS,
            random_state=seed,
        )
    raise KeyError(name)


def _group_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    work = df[cols].astype("string")
    valid = work.notna().all(axis=1)
    key = pd.Series(pd.NA, index=df.index, dtype="string")
    key.loc[valid] = work.loc[valid].agg("||".join, axis=1)
    return key


def fairness_metrics(y_true, y_pred, groups: pd.Series) -> Dict[str, float]:
    """
    Generalised multi-group fairness metrics.

    Raw all-group metrics are retained for continuity. Because intersectional
    evaluation naturally creates tiny groups, the function also reports the
    same extrema/means for groups with support >= 5, 10, and 20. No row or
    group is removed from training or generation; these are evaluation-only
    sensitivity views.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    g = groups.astype("string")
    valid = g.notna().to_numpy()
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    g = g[valid].astype(str).to_numpy()

    base_names = [
        "dp_mean", "dp_max", "eo_mean", "eo_max",
        "aod_mean", "aod_max", "worst_group_ba",
        "worst_group_tpr", "worst_group_fpr",
        "n_eval_groups", "min_eval_group_support",
    ]
    threshold_names = []
    for t in (5, 10, 20):
        threshold_names.extend([
            f"dp_mean_s{t}", f"dp_max_s{t}",
            f"eo_mean_s{t}", f"eo_max_s{t}",
            f"aod_mean_s{t}", f"aod_max_s{t}",
            f"worst_group_ba_s{t}",
            f"n_eval_groups_s{t}",
        ])

    if len(y_true) == 0:
        return {k: np.nan for k in base_names + threshold_names}

    global_pos = float(np.mean(y_pred))
    global_tpr = float(np.mean(y_pred[y_true == 1])) if np.any(y_true == 1) else np.nan
    global_fpr = float(np.mean(y_pred[y_true == 0])) if np.any(y_true == 0) else np.nan

    per_group = []
    for gv in sorted(np.unique(g)):
        m = g == gv
        yt = y_true[m]
        yp = y_pred[m]
        support = int(m.sum())

        p = float(np.mean(yp))
        dp = abs(p - global_pos)

        has_pos = np.any(yt == 1)
        has_neg = np.any(yt == 0)
        tpr = float(np.mean(yp[yt == 1])) if has_pos else np.nan
        fpr = float(np.mean(yp[yt == 0])) if has_neg else np.nan

        parts = []
        if has_pos and np.isfinite(global_tpr):
            parts.append(abs(tpr - global_tpr))
        if has_neg and np.isfinite(global_fpr):
            parts.append(abs(fpr - global_fpr))

        eo = float(max(parts)) if parts else np.nan
        aod = float(np.mean(parts)) if parts else np.nan
        ba = float(balanced_accuracy_score(yt, yp)) if has_pos and has_neg else np.nan

        per_group.append({
            "support": support,
            "dp": dp,
            "eo": eo,
            "aod": aod,
            "ba": ba,
            "tpr": tpr,
            "fpr": fpr,
        })

    pg = pd.DataFrame(per_group)

    def finite_mean(x):
        x = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if len(x) else np.nan

    def finite_max(x):
        x = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        return float(np.max(x)) if len(x) else np.nan

    def finite_min(x):
        x = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        return float(np.min(x)) if len(x) else np.nan

    out = {
        "dp_mean": finite_mean(pg["dp"]),
        "dp_max": finite_max(pg["dp"]),
        "eo_mean": finite_mean(pg["eo"]),
        "eo_max": finite_max(pg["eo"]),
        "aod_mean": finite_mean(pg["aod"]),
        "aod_max": finite_max(pg["aod"]),
        "worst_group_ba": finite_min(pg["ba"]),
        "worst_group_tpr": finite_min(pg["tpr"]),
        "worst_group_fpr": finite_max(pg["fpr"]),
        "n_eval_groups": int(len(pg)),
        "min_eval_group_support": int(pg["support"].min()) if len(pg) else 0,
    }

    for t in (5, 10, 20):
        z = pg[pg["support"] >= t]
        out.update({
            f"dp_mean_s{t}": finite_mean(z["dp"]) if len(z) else np.nan,
            f"dp_max_s{t}": finite_max(z["dp"]) if len(z) else np.nan,
            f"eo_mean_s{t}": finite_mean(z["eo"]) if len(z) else np.nan,
            f"eo_max_s{t}": finite_max(z["eo"]) if len(z) else np.nan,
            f"aod_mean_s{t}": finite_mean(z["aod"]) if len(z) else np.nan,
            f"aod_max_s{t}": finite_max(z["aod"]) if len(z) else np.nan,
            f"worst_group_ba_s{t}": finite_min(z["ba"]) if len(z) else np.nan,
            f"n_eval_groups_s{t}": int(len(z)),
        })

    return out


def _align_synthetic_features(syn: pd.DataFrame, real: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=np.arange(len(syn)))
    for c in cols:
        out[c] = syn[c].to_numpy() if c in syn.columns else pd.NA
    return out


def downstream_metrics(
    train_like: pd.DataFrame,
    test_real: pd.DataFrame,
    dataset: str,
    eval_k: int,
    seed: int,
    classifiers=("rf",),
    return_predictions: bool = False,
):
    cols = model_columns(test_real, dataset)
    Xtr = _align_synthetic_features(train_like, test_real, cols)
    Xte = test_real[cols].copy()
    ytr = pd.to_numeric(train_like["__target__"], errors="coerce").fillna(0).astype(int).to_numpy()
    yte = pd.to_numeric(test_real["__target__"], errors="coerce").astype(int).to_numpy()

    groups = DATASETS[dataset]["groups"][:eval_k]
    gkey = _group_key(test_real, groups)

    rows = []
    predictions = {}
    for clf_name in classifiers:
        pre, _, _ = _build_preprocessor(Xtr)
        clf = _classifier(clf_name, seed)
        pipe = Pipeline([("prep", pre), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(Xte)[:, 1]
        else:
            score = pipe.decision_function(Xte)
            proba = 1.0 / (1.0 + np.exp(-score))
        pred = (proba >= 0.5).astype(int)

        row = {
            "classifier": clf_name,
            "roc_auc": float(roc_auc_score(yte, proba)),
            "pr_auc": float(average_precision_score(yte, proba)),
            "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
            "macro_f1": float(f1_score(yte, pred, average="macro")),
        }
        row.update(fairness_metrics(yte, pred, gkey))
        rows.append(row)
        predictions[clf_name] = pred

    if return_predictions:
        return rows, predictions, yte
    return rows


def _tv_distance(a: pd.Series, b: pd.Series) -> float:
    aa = a.astype("string").fillna("__MISSING__")
    bb = b.astype("string").fillna("__MISSING__")
    cats = sorted(set(aa.unique()).union(set(bb.unique())))
    pa = aa.value_counts(normalize=True).reindex(cats, fill_value=0)
    pb = bb.value_counts(normalize=True).reindex(cats, fill_value=0)
    return float(0.5 * np.abs(pa - pb).sum())


def fidelity_metrics(real: pd.DataFrame, syn: pd.DataFrame, dataset: str, seed: int, detection_cap: int = 8000):
    cols = model_columns(real, dataset)
    synx = _align_synthetic_features(syn, real, cols)
    realx = real[cols].copy()

    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(realx[c])]
    categorical = [c for c in cols if c not in numeric]

    ks = []
    for c in numeric:
        a = pd.to_numeric(realx[c], errors="coerce").dropna()
        b = pd.to_numeric(synx[c], errors="coerce").dropna()
        if len(a) and len(b):
            ks.append(float(ks_2samp(a, b).statistic))

    tv = [_tv_distance(realx[c], synx[c]) for c in categorical]

    corr_dist = np.nan
    if len(numeric) >= 2:
        ra = realx[numeric].apply(pd.to_numeric, errors="coerce").corr(method="spearman").fillna(0)
        rb = synx[numeric].apply(pd.to_numeric, errors="coerce").corr(method="spearman").fillna(0)
        corr_dist = float(np.mean(np.abs(ra.to_numpy() - rb.to_numpy())))

    rng = np.random.default_rng(seed)
    nr = min(len(realx), detection_cap)
    ns = min(len(synx), detection_cap)
    ridx = rng.choice(len(realx), nr, replace=False)
    sidx = rng.choice(len(synx), ns, replace=False)
    det = pd.concat([realx.iloc[ridx], synx.iloc[sidx]], ignore_index=True)
    lab = np.r_[np.zeros(nr, dtype=int), np.ones(ns, dtype=int)]
    tr, te = train_test_split(
        np.arange(len(det)), test_size=0.3, random_state=seed, stratify=lab
    )
    pre, _, _ = _build_preprocessor(det.iloc[tr])
    clf = LogisticRegression(
        max_iter=700, tol=1e-3, solver="saga", random_state=seed
    )
    pipe = Pipeline([("prep", pre), ("clf", clf)])
    pipe.fit(det.iloc[tr], lab[tr])
    auc = float(roc_auc_score(lab[te], pipe.predict_proba(det.iloc[te])[:, 1]))
    return {
        "numeric_ks_mean": float(np.mean(ks)) if ks else np.nan,
        "categorical_tv_mean": float(np.mean(tv)) if tv else np.nan,
        "correlation_distance": corr_dist,
        "detection_auc": auc,
        "detection_distance": float(2 * abs(auc - 0.5)),
    }


def support_bucket_metrics(y_true, y_pred, test_real: pd.DataFrame, group_cols: List[str]) -> List[Dict[str, Any]]:
    key = _group_key(test_real, group_cols)
    tmp = pd.DataFrame({
        "g": key,
        "y": np.asarray(y_true, dtype=int),
        "p": np.asarray(y_pred, dtype=int),
    }).dropna()
    if tmp.empty:
        return []

    counts = tmp["g"].value_counts()
    bins = [
        ("1-4", 1, 4), ("5-9", 5, 9), ("10-19", 10, 19),
        ("20-49", 20, 49), ("50-99", 50, 99), ("100+", 100, np.inf),
    ]
    rows = []
    global_tpr = tmp.loc[tmp.y == 1, "p"].mean()
    global_fpr = tmp.loc[tmp.y == 0, "p"].mean()

    group_stats = []
    for g, n in counts.items():
        z = tmp[tmp.g == g]
        tpr = z.loc[z.y == 1, "p"].mean()
        fpr = z.loc[z.y == 0, "p"].mean()
        parts = []
        if np.isfinite(tpr) and np.isfinite(global_tpr):
            parts.append(abs(tpr - global_tpr))
        if np.isfinite(fpr) and np.isfinite(global_fpr):
            parts.append(abs(fpr - global_fpr))
        if parts:
            group_stats.append((g, int(n), float(max(parts)), float(np.mean(parts))))

    for label, lo, hi in bins:
        sel = [x for x in group_stats if x[1] >= lo and x[1] <= hi]
        rows.append({
            "support_bucket": label,
            "n_groups": len(sel),
            "eo_max_mean_within_bucket": float(np.mean([x[2] for x in sel])) if sel else np.nan,
            "aod_mean_within_bucket": float(np.mean([x[3] for x in sel])) if sel else np.nan,
        })
    return rows
