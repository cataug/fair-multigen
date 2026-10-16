from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from .config import DATASETS


def project_root(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    # Designed for /home/tahiti/MultiFairGDT, but portable.
    here = Path.cwd().resolve()
    if (here / "data" / "processed").exists():
        return here
    env = Path("/home/tahiti/MultiFairGDT")
    if env.exists():
        return env
    raise FileNotFoundError(
        "Cannot locate project root. Run from MultiFairGDT or pass --root."
    )


def load_processed(root: Path, dataset: str) -> pd.DataFrame:
    spec = DATASETS[dataset]
    path = root / "data" / "processed" / spec["file"]
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    if "__target__" not in df.columns:
        raise ValueError(f"{dataset}: __target__ missing")
    return df


def _safe_str(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    return out.replace({"": pd.NA, "?": pd.NA, "nan": pd.NA, "None": pd.NA})


def fit_group_state(df_train: pd.DataFrame, dataset: str) -> Dict[str, Any]:
    spec = DATASETS[dataset]
    state: Dict[str, Any] = {"dataset": dataset}

    if spec.get("quantile_groups"):
        edges = {}
        for source in spec["group_sources"]:
            x = pd.to_numeric(df_train[source], errors="coerce").dropna()
            if len(x) == 0:
                edges[source] = [-np.inf, np.inf]
                continue
            qs = np.quantile(x, [0, .25, .5, .75, 1.0])
            qs = np.unique(qs)
            if len(qs) < 2:
                qs = np.array([x.min(), x.max() + 1e-12])
            qs[0] = -np.inf
            qs[-1] = np.inf
            edges[source] = [float(v) for v in qs]
        state["quantile_edges"] = edges

    return state


def apply_group_state(df: pd.DataFrame, dataset: str, state: Dict[str, Any]) -> pd.DataFrame:
    """Recompute fairness groups without using labels/test-distribution statistics."""
    out = df.copy()
    spec = DATASETS[dataset]
    groups = spec["groups"]

    if dataset in {"adult", "compas", "census_income_kdd"}:
        out[groups[0]] = _safe_str(out[spec["group_sources"][0]])
        out[groups[1]] = _safe_str(out[spec["group_sources"][1]])
        age = pd.to_numeric(out[spec["age_source"]], errors="coerce")
        out[groups[2]] = pd.cut(
            age, bins=spec["age_bins"], labels=spec["age_labels"],
            right=False, include_lowest=True
        ).astype("string")
        if len(groups) == 4:
            out[groups[3]] = _safe_str(out[spec["group_sources"][3]])

    elif dataset == "default_credit":
        out[groups[0]] = _safe_str(out["SEX"])
        age = pd.to_numeric(out["AGE"], errors="coerce")
        out[groups[1]] = pd.cut(
            age, bins=spec["age_bins"], labels=spec["age_labels"],
            right=False, include_lowest=True
        ).astype("string")
        out[groups[2]] = _safe_str(out["MARRIAGE"])
        out[groups[3]] = _safe_str(out["EDUCATION"])

    elif dataset == "bank_marketing":
        age = pd.to_numeric(out["age"], errors="coerce")
        out[groups[0]] = pd.cut(
            age, bins=spec["age_bins"], labels=spec["age_labels"],
            right=False, include_lowest=True
        ).astype("string")
        out[groups[1]] = _safe_str(out["marital"])
        out[groups[2]] = _safe_str(out["education"])

    elif dataset == "communities_crime":
        edges = state["quantile_edges"]
        for source, group in zip(spec["group_sources"], groups):
            x = pd.to_numeric(out[source], errors="coerce")
            e = np.asarray(edges[source], dtype=float)
            labels = [f"Q{i+1}" for i in range(len(e) - 1)]
            out[group] = pd.cut(
                x, bins=e, labels=labels, include_lowest=True, right=True
            ).astype("string")

    elif dataset == "german_credit":
        out[groups[0]] = _safe_str(out["personal_status_sex"]).map(spec["sex_map"]).astype("string")
        age = pd.to_numeric(out["age"], errors="coerce")
        out[groups[1]] = pd.cut(
            age, bins=spec["age_bins"], labels=spec["age_labels"],
            right=False, include_lowest=True
        ).astype("string")
        out[groups[2]] = _safe_str(out["foreign_worker"])

    elif dataset == "diabetes130":
        gender = _safe_str(out["gender"])
        # Keep rows for modelling, but do not create a third tiny fairness group.
        out[groups[0]] = gender.where(gender.isin(["Male", "Female"]), pd.NA)
        out[groups[1]] = _safe_str(out["race"])
        out[groups[2]] = _safe_str(out["age"])

    elif dataset == "law_school":
        out[groups[0]] = _safe_str(out["gender"])
        out[groups[1]] = _safe_str(out["race1"])
        age = pd.to_numeric(out["age"], errors="coerce")
        out[groups[2]] = pd.cut(
            age, bins=spec["age_bins"], labels=spec["age_labels"],
            right=False, include_lowest=True
        ).astype("string")
    else:
        raise KeyError(dataset)

    return out


def make_split_indices(df: pd.DataFrame, dataset: str, seed: int) -> Dict[str, np.ndarray]:
    y = pd.to_numeric(df["__target__"], errors="coerce").astype(int).to_numpy()
    idx = np.arange(len(df))

    group_col = DATASETS[dataset].get("group_split")
    if group_col:
        groups = df[group_col].astype("string").fillna("__MISSING__").to_numpy()
        g1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        trainval_pos, test_pos = next(g1.split(idx, y, groups))
        trainval_idx = idx[trainval_pos]
        test_idx = idx[test_pos]
        groups_tv = groups[trainval_pos]
        y_tv = y[trainval_pos]

        # 15 / 85 = 0.176470588 of the remaining rows.
        g2 = GroupShuffleSplit(n_splits=1, test_size=0.17647058823529413, random_state=seed + 10000)
        tr_pos, va_pos = next(g2.split(trainval_idx, y_tv, groups_tv))
        train_idx = trainval_idx[tr_pos]
        val_idx = trainval_idx[va_pos]
    else:
        trainval_idx, test_idx = train_test_split(
            idx, test_size=0.15, random_state=seed, stratify=y
        )
        y_tv = y[trainval_idx]
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=0.17647058823529413,
            random_state=seed + 10000, stratify=y_tv
        )

    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def split_paths(root: Path, dataset: str, seed: int) -> Tuple[Path, Path]:
    d = root / "data" / "splits" / dataset
    return d / f"seed_{seed}.npz", d / f"seed_{seed}_group_state.json"


def save_split(root: Path, dataset: str, seed: int) -> Dict[str, Any]:
    df = load_processed(root, dataset)
    split = make_split_indices(df, dataset, seed)
    train = df.iloc[split["train"]]
    state = fit_group_state(train, dataset)

    npz, js = split_paths(root, dataset, seed)
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, **split)
    js.write_text(json.dumps(state, indent=2, allow_nan=True), encoding="utf-8")

    info = {
        "dataset": dataset, "seed": seed,
        "n": len(df),
        "train": len(split["train"]),
        "val": len(split["val"]),
        "test": len(split["test"]),
        "train_positive": float(df.iloc[split["train"]]["__target__"].mean()),
        "val_positive": float(df.iloc[split["val"]]["__target__"].mean()),
        "test_positive": float(df.iloc[split["test"]]["__target__"].mean()),
    }
    return info


def load_split(root: Path, dataset: str, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    npz, js = split_paths(root, dataset, seed)
    if not npz.exists() or not js.exists():
        save_split(root, dataset, seed)

    arr = np.load(npz)
    state = json.loads(js.read_text(encoding="utf-8"))
    df = load_processed(root, dataset)
    df = apply_group_state(df, dataset, state)

    return (
        df.iloc[arr["train"]].reset_index(drop=True),
        df.iloc[arr["val"]].reset_index(drop=True),
        df.iloc[arr["test"]].reset_index(drop=True),
        state,
    )


def model_columns(df: pd.DataFrame, dataset: str) -> List[str]:
    spec = DATASETS[dataset]
    drop = {"__row_id__", "__target__", *spec["groups"], *spec.get("drop_features", [])}
    return [c for c in df.columns if c not in drop]


def modelling_table(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    cols = model_columns(df, dataset)
    out = df[cols].copy()
    out["__target__"] = pd.to_numeric(df["__target__"], errors="coerce").astype(int)
    return out
