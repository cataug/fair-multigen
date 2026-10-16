from __future__ import annotations

from typing import Dict, Any, Tuple, List
import time
import warnings
import numpy as np
import pandas as pd

from .data import modelling_table, apply_group_state
from .config import DATASETS


def _impute_for_external(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    info = {"numeric": {}, "categorical": {}}
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            x = pd.to_numeric(out[c], errors="coerce")
            med = float(x.median()) if x.notna().any() else 0.0
            out[c] = x.fillna(med)
            info["numeric"][c] = med
        else:
            out[c] = out[c].astype("string").fillna("__MISSING__").astype(str)
            info["categorical"][c] = "__MISSING__"
    return out, info


def _coerce_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    x = pd.to_numeric(out["__target__"], errors="coerce").fillna(0)
    out["__target__"] = (x >= 0.5).astype(int)
    return out


def _normalize_external_schema(
    syn: pd.DataFrame,
    train: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    """
    Force external-generator output back to the schema of the real
    modelling table. SDV generators can emit mixed Python types inside
    nominal columns (e.g. ints together with strings), which later breaks
    sklearn's categorical encoders.
    """
    out = syn.copy()
    ref = modelling_table(train, dataset)

    for c in ref.columns:
        if c not in out.columns:
            continue

        if c == "__target__":
            continue

        if pd.api.types.is_numeric_dtype(ref[c]):
            xr = pd.to_numeric(ref[c], errors="coerce")
            med = float(xr.median()) if xr.notna().any() else 0.0
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(med)
        else:
            out[c] = (
                out[c]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
            )

    return _coerce_target(out)


def generate_bootstrap(train: pd.DataFrame, dataset: str, n: int, seed: int, group_state):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    tab = modelling_table(train, dataset)
    idx = rng.choice(len(tab), size=n, replace=True)
    syn = tab.iloc[idx].reset_index(drop=True).copy()
    syn = apply_group_state(syn, dataset, group_state)
    return syn, {"fit_seconds": 0.0, "generation_seconds": time.perf_counter() - t0}


def _sdv_synth(train: pd.DataFrame, dataset: str, n: int, seed: int, kind: str, epochs: int = 300):
    try:
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
    except Exception as e:
        raise RuntimeError(
            "SDV baseline requested but SDV is unavailable. Install requirements-deep.txt"
        ) from e

    tab = modelling_table(train, dataset)
    tab, _ = _impute_for_external(tab)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(tab)

    kwargs = {"metadata": metadata}
    if kind == "gaussian_copula":
        model = GaussianCopulaSynthesizer(**kwargs)
    elif kind == "ctgan":
        model = CTGANSynthesizer(**kwargs, epochs=epochs, verbose=False)
    elif kind == "tvae":
        model = TVAESynthesizer(**kwargs, epochs=epochs)
    else:
        raise KeyError(kind)

    t0 = time.perf_counter()
    model.fit(tab)
    fit_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    syn = model.sample(num_rows=n)
    gen_seconds = time.perf_counter() - t1
    return _coerce_target(syn), {"fit_seconds": fit_seconds, "generation_seconds": gen_seconds}


def _synthcity_synth(train: pd.DataFrame, dataset: str, n: int, seed: int, kind: str, n_iter: int = 300):
    try:
        import os
        os.environ["POLARS_FORCE_PKG"] = "compat"

        from synthcity.plugins.core.dataloader import GenericDataLoader
        from synthcity.plugins.generic.plugin_ddpm import TabDDPMPlugin
        from synthcity.plugins.privacy.plugin_decaf import DECAFPlugin
    except Exception as e:
        raise RuntimeError(
            f"Synthcity import failed: {type(e).__name__}: {e}"
        ) from e

    tab = modelling_table(train, dataset)
    tab, _ = _impute_for_external(tab)
    sens = DATASETS[dataset]["groups"][:1]

    # Group columns are not duplicated in the generator table. For DECAF's
    # sensitive_columns interface, append the first group deterministically.
    if kind == "decaf":
        first_group = DATASETS[dataset]["groups"][0]
        tab[first_group] = train[first_group].astype("string").fillna("__MISSING__").astype(str).to_numpy()
        loader = GenericDataLoader(
            tab, target_column="__target__", sensitive_columns=[first_group],
            random_state=seed
        )
        model = DECAFPlugin(n_iter=n_iter, random_state=seed)
    elif kind == "tabddpm":
        loader = GenericDataLoader(tab, target_column="__target__", random_state=seed)
        model = TabDDPMPlugin(n_iter=n_iter, random_state=seed)
    else:
        raise KeyError(kind)

    t0 = time.perf_counter()
    model.fit(loader)
    fit_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    if kind == "decaf":
        # In DECAF/Synthcity, biased_edges maps a child to parent(s) whose
        # incoming causal edge is removed at inference. Remove the direct
        # sensitive -> target edge for the single-sensitive comparison.
        syn = model.generate(
            count=n,
            random_state=seed,
            biased_edges={"__target__": [first_group]},
        ).dataframe()
    else:
        syn = model.generate(count=n, random_state=seed).dataframe()
    gen_seconds = time.perf_counter() - t1

    if kind == "decaf":
        first_group = DATASETS[dataset]["groups"][0]
        if first_group in syn:
            syn = syn.drop(columns=[first_group])
    return _coerce_target(syn), {"fit_seconds": fit_seconds, "generation_seconds": gen_seconds}


def _tabfairgan_synth(train: pd.DataFrame, dataset: str, n: int, seed: int, epochs: int = 300):
    try:
        from tabfairgan import TFG
    except Exception as e:
        raise RuntimeError(
            "TabFairGAN requested but tabfairgan is unavailable. "
            "Install requirements-deep.txt."
        ) from e

    first_group = DATASETS[dataset]["groups"][0]
    tab = modelling_table(train, dataset)
    tab[first_group] = train[first_group].astype("string").to_numpy()

    # TabFairGAN is a single-sensitive-attribute baseline. Restrict its
    # training view to the two non-missing groups if tiny unknown categories
    # exist (e.g. three Diabetes records), while leaving the main benchmark
    # unchanged.
    valid = tab[first_group].notna() & (tab[first_group].astype(str) != "<NA>")
    tab = tab.loc[valid].reset_index(drop=True)

    counts = tab[first_group].value_counts()
    if len(counts) != 2:
        top2 = counts.head(2).index
        tab = tab[tab[first_group].isin(top2)].reset_index(drop=True)
        counts = tab[first_group].value_counts()
    if len(counts) != 2:
        raise RuntimeError(f"{dataset}: TabFairGAN requires a binary sensitive attribute")

    # TabFairGAN 0.1.0 expects the under-privileged sensitive value
    # to be a string. Use explicit categorical string levels.
    groups = list(counts.index)
    smap = {groups[0]: "0", groups[1]: "1"}
    tab[first_group] = tab[first_group].map(smap).astype(str)

    rates = tab.groupby(first_group)["__target__"].mean()
    under = str(rates.idxmin())

    tab, _ = _impute_for_external(tab)
    fairness_config = {
        "fair_epochs": epochs,
        "lamda": 0.5,
        "S": first_group,
        "Y": "__target__",
        "S_under": under,
        "Y_desire": 1,
    }

    t0 = time.perf_counter()
    model = TFG(
        tab,
        epochs=epochs,
        batch_size=256,
        fairness_config=fairness_config,
    )
    model.train()
    fit_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    syn = model.generate_fake_df(num_rows=n)
    gen_seconds = time.perf_counter() - t1

    if first_group in syn.columns:
        inv = {str(v): k for k, v in smap.items()}
        sv = syn[first_group].astype("string")
        sv = sv.str.replace(r"\\.0$", "", regex=True)
        syn[first_group] = sv.map(inv)
        syn = syn.drop(columns=[first_group])

    return _coerce_target(syn), {"fit_seconds": fit_seconds, "generation_seconds": gen_seconds}


def generate_external(
    method: str,
    train: pd.DataFrame,
    dataset: str,
    n: int,
    seed: int,
    group_state: Dict[str, Any],
    epochs: int = 300,
):
    if method == "bootstrap":
        return generate_bootstrap(train, dataset, n, seed, group_state)

    if method in {"gaussian_copula", "ctgan", "tvae"}:
        syn, meta = _sdv_synth(train, dataset, n, seed, method, epochs=epochs)
    elif method == "tabddpm":
        syn, meta = _synthcity_synth(train, dataset, n, seed, "tabddpm", n_iter=epochs)
    elif method == "decaf":
        syn, meta = _synthcity_synth(train, dataset, n, seed, "decaf", n_iter=epochs)
    elif method == "tabfairgan":
        syn, meta = _tabfairgan_synth(train, dataset, n, seed, epochs=epochs)
    else:
        raise KeyError(method)

    # External generators may alter dtypes. Restore the training schema
    # before reconstructing fairness groups and before downstream sklearn.
    syn = _normalize_external_schema(syn, train, dataset)
    syn = apply_group_state(syn, dataset, group_state)
    meta["samples_per_second"] = float(n / max(meta["generation_seconds"], 1e-9))
    return syn, meta
