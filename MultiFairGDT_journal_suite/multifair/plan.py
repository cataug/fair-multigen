from __future__ import annotations

import hashlib
import json
from typing import Dict, Any, List
import pandas as pd

from .config import (
    ALL_DATASETS, DATASETS, SEEDS_MAIN, SEEDS_SWEEP,
    GENERIC_BASELINES, FAIR_BASELINES, FAIR_BASELINE_DATASETS,
    ABLATIONS, CONSTRAINT_VARIANTS, SWEEPS, DEFAULTS,
    DEV_DATASETS, RUNTIME_SIZES,
)


def canonical_job(job: Dict[str, Any]) -> Dict[str, Any]:
    # family is deliberately excluded: identical configurations shared by
    # core/ablation/sweep are executed once and tagged with multiple families.
    keys = [
        "dataset", "seed", "method", "k", "runtime_n", "timing_only",
        "lambda_fair", "lambda_u", "alpha", "eta", "epsilon", "update_batch",
        "constraint_mode", "adapt", "group_mode", "penalty_mode",
        "epochs",
    ]
    return {k: job.get(k) for k in keys}


def _hash(job):
    raw = json.dumps(canonical_job(job), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _base(dataset, seed, method, family, k=0):
    d = {
        "dataset": dataset,
        "seed": int(seed),
        "method": method,
        "family": family,
        "k": int(k),
        "runtime_n": "",
        "timing_only": False,
        "epochs": 300,
        "ablation": "",
        **DEFAULTS,
    }
    return d


def build_plan() -> pd.DataFrame:
    jobs: List[Dict[str, Any]] = []

    # Real-data reference: no synthetic generator, one job evaluates all k.
    for ds in ALL_DATASETS:
        for seed in SEEDS_MAIN:
            jobs.append(_base(ds, seed, "real", "reference", 0))

    # Core: GDT and SingleFair generate once and are evaluated at every k;
    # MultiFair is regenerated for each intersection depth.
    for ds in ALL_DATASETS:
        kmax = len(DATASETS[ds]["groups"])
        for seed in SEEDS_MAIN:
            jobs.append(_base(ds, seed, "gdt", "core", 0))
            jobs.append(_base(ds, seed, "singlefair", "core", 0))
            for k in range(1, kmax + 1):
                jobs.append(_base(ds, seed, "multifair", "core", k))

    # Generic synthetic baselines: one generated table per dataset/seed, all k.
    for ds in ALL_DATASETS:
        for seed in SEEDS_MAIN:
            for method in GENERIC_BASELINES:
                jobs.append(_base(ds, seed, method, "generic_baselines", 0))

    # Single-sensitive fairness baselines: k=1 only.
    for ds in FAIR_BASELINE_DATASETS:
        for seed in SEEDS_MAIN:
            for method in FAIR_BASELINES:
                jobs.append(_base(ds, seed, method, "fair_baselines", 1))

    # Ablations at maximal intersection.
    for ds in ALL_DATASETS:
        kmax = len(DATASETS[ds]["groups"])
        for seed in SEEDS_MAIN:
            for name, overrides in ABLATIONS.items():
                j = _base(ds, seed, "multifair", "ablation", kmax)
                j.update(overrides)
                j["ablation"] = name
                jobs.append(j)

    # Constraint variants. Full DP+EO is already the core job.
    for ds in ALL_DATASETS:
        kmax = len(DATASETS[ds]["groups"])
        for seed in SEEDS_MAIN:
            for name, overrides in CONSTRAINT_VARIANTS.items():
                j = _base(ds, seed, "multifair", "constraints", kmax)
                j.update(overrides)
                j["ablation"] = name
                jobs.append(j)

    # One-factor-at-a-time parameter sweeps on original six dev datasets.
    # The union is de-duplicated, then de-duplicated again against core/ablations.
    for ds in DEV_DATASETS:
        kmax = len(DATASETS[ds]["groups"])
        for seed in SEEDS_SWEEP:
            seen_local = set()
            # Explicit default point.
            variants = [dict(DEFAULTS)]
            for param, values in SWEEPS.items():
                for v in values:
                    cfg = dict(DEFAULTS)
                    cfg[param] = v
                    variants.append(cfg)
            for cfg in variants:
                sig = json.dumps(cfg, sort_keys=True)
                if sig in seen_local:
                    continue
                seen_local.add(sig)
                j = _base(ds, seed, "multifair", "sweep", kmax)
                j.update(cfg)
                jobs.append(j)

    # Runtime scaling is kept distinct because timings should come from a
    # controlled timing run rather than from cached core outputs.
    for ds, sizes in RUNTIME_SIZES.items():
        kmax = len(DATASETS[ds]["groups"])
        for seed in SEEDS_SWEEP:
            for size in sizes:
                for method in ("gdt", "multifair"):
                    j = _base(ds, seed, method, "runtime", kmax if method == "multifair" else 0)
                    j["runtime_n"] = size
                    j["timing_only"] = True
                    jobs.append(j)

    # Merge truly identical configurations and accumulate family tags.
    merged = {}
    for j in jobs:
        jid = _hash(j)
        if jid in merged:
            old = merged[jid]
            fams = set(str(old["family"]).split("|")) | set(str(j["family"]).split("|"))
            old["family"] = "|".join(sorted(fams))
            # Preserve a useful ablation label if one exists.
            if not old.get("ablation") and j.get("ablation"):
                old["ablation"] = j["ablation"]
        else:
            j = dict(j)
            j["job_id"] = jid
            merged[jid] = j

    df = pd.DataFrame(list(merged.values()))
    order = [
        "job_id", "family", "dataset", "seed", "method", "k",
        "runtime_n", "timing_only", "ablation",
        "lambda_fair", "lambda_u", "alpha", "eta", "epsilon",
        "update_batch", "constraint_mode", "adapt", "group_mode",
        "penalty_mode", "epochs", "max_depth", "min_samples_leaf",
    ]
    for c in order:
        if c not in df:
            df[c] = ""
    return df[order].sort_values(
        ["family", "dataset", "seed", "method", "k", "job_id"]
    ).reset_index(drop=True)


def plan_summary(df: pd.DataFrame) -> Dict[str, Any]:
    synthetic = df[df.method != "real"]
    out = {
        "total_execution_jobs": int(len(df)),
        "synthetic_generator_jobs": int(len(synthetic)),
        "real_reference_jobs": int((df.method == "real").sum()),
        "by_method": df.method.value_counts().to_dict(),
    }
    return out
