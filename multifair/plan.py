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

SUITE_VERSION = 21


def canonical_job(job: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "suite_version",
        "dataset", "seed", "method", "k", "runtime_n", "timing_only",
        "lambda_fair", "lambda_u", "alpha", "eta", "epsilon", "update_batch",
        "candidate_count", "constraint_mode", "adapt", "group_mode",
        "penalty_mode", "feature_correction", "epochs",
        "max_depth", "min_samples_leaf",
    ]
    return {k: job.get(k) for k in keys}


def _hash(job):
    raw = json.dumps(canonical_job(job), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _base(dataset, seed, method, family, k=0):
    return {
        "suite_version": SUITE_VERSION,
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


def build_plan() -> pd.DataFrame:
    jobs: List[Dict[str, Any]] = []

    # Real-data reference: maximal intersection only in the runner; all three
    # classifiers are used there for robustness.
    for ds in ALL_DATASETS:
        for seed in SEEDS_MAIN:
            jobs.append(_base(ds, seed, "real", "reference", 0))

    # Core scaling: GDT and SingleFair generate once and are evaluated at all k;
    # full MultiFair regenerates at each k because the correction changes.
    for ds in ALL_DATASETS:
        kmax = len(DATASETS[ds]["groups"])
        for seed in SEEDS_MAIN:
            jobs.append(_base(ds, seed, "gdt", "core", 0))
            jobs.append(_base(ds, seed, "singlefair", "core", 0))
            for k in range(1, kmax + 1):
                jobs.append(_base(ds, seed, "multifair", "core", k))

    # Generic synthetic baselines, evaluated at maximal intersection.
    for ds in ALL_DATASETS:
        for seed in SEEDS_MAIN:
            for method in GENERIC_BASELINES:
                jobs.append(_base(ds, seed, method, "generic_baselines", 0))

    # Single-sensitive fair-generator comparison at k=1.
    for ds in FAIR_BASELINE_DATASETS:
        for seed in SEEDS_MAIN:
            for method in FAIR_BASELINES:
                jobs.append(_base(ds, seed, method, "fair_baselines", 1))

    # Ablations at maximal intersection. target_only is the previous narrower
    # implementation and directly tests whether feature-level correction helps.
    for ds in ALL_DATASETS:
        kmax = len(DATASETS[ds]["groups"])
        for seed in SEEDS_MAIN:
            for name, overrides in ABLATIONS.items():
                j = _base(ds, seed, "multifair", "ablation", kmax)
                j.update(overrides)
                j["ablation"] = name
                jobs.append(j)

    # Constraint modes. Joint DP+EO is the core default.
    for ds in ALL_DATASETS:
        kmax = len(DATASETS[ds]["groups"])
        for seed in SEEDS_MAIN:
            for name, overrides in CONSTRAINT_VARIANTS.items():
                j = _base(ds, seed, "multifair", "constraints", kmax)
                j.update(overrides)
                j["ablation"] = name
                jobs.append(j)

    # OFAT parameter sweeps on the original six development datasets, three
    # seeds. Defaults are de-duplicated against core and relevant ablations.
    for ds in DEV_DATASETS:
        kmax = len(DATASETS[ds]["groups"])
        for seed in SEEDS_SWEEP:
            seen_local = set()
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

    # Controlled runtime scaling, three seeds.
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
            if not old.get("ablation") and j.get("ablation"):
                old["ablation"] = j["ablation"]
        else:
            j = dict(j)
            j["job_id"] = jid
            merged[jid] = j

    df = pd.DataFrame(list(merged.values()))
    order = [
        "job_id", "suite_version", "family", "dataset", "seed", "method", "k",
        "runtime_n", "timing_only", "ablation",
        "lambda_fair", "lambda_u", "alpha", "eta", "epsilon",
        "update_batch", "candidate_count", "constraint_mode", "adapt",
        "group_mode", "penalty_mode", "feature_correction",
        "epochs", "max_depth", "min_samples_leaf",
    ]
    for c in order:
        if c not in df:
            df[c] = ""
    return df[order].sort_values(
        ["family", "dataset", "seed", "method", "k", "job_id"]
    ).reset_index(drop=True)


def plan_summary(df: pd.DataFrame) -> Dict[str, Any]:
    synthetic = df[df.method != "real"]
    return {
        "total_execution_jobs": int(len(df)),
        "synthetic_generator_jobs": int(len(synthetic)),
        "real_reference_jobs": int((df.method == "real").sum()),
        "by_method": df.method.value_counts().to_dict(),
    }
