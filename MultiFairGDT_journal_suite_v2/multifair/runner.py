from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import json
import os
import socket
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import psutil

from .config import DATASETS
from .data import load_split
from .tree_generator import AutoRegressiveTreeGenerator, FairConfig
from .baselines import generate_external
from .evaluate import downstream_metrics, fidelity_metrics, support_bucket_metrics


LOCAL_METHODS = {"gdt", "singlefair", "multifair"}
EXTERNAL_METHODS = {
    "bootstrap", "gaussian_copula", "ctgan", "tvae",
    "tabddpm", "tabfairgan", "decaf",
}
CACHE_VERSION = "v2_full"


def result_root(root: Path) -> Path:
    p = root / "results" / "journal2026_v2"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _job_dir(root: Path, job_id: str) -> Path:
    p = result_root(root) / "runs" / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _success(job_dir: Path) -> bool:
    p = job_dir / "status.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text())["status"] == "success"
    except Exception:
        return False


def _tree_cache_path(root: Path, dataset: str, seed: int, runtime_n, max_depth, min_samples_leaf):
    tag = "full" if runtime_n in ("", None, "full") else f"n{runtime_n}"
    d = result_root(root) / "cache" / "trees"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{CACHE_VERSION}_{dataset}_s{seed}_{tag}_d{max_depth}_leaf{min_samples_leaf}.joblib"


def _subsample_train(train: pd.DataFrame, n, seed: int) -> pd.DataFrame:
    if n in ("", None, "full") or str(n) == "full":
        return train
    n = int(n)
    if n >= len(train):
        return train
    rng = np.random.default_rng(seed + 777)
    parts = []
    for _, z in train.groupby("__target__", sort=True):
        take = max(1, round(n * len(z) / len(train)))
        take = min(take, len(z))
        parts.append(z.iloc[rng.choice(len(z), take, replace=False)])
    return pd.concat(parts).sample(frac=1, random_state=seed).head(n).reset_index(drop=True)


def _load_or_fit_tree(root, dataset, seed, train, job):
    cache = _tree_cache_path(
        root, dataset, seed, job.get("runtime_n"),
        int(job["max_depth"]), int(job["min_samples_leaf"])
    )
    if cache.exists():
        try:
            return joblib.load(cache), True
        except Exception:
            pass

    gen = AutoRegressiveTreeGenerator(
        dataset=dataset,
        max_depth=int(job["max_depth"]),
        min_samples_leaf=int(job["min_samples_leaf"]),
        seed=int(seed),
    ).fit(train)
    joblib.dump(gen, cache, compress=3)
    return gen, False


def _to_bool(v):
    return v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")


def _fair_config(job) -> FairConfig:
    return FairConfig(
        lambda_fair=float(job["lambda_fair"]),
        lambda_u=float(job["lambda_u"]),
        alpha=float(job["alpha"]),
        eta=float(job["eta"]),
        epsilon=float(job["epsilon"]),
        update_batch=int(job["update_batch"]),
        candidate_count=int(job.get("candidate_count", 16)),
        constraint_mode=str(job["constraint_mode"]),
        adapt=_to_bool(job["adapt"]),
        group_mode=str(job["group_mode"]),
        penalty_mode=str(job["penalty_mode"]),
        feature_correction=_to_bool(job.get("feature_correction", True)),
    )


def _families(job):
    return set(str(job.get("family", "")).split("|"))


def _eval_ks(dataset: str, job: Dict[str, Any]) -> List[int]:
    kmax = len(DATASETS[dataset]["groups"])
    method = str(job["method"])
    fam = _families(job)
    k = int(job.get("k", 0) or 0)

    if method in {"tabfairgan", "decaf"}:
        return [1]
    if "core" in fam and method in {"gdt", "singlefair"}:
        return list(range(1, kmax + 1))
    if "core" in fam and method == "multifair":
        return [max(1, k)]
    if "reference" in fam or "generic_baselines" in fam:
        return [kmax]
    if "fair_baselines" in fam:
        return [1]
    return [kmax if k <= 0 else k]


def _classifiers_for(job: Dict[str, Any], dataset: str, k: int):
    """
    Primary benchmark uses RF. Classifier robustness is embedded at kmax for
    core/reference jobs, avoiding a second set of generator runs.
    """
    kmax = len(DATASETS[dataset]["groups"])
    fam = _families(job)
    if k == kmax and ("core" in fam or "reference" in fam):
        return ("lr", "rf", "xgb")
    return ("rf",)


def _needs_fidelity(job: Dict[str, Any], dataset: str) -> bool:
    method = str(job["method"])
    if method == "real":
        return False
    fam = _families(job)
    kmax = len(DATASETS[dataset]["groups"])
    k = int(job.get("k", 0) or 0)

    if "generic_baselines" in fam or "fair_baselines" in fam:
        return True
    if "core" in fam:
        if method in {"gdt", "singlefair"}:
            return True  # one table serves k=1..kmax
        return method == "multifair" and k == kmax
    return False


def run_job(root: Path, job: Dict[str, Any], resume=True, save_synthetic=False) -> Dict[str, Any]:
    jid = str(job["job_id"])
    outdir = _job_dir(root, jid)
    if resume and _success(outdir):
        return {"job_id": jid, "status": "skipped_cached"}

    status = {
        "job_id": jid,
        "status": "running",
        "started": time.time(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    (outdir / "job.json").write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")
    (outdir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    process = psutil.Process(os.getpid())
    rss0 = process.memory_info().rss
    try:
        dataset = str(job["dataset"])
        seed = int(job["seed"])
        method = str(job["method"])

        print("  [load] reading split...", flush=True)
        train, val, test, group_state = load_split(root, dataset, seed)
        train = _subsample_train(train, job.get("runtime_n"), seed)
        print(f"  [load] train={len(train):,} val={len(val):,} test={len(test):,}", flush=True)

        t_job = time.perf_counter()
        meta = {}

        if method == "real":
            print("  [reference] using real training data", flush=True)
            syn = train.copy()
            meta = {"fit_seconds": 0.0, "generation_seconds": 0.0, "samples_per_second": np.nan}

        elif method in LOCAL_METHODS:
            print(f"  [fit] {method}: fitting/loading tree generator...", flush=True)
            t_stage = time.perf_counter()
            gen, loaded_cache = _load_or_fit_tree(root, dataset, seed, train, job)
            print(
                f"  [fit] done in {time.perf_counter()-t_stage:.1f}s "
                f"(cache_hit={loaded_cache}; original_fit={gen.fit_seconds:.1f}s)",
                flush=True,
            )
            fair = _fair_config(job)
            print(
                f"  [generate] {len(train):,} rows; k={job.get('k', 0)}; "
                f"feature_correction={fair.feature_correction}; C={fair.candidate_count}",
                flush=True,
            )
            t_stage = time.perf_counter()
            syn, meta = gen.generate(
                n=len(train),
                group_state=group_state,
                seed=seed + 123456,
                method=method,
                k=(None if int(job.get("k", 0) or 0) <= 0 else int(job["k"])),
                fair=fair,
            )
            meta["tree_cache_hit"] = bool(loaded_cache)
            print(
                f"  [generate] done in {time.perf_counter()-t_stage:.1f}s "
                f"({meta.get('samples_per_second', float('nan')):.1f} rows/s)",
                flush=True,
            )

        elif method in EXTERNAL_METHODS:
            print(f"  [external] running {method}...", flush=True)
            t_stage = time.perf_counter()
            syn, meta = generate_external(
                method=method,
                train=train,
                dataset=dataset,
                n=len(train),
                seed=seed,
                group_state=group_state,
                epochs=int(job.get("epochs", 300)),
            )
            print(f"  [external] done in {time.perf_counter()-t_stage:.1f}s", flush=True)
        else:
            raise KeyError(f"Unknown method: {method}")

        syn["__target__"] = (
            pd.to_numeric(syn["__target__"], errors="coerce")
            .fillna(0).round().clip(0, 1).astype(int)
        )

        if save_synthetic:
            syn.to_pickle(outdir / "synthetic.pkl.gz", compression="gzip")

        metrics = []
        support_rows = []
        if not bool(job.get("timing_only", False)):
            if _needs_fidelity(job, dataset):
                print("  [fidelity] computing synthetic-data quality...", flush=True)
                t_stage = time.perf_counter()
                fidelity = fidelity_metrics(train, syn, dataset, seed)
                print(f"  [fidelity] done in {time.perf_counter()-t_stage:.1f}s", flush=True)
            else:
                fidelity = {}

            for k in _eval_ks(dataset, job):
                classifiers = _classifiers_for(job, dataset, k)
                print(
                    f"  [evaluate] k={k}: {' + '.join(x.upper() for x in classifiers)}...",
                    flush=True,
                )
                t_stage = time.perf_counter()
                rows, preds, yte = downstream_metrics(
                    train_like=syn,
                    test_real=test,
                    dataset=dataset,
                    eval_k=k,
                    seed=seed,
                    classifiers=classifiers,
                    return_predictions=True,
                )
                print(f"  [evaluate] k={k} done in {time.perf_counter()-t_stage:.1f}s", flush=True)

                for r in rows:
                    rec = {
                        "job_id": jid,
                        "family": job["family"],
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "k": k,
                        "ablation": job.get("ablation", ""),
                        **{kk: job.get(kk) for kk in [
                            "lambda_fair", "lambda_u", "alpha", "eta", "epsilon",
                            "update_batch", "candidate_count", "constraint_mode", "adapt",
                            "group_mode", "penalty_mode", "feature_correction",
                        ]},
                        **r,
                        **fidelity,
                    }
                    metrics.append(rec)

                # Sparse-support analysis is cheap once RF predictions exist.
                if "rf" in preds:
                    for br in support_bucket_metrics(
                        yte,
                        preds["rf"],
                        test,
                        DATASETS[dataset]["groups"][:k],
                    ):
                        support_rows.append({
                            "job_id": jid,
                            "family": job["family"],
                            "dataset": dataset,
                            "seed": seed,
                            "method": method,
                            "k": k,
                            "ablation": job.get("ablation", ""),
                            "classifier": "rf",
                            **br,
                        })

            pd.DataFrame(metrics).to_csv(outdir / "metrics.csv", index=False)
            if support_rows:
                pd.DataFrame(support_rows).to_csv(outdir / "support_buckets.csv", index=False)

        rss1 = process.memory_info().rss
        runtime = {
            "job_id": jid,
            "family": job["family"],
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "k": int(job.get("k", 0) or 0),
            "runtime_n": job.get("runtime_n", ""),
            "train_rows": int(len(train)),
            "fit_seconds": float(meta.get("fit_seconds", np.nan)),
            "generation_seconds": float(meta.get("generation_seconds", np.nan)),
            "feature_generation_seconds": float(meta.get("feature_generation_seconds", np.nan)),
            "samples_per_second": float(meta.get("samples_per_second", np.nan)),
            "job_seconds": float(time.perf_counter() - t_job),
            "rss_start_mb": float(rss0 / 1024**2),
            "rss_end_mb": float(rss1 / 1024**2),
            "final_lambda_dp": meta.get("final_lambda_dp", np.nan),
            "final_lambda_eo": meta.get("final_lambda_eo", np.nan),
            "feature_lambda_dp": meta.get("feature_lambda_dp", np.nan),
            "feature_lambda_eo": meta.get("feature_lambda_eo", np.nan),
            "feature_corrected_column": meta.get("feature_corrected_column", ""),
            "candidate_count": meta.get("candidate_count", job.get("candidate_count", np.nan)),
            "tree_cache_hit": meta.get("tree_cache_hit", False),
        }
        pd.DataFrame([runtime]).to_csv(outdir / "runtime.csv", index=False)

        print(f"  [done] total job time {time.perf_counter()-t_job:.1f}s", flush=True)
        status.update({
            "status": "success",
            "finished": time.time(),
            "metrics_rows": len(metrics),
        })
        (outdir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return status

    except Exception as e:
        status.update({
            "status": "failed",
            "finished": time.time(),
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        (outdir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"  [failed] {type(e).__name__}: {e}", flush=True)
        return status


def run_plan(
    root: Path,
    plan: pd.DataFrame,
    family: str | None = None,
    method: str | None = None,
    dataset: str | None = None,
    seed: int | None = None,
    limit: int | None = None,
    resume: bool = True,
    save_synthetic: bool = False,
):
    sel = plan.copy()
    if family:
        sel = sel[sel["family"].astype(str).str.split("|").map(lambda xs: family in xs)]
    if method:
        sel = sel[sel.method == method]
    if dataset:
        sel = sel[sel.dataset == dataset]
    if seed is not None:
        sel = sel[sel.seed.astype(int) == int(seed)]
    if limit is not None:
        sel = sel.head(int(limit))

    print(f"Selected {len(sel)} jobs")
    counts = {"success": 0, "failed": 0, "skipped_cached": 0}
    for i, (_, row) in enumerate(sel.iterrows(), 1):
        job = row.to_dict()
        print(
            f"[{i}/{len(sel)}] {job['job_id']} "
            f"{job['family']} {job['dataset']} {job['method']} "
            f"seed={job['seed']} k={job['k']}",
            flush=True,
        )
        st = run_job(root, job, resume=resume, save_synthetic=save_synthetic)
        counts[st["status"]] = counts.get(st["status"], 0) + 1
        if st["status"] == "failed":
            print(f"  FAILED: {st.get('error_type')}: {st.get('error')}", flush=True)
        else:
            print(f"  {st['status']}", flush=True)
    print("Run summary:", counts)
    return counts
