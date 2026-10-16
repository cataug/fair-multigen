from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import json
import os
import platform
import socket
import time
import traceback

import joblib
import numpy as np
import pandas as pd
import psutil

from .config import DATASETS, DEFAULTS
from .data import load_split, model_columns
from .tree_generator import AutoRegressiveTreeGenerator, FairConfig
from .baselines import generate_external
from .evaluate import downstream_metrics, fidelity_metrics


LOCAL_METHODS = {"gdt", "singlefair", "multifair"}
EXTERNAL_METHODS = {
    "bootstrap", "gaussian_copula", "ctgan", "tvae",
    "tabddpm", "tabfairgan", "decaf",
}


def result_root(root: Path) -> Path:
    p = root / "results" / "journal2026"
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
    return d / f"{dataset}_s{seed}_{tag}_d{max_depth}_leaf{min_samples_leaf}.joblib"


def _subsample_train(train: pd.DataFrame, n, seed: int) -> pd.DataFrame:
    if n in ("", None, "full") or str(n) == "full":
        return train
    n = int(n)
    if n >= len(train):
        return train
    # Stratified deterministic subset.
    rng = np.random.default_rng(seed + 777)
    parts = []
    for yv, z in train.groupby("__target__", sort=True):
        take = max(1, round(n * len(z) / len(train)))
        take = min(take, len(z))
        parts.append(z.iloc[rng.choice(len(z), take, replace=False)])
    out = pd.concat(parts).sample(frac=1, random_state=seed).head(n).reset_index(drop=True)
    return out


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


def _fair_config(job) -> FairConfig:
    return FairConfig(
        lambda_fair=float(job["lambda_fair"]),
        lambda_u=float(job["lambda_u"]),
        alpha=float(job["alpha"]),
        eta=float(job["eta"]),
        epsilon=float(job["epsilon"]),
        update_batch=int(job["update_batch"]),
        constraint_mode=str(job["constraint_mode"]),
        adapt=str(job["adapt"]).lower() in ("true", "1", "yes") if not isinstance(job["adapt"], bool) else job["adapt"],
        group_mode=str(job["group_mode"]),
        penalty_mode=str(job["penalty_mode"]),
    )


def _eval_ks(dataset: str, job: Dict[str, Any]) -> List[int]:
    kmax = len(DATASETS[dataset]["groups"])
    method = job["method"]
    if method in {"tabfairgan", "decaf"}:
        return [1]
    k = int(job.get("k", 0))
    if k <= 0:
        return list(range(1, kmax + 1))
    return [k]


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
        train, val, test, group_state = load_split(root, dataset, seed)
        train = _subsample_train(train, job.get("runtime_n"), seed)

        t_job = time.perf_counter()
        meta = {}

        if method == "real":
            syn = train.copy()
            meta = {"fit_seconds": 0.0, "generation_seconds": 0.0, "samples_per_second": np.nan}
        elif method in LOCAL_METHODS:
            gen, loaded_cache = _load_or_fit_tree(root, dataset, seed, train, job)
            fair = _fair_config(job)
            syn, meta = gen.generate(
                n=len(train),
                group_state=group_state,
                seed=seed + 123456,
                method=method,
                k=(None if int(job.get("k", 0)) <= 0 else int(job["k"])),
                fair=fair,
            )
            meta["tree_cache_hit"] = bool(loaded_cache)
        elif method in EXTERNAL_METHODS:
            syn, meta = generate_external(
                method=method,
                train=train,
                dataset=dataset,
                n=len(train),
                seed=seed,
                group_state=group_state,
                epochs=int(job.get("epochs", 300)),
            )
        else:
            raise KeyError(f"Unknown method: {method}")

        # Ensure binary target and deterministic groups.
        syn["__target__"] = pd.to_numeric(syn["__target__"], errors="coerce").fillna(0).round().clip(0, 1).astype(int)

        if save_synthetic:
            syn.to_pickle(outdir / "synthetic.pkl.gz", compression="gzip")

        metrics = []
        support_buckets = []
        if not bool(job.get("timing_only", False)):
            fidelity = {} if method == "real" else fidelity_metrics(train, syn, dataset, seed)
            for k in _eval_ks(dataset, job):
                rows, buckets = downstream_metrics(
                    train_like=syn,
                    test_real=test,
                    dataset=dataset,
                    eval_k=k,
                    seed=seed,
                    classifiers=("lr", "rf", "xgb"),
                )
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
                            "update_batch", "constraint_mode", "adapt",
                            "group_mode", "penalty_mode"
                        ]},
                        **r,
                        **fidelity,
                    }
                    metrics.append(rec)
                for b in buckets:
                    support_buckets.append({
                        "job_id": jid,
                        "family": job["family"],
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "ablation": job.get("ablation", ""),
                        **b,
                    })

            pd.DataFrame(metrics).to_csv(outdir / "metrics.csv", index=False)
            pd.DataFrame(support_buckets).to_csv(outdir / "support_buckets.csv", index=False)

        rss1 = process.memory_info().rss
        runtime = {
            "job_id": jid,
            "family": job["family"],
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "k": int(job.get("k", 0)),
            "runtime_n": job.get("runtime_n", ""),
            "train_rows": int(len(train)),
            "fit_seconds": float(meta.get("fit_seconds", np.nan)),
            "generation_seconds": float(meta.get("generation_seconds", np.nan)),
            "samples_per_second": float(meta.get("samples_per_second", np.nan)),
            "job_seconds": float(time.perf_counter() - t_job),
            "rss_start_mb": float(rss0 / 1024**2),
            "rss_end_mb": float(rss1 / 1024**2),
            "final_lambda_dp": meta.get("final_lambda_dp", np.nan),
            "final_lambda_eo": meta.get("final_lambda_eo", np.nan),
            "tree_cache_hit": meta.get("tree_cache_hit", False),
        }
        pd.DataFrame([runtime]).to_csv(outdir / "runtime.csv", index=False)

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
