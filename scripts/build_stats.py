from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
import os
import re
import math
import shutil
import subprocess

ROOT = Path("/home/tahiti/MultiFairGDT")
EXP = ROOT / "results" / "journal2026_v21"
PLAN = EXP / "jobs.csv"
STATS = ROOT / "STATS"

NOW = datetime.now()
NOW_TS = NOW.timestamp()

# ---------------------------------------------------------------------
# Fresh STATS directory, preserving an older snapshot if it exists
# ---------------------------------------------------------------------
if STATS.exists():
    backup = ROOT / f"STATS_previous_{NOW.strftime('%Y%m%d_%H%M%S')}"
    STATS.rename(backup)
    print(f"[backup] {STATS} -> {backup}")

STATS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def safe_json(path):
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return {}

def epoch(v):
    if v is None:
        return np.nan
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip()
    if not s:
        return np.nan
    try:
        return float(s)
    except Exception:
        pass
    try:
        return pd.Timestamp(s).timestamp()
    except Exception:
        return np.nan

def fmt_time(v):
    if pd.isna(v):
        return ""
    try:
        return datetime.fromtimestamp(float(v)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""

def pid_alive(pid):
    try:
        pid = int(pid)
        os.kill(pid, 0)
        return True
    except Exception:
        return False

def clean_error(x):
    if x is None:
        return ""
    return " ".join(str(x).replace("\x00", " ").split())

def classify_error(err):
    s = err.lower()

    if not s:
        return ""

    if "out of memory" in s or "cuda oom" in s:
        return "cuda_oom"

    if "device-side assert" in s or "scattergatherkernel" in s:
        return "cuda_device_assert"

    if (
        "index 1 is out of bounds for axis 1 with size 1" in s
        or ("predict_proba" in s and "index 1" in s)
    ):
        return "one_class_predict_proba"

    if "synthcity" in s and (
        "unavailable" in s
        or "requirements-deep" in s
        or "could not import" in s
    ):
        return "synthcity_unavailable"

    if (
        "not supported between instances of" in s
        and ("float" in s and "str" in s)
    ):
        return "mixed_float_str"

    if (
        "mixed type" in s
        or "mixed dtype" in s
        or "uniformly strings or numbers" in s
        or "encoders require their input" in s
    ):
        return "mixed_dtype_encoding"

    if "s_under" in s or (
        "tabfairgan" in s and ("string" in s or "str" in s)
    ):
        return "tabfairgan_sensitive_type"

    if "polars" in s and ("illegal instruction" in s or "cpu" in s):
        return "polars_cpu_runtime"

    if "keyerror" in s:
        return "key_error"

    if "filenotfounderror" in s or "no such file" in s:
        return "missing_file"

    if "valueerror" in s:
        return "value_error"

    if "typeerror" in s:
        return "type_error"

    if "indexerror" in s:
        return "index_error"

    return "other"

def pct(a, b):
    return 100.0 * a / b if b else np.nan

def write_csv(df, name):
    p = STATS / name
    df.to_csv(p, index=False)
    print(f"[write] {name}: {len(df):,} rows")

def scalar_metrics(obj, prefix=""):
    out = []

    if isinstance(obj, bool):
        return out

    if isinstance(obj, (int, float, np.integer, np.floating)):
        try:
            v = float(obj)
            if math.isfinite(v):
                out.append((prefix or "value", v))
        except Exception:
            pass
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            kk = str(k)
            p = f"{prefix}.{kk}" if prefix else kk
            out.extend(scalar_metrics(v, p))
        return out

    # Avoid expanding huge prediction arrays.
    if isinstance(obj, (list, tuple)) and len(obj) <= 50:
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            out.extend(scalar_metrics(v, p))
        return out

    return out

def group_summary(df, keys):
    x = (
        df.groupby(keys + ["status"], dropna=False)
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )

    wanted = ["success", "failed", "running", "pending", "broken", "unknown"]
    for c in wanted:
        if c not in x.columns:
            x[c] = 0

    x["total"] = x[wanted].sum(axis=1)
    x["left"] = x["total"] - x["success"]
    x["success_pct"] = np.where(
        x["total"] > 0,
        100.0 * x["success"] / x["total"],
        np.nan
    )

    return x[keys + [
        "total", "success", "failed", "running",
        "pending", "broken", "unknown",
        "left", "success_pct"
    ]]

def runtime_summary(df, keys):
    q = df[
        (df["status"] == "success")
        & df["elapsed_min"].notna()
    ].copy()

    if q.empty:
        return pd.DataFrame()

    def one(g):
        a = g["elapsed_min"].astype(float)
        return pd.Series({
            "n": len(a),
            "min_min": a.min(),
            "median_min": a.median(),
            "mean_min": a.mean(),
            "p90_min": a.quantile(.90),
            "max_min": a.max(),
            "total_worker_hours": a.sum() / 60.0,
        })

    z = q.groupby(keys, dropna=False).apply(one).reset_index()

    numeric = [
        "min_min", "median_min", "mean_min",
        "p90_min", "max_min", "total_worker_hours"
    ]
    z[numeric] = z[numeric].round(3)
    return z

# ---------------------------------------------------------------------
# Load plan
# ---------------------------------------------------------------------
if not PLAN.exists():
    raise SystemExit(f"Missing plan: {PLAN}")

plan = pd.read_csv(PLAN, keep_default_na=False)

if "job_id" not in plan.columns:
    raise SystemExit("jobs.csv has no job_id column")

rows = []

# ---------------------------------------------------------------------
# Harmonize all jobs
# ---------------------------------------------------------------------
for _, job in plan.iterrows():
    job_id = str(job["job_id"])
    run_dir = EXP / "runs" / job_id
    sp = run_dir / "status.json"

    st = safe_json(sp) if sp.exists() else {}

    status = str(st.get("status", "pending" if not sp.exists() else "unknown"))

    started = epoch(st.get("started"))
    finished = epoch(st.get("finished"))

    if status == "running" and not pd.isna(started):
        elapsed_s = NOW_TS - started
    elif not pd.isna(started) and not pd.isna(finished):
        elapsed_s = finished - started
    else:
        elapsed_s = np.nan

    pid = st.get("pid", "")
    alive = pid_alive(pid) if str(pid).strip() else False

    err = clean_error(
        st.get("error")
        or st.get("exception")
        or st.get("message")
        or ""
    )

    # Count files belonging to this run.
    n_files = 0
    bytes_total = 0
    if run_dir.exists():
        try:
            for f in run_dir.rglob("*"):
                if f.is_file():
                    n_files += 1
                    try:
                        bytes_total += f.stat().st_size
                    except Exception:
                        pass
        except Exception:
            pass

    base = job.to_dict()
    base.update({
        "status": status,
        "pid": pid,
        "pid_alive": alive,
        "started_epoch": started,
        "finished_epoch": finished,
        "started_time": fmt_time(started),
        "finished_time": fmt_time(finished),
        "elapsed_sec": elapsed_s,
        "elapsed_min": elapsed_s / 60.0 if not pd.isna(elapsed_s) else np.nan,
        "elapsed_hours": elapsed_s / 3600.0 if not pd.isna(elapsed_s) else np.nan,
        "error_group": classify_error(err),
        "error": err,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "status_exists": sp.exists(),
        "run_file_count": n_files,
        "run_bytes": bytes_total,
        "run_mib": bytes_total / 1024**2,
    })

    rows.append(base)

jobs = pd.DataFrame(rows)

# Common sort.
sort_cols = [
    c for c in ["method", "dataset", "seed", "job_id"]
    if c in jobs.columns
]
jobs = jobs.sort_values(sort_cols).reset_index(drop=True)

write_csv(jobs, "00_jobs_harmonized.csv")

# ---------------------------------------------------------------------
# Overall status
# ---------------------------------------------------------------------
overall = (
    jobs["status"]
    .value_counts(dropna=False)
    .rename_axis("status")
    .reset_index(name="count")
)
overall["pct"] = (100 * overall["count"] / len(jobs)).round(3)
write_csv(overall, "01_overall_status.csv")

# ---------------------------------------------------------------------
# Coverage summaries
# ---------------------------------------------------------------------
method_summary = group_summary(jobs, ["method"])
dataset_summary = group_summary(jobs, ["dataset"])
method_dataset_summary = group_summary(jobs, ["method", "dataset"])

write_csv(method_summary, "02_method_summary.csv")
write_csv(dataset_summary, "03_dataset_summary.csv")
write_csv(method_dataset_summary, "04_method_dataset_summary.csv")

# Coverage matrix success/total
cm = method_dataset_summary.copy()
cm["coverage"] = (
    cm["success"].astype(str)
    + "/"
    + cm["total"].astype(str)
)
coverage_matrix = cm.pivot(
    index="method",
    columns="dataset",
    values="coverage"
).reset_index()
write_csv(coverage_matrix, "05_coverage_matrix_success_over_total.csv")

# ---------------------------------------------------------------------
# Runtime statistics
# ---------------------------------------------------------------------
success_runtime = jobs[
    (jobs.status == "success")
    & jobs.elapsed_min.notna()
].copy()

runtime_cols = [
    c for c in [
        "job_id", "method", "dataset", "seed",
        "status", "started_time", "finished_time",
        "elapsed_sec", "elapsed_min", "elapsed_hours"
    ] if c in success_runtime.columns
]
write_csv(success_runtime[runtime_cols], "06_success_runtime_jobs.csv")

r_method = runtime_summary(jobs, ["method"])
r_dataset = runtime_summary(jobs, ["dataset"])
r_md = runtime_summary(jobs, ["method", "dataset"])

write_csv(r_method, "07_runtime_by_method.csv")
write_csv(r_dataset, "08_runtime_by_dataset.csv")
write_csv(r_md, "09_runtime_by_method_dataset.csv")

# ---------------------------------------------------------------------
# Failed / incomplete / running
# ---------------------------------------------------------------------
failed = jobs[jobs.status == "failed"].copy()
running = jobs[jobs.status == "running"].copy()
incomplete = jobs[jobs.status != "success"].copy()

detail_cols = [
    c for c in [
        "job_id", "method", "dataset", "seed",
        "status", "pid", "pid_alive",
        "started_time", "finished_time",
        "elapsed_hours", "error_group", "error"
    ] if c in jobs.columns
]

write_csv(failed[detail_cols], "10_failed_jobs.csv")
write_csv(running[detail_cols], "11_running_jobs.csv")
write_csv(incomplete[detail_cols], "12_incomplete_jobs.csv")

if not failed.empty:
    fg = (
        failed.groupby(["method", "error_group"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["method", "count"], ascending=[True, False])
    )
else:
    fg = pd.DataFrame(columns=["method", "error_group", "count"])

write_csv(fg, "13_failure_groups.csv")

if not failed.empty:
    fgd = (
        failed.groupby(
            ["method", "dataset", "error_group"],
            dropna=False
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            ["method", "dataset", "count"],
            ascending=[True, True, False]
        )
    )
else:
    fgd = pd.DataFrame(
        columns=["method", "dataset", "error_group", "count"]
    )

write_csv(fgd, "14_failure_groups_by_dataset.csv")

# ---------------------------------------------------------------------
# Artifact inventory
# ---------------------------------------------------------------------
artifacts = []

for _, j in jobs.iterrows():
    run_dir = ROOT / str(j["run_dir"])

    if not run_dir.exists():
        continue

    for f in run_dir.rglob("*"):
        if not f.is_file():
            continue

        try:
            st = f.stat()
        except Exception:
            continue

        artifacts.append({
            "job_id": j["job_id"],
            "method": j.get("method", ""),
            "dataset": j.get("dataset", ""),
            "seed": j.get("seed", ""),
            "status": j["status"],
            "path": str(f.relative_to(ROOT)),
            "filename": f.name,
            "suffix": f.suffix.lower(),
            "bytes": st.st_size,
            "mib": st.st_size / 1024**2,
            "mtime_epoch": st.st_mtime,
            "mtime": fmt_time(st.st_mtime),
        })

artifact_df = pd.DataFrame(artifacts)
write_csv(artifact_df, "15_artifact_inventory.csv")

# ---------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------
metric_rows = []
metric_words = (
    "metric", "result", "eval", "score",
    "summary", "performance"
)

for _, j in jobs.iterrows():
    run_dir = ROOT / str(j["run_dir"])
    if not run_dir.exists():
        continue

    candidates = []
    for f in run_dir.rglob("*"):
        if not f.is_file():
            continue
        low = f.name.lower()
        if f.suffix.lower() not in {".json", ".csv"}:
            continue
        if f.name == "status.json":
            continue
        if any(w in low for w in metric_words):
            candidates.append(f)

    for f in candidates:
        rel = str(f.relative_to(run_dir))

        try:
            if f.suffix.lower() == ".json":
                obj = safe_json(f)
                pairs = scalar_metrics(obj)

                for key, value in pairs:
                    metric_rows.append({
                        "job_id": j["job_id"],
                        "method": j.get("method", ""),
                        "dataset": j.get("dataset", ""),
                        "seed": j.get("seed", ""),
                        "status": j["status"],
                        "source": rel,
                        "metric": key,
                        "metric_id": f"{rel}::{key}",
                        "value": value,
                    })

            elif f.suffix.lower() == ".csv":
                # Do not accidentally load enormous prediction dumps.
                if f.stat().st_size > 20 * 1024**2:
                    continue

                x = pd.read_csv(f)

                if len(x) == 1:
                    for c in x.columns:
                        try:
                            v = float(x.iloc[0][c])
                            if math.isfinite(v):
                                metric_rows.append({
                                    "job_id": j["job_id"],
                                    "method": j.get("method", ""),
                                    "dataset": j.get("dataset", ""),
                                    "seed": j.get("seed", ""),
                                    "status": j["status"],
                                    "source": rel,
                                    "metric": str(c),
                                    "metric_id": f"{rel}::{c}",
                                    "value": v,
                                })
                        except Exception:
                            pass

                else:
                    lower = {str(c).lower(): c for c in x.columns}

                    if "metric" in lower and "value" in lower:
                        mc = lower["metric"]
                        vc = lower["value"]

                        for _, rr in x.iterrows():
                            try:
                                v = float(rr[vc])
                                if math.isfinite(v):
                                    key = str(rr[mc])
                                    metric_rows.append({
                                        "job_id": j["job_id"],
                                        "method": j.get("method", ""),
                                        "dataset": j.get("dataset", ""),
                                        "seed": j.get("seed", ""),
                                        "status": j["status"],
                                        "source": rel,
                                        "metric": key,
                                        "metric_id": f"{rel}::{key}",
                                        "value": v,
                                    })
                            except Exception:
                                pass
        except Exception:
            pass

metrics_long = pd.DataFrame(metric_rows)

if not metrics_long.empty:
    metrics_long = metrics_long.sort_values(
        ["method", "dataset", "seed", "metric_id"]
    )
write_csv(metrics_long, "16_metrics_long.csv")

if not metrics_long.empty:
    mw = (
        metrics_long.pivot_table(
            index=["job_id", "method", "dataset", "seed", "status"],
            columns="metric_id",
            values="value",
            aggfunc="last",
        )
        .reset_index()
    )
else:
    mw = pd.DataFrame()

write_csv(mw, "17_metrics_wide.csv")

if not metrics_long.empty:
    ms = (
        metrics_long[
            metrics_long.status == "success"
        ]
        .groupby(
            ["method", "dataset", "metric_id"],
            dropna=False
        )["value"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
else:
    ms = pd.DataFrame()

write_csv(ms, "18_metrics_summary.csv")

# ---------------------------------------------------------------------
# Checkpoint inventory
# ---------------------------------------------------------------------
checkpoint_roots = [
    ROOT / "checkpoints",
    EXP / "checkpoints",
]

checkpoint_rows = []
seen = set()

for cr in checkpoint_roots:
    if not cr.exists():
        continue

    for f in cr.rglob("*"):
        if not f.is_file():
            continue

        if f.suffix.lower() not in {".ckpt", ".pt", ".pth"}:
            continue

        rp = str(f.resolve())
        if rp in seen:
            continue
        seen.add(rp)

        st = f.stat()

        epoch_n = np.nan
        step_n = np.nan
        version_n = np.nan

        m = re.search(
            r"epoch=(\d+)-step=(\d+)(?:-v(\d+))?",
            f.name
        )
        if m:
            epoch_n = int(m.group(1))
            step_n = int(m.group(2))
            if m.group(3) is not None:
                version_n = int(m.group(3))

        checkpoint_rows.append({
            "path": str(f.relative_to(ROOT)),
            "filename": f.name,
            "bytes": st.st_size,
            "mib": st.st_size / 1024**2,
            "mtime_epoch": st.st_mtime,
            "mtime": fmt_time(st.st_mtime),
            "epoch": epoch_n,
            "global_step_from_name": step_n,
            "version_from_name": version_n,
        })

ckpt = pd.DataFrame(checkpoint_rows)

if not ckpt.empty:
    ckpt = ckpt.sort_values("mtime_epoch")

write_csv(ckpt, "19_checkpoint_inventory.csv")

# ---------------------------------------------------------------------
# Candidate mapping: failed DECAF jobs -> nearby Lightning checkpoint
# ---------------------------------------------------------------------
candidates = []

if not ckpt.empty:
    lightning = ckpt[
        ckpt["filename"].astype(str).str.endswith(".ckpt")
    ].copy()

    decaf = jobs[
        (jobs["method"] == "decaf")
        & jobs["finished_epoch"].notna()
    ].copy()

    for _, j in decaf.iterrows():
        if lightning.empty:
            continue

        diffs = (
            lightning["mtime_epoch"].astype(float)
            - float(j["finished_epoch"])
        ).abs()

        idx = diffs.idxmin()
        c = lightning.loc[idx]
        delta = float(diffs.loc[idx])

        candidates.append({
            "job_id": j["job_id"],
            "dataset": j["dataset"],
            "seed": j["seed"],
            "status": j["status"],
            "finished_time": j["finished_time"],
            "checkpoint": c["path"],
            "checkpoint_time": c["mtime"],
            "checkpoint_epoch": c["epoch"],
            "checkpoint_step": c["global_step_from_name"],
            "time_difference_sec": round(delta, 3),
            "very_likely_same_job": delta <= 120,
        })

decaf_ckpt = pd.DataFrame(candidates)

if not decaf_ckpt.empty:
    decaf_ckpt = decaf_ckpt.sort_values(
        ["dataset", "seed"]
    )

write_csv(decaf_ckpt, "20_decaf_checkpoint_candidates.csv")

# ---------------------------------------------------------------------
# Git snapshot
# ---------------------------------------------------------------------
def git(args):
    try:
        return subprocess.check_output(
            ["git"] + args,
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""

git_head = git(["rev-parse", "HEAD"])
git_branch = git(["branch", "--show-current"])
git_status = git(["status", "--porcelain"])

snapshot = {
    "generated": NOW.isoformat(),
    "project_root": str(ROOT),
    "experiment_root": str(EXP),
    "plan_jobs": int(len(jobs)),
    "git_head": git_head,
    "git_branch": git_branch,
    "git_dirty": bool(git_status),
    "status_counts": {
        str(k): int(v)
        for k, v in jobs.status.value_counts().items()
    },
}

(STATS / "21_snapshot.json").write_text(
    json.dumps(snapshot, indent=2, ensure_ascii=False)
)

# ---------------------------------------------------------------------
# Human-readable README
# ---------------------------------------------------------------------
counts = jobs.status.value_counts().to_dict()
success = int(counts.get("success", 0))
failed_n = int(counts.get("failed", 0))
running_n = int(counts.get("running", 0))
pending_n = int(counts.get("pending", 0))

readme = []
readme.append("# MultiFairGDT — harmonized experiment statistics")
readme.append("")
readme.append(f"Generated: **{NOW.strftime('%Y-%m-%d %H:%M:%S')}**")
readme.append("")
readme.append("## Global state")
readme.append("")
readme.append(f"- Planned jobs: **{len(jobs)}**")
readme.append(
    f"- Success: **{success} "
    f"({pct(success, len(jobs)):.2f}%)**"
)
readme.append(f"- Failed: **{failed_n}**")
readme.append(f"- Running: **{running_n}**")
readme.append(f"- Pending: **{pending_n}**")
readme.append(
    f"- Not successful yet: **{len(jobs) - success}**"
)
readme.append("")

readme.append("## Methods")
readme.append("")
readme.append(
    "| method | total | success | failed | running | left | success % |"
)
readme.append(
    "|---|---:|---:|---:|---:|---:|---:|"
)

for _, r in method_summary.sort_values("method").iterrows():
    readme.append(
        f"| {r['method']} | {int(r['total'])} | "
        f"{int(r['success'])} | {int(r['failed'])} | "
        f"{int(r['running'])} | {int(r['left'])} | "
        f"{float(r['success_pct']):.1f} |"
    )

readme.append("")
readme.append("## Failure groups")
readme.append("")

if fg.empty:
    readme.append("No failed jobs.")
else:
    readme.append("| method | error group | count |")
    readme.append("|---|---|---:|")

    for _, r in fg.iterrows():
        readme.append(
            f"| {r['method']} | {r['error_group']} | "
            f"{int(r['count'])} |"
        )

readme.append("")
readme.append("## Important files")
readme.append("")
readme.append("- `00_jobs_harmonized.csv` — one row per planned job.")
readme.append("- `04_method_dataset_summary.csv` — experiment coverage.")
readme.append("- `05_coverage_matrix_success_over_total.csv` — compact coverage matrix.")
readme.append("- `07_runtime_by_method.csv` — real successful runtimes.")
readme.append("- `09_runtime_by_method_dataset.csv` — runtime by exact method/dataset.")
readme.append("- `10_failed_jobs.csv` — all current failures and errors.")
readme.append("- `13_failure_groups.csv` — harmonized failure taxonomy.")
readme.append("- `15_artifact_inventory.csv` — files generated by every run.")
readme.append("- `16_metrics_long.csv` — harmonized discovered metrics.")
readme.append("- `17_metrics_wide.csv` — metrics in job × metric form.")
readme.append("- `19_checkpoint_inventory.csv` — all project checkpoints.")
readme.append("- `20_decaf_checkpoint_candidates.csv` — DECAF checkpoint/job matching.")
readme.append("- `21_snapshot.json` — machine-readable snapshot metadata.")
readme.append("")

if not decaf_ckpt.empty:
    likely = decaf_ckpt[
        decaf_ckpt["very_likely_same_job"] == True
    ]

    readme.append("## DECAF checkpoint recovery")
    readme.append("")
    readme.append(
        f"Checkpoint candidates within 120 s of job completion: "
        f"**{len(likely)}**."
    )
    readme.append("")

(STATS / "README.md").write_text(
    "\n".join(readme) + "\n",
    encoding="utf-8"
)

# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------
manifest = []
for f in sorted(STATS.iterdir()):
    if f.is_file():
        manifest.append(
            f"{f.name}\t{f.stat().st_size} bytes"
        )

(STATS / "MANIFEST.txt").write_text(
    "\n".join(manifest) + "\n"
)

# ---------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------
print()
print("=" * 78)
print("STATS BUILD COMPLETE")
print("=" * 78)
print(f"folder : {STATS}")
print(f"jobs   : {len(jobs)}")
print(f"success: {success}")
print(f"failed : {failed_n}")
print(f"running: {running_n}")
print(f"pending: {pending_n}")
print(f"done   : {pct(success, len(jobs)):.2f}%")
print()

print("=== METHODS ===")
print(
    method_summary[
        ["method", "total", "success", "failed", "running", "left", "success_pct"]
    ].to_string(index=False)
)

print()
print("=== FAILURE GROUPS ===")
if fg.empty:
    print("none")
else:
    print(fg.to_string(index=False))

print()
print("=== RUNNING ===")
if running.empty:
    print("none")
else:
    print(
        running[
            ["method", "dataset", "seed", "pid", "pid_alive", "elapsed_hours"]
        ].to_string(index=False)
    )

print()
print("=== LIKELY DECAF CHECKPOINT MATCHES ===")
if decaf_ckpt.empty:
    print("none")
else:
    print(
        decaf_ckpt[
            decaf_ckpt["very_likely_same_job"] == True
        ][
            [
                "dataset", "seed", "status",
                "checkpoint", "time_difference_sec"
            ]
        ].to_string(index=False)
    )

print()
print("README :", STATS / "README.md")
print("TABLE  :", STATS / "00_jobs_harmonized.csv")
print("=" * 78)
