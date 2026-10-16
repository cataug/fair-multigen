from pathlib import Path
from collections import deque
import subprocess
import pandas as pd
import json
import os
import time
import fcntl
import sys

ROOT = Path("/home/tahiti/MultiFairGDT")
RESULTS = ROOT / "results/journal2026_v21"
PLAN = RESULTS / "jobs.csv"

PYTHON = str(ROOT / ".venv_external/bin/python")
SCRIPT = str(ROOT / "scripts/journal_experiments.py")

GPU_METHODS = {
    "ctgan",
    "tvae",
    "tabddpm",
    "decaf",
    "tabfairgan",
}

# We already measured ~100% GPU with ~3 workers.
MAX_GPU_WORKERS = 8
START_BELOW_UTIL = 95
MEMORY_LIMIT_MIB = 36000
LAUNCH_COOLDOWN = 10
QUIET_BEFORE_START = 30
MAX_ATTEMPTS = 2

env = os.environ.copy()
env.update({
    "CUDA_VISIBLE_DEVICES": "0",
    "POLARS_FORCE_PKG": "compat",
    "MULTIFAIR_RESUME_ROOT":
        str(RESULTS / "checkpoints"),
    "MULTIFAIR_THREADS": "3",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
})


# Prevent two copies of this scheduler.
lock_path = RESULTS / "adaptive_gpu.lock"
lock_path.parent.mkdir(parents=True, exist_ok=True)

lock_handle = open(lock_path, "w")

try:
    fcntl.flock(
        lock_handle,
        fcntl.LOCK_EX | fcntl.LOCK_NB
    )
except BlockingIOError:
    print("Another adaptive scheduler is already running.")
    sys.exit(2)


def status_of(job_id):
    p = RESULTS / "runs" / str(job_id) / "status.json"

    if not p.exists():
        return "pending"

    try:
        return json.loads(
            p.read_text()
        ).get("status", "unknown")
    except Exception:
        return "broken"


def gpu_state():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu="
                "utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()[0]

        util, used, total = [
            int(v.strip())
            for v in out.split(",")
        ]

        return util, used, total

    except Exception:
        return 0, 0, 40960


def existing_external_jobs():
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,args"],
            text=True
        )
    except Exception:
        return []

    return [
        line.strip()
        for line in out.splitlines()
        if "journal_experiments.py run" in line
    ]


print("=" * 78)
print("MULTIFAIR ADAPTIVE RESUME SCHEDULER")
print("GPU target       : 90-100%")
print("Max GPU workers  :", MAX_GPU_WORKERS)
print("Checkpoint root  :", RESULTS / "checkpoints")
print("=" * 78)


# ------------------------------------------------------------
# Wait for any already-running pre-existing jobs.
# ------------------------------------------------------------

quiet_since = None

while True:
    old = existing_external_jobs()

    if old:
        quiet_since = None
        util, used, total = gpu_state()

        print(
            f"[WAIT] existing={len(old)} "
            f"GPU={util}% "
            f"VRAM={used}/{total} MiB",
            flush=True,
        )

    else:
        if quiet_since is None:
            quiet_since = time.time()

        quiet = time.time() - quiet_since

        print(
            f"[WAIT] quiet {quiet:.0f}/"
            f"{QUIET_BEFORE_START}s",
            flush=True,
        )

        if quiet >= QUIET_BEFORE_START:
            break

    time.sleep(10)


plan = pd.read_csv(
    PLAN,
    keep_default_na=False
)

priority = {
    "decaf": 0,
    "tabfairgan": 1,
    "tabddpm": 2,
    "tvae": 3,
    "ctgan": 4,
}

jobs = []

for _, j in plan.iterrows():
    method = str(j["method"])

    if method not in GPU_METHODS:
        continue

    if status_of(j["job_id"]) == "success":
        continue

    jobs.append({
        "job_id": str(j["job_id"]),
        "method": method,
        "dataset": str(j["dataset"]),
        "seed": int(j["seed"]),
    })

jobs.sort(
    key=lambda j: (
        priority.get(j["method"], 99),
        j["dataset"],
        j["seed"],
    )
)

todo = deque(jobs)
active = {}
attempts = {}

print()
print("Remaining GPU jobs:", len(todo))

counts = {}
for j in todo:
    counts[j["method"]] = (
        counts.get(j["method"], 0) + 1
    )
print("By method:", counts)
print()


def launch(j):
    attempts[j["job_id"]] = (
        attempts.get(j["job_id"], 0) + 1
    )

    cmd = [
        PYTHON,
        SCRIPT,
        "run",
        "--method", j["method"],
        "--dataset", j["dataset"],
        "--seed", str(j["seed"]),
    ]

    print()
    print("=" * 78)
    print(
        f"START {j['method']} "
        f"{j['dataset']} "
        f"seed={j['seed']} "
        f"attempt={attempts[j['job_id']]}"
    )
    print("=" * 78)

    p = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
    )

    active[p.pid] = {
        "process": p,
        "job": j,
        "started": time.time(),
    }


util_hist = deque(maxlen=3)
last_launch = 0
last_report = 0

while todo or active:

    # Finished processes.
    for pid in list(active):
        entry = active[pid]
        proc = entry["process"]
        rc = proc.poll()

        if rc is None:
            continue

        j = entry["job"]
        st = status_of(j["job_id"])

        print()
        print(
            f"FINISH pid={pid} "
            f"{j['method']} "
            f"{j['dataset']} "
            f"seed={j['seed']} "
            f"rc={rc} status={st}",
            flush=True,
        )

        del active[pid]

        if (
            st != "success"
            and attempts.get(j["job_id"], 0)
            < MAX_ATTEMPTS
        ):
            print(
                "  -> queued for one retry",
                flush=True,
            )
            todo.append(j)

    util, used, total = gpu_state()
    util_hist.append(util)

    avg_util = (
        sum(util_hist) / len(util_hist)
        if util_hist else 0
    )

    now = time.time()

    if now - last_report >= 20:
        print(
            f"[GPU] now={util:3d}% "
            f"avg={avg_util:5.1f}% "
            f"VRAM={used:5d}/{total} MiB "
            f"active={len(active)} "
            f"waiting={len(todo)}",
            flush=True,
        )
        last_report = now

    # Always keep at least one worker.
    if todo and not active:
        j = todo.popleft()

        if status_of(j["job_id"]) != "success":
            launch(j)
            last_launch = time.time()
            util_hist.clear()

        time.sleep(5)
        continue

    # Add one worker only if GPU is actually underfed.
    if (
        todo
        and len(active) < MAX_GPU_WORKERS
        and used < MEMORY_LIMIT_MIB
        and now - last_launch >= LAUNCH_COOLDOWN
        and avg_util < START_BELOW_UTIL
    ):
        j = todo.popleft()

        if status_of(j["job_id"]) != "success":
            launch(j)
            last_launch = time.time()
            util_hist.clear()

    time.sleep(10)


print()
print("=" * 78)
print("GPU QUEUE FINISHED")
print("=" * 78)


# ------------------------------------------------------------
# Small CPU-only GaussianCopula tail.
# Existing successes are cached and skipped.
# ------------------------------------------------------------

print()
print("===== GAUSSIAN COPULA RETRY =====")

for seed in [42, 43, 44, 45, 46]:
    subprocess.run(
        [
            PYTHON,
            SCRIPT,
            "run",
            "--method", "gaussian_copula",
            "--seed", str(seed),
        ],
        cwd=str(ROOT),
        env=env,
    )


print()
print("===== AGGREGATE =====")

subprocess.run(
    [
        PYTHON,
        SCRIPT,
        "aggregate",
    ],
    cwd=str(ROOT),
    env=env,
)


# ------------------------------------------------------------
# Final state.
# ------------------------------------------------------------

rows = []

for _, j in plan.iterrows():
    rows.append({
        "method": j["method"],
        "status": status_of(j["job_id"]),
    })

df = pd.DataFrame(rows)

print()
print("=== FINAL OVERALL ===")
print(
    df["status"]
    .value_counts()
    .to_string()
)

print()
print("=== FINAL BY METHOD ===")
print(
    df.groupby(["method", "status"])
      .size()
      .unstack(fill_value=0)
      .to_string()
)
