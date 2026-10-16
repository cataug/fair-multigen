#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

# Allow running directly from project/scripts after copying the suite.
HERE = Path(__file__).resolve()
SUITE = HERE.parent.parent
if str(SUITE) not in sys.path:
    sys.path.insert(0, str(SUITE))

from multifair.config import ALL_DATASETS, SEEDS_MAIN
from multifair.data import project_root, save_split, load_split
from multifair.plan import build_plan, plan_summary
from multifair.runner import run_plan, run_job
from multifair.aggregate import aggregate
from multifair.paper_outputs import make_paper_outputs


def write_plan(root):
    p = root / "results" / "journal2026"
    p.mkdir(parents=True, exist_ok=True)
    df = build_plan()
    path = p / "jobs.csv"
    df.to_csv(path, index=False)
    print(df.groupby(["family", "method"]).size().to_string())
    print()
    print(plan_summary(df))
    print(f"\nWrote {len(df)} unique execution jobs -> {path}")
    return df


def load_plan(root):
    p = root / "results" / "journal2026" / "jobs.csv"
    if not p.exists():
        return write_plan(root)
    return pd.read_csv(p, keep_default_na=False)


def cmd_splits(root):
    rows = []
    for ds in ALL_DATASETS:
        for seed in SEEDS_MAIN:
            info = save_split(root, ds, seed)
            rows.append(info)
            print(
                f"{ds:22s} seed={seed} "
                f"train={info['train']:7d} val={info['val']:6d} test={info['test']:6d} "
                f"pos={info['train_positive']:.4f}/{info['val_positive']:.4f}/{info['test_positive']:.4f}"
            )
    out = root / "manifests" / "split_manifest.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Wrote", out)


def cmd_smoke(root, dataset, seed):
    plan = build_plan()
    z = plan[
        (plan.dataset == dataset)
        & (plan.seed.astype(int) == int(seed))
        & (plan.method == "multifair")
        & (plan.k.astype(int) == 1)
        & (plan.family.astype(str).str.contains("core"))
    ].head(1)
    if z.empty:
        raise RuntimeError("Smoke job not found")
    # Smoke uses a tiny timing-style training subset but still writes its own
    # dedicated job id by overriding runtime fields locally.
    job = z.iloc[0].to_dict()
    job["job_id"] = "smoke_" + str(job["job_id"])
    job["runtime_n"] = 1000
    job["timing_only"] = False
    print("SMOKE:", job)
    st = run_job(root, job, resume=False, save_synthetic=False)
    print(st)
    if st["status"] != "success":
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(
        description="MultiFairGDT journal experiment runner"
    )
    ap.add_argument("--root", default=None)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("plan")
    sub.add_parser("make-splits")

    sm = sub.add_parser("smoke")
    sm.add_argument("--dataset", default="adult")
    sm.add_argument("--seed", type=int, default=42)

    run = sub.add_parser("run")
    run.add_argument("--family", default=None)
    run.add_argument("--method", default=None)
    run.add_argument("--dataset", default=None)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--no-resume", action="store_true")
    run.add_argument("--save-synthetic", action="store_true")

    sub.add_parser("aggregate")
    sub.add_parser("paper")

    args = ap.parse_args()
    root = project_root(args.root)

    if args.command == "plan":
        write_plan(root)
    elif args.command == "make-splits":
        cmd_splits(root)
    elif args.command == "smoke":
        cmd_smoke(root, args.dataset, args.seed)
    elif args.command == "run":
        df = load_plan(root)
        run_plan(
            root=root,
            plan=df,
            family=args.family,
            method=args.method,
            dataset=args.dataset,
            seed=args.seed,
            limit=args.limit,
            resume=not args.no_resume,
            save_synthetic=args.save_synthetic,
        )
    elif args.command == "aggregate":
        aggregate(root)
    elif args.command == "paper":
        make_paper_outputs(root)


if __name__ == "__main__":
    main()
