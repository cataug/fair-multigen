# MultiFairGDT journal experiment suite

This suite assumes the project root is:

    /home/tahiti/MultiFairGDT

and that the nine processed CSV files already exist under `data/processed/`.

## What the plan contains

The experiment design keeps all 9 datasets in the main paper.

Core:
- GDT: one synthetic table per dataset/seed; evaluate every k.
- SingleFair: one synthetic table per dataset/seed; fairness correction uses the first group; evaluate every k.
- MultiFair: regenerate for every k=1..kmax.

Generic baselines:
- bootstrap
- Gaussian Copula
- CTGAN
- TVAE
- TabDDPM

Single-sensitive fairness baselines:
- TabFairGAN
- DECAF

Ablations at kmax:
- no adaptive lambda
- no smoothing
- marginal groups rather than intersections
- no extra utility term
- absolute rather than incremental candidate penalty

Constraint modes:
- DP only
- EO only
- joint DP+EO (the core method)

Parameter sweeps on the original six development datasets, seeds 42/43/44:
- lambda_fair = 0, .1, .25, .5, 1, 2, 4
- lambda_u = 0, .25, .5, 1, 2
- alpha = 0, .25, .5, 1, 2, 5
- eta = .01, .025, .05, .1, .2
- epsilon = 0, .01, .025, .05, .1
- update_batch = 128, 256, 512, 1024

Runtime scaling:
- Adult: 5k, 10k, 25k, full
- Diabetes: 5k, 10k, 25k, 50k, full
- Census KDD: 5k, 10k, 25k, 50k, 100k, full
- GDT and MultiFair, seeds 42/43/44.

## Run count

The naive family-by-family design contains 1,431 synthetic-generator cells.

The plan builder merges:
- the default sweep point with core MultiFair;
- alpha=0 with the no-smoothing ablation;
- lambda_u=0 with the no-utility ablation.

This leaves 1,377 unique synthetic-generator jobs.

There are also 45 real-data reference evaluation jobs (9 datasets x 5 seeds),
so the execution plan contains 1,422 unique jobs.

The three downstream classifiers (LR, RF, XGBoost) are evaluated from each
generated table and DO NOT multiply the number of generator runs.

## Installation

Do not modify the system Python.

Core environment (the existing project `.venv` is fine):

    cd /home/tahiti/MultiFairGDT
    source .venv/bin/activate
    python -m pip install -r requirements-core.txt

For deep baselines, a second environment is recommended because SDV/Synthcity
may impose different dependency versions:

    cd /home/tahiti/MultiFairGDT
    python3 -m venv .venv_deep
    source .venv_deep/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements-deep.txt

Both environments write the same result format, so core/deep families can be
run separately and aggregated afterwards.

## First checks

From the project root:

    source .venv/bin/activate

    python -m py_compile \
      multifair/config.py \
      multifair/data.py \
      multifair/tree_generator.py \
      multifair/baselines.py \
      multifair/evaluate.py \
      multifair/plan.py \
      multifair/runner.py \
      multifair/aggregate.py \
      scripts/journal_experiments.py

Create immutable split-index files:

    python scripts/journal_experiments.py make-splits

Create the complete de-duplicated job plan:

    python scripts/journal_experiments.py plan

Smoke test:

    python scripts/journal_experiments.py smoke --dataset adult --seed 42

## Recommended execution order

Run foreground jobs with live output.

1. Real reference:

    python scripts/journal_experiments.py run --family reference

2. Core GDT / SingleFair / MultiFair:

    python scripts/journal_experiments.py run --family core

3. Cheap/generic baselines. Bootstrap first:

    python scripts/journal_experiments.py run --family generic_baselines --method bootstrap

Then Gaussian Copula:

    python scripts/journal_experiments.py run --family generic_baselines --method gaussian_copula

Then CTGAN/TVAE/TabDDPM from the deep environment:

    python scripts/journal_experiments.py run --family generic_baselines --method ctgan
    python scripts/journal_experiments.py run --family generic_baselines --method tvae
    python scripts/journal_experiments.py run --family generic_baselines --method tabddpm

4. Dedicated fair baselines:

    python scripts/journal_experiments.py run --family fair_baselines --method tabfairgan
    python scripts/journal_experiments.py run --family fair_baselines --method decaf

5. Ablations:

    python scripts/journal_experiments.py run --family ablation

6. Constraint variants:

    python scripts/journal_experiments.py run --family constraints

7. Parameter sweeps:

    python scripts/journal_experiments.py run --family sweep

8. Runtime scaling:

    python scripts/journal_experiments.py run --family runtime

9. Aggregate:

    python scripts/journal_experiments.py aggregate

All runs are resumable by default. Successful job IDs are skipped.

## Useful partial commands

Only Adult:

    python scripts/journal_experiments.py run --family core --dataset adult

Only seed 42:

    python scripts/journal_experiments.py run --family core --seed 42

First 5 selected jobs:

    python scripts/journal_experiments.py run --family core --limit 5

Re-run even if cached:

    python scripts/journal_experiments.py run --family core --no-resume

## Output layout

    results/journal2026/
        jobs.csv
        cache/trees/
        runs/<job_id>/
            job.json
            status.json
            metrics.csv
            runtime.csv
        aggregate/
            all_metrics.csv
            all_runtime.csv
            job_status.csv
            summary_by_dataset.csv
            paired_differences.csv

Synthetic tables are not saved by default because Census KDD makes the sweep
outputs enormous. Add `--save-synthetic` only when a specific family/output
needs to be retained.

## Methodological details

- 70/15/15 train/validation/test splits.
- Five primary seeds: 42..46.
- Diabetes is split by patient_nbr to prevent patient overlap.
- Group transforms are re-fit from training data when needed (Communities
  quartile edges) and then applied to validation/test/generated rows.
- Small groups are not discarded.
- The primary fairness reports include mean and max DP/EO/AOD.
- Detection quality uses D_det = 2*|AUC-0.5|, so 0 means indistinguishable.
- The target correction uses a real-train proxy only as an internal surrogate;
  final fairness is always measured on independent real test rows using LR,
  RF, and XGBoost trained on the synthetic table.


## Paper-ready aggregate tables and figures

After all desired families are complete:

    python scripts/journal_experiments.py aggregate
    python scripts/journal_experiments.py paper

The second command writes PDF figures to:

    results/journal2026/figures/

The aggregate command also writes:
- `main_maxk_summary.csv`
- `fair_baseline_k1_summary.csv`
- `k_scaling_summary.csv`
- `ablation_summary.csv`
- `sweep_summary.csv`
- `support_bucket_summary.csv`
- `runtime_summary.csv`
- `statistical_tests.csv`

`statistical_tests.csv` averages seeds within each dataset first and then uses
the datasets as paired units. It reports dataset-bootstrap 95% confidence
intervals for MultiFair-minus-baseline differences and two-sided Wilcoxon
signed-rank p-values with Holm correction.

For DECAF, the Synthcity adapter uses inference-time `biased_edges` to remove
the direct first-sensitive-variable -> target edge in the single-sensitive
comparison. This should be smoke-tested after installing the exact Synthcity
version used for the final paper.
