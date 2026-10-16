# MultiFairGDT journal experiment suite v2

Project root expected at:

    /home/tahiti/MultiFairGDT

The nine processed datasets must already exist under `data/processed/`.

## What changed in v2

v2 replaces the narrower target-only correction with the method actually needed
for the journal paper:

- the base GDT samples every feature from the empirical CART leaf proposal;
- full MultiFair additionally performs soft candidate reweighting at the last
  autoregressive source column needed to define the requested k-way group;
- C donor candidates are drawn from the leaf proposal, so the proposal itself
  retains the local empirical q distribution;
- a prefix target CART estimates each candidate's outcome effect;
- DP/EO incremental costs and a local utility-distortion term determine the
  candidate weights;
- the target conditional is then fairness-corrected using the realized groups;
- the old target-only implementation remains as the `target_only` ablation.

The separate heavy HistGradientBoosting proxy used in v1 has been removed. The
already fitted target CART and small prefix target CARTs are used instead.

v2 writes to a new result directory:

    results/journal2026_v21/

so the earlier `results/journal2026/` smoke/GDT runs are left untouched and do
not contaminate the new aggregate results.

## Experiment plan

All 9 datasets remain in the main evaluation.

Core:
- GDT: one synthetic table per dataset/seed; RF evaluation for every k; at kmax
  also LR and XGBoost.
- SingleFair: one synthetic table per dataset/seed; correction uses only the
  first sensitive/grouping attribute; same evaluation policy.
- MultiFair: regenerate for k=1..kmax because the correction itself changes.

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
- target-only correction
- no adaptive lambda
- no smoothing
- marginal rather than intersectional group correction
- no extra utility term
- absolute rather than incremental candidate penalty

Constraint modes:
- DP only
- EO only
- joint DP+EO (default)

OFAT sweeps on the original six development datasets, seeds 42/43/44:
- lambda_fair = 0, .1, .25, .5, 1, 2, 4
- lambda_u = 0, .25, .5, 1, 2
- alpha = 0, .25, .5, 1, 2, 5
- eta = .01, .025, .05, .1, .2
- epsilon = 0, .01, .025, .05, .1
- candidate_count C = 8, 16, 32, 64

Runtime scaling:
- Adult: 5k, 10k, 25k, full
- Diabetes: 5k, 10k, 25k, 50k, full
- Census KDD: 5k, 10k, 25k, 50k, 100k, full
- GDT and MultiFair, seeds 42/43/44.

## Run count

After de-duplicating configurations shared by core/sweep/ablations:

- 1,422 unique synthetic-generator jobs;
- 45 real-data reference evaluation jobs;
- 1,467 total execution jobs.

LR/RF/XGBoost do **not** multiply the generator count. RF is the primary
classifier. LR and XGBoost are added only at maximal intersection for the
core/reference robustness analysis. Generic baselines, sweeps and ablations use
RF only unless explicitly extended later.

## CPU policy

RF and XGBoost use 8 threads by default rather than `n_jobs=-1`.
Override if needed:

    export MULTIFAIR_THREADS=8

This avoids one evaluation consuming every CPU core on the server.

## Install/update into the existing project

From the unpacked suite directory:

    ./install_into_project.sh /home/tahiti/MultiFairGDT

The installer only replaces experiment code and requirement files. It does not
modify `data/raw/`, `data/processed/`, or the existing split indices.

## Core environment

    cd /home/tahiti/MultiFairGDT
    source .venv/bin/activate
    python -m pip install -r requirements-core.txt

Deep baselines should use a separate environment because SDV/Synthcity/PyTorch
may have different dependency requirements:

    cd /home/tahiti/MultiFairGDT
    python3 -m venv .venv_deep
    source .venv_deep/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements-deep.txt

## Syntax check

    cd /home/tahiti/MultiFairGDT
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
      multifair/paper_outputs.py \
      scripts/journal_experiments.py

## Splits

The existing split files can be reused. Regenerating them is deterministic:

    python scripts/journal_experiments.py make-splits

Diabetes is split by `patient_nbr`; the other datasets use stratified row
splits.

## Build v2 plan

    python scripts/journal_experiments.py plan

Expected totals:

    total_execution_jobs      1467
    synthetic_generator_jobs  1422
    real_reference_jobs         45

The job ids contain `suite_version=2`, so they are distinct from v1.

## Smoke test

Run this before any full experiment family:

    python scripts/journal_experiments.py smoke --dataset adult --seed 42

v2 prints stage-level live progress:

    [load]
    [fit]
    [generate]
    [fidelity]      # only where required
    [evaluate]
    [done]

The k=1 smoke uses only RF and 1000 training rows.

## Recommended execution order

All commands run in the foreground and successful job ids are resumable.

Real reference:

    python scripts/journal_experiments.py run --family reference

Core:

    python scripts/journal_experiments.py run --family core

Cheap baselines:

    python scripts/journal_experiments.py run --family generic_baselines --method bootstrap
    python scripts/journal_experiments.py run --family generic_baselines --method gaussian_copula

Deep baselines from `.venv_deep`:

    python scripts/journal_experiments.py run --family generic_baselines --method ctgan
    python scripts/journal_experiments.py run --family generic_baselines --method tvae
    python scripts/journal_experiments.py run --family generic_baselines --method tabddpm

Fair baselines:

    python scripts/journal_experiments.py run --family fair_baselines --method tabfairgan
    python scripts/journal_experiments.py run --family fair_baselines --method decaf

Ablations:

    python scripts/journal_experiments.py run --family ablation

Constraint variants:

    python scripts/journal_experiments.py run --family constraints

Sweeps:

    python scripts/journal_experiments.py run --family sweep

Runtime:

    python scripts/journal_experiments.py run --family runtime

Aggregate and figures:

    python scripts/journal_experiments.py aggregate
    python scripts/journal_experiments.py paper

## Useful partial runs

One dataset:

    python scripts/journal_experiments.py run --family core --dataset adult

One seed:

    python scripts/journal_experiments.py run --family core --seed 42

First five selected jobs:

    python scripts/journal_experiments.py run --family core --limit 5

Force re-run rather than resume:

    python scripts/journal_experiments.py run --family core --no-resume

## Outputs

    results/journal2026_v21/
        jobs.csv
        cache/trees/
        runs/<job_id>/
            job.json
            status.json
            metrics.csv
            runtime.csv
            support_buckets.csv
        aggregate/
            all_metrics.csv
            all_runtime.csv
            all_support_buckets.csv
            job_status.csv
            summary_by_dataset.csv
            main_maxk_summary.csv
            fair_baseline_k1_summary.csv
            k_scaling_summary.csv
            ablation_summary.csv
            sweep_summary.csv
            support_bucket_summary.csv
            runtime_summary.csv
            statistical_tests.csv
        figures/

Synthetic tables are not saved by default; use `--save-synthetic` only for a
specific diagnostic run.

## Metrics

Utility:
- ROC-AUC
- PR-AUC
- balanced accuracy
- macro-F1

Fairness:
- DP mean/max
- EO mean/max
- AOD mean/max
- worst-group balanced accuracy
- worst-group TPR/FPR

Fidelity at the maximal/main settings:
- mean numeric KS statistic
- mean categorical total variation
- Spearman correlation distance
- detection distance D_det = 2*|AUC-0.5|

Sparse-group analysis uses support buckets:
- 1-4
- 5-9
- 10-19
- 20-49
- 50-99
- 100+

No group is discarded merely because its support is small.

## Statistical analysis

Seeds are averaged within each dataset first. Dataset-level means are then used
as the paired units for:

- bootstrap 95% confidence intervals of MultiFair-minus-baseline differences;
- two-sided Wilcoxon signed-rank tests;
- Holm correction over the reported comparisons.

## External fair baselines

TabFairGAN is invoked through its current `TFG` interface with the first binary
sensitive attribute. DECAF is invoked through Synthcity. At generation time,
DECAF receives `biased_edges={"__target__": [first_sensitive_group]}` so its
single-sensitive comparison suppresses the direct sensitive-parent influence
on the target generation path. These adapters should still be smoke-tested in
the exact deep environment before launching all 70 fair-baseline jobs.


## v2.1 methodological safeguards

- Smoke-test observations are tagged `family=smoke` and excluded from aggregation.
- Parameter sweeps are evaluated on the validation split only.
- If a configuration is shared by `core|sweep` or `ablation|sweep`, the
  synthetic table is generated once but evaluated on validation for the sweep
  and on test for the main/ablation result.
- Main tables and statistical tests use `eval_split=test`.
- Sweep summaries use `eval_split=validation`.
- Intersectional fairness metrics retain raw all-group values and additionally
  report support-sensitive variants for group support >=5, >=10, and >=20.
  These thresholds are evaluation diagnostics only; groups are never removed
  from generation or training.
- `lambda_fair_calibration.csv` recommends a validation-only lambda_fair from
  the OFAT lambda sweep under a mean absolute ROC-AUC-loss budget of 0.01.
