# Fair-MultiGen

### Fairness-aware tabular data generation under multi-group constraints

**Fair-MultiGen** is an experimental framework for studying the trade-off between **predictive utility**, **multi-group fairness**, and **computational cost** in synthetic tabular data generation.

The repository contains the full experimental pipeline used to compare the proposed **MultiFair** approach against classical resampling, statistical synthesizers, and neural generative baselines across multiple fairness-sensitive tabular datasets.

The project is designed around a simple principle:

> A synthetic-data method should not be considered fair merely because group disparities are small.  
> Fairness must be evaluated jointly with predictive utility, worst-group performance, robustness to group-support definitions, and computational cost.

---

## Highlights

- **Multi-group fairness-aware synthetic tabular generation**
- **9 heterogeneous benchmark datasets**
- **11 experimental method families**
- **1,467 planned experimental jobs**
- multi-seed evaluation
- utility, fairness, worst-group, and support-sensitive metrics
- runtime and computational-efficiency analysis
- configuration-level MultiFair sweeps
- Pareto analysis of fairness–utility trade-offs
- harmonized experiment statistics
- publication-ready figures
- reproducible job-level outputs and logs

The repository intentionally keeps **datasets and large model checkpoints outside Git**, while retaining source code, experiment definitions, logs, metrics, statistics, and figures.

---

# Experimental scope

## Datasets

The main journal experiment suite covers:

| Dataset | Domain |
|---|---|
| `adult` | income prediction |
| `bank_marketing` | marketing response |
| `census_income_kdd` | large-scale census income |
| `communities_crime` | community-level crime prediction |
| `compas` | recidivism risk |
| `default_credit` | credit default |
| `diabetes130` | hospital / diabetes outcomes |
| `german_credit` | credit risk |
| `law_school` | law-school outcomes |

Raw datasets are **not committed to the repository**.

Dataset preparation, schema handling, sensitive attributes, and experiment construction are implemented in the project scripts.

---

# Methods

The experiment suite includes the proposed method together with lightweight, statistical, and neural baselines.

| Method | Role |
|---|---|
| **MultiFair** | proposed multi-group fairness-aware method |
| **SingleFair** | single-fairness comparison |
| **GDT** | tree-based generation baseline |
| **Bootstrap** | non-generative resampling baseline |
| **Real** | real-data reference |
| **Gaussian Copula** | statistical synthetic-data baseline |
| **CTGAN** | GAN-based tabular synthesizer |
| **TVAE** | variational tabular synthesizer |
| **TabDDPM** | diffusion-based tabular generation |
| **DECAF** | causality/fairness-oriented neural generator |
| **TabFairGAN** | fairness-oriented GAN baseline |

Not every method is applicable to every dataset or experimental configuration; experiment coverage is recorded explicitly rather than silently imputed.

---

# What is measured?

Fair synthetic data generation is treated as a **multi-objective problem**.

## Predictive utility

Main downstream utility metrics include:

- **Macro-F1**
- **ROC-AUC**
- **PR-AUC**

Higher values are better.

---

## Group fairness

The framework evaluates disparities across sensitive groups using several complementary metrics.

### Demographic Parity disparity

`dp_max`

Measures differences in positive prediction rates across groups.

**Lower is better.**

---

### Equalized Odds disparity

`eo_mean`

Measures differences in error behavior across groups, including group-dependent true-positive and false-positive rates.

**Lower is better.**

---

### Average Odds Difference

`aod_mean`

Summarizes group differences in prediction error rates.

**Lower is better.**

---

## Worst-group performance

Fairness disparities alone can be misleading.

A degenerate classifier can obtain nearly zero disparity simply by predicting almost the same outcome for everyone.

For this reason, the project also reports:

- `worst_group_ba`
- `worst_group_tpr`
- `worst_group_fpr`

with particular emphasis on **worst-group balanced accuracy**.

This separates genuinely useful fairness improvements from trivial low-disparity solutions with poor predictive performance.

---

# Different aggregation levels

A central design choice of this repository is that **datasets are not collapsed into a single cross-dataset average**.

Results are analyzed at several levels:

1. individual experimental jobs;
2. individual random seeds;
3. method × dataset summaries;
4. group-support-specific evaluations;
5. MultiFair configuration-level results;
6. hyperparameter sensitivity;
7. fairness–utility Pareto fronts;
8. fairness–utility–runtime interactions.

This avoids hiding dataset-specific behavior behind a single global score.

---

# Group-support sensitivity

Fairness metrics can become unstable when intersectional groups contain very few examples.

The experiment suite therefore evaluates several minimum-support regimes, including quantities such as:

- `eo_mean_s5`
- `eo_mean_s10`
- `dp_max_s5`
- `dp_max_s10`
- `aod_mean_s5`
- `aod_mean_s10`
- `worst_group_ba_s5`
- `worst_group_ba_s10`

This makes it possible to distinguish:

- broad group-level behavior;
- sparse intersectional effects;
- apparent disparities driven by extremely small groups.

---

# Computational efficiency

Runtime is treated as a first-class experimental quantity rather than an implementation detail.

The archived experiment snapshot shows a very large difference in computational cost between lightweight methods and deep generative baselines.

Representative successful-job runtimes observed in the experiment suite include:

| Method | Median runtime | Mean runtime |
|---|---:|---:|
| Bootstrap | ~0.09 min | ~0.45 min |
| MultiFair | ~0.21 min | ~0.47 min |
| Gaussian Copula | ~0.31 min | ~1.67 min |
| TVAE | ~4.15 min | ~8.21 min |
| CTGAN | ~15.1 min | ~54.7 min |
| TabDDPM | ~12.0 min | ~86.9 min |
| DECAF | ~640 min | ~472 min |

These values should be interpreted together with dataset size: large datasets such as `census_income_kdd` are substantially more expensive for neural synthesizers.

The key comparison is therefore not simply:

> Which method has the lowest fairness disparity?

but rather:

> Which method provides a useful fairness–utility trade-off at a computational cost that is practical?

---

# Archived experiment snapshot

The harmonized experiment snapshot currently stored under `STATS/` contains:

| Quantity | Value |
|---|---:|
| Planned jobs | **1,467** |
| Successful jobs | **1,392** |
| Successful fraction | **94.89%** |
| Failed jobs | **69** |
| Recorded running states | **6** |
| Harmonized metric values | **55,228** |
| Distinct discovered metrics | **56** |

Some recorded `running` states may correspond to stale process metadata from long-running experiments. The raw status files are intentionally preserved for auditability.

The statistics can be regenerated from the current experiment directory rather than relying on these archived numbers.

---

# Repository structure

```text
fair-multigen/
│
├── multifair/
│   └── core implementation and baseline adapters
│
├── scripts/
│   ├── experiment runners
│   ├── resume / scheduling utilities
│   ├── statistics generation
│   └── figure generation
│
├── results/
│   └── journal2026_v21/
│       ├── jobs.csv
│       ├── runs/
│       └── experiment metadata
│
├── STATS/
│   ├── harmonized job tables
│   ├── method × dataset coverage
│   ├── runtime summaries
│   ├── failure summaries
│   ├── metric tables
│   └── checkpoint inventories
│
├── FIGURES/
│   └── exploratory figure suite
│
├── FIGURES_MAIN/
│   └── compact publication-oriented figure suite
│
├── requirements*.txt
├── *.ipynb
└── README.md
