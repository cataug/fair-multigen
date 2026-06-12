Вот полный **README.md** в одном окне, готовый к копированию в файл:

```markdown
# MAF-TabGen: Multi-Attribute Fairness-Aware Tabular Data Generation

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

MAF-TabGen is a Python-based framework for **multi-attribute fairness-aware synthetic tabular data generation**.  
It extends tree-based tabular generators (like GDT) to support **multi-attribute and non-binary sensitive features** with multiple fairness constraints (e.g., Equalized Odds, Demographic Parity).

This repository contains **code, datasets preprocessing scripts, experiments, and evaluation pipelines** for reproducing ICDM-style experiments on fairness-utility trade-offs.

---

## Features

- Generate synthetic tabular datasets while enforcing **multi-attribute fairness**.
- Supports **single-attribute** and **intersectional** fairness settings.
- Implements **soft leaf resampling** in decision-tree-based generators.
- Evaluate synthetic data with:
  - Predictive utility (AUC, F1, Balanced Accuracy)
  - Fairness metrics (DP Diff, EO Diff, Avg Odds Diff)
  - Synthetic quality metrics (KS, TVD, correlation distance, detection AUC)
- Parallel CPU execution for large-scale experiments.
- Optional integration with **CTGAN** and **TVAE** (via SDV).

---

## Repository Structure

```

DataGenaration/
├── processed/               # Preprocessed datasets (Adult, COMPAS, etc.)
├── RESULTS/                 # Experiment results and plots
│   ├── icdm_fairness_final_cpu/
│   └── icdm_fairness_extra_series_cpu/
├── scripts/                 # Experiment scripts, synthetic generators
├── notebooks/               # Optional analysis notebooks
├── run_icdm_extra_series_cpu.py   # Main CPU experiment runner
└── README.md

````

---

## Installation

Clone via SSH:

```bash
git clone git@github.com:cataug/fair-multigen.git
cd fair-multigen
````

Set up environment (Python ≥ 3.10 recommended):

```bash
python -m venv .venv_a100
source .venv_a100/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Optional: install SDV for CTGAN/TVAE baselines:
>
> ```bash
> pip install sdv
> ```

---

## Usage

### 1. Preprocess datasets

Processed datasets are already included in `processed/`.
If you need to recreate:

```bash
python scripts/preprocess_datasets.py
```

### 2. Run experiments (CPU)

```bash
python run_icdm_extra_series_cpu.py
```

This runs the **extra experimental series**:

* Lambda sweep for fairness-utility trade-offs
* Minimum group support sensitivity
* Intersectional stress tests
* Runtime scaling
* Optional SDV baselines (CTGAN, TVAE)

Results will be saved in:

```
RESULTS/icdm_fairness_extra_series_cpu/
```

---

### 3. Evaluate and plot

After running:

```bash
# Aggregate results and generate summary tables + plots
python run_icdm_extra_series_cpu.py  # runs aggregation and plotting automatically
```

Generated outputs:

* `extra_all_results.csv` — all runs
* `*_summary_mean_std.csv` — series-specific summaries
* `plots/` — fairness-utility Pareto plots, stress test plots, runtime scaling plots
* `latex_tables/` — ready-to-use LaTeX tables for ICDM-style papers

---

### Example: Generating synthetic data

```python
from scripts.synthetic_generators import generate_method, load_dataset

df = load_dataset("adult")
sensitive_cols = ["sex", "race", "age_group"]

synth = generate_method(
    train_df=df,
    method="our_multi_fair_gdt",
    n_samples=len(df),
    seed=42,
    sensitive_cols=sensitive_cols,
    lambda_fair=1.0
)
```

---

## Metrics

* **Utility**: ROC-AUC, F1-score, Balanced Accuracy
* **Fairness**:

  * Demographic Parity Difference (DP Diff)
  * Equalized Odds Difference (EO Diff)
  * Average Odds Difference
* **Synthetic Quality**:

  * Numeric KS distance
  * Categorical TVD
  * Correlation distance
  * Detection AUC (real vs synthetic classifier)



---

## License

This repository is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.


