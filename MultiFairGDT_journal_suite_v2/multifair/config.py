from pathlib import Path
from typing import Dict, Any
import math

SEEDS_MAIN = [42, 43, 44, 45, 46]
SEEDS_SWEEP = [42, 43, 44]
DEV_DATASETS = [
    "adult",
    "compas",
 "default_credit",
    "bank_marketing",
    "communities_crime",
    "german_credit",
]
TRANSFER_DATASETS = ["census_income_kdd", "diabetes130", "law_school"]

DEFAULTS = {
    "lambda_fair": 1.0,
    "lambda_u": 0.5,
    "alpha": 1.0,
    "eta": 0.05,
    "epsilon": 0.025,
    "update_batch": 512,
    "candidate_count": 16,
    "constraint_mode": "joint",   # joint = DP + EO
    "adapt": True,
    "group_mode": "joint",        # joint intersection
    "penalty_mode": "incremental",
    "feature_correction": True,
    "max_depth": 7,
    "min_samples_leaf": 40,
}

# One-factor-at-a-time sweeps. update_batch is kept fixed; candidate_count is
# more directly tied to the soft leaf-resampling approximation.
SWEEPS = {
    "lambda_fair": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
    "lambda_u": [0.0, 0.25, 0.5, 1.0, 2.0],
    "alpha": [0.0, 0.25, 0.5, 1.0, 2.0, 5.0],
    "eta": [0.01, 0.025, 0.05, 0.1, 0.2],
    "epsilon": [0.0, 0.01, 0.025, 0.05, 0.1],
    "candidate_count": [8, 16, 32, 64],
}

RUNTIME_SIZES = {
    "adult": [5000, 10000, 25000, "full"],
    "diabetes130": [5000, 10000, 25000, 50000, "full"],
    "census_income_kdd": [5000, 10000, 25000, 50000, 100000, "full"],
}

GENERIC_BASELINES = ["bootstrap", "gaussian_copula", "ctgan", "tvae", "tabddpm"]

FAIR_BASELINE_DATASETS = [
    "adult",
    "compas",
    "default_credit",
    "german_credit",
    "census_income_kdd",
    "diabetes130",
    "law_school",
]
FAIR_BASELINES = ["tabfairgan", "decaf"]

ABLATIONS = {
    "target_only": {"feature_correction": False},
    "no_adapt": {"adapt": False},
    "no_smooth": {"alpha": 0.0},
    "marginal": {"group_mode": "marginal"},
    "no_util": {"lambda_u": 0.0},
    "absolute": {"penalty_mode": "absolute"},
}

CONSTRAINT_VARIANTS = {
    "dp_only": {"constraint_mode": "dp"},
    "eo_only": {"constraint_mode": "eo"},
}

DATASETS: Dict[str, Dict[str, Any]] = {
    "adult": {
        "file": "adult.csv",
        "groups": [
            "__group_sex__", "__group_race__", "__group_age__", "__group_marital__"
        ],
        "group_sources": ["sex", "race", "age", "marital-status"],
        "age_source": "age",
        "age_bins": [0, 25, 35, 45, 55, 65, math.inf],
        "age_labels": ["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
        "drop_features": [],
    },
    "compas": {
        "file": "compas.csv",
        "groups": ["__group_sex__", "__group_race__", "__group_age__"],
        "group_sources": ["sex", "race", "age"],
        "age_source": "age",
        "age_bins": [0, 25, 35, 45, 55, 65, math.inf],
        "age_labels": ["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
        "drop_features": [
            "name", "first", "last", "dob", "compas_screening_date",
            "c_jail_in", "c_jail_out", "c_offense_date", "c_arrest_date",
            "r_offense_date", "r_jail_in", "r_jail_out", "vr_offense_date",
        ],
    },
    "default_credit": {
        "file": "default_credit.csv",
        "groups": [
            "__group_sex__", "__group_age__", "__group_marriage__", "__group_education__"
        ],
        "group_sources": ["SEX", "AGE", "MARRIAGE", "EDUCATION"],
        "age_source": "AGE",
        "age_bins": [0, 25, 35, 45, 55, 65, math.inf],
        "age_labels": ["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
        "drop_features": ["ID"],
    },
    "bank_marketing": {
        "file": "bank_marketing.csv",
        "groups": ["__group_age__", "__group_marital__", "__group_education__"],
        "group_sources": ["age", "marital", "education"],
        "age_source": "age",
        "age_bins": [0, 25, 35, 45, 55, 65, math.inf],
        "age_labels": ["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
        "drop_features": [],
    },
    "communities_crime": {
        "file": "communities_crime.csv",
        "groups": [
            "__group_racepctblack__", "__group_racepctwhite__",
            "__group_racepctasian__", "__group_racepcthisp__"
        ],
        "group_sources": [
            "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp"
        ],
        "quantile_groups": True,
        "drop_features": [
            "ViolentCrimesPerPop_raw", "communityname", "county", "community", "fold"
        ],
    },
    "german_credit": {
        "file": "german_credit.csv",
        "groups": ["__group_sex__", "__group_age__", "__group_foreign_worker__"],
        "group_sources": ["personal_status_sex", "age", "foreign_worker"],
        "age_source": "age",
        "age_bins": [0, 25, 35, 45, 55, 65, math.inf],
        "age_labels": ["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
        "sex_map": {"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"},
        "drop_features": [],
    },
    "census_income_kdd": {
        "file": "census_income_kdd.csv",
        "groups": [
            "__group_sex__", "__group_race__", "__group_age__", "__group_marital__"
        ],
        "group_sources": ["sex", "race", "age", "marital stat"],
        "age_source": "age",
        "age_bins": [0, 25, 35, 45, 55, 65, math.inf],
        "age_labels": ["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
        "drop_features": [],
    },
    "diabetes130": {
        "file": "diabetes130.csv",
        "groups": ["__group_gender__", "__group_race__", "__group_age__"],
        "group_sources": ["gender", "race", "age"],
        "drop_features": ["encounter_id", "patient_nbr"],
        "group_split": "patient_nbr",
    },
    "law_school": {
        "file": "law_school.csv",
        "groups": ["__group_gender__", "__group_race__", "__group_age__"],
        "group_sources": ["gender", "race1", "age"],
        "age_source": "age",
        "age_bins": [0, 23, 26, 30, 35, 45, math.inf],
        "age_labels": ["<23", "23-25", "26-29", "30-34", "35-44", "45+"],
        "drop_features": [],
    },
}

ALL_DATASETS = list(DATASETS)
