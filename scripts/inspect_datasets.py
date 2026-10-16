from pathlib import Path
import csv
import json
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path("/home/tahiti/MultiFairGDT")
RAW = ROOT / "data" / "raw"
OUT = ROOT / "manifests" / "dataset_profile.csv"


def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def clean_strings(df):
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip()
        df[c] = df[c].replace({
            "?": pd.NA,
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
        })
    return df


def find_col(df, candidates):
    lookup = {norm(c): c for c in df.columns}

    for cand in candidates:
        n = norm(cand)
        if n in lookup:
            return lookup[n]

    for cand in candidates:
        n = norm(cand)
        for key, original in lookup.items():
            if n == key or n in key or key in n:
                return original

    return None


def summarize(
    dataset_id,
    df,
    target_candidates,
    sensitive_candidates,
    target_mode="binary",
    notes="",
):
    df = clean_strings(df.copy())

    target = find_col(df, target_candidates)

    sensitive = []
    for candidates in sensitive_candidates:
        c = find_col(df, candidates)
        if c is not None and c != target and c not in sensitive:
            sensitive.append(c)

    n_rows = len(df)
    n_cols = len(df.columns)

    missing_cells = int(df.isna().sum().sum())
    total_cells = int(max(1, n_rows * n_cols))
    missing_fraction = missing_cells / total_cells

    target_cardinality = ""
    target_positive_rate = ""
    target_distribution = ""

    if target is not None:
        vc = df[target].value_counts(dropna=False)
        target_cardinality = int(df[target].nunique(dropna=True))

        target_distribution = json.dumps(
            {
                str(k): int(v)
                for k, v in vc.head(20).items()
            },
            ensure_ascii=False,
        )

        if target_mode == "binary" and target_cardinality == 2:
            vals = [x for x in df[target].dropna().unique()]
            vals_sorted = sorted(vals, key=lambda x: str(x))

            # Inspection only. We do not impose semantic positive class here.
            positive = vals_sorted[-1]
            target_positive_rate = float(
                (df[target] == positive).mean()
            )

    sens_cards = {}
    sens_missing = {}

    for c in sensitive:
        sens_cards[c] = int(df[c].nunique(dropna=True))
        sens_missing[c] = float(df[c].isna().mean())

    smallest_intersection = ""
    intersection_groups = ""

    if len(sensitive) >= 2:
        tmp = df[sensitive].dropna()

        if len(tmp):
            counts = tmp.groupby(
                sensitive,
                dropna=False,
                observed=True
            ).size()

            if len(counts):
                smallest_intersection = int(counts.min())
                intersection_groups = int(len(counts))

    return {
        "dataset_id": dataset_id,
        "rows": n_rows,
        "columns": n_cols,
        "target": target or "",
        "target_mode": target_mode,
        "target_cardinality": target_cardinality,
        "target_positive_rate_inspection_only": target_positive_rate,
        "target_distribution": target_distribution,
        "sensitive_attributes": "|".join(sensitive),
        "sensitive_cardinalities": json.dumps(
            sens_cards, ensure_ascii=False
        ),
        "sensitive_missing_fraction": json.dumps(
            sens_missing, ensure_ascii=False
        ),
        "intersection_group_count": intersection_groups,
        "smallest_complete_intersection_group": smallest_intersection,
        "missing_cells": missing_cells,
        "missing_fraction": missing_fraction,
        "notes": notes,
    }


def load_adult():
    cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    train = pd.read_csv(
        RAW / "adult/adult.data",
        header=None,
        names=cols,
        skipinitialspace=True,
        na_values="?",
    )

    test = pd.read_csv(
        RAW / "adult/adult.test",
        header=None,
        names=cols,
        skiprows=1,
        skipinitialspace=True,
        na_values="?",
    )

    test["income"] = (
        test["income"]
        .astype("string")
        .str.replace(".", "", regex=False)
    )

    df = pd.concat([train, test], ignore_index=True)

    return summarize(
        "adult",
        df,
        ["income"],
        [
            ["sex"],
            ["race"],
            ["age"],
            ["marital-status"],
        ],
        "binary",
        "UCI train and test combined for structural inspection only.",
    )


def load_compas():
    df = pd.read_csv(
        RAW / "compas/compas-scores-two-years.csv"
    )

    return summarize(
        "compas",
        df,
        ["two_year_recid"],
        [
            ["sex"],
            ["race"],
            ["age"],
        ],
        "binary",
        "Raw ProPublica file; standard COMPAS filtering has not yet been applied.",
    )


def load_default_credit():
    path = RAW / "default_credit/default of credit card clients.xls"

    try:
        df = pd.read_excel(path, header=1)
    except ImportError:
        print(
            "\nERROR: reading the .xls file requires xlrd.\n"
            "Install it with:\n"
            "  python3 -m pip install xlrd\n",
            file=sys.stderr,
        )
        raise

    return summarize(
        "default_credit",
        df,
        [
            "default payment next month",
            "default_payment_next_month",
        ],
        [
            ["SEX", "sex"],
            ["AGE", "age"],
            ["MARRIAGE", "marriage"],
            ["EDUCATION", "education"],
        ],
        "binary",
    )


def load_bank():
    df = pd.read_csv(
        RAW / "bank_marketing/bank/bank-full.csv",
        sep=";",
    )

    return summarize(
        "bank_marketing",
        df,
        ["y"],
        [
            ["age"],
            ["marital"],
            ["education"],
        ],
        "binary",
        "Age, marital status, and education are grouping variables here; "
        "their interpretation as protected attributes must be stated explicitly.",
    )


def parse_communities_names():
    names_path = RAW / "communities_crime/communities.names"

    attrs = []

    with names_path.open(
        "r",
        encoding="latin-1",
        errors="ignore"
    ) as f:
        for line in f:
            line = line.strip()

            m = re.match(
                r"@attribute\s+([^\s]+)",
                line,
                flags=re.I,
            )

            if m:
                attrs.append(m.group(1))

    return attrs


def load_communities():
    attrs = parse_communities_names()

    path = RAW / "communities_crime/communities.data"

    if attrs:
        df = pd.read_csv(
            path,
            header=None,
            names=attrs,
            na_values="?",
        )
    else:
        df = pd.read_csv(
            path,
            header=None,
            na_values="?",
        )

    return summarize(
        "communities_crime",
        df,
        ["ViolentCrimesPerPop"],
        [
            ["racePctBlack"],
            ["racePctWhite"],
            ["racePctAsian"],
            ["racePctHisp"],
        ],
        "continuous",
        "Original target is continuous. No classification threshold is imposed during inspection.",
    )


def load_german():
    path = RAW / "german_credit/german.data"

    cols = [
        "status_checking",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings",
        "employment",
        "installment_rate",
        "personal_status_sex",
        "other_debtors",
        "residence_since",
        "property",
        "age",
        "other_installment",
        "housing",
        "existing_credits",
        "job",
        "people_liable",
        "telephone",
        "foreign_worker",
        "credit_risk",
    ]

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=cols,
    )

    return summarize(
        "german_credit",
        df,
        ["credit_risk"],
        [
            ["personal_status_sex"],
            ["age"],
            ["foreign_worker"],
        ],
        "binary",
        "Sex is encoded jointly with personal status in the original UCI coding; "
        "we will derive sex explicitly during preprocessing.",
    )


def load_census_kdd():
    names_path = (
        RAW
        / "census_income_kdd/extracted/census-income.names"
    )

    attrs = []

    with names_path.open(
        "r",
        encoding="latin-1",
        errors="ignore"
    ) as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("|"):
                continue

            if ":" in line:
                left = line.split(":", 1)[0].strip()

                # Skip prose-like/header lines.
                if left and len(left) < 80:
                    attrs.append(left)

    data_path = (
        RAW
        / "census_income_kdd/extracted/census-income.data"
    )

    test_path = (
        RAW
        / "census_income_kdd/extracted/census-income.test"
    )

    # UCI Census Income has 41 predictor fields + target.
    expected_cols = 42

    if len(attrs) == expected_cols - 1:
        cols = attrs + ["income"]
    else:
        cols = None

    train = pd.read_csv(
        data_path,
        header=None,
        names=cols,
        skipinitialspace=True,
        na_values="?",
    )

    test = pd.read_csv(
        test_path,
        header=None,
        names=cols,
        skipinitialspace=True,
        na_values="?",
    )

    df = pd.concat([train, test], ignore_index=True)

    if cols is None:
        # Known positions from the UCI KDD Census schema.
        rename = {
            0: "age",
            7: "marital_status",
            10: "race",
            12: "sex",
            len(df.columns) - 1: "income",
        }
        df = df.rename(columns=rename)

    return summarize(
        "census_income_kdd",
        df,
        ["income", "income_label"],
        [
            ["sex"],
            ["race"],
            ["age"],
            ["marital_status", "marital stat"],
        ],
        "binary",
        "Official UCI train and test combined for structural inspection only.",
    )


def load_diabetes():
    df = pd.read_csv(
        RAW / "diabetes130/diabetic_data.csv",
        na_values=["?", "NULL"],
    )

    return summarize(
        "diabetes130",
        df,
        ["readmitted"],
        [
            ["gender"],
            ["race"],
            ["age"],
        ],
        "multiclass",
        "Raw readmitted target has multiple values; binary transformation will be defined later.",
    )


def load_law_school():
    path = (
        RAW
        / "law_school/law-school-admission-binary.arff"
    )

    try:
        from scipy.io import arff
    except ImportError:
        print(
            "\nERROR: scipy is required for the ARFF file.\n"
            "Install it with:\n"
            "  python3 -m pip install scipy\n",
            file=sys.stderr,
        )
        raise

    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(
                lambda x:
                    x.decode("utf-8", errors="replace")
                    if isinstance(x, bytes)
                    else x
            )

    return summarize(
        "law_school",
        df,
        [
            "target",
            "bar",
            "pass_bar",
            "bar_pass",
            "admit",
        ],
        [
            ["gender", "sex"],
            ["race"],
            ["age"],
        ],
        "binary",
        "Target and sensitive column names are resolved from the downloaded ARFF.",
    )


LOADERS = [
    ("adult", load_adult),
    ("compas", load_compas),
    ("default_credit", load_default_credit),
    ("bank_marketing", load_bank),
    ("communities_crime", load_communities),
    ("german_credit", load_german),
    ("census_income_kdd", load_census_kdd),
    ("diabetes130", load_diabetes),
    ("law_school", load_law_school),
]


def main():
    rows = []

    print("=" * 88)
    print("DATASET INSPECTION")
    print("=" * 88)

    for dataset_id, loader in LOADERS:
        print(f"\n[{dataset_id}]")

        try:
            row = loader()
            rows.append(row)

            print(
                f"  rows={row['rows']:,} "
                f"cols={row['columns']} "
                f"target={row['target'] or 'NOT FOUND'}"
            )
            print(
                f"  sensitive="
                f"{row['sensitive_attributes'] or 'NONE FOUND'}"
            )
            print(
                f"  missing={row['missing_fraction']:.4f}"
            )

            if row["intersection_group_count"] != "":
                print(
                    f"  intersections="
                    f"{row['intersection_group_count']} "
                    f"min_group="
                    f"{row['smallest_complete_intersection_group']}"
                )

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

            rows.append({
                "dataset_id": dataset_id,
                "rows": "",
                "columns": "",
                "target": "",
                "target_mode": "",
                "target_cardinality": "",
                "target_positive_rate_inspection_only": "",
                "target_distribution": "",
                "sensitive_attributes": "",
                "sensitive_cardinalities": "",
                "sensitive_missing_fraction": "",
                "intersection_group_count": "",
                "smallest_complete_intersection_group": "",
                "missing_cells": "",
                "missing_fraction": "",
                "notes": f"LOAD_ERROR: {type(e).__name__}: {e}",
            })

    fieldnames = [
        "dataset_id",
        "rows",
        "columns",
        "target",
        "target_mode",
        "target_cardinality",
        "target_positive_rate_inspection_only",
        "target_distribution",
        "sensitive_attributes",
        "sensitive_cardinalities",
        "sensitive_missing_fraction",
        "intersection_group_count",
        "smallest_complete_intersection_group",
        "missing_cells",
        "missing_fraction",
        "notes",
    ]

    with OUT.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 88)
    print(f"WROTE: {OUT}")
    print("=" * 88)


if __name__ == "__main__":
    main()
