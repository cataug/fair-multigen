from pathlib import Path
import csv
import json
import re

import numpy as np
import pandas as pd
from scipy.io import arff

ROOT = Path("/home/tahiti/MultiFairGDT")
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
MAN = ROOT / "manifests"

OUT.mkdir(parents=True, exist_ok=True)
MAN.mkdir(parents=True, exist_ok=True)

REPORT = MAN / "preprocessing_report.csv"
SUPPORT = MAN / "intersection_support.csv"
SCHEMA = MAN / "processed_schema.csv"

AGE_BINS = [0, 25, 35, 45, 55, 65, np.inf]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]

LSAC_AGE_BINS = [0, 23, 26, 30, 35, 45, np.inf]
LSAC_AGE_LABELS = ["<23", "23-25", "26-29", "30-34", "35-44", "45+"]


def clean_strings(df):
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip()
        df[c] = df[c].replace({
            "?": pd.NA,
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "NULL": pd.NA,
        })
    return df


def add_age_bin(df, source, dest="__group_age__", bins=None, labels=None):
    if bins is None:
        bins = AGE_BINS
    if labels is None:
        labels = AGE_LABELS

    x = pd.to_numeric(df[source], errors="coerce")
    df[dest] = pd.cut(
        x,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    ).astype("string")
    return df


def add_copy_group(df, source, dest):
    df[dest] = df[source].astype("string")
    df.loc[df[source].isna(), dest] = pd.NA
    return df


def normalize_binary(series, positive_values, negative_values=None):
    s = series.astype("string").str.strip()

    pos = {str(x).strip().lower() for x in positive_values}
    neg = None
    if negative_values is not None:
        neg = {str(x).strip().lower() for x in negative_values}

    low = s.str.lower()

    y = pd.Series(pd.NA, index=series.index, dtype="Int64")
    y.loc[low.isin(pos)] = 1

    if neg is None:
        y.loc[~low.isin(pos) & low.notna()] = 0
    else:
        y.loc[low.isin(neg)] = 0

    return y


def support_rows(dataset_id, df, group_cols):
    rows = []

    for k in range(1, len(group_cols) + 1):
        cols = group_cols[:k]
        valid = df[cols].dropna()

        if len(valid) == 0:
            continue

        counts = (
            valid
            .groupby(cols, observed=True, dropna=False)
            .size()
        )

        if len(counts) == 0:
            continue

        rows.append({
            "dataset_id": dataset_id,
            "k": k,
            "group_columns": "|".join(cols),
            "n_complete_rows": int(len(valid)),
            "n_groups": int(len(counts)),
            "min_support": int(counts.min()),
            "median_support": float(counts.median()),
            "mean_support": float(counts.mean()),
            "max_support": int(counts.max()),
            "groups_lt_5": int((counts < 5).sum()),
            "groups_lt_10": int((counts < 10).sum()),
            "groups_lt_20": int((counts < 20).sum()),
            "fraction_groups_lt_5": float((counts < 5).mean()),
            "fraction_groups_lt_10": float((counts < 10).mean()),
            "fraction_groups_lt_20": float((counts < 20).mean()),
        })

    return rows


def finalize(dataset_id, df, group_cols, notes="", target_mapping=""):
    df = df.copy()

    if "__target__" not in df.columns:
        raise RuntimeError(f"{dataset_id}: __target__ not created")

    before = len(df)

    # Target missing values cannot be used in classification experiments.
    df = df.loc[df["__target__"].notna()].copy()
    df["__target__"] = df["__target__"].astype(int)

    # Stable row id after preprocessing.
    df.insert(0, "__row_id__", np.arange(len(df), dtype=np.int64))

    out_path = OUT / f"{dataset_id}.csv"
    df.to_csv(out_path, index=False)

    target_rate = float(df["__target__"].mean())

    group_complete = df[group_cols].dropna() if group_cols else pd.DataFrame()

    if group_cols and len(group_complete):
        counts = (
            group_complete
            .groupby(group_cols, observed=True, dropna=False)
            .size()
        )
        n_groups = int(len(counts))
        min_group = int(counts.min())
        median_group = float(counts.median())
    else:
        n_groups = 0
        min_group = ""
        median_group = ""

    report = {
        "dataset_id": dataset_id,
        "processed_path": str(out_path.relative_to(ROOT)),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "dropped_missing_target": int(before - len(df)),
        "target_positive_rate": target_rate,
        "group_columns": "|".join(group_cols),
        "max_intersection_groups": n_groups,
        "max_intersection_min_support": min_group,
        "max_intersection_median_support": median_group,
        "overall_missing_fraction": float(df.isna().mean().mean()),
        "target_mapping": target_mapping,
        "notes": notes,
    }

    schema = []
    for c in df.columns:
        schema.append({
            "dataset_id": dataset_id,
            "column": c,
            "dtype": str(df[c].dtype),
            "role": (
                "row_id" if c == "__row_id__"
                else "target" if c == "__target__"
                else "group" if c in group_cols
                else "feature"
            ),
            "n_unique": int(df[c].nunique(dropna=True)),
            "missing_fraction": float(df[c].isna().mean()),
        })

    return report, support_rows(dataset_id, df, group_cols), schema


# ---------------------------------------------------------------------
# Adult
# ---------------------------------------------------------------------

def prepare_adult():
    cols = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
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

    df = clean_strings(pd.concat([train, test], ignore_index=True))

    df["__target__"] = normalize_binary(
        df["income"],
        positive_values=[">50K"],
        negative_values=["<=50K"],
    )

    df = add_copy_group(df, "sex", "__group_sex__")
    df = add_copy_group(df, "race", "__group_race__")
    df = add_age_bin(df, "age")
    df = add_copy_group(df, "marital-status", "__group_marital__")

    df = df.drop(columns=["income"])

    groups = [
        "__group_sex__",
        "__group_race__",
        "__group_age__",
        "__group_marital__",
    ]

    return finalize(
        "adult",
        df,
        groups,
        notes="Official UCI train and test combined before later experimental splitting.",
        target_mapping=">50K -> 1; <=50K -> 0",
    )


# ---------------------------------------------------------------------
# COMPAS
# ---------------------------------------------------------------------

def prepare_compas():
    df = pd.read_csv(
        RAW / "compas/compas-scores-two-years.csv",
        low_memory=False,
    )
    df = clean_strings(df)

    raw_n = len(df)

    days = pd.to_numeric(df["days_b_screening_arrest"], errors="coerce")

    mask = (
        days.between(-30, 30, inclusive="both")
        & (pd.to_numeric(df["is_recid"], errors="coerce") != -1)
        & (df["c_charge_degree"].astype("string") != "O")
        & (df["score_text"].astype("string") != "N/A")
    )

    df = df.loc[mask].copy()

    df["__target__"] = pd.to_numeric(
        df["two_year_recid"],
        errors="coerce"
    ).astype("Int64")

    df = add_copy_group(df, "sex", "__group_sex__")
    df = add_copy_group(df, "race", "__group_race__")
    df = add_age_bin(df, "age")

    df = df.drop(columns=["two_year_recid"])

    groups = [
        "__group_sex__",
        "__group_race__",
        "__group_age__",
    ]

    return finalize(
        "compas",
        df,
        groups,
        notes=f"Standard ProPublica-style filtering applied: {raw_n} raw rows before filter.",
        target_mapping="two_year_recid: 1 -> 1; 0 -> 0",
    )


# ---------------------------------------------------------------------
# Default Credit
# ---------------------------------------------------------------------

def prepare_default_credit():
    df = pd.read_excel(
        RAW / "default_credit/default of credit card clients.xls",
        header=1,
    )
    df = clean_strings(df)

    target = "default payment next month"

    df["__target__"] = pd.to_numeric(
        df[target],
        errors="coerce"
    ).astype("Int64")

    df = add_copy_group(df, "SEX", "__group_sex__")
    df = add_age_bin(df, "AGE")
    df = add_copy_group(df, "MARRIAGE", "__group_marriage__")
    df = add_copy_group(df, "EDUCATION", "__group_education__")

    df = df.drop(columns=[target])

    groups = [
        "__group_sex__",
        "__group_age__",
        "__group_marriage__",
        "__group_education__",
    ]

    return finalize(
        "default_credit",
        df,
        groups,
        target_mapping="UCI default-payment label preserved as 0/1",
    )


# ---------------------------------------------------------------------
# Bank Marketing
# ---------------------------------------------------------------------

def prepare_bank():
    df = pd.read_csv(
        RAW / "bank_marketing/bank/bank-full.csv",
        sep=";",
    )
    df = clean_strings(df)

    df["__target__"] = normalize_binary(
        df["y"],
        positive_values=["yes"],
        negative_values=["no"],
    )

    df = add_age_bin(df, "age")
    df = add_copy_group(df, "marital", "__group_marital__")
    df = add_copy_group(df, "education", "__group_education__")

    df = df.drop(columns=["y"])

    groups = [
        "__group_age__",
        "__group_marital__",
        "__group_education__",
    ]

    return finalize(
        "bank_marketing",
        df,
        groups,
        notes=(
            "Marital status and education are treated as grouping variables; "
            "they are not described as universally protected attributes."
        ),
        target_mapping="yes -> 1; no -> 0",
    )


# ---------------------------------------------------------------------
# Communities and Crime
# ---------------------------------------------------------------------

def parse_communities_names():
    attrs = []

    with (
        RAW / "communities_crime/communities.names"
    ).open("r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            m = re.match(
                r"\s*@attribute\s+([^\s]+)",
                line,
                flags=re.I,
            )
            if m:
                attrs.append(m.group(1))

    return attrs


def prepare_communities():
    attrs = parse_communities_names()

    df = pd.read_csv(
        RAW / "communities_crime/communities.data",
        header=None,
        names=attrs if attrs else None,
        na_values="?",
    )

    df = clean_strings(df)

    target = "ViolentCrimesPerPop"
    y_raw = pd.to_numeric(df[target], errors="coerce")

    # Explicit reproducible threshold.
    threshold = float(y_raw.median())

    df["__target__"] = pd.Series(
        np.where(y_raw.isna(), pd.NA, (y_raw >= threshold).astype(int)),
        index=df.index,
        dtype="Int64",
    )

    group_sources = [
        "racepctblack",
        "racePctWhite",
        "racePctAsian",
        "racePctHisp",
    ]

    groups = []

    for src in group_sources:
        x = pd.to_numeric(df[src], errors="coerce")
        dest = "__group_" + re.sub(r"[^A-Za-z0-9]+", "_", src).lower() + "__"

        # Quartiles are computed separately for each composition variable.
        try:
            binned = pd.qcut(
                x,
                q=4,
                labels=["Q1", "Q2", "Q3", "Q4"],
                duplicates="drop",
            )
            df[dest] = binned.astype("string")
        except ValueError:
            df[dest] = pd.NA

        groups.append(dest)

    df = df.rename(columns={target: "ViolentCrimesPerPop_raw"})

    return finalize(
        "communities_crime",
        df,
        groups,
        notes=(
            "Group variables are community-level racial-composition quartiles, "
            "not individual protected attributes."
        ),
        target_mapping=f"ViolentCrimesPerPop >= raw median {threshold:.6g} -> 1",
    )


# ---------------------------------------------------------------------
# German Credit
# ---------------------------------------------------------------------

def prepare_german():
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
        RAW / "german_credit/german.data",
        sep=r"\s+",
        header=None,
        names=cols,
    )

    df = clean_strings(df)

    risk = pd.to_numeric(df["credit_risk"], errors="coerce")
    df["__target__"] = pd.Series(
        np.where(risk.isna(), pd.NA, (risk == 2).astype(int)),
        index=df.index,
        dtype="Int64",
    )

    sex_map = {
        "A91": "male",
        "A92": "female",
        "A93": "male",
        "A94": "male",
        "A95": "female",
    }

    df["__group_sex__"] = (
        df["personal_status_sex"]
        .astype("string")
        .map(sex_map)
        .astype("string")
    )

    df = add_age_bin(df, "age")
    df = add_copy_group(
        df,
        "foreign_worker",
        "__group_foreign_worker__",
    )

    df = df.drop(columns=["credit_risk"])

    groups = [
        "__group_sex__",
        "__group_age__",
        "__group_foreign_worker__",
    ]

    return finalize(
        "german_credit",
        df,
        groups,
        notes="Sex derived from the original combined personal-status/sex UCI code.",
        target_mapping="credit_risk 2 (bad) -> 1; 1 (good) -> 0",
    )


# ---------------------------------------------------------------------
# Census Income KDD
# ---------------------------------------------------------------------

def census_names():
    path = RAW / "census_income_kdd/extracted/census-income.names"
    attrs = []

    with path.open("r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("|"):
                continue

            if ":" in line:
                name = line.split(":", 1)[0].strip()
                if name:
                    attrs.append(name)

    return attrs


def prepare_census():
    attrs = census_names()

    data_path = RAW / "census_income_kdd/extracted/census-income.data"
    test_path = RAW / "census_income_kdd/extracted/census-income.test"

    # The source contains 41 predictors and one target.
    if len(attrs) == 41:
        cols = attrs + ["income"]
    else:
        cols = None

    train = pd.read_csv(
        data_path,
        header=None,
        names=cols,
        skipinitialspace=True,
        na_values="?",
        low_memory=False,
    )

    test = pd.read_csv(
        test_path,
        header=None,
        names=cols,
        skipinitialspace=True,
        na_values="?",
        low_memory=False,
    )

    df = clean_strings(pd.concat([train, test], ignore_index=True))

    if cols is None:
        df = df.rename(columns={
            0: "age",
            7: "marital stat",
            10: "race",
            12: "sex",
            len(df.columns) - 1: "income",
        })

    income_s = df["income"].astype("string")

    df["__target__"] = pd.Series(
        np.where(
            income_s.isna(),
            pd.NA,
            income_s.str.contains("50000\\+", regex=True).astype(int)
        ),
        index=df.index,
        dtype="Int64",
    )

    df = add_copy_group(df, "sex", "__group_sex__")
    df = add_copy_group(df, "race", "__group_race__")
    df = add_age_bin(df, "age")
    df = add_copy_group(df, "marital stat", "__group_marital__")

    df = df.drop(columns=["income"])

    groups = [
        "__group_sex__",
        "__group_race__",
        "__group_age__",
        "__group_marital__",
    ]

    return finalize(
        "census_income_kdd",
        df,
        groups,
        notes="Official train and test files combined before later experimental splitting.",
        target_mapping="50000+ -> 1; -50000 -> 0",
    )


# ---------------------------------------------------------------------
# Diabetes 130-US
# ---------------------------------------------------------------------

def prepare_diabetes():
    df = pd.read_csv(
        RAW / "diabetes130/diabetic_data.csv",
        na_values=["?", "NULL"],
        low_memory=False,
    )
    df = clean_strings(df)

    df["__target__"] = normalize_binary(
        df["readmitted"],
        positive_values=["<30"],
        negative_values=[">30", "NO"],
    )

    df = add_copy_group(df, "gender", "__group_gender__")
    df = add_copy_group(df, "race", "__group_race__")
    df = add_copy_group(df, "age", "__group_age__")

    df = df.drop(columns=["readmitted"])

    groups = [
        "__group_gender__",
        "__group_race__",
        "__group_age__",
    ]

    return finalize(
        "diabetes130",
        df,
        groups,
        notes="Age is already provided as categorical intervals in the source data.",
        target_mapping="<30 readmission -> 1; >30 or NO -> 0",
    )


# ---------------------------------------------------------------------
# Law School
# ---------------------------------------------------------------------

def prepare_law_school():
    path = RAW / "law_school/law-school-admission-binary.arff"

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

    df = clean_strings(df)

    target = "bar"

    raw_target = df[target]

    if pd.api.types.is_numeric_dtype(raw_target):
        vals = sorted(pd.to_numeric(raw_target, errors="coerce").dropna().unique())

        if set(vals).issubset({0, 1}):
            y = pd.to_numeric(raw_target, errors="coerce").astype("Int64")
            mapping = "bar source values 0/1 preserved"
        elif len(vals) == 2:
            y = pd.Series(
                np.where(
                    pd.isna(raw_target),
                    pd.NA,
                    (pd.to_numeric(raw_target, errors="coerce") == vals[-1]).astype(int)
                ),
                index=df.index,
                dtype="Int64",
            )
            mapping = f"bar {vals[-1]} -> 1; {vals[0]} -> 0"
        else:
            raise RuntimeError(f"Unexpected Law School bar values: {vals}")
    else:
        s = raw_target.astype("string").str.strip()
        uniq = [x for x in s.dropna().unique()]

        known_pos = {
            "1", "true", "yes", "pass", "passed",
            "success", "positive"
        }

        low = s.str.lower()

        if any(x.lower() in known_pos for x in uniq):
            pos = [x for x in uniq if x.lower() in known_pos]
            y = normalize_binary(s, positive_values=pos)
            mapping = f"bar positive values: {pos}"
        elif len(uniq) == 2:
            uniq_sorted = sorted(uniq)
            y = normalize_binary(
                s,
                positive_values=[uniq_sorted[-1]],
                negative_values=[uniq_sorted[0]],
            )
            mapping = (
                f"two-class fallback: {uniq_sorted[-1]} -> 1; "
                f"{uniq_sorted[0]} -> 0"
            )
        else:
            raise RuntimeError(f"Unexpected Law School bar values: {uniq}")

    df["__target__"] = y

    gender_col = "gender"
    race_col = "race1"

    df = add_copy_group(df, gender_col, "__group_gender__")
    df = add_copy_group(df, race_col, "__group_race__")
    df = add_age_bin(
        df,
        "age",
        bins=LSAC_AGE_BINS,
        labels=LSAC_AGE_LABELS,
    )

    df = df.drop(columns=[target])

    groups = [
        "__group_gender__",
        "__group_race__",
        "__group_age__",
    ]

    return finalize(
        "law_school",
        df,
        groups,
        notes="Downloaded OpenML binary Law School benchmark.",
        target_mapping=mapping,
    )


PREPARERS = [
    ("adult", prepare_adult),
    ("compas", prepare_compas),
    ("default_credit", prepare_default_credit),
    ("bank_marketing", prepare_bank),
    ("communities_crime", prepare_communities),
    ("german_credit", prepare_german),
    ("census_income_kdd", prepare_census),
    ("diabetes130", prepare_diabetes),
    ("law_school", prepare_law_school),
]


def main():
    reports = []
    supports = []
    schemas = []

    print("=" * 92)
    print("PREPARING DATASETS")
    print("RAW DATA ARE READ-ONLY; OUTPUT GOES TO data/processed/")
    print("=" * 92)

    failed = []

    for dataset_id, fn in PREPARERS:
        print(f"\n[{dataset_id}]")

        try:
            report, support, schema = fn()

            reports.append(report)
            supports.extend(support)
            schemas.extend(schema)

            print(
                f"  rows={report['rows']:,} "
                f"cols={report['columns']} "
                f"positive={report['target_positive_rate']:.4f}"
            )

            print(
                f"  groups={report['group_columns']}"
            )

            print(
                f"  max intersection: "
                f"{report['max_intersection_groups']} groups, "
                f"min={report['max_intersection_min_support']}, "
                f"median={report['max_intersection_median_support']}"
            )

            print(
                f"  -> {report['processed_path']}"
            )

        except Exception as e:
            failed.append((dataset_id, type(e).__name__, str(e)))
            print(
                f"  ERROR: {type(e).__name__}: {e}"
            )

    pd.DataFrame(reports).to_csv(REPORT, index=False)
    pd.DataFrame(supports).to_csv(SUPPORT, index=False)
    pd.DataFrame(schemas).to_csv(SCHEMA, index=False)

    print("\n" + "=" * 92)
    print(f"WROTE {REPORT}")
    print(f"WROTE {SUPPORT}")
    print(f"WROTE {SCHEMA}")

    if failed:
        print("\nFAILED DATASETS:")
        for dataset_id, typ, msg in failed:
            print(f"  {dataset_id}: {typ}: {msg}")
    else:
        print("\nALL DATASETS PREPARED SUCCESSFULLY")

    print("=" * 92)


if __name__ == "__main__":
    main()
