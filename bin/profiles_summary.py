#!/usr/bin/env python3
"""Summarize generated profile combinations, descriptions, and metadata coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

DEFAULT_DATASETS = ["CommonVoice", "GigaSpeech", "MLS-en", "Emilia-en"]
DESC_COLUMNS = [f"desc{i}" for i in range(10)]
EXTRA_DESC_COLUMNS = ["capspeech_desc"]
DEFAULT_METADATA_FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]
SPLITS = ["train", "dev", "test"]
UNKNOWN_VALUES = {"", "nan", "none", "unknown", "not_computed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print and save profile generation summary table.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--filtered-datasets",
        nargs="*",
        default=["Capspeech_min100"],
        help="Filtered aggregate datasets to show after the unfiltered TOTAL row.",
    )
    parser.add_argument("--metadata-fields", nargs="*", default=DEFAULT_METADATA_FIELDS)
    parser.add_argument("--latex-output", type=Path, default=Path("exp/tables/profiles.txt"))
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def count_descriptions(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    desc_columns = [column for column in [*DESC_COLUMNS, *EXTRA_DESC_COLUMNS] if column in df.columns]
    total = 0
    for column in desc_columns:
        values = df[column].dropna().astype(str).str.strip()
        total += int(((values != "") & (values.str.lower() != "nan")).sum())
    return total


def category_count(df: pd.DataFrame, field: str) -> int:
    if df.empty or field not in df.columns:
        return 0
    values = df[field].dropna().astype(str).str.strip()
    values = values[~values.str.lower().isin(UNKNOWN_VALUES)]
    return int(values.nunique())


def category_counts_from_splits(data_root: Path, dataset: str, metadata_fields: list[str]) -> dict[str, int]:
    values_by_field = {field: set() for field in metadata_fields}
    for split in SPLITS:
        path = data_root / dataset / f"{split}.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            header = pd.read_csv(path, nrows=0)
        except EmptyDataError:
            continue
        usecols = [field for field in metadata_fields if field in header.columns]
        if not usecols:
            continue
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=200_000):
            for field in usecols:
                values = chunk[field].dropna().astype(str).str.strip()
                values = values[~values.str.lower().isin(UNKNOWN_VALUES)]
                values_by_field[field].update(values.tolist())
    return {field: len(values) for field, values in values_by_field.items()}


def profile_summary_row(
    data_root: Path,
    dataset: str,
    metadata_fields: list[str],
    label: str | None = None,
) -> dict[str, object]:
    combinations = read_csv(data_root / dataset / "profile_combinations.csv")
    prompts = read_csv(data_root / dataset / "profile_prompts.csv")
    split_counts = None if not combinations.empty else category_counts_from_splits(data_root, dataset, metadata_fields)
    row: dict[str, object] = {
        "dataset": label or dataset,
        "profiles": len(combinations),
        "descriptions": count_descriptions(prompts),
    }
    for field in metadata_fields:
        row[field] = category_count(combinations, field) if not combinations.empty else split_counts[field]
    return row


def filtered_label(dataset: str) -> str:
    if "_min" in dataset:
        base, threshold = dataset.rsplit("_min", 1)
        return f"TOTAL >= {threshold} ({dataset})" if threshold else f"FILTERED ({dataset})"
    return f"FILTERED ({dataset})"


def build_rows(
    data_root: Path,
    datasets: list[str],
    metadata_fields: list[str],
    filtered_datasets: list[str],
) -> list[dict[str, object]]:
    rows = [profile_summary_row(data_root, dataset, metadata_fields) for dataset in datasets]
    rows.append(profile_summary_row(data_root, "Capspeech", metadata_fields, label="TOTAL before filtering"))
    for dataset in filtered_datasets:
        if (data_root / dataset).is_dir():
            rows.append(profile_summary_row(data_root, dataset, metadata_fields, label=filtered_label(dataset)))
    return rows


def print_table(rows: list[dict[str, object]], metadata_fields: list[str]) -> None:
    columns = [("Dataset", "dataset"), ("Profiles", "profiles"), ("Descriptions", "descriptions")]
    columns.extend((field, field) for field in metadata_fields)
    widths = []
    for label, key in columns:
        values = [label]
        for row in rows:
            value = str(row[key]) if key == "dataset" else f"{int(row[key]):,}"
            values.append(value)
        widths.append(max(len(value) for value in values))

    header = "  ".join(label.ljust(width) if key == "dataset" else label.rjust(width) for (label, key), width in zip(columns, widths))
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for (_label, key), width in zip(columns, widths):
            if key == "dataset":
                cells.append(str(row[key]).ljust(width))
            else:
                cells.append(f"{int(row[key]):>{width},}")
        print("  ".join(cells))


def latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def write_latex(rows: list[dict[str, object]], metadata_fields: list[str], output: Path) -> None:
    row_end = r"\\"
    align = "l" + "r" * (2 + len(metadata_fields))
    header = ["Dataset", "Profiles", "Descriptions", *metadata_fields]
    lines = [
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(latex_escape(column) for column in header) + f" {row_end}",
        r"\midrule",
    ]
    for row in rows:
        values = [latex_escape(row["dataset"]), f"{int(row['profiles']):,}", f"{int(row['descriptions']):,}"]
        values.extend(f"{int(row[field]):,}" for field in metadata_fields)
        lines.append(" & ".join(values) + f" {row_end}")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote LaTeX table to {output}", flush=True)


def main() -> int:
    args = parse_args()
    rows = build_rows(args.data_root, args.datasets, args.metadata_fields, args.filtered_datasets)
    print_table(rows, args.metadata_fields)
    write_latex(rows, args.metadata_fields, args.latex_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
