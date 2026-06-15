#!/usr/bin/env python3
"""Create single-known-field profile combinations from merged profiles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]
UNKNOWN = "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create profile rows where exactly one profile field is known."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/Capspeech/profile_combinations.csv"),
        help="Merged profile_combinations.csv to read.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/Capspeech/unique_profile_combinations.csv"),
        help="CSV to write single-known-field profile combinations to.",
    )
    parser.add_argument(
        "--fields",
        default=",".join(DEFAULT_FIELDS),
        help="Comma-separated profile fields to use.",
    )
    return parser.parse_args()


def normalize(value: Any) -> str:
    if pd.isna(value):
        return UNKNOWN
    text = str(value).strip()
    return text if text else UNKNOWN


def make_unique_profiles(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    count_column = "num_audios" if "num_audios" in df.columns else None
    capspeech_desc_column = "capspeech_desc" if "capspeech_desc" in df.columns else None
    if count_column:
        df[count_column] = pd.to_numeric(df[count_column], errors="coerce").fillna(0).astype(int)

    for field in fields:
        known = df.loc[df[field] != UNKNOWN]
        values = sorted(known[field].dropna().astype(str).unique())
        for value in values:
            row = {profile_field: UNKNOWN for profile_field in fields}
            row[field] = value
            if count_column:
                row[count_column] = int(known.loc[known[field] == value, count_column].sum())
            if capspeech_desc_column:
                row[capspeech_desc_column] = ""
            rows.append(row)

    columns = [*fields]
    if count_column:
        columns.append(count_column)
    if capspeech_desc_column:
        columns.append(capspeech_desc_column)
    return pd.DataFrame(rows, columns=columns)


def main() -> int:
    args = parse_args()
    fields = [item.strip() for item in args.fields.split(",") if item.strip()]
    df = pd.read_csv(args.input)

    missing_fields = [field for field in fields if field not in df.columns]
    if missing_fields:
        raise ValueError(f"{args.input} is missing profile fields: {missing_fields}")
    for field in fields:
        df[field] = df[field].map(normalize)

    unique_profiles = make_unique_profiles(df, fields)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    unique_profiles.to_csv(args.output, index=False)
    print(f"Wrote {len(unique_profiles)} single-known-field profiles to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
