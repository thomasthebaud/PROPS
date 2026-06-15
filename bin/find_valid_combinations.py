#!/usr/bin/env python3
"""Find valid profile combinations for one generated metadata dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]
UNKNOWN = "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create profile_combinations.csv for one dataset directory.")
    parser.add_argument("--dataset-name", default="CommonVoice", help="Dataset directory under --data-root.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--splits", default="train,dev,test", help="Comma-separated split CSV stems to read.")
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS), help="Comma-separated profile fields to use when present.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def normalize(value: Any) -> str:
    if pd.isna(value):
        return UNKNOWN
    text = str(value).strip()
    return text if text else UNKNOWN


def load_dataset(data_root: Path, dataset_name: str, splits: list[str]) -> pd.DataFrame:
    frames = []
    for split in splits:
        path = data_root / dataset_name / f"{split}.csv"
        if not path.exists() or path.stat().st_size == 0:
            print(f"WARNING: skipping missing/empty {path}", flush=True)
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f"WARNING: skipping header-only {path}", flush=True)
            continue
        df["split"] = split
        frames.append(df)
    if not frames:
        raise ValueError(f"No split CSVs found for {dataset_name} under {data_root}")
    return pd.concat(frames, ignore_index=True, sort=False)


def clean_prompt(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def first_nonempty(values: pd.Series) -> str:
    for value in values:
        text = clean_prompt(value)
        if text:
            return text
    return ""


def existing_combinations(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    grouped = df.groupby(fields, dropna=False, sort=True)
    profiles = grouped.size().reset_index(name="num_audios")

    if "capspeech_prompt" in df.columns:
        descs = grouped["capspeech_prompt"].agg(first_nonempty).reset_index(name="capspeech_desc")
        profiles = profiles.merge(descs, on=fields, how="left")
        if not profiles["capspeech_desc"].astype(bool).any():
            profiles = profiles.drop(columns=["capspeech_desc"])
        else:
            profiles["capspeech_desc"] = profiles["capspeech_desc"].fillna("")

    return profiles.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    requested_fields = [item.strip() for item in args.fields.split(",") if item.strip()]
    df = load_dataset(args.data_root, args.dataset_name, splits)

    fields = [field for field in requested_fields if field in df.columns]
    if not fields:
        raise ValueError(f"None of requested fields {requested_fields} are present in {args.dataset_name}")
    for field in fields:
        df[field] = df[field].map(normalize)

    combinations = existing_combinations(df, fields)
    print(f"{args.dataset_name}: found {len(combinations)} existing combinations over fields={fields}", flush=True)

    output = args.output or (args.data_root / args.dataset_name / "profile_combinations.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    combinations.to_csv(output, index=False)
    print(f"Saved {len(combinations)} existing combinations to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
