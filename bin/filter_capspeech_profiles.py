#!/usr/bin/env python3
"""Filter a Capspeech-style dataset to profiles with at least N train segments."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROFILE_FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]
UNKNOWN = "unknown"
SPLIT_FILES = ["train.csv", "dev.csv", "test.csv"]
PROFILE_FILES = [
    "profile_combinations.csv",
    "profile_prompts.csv",
    "unique_profile_combinations.csv",
    "unique_profile_prompts.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create data/Capspeech_minN with only profiles having at least N segments."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/Capspeech"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-segments", type=int, default=100)
    parser.add_argument("--profile-fields", default=",".join(DEFAULT_PROFILE_FIELDS))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_value(value: Any) -> str:
    if pd.isna(value):
        return UNKNOWN
    text = str(value).strip()
    return text if text else UNKNOWN


def normalize_fields(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    df = df.copy()
    for field in fields:
        if field not in df.columns:
            df[field] = UNKNOWN
        df[field] = df[field].map(normalize_value)
    return df


def profile_keys(df: pd.DataFrame, fields: list[str]) -> pd.Series:
    return pd.Series([tuple(row) for row in df[fields].to_numpy(dtype=object)], index=df.index)


def read_profiles(path: Path, fields: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required profile CSV: {path}")
    profiles = pd.read_csv(path)
    missing = [field for field in fields if field not in profiles.columns]
    if missing:
        raise ValueError(f"{path} is missing profile fields: {missing}")
    if "num_audios" not in profiles.columns:
        raise ValueError(f"{path} is missing required column: num_audios")
    profiles = normalize_fields(profiles, fields)
    profiles["num_audios"] = pd.to_numeric(profiles["num_audios"], errors="coerce").fillna(0).astype(int)
    return profiles


def filter_profile_file(path: Path, output_path: Path, fields: list[str], min_segments: int) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    if "num_audios" not in df.columns:
        shutil.copy2(path, output_path)
        print(f"Copied {path} -> {output_path} (no num_audios column)", flush=True)
        return
    df["num_audios"] = pd.to_numeric(df["num_audios"], errors="coerce").fillna(0).astype(int)
    filtered = df.loc[df["num_audios"] >= min_segments].copy()
    filtered.to_csv(output_path, index=False)
    print(f"Wrote {len(filtered):,}/{len(df):,} rows to {output_path}", flush=True)


def filter_split_file(path: Path, output_path: Path, fields: list[str], kept_keys: set[tuple[Any, ...]]) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = normalize_fields(df, fields)
    mask = profile_keys(df, fields).isin(kept_keys)
    filtered = df.loc[mask].copy()
    filtered.to_csv(output_path, index=False)
    print(f"Wrote {len(filtered):,}/{len(df):,} rows to {output_path}", flush=True)


def main() -> int:
    args = parse_args()
    if args.min_segments < 1:
        raise ValueError("--min-segments must be >= 1")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Missing input directory: {args.input_dir}")

    output_dir = args.output_dir or args.input_dir.with_name(f"{args.input_dir.name}_min{args.min_segments}")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace its CSV files")
        if not output_dir.is_dir():
            raise NotADirectoryError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = [field.strip() for field in args.profile_fields.split(",") if field.strip()]
    combinations_path = args.input_dir / "profile_combinations.csv"
    combinations = read_profiles(combinations_path, fields)
    kept_combinations = combinations.loc[combinations["num_audios"] >= args.min_segments].copy()
    kept_keys = set(profile_keys(kept_combinations, fields))
    if not kept_keys:
        raise ValueError(f"No profiles in {combinations_path} have num_audios >= {args.min_segments}")

    for filename in PROFILE_FILES:
        filter_profile_file(args.input_dir / filename, output_dir / filename, fields, args.min_segments)

    for filename in SPLIT_FILES:
        filter_split_file(args.input_dir / filename, output_dir / filename, fields, kept_keys)

    for path in args.input_dir.iterdir():
        if not path.is_file() or path.name in set(PROFILE_FILES + SPLIT_FILES):
            continue
        shutil.copy2(path, output_dir / path.name)
        print(f"Copied {path} -> {output_dir / path.name}", flush=True)

    print(
        f"Kept {len(kept_combinations):,}/{len(combinations):,} full profiles with num_audios >= {args.min_segments} in {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
