#!/usr/bin/env python3
"""Create a smaller Capspeech-style dataset with capped train rows per profile."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PROFILE_FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create data/Capspeech_N with at most N train audios per profile.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/Capspeech"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-audios-per-profile", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--profile-fields", default=",".join(DEFAULT_PROFILE_FIELDS))
    parser.add_argument("--dataset-column", default="dataset")
    return parser.parse_args()


def normalize_profile_columns(df: pd.DataFrame, profile_fields: list[str]) -> list[str]:
    fields = [field for field in profile_fields if field in df.columns]
    if not fields:
        raise ValueError(f"None of the requested profile fields are present: {profile_fields}")
    for field in fields:
        df[field] = df[field].fillna("unknown").astype(str).str.strip()
        df.loc[df[field].eq(""), field] = "unknown"
    return fields


def allocate_counts(group_sizes: pd.Series, total: int) -> dict[object, int]:
    if group_sizes.sum() <= total:
        return group_sizes.astype(int).to_dict()

    raw = group_sizes / group_sizes.sum() * total
    allocations = np.floor(raw).astype(int)
    allocations = allocations.where(group_sizes <= 0, allocations.clip(lower=1))
    allocations = allocations.clip(upper=group_sizes)

    while allocations.sum() > total:
        candidates = allocations[allocations > 0]
        key = (raw.loc[candidates.index] - allocations.loc[candidates.index]).idxmin()
        allocations.loc[key] -= 1

    while allocations.sum() < total:
        remaining = group_sizes - allocations
        candidates = remaining[remaining > 0]
        if candidates.empty:
            break
        key = (raw.loc[candidates.index] - allocations.loc[candidates.index]).idxmax()
        allocations.loc[key] += 1

    return allocations.astype(int).to_dict()


def sample_profile_group(
    group: pd.DataFrame,
    max_audios: int,
    dataset_column: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if len(group) <= max_audios:
        return group

    if dataset_column not in group.columns:
        return group.sample(n=max_audios, random_state=int(rng.integers(0, 2**32 - 1)))

    source_key = group[dataset_column].fillna("unknown").astype(str)
    dataset_sizes = source_key.groupby(source_key, sort=False).size()
    allocations = allocate_counts(dataset_sizes, max_audios)
    samples = []
    for dataset_value, count in allocations.items():
        if count <= 0:
            continue
        subset = group[source_key.eq(dataset_value)]
        samples.append(subset.sample(n=count, random_state=int(rng.integers(0, 2**32 - 1))))
    return pd.concat(samples, ignore_index=False)


def cap_train_profiles(
    train_df: pd.DataFrame,
    profile_fields: list[str],
    max_audios: int,
    dataset_column: str,
    seed: int,
) -> pd.DataFrame:
    fields = normalize_profile_columns(train_df, profile_fields)
    rng = np.random.default_rng(seed)
    sampled_groups = []
    grouped = train_df.groupby(fields, dropna=False, sort=False)

    num_capped = 0
    removed = 0
    for _, group in grouped:
        sampled = sample_profile_group(group, max_audios, dataset_column, rng)
        if len(group) > len(sampled):
            num_capped += 1
            removed += len(group) - len(sampled)
        sampled_groups.append(sampled)

    result = pd.concat(sampled_groups, ignore_index=False).sort_index()
    print(
        f"Capped {num_capped} profiles at {max_audios} train audios; "
        f"removed {removed:,} of {len(train_df):,} train rows.",
        flush=True,
    )
    return result


def copy_capspeech_files(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in input_dir.iterdir():
        if path.name == "train.csv" or not path.is_file():
            continue
        shutil.copy2(path, output_dir / path.name)
        print(f"Copied {path} -> {output_dir / path.name}", flush=True)


def main() -> int:
    args = parse_args()
    if args.max_audios_per_profile < 1:
        raise ValueError("--max-audios-per-profile must be >= 1")
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Missing input directory: {args.input_dir}")

    output_dir = args.output_dir or args.input_dir.with_name(f"{args.input_dir.name}_{args.max_audios_per_profile}")
    profile_fields = [field.strip() for field in args.profile_fields.split(",") if field.strip()]

    copy_capspeech_files(args.input_dir, output_dir)

    train_path = args.input_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train CSV: {train_path}")
    print(f"Reading {train_path}", flush=True)
    train_df = pd.read_csv(train_path)
    smaller_train = cap_train_profiles(
        train_df,
        profile_fields,
        args.max_audios_per_profile,
        args.dataset_column,
        args.seed,
    )

    output_train_path = output_dir / "train.csv"
    smaller_train.to_csv(output_train_path, index=False)
    print(f"Wrote {len(smaller_train):,} rows to {output_train_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
