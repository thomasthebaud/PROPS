#!/usr/bin/env python3
"""Merge per-dataset profile CSVs under data/Capspeech."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DATASETS = ["CommonVoice", "GigaSpeech", "MLS-en", "Emilia-en"]
PROFILE_FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]


def first_nonempty(values: pd.Series) -> str:
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def merge_profile_counts(df: pd.DataFrame) -> pd.DataFrame:
    fields = [field for field in PROFILE_FIELDS if field in df.columns]
    if not fields or "num_audios" not in df.columns:
        return df

    df = df.copy()
    df["num_audios"] = pd.to_numeric(df["num_audios"], errors="coerce").fillna(0).astype(int)
    aggregations = {"num_audios": "sum"}
    if "capspeech_desc" in df.columns:
        aggregations["capspeech_desc"] = first_nonempty
    for column in df.columns:
        if column not in fields and column not in aggregations:
            aggregations[column] = first_nonempty
    return df.groupby(fields, dropna=False, sort=True, as_index=False).agg(aggregations)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge one profile CSV filename across dataset directories.")
    parser.add_argument("filename", help="CSV basename to merge, e.g. profile_prompts.csv")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/Capspeech"))
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames = []
    for dataset in args.datasets:
        path = args.data_root / dataset / args.filename
        if not path.exists() or path.stat().st_size == 0:
            print(f"WARNING: missing/empty {path}; skipping", flush=True)
            continue
        df = pd.read_csv(path)
        # df.insert(0, "dataset", dataset)
        frames.append(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / args.filename
    if not frames:
        pd.DataFrame().to_csv(output, index=False)
        print(f"Wrote 0 rows to {output}", flush=True)
        return 0

    merged = pd.concat(frames, ignore_index=True, sort=False)
    before_merge = len(merged)
    merged = merge_profile_counts(merged)
    merged.to_csv(output, index=False)
    print(f"Wrote {len(merged)} rows to {output} from {before_merge} input rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
