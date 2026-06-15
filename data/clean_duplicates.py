#!/usr/bin/env python3
"""Normalize duplicate Capspeech metadata labels across split CSVs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd


SPLITS = ["train", "dev", "test"]
FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]
NORMALIZATIONS = {
    "pitch": {
        "high-pitched": "high-pitched",
        "low-pitched": "low-pitch",
        "medium-pitched": "moderate pitch",
    },
    "speaking_rate": {
        "fast speed": "fast",
        "measured speed": "moderate speed",
        "slow speed": "slowly",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean duplicate label values in Capspeech split metadata.")
    parser.add_argument("--dataset", default="Capspeech", help="Dataset directory under --data-root.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--chunksize", type=int, default=200_000)
    return parser.parse_args()


def normalize_value(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def format_values(values: set[str]) -> str:
    return "{" + ", ".join(sorted(values)) + "}"


def update_value_sets(values_by_field: dict[str, set[str]], chunk: pd.DataFrame) -> None:
    for field in FIELDS:
        if field in chunk.columns:
            values_by_field[field].update(chunk[field].map(normalize_value).unique())


def normalize_chunk(chunk: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    for field, replacements in NORMALIZATIONS.items():
        if field not in chunk.columns:
            continue
        chunk[field] = chunk[field].map(normalize_value).replace(replacements)
    for field in FIELDS:
        if field in chunk.columns and field not in NORMALIZATIONS:
            chunk[field] = chunk[field].map(normalize_value)

    dropped_rows = 0
    if "accent" in chunk.columns:
        dashed_accent_mask = chunk["accent"].map(
            lambda value: normalize_value(value) != "unknown" and "-" in normalize_value(value)
        )
        dropped_rows = int(dashed_accent_mask.sum())
        chunk = chunk.loc[~dashed_accent_mask].copy()

    return chunk, dropped_rows


def process_split(path: Path, chunksize: int) -> tuple[str, int, int, dict[str, set[str]], dict[str, set[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {path}")

    before = {field: set() for field in FIELDS}
    after = {field: set() for field in FIELDS}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    rows = 0
    dropped_rows = 0
    wrote_header = False

    try:
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            rows += len(chunk)
            update_value_sets(before, chunk)
            chunk, chunk_dropped_rows = normalize_chunk(chunk)
            dropped_rows += chunk_dropped_rows
            update_value_sets(after, chunk)
            chunk.to_csv(tmp_path, mode="w" if not wrote_header else "a", header=not wrote_header, index=False)
            wrote_header = True
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return path.stem, rows, dropped_rows, before, after


def merge_value_sets(results: list[tuple[str, int, int, dict[str, set[str]], dict[str, set[str]]]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    before = {field: set() for field in FIELDS}
    after = {field: set() for field in FIELDS}
    for _split, _rows, _dropped_rows, split_before, split_after in results:
        for field in FIELDS:
            before[field].update(split_before[field])
            after[field].update(split_after[field])
    return before, after


def print_value_sets(title: str, values_by_field: dict[str, set[str]]) -> None:
    print(title, flush=True)
    for field in FIELDS:
        print(f"{field}: {format_values(values_by_field[field])}", flush=True)


def main() -> int:
    args = parse_args()
    num_workers = max(1, args.num_workers)
    dataset_dir = args.data_root / args.dataset
    if not dataset_dir.is_dir():
        raise SystemExit(f"Missing dataset directory: {dataset_dir}")

    split_paths = [dataset_dir / f"{split}.csv" for split in SPLITS]
    print(
        f"Cleaning {args.dataset} metadata with {num_workers} worker thread(s): "
        f"{', '.join(str(path) for path in split_paths)}",
        flush=True,
    )

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_split, path, args.chunksize): path for path in split_paths}
        for future in as_completed(futures):
            split, rows, dropped_rows, before, after = future.result()
            results.append((split, rows, dropped_rows, before, after))
            print(
                f"Cleaned {rows:,} rows in {futures[future]}; "
                f"removed_dashed_accent_rows={dropped_rows:,}",
                flush=True,
            )

    before, after = merge_value_sets(results)
    print_value_sets("Before cleaning:", before)
    print_value_sets("After cleaning:", after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
