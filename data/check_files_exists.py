#!/usr/bin/env python3
"""Verify that generated metadata points to existing audio files."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

SPLITS = ["train", "dev", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that every storage_path in a dataset's split CSVs exists.")
    parser.add_argument("dataset_name", help="Dataset directory under --data-root, for example MLS-en.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--num-workers", type=int, default=16)
    return parser.parse_args()


def check_path(row_number: int, audio_path: str) -> tuple[int, str, bool]:
    return row_number, audio_path, Path(audio_path).exists()


def submit_next(
    reader: csv.DictReader,
    executor: ThreadPoolExecutor,
    pending: set[Future[tuple[int, str, bool]]],
) -> bool:
    try:
        row = next(reader)
    except StopIteration:
        return False

    if "storage_path" not in row:
        raise SystemExit("Missing required column `storage_path`.")

    row_number = reader.line_num
    audio_path = str(row["storage_path"]).strip()
    if not audio_path:
        raise SystemExit(f"Missing storage_path value at CSV line {row_number}.")
    pending.add(executor.submit(check_path, row_number, audio_path))
    return True


def check_split(path: Path, num_workers: int) -> int:
    if not path.exists():
        raise SystemExit(f"Missing metadata CSV: {path}")

    checked = 0
    max_pending = num_workers * 8
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "storage_path" not in (reader.fieldnames or []):
            raise SystemExit(f"{path} is missing required column `storage_path`.")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pending: set[Future[tuple[int, str, bool]]] = set()
            for _ in range(max_pending):
                if not submit_next(reader, executor, pending):
                    break

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row_number, audio_path, exists = future.result()
                    checked += 1
                    if not exists:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise SystemExit(f"Missing audio file in {path} at CSV line {row_number}: {audio_path}")
                    submit_next(reader, executor, pending)

    return checked


def main() -> int:
    args = parse_args()
    num_workers = max(1, args.num_workers)
    dataset_dir = args.data_root / args.dataset_name
    if not dataset_dir.is_dir():
        raise SystemExit(f"Missing dataset directory: {dataset_dir}")

    total = 0
    for split in SPLITS:
        split_path = dataset_dir / f"{split}.csv"
        checked = check_split(split_path, num_workers)
        total += checked
        print(f"Checked {checked:,} {split} audio paths in {split_path}", flush=True)

    print(f"All {total:,} audio files exist for {args.dataset_name}.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
