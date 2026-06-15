#!/usr/bin/env python3
"""Verify that every row in dataset split CSVs has an extracted xvector."""

from __future__ import annotations

import argparse
import csv
import re
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path


DEFAULT_SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that all expected xvector .npz files exist for dataset split CSVs."
    )
    parser.add_argument("dataset_names", nargs="+", help="Dataset directories under --data-root.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--xvector-root", type=Path, default=Path("exp/xvectors"))
    parser.add_argument("--model-name", default="ecapa_tdnn")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="Comma-separated split CSV stems.")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--max-missing",
        type=int,
        default=20,
        help="Maximum number of missing paths to print before summarizing.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "model"


def expected_xvector_path(
    xvector_root: Path,
    model_name: str,
    dataset_name: str,
    split: str,
    utterance_id: str,
) -> Path:
    output_dataset_name = f"{dataset_name}_{split}"
    return xvector_root / safe_name(model_name) / output_dataset_name / "xvectors" / f"{safe_name(utterance_id)}.npz"


def check_path(row_number: int, utterance_id: str, path: Path) -> tuple[int, str, Path, bool]:
    return row_number, utterance_id, path, path.exists()


def submit_next(
    reader: csv.DictReader,
    executor: ThreadPoolExecutor,
    pending: set[Future[tuple[int, str, Path, bool]]],
    xvector_root: Path,
    model_name: str,
    dataset_name: str,
    split: str,
) -> bool:
    try:
        row = next(reader)
    except StopIteration:
        return False

    if "id" not in row:
        raise SystemExit("Missing required column `id`.")

    row_number = reader.line_num
    utterance_id = str(row["id"]).strip()
    if not utterance_id:
        raise SystemExit(f"Missing id value at CSV line {row_number}.")

    path = expected_xvector_path(xvector_root, model_name, dataset_name, split, utterance_id)
    pending.add(executor.submit(check_path, row_number, utterance_id, path))
    return True


def check_split(
    csv_path: Path,
    xvector_root: Path,
    model_name: str,
    dataset_name: str,
    split: str,
    num_workers: int,
    max_missing: int,
) -> tuple[int, int]:
    if not csv_path.exists():
        raise SystemExit(f"Missing metadata CSV: {csv_path}")

    checked = 0
    missing = 0
    max_pending = num_workers * 8
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "id" not in (reader.fieldnames or []):
            raise SystemExit(f"{csv_path} is missing required column `id`.")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pending: set[Future[tuple[int, str, Path, bool]]] = set()
            for _ in range(max_pending):
                if not submit_next(reader, executor, pending, xvector_root, model_name, dataset_name, split):
                    break

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row_number, utterance_id, path, exists = future.result()
                    checked += 1
                    if not exists:
                        missing += 1
                        if missing <= max_missing:
                            print(
                                f"Missing xvector in {csv_path} at line {row_number} "
                                f"for id={utterance_id}: {path}",
                                flush=True,
                            )
                    submit_next(reader, executor, pending, xvector_root, model_name, dataset_name, split)

    print(f"Checked {checked:,} expected xvectors for {dataset_name}/{split}: missing={missing:,}", flush=True)
    return checked, missing


def main() -> int:
    args = parse_args()
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    if not splits:
        raise SystemExit("No splits requested.")

    num_workers = max(1, args.num_workers)
    total_checked = 0
    total_missing = 0

    for dataset_name in args.dataset_names:
        dataset_dir = args.data_root / dataset_name
        if not dataset_dir.is_dir():
            raise SystemExit(f"Missing dataset directory: {dataset_dir}")

        dataset_checked = 0
        dataset_missing = 0
        for split in splits:
            checked, missing = check_split(
                dataset_dir / f"{split}.csv",
                args.xvector_root,
                args.model_name,
                dataset_name,
                split,
                num_workers,
                max(0, args.max_missing - total_missing),
            )
            dataset_checked += checked
            dataset_missing += missing

        total_checked += dataset_checked
        total_missing += dataset_missing
        print(
            f"{dataset_name}: checked {dataset_checked:,} xvectors across {len(splits)} splits; "
            f"missing={dataset_missing:,}",
            flush=True,
        )

    if total_missing:
        print(f"ERROR: missing {total_missing:,} of {total_checked:,} expected xvectors.", flush=True)
        return 1

    print(f"All {total_checked:,} expected xvectors exist.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
