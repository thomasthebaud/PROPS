#!/usr/bin/env python3
"""Prepare CapSpeech MLS-en metadata in the unified PROPS CSV format."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

BASE_FIELDS = ["id", "duration", "speaker", "split", "storage_path", "sample_freq"]
OPTIONAL_FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent", "capspeech_prompt"]
DEFAULT_METADATA_ROOT = Path("/export/fs05/corpora7/capspeechset/capspeechset1")
DEFAULT_AUDIO_ROOT = Path("/export/fs05/corpora7/CapSpeech-MLS")
SPLIT_DIRS = {"train": "train", "dev": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create data/<dataset_name>/{train,dev,test}.csv for CapSpeech MLS-en.")
    parser.add_argument("dataset_name", nargs="?", default="MLS-en")
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--skip-missing-audio", action="store_true")
    return parser.parse_args()


def load_dataset(path: Path):
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise SystemExit("MLS-en metadata is stored as HuggingFace Arrow data. Activate the conda env or install `datasets`.") from exc
    return load_from_disk(str(path))


def normalize_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def safe_id(audio_path: object) -> str:
    return Path(str(audio_path)).stem


def speaker_from_path(audio_path: object) -> str:
    parts = Path(str(audio_path)).parts
    if len(parts) >= 3:
        return parts[-3]
    return safe_id(audio_path).split("_")[0]


def row_to_record(row: dict, split: str, audio_root: Path) -> dict[str, object]:
    audio_path = str(row["audio_path"])
    record = {
        "id": safe_id(audio_path),
        "duration": row.get("speech_duration", ""),
        "speaker": speaker_from_path(audio_path),
        "split": split,
        "storage_path": str(audio_root / audio_path),
        "sample_freq": 16000,
        "capspeech_prompt": normalize_value(row.get("caption")),
    }
    for field in ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]:
        if field in row:
            record[field] = normalize_value(row.get(field))
    return record


def records_for_split(args: argparse.Namespace, split: str) -> Iterable[dict[str, object]]:
    dataset_path = args.metadata_root / SPLIT_DIRS[split]
    dataset = load_dataset(dataset_path)
    for row in tqdm(dataset, total=len(dataset), desc=f"Preparing MLS-en {split}", unit="row"):
        record = row_to_record(row, split, args.audio_root)
        if args.skip_missing_audio and not Path(str(record["storage_path"])).exists():
            continue
        yield record


def write_split(args: argparse.Namespace, split: str, output_dir: Path) -> None:
    output_path = output_dir / f"{split}.csv"
    count = 0
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_FIELDS + OPTIONAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records_for_split(args, split):
            writer.writerow(record)
            count += 1
    print(f"Wrote {count} rows to {output_path}", flush=True)


def main() -> int:
    args = parse_args()
    output_dir = args.output_root / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in tqdm(["train", "dev", "test"], desc="Writing MLS-en splits", unit="split"):
        write_split(args, split, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
