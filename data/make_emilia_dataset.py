#!/usr/bin/env python3
"""Prepare CapSpeech Emilia-en metadata in the unified PROPS CSV format."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_FIELDS = ["id", "duration", "speaker", "split", "storage_path", "sample_freq"]
OPTIONAL_FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent", "capspeech_prompt"]
DEFAULT_METADATA_PATH = Path("/export/fs05/corpora7/emilia_en/filtered_paraspeechcaps_cleaned")
DEFAULT_AUDIO_ROOT = Path("/export/fs05/corpora7/emilia_en")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create data/<dataset_name>/{train,dev,test}.csv for CapSpeech Emilia-en.")
    parser.add_argument("dataset_name", nargs="?", default="Emilia-en")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--dev-percent", type=int, default=5)
    parser.add_argument("--test-percent", type=int, default=5)
    parser.add_argument("--skip-missing-audio", action="store_true")
    return parser.parse_args()


def load_dataset(path: Path):
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise SystemExit("Emilia-en metadata is stored as HuggingFace Arrow data. Activate the conda env or install `datasets`.") from exc
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            return dataset["train"]
        first_key = next(iter(dataset.keys()))
        return dataset[first_key]
    return dataset


def normalize_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def safe_id(audio_path: object) -> str:
    return Path(str(audio_path)).stem


def split_for_id(segment_id: str, dev_percent: int, test_percent: int) -> str:
    bucket = int(hashlib.sha1(segment_id.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < test_percent:
        return "test"
    if bucket < test_percent + dev_percent:
        return "dev"
    return "train"


def speaker_from_path(audio_path: object) -> str:
    path = Path(str(audio_path).strip().strip('"'))
    parts = path.parts
    if len(parts) > 1:
        return parts[0]
    stem_parts = path.stem.split("_")
    if len(stem_parts) >= 3 and stem_parts[0] == "EN" and stem_parts[1].startswith("B") and stem_parts[2].startswith("S"):
        return f"{stem_parts[0]}_{stem_parts[1]}_{stem_parts[2]}"
    return safe_id(path)


def storage_path(audio_root: Path, audio_path: str) -> str:
    path = Path(str(audio_path).strip().strip('"'))
    if path.is_absolute():
        return str(path)
    if len(path.parts) > 1:
        return str(audio_root / path)

    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0] == "EN" and parts[1].startswith("B") and parts[2].startswith("S"):
        batch_dir = f"{parts[0]}_{parts[1]}"
        segment_dir = f"{parts[0]}_{parts[1]}_{parts[2]}"
        return str(audio_root / batch_dir / segment_dir / "mp3" / path.name)
    return str(audio_root / path)


def row_to_record(row: dict, args: argparse.Namespace) -> dict[str, object]:
    audio_path = str(row["audio_path"]).strip().strip('"')
    segment_id = safe_id(audio_path)
    record = {
        "id": segment_id,
        "duration": row.get("speech_duration", ""),
        "speaker": speaker_from_path(audio_path),
        "split": split_for_id(segment_id, args.dev_percent, args.test_percent),
        "storage_path": storage_path(args.audio_root, audio_path),
        "sample_freq": 16000,
        "capspeech_prompt": normalize_value(row.get("caption")),
    }
    for field in ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]:
        if field in row:
            record[field] = normalize_value(row.get(field))
    return record


def main() -> int:
    args = parse_args()
    if args.dev_percent < 0 or args.test_percent < 0 or args.dev_percent + args.test_percent >= 100:
        raise ValueError("dev/test percentages must be non-negative and sum to less than 100")

    output_dir = args.output_root / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = BASE_FIELDS + OPTIONAL_FIELDS
    output_paths = {split: output_dir / f"{split}.csv" for split in ["train", "dev", "test"]}
    counts = {"train": 0, "dev": 0, "test": 0}
    files = {}
    writers = {}

    try:
        for split, output_path in output_paths.items():
            files[split] = output_path.open("w", newline="", encoding="utf-8")
            writers[split] = csv.DictWriter(files[split], fieldnames=fieldnames, extrasaction="ignore")
            writers[split].writeheader()
        dataset = load_dataset(args.metadata_path)
        for row in tqdm(dataset, total=len(dataset), desc="Preparing Emilia-en metadata", unit="row"):
            record = row_to_record(row, args)
            if args.skip_missing_audio and not Path(str(record["storage_path"])).exists():
                continue
            writers[record["split"]].writerow(record)
            counts[record["split"]] += 1
    finally:
        for file_obj in files.values():
            file_obj.close()

    for split, output_path in output_paths.items():
        print(f"Wrote {counts[split]} rows to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
