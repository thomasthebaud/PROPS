#!/usr/bin/env python3
"""Prepare CapSpeech CommonVoice metadata in the unified PROPS CSV format."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

BASE_FIELDS = ["id", "duration", "speaker", "split", "storage_path", "sample_freq"]
OPTIONAL_FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent", "capspeech_prompt"]
DEFAULT_METADATA_ROOT = Path("/export/fs05/corpora7/capspeechset/capspeechset1")
DEFAULT_AUDIO_ROOT = Path("/export/fs05/corpora7/CapSpeech-CommonVoice")
SPLIT_DIRS = {"train": "train", "dev": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create data/<dataset_name>/{train,dev,test}.csv for CapSpeech CommonVoice.")
    parser.add_argument("dataset_name", nargs="?", default="CommonVoice")
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--num-workers", type=int, default=16, help="Number of threads used for CommonVoice row conversion.")
    parser.add_argument("--skip-missing-audio", action="store_true")
    return parser.parse_args()


def load_dataset(path: Path):
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise SystemExit("CommonVoice metadata is stored as HuggingFace Arrow data. Activate the conda env or install `datasets`.") from exc
    return load_from_disk(str(path))


def normalize_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def clean_audio_path(audio_path: object) -> str:
    return str(audio_path).strip().strip('"')


def safe_id(audio_path: object) -> str:
    return Path(clean_audio_path(audio_path)).stem


def speaker_from_path(audio_path: object) -> str:
    parts = Path(clean_audio_path(audio_path)).parts
    if len(parts) >= 2:
        return parts[-2]
    return safe_id(audio_path).split("_")[0]


def storage_path(audio_root: Path, audio_path: str) -> str:
    path = Path(audio_path)
    if path.is_absolute():
        return str(path)
    return str(audio_root / path)


def row_to_record(row: dict, split: str, audio_root: Path) -> dict[str, object]:
    audio_path = clean_audio_path(row["audio_path"])
    record = {
        "id": safe_id(audio_path),
        "duration": row.get("speech_duration", ""),
        "speaker": speaker_from_path(audio_path),
        "split": split,
        "storage_path": storage_path(audio_root, audio_path),
        "sample_freq": 16000,
        "capspeech_prompt": normalize_value(row.get("caption")),
    }
    for field in ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]:
        if field in row:
            record[field] = normalize_value(row.get(field))
    return record


def should_keep_record(record: dict[str, object], skip_missing_audio: bool) -> bool:
    return not skip_missing_audio or Path(str(record["storage_path"])).exists()


def records_for_split(args: argparse.Namespace, split: str) -> Iterable[dict[str, object]]:
    dataset_path = args.metadata_root / SPLIT_DIRS[split]
    dataset = load_dataset(dataset_path)
    desc = f"Preparing CommonVoice {split}"
    num_workers = max(1, args.num_workers)

    def convert_if_commonvoice(row: dict) -> dict[str, object] | None:
        if row.get("source") != "commonvoice":
            return None
        record = row_to_record(row, split, args.audio_root)
        if not should_keep_record(record, args.skip_missing_audio):
            return None
        return record

    if num_workers == 1:
        for row in tqdm(dataset, total=len(dataset), desc=desc, unit="row"):
            record = convert_if_commonvoice(row)
            if record is not None:
                yield record
        return

    max_pending = num_workers * 4
    rows = iter(dataset)
    pending = set()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(max_pending):
            try:
                pending.add(executor.submit(convert_if_commonvoice, next(rows)))
            except StopIteration:
                break

        with tqdm(total=len(dataset), desc=desc, unit="row") as progress:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    progress.update(1)
                    record = future.result()
                    if record is not None:
                        yield record
                    try:
                        pending.add(executor.submit(convert_if_commonvoice, next(rows)))
                    except StopIteration:
                        pass


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
    for split in tqdm(["train", "dev", "test"], desc="Writing CommonVoice splits", unit="split"):
        write_split(args, split, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
