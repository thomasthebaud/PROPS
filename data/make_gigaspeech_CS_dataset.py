#!/usr/bin/env python3
"""Prepare CapSpeech-GigaSpeech metadata in the unified PROPS CSV format."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from pandas.errors import EmptyDataError

BASE_FIELDS = ["id", "duration", "speaker", "split", "storage_path", "sample_freq"]
OPTIONAL_FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent", "capspeech_prompt"]
DEFAULT_METADATA_ROOT = Path("/export/fs05/corpora7/capspeechset")
DEFAULT_AUDIO_ROOT = Path("/export/fs05/corpora7/CapSpeech-GigaSpeech")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create data/<dataset_name>/{train,dev,test}.csv for CapSpeech-GigaSpeech.")
    parser.add_argument("dataset_name", nargs="?", default="GigaSpeech")
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--num-file-check-workers", type=int, default=32)
    return parser.parse_args()


def source_split(split: str) -> str:
    return "validation" if split == "dev" else split


def safe_id(audio_path: object) -> str:
    return Path(str(audio_path)).stem


def speaker_from_id(segment_id: str) -> str:
    return segment_id.split("_")[0]


def read_csv_or_empty(path: Path, split: str, role: str) -> pd.DataFrame:
    if not path.exists():
        print(f"WARNING: missing {path}; treating {split} {role} metadata as empty", flush=True)
        return pd.DataFrame()
    if path.stat().st_size == 0:
        print(f"WARNING: empty {path}; treating {split} {role} metadata as empty", flush=True)
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        print(f"WARNING: no columns in {path}; treating {split} {role} metadata as empty", flush=True)
        return pd.DataFrame()


def file_exists(path: str) -> bool:
    return Path(path).exists()


def keep_existing_audio(df: pd.DataFrame, split: str, num_workers: int) -> pd.DataFrame:
    if df.empty:
        return df
    workers = max(1, num_workers)
    paths = df["storage_path"].astype(str).tolist()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        exists_mask = list(tqdm(
            executor.map(file_exists, paths),
            total=len(paths),
            desc=f"Checking {split} audio files",
            unit="file",
        ))
    missing = len(exists_mask) - sum(exists_mask)
    if missing:
        print(f"{split}: removed {missing} rows with missing audio", flush=True)
    else:
        print(f"{split}: all {len(exists_mask)} audio files exist", flush=True)
    return df.loc[exists_mask].copy()


def read_split(args: argparse.Namespace, split: str) -> pd.DataFrame:
    source = source_split(split)
    caption_path = args.metadata_root / f"{source}_PT_gigaspeech_caption.csv"
    metadata_path = args.metadata_root / f"{source}_gigaspeech.csv"
    caption_df = read_csv_or_empty(caption_path, split, "caption")
    if caption_df.empty:
        return caption_df
    metadata_df = read_csv_or_empty(metadata_path, split, "profile")
    if metadata_df.empty:
        print(f"WARNING: using caption metadata only for {split}", flush=True)
        df = caption_df
    else:
        df = pd.merge(caption_df, metadata_df, on="audio_path", suffixes=("", "_metadata"), how="left")
    if df.empty:
        return df
    df["split"] = split
    df["duration"] = df.get("speech_duration")
    df["storage_path"] = df["audio_path"].map(lambda value: str(args.audio_root / str(value)))
    df["sample_freq"] = 16000
    df["id"] = df["audio_path"].map(safe_id)
    df["speaker"] = df["id"].map(speaker_from_id)
    if "caption" in df.columns:
        df["capspeech_prompt"] = df["caption"]
    return keep_existing_audio(df, split, args.num_file_check_workers)


def write_split(df: pd.DataFrame, output_dir: Path, split: str) -> None:
    output_path = output_dir / f"{split}.csv"
    columns = BASE_FIELDS + [field for field in OPTIONAL_FIELDS if field in df.columns]
    if df.empty:
        pd.DataFrame(columns=BASE_FIELDS).to_csv(output_path, index=False)
    else:
        df[columns].to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}", flush=True)


def main() -> int:
    args = parse_args()
    output_dir = args.output_root / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in tqdm(["train", "dev", "test"], desc="Preparing GigaSpeech metadata", unit="split"):
        write_split(read_split(args, split), output_dir, split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
