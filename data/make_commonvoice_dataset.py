#!/usr/bin/env python3
"""Prepare CommonVoice metadata in the unified PROPS CSV format."""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_FIELDS = ["id", "duration", "speaker", "split", "storage_path", "sample_freq"]
OPTIONAL_FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent", "capspeech_prompt"]
DEFAULT_ROOT = Path("/export/fs05/corpora7/CommonVoice-19.0/en")
SPLITS = ["train", "dev", "test"]

AGE_MAP = {
    "teens": "teenager",
    "twenties": "young adult",
    "thirties": "young adult",
    "fourties": "middle-aged adult",
    "forties": "middle-aged adult",
    "fifties": "middle-aged adult",
    "sixties": "elderly",
    "seventies": "elderly",
    "eighties": "elderly",
    "nineties": "elderly",
}
GENDER_MAP = {"male_masculine": "male", "female_feminine": "female"}
ACCENT_MAP = {
    "american": [
        "united states", "american", "california", "texas", "midwest", "new york",
        "boston", "chicago", "florida", "appalachian", "oregon", "washington",
        "rhode island", "pennsylvania", "philadelphia", "kentucky", "ohio", "new england",
    ],
    "british": [
        "england", "british", "london", "rp", "received pronunciation", "yorkshire",
        "liverpool", "lancashire", "cumbrian", "midlands", "sussex", "east anglian",
        "northumbrian", "home counties",
    ],
    "scottish": ["scottish", "scotland"],
    "irish": ["irish", "northern irish", "belfast"],
    "welsh": ["welsh"],
    "canadian": ["canadian", "ontario", "toronto"],
    "australian": ["australian", "australia", "sydney"],
    "new zealand": ["new zealand", "kiwi"],
    "indian": ["india", "indian", "south asia", "south indian"],
    "filipino": ["filipino", "philippines"],
    "japanese": ["japanese"],
    "cantonese": ["cantonese", "hong kong"],
    "german": ["german", "germany", "austrian", "swiss", "alemannic"],
    "dutch": ["dutch"],
    "french": ["french", "france"],
    "italian": ["italian"],
    "mexican": ["mexican"],
    "brazilian": ["brazilian", "brazillian"],
    "colombian-american": ["colombian"],
    "russian": ["russian"],
    "czech": ["czech"],
    "hungarian": ["hungarian"],
    "norwegian": ["norwegian"],
    "portuguese": ["portuguese"],
    "belgian": ["belgian"],
    "slovenian": ["slovenian"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create data/<dataset_name>/{train,dev,test}.csv for CommonVoice.")
    parser.add_argument("dataset_name", nargs="?", default="CommonVoice")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--min-duration", type=float, default=5.0)
    parser.add_argument("--num-file-check-workers", type=int, default=32)
    return parser.parse_args()


def normalize_text(value: object) -> str:
    text = str(value).lower().strip()
    text = re.sub(r"[_/,-]+", " ", text)
    return re.sub(r"\s+", " ", text)


def categorize_accent(accent: object) -> str:
    text = normalize_text(accent)
    matches = [category for category, keywords in ACCENT_MAP.items() if any(keyword in text for keyword in keywords)]
    return matches[0] if len(matches) == 1 else "unknown"


def commonvoice_id(path_value: object) -> str:
    stem = Path(str(path_value)).stem
    return f"cv_{stem.split('_')[-1]}"


def build_speaker_ids(client_ids: pd.Series) -> dict[str, str]:
    return {client_id: f"cv_{idx}" for idx, client_id in enumerate(sorted(client_ids.dropna().astype(str).unique()))}


def file_exists(path: str) -> bool:
    return Path(path).exists()


def keep_existing_audio(df: pd.DataFrame, split: str, num_workers: int) -> pd.DataFrame:
    if df.empty:
        return df
    workers = max(1, num_workers)
    paths = df["storage_path"].astype(str).tolist()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        exists_mask = list(
            tqdm(
                executor.map(file_exists, paths),
                total=len(paths),
                desc=f"Checking CommonVoice {split} files",
                unit="file",
            )
        )
    missing = len(exists_mask) - sum(exists_mask)
    if missing:
        print(f"{split}: removed {missing} rows with missing audio", flush=True)
    else:
        print(f"{split}: all {len(exists_mask)} audio files exist", flush=True)
    return df.loc[exists_mask].copy()


def read_duration_lookup(source_root: Path) -> dict[str, float]:
    duration_path = source_root / "clip_durations.tsv"
    if not duration_path.exists():
        raise FileNotFoundError(f"Missing {duration_path}")
    durations = pd.read_csv(duration_path, sep="\t")
    return dict(zip(durations["clip"].astype(str), durations["duration[ms]"] / 1000.0))


def read_split(source_root: Path, split: str) -> pd.DataFrame:
    split_path = source_root / f"{split}.tsv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}")
    df = pd.read_csv(split_path, sep="\t")
    df["split"] = split
    return df


def prepare_metadata(args: argparse.Namespace) -> pd.DataFrame:
    duration_by_clip = read_duration_lookup(args.source_root)
    frames = [read_split(args.source_root, split) for split in tqdm(SPLITS, desc="Reading CommonVoice splits", unit="split")]
    df = pd.concat(frames, ignore_index=True, sort=False)

    df = df.dropna(subset=["age", "gender", "accents", "client_id", "path"])
    df = df[df["age"].isin(AGE_MAP)]
    df = df[df["gender"].isin(GENDER_MAP)]
    df["duration"] = df["path"].astype(str).map(duration_by_clip)
    df = df.dropna(subset=["duration"])
    df = df[df["duration"] > args.min_duration]

    df["age"] = df["age"].map(AGE_MAP).fillna("unknown")
    df["gender"] = df["gender"].map(GENDER_MAP).fillna("unknown")
    df["accent"] = df["accents"].map(categorize_accent).fillna("unknown")
    known_profile_mask = df[["gender", "age", "accent"]].ne("unknown").all(axis=1)
    removed_unknown_profiles = int((~known_profile_mask).sum())
    if removed_unknown_profiles:
        print(f"Removed {removed_unknown_profiles} rows with unknown gender, age, or accent", flush=True)
    df = df.loc[known_profile_mask].copy()

    speaker_map = build_speaker_ids(df["client_id"])
    df["speaker"] = df["client_id"].astype(str).map(speaker_map)
    df["id"] = df["path"].map(commonvoice_id)
    df["storage_path"] = df["path"].map(lambda value: str(args.source_root / "clips" / str(value)))
    df["sample_freq"] = 16000
    df["pitch"] = "unknown"
    df["speaking_rate"] = "unknown"
    df["speech_monotony"] = "unknown"
    df["capspeech_prompt"] = ""
    return df


def write_split(df: pd.DataFrame, output_dir: Path, split: str, num_workers: int) -> None:
    output_path = output_dir / f"{split}.csv"
    split_df = df.loc[df["split"] == split].copy()
    split_df = keep_existing_audio(split_df, split, num_workers)
    columns = BASE_FIELDS + OPTIONAL_FIELDS
    split_df[columns].to_csv(output_path, index=False)
    print(f"Wrote {len(split_df)} rows to {output_path}", flush=True)


def main() -> int:
    args = parse_args()
    output_dir = args.output_root / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_metadata(args)
    for split in tqdm(SPLITS, desc="Writing CommonVoice metadata", unit="split"):
        write_split(df, output_dir, split, args.num_file_check_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
