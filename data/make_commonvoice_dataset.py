#!/usr/bin/env python3
"""Prepare CommonVoice metadata in the unified PROPS CSV format."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

OPTIONAL_FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]
BASE_FIELDS = ["id", "duration", "speaker", "split", "storage_path", "sample_freq"]
DEFAULT_ROOT = Path("/export/fs05/corpora7/CommonVoice-19.0/en")

CATEGORIES = {
    "US": ["united states", "american", "california", "texas", "midwest", "new york", "boston", "chicago", "florida", "appalachian", "oregon", "washington", "rhode island", "pennsylvania", "philadelphia", "kentucky", "ohio", "new england"],
    "England": ["england", "british", "london", "rp", "received pronunciation", "yorkshire", "liverpool", "lancashire", "cumbrian", "midlands", "sussex", "east anglian", "northumbrian", "home counties"],
    "Scotland": ["scottish", "scotland"],
    "Ireland": ["irish", "northern irish", "belfast"],
    "Wales": ["welsh"],
    "Canada": ["canadian", "ontario", "toronto"],
    "Australia": ["australian", "australia", "sydney"],
    "New Zealand": ["new zealand", "kiwi"],
    "South Asia": ["india", "indian", "south asia", "pakistan", "sri lanka", "nepal", "nepalese", "bangladesh", "bangladeshi", "bengali", "south indian"],
    "Southeast Asia": ["singapore", "singaporean", "malaysia", "malaysian", "filipino", "philippines", "indonesia", "indonesian", "javanese", "thai", "vietnam"],
    "East Asia": ["chinese", "hong kong", "japanese", "hmong", "east asian", "tibetan"],
    "Africa": ["african", "south african", "southern african", "nigerian", "nigeria", "kenyan", "ghanaian", "west african", "east african", "afrikaans"],
    "Caribbean": ["west indies", "bermuda", "jamaica", "trinidad", "patois", "caribbean", "west indian"],
    "Latin America": ["latin", "latino", "hispanic", "mexican", "argentinian", "brazilian", "brazillian", "central american"],
    "French": ["french", "france"],
    "Germanic": ["german", "germany", "austrian", "swiss", "alemannic", "dutch"],
    "Nordic": ["swedish", "scandinavian", "danish", "norwegian", "finnish", "icelandic"],
    "Slavic / Eastern Europe": ["slavic", "polish", "russian", "ukrainian", "bulgarian", "czech", "croatian", "serbian", "slovak", "latvian", "romanian", "hungarian", "georgian", "kazakhstan"],
    "Southern Europe": ["italian", "spanish", "catalan", "greek", "cretan"],
    "Middle East": ["israeli", "turkish"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create data/<dataset_name>/{train,dev,test}.csv for CommonVoice.")
    parser.add_argument("dataset_name", nargs="?", default="CommonVoice")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--min-duration", type=float, default=5.0)
    return parser.parse_args()


def normalize_text(text: object) -> str:
    value = str(text).lower().strip()
    value = re.sub(r"[_/,-]+", " ", value)
    return re.sub(r"\s+", " ", value)


def categorize_accent(accent: object) -> str:
    value = normalize_text(accent)
    matches = [category for category, keywords in CATEGORIES.items() if any(keyword in value for keyword in keywords)]
    return matches[0] if len(matches) == 1 else "unknown"


def commonvoice_id(path_value: object) -> str:
    stem = Path(str(path_value)).stem
    return f"cv_{stem.split('_')[-1]}"


def build_speaker_ids(client_ids: pd.Series) -> dict[str, str]:
    return {client_id: f"cv_{idx}" for idx, client_id in enumerate(sorted(client_ids.dropna().astype(str).unique()))}


def write_split(df: pd.DataFrame, output_dir: Path, split: str) -> None:
    split_df = df.loc[df["split"] == split].copy()
    output_path = output_dir / f"{split}.csv"
    columns = BASE_FIELDS + [field for field in OPTIONAL_FIELDS if field in split_df.columns]
    split_df[columns].to_csv(output_path, index=False)
    print(f"Wrote {len(split_df)} rows to {output_path}", flush=True)


def main() -> int:
    args = parse_args()
    output_dir = args.output_root / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    duration_path = args.source_root / "clip_durations.tsv"
    if not duration_path.exists():
        raise FileNotFoundError(f"Missing {duration_path}")
    durations = pd.read_csv(duration_path, sep="\t")
    duration_by_clip = dict(zip(durations["clip"].astype(str), durations["duration[ms]"] / 1000.0))

    frames = []
    for split in tqdm(["train", "dev", "test"], desc="Reading CommonVoice splits", unit="split"):
        split_path = args.source_root / f"{split}.tsv"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing {split_path}")
        df = pd.read_csv(split_path, sep="\t")
        df["split"] = split
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["age", "gender", "accents", "client_id", "path"])
    df["duration"] = df["path"].astype(str).map(duration_by_clip)
    df = df.dropna(subset=["duration"])
    df = df[df["duration"] > args.min_duration]

    gender_map = {"male_masculine": "male", "female_feminine": "female"}
    df = df[df["gender"].isin(gender_map)]
    df["gender"] = df["gender"].map(gender_map)
    df["accent"] = df["accents"].map(categorize_accent)
    df = df[df["accent"] != "unknown"]

    speaker_map = build_speaker_ids(df["client_id"])
    df["speaker"] = df["client_id"].astype(str).map(speaker_map)
    df["id"] = df["path"].map(commonvoice_id)
    df["storage_path"] = df["path"].map(lambda value: str(args.source_root / "clips" / str(value)))
    df["sample_freq"] = 16000

    for split in tqdm(["train", "dev", "test"], desc="Writing CommonVoice metadata", unit="split"):
        write_split(df, output_dir, split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
