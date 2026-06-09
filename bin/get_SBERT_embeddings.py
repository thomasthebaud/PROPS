#!/usr/bin/env python3
"""Create SBERT embeddings for profile prompt descriptions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DESC_COLUMNS = [f"desc{i}" for i in range(10)]
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed desc0..desc9 from data/profile_prompts.csv with SBERT."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/profile_prompts.csv"),
        help="CSV containing desc0..desc9 columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/SBERT_embs"),
        help="Directory where one compressed .npz per description is written.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformers model name.")
    parser.add_argument("--device", default=None, help="Optional device, e.g. cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of descriptions to encode per batch.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate files that already exist.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N CSV rows.")
    return parser.parse_args()


def load_model(model_name: str, device: str | None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install sentence-transformers in this environment "
            "with `python -m pip install sentence-transformers`."
        ) from exc

    if device:
        return SentenceTransformer(model_name, device=device)
    return SentenceTransformer(model_name)


def output_path(output_dir: Path, row_index: int, desc_column: str) -> Path:
    return output_dir / f"{row_index}/{desc_column}.npz"


def iter_jobs(df: pd.DataFrame, output_dir: Path, overwrite: bool) -> list[tuple[int, str, str, Path]]:
    jobs: list[tuple[int, str, str, Path]] = []
    for row_index, row in df.iterrows():
        for desc_column in DESC_COLUMNS:
            description = str(row[desc_column]).strip()
            if not description or description.lower() == "nan":
                continue

            path = output_path(output_dir, int(row_index), desc_column)
            if path.exists() and not overwrite:
                continue
            jobs.append((int(row_index), desc_column, description, path))
    return jobs


def save_embedding(
    path: Path,
    row_index: int,
    desc_column: str,
    description: str,
    model_name: str,
    embedding: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(
            f,
            embedding=np.asarray(embedding, dtype=np.float32),
            index=np.asarray(row_index, dtype=np.int64),
            desc_column=desc_column,
            description=description,
            model=model_name,
        )
    tmp_path.replace(path)


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)

    missing_columns = [column for column in DESC_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{args.input} is missing columns: {missing_columns}")

    if args.limit is not None:
        df = df.head(args.limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = iter_jobs(df=df, output_dir=args.output_dir, overwrite=args.overwrite)
    print(f"Loaded {len(df)} rows from {args.input}", flush=True)
    print(f"Embedding {len(jobs)} descriptions with {args.model}", flush=True)

    if not jobs:
        print("Nothing to do.", flush=True)
        return 0

    model = load_model(args.model, args.device)

    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start : start + args.batch_size]
        descriptions = [job[2] for job in batch]
        embeddings = model.encode(
            descriptions,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        for (row_index, desc_column, description, path), embedding in zip(batch, embeddings):
            save_embedding(
                path=path,
                row_index=row_index,
                desc_column=desc_column,
                description=description,
                model_name=args.model,
                embedding=embedding,
            )

        processed = min(start + len(batch), len(jobs))
        print(f"Saved {processed}/{len(jobs)} embeddings", flush=True)

    print(f"Saved embeddings in {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
