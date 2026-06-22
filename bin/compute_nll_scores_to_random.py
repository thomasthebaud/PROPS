#!/usr/bin/env python3
"""Score dev/test xvectors against random baseline distributions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from utils import (
    FIELDS,
    candidate_xvector_paths,
    gmm_log_likelihood,
    normalize_characteristic_value,
    normalize_value,
    normalize_xvector,
    row_xvector_dataset,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score split xvectors against N(0,1) or a random GMM.")
    parser.add_argument("--dataset-name", default="Capspeech_min100")
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--model", choices=["normal", "random"], required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--xvector-root", type=Path, default=Path("exp/xvectors/ecapa_tdnn"))
    parser.add_argument("--output-root", type=Path, default=Path("exp/NLL_scores"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_csv_for(args: argparse.Namespace) -> Path:
    if args.output_csv is not None:
        return args.output_csv
    return args.output_root / "random" / f"{args.model}_{args.split}.csv"


def load_split_xvectors(dataset_name: str, split: str, xvector_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    metadata_path = Path("data") / dataset_name / f"{split}.csv"
    metadata = pd.read_csv(metadata_path)
    if "id" not in metadata.columns:
        raise ValueError(f"{metadata_path} is missing required column: id")

    rows: list[dict[str, Any]] = []
    xvectors: list[np.ndarray] = []
    fallback_dataset = f"{dataset_name}_{split}"
    for row_index, utterance in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc=f"loading {dataset_name}-{split} xvectors",
        unit="xvec",
    ):
        for field in FIELDS:
            if field not in utterance:
                utterance[field] = "unknown"
        row_dataset = row_xvector_dataset(utterance, fallback_dataset)
        xvector_dir = xvector_root / row_dataset / "xvectors"
        existing_path = next((path for path in candidate_xvector_paths(xvector_dir, utterance) if path.exists()), None)
        if existing_path is None:
            continue
        with np.load(existing_path) as data:
            xvector = normalize_xvector(np.asarray(data["xvector"], dtype=np.float64))
        rows.append(
            {
                "score_index": int(row_index),
                "utterance_id": normalize_value(utterance.get("id", row_index)),
                "xvector_path": str(existing_path),
                **{
                    field: normalize_characteristic_value(field, utterance.get(field, "unknown"))
                    for field in FIELDS
                },
            }
        )
        xvectors.append(xvector)

    if not xvectors:
        raise ValueError(f"No xvectors loaded for {dataset_name}_{split}")
    return pd.DataFrame(rows), np.stack(xvectors)


def score_standard_normal(xvectors: np.ndarray) -> np.ndarray:
    dim = xvectors.shape[1]
    log_likelihood = -0.5 * (np.sum(xvectors * xvectors, axis=1) + dim * np.log(2.0 * np.pi))
    return -log_likelihood


def random_gmm(dim: int, k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pi = np.full(k, 1.0 / k, dtype=np.float64)
    mu = rng.normal(size=(k, dim))
    mu = np.stack([normalize_xvector(vector) for vector in mu])
    sigma = np.ones((k, dim), dtype=np.float64)
    return pi, mu, sigma


def write_scores(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        [
            "evaluation_set",
            "gmm_variant",
            "model",
            "train_fraction",
            "gmm_split",
            "desc_column",
            "weighting",
            "score_index",
            "utterance_id",
            "xvector_path",
            "nll",
        ],
    )


def main() -> int:
    args = parse_args()
    output_csv = output_csv_for(args)
    if output_csv.exists() and not args.overwrite:
        print(f"Skipping existing {output_csv}", flush=True)
        return 0

    xvector_rows, xvectors = load_split_xvectors(args.dataset_name, args.split, args.xvector_root)
    rng = np.random.default_rng(args.seed)
    if args.model == "normal":
        nlls = score_standard_normal(xvectors)
        model_label = "normal"
        variant = f"normal_{args.split}"
    else:
        pi, mu, sigma = random_gmm(xvectors.shape[1], args.k, rng)
        nlls = -gmm_log_likelihood(xvectors, pi, mu, sigma)
        model_label = "random_gmm"
        variant = f"random_gmm_K={args.k}_{args.split}"

    rows: list[dict[str, Any]] = []
    for row, nll in tqdm(
        zip(xvector_rows.to_dict("records"), nlls),
        total=len(nlls),
        desc=f"scoring {args.split} against {args.model}",
        unit="xvec",
    ):
        rows.append(
            {
                "evaluation_set": args.split,
                "gmm_variant": variant,
                "model": model_label,
                "train_fraction": "",
                "gmm_split": args.split,
                "desc_column": "all",
                "weighting": "baseline",
                "score_index": row["score_index"],
                "utterance_id": row["utterance_id"],
                "xvector_path": row["xvector_path"],
                "nll": float(nll),
            }
        )

    write_scores(output_csv, rows)
    print(f"Wrote {output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
