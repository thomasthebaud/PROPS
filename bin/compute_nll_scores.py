#!/usr/bin/env python3
"""Compute xvector negative log-likelihoods for generated MDN GMMs."""

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
    load_gmm,
    load_gmm_metadata,
    normalize_characteristic_value,
    normalize_value,
    normalize_xvector,
    row_xvector_dataset,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score dev/test xvectors against generated MDN GMMs.")
    parser.add_argument("--dataset-name", default="Capspeech_min100")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--gmm-root", type=Path, default=Path("exp/GMMs"))
    parser.add_argument("--xvector-root", type=Path, default=Path("exp/xvectors/ecapa_tdnn"))
    parser.add_argument("--output-root", type=Path, default=Path("exp/NLL_scores"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", choices=["pretrain", "finetune"], default=None)
    parser.add_argument("--train-fraction", default=None)
    parser.add_argument("--gmm-split", choices=["dev", "test"], default=None)
    parser.add_argument("--eval-set", choices=["dev", "test"], default=None)
    parser.add_argument("--desc-column", default=None)
    parser.add_argument("--weighting", choices=["learned", "uniform_pi"], default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def model_variants(dataset_name: str, k: int) -> list[dict[str, str]]:
    return [
        {"model": "pretrain", "train_fraction": "1", "split": "test", "desc_column": "desc0"},
        {"model": "pretrain", "train_fraction": "1", "split": "test", "desc_column": "capspeech_prompt"},
        {"model": "pretrain", "train_fraction": "1", "split": "dev", "desc_column": "desc1"},
        {"model": "pretrain", "train_fraction": "1", "split": "dev", "desc_column": "capspeech_prompt"},
        {"model": "finetune", "train_fraction": "0.1", "split": "test", "desc_column": "desc0"},
        {"model": "finetune", "train_fraction": "0.1", "split": "test", "desc_column": "capspeech_prompt"},
        {"model": "finetune", "train_fraction": "0.1", "split": "dev", "desc_column": "desc1"},
        {"model": "finetune", "train_fraction": "0.1", "split": "dev", "desc_column": "capspeech_prompt"},
    ]


def run_name(k: int, dataset_name: str, train_fraction: str, model: str) -> str:
    return f"{k}_components_{dataset_name}_p={train_fraction}_{model}"


def variant_name(variant: dict[str, str]) -> str:
    return (
        f"{variant['model']}_p={variant['train_fraction']}_"
        f"{variant['split']}_{variant['desc_column']}"
    )


def variant_dir(gmm_root: Path, dataset_name: str, k: int, variant: dict[str, str]) -> Path:
    return (
        gmm_root
        / f"{dataset_name}_{variant['split']}"
        / f"{run_name(k, dataset_name, variant['train_fraction'], variant['model'])}_{variant['desc_column']}"
    )


def load_split_xvectors(
    dataset_name: str,
    split: str,
    xvector_root: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
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
        paths = candidate_xvector_paths(xvector_dir, utterance)
        existing_path = next((path for path in paths if path.exists()), None)
        if existing_path is None:
            continue
        with np.load(existing_path) as data:
            xvector = normalize_xvector(np.asarray(data["xvector"], dtype=np.float64))
        rows.append(
            {
                "row_index": int(row_index),
                "utterance_id": normalize_value(utterance.get("id", row_index)),
                "dataset": row_dataset,
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


def uniform_pi(pi: np.ndarray) -> np.ndarray:
    return np.full(len(pi), 1.0 / len(pi), dtype=np.float64)


def profile_key(row: pd.Series) -> tuple[str, ...]:
    return tuple(normalize_characteristic_value(field, row.get(field, "unknown")) for field in FIELDS)


def build_profile_gmm_lookup(gmm_metadata: pd.DataFrame) -> dict[tuple[str, ...], pd.Series]:
    lookup: dict[tuple[str, ...], pd.Series] = {}
    for _, row in gmm_metadata.iterrows():
        lookup.setdefault(profile_key(row), row)
    return lookup


def build_prompt_gmm_lookup(gmm_metadata: pd.DataFrame) -> dict[int, pd.Series]:
    lookup: dict[int, pd.Series] = {}
    if "profile_index" not in gmm_metadata.columns:
        return lookup
    for _, row in gmm_metadata.iterrows():
        try:
            profile_index = int(row["profile_index"])
        except (TypeError, ValueError):
            continue
        lookup.setdefault(profile_index, row)
    return lookup


def matching_gmm_row(
    xvector_row: pd.Series,
    gmm_metadata: pd.DataFrame,
    profile_lookup: dict[tuple[str, ...], pd.Series],
    prompt_lookup: dict[int, pd.Series],
    desc_column: str,
) -> tuple[pd.Series | None, str]:
    if desc_column == "capspeech_prompt":
        try:
            row_index = int(xvector_row["row_index"])
        except (KeyError, TypeError, ValueError):
            return None, "missing_prompt_row"
        return prompt_lookup.get(row_index), "prompt_row"

    key = profile_key(xvector_row)
    return profile_lookup.get(key), "profile"


def score_against_gmm_rows(
    xvectors: np.ndarray,
    xvector_rows: pd.DataFrame,
    gmm_metadata: pd.DataFrame,
    *,
    desc_column: str,
    force_uniform_pi: bool,
    progress_desc: str,
) -> list[dict[str, Any]]:
    if gmm_metadata.empty:
        raise ValueError("Cannot score against empty GMM metadata")

    profile_lookup = build_profile_gmm_lookup(gmm_metadata)
    prompt_lookup = build_prompt_gmm_lookup(gmm_metadata)
    loaded_gmms: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    missing_matches = 0
    for score_index, xvector in tqdm(
        enumerate(xvectors),
        total=len(xvectors),
        desc=progress_desc,
        unit="score",
    ):
        xvector_row = xvector_rows.iloc[score_index]
        gmm_row, match_type = matching_gmm_row(
            xvector_row,
            gmm_metadata,
            profile_lookup,
            prompt_lookup,
            desc_column,
        )
        if gmm_row is None:
            missing_matches += 1
            continue

        gmm_path = str(gmm_row["gmm_path"])
        if gmm_path not in loaded_gmms:
            _pi_logits, pi, mu, sigma = load_gmm(Path(gmm_path))
            loaded_gmms[gmm_path] = (pi, mu, sigma)
        pi, mu, sigma = loaded_gmms[gmm_path]
        if force_uniform_pi:
            pi = uniform_pi(pi)
        nll = -float(gmm_log_likelihood(xvector.reshape(1, -1), pi, mu, sigma)[0])
        rows.append(
            {
                "score_index": score_index,
                "utterance_id": str(xvector_row.get("utterance_id", score_index)),
                "xvector_path": str(xvector_row.get("xvector_path", "")),
                "gmm_id": str(gmm_row.get("gmm_id", "unknown")),
                "gmm_path": gmm_path,
                "nll": nll,
            }
        )
    if missing_matches:
        print(
            f"WARNING: skipped {missing_matches}/{len(xvectors)} xvectors with no matching {desc_column} GMM",
            flush=True,
        )
    return rows

def load_available_metadata(metadata_path: Path) -> pd.DataFrame:
    metadata = load_gmm_metadata(metadata_path)
    available = metadata["gmm_path"].map(lambda value: Path(str(value)).exists())
    metadata = metadata.loc[available].reset_index(drop=True)
    if metadata.empty:
        raise FileNotFoundError(f"No GMM files found from metadata: {metadata_path}")
    return metadata


def write_score_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
            "gmm_id",
            "gmm_path",
            "nll",
        ],
    )


def build_eval_sets(
    dataset_name: str,
    xvector_root: Path,
    target_eval_set: str | None = None,
) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    eval_sets: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    requested_sets = [target_eval_set] if target_eval_set is not None else ["dev", "test"]

    for split in ["dev", "test"]:
        if split not in requested_sets:
            continue
        rows, xvectors = load_split_xvectors(dataset_name, split, xvector_root)
        eval_sets[split] = (rows, xvectors)
        print(f"Loaded {len(xvectors)} {split} xvectors", flush=True)

    return eval_sets

def resolve_single_variant(args: argparse.Namespace) -> dict[str, str] | None:
    provided = [args.model, args.train_fraction, args.gmm_split, args.eval_set, args.desc_column, args.weighting]
    if not any(value is not None for value in provided):
        return None
    missing = [
        name
        for name, value in [
            ("--model", args.model),
            ("--train-fraction", args.train_fraction),
            ("--gmm-split", args.gmm_split),
            ("--eval-set", args.eval_set),
            ("--desc-column", args.desc_column),
            ("--weighting", args.weighting),
        ]
        if value is None
    ]
    if missing:
        raise ValueError(f"Single-job scoring requires: {', '.join(missing)}")
    return {
        "model": str(args.model),
        "train_fraction": str(args.train_fraction),
        "split": str(args.gmm_split),
        "desc_column": str(args.desc_column),
    }


def output_csv_for(args: argparse.Namespace, variant: dict[str, str], weighting: str) -> Path:
    if args.output_csv is not None:
        return args.output_csv
    return args.output_root / "jobs" / f"{args.eval_set}_{variant_name(variant)}_{weighting}.csv"


def score_variant(
    args: argparse.Namespace,
    variant: dict[str, str],
    weighting: str,
    eval_sets: dict[str, tuple[pd.DataFrame, np.ndarray]],
) -> tuple[Path, list[dict[str, Any]]]:
    output_path = output_csv_for(args, variant, weighting)
    if output_path.exists() and not args.overwrite:
        print(f"Skipping existing {output_path}", flush=True)
        return output_path, pd.read_csv(output_path).to_dict("records")

    gmm_dir = variant_dir(args.gmm_root, args.dataset_name, args.k, variant)
    metadata_path = gmm_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = load_available_metadata(metadata_path)

    evaluation_set = str(args.eval_set)
    xvector_rows, xvectors = eval_sets[evaluation_set]
    rows = score_against_gmm_rows(
        xvectors,
        xvector_rows,
        metadata,
        desc_column=variant["desc_column"],
        force_uniform_pi=(weighting == "uniform_pi"),
        progress_desc=f"scoring {evaluation_set} {variant_name(variant)} {weighting}",
    )
    decorated_rows = [
        {
            "evaluation_set": evaluation_set,
            "gmm_variant": variant_name(variant),
            "model": variant["model"],
            "train_fraction": variant["train_fraction"],
            "gmm_split": variant["split"],
            "desc_column": variant["desc_column"],
            "weighting": weighting,
            **row,
        }
        for row in rows
    ]

    write_score_csv(output_path, decorated_rows)
    print(f"Wrote {output_path}", flush=True)
    return output_path, decorated_rows


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    single_variant = resolve_single_variant(args)
    eval_sets = build_eval_sets(
        args.dataset_name,
        args.xvector_root,
        target_eval_set=args.eval_set if single_variant is not None else None,
    )

    if single_variant is not None:
        score_variant(args, single_variant, str(args.weighting), eval_sets)
        return 0

    for evaluation_set in ["dev", "test"]:
        args.eval_set = evaluation_set
        for variant in model_variants(args.dataset_name, args.k):
            weightings = ["learned"]
            if variant["model"] == "pretrain":
                weightings.append("uniform_pi")
            for weighting in weightings:
                score_variant(args, variant, weighting, eval_sets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
