#!/usr/bin/env python3
"""Compute test-set xvector log-likelihoods under generated profile GMMs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]
DEFAULT_TEST_DATASETS = ["CommonVoice_test", "GigaSpeech_test"]


def parse_csv_paths(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute xvector likelihoods for generated profile GMMs.")
    parser.add_argument(
        "--test-csv-paths",
        type=parse_csv_paths,
        default=DEFAULT_TEST_DATASETS,
        help="Comma-separated dataset names under data/ to evaluate.",
    )
    parser.add_argument(
        "--profile-combinations-csv",
        type=Path,
        default=Path("data/profile_combinations.csv"),
        help="CSV containing profile conditions to evaluate.",
    )
    parser.add_argument(
        "--gmm-metadata-csv",
        type=Path,
        default=Path("exp/GMMs/metadata.csv"),
        help="Metadata CSV written by step 06.",
    )
    parser.add_argument(
        "--xvector-root",
        type=Path,
        default=Path("exp/xvectors/ecapa_tdnn"),
        help="Root directory containing test-set xvectors.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/metrics/loglikelihood"),
        help="Directory for likelihood summary CSV and full likelihood lists.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first N profile conditions.")
    return parser.parse_args()


def normalize_value(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def normalize_xvector(xvector: np.ndarray) -> np.ndarray:
    xvector = np.asarray(xvector, dtype=np.float64)
    norm = np.linalg.norm(xvector)
    if norm > 0:
        xvector = xvector / norm
    return xvector


def load_test_xvectors(
    dataset_names: list[str],
    xvector_root: Path,
    condition_fields: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    xvectors: list[np.ndarray] = []

    for dataset_name in dataset_names:
        segments_path = Path("data") / dataset_name / "segments.csv"
        segments = pd.read_csv(segments_path)
        for field in condition_fields:
            if field not in segments.columns:
                segments[field] = "unknown"
            segments[field] = segments[field].map(normalize_value)

        for _, segment in tqdm(
            segments.iterrows(),
            total=len(segments),
            desc=f"Loading xvectors {dataset_name}",
            unit="xvec",
        ):
            utterance_id = str(segment["id"])
            xvector_path = xvector_root / dataset_name / "xvectors" / f"{utterance_id}.npz"
            if not xvector_path.exists():
                raise FileNotFoundError(f"Missing xvector: {xvector_path}")
            with np.load(xvector_path) as data:
                xvectors.append(normalize_xvector(data["xvector"]))

            rows.append(
                {
                    "dataset": dataset_name,
                    "utterance_id": utterance_id,
                    "speaker": normalize_value(segment.get("speaker", "unknown")),
                    **{field: str(segment[field]) for field in condition_fields},
                }
            )

    if not xvectors:
        raise ValueError(f"No xvectors loaded from {dataset_names}")

    return pd.DataFrame(rows), np.stack(xvectors)


def condition_mask(
    utterances: pd.DataFrame,
    condition: pd.Series,
    condition_fields: list[str] | None = None,
) -> np.ndarray:
    condition_fields = condition_fields or FIELDS
    mask = np.ones(len(utterances), dtype=bool)
    for field in condition_fields:
        value = normalize_value(condition[field])
        if value != "unknown":
            mask &= utterances[field].to_numpy() == value
    return mask


def condition_key(row: pd.Series, condition_fields: list[str] | None = None) -> tuple[str, ...]:
    condition_fields = condition_fields or FIELDS
    return tuple(normalize_value(row[field]) for field in condition_fields)


def load_profile_conditions(path: Path) -> tuple[pd.DataFrame, list[str]]:
    conditions = pd.read_csv(path).fillna("unknown")
    condition_fields = [column for column in conditions.columns if column in FIELDS]
    if not condition_fields:
        raise ValueError(f"{path} must contain at least one profile field from {FIELDS}")
    for field in condition_fields:
        conditions[field] = conditions[field].map(normalize_value)
    return conditions, condition_fields


def load_gmm_metadata(path: Path, condition_fields: list[str]) -> dict[tuple[str, ...], pd.Series]:
    metadata = pd.read_csv(path).fillna("unknown")
    missing = [column for column in ["gmm_path", *condition_fields] if column not in metadata.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    lookup: dict[tuple[str, ...], pd.Series] = {}
    for _, row in metadata.iterrows():
        lookup.setdefault(condition_key(row, condition_fields), row)
    return lookup


def load_gmm(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        pi_logits = np.asarray(data["pi_logits"], dtype=np.float64)
        pi = np.asarray(data["pi"], dtype=np.float64)
        mu = np.asarray(data["mu"], dtype=np.float64)
        sigma = np.asarray(data["sigma"], dtype=np.float64)
    return pi_logits, pi, mu, sigma


def gmm_log_likelihood(xvectors: np.ndarray, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    if xvectors.ndim != 2:
        raise ValueError(f"Expected xvectors shape [N, D], got {xvectors.shape}")
    if mu.ndim != 2 or sigma.ndim != 2 or pi.ndim != 1:
        raise ValueError(f"Expected pi [K], mu/sigma [K, D], got pi={pi.shape} mu={mu.shape} sigma={sigma.shape}")
    if xvectors.shape[1] != mu.shape[1]:
        raise ValueError(f"xvector dim {xvectors.shape[1]} does not match GMM dim {mu.shape[1]}")

    sigma = np.maximum(sigma, 1e-12)
    pi = np.maximum(pi, 1e-300)
    diff = (xvectors[:, None, :] - mu[None, :, :]) / sigma[None, :, :]
    component_log_probs = -0.5 * (
        np.sum(diff * diff, axis=-1)
        + np.sum(2.0 * np.log(sigma), axis=-1)[None, :]
        + xvectors.shape[1] * np.log(2.0 * np.pi)
    )
    weighted = component_log_probs + np.log(pi)[None, :]
    max_log = np.max(weighted, axis=1, keepdims=True)
    return (max_log[:, 0] + np.log(np.sum(np.exp(weighted - max_log), axis=1))).astype(np.float64)


def safe_file_stem(gmm_id: str, profile_index: Any) -> str:
    text = str(gmm_id).strip() or f"profile_{profile_index}_gmm"
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)


def write_summary(path: Path, rows: list[dict[str, Any]], condition_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "gmm_id",
        "profile_index",
        "desc_column",
        "num_xvectors",
        "mean_log_likelihood",
        "std_log_likelihood",
        "full_loglikelihood_path",
        "gmm_path",
        *condition_fields,
    ]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def main() -> int:
    args = parse_args()
    full_lists_dir = args.output_dir / "full_lists"
    full_lists_dir.mkdir(parents=True, exist_ok=True)

    conditions, condition_fields = load_profile_conditions(args.profile_combinations_csv)
    if args.limit is not None:
        conditions = conditions.head(args.limit)

    utterances, xvectors = load_test_xvectors(args.test_csv_paths, args.xvector_root, condition_fields)
    gmm_lookup = load_gmm_metadata(args.gmm_metadata_csv, condition_fields)
    summary_rows: list[dict[str, Any]] = []

    for _, condition in tqdm(
        conditions.iterrows(),
        total=len(conditions),
        desc="Computing likelihoods",
        unit="profile",
    ):
        key = condition_key(condition, condition_fields)
        gmm_row = gmm_lookup.get(key)
        if gmm_row is None:
            raise KeyError(f"No generated GMM metadata found for condition={dict(zip(condition_fields, key))}")

        mask = condition_mask(utterances, condition, condition_fields)
        selected_xvectors = xvectors[mask]
        if len(selected_xvectors) == 0:
            continue

        gmm_path = Path(str(gmm_row["gmm_path"]))
        if not gmm_path.exists():
            raise FileNotFoundError(f"Missing GMM file: {gmm_path}")

        _pi_logits, pi, mu, sigma = load_gmm(gmm_path)
        log_likelihoods = gmm_log_likelihood(selected_xvectors, pi, mu, sigma)
        mean_log_likelihood = float(np.mean(log_likelihoods))
        std_log_likelihood = float(np.std(log_likelihoods))

        gmm_id = str(gmm_row.get("gmm_id", ""))
        profile_index = gmm_row.get("profile_index", "unknown")
        full_path = full_lists_dir / f"{safe_file_stem(gmm_id, profile_index)}_loglikelihoods.npz"
        np.savez_compressed(
            full_path,
            log_likelihoods=log_likelihoods.astype(np.float32),
            num_xvectors=np.asarray(len(log_likelihoods), dtype=np.int64),
        )

        summary_rows.append(
            {
                "gmm_id": gmm_id,
                "profile_index": profile_index,
                "desc_column": gmm_row.get("desc_column", "unknown"),
                "num_xvectors": len(log_likelihoods),
                "mean_log_likelihood": mean_log_likelihood,
                "std_log_likelihood": std_log_likelihood,
                "full_loglikelihood_path": str(full_path),
                "gmm_path": str(gmm_path),
                **{field: key[idx] for idx, field in enumerate(condition_fields)},
            }
        )

    summary_path = args.output_dir / "test_likelihoods.csv"
    write_summary(summary_path, summary_rows, condition_fields)
    print(f"Wrote summary to {summary_path}", flush=True)
    print(f"Wrote full likelihood lists to {full_lists_dir}", flush=True)

    if summary_rows:
        mean_values = np.asarray(
            [row["mean_log_likelihood"] for row in summary_rows],
            dtype=np.float64,
        )
        print(
            "Summary across evaluated profiles: "
            f"average mean log-likelihood={np.mean(mean_values):.6f}, "
            f"std of mean log-likelihoods={np.std(mean_values):.6f}, "
            f"profiles={len(summary_rows)}",
            flush=True,
        )
    else:
        print("Summary across evaluated profiles: no profiles had matching test xvectors", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
