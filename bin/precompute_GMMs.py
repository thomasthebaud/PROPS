#!/usr/bin/env python3
"""Precompute profile GMMs from all xvectors in a dataset."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


UNKNOWN_VALUE = "unknown"


def parse_csv_paths(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute one diagonal GMM per fully-known profile from the full "
            "dataset supplied in --train-csv-paths."
        )
    )
    parser.add_argument(
        "--train-csv-paths",
        type=parse_csv_paths,
        default=["CommonVoice_train"],
        help="Comma-separated dataset names under data/ to load in full.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--xvector-root", type=Path, default=Path("exp/xvectors/ecapa_tdnn"))
    parser.add_argument(
        "--profile-combinations-csv",
        type=Path,
        default=Path("data/profile_combinations.csv"),
        help="CSV containing profile combinations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/GMMs/precomputed"),
        help="Output directory for precomputed profile GMMs and metadata.csv.",
    )
    parser.add_argument("--K", type=int, default=1, help="Number of diagonal Gaussian components per profile GMM.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for GMM initialization.")
    parser.add_argument("--num-load-workers", type=int, default=16, help="Number of threads for xvector loading.")
    parser.add_argument("--min-sigma", type=float, default=1e-3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def normalize_value(value: Any) -> str:
    if pd.isna(value):
        return UNKNOWN_VALUE
    text = str(value).strip()
    return text if text else UNKNOWN_VALUE


def normalize_xvector(xvector: np.ndarray) -> np.ndarray:
    xvector = np.asarray(xvector, dtype=np.float64)
    norm = np.linalg.norm(xvector)
    if norm > 0.0:
        xvector = xvector / norm
    return xvector


def safe_file_value(value: Any) -> str:
    text = normalize_value(value).lower()
    safe_chars = [char if char.isalnum() else "_" for char in text]
    safe = "_".join("".join(safe_chars).split("_"))
    return safe or UNKNOWN_VALUE


def profile_file_stem(profile_index: int, profile: pd.Series, profile_fields: list[str]) -> str:
    parts = [f"profile_{profile_index}"]
    parts.extend(f"{field}_{safe_file_value(profile[field])}" for field in profile_fields)
    return "__".join(parts)


def load_profiles(path: Path) -> tuple[pd.DataFrame, list[str]]:
    profiles = pd.read_csv(path, dtype=str).fillna(UNKNOWN_VALUE)
    if profiles.empty:
        raise ValueError(f"{path} does not contain any profiles")

    profile_fields = [column for column in profiles.columns if not column.startswith("desc")]
    if not profile_fields:
        raise ValueError(f"{path} does not contain profile fields")

    for field in profile_fields:
        profiles[field] = profiles[field].map(normalize_value)

    known_mask = np.ones(len(profiles), dtype=bool)
    for field in profile_fields:
        known_mask &= profiles[field].str.lower().to_numpy() != UNKNOWN_VALUE

    return profiles.loc[known_mask].reset_index(drop=False).rename(columns={"index": "source_profile_index"}), profile_fields


def load_dataset_xvectors(
    dataset_names: list[str],
    xvector_root: Path,
    data_root: Path,
    profile_fields: list[str],
    num_load_workers: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    xvectors: list[np.ndarray] = []

    for dataset_name in dataset_names:
        segments_path = data_root / dataset_name / "segments.csv"
        if not segments_path.exists():
            raise FileNotFoundError(f"Missing segments file: {segments_path}")

        segments = pd.read_csv(segments_path, dtype=str).fillna(UNKNOWN_VALUE)
        if "id" not in segments.columns:
            raise ValueError(f"{segments_path} must contain an 'id' column")

        for field in profile_fields:
            if field not in segments.columns:
                segments[field] = UNKNOWN_VALUE
            segments[field] = segments[field].map(normalize_value)

        xvector_dataset_dir = xvector_root / dataset_name / "xvectors"
        load_items = []
        for _, row in segments.iterrows():
            utterance_id = normalize_value(row["id"])
            xvector_path = xvector_dataset_dir / f"{utterance_id}.npz"
            row_info = {
                "dataset": dataset_name,
                "utterance_id": utterance_id,
                "speaker": normalize_value(row.get("speaker", UNKNOWN_VALUE)),
                **{field: normalize_value(row[field]) for field in profile_fields},
            }
            load_items.append((xvector_path, row_info))

        def load_one(item: tuple[Path, dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray]:
            xvector_path, row_info = item
            if not xvector_path.exists():
                raise FileNotFoundError(f"Missing xvector: {xvector_path}")
            with np.load(xvector_path) as data:
                xvector = normalize_xvector(data["xvector"])
            return row_info, xvector

        if num_load_workers <= 1:
            iterator = map(load_one, load_items)
        else:
            executor = ThreadPoolExecutor(max_workers=num_load_workers)
            iterator = executor.map(load_one, load_items)

        try:
            for row_info, xvector in tqdm(
                iterator,
                total=len(load_items),
                desc=f"Loading xvectors {dataset_name} ({max(1, num_load_workers)} threads)",
                unit="xvec",
            ):
                rows.append(row_info)
                xvectors.append(xvector)
        finally:
            if num_load_workers > 1:
                executor.shutdown(wait=True)

    if not xvectors:
        raise ValueError(f"No xvectors loaded from {dataset_names}")

    return pd.DataFrame(rows), np.stack(xvectors)


def profile_mask(utterances: pd.DataFrame, profile: pd.Series, profile_fields: list[str]) -> np.ndarray:
    mask = np.ones(len(utterances), dtype=bool)
    for field in profile_fields:
        mask &= utterances[field].to_numpy() == normalize_value(profile[field])
    return mask


def fit_diag_gmm(
    xvectors: np.ndarray,
    num_components: int,
    min_sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if num_components < 1:
        raise ValueError(f"--K must be at least 1, got {num_components}")
    if len(xvectors) < num_components:
        raise ValueError(
            f"Need at least K xvectors to fit a K={num_components} GMM, got {len(xvectors)}"
        )
    if min_sigma <= 0.0:
        raise ValueError(f"--min-sigma must be positive, got {min_sigma}")

    if num_components == 1:
        pi = np.asarray([1.0], dtype=np.float64)
        pi_logits = np.asarray([0.0], dtype=np.float64)
        mu = np.mean(xvectors, axis=0, keepdims=True).astype(np.float64)
        sigma = np.std(xvectors, axis=0, keepdims=True).astype(np.float64)
        sigma = np.maximum(sigma, min_sigma)
        return pi_logits, pi, mu, sigma

    gmm = GaussianMixture(
        n_components=num_components,
        covariance_type="diag",
        random_state=seed,
        reg_covar=min_sigma**2,
        max_iter=500,
    )
    gmm.fit(xvectors)

    pi = gmm.weights_.astype(np.float64)
    mu = gmm.means_.astype(np.float64)
    sigma = np.sqrt(gmm.covariances_.astype(np.float64))
    sigma = np.maximum(sigma, min_sigma)
    pi_logits = np.log(np.maximum(pi, 1e-300)).astype(np.float64)
    return pi_logits, pi, mu, sigma


def save_gmm(path: Path, pi_logits: np.ndarray, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, pi_logits=pi_logits, pi=pi, mu=mu, sigma=sigma)
    tmp_path.replace(path)


def write_metadata(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def main() -> int:
    args = parse_args()
    dataset_names = parse_csv_paths(args.train_csv_paths)
    if args.K < 1:
        raise ValueError(f"--K must be at least 1, got {args.K}")

    profiles, profile_fields = load_profiles(args.profile_combinations_csv)
    print(
        f"Loaded {len(profiles)} fully-known profiles from {args.profile_combinations_csv} "
        f"using fields={profile_fields}",
        flush=True,
    )

    utterances, xvectors = load_dataset_xvectors(
        dataset_names,
        args.xvector_root,
        args.data_root,
        profile_fields,
        num_load_workers=args.num_load_workers,
    )
    print(f"Loaded the full dataset: {len(utterances)} xvectors from {dataset_names}", flush=True)

    metadata_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for _, profile in tqdm(
        profiles.iterrows(),
        total=len(profiles),
        desc="Precomputing profile GMMs",
        unit="profile",
    ):
        source_profile_index = int(profile["source_profile_index"])
        gmm_id = profile_file_stem(source_profile_index, profile, profile_fields)
        gmm_path = args.output_dir / "profiles" / f"{gmm_id}.npz"

        mask = profile_mask(utterances, profile, profile_fields)
        selected_xvectors = xvectors[mask]
        num_xvectors = len(selected_xvectors)
        if num_xvectors == 0:
            skipped_rows.append(
                {
                    "profile_index": source_profile_index,
                    "reason": "no_matching_xvectors",
                    **{field: normalize_value(profile[field]) for field in profile_fields},
                }
            )
            continue
        if num_xvectors < args.K:
            skipped_rows.append(
                {
                    "profile_index": source_profile_index,
                    "reason": f"insufficient_xvectors_for_K={args.K}",
                    **{field: normalize_value(profile[field]) for field in profile_fields},
                }
            )
            continue

        if not gmm_path.exists() or args.overwrite:
            pi_logits, pi, mu, sigma = fit_diag_gmm(
                selected_xvectors,
                num_components=args.K,
                min_sigma=args.min_sigma,
                seed=args.seed + source_profile_index,
            )
            save_gmm(gmm_path, pi_logits, pi, mu, sigma)

        metadata_rows.append(
            {
                "gmm_id": gmm_id,
                "profile_index": source_profile_index,
                "desc_column": "precomputed",
                "desc_num": "",
                "description": "",
                "gmm_path": str(gmm_path),
                "source_datasets": ",".join(dataset_names),
                "num_xvectors": num_xvectors,
                "K": args.K,
                **{field: normalize_value(profile[field]) for field in profile_fields},
            }
        )

        if args.verbose and len(metadata_rows) % 100 == 0:
            print(f"Computed {len(metadata_rows)} profile GMMs", flush=True)

    metadata_fields = [
        "gmm_id",
        "profile_index",
        "desc_column",
        "desc_num",
        "description",
        "gmm_path",
        "source_datasets",
        "num_xvectors",
        "K",
        *profile_fields,
    ]
    write_metadata(args.output_dir / "metadata.csv", metadata_fields, metadata_rows)

    if skipped_rows:
        skipped_fields = ["profile_index", "reason", *profile_fields]
        write_metadata(args.output_dir / "skipped_profiles.csv", skipped_fields, skipped_rows)
        print(f"Skipped {len(skipped_rows)} profiles with no matching xvectors", flush=True)

    print(f"Saved {len(metadata_rows)} K={args.K} profile GMMs to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
