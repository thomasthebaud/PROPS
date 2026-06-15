#!/usr/bin/env python3
"""Precompute profile GMMs from all xvectors in a dataset."""

from __future__ import annotations

import argparse
import csv
import time
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
            "Precompute one diagonal GMM per profile from one or more train CSVs "
            "in the unified PROPS metadata format."
        )
    )
    parser.add_argument(
        "--train-csv-paths",
        type=parse_csv_paths,
        default=["data/Capspeech/train.csv"],
        help="Comma-separated train CSV paths, or legacy dataset names under data/.",
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
    parser.add_argument("--num-load-workers", type=int, default=16, help="Total thread budget for xvector loading across active profiles.")
    parser.add_argument("--num-profile-workers", type=int, default=16, help="Number of threads for profile-level GMM jobs.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of profile shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based profile shard index to process.")
    parser.add_argument("--min-sigma", type=float, default=1e-3)
    parser.add_argument(
        "--max-xvectors-per-profile",
        type=int,
        default=1000,
        help="Randomly sample at most this many existing xvectors per profile before fitting. Use 0 for no cap.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Recompute profile GMMs even when the output .npz already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated alias for --force-recompute.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.max_xvectors_per_profile < 0:
        parser.error("--max-xvectors-per-profile must be >= 0")
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= shard_index < num_shards")
    return args


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

    ignored_fields = {"dataset", "capspeech_desc", "num_audios"}
    profile_fields = [
        column
        for column in profiles.columns
        if not column.startswith("desc") and column not in ignored_fields
    ]
    if not profile_fields:
        raise ValueError(f"{path} does not contain profile fields")

    for field in profile_fields:
        profiles[field] = profiles[field].map(normalize_value)

    return profiles.reset_index(drop=False).rename(columns={"index": "source_profile_index"}), profile_fields


def resolve_train_csv_path(value: str, data_root: Path) -> Path:
    path = Path(value)
    if path.suffix == ".csv" or path.exists():
        return path
    legacy_segments = data_root / value / "segments.csv"
    if legacy_segments.exists():
        return legacy_segments
    legacy_split = data_root / f"{value}.csv"
    if legacy_split.exists():
        return legacy_split
    return path


def xvector_dataset_name(row: pd.Series, csv_path: Path) -> str:
    dataset = normalize_value(row.get("dataset", UNKNOWN_VALUE))
    split = normalize_value(row.get("split", UNKNOWN_VALUE))
    if dataset != UNKNOWN_VALUE and split != UNKNOWN_VALUE:
        return f"{dataset}_{split}"
    return csv_path.parent.name


def candidate_xvector_paths(row: pd.Series, xvector_dir: Path) -> list[Path]:
    candidates = []
    utterance_id = normalize_value(row["id"])
    candidates.append(xvector_dir / f"{utterance_id}.npz")
    storage_path = normalize_value(row.get("storage_path", UNKNOWN_VALUE))
    if storage_path != UNKNOWN_VALUE:
        storage_stem = Path(storage_path).stem
        candidates.append(xvector_dir / f"{storage_stem}.npz")
    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def load_train_metadata(
    csv_paths: list[str],
    xvector_root: Path,
    data_root: Path,
    profile_fields: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_value in csv_paths:
        segments_path = resolve_train_csv_path(csv_value, data_root)
        if not segments_path.exists():
            raise FileNotFoundError(f"Missing train CSV: {segments_path}")

        segments = pd.read_csv(segments_path, dtype=str).fillna(UNKNOWN_VALUE)
        if "id" not in segments.columns:
            raise ValueError(f"{segments_path} must contain an 'id' column")

        for field in profile_fields:
            if field not in segments.columns:
                segments[field] = UNKNOWN_VALUE
            segments[field] = segments[field].map(normalize_value)

        segments["utterance_id"] = segments["id"].map(normalize_value)
        segments["xvector_dataset"] = segments.apply(lambda row: xvector_dataset_name(row, segments_path), axis=1)
        if "dataset" not in segments.columns:
            segments["dataset"] = segments["xvector_dataset"].map(lambda value: value.rsplit("_", 1)[0])
        if "split" not in segments.columns:
            segments["split"] = segments["xvector_dataset"].map(lambda value: value.rsplit("_", 1)[-1])
        segments["dataset"] = segments["dataset"].map(normalize_value)
        segments["split"] = segments["split"].map(normalize_value)
        segments["xvector_dir"] = segments["xvector_dataset"].map(lambda value: str(xvector_root / value / "xvectors"))
        frames.append(segments)

    if not frames:
        raise ValueError(f"No train CSVs loaded from {csv_paths}")
    return pd.concat(frames, ignore_index=True, sort=False)


def resolve_xvector_path(row: pd.Series) -> Path | None:
    xvector_dir = Path(normalize_value(row["xvector_dir"]))
    for path in candidate_xvector_paths(row, xvector_dir):
        if path.exists():
            return path
    return None


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def estimate_remaining(elapsed: float, completed: int, total: int) -> str:
    if completed <= 0:
        return "estimating"
    seconds_per_profile = elapsed / completed
    return format_duration(seconds_per_profile * max(0, total - completed))


def rows_with_xvectors(profile_rows: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    rows = []
    for _, row in profile_rows.iterrows():
        xvector_path = resolve_xvector_path(row)
        if xvector_path is None:
            if verbose:
                xvector_dir = Path(normalize_value(row["xvector_dir"]))
                candidates = candidate_xvector_paths(row, xvector_dir)
                print(f"WARNING: missing xvector candidates: {', '.join(str(path) for path in candidates)}", flush=True)
            continue
        row_copy = row.copy()
        row_copy["resolved_xvector_path"] = str(xvector_path)
        rows.append(row_copy)
    if not rows:
        return profile_rows.iloc[0:0].copy()
    return pd.DataFrame(rows).reset_index(drop=True)


def sample_xvector_rows(
    rows: pd.DataFrame,
    max_xvectors: int,
    seed: int,
) -> pd.DataFrame:
    if max_xvectors <= 0 or len(rows) <= max_xvectors:
        return rows
    return rows.sample(n=max_xvectors, random_state=seed).sort_index().reset_index(drop=True)


def load_profile_xvectors(
    profile_rows: pd.DataFrame,
    num_load_workers: int,
    verbose: bool,
    show_progress: bool,
    profile_number: int,
) -> np.ndarray:
    def load_one(item: tuple[int, pd.Series]) -> np.ndarray | None:
        _, row = item
        xvector_path_value = row.get("resolved_xvector_path", UNKNOWN_VALUE)
        xvector_path = Path(normalize_value(xvector_path_value))
        if normalize_value(xvector_path_value) == UNKNOWN_VALUE or not xvector_path.exists():
            xvector_path = resolve_xvector_path(row)
        if xvector_path is None:
            if verbose:
                xvector_dir = Path(normalize_value(row["xvector_dir"]))
                candidates = candidate_xvector_paths(row, xvector_dir)
                print(f"WARNING: missing xvector candidates: {', '.join(str(path) for path in candidates)}", flush=True)
            return None
        with np.load(xvector_path) as data:
            return normalize_xvector(data["xvector"])

    load_items = list(profile_rows.iterrows())
    if num_load_workers <= 1:
        iterator = map(load_one, load_items)
    else:
        executor = ThreadPoolExecutor(max_workers=num_load_workers)
        iterator = executor.map(load_one, load_items)

    xvectors = []
    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(load_items),
            desc=f"Profile {profile_number} xvectors",
            unit="xvec",
            leave=False,
        )
    try:
        for xvector in iterator:
            if xvector is not None:
                xvectors.append(xvector)
    finally:
        if num_load_workers > 1:
            executor.shutdown(wait=True)

    if not xvectors:
        return np.empty((0, 0), dtype=np.float64)
    return np.stack(xvectors)


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


def load_existing_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "gmm_id" not in reader.fieldnames:
            return {}
        return {row["gmm_id"]: row for row in reader if row.get("gmm_id")}


def parse_saved_int(row: dict[str, str] | None, key: str) -> int | None:
    if row is None:
        return None
    try:
        return int(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def count_available_xvectors(profile_rows: pd.DataFrame) -> int:
    return sum(1 for _, row in profile_rows.iterrows() if resolve_xvector_path(row) is not None)


def main() -> int:
    args = parse_args()
    train_csv_paths = parse_csv_paths(args.train_csv_paths)
    if args.K < 1:
        raise ValueError(f"--K must be at least 1, got {args.K}")
    show_progress = args.shard_index == 0

    profiles, profile_fields = load_profiles(args.profile_combinations_csv)
    if show_progress:
        print(
            f"Loaded {len(profiles)} profiles from {args.profile_combinations_csv} "
            f"using fields={profile_fields}",
            flush=True,
        )

    utterances = load_train_metadata(
        train_csv_paths,
        args.xvector_root,
        args.data_root,
        profile_fields,
    )
    if show_progress:
        print(f"Loaded train metadata: {len(utterances)} rows from {train_csv_paths}", flush=True)

    existing_metadata = load_existing_metadata(args.output_dir / "metadata.csv")
    process_start = time.monotonic()
    metadata_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    effective_profile_workers = max(1, args.num_profile_workers)
    load_workers_per_profile = max(1, args.num_load_workers // effective_profile_workers)
    if show_progress:
        print(
            f"Profile workers={effective_profile_workers}; "
            f"xvector load workers per active profile={load_workers_per_profile}; "
            f"profile shard={args.shard_index}/{args.num_shards}",
            flush=True,
        )

    def process_profile(item: tuple[int, int, tuple[int, pd.Series]]) -> tuple[str, dict[str, Any]]:
        shard_position, profile_number, (_, profile) = item
        elapsed = time.monotonic() - process_start
        if show_progress:
            print(
                f"Starting profile {profile_number}/{total_profiles} "
                f"(shard item {shard_position}/{shard_total_profiles}) "
                f"elapsed={format_duration(elapsed)} "
                f"remaining_estimate={estimate_remaining(elapsed, shard_position - 1, shard_total_profiles)}",
                flush=True,
            )
        source_profile_index = int(profile["source_profile_index"])
        gmm_id = profile_file_stem(source_profile_index, profile, profile_fields)
        gmm_path = args.output_dir / "profiles" / f"{gmm_id}.npz"

        mask = profile_mask(utterances, profile, profile_fields)
        selected_rows = utterances.loc[mask]
        num_matching_rows = len(selected_rows)
        base_row = {
            "profile_index": source_profile_index,
            **{field: normalize_value(profile[field]) for field in profile_fields},
        }
        if num_matching_rows == 0:
            if show_progress:
                print(f"Skipping profile {profile_number}/{total_profiles}: no matching metadata rows", flush=True)
            return "skipped", {"reason": "no_matching_metadata_rows", **base_row}
        if num_matching_rows < args.K:
            if show_progress:
                print(f"Skipping profile {profile_number}/{total_profiles}: only {num_matching_rows} metadata rows for K={args.K}", flush=True)
            return "skipped", {"reason": f"insufficient_metadata_rows_for_K={args.K}", **base_row}

        available_xvector_rows = rows_with_xvectors(selected_rows, verbose=args.verbose)
        num_available_xvectors = len(available_xvector_rows)
        if num_available_xvectors == 0:
            if show_progress:
                print(f"Skipping profile {profile_number}/{total_profiles}: no matching xvectors", flush=True)
            return "skipped", {"reason": "no_matching_xvectors", **base_row}
        if num_available_xvectors < args.K:
            if show_progress:
                print(f"Skipping profile {profile_number}/{total_profiles}: only {num_available_xvectors} xvectors for K={args.K}", flush=True)
            return "skipped", {"reason": f"insufficient_xvectors_for_K={args.K}", **base_row}

        sampled_rows = sample_xvector_rows(
            available_xvector_rows,
            max_xvectors=args.max_xvectors_per_profile,
            seed=args.seed + source_profile_index,
        )
        num_xvectors = len(sampled_rows)

        saved_metadata = existing_metadata.get(gmm_id)
        saved_num_xvectors = parse_saved_int(saved_metadata, "num_xvectors")
        saved_k = parse_saved_int(saved_metadata, "K")
        recompute = (
            args.force_recompute
            or args.overwrite
            or not gmm_path.exists()
            or saved_num_xvectors != num_xvectors
            or saved_k != args.K
        )
        computed_now = False
        if recompute:
            if show_progress:
                cap_text = "uncapped" if args.max_xvectors_per_profile <= 0 else str(args.max_xvectors_per_profile)
                print(
                    f"Loading xvectors for profile {profile_number}/{total_profiles} "
                    f"({num_available_xvectors} available, using {num_xvectors}, cap={cap_text})",
                    flush=True,
                )
            selected_xvectors = load_profile_xvectors(
                sampled_rows,
                num_load_workers=load_workers_per_profile,
                verbose=args.verbose,
                show_progress=show_progress,
                profile_number=profile_number,
            )
            num_xvectors = len(selected_xvectors)
            if num_xvectors == 0:
                if show_progress:
                    print(f"Skipping profile {profile_number}/{total_profiles} after loading: no matching xvectors", flush=True)
                return "skipped", {"reason": "no_matching_xvectors", **base_row}
            if num_xvectors < args.K:
                if show_progress:
                    print(f"Skipping profile {profile_number}/{total_profiles} after loading: only {num_xvectors} xvectors for K={args.K}", flush=True)
                return "skipped", {"reason": f"insufficient_xvectors_for_K={args.K}", **base_row}
            if show_progress:
                print(f"Computing GMM for profile {profile_number}/{total_profiles} with {num_xvectors} xvectors, K={args.K}", flush=True)
            pi_logits, pi, mu, sigma = fit_diag_gmm(
                selected_xvectors,
                num_components=args.K,
                min_sigma=args.min_sigma,
                seed=args.seed + source_profile_index,
            )
            save_gmm(gmm_path, pi_logits, pi, mu, sigma)
            if show_progress:
                print(f"Saved GMM for profile {profile_number}/{total_profiles} to {gmm_path}", flush=True)
            computed_now = True
        elif show_progress:
            print(f"Reusing GMM for profile {profile_number}/{total_profiles} from {gmm_path} ({num_xvectors} xvectors)", flush=True)

        return "metadata", {
            "gmm_id": gmm_id,
            "profile_index": source_profile_index,
            "desc_column": "precomputed",
            "desc_num": "",
            "description": "",
            "gmm_path": str(gmm_path),
            "source_datasets": ",".join(train_csv_paths),
            "num_xvectors": num_xvectors,
            "K": args.K,
            "computed_now": computed_now,
            **{field: normalize_value(profile[field]) for field in profile_fields},
        }

    all_profile_items = list(enumerate(profiles.iterrows(), start=1))
    total_profiles = len(all_profile_items)
    profile_items = [
        item for item in all_profile_items if (item[0] - 1) % args.num_shards == args.shard_index
    ]
    profile_items = [
        (shard_position, profile_number, profile_item)
        for shard_position, (profile_number, profile_item) in enumerate(profile_items, start=1)
    ]
    shard_total_profiles = len(profile_items)
    if show_progress:
        print(f"Shard {args.shard_index}/{args.num_shards}: processing {shard_total_profiles}/{total_profiles} profiles", flush=True)
    if effective_profile_workers <= 1:
        profile_results = map(process_profile, profile_items)
    else:
        profile_executor = ThreadPoolExecutor(max_workers=effective_profile_workers)
        profile_results = profile_executor.map(process_profile, profile_items)

    try:
        for kind, row in tqdm(
            profile_results,
            total=len(profile_items),
            desc=f"Precomputing profile GMMs shard {args.shard_index}/{args.num_shards} ({effective_profile_workers} threads)",
            unit="profile",
            disable=not show_progress,
        ):
            if kind == "metadata":
                metadata_rows.append(row)
            else:
                skipped_rows.append(row)
            if args.verbose and len(metadata_rows) and len(metadata_rows) % 100 == 0:
                print(f"Computed {len(metadata_rows)} profile GMMs", flush=True)
    finally:
        if effective_profile_workers > 1:
            profile_executor.shutdown(wait=True)

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
        "computed_now",
        *profile_fields,
    ]
    metadata_path = args.output_dir / "metadata.csv"
    skipped_path = args.output_dir / "skipped_profiles.csv"
    if args.num_shards > 1:
        metadata_path = args.output_dir / f"metadata.shard{args.shard_index}.csv"
        skipped_path = args.output_dir / f"skipped_profiles.shard{args.shard_index}.csv"
    write_metadata(metadata_path, metadata_fields, metadata_rows)

    if skipped_rows:
        skipped_fields = ["profile_index", "reason", *profile_fields]
        write_metadata(skipped_path, skipped_fields, skipped_rows)
        if show_progress:
            print(f"Skipped {len(skipped_rows)} profiles with no matching xvectors", flush=True)

    computed_now = sum(1 for row in metadata_rows if row["computed_now"])
    reused = len(metadata_rows) - computed_now
    if show_progress:
        print(
            f"Saved {len(metadata_rows)} K={args.K} profile GMM metadata rows to {metadata_path} "
            f"computed_now={computed_now} reused_existing={reused}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
