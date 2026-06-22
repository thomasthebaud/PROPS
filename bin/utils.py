#!/usr/bin/env python3
"""Shared helpers for profile-conditioned xvector/GMM experiments."""

from __future__ import annotations

import csv
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]
DEFAULT_TEST_DATASETS = ["CommonVoice_test"]
AGE_LABEL_ORDER = [
    "child",
    "teenager",
    "young adult",
    "middle-aged adult",
    "elderly",
    "teens",
    "twenties",
    "thirties",
    "fourties",
    "forties",
    "fifties",
    "sixties",
    "seventies",
    "eighties",
    "nineties",
]
PITCH_LABEL_ORDER = [
    "very low-pitch",
    "low-pitch",
    "slightly low-pitch",
    "moderate pitch",
    "slightly high-pitch",
    "high-pitch",
    "very high-pitch",
]
SPEAKING_RATE_LABEL_ORDER = [
    "slowly",
    "slightly slowly",
    "moderate speed",
    "slightly fast",
    "fast",
]
SPEECH_MONOTONY_LABEL_ORDER = [
    "very monotone",
    "monotone",
    "slightly expressive and animated",
    "expressive and animated",
    "very expressive and animated",
]
CHARACTERISTIC_LABEL_ORDERS = {
    "age": AGE_LABEL_ORDER,
    "pitch": PITCH_LABEL_ORDER,
    "speaking_rate": SPEAKING_RATE_LABEL_ORDER,
    "speech_monotony": SPEECH_MONOTONY_LABEL_ORDER,
}
CHARACTERISTIC_LABEL_ALIASES = {
    "pitch": {
        "high-pitched": "high-pitch",
        "high_pitch": "high-pitch",
        "high_pitched": "high-pitch",
        "low-pitched": "low-pitch",
        "low_pitch": "low-pitch",
        "low_pitched": "low-pitch",
        "medium-pitched": "moderate pitch",
        "medium_pitch": "moderate pitch",
        "medium_pitched": "moderate pitch",
        "moderate_pitch": "moderate pitch",
    },
    "speaking_rate": {
        "slow speed": "slow",
        "slow_speed": "slow",
        "fast speed": "fast",
        "fast_speed": "fast",
    },
}


def parse_csv_paths(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_value(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def normalize_characteristic_value(characteristic: str, value: Any) -> str:
    text = normalize_value(value)
    aliases = CHARACTERISTIC_LABEL_ALIASES.get(characteristic, {})
    return aliases.get(text, text)


def sort_characteristic_labels(characteristic: str, labels: Iterable[str]) -> list[str]:
    labels = sorted({normalize_characteristic_value(characteristic, label) for label in labels})
    label_order = CHARACTERISTIC_LABEL_ORDERS.get(characteristic)
    if label_order is None:
        return sorted(labels)

    rank = {label: idx for idx, label in enumerate(label_order)}
    return sorted(labels, key=lambda label: (rank.get(label, len(rank)), label))


def normalize_xvector(xvector: np.ndarray) -> np.ndarray:
    xvector = np.asarray(xvector, dtype=np.float64)
    norm = np.linalg.norm(xvector)
    if norm > 0:
        xvector = xvector / norm
    return xvector


def condition_key(row: pd.Series | dict[str, Any], fields: list[str]) -> tuple[str, ...]:
    return tuple(normalize_characteristic_value(field, row.get(field, "unknown")) for field in fields)


def condition_mask(
    utterances: pd.DataFrame,
    condition: pd.Series | dict[str, Any],
    fields: list[str],
    unknown_is_wildcard: bool = True,
) -> np.ndarray:
    mask = np.ones(len(utterances), dtype=bool)
    for field in fields:
        if field not in utterances.columns:
            continue
        value = normalize_characteristic_value(field, condition.get(field, "unknown"))
        if unknown_is_wildcard and value == "unknown":
            continue
        mask &= utterances[field].map(normalize_value).to_numpy() == value
    return mask


def safe_file_value(value: Any) -> str:
    text = normalize_value(value).lower()
    safe_chars = [char if char.isalnum() else "_" for char in text]
    safe = "_".join("".join(safe_chars).split("_"))
    return safe or "unknown"


def parse_profile_fields_from_gmm_id(gmm_id: Any) -> dict[str, str]:
    """Recover profile fields from descriptive generated-GMM filename stems."""
    stem = Path(normalize_value(gmm_id)).stem
    if stem.endswith("_gmm"):
        stem = stem[: -len("_gmm")]

    parsed: dict[str, str] = {}
    fields_by_length = sorted(FIELDS, key=len, reverse=True)
    for part in stem.split("__"):
        for field in fields_by_length:
            prefix = f"{field}_"
            if part.startswith(prefix):
                value = part[len(prefix) :]
                if value and value != "unknown":
                    parsed[field] = value
                break
    return parsed


def values_match(left: Any, right: Any) -> bool:
    left_value = normalize_value(left)
    right_value = normalize_value(right)
    return left_value == right_value or safe_file_value(left_value) == safe_file_value(right_value)


def safe_file_stem(text: Any) -> str:
    value = normalize_value(text)
    return "".join(char if char.isalnum() or char in "._=-" else "_" for char in value)


def safe_xvector_stem(text: Any) -> str:
    value = normalize_value(text)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


SPLIT_NAMES = {"train", "dev", "test"}


def resolve_dataset_metadata(dataset_name: str) -> tuple[Path, str]:
    if "_" in dataset_name:
        base_name, split = dataset_name.rsplit("_", 1)
        if split in SPLIT_NAMES:
            split_csv_path = Path("data") / base_name / f"{split}.csv"
            if split_csv_path.exists():
                return split_csv_path, dataset_name

    dataset_dir = Path("data") / dataset_name
    if dataset_dir.is_dir():
        existing_splits = [split for split in ("test", "dev", "train") if (dataset_dir / f"{split}.csv").exists()]
        if "test" in existing_splits:
            return dataset_dir / "test.csv", f"{dataset_name}_test"
        if len(existing_splits) == 1:
            split = existing_splits[0]
            return dataset_dir / f"{split}.csv", f"{dataset_name}_{split}"
        if existing_splits:
            raise FileNotFoundError(
                f"{dataset_name!r} is ambiguous and has no test.csv; pass one of "
                f"{', '.join(f'{dataset_name}_{split}' for split in existing_splits)}"
            )

    raise FileNotFoundError(
        f"Could not find split metadata for {dataset_name!r}. Expected "
        f"data/<dataset>/<train|dev|test>.csv with a dataset name like CommonVoice_test."
    )


def row_xvector_dataset(row: pd.Series, fallback_dataset: str) -> str:
    row_dataset = normalize_value(row.get("dataset", "unknown"))
    row_split = normalize_value(row.get("split", "unknown"))
    if row_dataset != "unknown" and row_split != "unknown":
        return f"{row_dataset}_{row_split}"
    if row_dataset != "unknown":
        return row_dataset
    return fallback_dataset


def unique_preserving_order(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        text = normalize_value(value)
        if text == "unknown" or text in seen:
            continue
        seen.add(text)
        unique_values.append(text)
    return unique_values


def candidate_xvector_paths(xvector_dir: Path, row: pd.Series) -> list[Path]:
    utterance_id = normalize_value(row.get("id", "unknown"))
    storage_path = normalize_value(row.get("storage_path", "unknown"))
    stems = [safe_xvector_stem(utterance_id), utterance_id, safe_file_stem(utterance_id)]
    if storage_path != "unknown":
        stems.append(safe_xvector_stem(Path(storage_path).stem))
    return [xvector_dir / f"{stem}.npz" for stem in unique_preserving_order(stems)]


def load_profile_conditions(path: Path, fields: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    conditions = pd.read_csv(path).fillna("unknown")
    condition_fields = fields or [column for column in conditions.columns if column in FIELDS]
    if not condition_fields:
        raise ValueError(f"{path} must contain at least one profile field from {FIELDS}")
    for field in condition_fields:
        if field not in conditions.columns:
            conditions[field] = "unknown"
        conditions[field] = conditions[field].map(normalize_value)
    return conditions, condition_fields


def save_gmm(
    path: Path,
    *,
    pi: np.ndarray | None = None,
    mu: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    pi_logits: np.ndarray | None = None,
    sklearn_gmm: Any | None = None,
) -> None:
    """Save a diagonal GMM from model arrays or a fitted sklearn GaussianMixture."""
    if sklearn_gmm is not None:
        pi = np.asarray(sklearn_gmm.weights_, dtype=np.float64)
        mu = np.asarray(sklearn_gmm.means_, dtype=np.float64)
        covariances = np.asarray(sklearn_gmm.covariances_, dtype=np.float64)
        if sklearn_gmm.covariance_type == "diag":
            variances = covariances
        elif sklearn_gmm.covariance_type == "full":
            variances = np.stack([np.diag(covariance) for covariance in covariances])
        elif sklearn_gmm.covariance_type == "tied":
            variances = np.tile(np.diag(covariances), (len(pi), 1))
        elif sklearn_gmm.covariance_type == "spherical":
            variances = np.tile(covariances[:, None], (1, mu.shape[1]))
        else:
            raise ValueError(f"Unsupported sklearn covariance_type={sklearn_gmm.covariance_type!r}")
        sigma = np.sqrt(np.maximum(variances, 1e-12))
        pi_logits = np.log(np.maximum(pi, 1e-300))

    if pi is None or mu is None or sigma is None:
        raise ValueError("save_gmm requires either sklearn_gmm or pi/mu/sigma arrays")

    pi = np.asarray(pi, dtype=np.float32)
    mu = np.asarray(mu, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.float32)
    if pi_logits is None:
        pi_logits = np.log(np.maximum(pi.astype(np.float64), 1e-300))

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(
            f,
            pi_logits=np.asarray(pi_logits, dtype=np.float32),
            pi=pi,
            mu=mu,
            sigma=sigma,
        )
    tmp_path.replace(path)


def load_gmm(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        pi = np.asarray(data["pi"], dtype=np.float64)
        if "pi_logits" in data.files:
            pi_logits = np.asarray(data["pi_logits"], dtype=np.float64)
        else:
            pi_logits = np.log(np.maximum(pi, 1e-300))
        mu = np.asarray(data["mu"], dtype=np.float64)
        sigma = np.asarray(data["sigma"], dtype=np.float64)
    if mu.ndim == 1:
        mu = mu.reshape(1, -1)
    if sigma.ndim == 1:
        sigma = sigma.reshape(1, -1)
    if pi.ndim != 1 or mu.ndim != 2 or sigma.ndim != 2:
        raise ValueError(f"Invalid GMM shapes in {path}: pi={pi.shape} mu={mu.shape} sigma={sigma.shape}")
    return pi_logits, pi, mu, sigma


def prune_gmm_components(
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    threshold_multiplier: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # return pi, mu, sigma
    pi = np.asarray(pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    threshold = threshold_multiplier / len(pi)
    keep = pi >= threshold
    if not np.any(keep):
        keep[np.argmax(pi)] = True

    kept_pi = pi[keep]
    kept_pi = kept_pi / kept_pi.sum()
    return kept_pi, mu[keep], sigma[keep]


def sample_gmm(
    gmm_path_or_arrays: Path | tuple[np.ndarray, np.ndarray, np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if isinstance(gmm_path_or_arrays, (str, Path)):
        _pi_logits, pi, mu, sigma = load_gmm(Path(gmm_path_or_arrays))
    else:
        pi, mu, sigma = gmm_path_or_arrays

    pi = np.asarray(pi, dtype=np.float64)
    pi = np.maximum(pi, 0.0)
    pi = pi / pi.sum()
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)

    counts = rng.multinomial(n_samples, pi)
    samples = []
    for component_idx, count in enumerate(counts):
        if count == 0:
            continue
        noise = rng.normal(size=(count, mu.shape[1]))
        samples.append(mu[component_idx] + sigma[component_idx] * noise)
    if not samples:
        raise ValueError("No samples drawn from GMM")
    xvectors = np.vstack(samples)
    rng.shuffle(xvectors, axis=0)
    return xvectors


def sample_gmm_rows(
    gmm_rows: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator,
    *,
    prune_components: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    if gmm_rows.empty:
        raise ValueError("Cannot sample from an empty set of GMM rows")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    counts = rng.multinomial(n_samples, np.full(len(gmm_rows), 1.0 / len(gmm_rows)))
    batches = []
    sampled_row_indices = []
    for row_position, (count, (_, gmm_row)) in enumerate(zip(counts, gmm_rows.iterrows())):
        if count == 0:
            continue
        gmm_path = Path(str(gmm_row["gmm_path"]))
        if prune_components:
            _pi_logits, pi, mu, sigma = load_gmm(gmm_path)
            pi, mu, sigma = prune_gmm_components(pi, mu, sigma)
            batches.append(sample_gmm((pi, mu, sigma), int(count), rng))
        else:
            batches.append(sample_gmm(gmm_path, int(count), rng))
        sampled_row_indices.append(row_position)

    if not batches:
        raise ValueError("No samples drawn from matching GMM rows")
    return np.vstack(batches), gmm_rows.iloc[sampled_row_indices]


def gmm_log_likelihood(xvectors: np.ndarray, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    xvectors = np.asarray(xvectors, dtype=np.float64)
    pi = np.maximum(np.asarray(pi, dtype=np.float64), 1e-300)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    if xvectors.ndim != 2 or mu.ndim != 2 or sigma.ndim != 2 or pi.ndim != 1:
        raise ValueError(f"Expected xvectors [N,D], pi [K], mu/sigma [K,D]; got {xvectors.shape}, {pi.shape}, {mu.shape}, {sigma.shape}")
    if xvectors.shape[1] != mu.shape[1]:
        raise ValueError(f"xvector dim {xvectors.shape[1]} does not match GMM dim {mu.shape[1]}")

    diff = (xvectors[:, None, :] - mu[None, :, :]) / sigma[None, :, :]
    component_log_probs = -0.5 * (
        np.sum(diff * diff, axis=-1)
        + np.sum(2.0 * np.log(sigma), axis=-1)[None, :]
        + xvectors.shape[1] * np.log(2.0 * np.pi)
    )
    weighted = component_log_probs + np.log(pi / pi.sum())[None, :]
    max_log = np.max(weighted, axis=1, keepdims=True)
    return max_log[:, 0] + np.log(np.sum(np.exp(weighted - max_log), axis=1))


def load_gmm_metadata(path: Path) -> pd.DataFrame:
    metadata_dir = path.parent
    metadata = pd.read_csv(path).fillna("unknown")
    if "gmm_id" not in metadata.columns:
        if "gmm_path" in metadata.columns:
            metadata["gmm_id"] = metadata["gmm_path"].map(lambda item: Path(str(item)).stem)
        else:
            raise ValueError(f"{path} must contain a gmm_id or gmm_path column")

    for field in FIELDS:
        if field not in metadata.columns:
            metadata[field] = "unknown"
        metadata[field] = metadata[field].map(lambda value, field=field: normalize_characteristic_value(field, value))

    for index, row in metadata.iterrows():
        parsed_fields = parse_profile_fields_from_gmm_id(row.get("gmm_id", "unknown"))
        for field, value in parsed_fields.items():
            if normalize_value(metadata.at[index, field]).lower() == "unknown":
                metadata.at[index, field] = normalize_characteristic_value(field, value)

    if "gmm_path" not in metadata.columns:
        metadata["gmm_path"] = metadata["gmm_id"].map(
            lambda gmm_id: str(metadata_dir / "profiles" / f"{gmm_id}.npz")
        )
    else:
        def resolve_gmm_path(value: Any, gmm_id: Any) -> str:
            gmm_path = Path(str(value))
            if gmm_path.exists() or gmm_path.is_absolute():
                return str(gmm_path)
            metadata_relative = metadata_dir / gmm_path
            if metadata_relative.exists():
                return str(metadata_relative)
            sibling_profile = metadata_dir / "profiles" / f"{normalize_value(gmm_id)}.npz"
            if sibling_profile.exists():
                return str(sibling_profile)
            return str(gmm_path)

        metadata["gmm_path"] = [
            resolve_gmm_path(row.get("gmm_path", "unknown"), row.get("gmm_id", "unknown"))
            for _, row in metadata.iterrows()
        ]
    return metadata


def generated_gmm_matches(
    metadata: pd.DataFrame,
    condition: dict[str, Any],
    *,
    strict_unknown_other_fields: bool = True,
    unknown_is_wildcard: bool = False,
) -> pd.DataFrame:
    mask = np.ones(len(metadata), dtype=bool)
    for field in FIELDS:
        target = normalize_characteristic_value(field, condition.get(field, "unknown"))
        values = metadata[field].map(lambda value, field=field: normalize_characteristic_value(field, value)).to_numpy()
        if field in condition:
            if unknown_is_wildcard and target == "unknown":
                continue
            mask &= np.array([values_match(value, target) for value in values], dtype=bool)
        elif strict_unknown_other_fields:
            mask &= np.array([safe_file_value(value) == "unknown" for value in values], dtype=bool)
    return metadata.loc[mask]


def find_generated_gmm(
    metadata: pd.DataFrame,
    condition: dict[str, Any],
    strict_unknown_other_fields: bool = True,
    unknown_is_wildcard: bool = False,
) -> pd.Series | None:
    matches = generated_gmm_matches(
        metadata,
        condition,
        strict_unknown_other_fields=strict_unknown_other_fields,
        unknown_is_wildcard=unknown_is_wildcard,
    )
    if matches.empty:
        return None
    return matches.iloc[0]


def load_xvectors(
    dataset_names: list[str | tuple[str, float | None]],
    xvector_root: Path,
    profile: dict[str, Any] | None = None,
    profile_fields: list[str] | None = None,
    num_workers: int = 16,
    normalize: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    profile_fields = profile_fields or FIELDS
    rows: list[dict[str, Any]] = []
    xvectors: list[np.ndarray] = []

    for dataset_spec in dataset_names:
        if isinstance(dataset_spec, tuple):
            dataset_name, proportion = dataset_spec
        else:
            dataset_name, proportion = dataset_spec, None

        metadata_path, xvector_dataset = resolve_dataset_metadata(str(dataset_name))
        metadata = pd.read_csv(metadata_path)
        if "id" not in metadata.columns:
            raise ValueError(f"{metadata_path} is missing required column: id")
        if proportion is not None:
            if rng is None:
                rng = np.random.default_rng(0)
            sample_size = max(1, int(round(len(metadata) * proportion)))
            sample_indices = rng.choice(len(metadata), size=sample_size, replace=False)
            metadata = metadata.iloc[np.sort(sample_indices)].reset_index(drop=True)
            print(f"Using {len(metadata)} sampled rows from {dataset_name} ({proportion:g})", flush=True)

        for field in FIELDS:
            if field not in metadata.columns:
                metadata[field] = "unknown"
            metadata[field] = metadata[field].map(lambda value, field=field: normalize_characteristic_value(field, value))
        if profile is not None:
            metadata = metadata.loc[condition_mask(metadata, profile, profile_fields)].reset_index(drop=True)

        load_items = []
        for _, utterance in metadata.iterrows():
            utterance_id = normalize_value(utterance.get("id", "unknown"))
            row_dataset = row_xvector_dataset(utterance, xvector_dataset)
            xvector_dir = xvector_root / row_dataset / "xvectors"
            paths = candidate_xvector_paths(xvector_dir, utterance)
            row_info = {
                "dataset": row_dataset,
                "source_dataset": normalize_value(utterance.get("dataset", "unknown")),
                "split": normalize_value(utterance.get("split", "unknown")),
                "metadata_path": str(metadata_path),
                "utterance_id": utterance_id,
                "speaker": normalize_value(utterance.get("speaker", "unknown")),
                "storage_path": normalize_value(utterance.get("storage_path", "unknown")),
                **{field: normalize_characteristic_value(field, utterance.get(field, "unknown")) for field in FIELDS},
            }
            load_items.append((paths, row_info))

        def load_one(item: tuple[list[Path], dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray]:
            xvector_paths, row_info = item
            existing_path = next((candidate for candidate in xvector_paths if candidate.exists()), None)
            if existing_path is None:
                tried = ", ".join(str(candidate) for candidate in xvector_paths)
                raise FileNotFoundError(f"Missing xvector for id={row_info['utterance_id']!r}. Tried: {tried}")
            with np.load(existing_path) as data:
                xvector = np.asarray(data["xvector"], dtype=np.float64)
            if normalize:
                xvector = normalize_xvector(xvector)
            row_info["xvector_path"] = str(existing_path)
            return row_info, xvector

        if num_workers <= 1:
            iterator = map(load_one, load_items)
            executor = None
        else:
            executor = ThreadPoolExecutor(max_workers=num_workers)
            iterator = executor.map(load_one, load_items)
        try:
            for row_info, xvector in tqdm(
                iterator,
                total=len(load_items),
                desc=f"Loading xvectors {xvector_dataset}",
                unit="xvec",
            ):
                rows.append(row_info)
                xvectors.append(xvector)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    if not xvectors:
        return pd.DataFrame(rows), np.empty((0, 0), dtype=np.float64)
    return pd.DataFrame(rows), np.stack(xvectors)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)
