#!/usr/bin/env python3
"""Shared helpers for profile-conditioned xvector/GMM experiments."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]
PROFILE_FIELDS = ["age", "accent", "gender"]
DEFAULT_TEST_DATASETS = ["CommonVoice_test"]
AGE_LABEL_ORDER = [
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


def parse_csv_paths(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_value(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def sort_characteristic_labels(characteristic: str, labels: Iterable[str]) -> list[str]:
    labels = list(labels)
    if characteristic != "age":
        return sorted(labels)

    age_rank = {label: idx for idx, label in enumerate(AGE_LABEL_ORDER)}
    return sorted(labels, key=lambda label: (age_rank.get(label, len(age_rank)), label))


def normalize_xvector(xvector: np.ndarray) -> np.ndarray:
    xvector = np.asarray(xvector, dtype=np.float64)
    norm = np.linalg.norm(xvector)
    if norm > 0:
        xvector = xvector / norm
    return xvector


def condition_key(row: pd.Series | dict[str, Any], fields: list[str]) -> tuple[str, ...]:
    return tuple(normalize_value(row.get(field, "unknown")) for field in fields)


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
        value = normalize_value(condition.get(field, "unknown"))
        if unknown_is_wildcard and value == "unknown":
            continue
        mask &= utterances[field].map(normalize_value).to_numpy() == value
    return mask


def safe_file_stem(text: Any) -> str:
    value = normalize_value(text)
    return "".join(char if char.isalnum() or char in "._=-" else "_" for char in value)


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
    threshold_multiplier: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    metadata = pd.read_csv(path).fillna("unknown")
    if "gmm_path" not in metadata.columns:
        raise ValueError(f"{path} must contain a gmm_path column")
    for field in FIELDS:
        if field not in metadata.columns:
            metadata[field] = "unknown"
        metadata[field] = metadata[field].map(normalize_value)
    return metadata


def find_generated_gmm(metadata: pd.DataFrame, condition: dict[str, Any], strict_unknown_other_fields: bool = True) -> pd.Series | None:
    mask = np.ones(len(metadata), dtype=bool)
    for field in FIELDS:
        target = normalize_value(condition.get(field, "unknown"))
        if field in condition:
            mask &= metadata[field].map(normalize_value).to_numpy() == target
        elif strict_unknown_other_fields:
            mask &= metadata[field].map(normalize_value).str.lower().to_numpy() == "unknown"
    matches = metadata.loc[mask]
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

        segments_path = Path("data") / dataset_name / "segments.csv"
        segments = pd.read_csv(segments_path)
        if proportion is not None:
            if rng is None:
                rng = np.random.default_rng(0)
            sample_size = max(1, int(round(len(segments) * proportion)))
            sample_indices = rng.choice(len(segments), size=sample_size, replace=False)
            segments = segments.iloc[np.sort(sample_indices)].reset_index(drop=True)
            print(f"Using {len(segments)} sampled rows from {dataset_name} ({proportion:g})", flush=True)
        for field in FIELDS:
            if field not in segments.columns:
                segments[field] = "unknown"
            segments[field] = segments[field].map(normalize_value)
        if profile is not None:
            segments = segments.loc[condition_mask(segments, profile, profile_fields)].reset_index(drop=True)

        load_items = []
        for _, segment in segments.iterrows():
            utterance_id = str(segment["id"])
            xvector_path = xvector_root / dataset_name / "xvectors" / f"{utterance_id}.npz"
            row_info = {
                "dataset": dataset_name,
                "utterance_id": utterance_id,
                "speaker": normalize_value(segment.get("speaker", "unknown")),
                **{field: normalize_value(segment.get(field, "unknown")) for field in FIELDS},
            }
            load_items.append((xvector_path, row_info))

        def load_one(item: tuple[Path, dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray]:
            xvector_path, row_info = item
            if not xvector_path.exists():
                raise FileNotFoundError(f"Missing xvector: {xvector_path}")
            with np.load(xvector_path) as data:
                xvector = np.asarray(data["xvector"], dtype=np.float64)
            if normalize:
                xvector = normalize_xvector(xvector)
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
                desc=f"Loading xvectors {dataset_name}",
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
