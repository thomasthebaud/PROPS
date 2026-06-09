#!/usr/bin/env python3
"""Generate profile-conditioned GMMs from a trained MDN."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from model import GaussianMDN, LearnablePi_MDN, ComposedGMM_MDN


FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]
DESC_COLUMNS = [f"desc{i}" for i in range(10)]
DEFAULT_TEST_DATASETS = ["CommonVoice_test", "GigaSpeech_test"]


def parse_csv_paths(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MDN inference for profile descriptions.")
    parser.add_argument(
        "--test-csv-paths",
        type=parse_csv_paths,
        default=DEFAULT_TEST_DATASETS,
        help="Deprecated; kept for compatibility with older test-set inference commands.",
    )
    parser.add_argument(
        "--test-on-profile",
        default="desc0",
        choices=DESC_COLUMNS,
        help="Description column from --profile-csv-path to use for profile inference.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="MDN checkpoint. Defaults to the newest .pt in --checkpoint-dir.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("exp/MDN_models"),
        help="Directory used to find the latest checkpoint if --checkpoint is omitted.",
    )
    parser.add_argument(
        "--profile-csv-path",
        type=Path,
        default=Path("data/profile_prompts.csv"),
        help="CSV containing profile properties and desc0..desc9.",
    )
    parser.add_argument(
        "--profile-combinations-csv",
        type=Path,
        default=Path("data/profile_combinations.csv"),
        help="Only run inference for profiles present in this combinations CSV.",
    )
    parser.add_argument(
        "--sbert-root",
        type=Path,
        default=Path("exp/SBERT_embs"),
        help="Root directory containing SBERT .npz embeddings.",
    )
    parser.add_argument(
        "--original-xvector-root",
        type=Path,
        default=Path("exp/xvectors/ecapa_tdnn"),
        help="Deprecated; test-set original xvectors are no longer used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/GMMs"),
        help="Directory for generated GMM .npz files and metadata.csv.",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="MDN inference batch size.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing .npz files.")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N profiles.")
    parser.add_argument("--model-name", type=str, default="LearnablePi_MDN", choices=["LearnablePi_MDN", "GaussianMDN", "ComposedGMM_MDN"])
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoints found in {checkpoint_dir}")
    return checkpoints[-1]


def load_model(checkpoint_path: Path, device: torch.device, model_name: str):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config")
    if not config:
        raise ValueError(f"{checkpoint_path} does not contain model_config")

    if model_name == "LearnablePi_MDN":
        model = LearnablePi_MDN(
        input_dim=int(config["input_dim"]),
        output_dim=int(config["output_dim"]),
        num_components=int(config["num_components"]),
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=float(config.get("dropout", 0.0)),
        ).to(device)
    elif model_name == "GaussianMDN":
        model = GaussianMDN(
            input_dim=int(config["input_dim"]),
            output_dim=int(config["output_dim"]),
            num_components=int(config["num_components"]),
            hidden_dims=tuple(config["hidden_dims"]),
            dropout=float(config.get("dropout", 0.0)),
        ).to(device)
    elif model_name == "ComposedGMM_MDN":
        model = ComposedGMM_MDN(
            input_dim=int(config["input_dim"]),
            output_dim=int(config["output_dim"]),
            num_components=int(config["num_components"]),
            hidden_dims=tuple(config["hidden_dims"]),
            dropout=float(config.get("dropout", 0.0)),
            gmm_init_path=f"exp/GMMs/{config['num_components']}_components_precomputed",
        ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def normalize_value(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def profile_key(row: pd.Series, fields: list[str]) -> tuple[str, ...]:
    return tuple(normalize_value(row[field]) for field in fields)


def load_profile_combinations(path: Path) -> tuple[pd.DataFrame, list[str], set[tuple[str, ...]]]:
    combinations = pd.read_csv(path).fillna("unknown")
    combination_fields = [column for column in combinations.columns if column in FIELDS]
    if not combination_fields:
        raise ValueError(f"{path} must contain at least one profile field from {FIELDS}")
    for field in combination_fields:
        combinations[field] = combinations[field].map(normalize_value)
    allowed_keys = {profile_key(row, combination_fields) for _, row in combinations.iterrows()}
    return combinations, combination_fields, allowed_keys


def load_profiles(profile_csv_path: Path) -> pd.DataFrame:
    profiles = pd.read_csv(profile_csv_path).fillna("unknown")
    missing_desc = [column for column in DESC_COLUMNS if column not in profiles.columns]
    if missing_desc:
        raise ValueError(f"{profile_csv_path} is missing columns: {missing_desc}")
    for field in FIELDS:
        if field not in profiles.columns:
            profiles[field] = "unknown"
        profiles[field] = profiles[field].map(normalize_value)
    return profiles


def filter_profiles_to_combinations(
    profiles: pd.DataFrame,
    combination_fields: list[str],
    allowed_keys: set[tuple[str, ...]],
) -> pd.DataFrame:
    missing = [field for field in combination_fields if field not in profiles.columns]
    if missing:
        raise ValueError(f"profile CSV is missing combination fields: {missing}")

    mask = profiles.apply(lambda row: profile_key(row, combination_fields) in allowed_keys, axis=1)
    for field in FIELDS:
        if field not in combination_fields:
            mask &= profiles[field].map(normalize_value).str.lower() == "unknown"
    return profiles.loc[mask]


def sbert_path(root: Path, profile_index: int, desc_num: int) -> Path:
    return root / str(profile_index) / f"desc{desc_num}.npz"


def load_sbert_embedding(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["embedding"], dtype=np.float32)


def generated_profile_path(output_dir: Path, profile_index: int, desc_column: str) -> Path:
    return output_dir / "profiles" / f"profile_{profile_index}_{desc_column}_gmm.npz"


def save_gmm_npz(
    path: Path,
    pi_logits: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(
            f,
            pi_logits=np.asarray(pi_logits, dtype=np.float32),
            pi=np.asarray(pi, dtype=np.float32),
            mu=np.asarray(mu, dtype=np.float32),
            sigma=np.asarray(sigma, dtype=np.float32),
        )
    tmp_path.replace(path)


def prune_low_weight_components(
    pi_logits: np.ndarray,
    pi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    threshold = 2.0 / len(pi)
    keep = pi >= threshold
    if not np.any(keep):
        keep[np.argmax(pi)] = True

    kept_pi = pi[keep]
    kept_pi = kept_pi / kept_pi.sum()
    kept_pi_logits = np.log(np.maximum(kept_pi, 1e-300))
    return kept_pi_logits, kept_pi, mu[keep], sigma[keep]


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "gmm_id",
        "profile_index",
        "desc_column",
        "desc_num",
        "description",
        "gmm_path",
        "sbert_path",
        "checkpoint_path",
        *FIELDS,
    ]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def main() -> int:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    model, checkpoint = load_model(checkpoint_path, device, args.model_name)

    profiles = load_profiles(args.profile_csv_path)
    combinations, combination_fields, allowed_keys = load_profile_combinations(args.profile_combinations_csv)
    profiles = filter_profiles_to_combinations(profiles, combination_fields, allowed_keys)
    if args.limit is not None:
        profiles = profiles.head(args.limit)
    desc_column = args.test_on_profile
    desc_num = int(desc_column.removeprefix("desc"))
    metadata_rows: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], Path, np.ndarray]] = []

    print(f"Loaded checkpoint {checkpoint_path}", flush=True)
    print(f"Checkpoint epoch={checkpoint.get('epoch')} dev_loss={checkpoint.get('dev_loss')}", flush=True)
    print(f"Loaded {len(combinations)} profile combinations from {args.profile_combinations_csv}", flush=True)
    print(f"Filtering inference profiles on fields={combination_fields}", flush=True)
    print(f"Loaded {len(profiles)} matching profiles from {args.profile_csv_path}", flush=True)
    print(f"Using profile description column {desc_column}", flush=True)
    print(f"Generating one GMM per profile into {args.output_dir}", flush=True)

    for row_idx, profile_row in profiles.iterrows():
        profile_index = int(row_idx)
        emb_path = sbert_path(args.sbert_root, profile_index, desc_num)
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing SBERT embedding: {emb_path}")

        embedding = load_sbert_embedding(emb_path)
        properties = {field: str(profile_row[field]) for field in FIELDS}
        description = str(profile_row[desc_column])
        generated_path = generated_profile_path(args.output_dir, profile_index, desc_column)
        gmm_id = f"profile_{profile_index}_{desc_column}_gmm"
        metadata = {
            "gmm_id": gmm_id,
            "profile_index": profile_index,
            "desc_column": desc_column,
            "desc_num": desc_num,
            "description": description,
            "gmm_path": str(generated_path),
            "sbert_path": str(emb_path),
            "checkpoint_path": str(checkpoint_path),
            **properties,
        }
        metadata_rows.append(metadata)

        if not generated_path.exists() or args.overwrite:
            pending.append((metadata, generated_path, embedding))

        if len(metadata_rows) % 1000 == 0:
            print(f"Prepared {len(metadata_rows)}/{len(profiles)} profiles", flush=True)

    print(f"Need to generate {len(pending)} new files; metadata rows={len(metadata_rows)}", flush=True)

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        embeddings = torch.from_numpy(np.stack([item[2] for item in batch])).float().to(device)
        with torch.no_grad():
            pi_logits, mu, sigma = model(embeddings)
            pi = torch.softmax(pi_logits, dim=-1)

        pi_logits_np = pi_logits.cpu().numpy()
        pi_np = pi.cpu().numpy()
        mu_np = mu.cpu().numpy()
        sigma_np = sigma.cpu().numpy()

        for idx, (metadata, generated_path, _embedding) in enumerate(batch):
            kept_pi_logits, kept_pi, kept_mu, kept_sigma = prune_low_weight_components(
                pi_logits_np[idx],
                pi_np[idx],
                mu_np[idx],
                sigma_np[idx],
            )
            save_gmm_npz(
                path=generated_path,
                pi_logits=kept_pi_logits,
                pi=kept_pi,
                mu=kept_mu,
                sigma=kept_sigma,
            )

        processed = min(start + len(batch), len(pending))
        print(f"Generated {processed}/{len(pending)} files", flush=True)

    metadata_path = args.output_dir / "metadata.csv"
    write_metadata(metadata_path, metadata_rows)
    print(f"Wrote metadata to {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
