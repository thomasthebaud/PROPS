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
from tqdm.auto import tqdm

from model import GaussianMDN, LearnablePi_MDN, ComposedGMM_MDN


FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]
DESC_COLUMNS = [f"desc{i}" for i in range(10)]
PROFILE_TEXT_COLUMNS = [*DESC_COLUMNS, "capspeech_desc"]
UNKNOWN_VALUES = {"", "nan", "none", "unknown", "not_computed"}
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
        choices=PROFILE_TEXT_COLUMNS,
        help="Description column from --profile-csv-path to use for profile inference.",
    )
    parser.add_argument(
        "--test-metadata-csv",
        type=Path,
        default=None,
        help="If provided, generate one GMM per row in this metadata CSV using --test-desc-column.",
    )
    parser.add_argument(
        "--test-desc-column",
        default="capspeech_desc",
        help="Description column to encode when --test-metadata-csv is used.",
    )
    parser.add_argument(
        "--fallback-profile-csv",
        type=Path,
        default=None,
        help="Profile prompts CSV used to fill missing test descriptions with matching desc0.",
    )
    parser.add_argument(
        "--sbert-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model used to encode test metadata descriptions.",
    )
    parser.add_argument(
        "--sbert-device",
        default=None,
        help="Device for SentenceTransformer test-metadata encoding. Defaults to --device when possible.",
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
        "--dataset-name",
        default=None,
        help="Profile dataset name under data/ and exp/SBERT_embs/.",
    )
    parser.add_argument(
        "--profile-csv-path",
        type=Path,
        default=None,
        help="CSV containing profile properties and desc0..desc9. Defaults to data/<dataset-name>/profile_prompts.csv.",
    )
    parser.add_argument(
        "--profile-combinations-csv",
        type=Path,
        default=None,
        help="Deprecated no-op; inference now uses every row in --profile-csv-path.",
    )
    parser.add_argument(
        "--sbert-root",
        type=Path,
        default=None,
        help="Root directory containing SBERT .npz embeddings. Defaults to exp/SBERT_embs/<dataset-name>.",
    )
    parser.add_argument(
        "--gmm-init-path",
        type=Path,
        default=None,
        help="Precomputed GMM directory used to initialize ComposedGMM_MDN.",
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
    parser.add_argument("--batch-size", type=int, default=256, help="MDN inference batch size.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing .npz files.")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N profiles.")
    parser.add_argument("--model-name", type=str, default="LearnablePi_MDN", choices=["LearnablePi_MDN", "GaussianMDN", "ComposedGMM_MDN"])
    return parser.parse_args()


def resolve_profile_paths(args: argparse.Namespace) -> None:
    if args.dataset_name is None and (args.profile_csv_path is None or args.sbert_root is None):
        raise ValueError("Pass --dataset-name, or pass --profile-csv-path and --sbert-root explicitly")

    if args.profile_csv_path is None:
        args.profile_csv_path = Path("data") / args.dataset_name / "profile_prompts.csv"
    if args.sbert_root is None:
        args.sbert_root = Path("exp/SBERT_embs") / args.dataset_name


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


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    model_name: str,
    gmm_init_path: Path | None = None,
):
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
        init_path = gmm_init_path or Path(f"exp/GMMs/{config['num_components']}_components_precomputed")
        model = ComposedGMM_MDN(
            input_dim=int(config["input_dim"]),
            output_dim=int(config["output_dim"]),
            num_components=int(config["num_components"]),
            hidden_dims=tuple(config["hidden_dims"]),
            dropout=float(config.get("dropout", 0.0)),
            gmm_init_path=str(init_path),
        ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_sbert_model(model_name: str, device: str | None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("Missing dependency: install sentence-transformers.") from exc
    return SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)


def encode_descriptions(sbert_model, descriptions: list[str]) -> np.ndarray:
    embeddings = sbert_model.encode(descriptions, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


def normalize_value(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def safe_file_value(value: Any) -> str:
    text = normalize_value(value).lower()
    safe_chars = [char if char.isalnum() else "_" for char in text]
    safe = "_".join("".join(safe_chars).split("_"))
    return safe or "unknown"


def profile_file_stem(profile_index: int, profile: pd.Series, fields: list[str], desc_column: str) -> str:
    parts = [f"profile_{profile_index}", desc_column]
    parts.extend(
        f"{field}_{safe_file_value(profile[field])}"
        for field in fields
        if normalize_value(profile[field]).lower() != "unknown"
    )
    if len(parts) == 2:
        parts.append("all_unknown")
    return "__".join(parts) + "_gmm"


def load_profiles(profile_csv_path: Path, profile_text_column: str) -> pd.DataFrame:
    profiles = pd.read_csv(profile_csv_path).fillna("unknown")
    if profile_text_column not in profiles.columns:
        raise ValueError(f"{profile_csv_path} is missing column: {profile_text_column}")
    for field in FIELDS:
        if field not in profiles.columns:
            profiles[field] = "unknown"
        profiles[field] = profiles[field].map(normalize_value)
    return profiles


def sbert_path(root: Path, profile_index: int, profile_text_column: str) -> Path:
    return root / str(profile_index) / f"{profile_text_column}.npz"


def load_sbert_embedding(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["embedding"], dtype=np.float32)


def generated_profile_path(output_dir: Path, gmm_id: str) -> Path:
    return output_dir / "profiles" / f"{gmm_id}.npz"


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


def is_known_text(value: Any) -> bool:
    return normalize_value(value).lower() not in UNKNOWN_VALUES


def profile_condition_key(row: pd.Series) -> tuple[str, ...]:
    return tuple(normalize_value(row.get(field, "unknown")) for field in FIELDS)


def load_profile_desc0_by_condition(profile_csv_path: Path | None) -> dict[tuple[str, ...], str]:
    if profile_csv_path is None or not profile_csv_path.exists():
        return {}

    profiles = pd.read_csv(profile_csv_path).fillna("unknown")
    if "desc0" not in profiles.columns:
        return {}
    for field in FIELDS:
        if field not in profiles.columns:
            profiles[field] = "unknown"
        profiles[field] = profiles[field].map(normalize_value)

    desc0_by_condition: dict[tuple[str, ...], str] = {}
    for _, profile in profiles.iterrows():
        desc0 = normalize_value(profile.get("desc0", "unknown"))
        if not is_known_text(desc0):
            continue
        key = profile_condition_key(profile)
        desc0_by_condition.setdefault(key, desc0)
    return desc0_by_condition


def load_test_metadata(
    test_metadata_csv: Path,
    desc_column: str,
    fallback_profile_csv: Path | None,
) -> pd.DataFrame:
    metadata = pd.read_csv(test_metadata_csv).fillna("unknown")
    for field in FIELDS:
        if field not in metadata.columns:
            metadata[field] = "unknown"
        metadata[field] = metadata[field].map(normalize_value)

    if desc_column not in metadata.columns:
        metadata[desc_column] = ""

    desc0_by_condition = load_profile_desc0_by_condition(fallback_profile_csv)
    candidate_columns = []
    for column in [desc_column, "capspeech_desc", "capspeech_prompt"]:
        if column in metadata.columns and column not in candidate_columns:
            candidate_columns.append(column)

    descriptions: list[str] = []
    source_counts = {column: 0 for column in candidate_columns}
    fallback_count = 0
    missing_count = 0
    for _, row in metadata.iterrows():
        description = ""
        for column in candidate_columns:
            value = normalize_value(row.get(column, "unknown"))
            if is_known_text(value):
                description = value
                source_counts[column] += 1
                break
        if not description:
            description = desc0_by_condition.get(profile_condition_key(row), "")
            if description:
                fallback_count += 1
        if not description:
            missing_count += 1
        descriptions.append(description)

    metadata[desc_column] = descriptions
    metadata = metadata.loc[metadata[desc_column].map(is_known_text)].copy()
    metadata.attrs["description_source_counts"] = source_counts
    metadata.attrs["fallback_desc0_count"] = fallback_count
    metadata.attrs["missing_description_count"] = missing_count
    metadata.attrs["fallback_profile_csv"] = str(fallback_profile_csv) if fallback_profile_csv else ""
    return metadata


def test_gmm_id(row_idx: int) -> str:
    return str(row_idx)


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


def progress_description(args: argparse.Namespace, desc_column: str) -> str:
    if args.test_metadata_csv is not None:
        split_name = args.test_metadata_csv.stem
    else:
        split_name = "profiles"
    return f"generating {split_name}-{desc_column}"


def main() -> int:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    model, checkpoint = load_model(checkpoint_path, device, args.model_name, args.gmm_init_path)

    metadata_rows: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], Path, np.ndarray]] = []

    print(f"Loaded checkpoint {checkpoint_path}", flush=True)
    print(f"Checkpoint epoch={checkpoint.get('epoch')} dev_loss={checkpoint.get('dev_loss')}", flush=True)

    if args.test_metadata_csv is not None:
        desc_column = args.test_desc_column
        fallback_profile_csv = args.fallback_profile_csv
        if fallback_profile_csv is None and args.dataset_name is not None:
            fallback_profile_csv = Path("data") / args.dataset_name / "profile_prompts.csv"
        test_rows = load_test_metadata(args.test_metadata_csv, desc_column, fallback_profile_csv)
        if args.limit is not None:
            test_rows = test_rows.head(args.limit)
        desc_num = None
        sbert_device = args.sbert_device or ("cuda" if device.type == "cuda" else "cpu")
        sbert_model = load_sbert_model(args.sbert_model, sbert_device)
        print(f"Loaded {len(test_rows)} test metadata rows from {args.test_metadata_csv}", flush=True)
        source_counts = test_rows.attrs.get("description_source_counts", {})
        print(
            "Description sources: "
            f"{desc_column}={source_counts.get(desc_column, 0)}, "
            f"capspeech_desc={source_counts.get('capspeech_desc', 0)}, "
            f"capspeech_prompt={source_counts.get('capspeech_prompt', 0)}, "
            f"profile_desc0={test_rows.attrs.get('fallback_desc0_count', 0)}",
            flush=True,
        )
        print(
            f"Filled {test_rows.attrs.get('fallback_desc0_count', 0)} missing descriptions "
            f"from matching profile desc0 in {test_rows.attrs.get('fallback_profile_csv', '')}",
            flush=True,
        )
        if test_rows.attrs.get("missing_description_count", 0):
            print(
                f"Skipped {test_rows.attrs['missing_description_count']} test rows with no usable description",
                flush=True,
            )
        print(f"Encoding test descriptions from column {desc_column} with {args.sbert_model}", flush=True)
        print(f"Generating one GMM per test row into {args.output_dir}", flush=True)

        for start in range(0, len(test_rows), args.batch_size):
            batch_rows = test_rows.iloc[start : start + args.batch_size]
            descriptions = batch_rows[desc_column].astype(str).tolist()
            embeddings = encode_descriptions(sbert_model, descriptions)
            for offset, (row_idx, test_row) in enumerate(batch_rows.iterrows()):
                profile_index = int(row_idx)
                gmm_id = test_gmm_id(profile_index)
                generated_path = generated_profile_path(args.output_dir, gmm_id)
                properties = {field: str(test_row[field]) for field in FIELDS}
                metadata = {
                    "gmm_id": gmm_id,
                    "profile_index": profile_index,
                    "desc_column": desc_column,
                    "desc_num": desc_num,
                    "description": str(test_row[desc_column]),
                    "gmm_path": str(generated_path),
                    "sbert_path": "",
                    "checkpoint_path": str(checkpoint_path),
                    **properties,
                }
                metadata_rows.append(metadata)
                if not generated_path.exists() or args.overwrite:
                    pending.append((metadata, generated_path, embeddings[offset]))
            print(f"Prepared {min(start + len(batch_rows), len(test_rows))}/{len(test_rows)} test rows", flush=True)
    else:
        resolve_profile_paths(args)
        desc_column = args.test_on_profile
        profiles = load_profiles(args.profile_csv_path, desc_column)
        if args.limit is not None:
            profiles = profiles.head(args.limit)
        desc_num = int(desc_column.removeprefix("desc")) if desc_column in DESC_COLUMNS else None

        print(f"Loaded {len(profiles)} profiles from {args.profile_csv_path}", flush=True)
        print(f"Using profile description column {desc_column}", flush=True)
        print(f"Generating one GMM per profile into {args.output_dir}", flush=True)
        missing_embs=0
        for row_idx, profile_row in profiles.iterrows():
            profile_index = int(row_idx)
            emb_path = sbert_path(args.sbert_root, profile_index, desc_column)
            if not emb_path.exists():
                missing_embs+=1
                continue

            embedding = load_sbert_embedding(emb_path)
            properties = {field: str(profile_row[field]) for field in FIELDS}
            description = str(profile_row[desc_column])
            gmm_id = profile_file_stem(profile_index, profile_row, FIELDS, desc_column)
            generated_path = generated_profile_path(args.output_dir, gmm_id)
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

        print(f"Missing SBERT embeddings={missing_embs}", flush=True)

    print(f"Need to generate {len(pending)} new files; metadata rows={len(metadata_rows)}", flush=True)

    progress = tqdm(
        total=len(pending),
        desc=progress_description(args, desc_column),
        unit="file",
    )
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

        progress.update(len(batch))
    progress.close()

    metadata_path = args.output_dir / "metadata.csv"
    write_metadata(metadata_path, metadata_rows)
    print(f"Wrote metadata to {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
