#!/usr/bin/env python3
"""Fine-tune a pretrained ComposedGMM_MDN from per-utterance Capspeech descriptions."""

from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from model import ComposedGMM_MDN
from train import resolve_device, set_seed


FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]
DESC_COLUMNS = ["capspeech_desc", "capspeech_prompt"]
UNKNOWN_VALUES = {"", "nan", "none", "unknown", "not_computed"}


def safe_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "unknown"


def normalize_value(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def parse_csv_paths(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_dataset_fractions(value: str | None) -> dict[str, float]:
    if not value:
        return {}
    result: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, fraction = item.split("=", 1)
        parsed = float(fraction)
        if parsed > 1.0:
            parsed /= 100.0
        result[name.strip()] = parsed
    return result


class CapspeechDescriptionDataset(Dataset):
    def __init__(
        self,
        csv_paths: list[str],
        xvector_root: Path,
        desc_column: str = "auto",
        dataset_fractions: dict[str, float] | None = None,
        seed: int = 1234,
    ):
        self.xvector_root = xvector_root
        self.desc_column = desc_column
        self.dataset_fractions = dataset_fractions or {}
        self.rows: list[dict[str, Any]] = []
        self.missing_xvectors = 0
        self.missing_xvectors_by_dataset: dict[str, int] = {}
        rng = np.random.default_rng(seed)

        for csv_spec in csv_paths:
            csv_path = self.resolve_csv_path(csv_spec)
            needed_columns = {"id", "dataset", "split", "storage_path", *FIELDS, *DESC_COLUMNS}
            print(f"Reading {csv_path}...", flush=True)
            df = pd.read_csv(
                csv_path,
                low_memory=False,
                usecols=lambda column: column in needed_columns,
            )
            print(f"Read {len(df)} rows from {csv_path}", flush=True)
            if "id" not in df.columns:
                raise ValueError(f"{csv_path} is missing required column: id")
            active_desc_column = self.resolve_desc_column(df, csv_path)
            df[active_desc_column] = df[active_desc_column].fillna("").astype(str).str.strip()
            df = df.loc[~df[active_desc_column].str.lower().isin(UNKNOWN_VALUES)].copy()
            print(
                f"Kept {len(df)} rows from {csv_path} with non-empty {active_desc_column}",
                flush=True,
            )

            fraction = self.dataset_fractions.get(str(csv_spec), self.dataset_fractions.get(self.default_dataset_name(df, csv_path), 1.0))
            if fraction < 1.0:
                sample_size = max(1, int(round(len(df) * fraction)))
                indices = rng.choice(len(df), size=sample_size, replace=False)
                df = df.iloc[np.sort(indices)].reset_index(drop=True)
                print(f"Sampled {len(df)} rows from {csv_path} with fraction={fraction}", flush=True)

            for field in FIELDS:
                if field not in df.columns:
                    df[field] = "unknown"
                df[field] = df[field].map(normalize_value)

            for row_number, (_, row) in enumerate(df.iterrows(), start=1):
                if row_number % 100000 == 0:
                    print(
                        f"Scanned {row_number}/{len(df)} rows from {csv_path}; "
                        f"matched_xvectors={len(self.rows)} skipped_missing={self.missing_xvectors}",
                        flush=True,
                    )
                dataset_name = self.row_xvector_dataset(row, csv_path)
                item = {
                    "id": normalize_value(row.get("id", "unknown")),
                    "dataset": dataset_name,
                    "storage_path": normalize_value(row.get("storage_path", "unknown")),
                    "description": str(row[active_desc_column]).strip(),
                    **{field: normalize_value(row.get(field, "unknown")) for field in FIELDS},
                }
                xvector_path = self.resolve_xvector_path(item)
                if xvector_path is None:
                    self.missing_xvectors += 1
                    self.missing_xvectors_by_dataset[dataset_name] = self.missing_xvectors_by_dataset.get(dataset_name, 0) + 1
                    continue
                item["xvector_path"] = str(xvector_path)
                self.rows.append(item)

        if not self.rows:
            raise ValueError(f"No rows with Capspeech descriptions and xvectors found in {csv_paths}")
        print(
            f"Loaded {len(self.rows)} rows with Capspeech descriptions from {csv_paths}; "
            f"skipped_missing_xvectors={self.missing_xvectors}",
            flush=True,
        )
        if self.missing_xvectors_by_dataset:
            print("Missing xvectors by dataset:", flush=True)
            for dataset, count in sorted(self.missing_xvectors_by_dataset.items()):
                print(f"  {dataset}: {count}", flush=True)

    def resolve_desc_column(self, df: pd.DataFrame, csv_path: Path) -> str:
        if self.desc_column != "auto":
            if self.desc_column not in df.columns:
                raise ValueError(f"{csv_path} is missing description column: {self.desc_column}")
            return self.desc_column
        for column in DESC_COLUMNS:
            if column not in df.columns:
                continue
            values = df[column].fillna("").astype(str).str.strip().str.lower()
            if (~values.isin(UNKNOWN_VALUES)).any():
                return column
        raise ValueError(f"{csv_path} must contain non-empty descriptions in one of {DESC_COLUMNS}")

    @staticmethod
    def resolve_csv_path(value: str) -> Path:
        path = Path(value)
        if path.suffix == ".csv" or path.exists():
            return path
        if "_" in value:
            dataset, split = value.rsplit("_", 1)
            split_path = Path("data") / dataset / f"{split}.csv"
            if split_path.exists():
                return split_path
        return Path("data") / value

    @staticmethod
    def default_dataset_name(df: pd.DataFrame, csv_path: Path) -> str:
        if csv_path.name in {"train.csv", "dev.csv", "test.csv"}:
            return f"{csv_path.parent.name}_{csv_path.stem}"
        if "dataset" in df.columns and "split" in df.columns and not df.empty:
            dataset = normalize_value(df.iloc[0].get("dataset", "unknown"))
            split = normalize_value(df.iloc[0].get("split", "unknown"))
            if dataset != "unknown" and split != "unknown":
                return f"{dataset}_{split}"
        return csv_path.parent.name

    @staticmethod
    def row_xvector_dataset(row: pd.Series, csv_path: Path) -> str:
        dataset = normalize_value(row.get("dataset", "unknown"))
        split = normalize_value(row.get("split", "unknown"))
        if dataset != "unknown" and split != "unknown":
            return f"{dataset}_{split}"
        if csv_path.name in {"train.csv", "dev.csv", "test.csv"}:
            return f"{csv_path.parent.name}_{csv_path.stem}"
        return dataset if dataset != "unknown" else csv_path.parent.name

    def candidate_xvector_paths(self, row: dict[str, Any]) -> list[Path]:
        xvector_dir = self.xvector_root / str(row["dataset"]) / "xvectors"
        stems = [safe_name(row["id"]), str(row["id"])]
        storage_path = str(row.get("storage_path", "")).strip()
        if storage_path and storage_path != "unknown":
            stems.append(safe_name(Path(storage_path).stem))
        paths: list[Path] = []
        seen = set()
        for stem in stems:
            path = xvector_dir / f"{stem}.npz"
            if path not in seen:
                paths.append(path)
                seen.add(path)
        return paths

    def resolve_xvector_path(self, row: dict[str, Any]) -> Path | None:
        return next((candidate for candidate in self.candidate_xvector_paths(row) if candidate.exists()), None)

    def load_xvector(self, row: dict[str, Any]) -> torch.Tensor:
        path = Path(str(row["xvector_path"]))
        with np.load(path) as data:
            xvector = np.asarray(data["xvector"], dtype=np.float32)
        norm = np.linalg.norm(xvector)
        if norm > 0:
            xvector = xvector / norm
        return torch.from_numpy(xvector)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        row = self.rows[idx]
        return self.load_xvector(row), row["description"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MDN from per-utterance Capspeech descriptions.")
    parser.add_argument("--dataset-name", default="Capspeech_1000")
    parser.add_argument("--train-csv-paths", type=parse_csv_paths, default=None)
    parser.add_argument("--dev-csv-paths", type=parse_csv_paths, default=None)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--train-dataset-fractions", type=parse_dataset_fractions, default=None)
    parser.add_argument("--xvector-root", type=Path, default=Path("exp/xvectors/ecapa_tdnn"))
    parser.add_argument("--desc-column", default="auto", help="Description column to use, or auto for capspeech_desc/capspeech_prompt.")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gmm-init-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--mean-norm-weight", type=float, default=0.0)
    parser.add_argument("--variance-down-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--sbert-device", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_sbert_model(model_name: str, device: str | None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("Missing dependency: install sentence-transformers.") from exc
    return SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)


def encode_descriptions(sbert_model, descriptions: list[str], device: torch.device) -> torch.Tensor:
    embeddings = sbert_model.encode(descriptions, convert_to_numpy=True, show_progress_bar=False)
    return torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(device)


def load_pretrained_model(checkpoint_path: Path, device: torch.device, gmm_init_path: Path | None) -> tuple[ComposedGMM_MDN, dict[str, Any], dict[str, Any], int, int]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config")
    if not config:
        raise ValueError(f"{checkpoint_path} does not contain model_config")
    input_dim = int(config["input_dim"])
    output_dim = int(config["output_dim"])
    init_path = gmm_init_path or Path(checkpoint.get("gmm_init_path", f"exp/GMMs/{config['num_components']}_components_precomputed"))
    model = ComposedGMM_MDN(
        input_dim=input_dim,
        output_dim=output_dim,
        num_components=int(config["num_components"]),
        hidden_dims=tuple(config["hidden_dims"]),
        dropout=float(config.get("dropout", 0.0)),
        gmm_init_path=str(init_path),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint, config, input_dim, output_dim


def mdn_loss(model, embedding: torch.Tensor, xvector: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    nll = model.negative_log_likelihood(embedding, xvector)
    loss = nll
    entropy_loss = torch.zeros((), device=embedding.device)
    mean_norm_loss = torch.zeros((), device=embedding.device)
    variance_down_loss = torch.zeros((), device=embedding.device)
    if args.entropy_weight:
        entropy_loss = model.mixture_entropy_loss(embedding)
        loss = loss + args.entropy_weight * entropy_loss
    if args.mean_norm_weight:
        _pi, mu, _sigma = model(embedding)
        mean_norm_loss = torch.abs(1 - (mu ** 2).mean())
        loss = loss + args.mean_norm_weight * mean_norm_loss
    if args.variance_down_weight:
        variance_down_loss = model.variance_reduction_loss(embedding)
        loss = loss + args.variance_down_weight * variance_down_loss
    return loss, {
        "total_loss": loss,
        "nll": nll,
        "entropy_loss": entropy_loss,
        "mean_norm_loss": mean_norm_loss,
        "variance_down_loss": variance_down_loss,
    }


def run_epoch(
    model,
    sbert_model,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    optimizer=None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total_loss": 0.0, "nll": 0.0, "entropy_loss": 0.0, "mean_norm_loss": 0.0, "variance_down_loss": 0.0}
    total_items = 0
    phase = "train" if training else "dev"
    progress = tqdm(loader, desc=f"epoch {epoch:03d} {phase}", unit="batch")
    for xvector, descriptions in progress:
        xvector = xvector.float().to(device)
        embedding = encode_descriptions(sbert_model, list(descriptions), device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss, terms = mdn_loss(model, embedding, xvector, args)
            if training:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        batch_size = xvector.size(0)
        total_items += batch_size
        for name, value in terms.items():
            totals[name] += float(value.detach().cpu()) * batch_size
        progress.set_postfix(
            total=totals["total_loss"] / max(total_items, 1),
            nll=totals["nll"] / max(total_items, 1),
        )
    return {name: value / max(total_items, 1) for name, value in totals.items()}


def save_checkpoint(path: Path, model, optimizer, args, epoch: int, train_loss: float, dev_loss: float, model_config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "model_config": model_config,
            "dataset_name": args.dataset_name,
            "checkpoint": str(args.checkpoint),
            "desc_column": args.desc_column,
            "sbert_model": args.sbert_model,
            "fine_tune_capspeech_desc": True,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    if args.train_fraction > 1.0:
        args.train_fraction /= 100.0
    if args.train_csv_paths is None:
        args.train_csv_paths = [str(Path("data") / args.dataset_name / "train.csv")]
    if args.dev_csv_paths is None:
        args.dev_csv_paths = [str(Path("data") / args.dataset_name / "dev.csv")]
    if args.train_dataset_fractions is None:
        args.train_dataset_fractions = {args.train_csv_paths[0]: args.train_fraction}

    set_seed(args.seed)
    random.seed(args.seed)
    device = resolve_device(args.device)
    sbert_device = args.sbert_device or ("cuda" if device.type == "cuda" else "cpu")
    train_dataset = CapspeechDescriptionDataset(args.train_csv_paths, args.xvector_root, args.desc_column, args.train_dataset_fractions, args.seed)
    dev_dataset = CapspeechDescriptionDataset(args.dev_csv_paths, args.xvector_root, args.desc_column, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    model, checkpoint, model_config, input_dim, output_dim = load_pretrained_model(args.checkpoint, device, args.gmm_init_path)
    sbert_model = load_sbert_model(args.sbert_model, sbert_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Loaded pretrained checkpoint {args.checkpoint} epoch={checkpoint.get('epoch')}", flush=True)
    print(f"Train rows={len(train_dataset)} dev rows={len(dev_dataset)}", flush=True)
    print(f"Device={device} SBERT device={sbert_device}", flush=True)

    best_dev_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_metrics = run_epoch(model, sbert_model, train_loader, device, args, epoch, optimizer)
        with torch.no_grad():
            dev_metrics = run_epoch(model, sbert_model, dev_loader, device, args, epoch, optimizer=None)
        elapsed = time.perf_counter() - start
        print(
            f"epoch={epoch:03d} train_total={train_metrics['total_loss']:.6f} "
            f"train_nll={train_metrics['nll']:.6f} dev_nll={dev_metrics['nll']:.6f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )
        if dev_metrics["nll"] < best_dev_loss:
            best_dev_loss = dev_metrics["nll"]
            best_epoch = epoch
            save_checkpoint(args.output, model, optimizer, args, epoch, train_metrics["total_loss"], best_dev_loss, model_config)
            print(f"Saved checkpoint to {args.output}", flush=True)

    print(f"Done. Best epoch={best_epoch} best_dev_nll={best_dev_loss:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
