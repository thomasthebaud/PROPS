#!/usr/bin/env python3
"""Pretrain ComposedGMM_MDN pi routing from profile SBERT embeddings only."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model import ComposedGMM_MDN
from train import (
    build_profile_component_targets,
    pi_profile_cross_entropy,
    resolve_device,
    set_seed,
)

DEFAULT_PROFILE_FIELDS = ["pitch", "age", "gender", "speaking_rate", "speech_monotony", "accent"]

TRAIN_DESC_COLUMNS = [f"desc{i}" for i in range(1, 9)]
VAL_DESC_COLUMNS = ["desc9"]


class ProfileEmbeddingDataset(Dataset):
    def __init__(
        self,
        profile_csv_path: Path,
        sbert_root: Path,
        desc_columns: list[str],
    ):
        self.profile_csv_path = profile_csv_path
        self.sbert_root = sbert_root
        self.desc_columns = desc_columns
        self.profiles = pd.read_csv(profile_csv_path).fillna("unknown")
        for field in DEFAULT_PROFILE_FIELDS:
            if field not in self.profiles.columns:
                self.profiles[field] = "unknown"

        self.items: list[tuple[int, str, Path]] = []
        for profile_index, row in self.profiles.iterrows():
            for desc_column in desc_columns:
                if desc_column not in self.profiles.columns:
                    continue
                value = str(row.get(desc_column, "")).strip()
                if not value or value.lower() in {"nan", "unknown", "not_computed"}:
                    continue
                emb_path = self.sbert_root / str(int(profile_index)) / f"{desc_column}.npz"
                if emb_path.exists():
                    self.items.append((int(profile_index), desc_column, emb_path))

        if not self.items:
            raise ValueError(
                f"No SBERT embeddings found under {sbert_root} for {profile_csv_path} "
                f"columns={desc_columns}"
            )
        print(
            f"Loaded {len(self.profiles)} profiles and {len(self.items)} SBERT embedding rows "
            f"from {profile_csv_path}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        profile_index, _desc_column, emb_path = self.items[idx]
        with np.load(emb_path) as data:
            embedding = np.asarray(data["embedding"], dtype=np.float32)
        return torch.from_numpy(embedding), torch.tensor(profile_index, dtype=torch.long)


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain ComposedGMM_MDN pi head with profile CE only.")
    parser.add_argument("--dataset-name", default="Capspeech_1000")
    parser.add_argument("--profile-csv-path", type=Path, default=None)
    parser.add_argument("--sbert-root", type=Path, default=None)
    parser.add_argument("--gmm-init-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-components", type=int, default=4)
    parser.add_argument("--hidden-dims", default="1024,2048,1024")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pi-profile-ce-weight", type=float, default=1.0)
    parser.add_argument("--train-desc-columns", type=parse_csv_list, default=TRAIN_DESC_COLUMNS)
    parser.add_argument("--val-desc-columns", type=parse_csv_list, default=VAL_DESC_COLUMNS)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    model: ComposedGMM_MDN,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    train_loss: float,
    dev_loss: float,
    input_dim: int,
    output_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "model_config": {
                "input_dim": input_dim,
                "output_dim": output_dim,
                "num_components": args.num_components,
                "hidden_dims": tuple(int(x) for x in args.hidden_dims.split(",") if x.strip()),
                "dropout": args.dropout,
            },
            "dataset_name": args.dataset_name,
            "profile_csv_path": str(args.profile_csv_path),
            "sbert_root": str(args.sbert_root),
            "gmm_init_path": str(args.gmm_init_path),
            "pi_profile_ce_weight": args.pi_profile_ce_weight,
            "pretrain_only": True,
        },
        path,
    )


def run_epoch(
    model: ComposedGMM_MDN,
    loader: DataLoader,
    targets: torch.Tensor,
    device: torch.device,
    pi_profile_ce_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, int]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    for embedding, profile_indices in loader:
        embedding = embedding.float().to(device)
        profile_indices = profile_indices.long().to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            pi_logits, _mu, _sigma = model(embedding)
            ce_loss = pi_profile_cross_entropy(pi_logits, profile_indices, targets)
            loss = pi_profile_ce_weight * ce_loss
            if training:
                loss.backward()
                optimizer.step()

        batch_size = embedding.size(0)
        total_loss += float(ce_loss.detach().cpu()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1), total_items


def main() -> int:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.pi_profile_ce_weight <= 0:
        raise ValueError("--pi-profile-ce-weight must be positive for pretraining")

    if args.profile_csv_path is None:
        args.profile_csv_path = Path("data") / args.dataset_name / "profile_prompts.csv"
    if args.sbert_root is None:
        args.sbert_root = Path("exp/SBERT_embs") / args.dataset_name
    if args.gmm_init_path is None:
        args.gmm_init_path = Path(f"exp/GMMs/{args.num_components}_components_precomputed")
    if args.output is None:
        args.output = Path("exp/MDN_models") / f"{args.num_components}_components_{args.dataset_name}_pretrain.pt"

    set_seed(args.seed)
    device = resolve_device(args.device)
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    input_dim, output_dim = 384, 192

    train_dataset = ProfileEmbeddingDataset(args.profile_csv_path, args.sbert_root, args.train_desc_columns)
    val_dataset = ProfileEmbeddingDataset(args.profile_csv_path, args.sbert_root, args.val_desc_columns)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Dataset: {args.dataset_name}", flush=True)
    print(f"Profile CSV: {args.profile_csv_path}", flush=True)
    print(f"SBERT root: {args.sbert_root}", flush=True)
    print(f"ComposedGMM_MDN init GMM path: {args.gmm_init_path}", flush=True)
    print(f"Device: {device}", flush=True)

    model = ComposedGMM_MDN(
        input_dim=input_dim,
        output_dim=output_dim,
        num_components=args.num_components,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        gmm_init_path=str(args.gmm_init_path),
    ).to(device)
    targets = build_profile_component_targets(model, train_dataset.profiles, DEFAULT_PROFILE_FIELDS, device)
    if targets is None:
        raise ValueError("Could not build profile-component CE targets from GMM metadata")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_dev_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        train_loss, train_items = run_epoch(
            model,
            train_loader,
            targets,
            device,
            args.pi_profile_ce_weight,
            optimizer=optimizer,
        )
        with torch.no_grad():
            dev_loss, dev_items = run_epoch(
                model,
                val_loader,
                targets,
                device,
                args.pi_profile_ce_weight,
                optimizer=None,
            )

        elapsed = time.perf_counter() - start_time
        print(
            f"epoch={epoch:03d} train_pi_profile_ce={train_loss:.6f} "
            f"dev_pi_profile_ce={dev_loss:.6f} "
            f"train_items={train_items} dev_items={dev_items} time={elapsed:.1f}s",
            flush=True,
        )
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch
            save_checkpoint(args.output, model, optimizer, args, epoch, train_loss, dev_loss, input_dim, output_dim)
            print(f"Saved checkpoint to {args.output}", flush=True)

    print(f"Done. Best epoch={best_epoch} best_dev_pi_profile_ce={best_dev_loss:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
