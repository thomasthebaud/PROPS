#!/usr/bin/env python3
"""Train the MDN baseline from SBERT descriptions to ECAPA xvectors."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from model import LearnablePi_MDN, GaussianMDN, ComposedGMM_MDN
from my_dataset import ProfileDataset




def parse_csv_paths(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_dataset_fractions(value: str) -> dict[str, float]:
    fractions = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Expected DATASET=FRACTION entry, got {item!r}"
            )
        dataset, fraction_text = item.split("=", 1)
        dataset = dataset.strip()
        if not dataset:
            raise argparse.ArgumentTypeError(f"Missing dataset name in {item!r}")
        try:
            fraction = float(fraction_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid fraction for {dataset!r}: {fraction_text!r}"
            ) from exc
        if fraction > 1.0:
            fraction = fraction / 100.0
        if not 0.0 < fraction <= 1.0:
            raise argparse.ArgumentTypeError(
                f"Fraction for {dataset!r} must be in (0, 1] or a percent in (0, 100], got {fraction_text!r}"
            )
        fractions[dataset] = fraction
    return fractions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MDN speaker-profile baseline.")
    parser.add_argument(
        "--train-csv-paths",
        type=parse_csv_paths,
        help="Comma-separated dataset names under data/ for training.",
    )
    parser.add_argument(
        "--dev-csv-paths",
        type=parse_csv_paths,
        help="Comma-separated dataset names under data/ for dev validation.",
    )
    parser.add_argument(
        "--train-dataset-fractions",
        type=parse_dataset_fractions,
        default={},
        help=(
            "Optional comma-separated DATASET=FRACTION entries for training subsampling, "
            "for example CommonVoice_train=0.5,GigaSpeech_train=10."
        ),
    )
    parser.add_argument(
        "--profile-csv-path",
        help="CSV containing profile descriptions desc0..desc9.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exp/MDN_models/baseline.pt"),
        help="Path to save the trained checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-components", type=int, default=5)
    parser.add_argument("--hidden-dims", default="512,512,256")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--prop-dropout", type=float, default=0.2)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--mean-norm-weight", type=float, default=0.0)
    parser.add_argument("--pi-regularisation-weight", type=float, default=0.0)
    parser.add_argument("--variance-down-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model-name", type=str, default="LearnablePi_MDN", choices=["LearnablePi_MDN", "GaussianMDN", "ComposedGMM_MDN"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress/timing messages to diagnose stalls.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="When --verbose is set, print batch progress every N batches. Use 0 to log only phase boundaries.",
    )
    parser.add_argument(
        "--dev-split",
        type=float,
        default=0.2,
        help="If train and dev CSV paths are the same, split one dataset with this dev fraction.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def vprint(args: argparse.Namespace, message: str) -> None:
    if args.verbose:
        print(f"[verbose {time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    vprint(args, "Building training ProfileDataset")
    train_dataset = ProfileDataset(
        args.train_csv_paths,
        args.profile_csv_path,
        prop_dropout=args.prop_dropout,
        dataset_fractions=args.train_dataset_fractions,
        seed=args.seed,
    )
    vprint(args, f"Training ProfileDataset ready: len={len(train_dataset)}")

    vprint(args, "Building dev ProfileDataset")
    dev_dataset = ProfileDataset(
        args.dev_csv_paths,
        args.profile_csv_path,
        prop_dropout=args.prop_dropout,
        seed=args.seed,
    )
    vprint(args, f"Dev ProfileDataset ready: len={len(dev_dataset)}")

    pin_memory = torch.cuda.is_available()
    vprint(
        args,
        "Creating DataLoaders "
        f"batch_size={args.batch_size} num_workers={args.num_workers} pin_memory={pin_memory}",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    vprint(args, f"DataLoaders ready: train_batches={len(train_loader)} dev_batches={len(dev_loader)}")
    return train_loader, dev_loader


def to_device(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    xvector, embedding = batch
    return xvector.float().to(device), embedding.float().to(device)


def run_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    entropy_weight: float = 0.0,
    mean_norm_weight: float = 0.0,
    variance_down_weight: float = 0.0,
    pi_regularisation_weight: float = 0.0,
    grad_clip: float | None = None,
    verbose: bool = False,
    log_interval: int = 10,
    epoch: int | None = None,
    phase: str | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    phase_name = phase or ("train" if training else "dev")
    epoch_label = f" epoch={epoch}" if epoch is not None else ""
    model.train(training)
    totals = {
        "total_loss": 0.0,
        "nll": 0.0,
        "entropy_loss": 0.0,
        "entropy_term": 0.0,
        "mean_norm_loss": 0.0,
        "mean_norm_term": 0.0,
        "variance_down_loss": 0.0,
        "variance_down_term": 0.0,
        "pi_regularisation_loss":0.0
    }
    total_items = 0
    total_batches = len(loader)

    if verbose:
        print(
            f"[verbose {time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Starting {phase_name}{epoch_label}: batches={total_batches} training={training}",
            flush=True,
        )

    iterator = iter(loader)
    batch_index = 0
    while True:
        should_log_batch = verbose and (
            batch_index == 0 or (log_interval > 0 and batch_index % log_interval == 0)
        )
        if should_log_batch:
            print(
                f"[verbose {time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{phase_name}{epoch_label}: waiting for batch {batch_index + 1}/{total_batches}",
                flush=True,
            )
        load_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        load_seconds = time.perf_counter() - load_start
        if should_log_batch:
            print(
                f"[verbose {time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{phase_name}{epoch_label}: got batch {batch_index + 1}/{total_batches} "
                f"after {load_seconds:.3f}s",
                flush=True,
            )

        step_start = time.perf_counter()
        xvector, embedding = to_device(batch, device)
        if should_log_batch:
            print(
                f"[verbose {time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{phase_name}{epoch_label}: moved batch {batch_index + 1} to {device} "
                f"xvector_shape={tuple(xvector.shape)} embedding_shape={tuple(embedding.shape)}",
                flush=True,
            )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            nll = model.negative_log_likelihood(embedding, xvector)
            loss = nll

            entropy_loss = torch.zeros((), device=device)
            if entropy_weight:
                entropy_loss = model.mixture_entropy_loss(embedding)
                loss = loss + entropy_weight * entropy_loss

            mean_norm_loss = torch.zeros((), device=device)
            if mean_norm_weight:
                _, mu, _ = model(embedding)
                mean_norm_loss = torch.abs(1 - (mu ** 2).mean())
                loss = loss + mean_norm_weight * mean_norm_loss
            
            pi_regularisation_loss = torch.zeros((), device=device)
            if pi_regularisation_weight:
                pi, _, _ = model(embedding)
                pi_regularisation_loss = (torch.max(pi, dim=1)[0] - torch.tensor(1/model.K, device=device)).mean()
                loss = loss + pi_regularisation_weight * pi_regularisation_loss
            

            variance_down_loss = torch.zeros((), device=device)
            if variance_down_weight:
                variance_down_loss = model.variance_reduction_loss(embedding)
                loss = loss + variance_down_weight * variance_down_loss

            if training:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = xvector.size(0)
        total_items += batch_size
        batch_terms = {
            "total_loss": loss,
            "nll": nll,
            "entropy_loss": entropy_loss,
            "mean_norm_loss": mean_norm_loss,
            "variance_down_loss": variance_down_loss,
            "pi_regularisation_loss": pi_regularisation_loss
        }
        for name, value in batch_terms.items():
            totals[name] += float(value.detach().cpu()) * batch_size
        batch_index += 1

        if should_log_batch:
            step_seconds = time.perf_counter() - step_start
            running_loss = totals["total_loss"] / max(total_items, 1)
            running_nll = totals["nll"] / max(total_items, 1)
            print(
                f"[verbose {time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"{phase_name}{epoch_label}: finished batch {batch_index}/{total_batches} "
                f"step_time={step_seconds:.3f}s running_total={running_loss:.6f} "
                f"running_nll={running_nll:.6f}",
                flush=True,
            )

    if verbose:
        print(
            f"[verbose {time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Finished {phase_name}{epoch_label}: items={total_items}",
            flush=True,
        )
    return {name: value / max(total_items, 1) for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    model,
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
            "train_csv_paths": args.train_csv_paths,
            "dev_csv_paths": args.dev_csv_paths,
            "train_dataset_fractions": args.train_dataset_fractions,
            "profile_csv_path": args.profile_csv_path,
            "entropy_weight": args.entropy_weight,
            "mean_norm_weight": args.mean_norm_weight,
            "variance_down_weight": args.variance_down_weight,
            "pi_regularisation_weight": args.pi_regularisation_weight,
        },
        path,
    )
def load_precomputed_GMM(
        K: int,
    ) -> tuple[np.ndarray, np.ndarray]:
    """Load mu and sigma from male_only.npz and female_only.npz for a given K."""
    precomputed_dir = Path(f"exp/GMMs/{K//2}_components_precomputed")
    n_components = K // 2
    mu_init = []
    sigma_init = []

    for gender in ("male", "female"):
        path = precomputed_dir / f"{gender}_only.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing precomputed GMM file: {path}")

        with np.load(path) as data:
            mu = np.asarray(data["mu"], dtype=np.float64)
            sigma = np.asarray(data["sigma"], dtype=np.float64)

        if mu.ndim != 2 or sigma.ndim != 2:
            raise ValueError(f"Unexpected shape for {gender} GMM arrays: mu={mu.shape}, sigma={sigma.shape}")
        if mu.shape[0] != n_components or sigma.shape[0] != n_components:
            raise ValueError(
                f"{gender} GMM component count mismatch: expected {n_components}, "
                f"got mu.shape[0]={mu.shape[0]}, sigma.shape[0]={sigma.shape[0]}"
            )
        if mu.shape != sigma.shape:
            raise ValueError(f"{gender} mu/sigma shape mismatch: {mu.shape} vs {sigma.shape}")

        mu_init.append(mu)
        sigma_init.append(sigma)
    mu_init, sigma_init = np.concatenate(mu_init, axis=0), np.concatenate(sigma_init, axis=0)
    print(f"Loaded precomputed GMMs for {n_components} components: mu={mu_init.shape}, sigma={sigma_init.shape}")
    assert mu_init.shape[0]==K and sigma_init.shape[0]==K
    return torch.tensor(mu_init, dtype=torch.float32), torch.tensor(sigma_init, dtype=torch.float32)

def main() -> int:
    args = parse_args()
    vprint(args, f"Parsed args: {args}")
    vprint(args, f"Setting random seed: {args.seed}")
    set_seed(args.seed)
    vprint(args, f"Resolving device request: {args.device}")
    device = resolve_device(args.device)
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    vprint(args, f"Hidden dims: {hidden_dims}")

    train_loader, dev_loader = make_loaders(args)
    input_dim, output_dim = 384, 192
    print(f"Training datasets: {args.train_csv_paths}", flush=True)
    print(f"Dev datasets: {args.dev_csv_paths}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"MDN dims: input_dim={input_dim} output_dim={output_dim}", flush=True)

    vprint(args, f"Constructing {args.model_name}")
    if args.model_name == "LearnablePi_MDN":
        mu_init, sigma_init = load_precomputed_GMM(args.num_components)
        model = LearnablePi_MDN(
            input_dim=input_dim,
            output_dim=output_dim,
            num_components=args.num_components,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
            mu_init=mu_init,
            sigma_init=sigma_init
        )
    elif args.model_name == "GaussianMDN":
        model = GaussianMDN(
            input_dim=input_dim,
            output_dim=output_dim,
            num_components=args.num_components,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
        )
    elif args.model_name == "ComposedGMM_MDN":
        model = ComposedGMM_MDN(
            input_dim=input_dim,
            output_dim=output_dim,
            num_components=args.num_components,
            hidden_dims=hidden_dims,
            dropout=args.dropout,
            gmm_init_path=f"exp/GMMs/{args.num_components}_components_precomputed",
        )
    vprint(args, f"{args.model_name} constructed; moving model to device")
    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    vprint(args, f"{args.model_name} on {device}: parameters={param_count}")
    vprint(args, "Constructing AdamW optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    vprint(args, "Optimizer ready")

    best_dev_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            entropy_weight=args.entropy_weight,
            mean_norm_weight=args.mean_norm_weight,
            variance_down_weight=args.variance_down_weight,
            pi_regularisation_weight=args.pi_regularisation_weight,
            grad_clip=args.grad_clip,
            verbose=args.verbose,
            log_interval=args.log_interval,
            epoch=epoch,
            phase="train",
        )
        dev_metrics = run_epoch(
            model=model,
            loader=dev_loader,
            device=device,
            verbose=args.verbose,
            log_interval=args.log_interval,
            epoch=epoch,
            phase="dev",
        )
        train_loss = train_metrics["total_loss"]
        dev_loss = dev_metrics["nll"]
        print(
            f"epoch={epoch:03d} "
            f"train_total={train_metrics['total_loss']:.6f} "
            f"train_nll={train_metrics['nll']:.6f} "
            f"train_entropy={train_metrics['entropy_loss']:.6f} "
            f"train_mean_norm={train_metrics['mean_norm_loss']:.6f} "
            f"train_variance_down={train_metrics['variance_down_loss']:.6f} "
            f"train_pi_regularisation={train_metrics['pi_regularisation_loss']:.6f} "
            f"dev_nll={dev_loss:.6f}",
            flush=True,
        )

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch
            save_checkpoint(
                path=args.output,
                model=model,
                optimizer=optimizer,
                args=args,
                epoch=epoch,
                train_loss=train_loss,
                dev_loss=dev_loss,
                input_dim=input_dim,
                output_dim=output_dim,
            )
            print(f"Saved best checkpoint to {args.output}", flush=True)

    print(f"Done. Best epoch={best_epoch} best_dev_nll={best_dev_loss:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
