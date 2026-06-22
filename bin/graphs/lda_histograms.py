#!/usr/bin/env python3
"""Plot LDA/PCA histograms separating real and generated xvectors for requested profiles."""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from utils import (
    DEFAULT_TEST_DATASETS,
    FIELDS,
    condition_mask,
    generated_gmm_matches,
    load_gmm_metadata,
    load_xvectors,
    normalize_value,
    parse_csv_paths,
    safe_file_stem,
    sample_gmm_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LDA/PCA histograms for real vs generated profile xvectors.")
    parser.add_argument("--test-csv-paths", type=parse_csv_paths, default=DEFAULT_TEST_DATASETS)
    parser.add_argument("--xvector-root", type=Path, default=Path("exp/xvectors/ecapa_tdnn"))
    parser.add_argument("--gmm-metadata-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pitch", default="unknown")
    parser.add_argument("--age", default="unknown")
    parser.add_argument("--gender", default="unknown")
    parser.add_argument("--speaking-rate", default="unknown")
    parser.add_argument("--speech-monotony", default="unknown")
    parser.add_argument("--accent", default="unknown")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--n-min", type=int, default=50, help="Skip plotting when fewer than this many real xvectors match the profile.")
    parser.add_argument("--num-load-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--pca", action="store_true", help="Use PCA instead of LDA for the one-dimensional projection.")
    return parser.parse_args()


def parse_field_values(value: str) -> list[str]:
    values = [normalize_value(item) for item in str(value).split(",") if str(item).strip()]
    return values or ["unknown"]


def requested_conditions(args: argparse.Namespace) -> list[dict[str, str]]:
    values_by_field = {
        "pitch": parse_field_values(args.pitch),
        "age": parse_field_values(args.age),
        "gender": parse_field_values(args.gender),
        "speaking_rate": parse_field_values(args.speaking_rate),
        "speech_monotony": parse_field_values(args.speech_monotony),
        "accent": parse_field_values(args.accent),
    }
    return [
        dict(zip(FIELDS, values))
        for values in product(*(values_by_field[field] for field in FIELDS))
    ]


def profile_label(condition: dict[str, str]) -> str:
    parts = [f"{field}={value}" for field, value in condition.items() if value != "unknown"]
    return ", ".join(parts) or "all_unknown"


def build_projection_data(
    condition: dict[str, str],
    utterances,
    real_xvectors: np.ndarray,
    metadata,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, object] | None:
    projection_name = "PCA" if args.pca else "LDA"
    gmm_rows = generated_gmm_matches(
        metadata,
        condition,
        strict_unknown_other_fields=False,
        unknown_is_wildcard=True,
    )
    if gmm_rows.empty:
        print(f"WARNING: skipping {projection_name} histogram for {condition}: no generated GMM found", flush=True)
        return None

    mask = condition_mask(utterances, condition, FIELDS)
    selected_real = real_xvectors[mask]
    if len(selected_real) < args.n_min:
        print(
            f"WARNING: skipping {projection_name} histogram for {condition}: "
            f"only {len(selected_real)} matching real xvectors, n_min={args.n_min}",
            flush=True,
        )
        return None

    generated, sampled_gmm_rows = sample_gmm_rows(gmm_rows, args.samples, rng)
    first_gmm_path = Path(str(sampled_gmm_rows.iloc[0]["gmm_path"]))
    x = np.vstack([selected_real, generated])
    y = np.concatenate([np.zeros(len(selected_real), dtype=int), np.ones(len(generated), dtype=int)])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    if args.pca:
        projector = PCA(n_components=1, random_state=args.seed)
        projected = projector.fit_transform(x_scaled)[:, 0]
    else:
        projector = LinearDiscriminantAnalysis(n_components=1)
        projected = projector.fit_transform(x_scaled, y)[:, 0]

    return {
        "condition": condition,
        "label": profile_label(condition),
        "gmm_path": first_gmm_path,
        "num_generated_gmms": len(gmm_rows),
        "real_projection": projected[: len(selected_real)],
        "generated_projection": projected[len(selected_real) :],
        "num_real": len(selected_real),
        "num_generated": len(generated),
    }


def draw_projection_subplot(ax, projection_data: dict[str, object], projection_name: str, args: argparse.Namespace) -> None:
    real_projection = projection_data["real_projection"]
    generated_projection = projection_data["generated_projection"]
    num_real = projection_data["num_real"]
    num_generated = projection_data["num_generated"]
    label = projection_data["label"]

    ax.hist(real_projection, bins=40, density=True, alpha=0.35, label=f"Real hist ({num_real})")
    ax.hist(generated_projection, bins=40, density=True, alpha=0.35, label=f"Generated hist ({num_generated})")
    sns.kdeplot(x=real_projection, ax=ax, linewidth=2.0, label="Real KDE")
    sns.kdeplot(x=generated_projection, ax=ax, linewidth=2.0, label="Generated KDE")
    ax.set_title(f"{projection_name} real vs generated: {label}")
    ax.set_xlabel("PC1" if args.pca else "LD1")
    ax.set_ylabel("Density")
    ax.legend()


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.n_min < 2:
        raise ValueError("--n-min must be at least 2")

    conditions = requested_conditions(args)
    if all(all(value == "unknown" for value in condition.values()) for condition in conditions):
        raise ValueError("At least one profile field must be known")

    rng = np.random.default_rng(args.seed)
    utterances, real_xvectors = load_xvectors(
        args.test_csv_paths,
        args.xvector_root,
        num_workers=args.num_load_workers,
    )
    if len(real_xvectors) == 0:
        raise ValueError(f"No xvectors loaded from {args.test_csv_paths}")

    metadata = load_gmm_metadata(args.gmm_metadata_csv)
    projection_name = "PCA" if args.pca else "LDA"
    projections = [
        projection
        for condition in conditions
        if (projection := build_projection_data(condition, utterances, real_xvectors, metadata, args, rng)) is not None
    ]
    if not projections:
        print(f"No {projection_name} histograms were written", flush=True)
        return 0

    labels = [projection["label"] for projection in projections]
    output_stem = safe_file_stem("__".join(labels))
    output_path = args.output_dir / f"{projection_name}_histogram_{output_stem}.png"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")
    fig_height = max(5.0, 4.2 * len(projections))
    fig, axes = plt.subplots(len(projections), 1, figsize=(8.5, fig_height), squeeze=False)
    for ax, projection_data in zip(axes[:, 0], projections):
        draw_projection_subplot(ax, projection_data, projection_name, args)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    for projection in projections:
        print(f"Profile: {projection['condition']}", flush=True)
        print(f"Real xvectors: {projection['num_real']}", flush=True)
        print(f"Generated xvectors: {projection['num_generated']}", flush=True)
        print(f"Generated GMMs: {projection['num_generated_gmms']} matching rows (first: {projection['gmm_path']})", flush=True)
    print(f"Wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
