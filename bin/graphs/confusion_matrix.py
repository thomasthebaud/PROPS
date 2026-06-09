#!/usr/bin/env python3
"""Build log-likelihood confusion matrices for generated category GMMs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns

from utils import (
    DEFAULT_TEST_DATASETS,
    FIELDS,
    find_generated_gmm,
    gmm_log_likelihood,
    load_gmm,
    load_gmm_metadata,
    load_xvectors,
    parse_csv_paths,
    sort_characteristic_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot category xvector-vs-generated-GMM log-likelihood matrices.")
    parser.add_argument("--category", required=True, choices=FIELDS)
    parser.add_argument("--test-csv-paths", type=parse_csv_paths, default=DEFAULT_TEST_DATASETS)
    parser.add_argument("--xvector-root", type=Path, default=Path("exp/xvectors/ecapa_tdnn"))
    parser.add_argument("--gmm-metadata-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-load-workers", type=int, default=16)
    parser.add_argument("--n-min", type=int, default=10, help="Drop labels with fewer than this many real xvectors from rows and columns.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_min < 1:
        raise ValueError("--n-min must be at least 1")

    utterances, xvectors = load_xvectors(
        args.test_csv_paths,
        args.xvector_root,
        num_workers=args.num_load_workers,
    )
    if len(xvectors) == 0:
        raise ValueError(f"No xvectors loaded from {args.test_csv_paths}")

    metadata = load_gmm_metadata(args.gmm_metadata_csv)
    category_values = utterances[args.category].astype(str).to_numpy()
    category_labels = sort_characteristic_labels(
        args.category,
        (label for label in set(category_values) if label != "unknown"),
    )
    label_counts = {
        label: int(np.sum(category_values == label))
        for label in category_labels
    }
    real_labels = [label for label, count in label_counts.items() if count >= args.n_min]
    skipped_labels = {label: count for label, count in label_counts.items() if count < args.n_min}
    for label, count in skipped_labels.items():
        print(
            f"Skipping {args.category}={label}: only {count} real xvectors, n_min={args.n_min}",
            flush=True,
        )

    gmm_rows = []
    gmm_labels = []
    for label in real_labels:
        gmm_row = find_generated_gmm(metadata, {args.category: label})
        if gmm_row is None:
            print(f"Skipping generated column {label}: no all-unknown GMM found", flush=True)
            continue
        gmm_labels.append(label)
        gmm_rows.append(gmm_row)

    real_labels = [label for label in real_labels if label in set(gmm_labels)]
    if not real_labels or not gmm_rows:
        raise ValueError(f"No matrix labels available for {args.category} with n_min={args.n_min}")

    matrix = np.empty((len(real_labels), len(gmm_labels)), dtype=np.float64)
    counts = np.empty(len(real_labels), dtype=np.int64)
    for row_idx, real_label in enumerate(real_labels):
        selected = xvectors[category_values == real_label]
        counts[row_idx] = len(selected)
        for col_idx, gmm_row in enumerate(gmm_rows):
            _pi_logits, pi, mu, sigma = load_gmm(Path(str(gmm_row["gmm_path"])))
            matrix[row_idx, col_idx] = float(np.mean(gmm_log_likelihood(selected, pi, mu, sigma)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.category}_confusion_matrix.csv"
    matrix_df = pd.DataFrame(matrix, index=real_labels, columns=gmm_labels)
    matrix_df.index.name = f"real_{args.category}"
    matrix_df.to_csv(csv_path)
    plot_matrix_df = matrix_df.copy()
    plot_matrix_df.index = [
        f"{label} (n={count})"
        for label, count in zip(real_labels, counts)
    ]

    counts_path = args.output_dir / f"{args.category}_confusion_counts.csv"
    pd.DataFrame({"real_label": real_labels, "num_xvectors": counts}).to_csv(counts_path, index=False)

    png_path = args.output_dir / f"{args.category}_confusion_matrix.png"
    sns.set_theme(style="white", context="notebook")
    width = max(6.0, 0.55 * len(gmm_labels) + 3.0)
    height = max(5.0, 0.45 * len(real_labels) + 2.5)
    fig, ax = plt.subplots(figsize=(width, height))
    center = float(np.mean(matrix))
    vmin = float(np.min(matrix))
    vmax = float(np.max(matrix))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax) if vmin < center < vmax else None
    sns.heatmap(
        plot_matrix_df,
        annot=True,
        fmt=".1f",
        cmap="PuOr",
        norm=norm,
        center=center if norm is None else None,
        cbar_kws={"label": f"Mean LL, centered at matrix mean ({center:.1f})"},
        ax=ax,
    )
    ax.set_xlabel(f"Generated {args.category} GMM")
    ax.set_ylabel(f"Real {args.category} xvectors")
    ax.set_title(f"Mean log-likelihood by {args.category}")
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {png_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
