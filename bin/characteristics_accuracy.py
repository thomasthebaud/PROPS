#!/usr/bin/env python3
"""Train real-xvector characteristic classifiers and evaluate generated GMM samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from utils import (
    DEFAULT_TEST_DATASETS,
    FIELDS,
    find_generated_gmm,
    load_gmm_metadata,
    load_xvectors,
    parse_csv_paths,
    sample_gmm,
    sort_characteristic_labels,
    write_csv,
)


def save_confusion_matrix(
    output_path: Path,
    characteristic: str,
    labels: list[str],
    true_labels: list[str],
    predicted_labels: list[str],
) -> None:
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros(matrix.shape, dtype=np.float64),
        where=row_totals != 0,
    )

    width = max(6.0, 0.75 * len(labels) + 2.5)
    height = max(5.0, 0.65 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Row-normalized fraction", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Predicted {characteristic}")
    ax.set_ylabel(f"Generated {characteristic}")
    ax.set_title(f"{characteristic} classifier confusion matrix")

    threshold = 0.5
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = normalized[row_idx, col_idx]
            color = "white" if value > threshold else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{matrix[row_idx, col_idx]}\n{value:.1%}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated xvectors with SVM characteristic classifiers.")
    parser.add_argument("--test-csv-paths", type=parse_csv_paths, default=DEFAULT_TEST_DATASETS)
    parser.add_argument("--xvector-root", type=Path, default=Path("exp/xvectors/ecapa_tdnn"))
    parser.add_argument("--gmm-metadata-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--characteristics", type=parse_csv_paths, default=["gender", "age", "accent"])
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--num-load-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    unknown_characteristics = [item for item in args.characteristics if item not in FIELDS]
    if unknown_characteristics:
        raise ValueError(f"Unknown characteristics: {unknown_characteristics}; expected subset of {FIELDS}")

    rng = np.random.default_rng(args.seed)
    utterances, xvectors = load_xvectors(
        args.test_csv_paths,
        args.xvector_root,
        num_workers=args.num_load_workers,
    )
    if len(xvectors) == 0:
        raise ValueError(f"No xvectors loaded from {args.test_csv_paths}")

    metadata = load_gmm_metadata(args.gmm_metadata_csv)
    rows: list[dict[str, object]] = []
    output_dir = args.output_csv.parent

    for characteristic in args.characteristics:
        labels = utterances[characteristic].astype(str).to_numpy()
        train_mask = labels != "unknown"
        known_labels = sort_characteristic_labels(characteristic, set(labels[train_mask]))
        if len(known_labels) < 2:
            print(f"Skipping {characteristic}: need at least two known labels, found {known_labels}", flush=True)
            continue

        train_labels = labels[train_mask]
        sample_weights = compute_sample_weight(class_weight="balanced", y=train_labels)
        classifier = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", gamma="scale"),
        )
        classifier.fit(xvectors[train_mask], train_labels, svc__sample_weight=sample_weights)
        train_accuracy = float(accuracy_score(train_labels, classifier.predict(xvectors[train_mask])))
        print(
            f"{characteristic}: trained SVM on {int(train_mask.sum())} xvectors, "
            f"labels={known_labels}, train_accuracy={train_accuracy:.4f}",
            flush=True,
        )

        confusion_true_labels: list[str] = []
        confusion_predicted_labels: list[str] = []
        for label in known_labels:
            gmm_row = find_generated_gmm(metadata, {characteristic: label})
            if gmm_row is None:
                print(f"Skipping {characteristic}={label}: no all-unknown generated GMM found", flush=True)
                continue

            gmm_path = Path(str(gmm_row["gmm_path"]))
            generated_xvectors = sample_gmm(gmm_path, args.samples, rng)
            predictions = classifier.predict(generated_xvectors)
            accuracy = float(np.mean(predictions == label))
            confusion_true_labels.extend([label] * len(predictions))
            confusion_predicted_labels.extend(predictions.astype(str).tolist())
            rows.append(
                {
                    "characteristic": characteristic,
                    "label": label,
                    "accuracy": accuracy,
                    "num_generated_xvectors": len(generated_xvectors),
                    "num_classifier_train_xvectors": int(train_mask.sum()),
                    "classifier_train_accuracy": train_accuracy,
                    "gmm_path": str(gmm_path),
                }
            )

        if confusion_true_labels:
            confusion_path = output_dir / f"confusion_{characteristic}.png"
            save_confusion_matrix(
                confusion_path,
                characteristic,
                known_labels,
                confusion_true_labels,
                confusion_predicted_labels,
            )
            print(f"Wrote confusion matrix to {confusion_path}", flush=True)

    write_csv(
        args.output_csv,
        rows,
        [
            "characteristic",
            "label",
            "accuracy",
            "num_generated_xvectors",
            "num_classifier_train_xvectors",
            "classifier_train_accuracy",
            "gmm_path",
        ],
    )
    print(f"Wrote accuracies to {args.output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
