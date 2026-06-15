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


def confusion_matrix_values(
    labels: list[str],
    true_labels: list[str] | np.ndarray,
    predicted_labels: list[str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros(matrix.shape, dtype=np.float64),
        where=row_totals != 0,
    )
    return matrix, normalized


def matrix_accuracy(matrix: np.ndarray) -> float:
    total = int(matrix.sum())
    if total == 0:
        return float("nan")
    return float(np.trace(matrix) / total)


def labels_with_counts(labels: list[str], counts: np.ndarray) -> list[str]:
    return [f"{label} (n={int(count)})" for label, count in zip(labels, counts)]


def draw_confusion_panel(
    ax,
    labels: list[str],
    matrix: np.ndarray,
    normalized: np.ndarray,
    characteristic: str,
    title: str,
    ylabel: str,
):
    row_counts = matrix.sum(axis=1)
    col_counts = matrix.sum(axis=0)
    title_accuracy = matrix_accuracy(matrix)
    title_suffix = "n/a" if np.isnan(title_accuracy) else f"{title_accuracy:.1%}"

    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Predicted {characteristic}")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} acc={title_suffix}")

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
    return image


def save_confusion_matrix(
    output_path: Path,
    characteristic: str,
    labels: list[str],
    generated_true_labels: list[str],
    generated_predicted_labels: list[str],
    real_true_labels: list[str] | np.ndarray,
    real_predicted_labels: list[str] | np.ndarray,
) -> None:
    labels = sort_characteristic_labels(characteristic, labels)
    generated_matrix, generated_normalized = confusion_matrix_values(
        labels,
        generated_true_labels,
        generated_predicted_labels,
    )
    real_matrix, real_normalized = confusion_matrix_values(
        labels,
        real_true_labels,
        real_predicted_labels,
    )

    width = max(10.0, 1.45 * len(labels) + 5.0)
    height = max(5.0, 0.65 * len(labels) + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharey=True)
    image = draw_confusion_panel(
        axes[0],
        labels,
        generated_matrix,
        generated_normalized,
        characteristic,
        "Generated samples",
        f"Generated {characteristic}",
    )
    draw_confusion_panel(
        axes[1],
        labels,
        real_matrix,
        real_normalized,
        characteristic,
        "Real test xvectors",
        f"Real {characteristic}",
    )
    cbar = fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Row-normalized fraction", rotation=-90, va="bottom")
    fig.suptitle(f"{characteristic} classifier confusion matrices")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def format_accuracy(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "--"
    return f"{100.0 * value:.1f}"


def write_latex_results(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Characteristic & Train acc. & Real test acc. & Generated acc. & Train xvecs & Real test xvecs \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(row["characteristic"]),
                    format_accuracy(row["classifier_train_accuracy"]),
                    format_accuracy(row["real_test_accuracy"]),
                    format_accuracy(row["generated_accuracy"]),
                    str(row["total_num_classifier_train_xvectors"]),
                    str(row["real_test_num_xvectors"]),
                ]
            )
            + r" \\",
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated xvectors with SVM characteristic classifiers.")
    parser.add_argument("--classifier-train-csv-paths", type=parse_csv_paths, default=None)
    parser.add_argument("--test-csv-paths", type=parse_csv_paths, default=DEFAULT_TEST_DATASETS)
    parser.add_argument("--xvector-root", type=Path, default=Path("exp/xvectors/ecapa_tdnn"))
    parser.add_argument("--gmm-metadata-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--characteristics", type=parse_csv_paths, default=["gender", "age", "accent"])
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument(
        "--max-train-vectors",
        type=int,
        default=10000,
        help="Maximum real xvectors per characteristic label to use when training each SVM.",
    )
    parser.add_argument(
        "--min-label-xvectors",
        type=int,
        default=25,
        help="Drop characteristic labels with fewer than this many classifier-train xvectors.",
    )
    parser.add_argument("--num-load-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.max_train_vectors <= 0:
        raise ValueError("--max-train-vectors must be positive")
    if args.min_label_xvectors <= 0:
        raise ValueError("--min-label-xvectors must be positive")

    unknown_characteristics = [item for item in args.characteristics if item not in FIELDS]
    if unknown_characteristics:
        raise ValueError(f"Unknown characteristics: {unknown_characteristics}; expected subset of {FIELDS}")

    rng = np.random.default_rng(args.seed)
    classifier_train_csv_paths = args.classifier_train_csv_paths or args.test_csv_paths
    train_utterances, train_xvectors = load_xvectors(
        classifier_train_csv_paths,
        args.xvector_root,
        num_workers=args.num_load_workers,
    )
    if len(train_xvectors) == 0:
        raise ValueError(f"No classifier train xvectors loaded from {classifier_train_csv_paths}")

    test_utterances, test_xvectors = load_xvectors(
        args.test_csv_paths,
        args.xvector_root,
        num_workers=args.num_load_workers,
    )
    if len(test_xvectors) == 0:
        raise ValueError(f"No test xvectors loaded from {args.test_csv_paths}")

    metadata = load_gmm_metadata(args.gmm_metadata_csv)
    rows: list[dict[str, object]] = []
    latex_rows: list[dict[str, object]] = []
    output_dir = args.output_csv.parent

    for characteristic in args.characteristics:
        labels = train_utterances[characteristic].astype(str).to_numpy()
        test_labels = test_utterances[characteristic].astype(str).to_numpy()
        train_mask = labels != "unknown"
        known_labels = sort_characteristic_labels(characteristic, set(labels[train_mask]))
        if characteristic == "age" and "child" in known_labels:
            print(f"Skipping {characteristic}=child: excluded from age classifier", flush=True)
            known_labels = [label for label in known_labels if label != "child"]

        raw_label_counts = {label: int(np.sum(labels == label)) for label in known_labels}
        skipped_small_labels = {
            label: count
            for label, count in raw_label_counts.items()
            if count < args.min_label_xvectors
        }
        for label, count in skipped_small_labels.items():
            print(
                f"Skipping {characteristic}={label}: only {count} classifier-train xvectors, "
                f"min_label_xvectors={args.min_label_xvectors}",
                flush=True,
            )
        known_labels = [
            label
            for label in known_labels
            if raw_label_counts[label] >= args.min_label_xvectors
        ]
        if len(known_labels) < 2:
            print(f"Skipping {characteristic}: need at least two known labels, found {known_labels}", flush=True)
            continue

        train_indices_by_label: dict[str, np.ndarray] = {}
        capped_train_indices: list[np.ndarray] = []
        for label in known_labels:
            label_indices = np.flatnonzero(labels == label)
            if len(label_indices) > args.max_train_vectors:
                label_indices = np.sort(
                    rng.choice(label_indices, size=args.max_train_vectors, replace=False)
                )
            train_indices_by_label[label] = label_indices
            capped_train_indices.append(label_indices)

        train_indices = np.concatenate(capped_train_indices)
        train_labels = labels[train_indices]
        label_counts = {label: int(len(indices)) for label, indices in train_indices_by_label.items()}
        total_num_classifier_train_xvectors = int(len(train_indices))
        print(
            f"{characteristic}: training SVM with {total_num_classifier_train_xvectors} real xvectors "
            f"across {len(known_labels)} labels: {label_counts}",
            flush=True,
        )
        sample_weights = compute_sample_weight(class_weight="balanced", y=train_labels)
        classifier = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", gamma="scale"),
        )
        classifier.fit(train_xvectors[train_indices], train_labels, svc__sample_weight=sample_weights)
        train_accuracy = float(accuracy_score(train_labels, classifier.predict(train_xvectors[train_indices])))
        print(
            f"{characteristic}: trained SVM on {total_num_classifier_train_xvectors} xvectors, "
            f"labels={known_labels}, train_accuracy={train_accuracy:.4f}",
            flush=True,
        )

        test_mask = np.isin(test_labels, known_labels)
        real_test_accuracy = None
        real_test_true_labels = test_labels[test_mask]
        real_test_predictions: np.ndarray = np.array([], dtype=str)
        real_test_num_xvectors = int(np.sum(test_mask))
        if real_test_num_xvectors:
            real_test_predictions = classifier.predict(test_xvectors[test_mask])
            real_test_accuracy = float(accuracy_score(real_test_true_labels, real_test_predictions))
            print(
                f"{characteristic}: real test accuracy={real_test_accuracy:.4f} "
                f"({int(np.sum(real_test_predictions == real_test_true_labels))}/{real_test_num_xvectors} correct)",
                flush=True,
            )
        else:
            print(f"{characteristic}: no real test xvectors with labels={known_labels}", flush=True)

        confusion_true_labels: list[str] = []
        confusion_predicted_labels: list[str] = []
        for label in known_labels:
            gmm_row = find_generated_gmm(metadata, {characteristic: label})
            if gmm_row is None:
                print(f"Skipping {characteristic}={label}: no all-unknown generated GMM found", flush=True)
                continue

            gmm_id = str(gmm_row.get("gmm_id", Path(str(gmm_row["gmm_path"])).stem))
            gmm_path = Path(str(gmm_row["gmm_path"]))
            print(
                f"{characteristic}={label}: testing SVM with {args.samples} generated xvectors "
                f"sampled from {gmm_id}",
                flush=True,
            )
            generated_xvectors = sample_gmm(gmm_path, args.samples, rng)
            predictions = classifier.predict(generated_xvectors)
            accuracy = float(np.mean(predictions == label))
            print(
                f"{characteristic}={label}: testing accuracy={accuracy:.4f} "
                f"({int(np.sum(predictions == label))}/{len(predictions)} correct)",
                flush=True,
            )
            confusion_true_labels.extend([label] * len(predictions))
            confusion_predicted_labels.extend(predictions.astype(str).tolist())
            rows.append(
                {
                    "characteristic": characteristic,
                    "label": label,
                    "accuracy": accuracy,
                    "num_generated_xvectors": len(generated_xvectors),
                    "num_classifier_train_xvectors": label_counts[label],
                    "total_num_classifier_train_xvectors": total_num_classifier_train_xvectors,
                    "classifier_train_accuracy": train_accuracy,
                    "real_test_accuracy": real_test_accuracy if real_test_accuracy is not None else "",
                    "real_test_num_xvectors": real_test_num_xvectors,
                    "gmm_id": gmm_id,
                    "gmm_path": str(gmm_path),
                }
            )

        if confusion_true_labels:
            overall_accuracy = float(
                accuracy_score(confusion_true_labels, confusion_predicted_labels)
            )
            rows.append(
                {
                    "characteristic": characteristic,
                    "label": "overall",
                    "accuracy": overall_accuracy,
                    "num_generated_xvectors": len(confusion_true_labels),
                    "num_classifier_train_xvectors": "",
                    "total_num_classifier_train_xvectors": total_num_classifier_train_xvectors,
                    "classifier_train_accuracy": train_accuracy,
                    "real_test_accuracy": real_test_accuracy if real_test_accuracy is not None else "",
                    "real_test_num_xvectors": real_test_num_xvectors,
                    "gmm_id": "",
                    "gmm_path": "",
                }
            )
            latex_rows.append(
                {
                    "characteristic": characteristic,
                    "classifier_train_accuracy": train_accuracy,
                    "real_test_accuracy": real_test_accuracy,
                    "generated_accuracy": overall_accuracy,
                    "total_num_classifier_train_xvectors": total_num_classifier_train_xvectors,
                    "real_test_num_xvectors": real_test_num_xvectors,
                }
            )
            print(
                f"{characteristic}: overall generated accuracy={overall_accuracy:.4f} "
                f"({int(np.sum(np.asarray(confusion_true_labels) == np.asarray(confusion_predicted_labels)))}/"
                f"{len(confusion_true_labels)} correct)",
                flush=True,
            )

            confusion_path = output_dir / f"confusion_{characteristic}.png"
            save_confusion_matrix(
                confusion_path,
                characteristic,
                known_labels,
                confusion_true_labels,
                confusion_predicted_labels,
                real_test_true_labels,
                real_test_predictions,
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
            "total_num_classifier_train_xvectors",
            "classifier_train_accuracy",
            "real_test_accuracy",
            "real_test_num_xvectors",
            "gmm_id",
            "gmm_path",
        ],
    )
    results_tex = args.output_csv.parent / "results.tex"
    write_latex_results(results_tex, latex_rows)
    print(f"Wrote accuracies to {args.output_csv}", flush=True)
    print(f"Wrote LaTeX results to {results_tex}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
