#!/usr/bin/env python3
"""Train real-xvector characteristic classifiers and evaluate generated GMM samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils.class_weight import compute_sample_weight

from utils import (
    DEFAULT_TEST_DATASETS,
    FIELDS,
    generated_gmm_matches,
    load_gmm_metadata,
    load_xvectors,
    parse_csv_paths,
    sample_gmm_rows,
    sort_characteristic_labels,
    write_csv,
)


ORDINAL_CHARACTERISTICS = {"age", "speech_monotony", "pitch", "speaking_rate"}


def is_ordinal_characteristic(characteristic: str) -> bool:
    return characteristic in ORDINAL_CHARACTERISTICS


def label_rank_maps(characteristic: str, labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    ordered_labels = sort_characteristic_labels(characteristic, labels)
    label_to_rank = {label: rank for rank, label in enumerate(ordered_labels)}
    rank_to_label = {rank: label for label, rank in label_to_rank.items()}
    return label_to_rank, rank_to_label


def ordinal_targets(labels: np.ndarray, label_to_rank: dict[str, int]) -> np.ndarray:
    return np.asarray([label_to_rank[label] for label in labels], dtype=np.float64)


def ordinal_predict(classifier, xvectors: np.ndarray, rank_to_label: dict[int, str]) -> np.ndarray:
    numeric_predictions = np.asarray(classifier.predict(xvectors), dtype=np.float64)
    return ordinal_ranks_to_labels(numeric_predictions, rank_to_label)


def ordinal_ranks_to_labels(numeric_predictions: np.ndarray, rank_to_label: dict[int, str]) -> np.ndarray:
    max_rank = max(rank_to_label)
    ranks = np.rint(numeric_predictions).astype(int)
    ranks = np.clip(ranks, 0, max_rank)
    return np.asarray([rank_to_label[int(rank)] for rank in ranks], dtype=str)


def fused_ordinal_predict(
    svr_classifier,
    svc_classifier,
    xvectors: np.ndarray,
    label_to_rank: dict[str, int],
    rank_to_label: dict[int, str],
) -> np.ndarray:
    svr_ranks = np.asarray(svr_classifier.predict(xvectors), dtype=np.float64)
    svc_labels = svc_classifier.predict(xvectors)
    svc_ranks = np.asarray([label_to_rank[str(label)] for label in svc_labels], dtype=np.float64)
    fused_ranks = 0.5 * (svr_ranks + svc_ranks)
    return ordinal_ranks_to_labels(fused_ranks, rank_to_label)


def predict_ordinal_by_mode(
    mode: str,
    svr_classifier,
    svc_classifier,
    xvectors: np.ndarray,
    label_to_rank: dict[str, int],
    rank_to_label: dict[int, str],
) -> np.ndarray:
    if mode == "fusion":
        return fused_ordinal_predict(svr_classifier, svc_classifier, xvectors, label_to_rank, rank_to_label)
    if mode == "svr":
        return ordinal_predict(svr_classifier, xvectors, rank_to_label)
    if mode == "svc":
        return np.asarray(svc_classifier.predict(xvectors), dtype=str)
    raise ValueError(f"Unsupported ordinal classifier mode: {mode}")


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


def macro_accuracy_from_matrix(matrix: np.ndarray) -> float:
    row_totals = matrix.sum(axis=1)
    valid_rows = row_totals > 0
    if not np.any(valid_rows):
        return float("nan")
    per_label_accuracy = np.divide(
        np.diag(matrix)[valid_rows],
        row_totals[valid_rows],
        out=np.zeros(int(np.sum(valid_rows)), dtype=np.float64),
        where=row_totals[valid_rows] != 0,
    )
    return float(np.mean(per_label_accuracy))


def macro_accuracy(
    labels: list[str],
    true_labels: list[str] | np.ndarray,
    predicted_labels: list[str] | np.ndarray,
) -> float | None:
    if len(true_labels) == 0:
        return None
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    value = macro_accuracy_from_matrix(matrix)
    if np.isnan(value):
        return None
    return value


def label_correlation(
    characteristic: str,
    labels: list[str],
    true_labels: list[str] | np.ndarray,
    predicted_labels: list[str] | np.ndarray,
) -> float | None:
    if len(true_labels) < 2:
        return None
    label_to_rank, _ = label_rank_maps(characteristic, labels)
    try:
        true_ranks = np.asarray([label_to_rank[str(label)] for label in true_labels], dtype=np.float64)
        predicted_ranks = np.asarray([label_to_rank[str(label)] for label in predicted_labels], dtype=np.float64)
    except KeyError:
        return None
    if np.std(true_ranks) == 0.0 or np.std(predicted_ranks) == 0.0:
        return None
    correlation = float(np.corrcoef(true_ranks, predicted_ranks)[0, 1])
    if np.isnan(correlation):
        return None
    return correlation


def classification_metrics(
    characteristic: str,
    labels: list[str],
    true_labels: list[str] | np.ndarray,
    predicted_labels: list[str] | np.ndarray,
    correlation_labels: list[str] | None = None,
) -> dict[str, float | None]:
    if len(true_labels) == 0:
        return {
            "micro_accuracy": None,
            "macro_accuracy": None,
            "correlation": None,
            "micro_f1": None,
            "macro_f1": None,
        }
    return {
        "micro_accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_accuracy": macro_accuracy(labels, true_labels, predicted_labels),
        "correlation": label_correlation(characteristic, correlation_labels or labels, true_labels, predicted_labels),
        "micro_f1": float(f1_score(true_labels, predicted_labels, labels=labels, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, labels=labels, average="macro", zero_division=0)),
    }


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
    title_accuracy = macro_accuracy_from_matrix(matrix)
    title_suffix = "n/a" if np.isnan(title_accuracy) else f"{title_accuracy:.1%}"

    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Predicted {characteristic}")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} macro acc={title_suffix}")

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


def format_correlation(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "--"
    return f"{value:.3f}"


def display_characteristic_name(value: object) -> str:
    return str(value).replace("_", " ").title()


def write_latex_results(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '\\\\begin{tabular}{lrrrrrrrrrrr}',
        '\\\\hline',
        'Characteristic & Real micro acc. & Real macro acc. & Generated micro acc. & Generated macro acc. & Real corr. & Generated corr. & Real micro F1 & Real macro F1 & Generated micro F1 & Generated macro F1 & Number of classes \\\\',
        '\\\\hline',
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(display_characteristic_name(row["characteristic"])),
                    format_accuracy(row.get("real_test_micro_accuracy")),
                    format_accuracy(row.get("real_test_macro_accuracy")),
                    format_accuracy(row.get("generated_micro_accuracy")),
                    format_accuracy(row.get("generated_macro_accuracy")),
                    format_correlation(row.get("real_test_correlation")),
                    format_correlation(row.get("generated_correlation")),
                    format_accuracy(row.get("real_test_micro_f1")),
                    format_accuracy(row.get("real_test_macro_f1")),
                    format_accuracy(row.get("generated_micro_f1")),
                    format_accuracy(row.get("generated_macro_f1")),
                    str(row.get("num_classes", "")),
                ]
            )
            + ' \\\\'
        )
    lines.extend(['\\\\hline', '\\\\end{tabular}'])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated xvectors with SVM characteristic classifiers.")
    parser.add_argument("--classifier-train-csv-paths", type=parse_csv_paths, default=None)
    parser.add_argument("--test-csv-paths", type=parse_csv_paths, default=DEFAULT_TEST_DATASETS)
    parser.add_argument("--xvector-root", type=Path, default=Path("exp/xvectors/ecapa_tdnn"))
    parser.add_argument("--gmm-metadata-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--confusion-output-dir", type=Path, default=None)
    parser.add_argument("--characteristics", type=parse_csv_paths, default=["gender", "age", "accent"])
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument(
        "--ordinal-classifier-mode",
        choices=["fusion", "svc", "svr"],
        default="fusion",
        help="Classifier to use for ordinal characteristics: fused SVR+SVC, SVC only, or SVR only.",
    )
    parser.add_argument(
        "--svc-c",
        type=float,
        default=1.0,
        help="Regularization parameter C for SVC classifiers.",
    )
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
    if args.svc_c <= 0:
        raise ValueError("--svc-c must be positive")

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
    confusion_output_dir = args.confusion_output_dir or args.output_csv.parent

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
            f"{characteristic}: training classifier with {total_num_classifier_train_xvectors} real xvectors "
            f"across {len(known_labels)} labels: {label_counts}",
            flush=True,
        )
        sample_weights = compute_sample_weight(class_weight="balanced", y=train_labels)
        label_to_rank: dict[str, int] = {}
        rank_to_label: dict[int, str] = {}
        if is_ordinal_characteristic(characteristic):
            label_to_rank, rank_to_label = label_rank_maps(characteristic, known_labels)
            svr_classifier = None
            svc_classifier = None
            if args.ordinal_classifier_mode in {"fusion", "svr"}:
                svr_classifier = make_pipeline(
                    StandardScaler(),
                    SVR(kernel="rbf", gamma="scale", C=args.svc_c),
                )
                train_targets = ordinal_targets(train_labels, label_to_rank)
                svr_classifier.fit(train_xvectors[train_indices], train_targets, svr__sample_weight=sample_weights)
            if args.ordinal_classifier_mode in {"fusion", "svc"}:
                svc_classifier = make_pipeline(
                    StandardScaler(),
                    SVC(kernel="rbf", gamma="scale", C=args.svc_c),
                )
                svc_classifier.fit(train_xvectors[train_indices], train_labels, svc__sample_weight=sample_weights)
            train_predictions = predict_ordinal_by_mode(
                args.ordinal_classifier_mode,
                svr_classifier,
                svc_classifier,
                train_xvectors[train_indices],
                label_to_rank,
                rank_to_label,
            )
            classifier_type = f"ordinal {args.ordinal_classifier_mode.upper()}"
        else:
            classifier = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", gamma="scale", C=args.svc_c),
            )
            classifier.fit(train_xvectors[train_indices], train_labels, svc__sample_weight=sample_weights)
            train_predictions = classifier.predict(train_xvectors[train_indices])
            classifier_type = "SVM"

        train_accuracy = float(accuracy_score(train_labels, train_predictions))
        print(
            f"{characteristic}: trained {classifier_type} on {total_num_classifier_train_xvectors} xvectors, "
            f"labels={known_labels}, train_accuracy={train_accuracy:.4f}",
            flush=True,
        )

        test_mask = np.isin(test_labels, known_labels)
        real_test_accuracy = None
        real_test_metrics = {
            "micro_accuracy": None,
            "macro_accuracy": None,
            "correlation": None,
            "micro_f1": None,
            "macro_f1": None,
        }
        real_test_true_labels = test_labels[test_mask]
        real_test_predictions: np.ndarray = np.array([], dtype=str)
        real_test_num_xvectors = int(np.sum(test_mask))
        if real_test_num_xvectors:
            if is_ordinal_characteristic(characteristic):
                real_test_predictions = predict_ordinal_by_mode(
                    args.ordinal_classifier_mode,
                    svr_classifier,
                    svc_classifier,
                    test_xvectors[test_mask],
                    label_to_rank,
                    rank_to_label,
                )
            else:
                real_test_predictions = classifier.predict(test_xvectors[test_mask])
            real_test_metrics = classification_metrics(
                characteristic,
                known_labels,
                real_test_true_labels,
                real_test_predictions,
            )
            real_test_accuracy = real_test_metrics["macro_accuracy"]
            real_test_label_accuracies = {
                label: float(np.mean(real_test_predictions[real_test_true_labels == label] == label))
                for label in known_labels
                if np.any(real_test_true_labels == label)
            }
            print(
                f"{characteristic}: real test micro accuracy={real_test_metrics['micro_accuracy']:.4f} "
                f"macro accuracy={real_test_metrics['macro_accuracy']:.4f} "
                f"correlation={real_test_metrics['correlation'] if real_test_metrics['correlation'] is not None else 'n/a'} "
                f"micro_f1={real_test_metrics['micro_f1']:.4f} macro_f1={real_test_metrics['macro_f1']:.4f} "
                f"from per-label accuracies={real_test_label_accuracies}",
                flush=True,
            )
        else:
            print(f"{characteristic}: no real test xvectors with labels={known_labels}", flush=True)

        confusion_true_labels: list[str] = []
        confusion_predicted_labels: list[str] = []
        for label in known_labels:
            gmm_rows = generated_gmm_matches(
                metadata,
                {characteristic: label},
                strict_unknown_other_fields=False,
                unknown_is_wildcard=True,
            )
            if gmm_rows.empty:
                print(f"Skipping {characteristic}={label}: no matching generated GMM found", flush=True)
                continue

            generated_xvectors, sampled_gmm_rows = sample_gmm_rows(gmm_rows, args.samples, rng)
            first_gmm_row = sampled_gmm_rows.iloc[0]
            gmm_id = str(first_gmm_row.get("gmm_id", Path(str(first_gmm_row["gmm_path"])).stem))
            gmm_path = Path(str(first_gmm_row["gmm_path"]))
            print(
                f"{characteristic}={label}: testing SVM with {len(generated_xvectors)} generated xvectors "
                f"sampled across {len(gmm_rows)} matching GMMs (N={args.samples} total)",
                flush=True,
            )
            if is_ordinal_characteristic(characteristic):
                predictions = predict_ordinal_by_mode(
                    args.ordinal_classifier_mode,
                    svr_classifier,
                    svc_classifier,
                    generated_xvectors,
                    label_to_rank,
                    rank_to_label,
                )
            else:
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
                    "num_generated_gmms": len(gmm_rows),
                    "num_classifier_train_xvectors": label_counts[label],
                    "total_num_classifier_train_xvectors": total_num_classifier_train_xvectors,
                    "classifier_train_accuracy": train_accuracy,
                    "real_test_accuracy": real_test_accuracy if real_test_accuracy is not None else "",
                    "real_test_micro_accuracy": real_test_metrics["micro_accuracy"] if real_test_metrics["micro_accuracy"] is not None else "",
                    "real_test_macro_accuracy": real_test_metrics["macro_accuracy"] if real_test_metrics["macro_accuracy"] is not None else "",
                    "real_test_correlation": real_test_metrics["correlation"] if real_test_metrics["correlation"] is not None else "",
                    "real_test_micro_f1": real_test_metrics["micro_f1"] if real_test_metrics["micro_f1"] is not None else "",
                    "real_test_macro_f1": real_test_metrics["macro_f1"] if real_test_metrics["macro_f1"] is not None else "",
                    "real_test_num_xvectors": real_test_num_xvectors,
                    "gmm_id": gmm_id,
                    "gmm_path": str(gmm_path),
                }
            )

        if confusion_true_labels:
            generated_labels = [
                label
                for label in known_labels
                if any(true_label == label for true_label in confusion_true_labels)
            ]
            generated_metrics = classification_metrics(
                characteristic,
                generated_labels,
                confusion_true_labels,
                confusion_predicted_labels,
                correlation_labels=known_labels,
            )
            overall_accuracy = generated_metrics["macro_accuracy"]
            generated_label_accuracies = {
                label: float(
                    np.mean(
                        np.asarray(confusion_predicted_labels)[np.asarray(confusion_true_labels) == label] == label
                    )
                )
                for label in generated_labels
            }
            rows.append(
                {
                    "characteristic": characteristic,
                    "label": "overall",
                    "accuracy": overall_accuracy if overall_accuracy is not None else "",
                    "num_generated_xvectors": len(confusion_true_labels),
                    "num_generated_gmms": "",
                    "num_classifier_train_xvectors": "",
                    "total_num_classifier_train_xvectors": total_num_classifier_train_xvectors,
                    "classifier_train_accuracy": train_accuracy,
                    "real_test_accuracy": real_test_accuracy if real_test_accuracy is not None else "",
                    "real_test_micro_accuracy": real_test_metrics["micro_accuracy"] if real_test_metrics["micro_accuracy"] is not None else "",
                    "real_test_macro_accuracy": real_test_metrics["macro_accuracy"] if real_test_metrics["macro_accuracy"] is not None else "",
                    "real_test_correlation": real_test_metrics["correlation"] if real_test_metrics["correlation"] is not None else "",
                    "real_test_micro_f1": real_test_metrics["micro_f1"] if real_test_metrics["micro_f1"] is not None else "",
                    "real_test_macro_f1": real_test_metrics["macro_f1"] if real_test_metrics["macro_f1"] is not None else "",
                    "generated_micro_accuracy": generated_metrics["micro_accuracy"] if generated_metrics["micro_accuracy"] is not None else "",
                    "generated_macro_accuracy": generated_metrics["macro_accuracy"] if generated_metrics["macro_accuracy"] is not None else "",
                    "generated_correlation": generated_metrics["correlation"] if generated_metrics["correlation"] is not None else "",
                    "generated_micro_f1": generated_metrics["micro_f1"] if generated_metrics["micro_f1"] is not None else "",
                    "generated_macro_f1": generated_metrics["macro_f1"] if generated_metrics["macro_f1"] is not None else "",
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
                    "real_test_micro_accuracy": real_test_metrics["micro_accuracy"],
                    "real_test_macro_accuracy": real_test_metrics["macro_accuracy"],
                    "generated_micro_accuracy": generated_metrics["micro_accuracy"],
                    "generated_macro_accuracy": generated_metrics["macro_accuracy"],
                    "real_test_correlation": real_test_metrics["correlation"],
                    "generated_correlation": generated_metrics["correlation"],
                    "real_test_micro_f1": real_test_metrics["micro_f1"],
                    "real_test_macro_f1": real_test_metrics["macro_f1"],
                    "generated_micro_f1": generated_metrics["micro_f1"],
                    "generated_macro_f1": generated_metrics["macro_f1"],
                    "total_num_classifier_train_xvectors": total_num_classifier_train_xvectors,
                    "real_test_num_xvectors": real_test_num_xvectors,
                    "num_classes": len(known_labels),
                }
            )
            print(
                f"{characteristic}: overall generated micro accuracy={generated_metrics['micro_accuracy']:.4f} "
                f"macro accuracy={generated_metrics['macro_accuracy']:.4f} "
                f"correlation={generated_metrics['correlation'] if generated_metrics['correlation'] is not None else 'n/a'} "
                f"micro_f1={generated_metrics['micro_f1']:.4f} macro_f1={generated_metrics['macro_f1']:.4f} "
                f"from per-label accuracies={generated_label_accuracies}",
                flush=True,
            )

            confusion_path = confusion_output_dir / f"confusion_{characteristic}.png"
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

    if latex_rows:
        train_accuracies = [row["classifier_train_accuracy"] for row in latex_rows if row["classifier_train_accuracy"] is not None]
        average_train_accuracy = float(np.mean(train_accuracies)) if train_accuracies else None
        average_metrics = {
            key: (float(np.mean(values)) if values else None)
            for key in (
                "real_test_micro_accuracy",
                "real_test_macro_accuracy",
                "generated_micro_accuracy",
                "generated_macro_accuracy",
                "real_test_correlation",
                "generated_correlation",
                "real_test_micro_f1",
                "real_test_macro_f1",
                "generated_micro_f1",
                "generated_macro_f1",
            )
            for values in [[row[key] for row in latex_rows if row.get(key) is not None]]
        }
        average_real_test_accuracy = average_metrics["real_test_macro_accuracy"]
        average_generated_accuracy = average_metrics["generated_macro_accuracy"]
        rows.append(
            {
                "characteristic": "average",
                "label": "macro_over_characteristics",
                "accuracy": average_generated_accuracy,
                "num_generated_xvectors": "",
                "num_generated_gmms": "",
                "num_classifier_train_xvectors": "",
                "total_num_classifier_train_xvectors": "",
                "classifier_train_accuracy": average_train_accuracy,
                "real_test_accuracy": average_real_test_accuracy if average_real_test_accuracy is not None else "",
                "real_test_micro_accuracy": average_metrics["real_test_micro_accuracy"] if average_metrics["real_test_micro_accuracy"] is not None else "",
                "real_test_macro_accuracy": average_metrics["real_test_macro_accuracy"] if average_metrics["real_test_macro_accuracy"] is not None else "",
                "generated_micro_accuracy": average_metrics["generated_micro_accuracy"] if average_metrics["generated_micro_accuracy"] is not None else "",
                "generated_macro_accuracy": average_metrics["generated_macro_accuracy"] if average_metrics["generated_macro_accuracy"] is not None else "",
                "real_test_correlation": average_metrics["real_test_correlation"] if average_metrics["real_test_correlation"] is not None else "",
                "generated_correlation": average_metrics["generated_correlation"] if average_metrics["generated_correlation"] is not None else "",
                "real_test_micro_f1": average_metrics["real_test_micro_f1"] if average_metrics["real_test_micro_f1"] is not None else "",
                "real_test_macro_f1": average_metrics["real_test_macro_f1"] if average_metrics["real_test_macro_f1"] is not None else "",
                "generated_micro_f1": average_metrics["generated_micro_f1"] if average_metrics["generated_micro_f1"] is not None else "",
                "generated_macro_f1": average_metrics["generated_macro_f1"] if average_metrics["generated_macro_f1"] is not None else "",
                "real_test_num_xvectors": "",
                "gmm_id": "",
                "gmm_path": "",
            }
        )

    write_csv(
        args.output_csv,
        rows,
        [
            "characteristic",
            "label",
            "accuracy",
            "num_generated_xvectors",
            "num_generated_gmms",
            "num_classifier_train_xvectors",
            "total_num_classifier_train_xvectors",
            "classifier_train_accuracy",
            "real_test_accuracy",
            "real_test_micro_accuracy",
            "real_test_macro_accuracy",
            "generated_micro_accuracy",
            "generated_macro_accuracy",
            "real_test_correlation",
            "generated_correlation",
            "real_test_micro_f1",
            "real_test_macro_f1",
            "generated_micro_f1",
            "generated_macro_f1",
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
