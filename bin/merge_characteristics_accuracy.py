#!/usr/bin/env python3
"""Merge per-characteristic accuracy CSVs and rebuild the combined LaTeX table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any



FIELDNAMES = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-characteristic accuracy CSVs.")
    parser.add_argument("--input-csvs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--results-tex", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in FIELDNAMES} for row in rows])
    tmp_path.replace(path)


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


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


def format_accuracy(value: Any) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "--"
    return f"{100.0 * parsed:.1f}"


def format_correlation(value: Any) -> str:
    parsed = parse_float(value)
    if parsed is None:
        return "--"
    return f"{parsed:.3f}"


def metric_value(row: dict[str, Any], primary: str, fallback: str | None = None) -> Any:
    value = row.get(primary)
    if parse_float(value) is not None or fallback is None:
        return value
    return row.get(fallback)


def write_latex(path: Path, overall_rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\hline",
        "Characteristic & Real micro acc. & Real macro acc. & Generated micro acc. & Generated macro acc. & Real corr. & Generated corr. & Real micro F1 & Real macro F1 & Generated micro F1 & Generated macro F1 \\\\",
        r"\hline",
    ]
    for row in overall_rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(row["characteristic"]),
                    format_accuracy(row.get("real_test_micro_accuracy")),
                    format_accuracy(metric_value(row, "real_test_macro_accuracy", "real_test_accuracy")),
                    format_accuracy(row.get("generated_micro_accuracy")),
                    format_accuracy(metric_value(row, "generated_macro_accuracy", "accuracy")),
                    format_correlation(row.get("real_test_correlation")),
                    format_correlation(row.get("generated_correlation")),
                    format_accuracy(row.get("real_test_micro_f1")),
                    format_accuracy(row.get("real_test_macro_f1")),
                    format_accuracy(row.get("generated_micro_f1")),
                    format_accuracy(row.get("generated_macro_f1")),
                ]
            )
            + r" \\",
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    merged_rows: list[dict[str, Any]] = []
    for path in args.input_csvs:
        if not path.exists():
            raise FileNotFoundError(f"Missing per-characteristic CSV: {path}")
        for row in read_rows(path):
            if row.get("characteristic") == "average":
                continue
            merged_rows.append(row)

    overall_rows = [row for row in merged_rows if row.get("label") == "overall"]
    if overall_rows:
        average_train = mean_or_none([value for row in overall_rows if (value := parse_float(row.get("classifier_train_accuracy"))) is not None])
        average_metrics = {
            key: mean_or_none([value for row in overall_rows if (value := parse_float(row.get(key))) is not None])
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
        }
        average_generated = average_metrics["generated_macro_accuracy"]
        if average_generated is None:
            average_generated = mean_or_none([value for row in overall_rows if (value := parse_float(row.get("accuracy"))) is not None])
        average_real = average_metrics["real_test_macro_accuracy"]
        if average_real is None:
            average_real = mean_or_none([value for row in overall_rows if (value := parse_float(row.get("real_test_accuracy"))) is not None])
        average_row = {
            "characteristic": "average",
            "label": "macro_over_characteristics",
            "accuracy": average_generated if average_generated is not None else "",
            "num_generated_xvectors": "",
            "num_generated_gmms": "",
            "num_classifier_train_xvectors": "",
            "total_num_classifier_train_xvectors": "",
            "classifier_train_accuracy": average_train if average_train is not None else "",
            "real_test_accuracy": average_real if average_real is not None else "",
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
        merged_rows.append(average_row)
        overall_rows = [*overall_rows, average_row]

    write_rows(args.output_csv, merged_rows)
    write_latex(args.results_tex, overall_rows)
    print(f"Merged {len(args.input_csvs)} CSVs into {args.output_csv}", flush=True)
    print(f"Wrote {args.results_tex}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
