#!/usr/bin/env python3
"""Summarize generated metadata CSVs by dataset and split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

SPLITS = ["train", "dev", "test"]
DEFAULT_DATASETS = ["CommonVoice", "GigaSpeech", "MLS-en", "Emilia-en", "Capspeech"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print segment/hour summary tables for generated metadata CSVs.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset directories to summarize. Defaults to known generated datasets plus any directory with split CSVs.")
    parser.add_argument("--latex-output", type=Path, default=Path("data/latex_table.txt"))
    return parser.parse_args()


def discover_datasets(data_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    discovered = {
        path.name
        for path in data_root.iterdir()
        if path.is_dir() and any((path / f"{split}.csv").exists() for split in SPLITS)
    }
    ordered = [name for name in DEFAULT_DATASETS if name in discovered]
    ordered.extend(sorted(discovered - set(ordered)))
    return ordered


def read_split(path: Path) -> tuple[int, float]:
    if not path.exists() or path.stat().st_size == 0:
        return 0, 0.0

    try:
        header = pd.read_csv(path, nrows=0)
    except EmptyDataError:
        return 0, 0.0

    if "duration" not in header.columns:
        try:
            segments = sum(len(chunk) for chunk in pd.read_csv(path, chunksize=200_000))
        except EmptyDataError:
            return 0, 0.0
        return segments, 0.0

    segments = 0
    duration_seconds = 0.0
    try:
        for chunk in pd.read_csv(path, usecols=["duration"], chunksize=200_000):
            segments += len(chunk)
            duration_seconds += float(pd.to_numeric(chunk["duration"], errors="coerce").fillna(0.0).sum())
    except EmptyDataError:
        return 0, 0.0
    return segments, duration_seconds / 3600.0


def build_rows(data_root: Path, datasets: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        row: dict[str, object] = {"dataset": dataset}
        total_segments = 0
        total_hours = 0.0
        for split in SPLITS:
            segments, hours = read_split(data_root / dataset / f"{split}.csv")
            row[f"{split}_hours"] = hours
            row[f"{split}_segments"] = segments
            total_segments += segments
            total_hours += hours
        row["total_hours"] = total_hours
        row["total_segments"] = total_segments
        rows.append(row)
    return rows


def add_grand_total(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    total_row: dict[str, object] = {"dataset": "TOTAL"}
    for split in [*SPLITS, "total"]:
        total_row[f"{split}_hours"] = sum(float(row[f"{split}_hours"]) for row in rows)
        total_row[f"{split}_segments"] = sum(int(row[f"{split}_segments"]) for row in rows)
    return [*rows, total_row]


def print_table(rows: list[dict[str, object]]) -> None:
    columns = [
        ("Dataset", "dataset", "text"),
        ("train hours", "train_hours", "hours"),
        ("train segments", "train_segments", "segments"),
        ("dev hours", "dev_hours", "hours"),
        ("dev segments", "dev_segments", "segments"),
        ("test hours", "test_hours", "hours"),
        ("test segments", "test_segments", "segments"),
        ("total hours", "total_hours", "hours"),
        ("total segments", "total_segments", "segments"),
    ]
    widths = []
    for label, key, kind in columns:
        values = [label]
        for row in rows:
            if kind == "hours":
                values.append(f"{float(row[key]):,.2f}")
            elif kind == "segments":
                values.append(f"{int(row[key]):,}")
            else:
                values.append(str(row[key]))
        widths.append(max(len(value) for value in values))

    header = "  ".join(label.ljust(width) for (label, _key, _kind), width in zip(columns, widths))
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for (_label, key, kind), width in zip(columns, widths):
            if kind == "hours":
                cells.append(f"{float(row[key]):>{width},.2f}")
            elif kind == "segments":
                cells.append(f"{int(row[key]):>{width},}")
            else:
                cells.append(str(row[key]).ljust(width))
        print("  ".join(cells))


def latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def write_latex(rows: list[dict[str, object]], output_path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Dataset & \multicolumn{2}{c}{train} & \multicolumn{2}{c}{dev} & \multicolumn{2}{c}{test} & \multicolumn{2}{c}{total} \\",
        r"Name & hours & segments & hours & segments & hours & segments & hours & segments \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row['dataset'])} & "
            f"{float(row['train_hours']):,.2f} & {int(row['train_segments']):,} & "
            f"{float(row['dev_hours']):,.2f} & {int(row['dev_segments']):,} & "
            f"{float(row['test_hours']):,.2f} & {int(row['test_segments']):,} & "
            f"{float(row['total_hours']):,.2f} & {int(row['total_segments']):,} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote LaTeX table to {output_path}")


def main() -> int:
    args = parse_args()
    datasets = discover_datasets(args.data_root, args.datasets)
    if not datasets:
        raise SystemExit(f"No generated dataset split CSVs found under {args.data_root}")
    rows = add_grand_total(build_rows(args.data_root, datasets))
    print_table(rows)
    write_latex(rows, args.latex_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
