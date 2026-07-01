#!/usr/bin/env python3
"""Write per-metric C comparison CSVs from characteristic results.tex tables."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def split_tex_row(line: str) -> list[str]:
    line = line.strip()
    line = re.sub(r"\\+\s*$", "", line).strip()
    return [cell.strip().replace(r"\_", "_") for cell in line.split("&")]


def metric_slug(metric_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", metric_name.lower()).strip("_")
    return slug or "metric"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create C-by-characteristic CSV tables for each metric in results.tex files."
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        type=Path,
        help="Directory containing C=<value>/results.tex subdirectories.",
    )
    parser.add_argument(
        "--c-values",
        required=True,
        nargs="+",
        help="C values to read, in the desired output column order.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir

    metric_names: list[str] | None = None
    metric_values: dict[str, dict[str, dict[str, str]]] = {}
    characteristics: list[str] = []
    seen_characteristics: set[str] = set()
    available_c_values: list[str] = []

    for c_value in args.c_values:
        table_path = base_dir / f"C={c_value}" / "results.tex"
        if not table_path.exists():
            print(f"Skipping missing {table_path}", file=sys.stderr)
            continue

        available_c_values.append(c_value)
        with table_path.open(newline="") as table_file:
            for raw_line in table_file:
                line = raw_line.strip()
                if not line or line.startswith(r"\hline") or line.startswith(r"\begin") or line.startswith(r"\end"):
                    continue
                cells = split_tex_row(line)
                if cells[0] == "Characteristic":
                    current_metrics = cells[1:]
                    if metric_names is None:
                        metric_names = current_metrics
                    elif current_metrics != metric_names:
                        raise ValueError(
                            f"Metric header mismatch in {table_path}: {current_metrics} != {metric_names}"
                        )
                    continue
                if metric_names is None or len(cells) < len(metric_names) + 1:
                    continue

                characteristic = cells[0]
                if characteristic not in seen_characteristics:
                    characteristics.append(characteristic)
                    seen_characteristics.add(characteristic)
                for metric_name, value in zip(metric_names, cells[1:]):
                    metric_values.setdefault(metric_name, {}).setdefault(characteristic, {})[c_value] = value

    if metric_names is None:
        raise SystemExit(f"No results.tex tables found under {base_dir}/C=*/results.tex")

    for metric_name in metric_names:
        output_path = base_dir / f"C_compare_{metric_slug(metric_name)}.csv"
        with output_path.open("w", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["characteristic", *[f"C={c_value}" for c_value in available_c_values]])
            for characteristic in characteristics:
                row_values = metric_values.get(metric_name, {}).get(characteristic, {})
                writer.writerow([characteristic, *[row_values.get(c_value, "") for c_value in available_c_values]])
        print(f"Wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
