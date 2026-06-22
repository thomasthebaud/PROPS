#!/usr/bin/env python3
"""Aggregate NLL score CSVs into a LaTeX table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


COLUMN_ORDER = [
    ("test", "desc0", "test_gpt"),
    ("test", "capspeech_prompt", "test_capspeech"),
    ("dev", "desc1", "dev_gpt"),
    ("dev", "capspeech_prompt", "dev_capspeech"),
]
ROW_ORDER = [
    r"$\mathcal{N}(0,1)$",
    "Random GMMs",
    "Precomputed GMMs",
    "Pretrained MDN",
    "Finetuned MDN",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a LaTeX table from exp/NLL_scores CSVs.")
    parser.add_argument("--scores-root", type=Path, default=Path("exp/NLL_scores"))
    parser.add_argument("--output-path", type=Path, default=Path("exp/NLL_scores/table.tex"))
    parser.add_argument("--digits", type=int, default=0, help="Digits after the decimal point in table values.")
    return parser.parse_args()


def expected_score_files(scores_root: Path) -> list[Path]:
    files: list[Path] = []
    for model_variant in ["pretrain", "pretrain_uniform", "finetune"]:
        for split, desc_columns in [("test", ["desc0", "capspeech_prompt"]), ("dev", ["desc1", "capspeech_prompt"])]:
            for desc_column in desc_columns:
                files.append(scores_root / "jobs" / f"{model_variant}_{split}_{desc_column}.csv")
    for baseline_model in ["normal", "random"]:
        for split in ["dev", "test"]:
            files.append(scores_root / "random" / f"{baseline_model}_{split}.csv")
    return files


def load_scores(scores_root: Path) -> pd.DataFrame:
    print(f"Loading NLL scores from {scores_root}", flush=True)
    frames = []
    jobs_dir = scores_root / "jobs"
    random_dir = scores_root / "random"
    job_csvs = sorted(jobs_dir.glob("*.csv")) if jobs_dir.exists() else []
    random_csvs = sorted(random_dir.glob("*.csv")) if random_dir.exists() else []
    csv_paths = [*job_csvs, *random_csvs]
    print(
        f"Found {len(job_csvs)} job CSV(s) and {len(random_csvs)} baseline CSV(s)",
        flush=True,
    )
    for csv_path in tqdm(csv_paths, desc="Loading score CSVs", unit="csv"):
        frame = pd.read_csv(csv_path)
        tqdm.write(f"Loaded {csv_path} ({len(frame)} rows)")
        frames.append(frame)
    if not frames:
        print("No stage job/random CSVs found; checking legacy split directories", flush=True)
        split_dirs = sorted(path for path in scores_root.iterdir() if path.is_dir()) if scores_root.exists() else []
        for split_dir in tqdm(split_dirs, desc="Checking legacy split dirs", unit="dir"):
            csv_path = split_dir / "nll_scores.csv"
            if csv_path.exists():
                frame = pd.read_csv(csv_path)
                tqdm.write(f"Loaded {csv_path} ({len(frame)} rows)")
                frames.append(frame)
    if not frames:
        print(f"WARNING: no NLL score CSV files found under {scores_root}; table values will be NA", flush=True)
        return pd.DataFrame(columns=["evaluation_set", "model", "train_fraction", "gmm_split", "desc_column", "weighting", "nll"])
    print(f"Concatenating {len(frames)} score frame(s)", flush=True)
    scores = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(scores)} total score row(s)", flush=True)
    return scores


def format_mean(mean: float, digits: int) -> str:
    if not np.isfinite(mean):
        return "NA"
    return f"{mean:.{digits}f}"


def escape_latex(text: object) -> str:
    value = str(text)
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
    return "".join(replacements.get(char, char) for char in value)


def row_label(row: pd.Series) -> str | None:
    if row["model"] == "normal":
        return r"$\mathcal{N}(0,1)$"
    if row["model"] == "random_gmm":
        return "Random GMMs"
    if row["model"] == "pretrain" and row["weighting"] == "uniform_pi":
        return "Precomputed GMMs"
    if row["model"] == "pretrain" and row["weighting"] == "learned":
        return "Pretrained MDN"
    if row["model"] == "finetune" and row["weighting"] == "learned":
        return "Finetuned MDN"
    return None


def column_key(row: pd.Series) -> str | None:
    for gmm_split, desc_column, key in COLUMN_ORDER:
        if row["gmm_split"] == gmm_split and row["desc_column"] == desc_column:
            return key
    return None


def expand_baseline_columns(scores: pd.DataFrame) -> pd.DataFrame:
    print(f"Expanding baseline rows across table columns for {len(scores)} score row(s)", flush=True)
    expanded_rows = []
    for _, row in tqdm(scores.iterrows(), total=len(scores), desc="Expanding baseline rows", unit="row"):
        if row["model"] not in {"normal", "random_gmm"}:
            expanded_rows.append(row.to_dict())
            continue
        if row["gmm_split"] == "test":
            columns = ["test_gpt", "test_capspeech"]
        elif row["gmm_split"] == "dev":
            columns = ["dev_gpt", "dev_capspeech"]
        else:
            columns = []
        for column in columns:
            item = row.to_dict()
            item["table_column"] = column
            expanded_rows.append(item)
    expanded = pd.DataFrame(expanded_rows)
    print(f"Expanded to {len(expanded)} table candidate row(s)", flush=True)
    return expanded


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    print(f"Summarizing {len(scores)} loaded score row(s)", flush=True)
    scores = scores.copy()
    table_rows = []
    table_columns = []
    for _, row in tqdm(scores.iterrows(), total=len(scores), desc="Mapping rows to table cells", unit="row"):
        table_rows.append(row_label(row))
        table_columns.append(column_key(row))
    scores["table_row"] = table_rows
    scores["table_column"] = table_columns
    scores = expand_baseline_columns(scores)
    print(f"Filtering rows that match the Stage 5 table layout from {len(scores)} row(s)", flush=True)
    scores = scores.dropna(subset=["table_row", "table_column"])
    print(f"{len(scores)} row(s) remain after table-layout filtering", flush=True)
    if scores.empty:
        print("WARNING: no scores matched the Stage 5 table layout; table values will be NA", flush=True)
        return pd.DataFrame(columns=["table_row", "table_column", "mean_nll", "std_nll", "num_scores"])
    print("Grouping NLL scores by table row and column", flush=True)
    summary = (
        scores.groupby(["table_row", "table_column"], dropna=False)["nll"]
        .agg(mean_nll="mean", std_nll="std", num_scores="count")
        .reset_index()
    )
    summary["std_nll"] = summary["std_nll"].fillna(0.0)
    print(f"Computed {len(summary)} summary cell(s)", flush=True)
    return summary


def make_table(summary: pd.DataFrame, digits: int) -> str:
    print(f"Rendering LaTeX table with {len(summary)} summary cell(s)", flush=True)
    cells = {
        (row["table_row"], row["table_column"]): format_mean(row["mean_nll"], digits)
        for _, row in tqdm(summary.iterrows(), total=len(summary), desc="Indexing summary cells", unit="cell")
    }

    def cell(row_label: str, col_key: str) -> str:
        if (row_label, col_key) not in cells:
            print(f"WARNING: missing score for {row_label} / {col_key}; using NA", flush=True)
            return "NA"
        return cells[(row_label, col_key)]
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "\\multirow{2}{*}{Model} & \\multicolumn{2}{c}{Test} & \\multicolumn{2}{c}{Dev} " + r"\\",
        " & GPT 0 & CapSpeech & GPT 1 & CapSpeech " + r"\\",
        r"\midrule",
    ]
    for label in tqdm(ROW_ORDER, desc="Rendering LaTeX rows", unit="row"):
        lines.append(
            " & ".join(
                [
                    label if label.startswith("$") else escape_latex(label),
                    cell(label, "test_gpt"),
                    cell(label, "test_capspeech"),
                    cell(label, "dev_gpt"),
                    cell(label, "dev_capspeech"),
                ]
            )
            + r" \\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    print(
        f"Stage 5 table: scores_root={args.scores_root}, output_path={args.output_path}, digits={args.digits}",
        flush=True,
    )
    expected_files = expected_score_files(args.scores_root)
    print(f"Checking {len(expected_files)} expected score CSV(s)", flush=True)
    for csv_path in tqdm(expected_files, desc="Checking expected CSVs", unit="csv"):
        if not csv_path.exists():
            tqdm.write(f"WARNING: missing expected score CSV: {csv_path}")
    scores = load_scores(args.scores_root)
    summary = summarize(scores)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing LaTeX table to {args.output_path}", flush=True)
    args.output_path.write_text(make_table(summary, args.digits), encoding="utf-8")
    summary_path = args.output_path.with_suffix(".summary.csv")
    print(f"Writing summary CSV to {summary_path}", flush=True)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {args.output_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
