#!/usr/bin/env python3
"""Evaluate precomputed profile GMMs on dev/test xvectors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    FIELDS,
    condition_key,
    gmm_log_likelihood,
    load_gmm,
    load_gmm_metadata,
    load_xvectors,
    normalize_characteristic_value,
    parse_csv_paths,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score dev/test xvectors against matching precomputed profile GMMs.")
    parser.add_argument("--dataset-name", default="Capspeech_min100")
    parser.add_argument("--ks", type=parse_csv_paths, default=["1", "2", "4", "8", "16"])
    parser.add_argument("--gmm-root", type=Path, default=Path("exp/GMMs"))
    parser.add_argument("--xvector-root", type=Path, default=Path("exp/xvectors/ecapa_tdnn"))
    parser.add_argument("--output-dir", type=Path, default=Path("exp/graphs/Best_K"))
    parser.add_argument("--cache-dir", type=Path, default=Path("exp/Best_K"))
    parser.add_argument("--profile-counts-csv", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-load-workers", type=int, default=16)
    return parser.parse_args()


def parse_ks(values: list[str]) -> list[int]:
    ks: list[int] = []
    for value in values:
        try:
            k = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid K value: {value!r}") from exc
        if k <= 0:
            raise ValueError(f"K must be positive, got {k}")
        ks.append(k)
    return ks




def resolve_profile_counts_csv(dataset_name: str, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Missing profile counts CSV: {explicit_path}")
        return explicit_path

    dataset_dir = Path("data") / dataset_name
    candidates = [dataset_dir / "profile_prompts.csv", dataset_dir / "profile_combinations.csv"]
    for candidate in candidates:
        if candidate.exists():
            header = pd.read_csv(candidate, nrows=0)
            if "num_audios" in header.columns:
                return candidate
    raise FileNotFoundError(
        f"Could not find num_audios in profile_prompts.csv or profile_combinations.csv under {dataset_dir}"
    )


def compute_profile_counts_by_k(profile_counts_csv: Path, ks: list[int]) -> dict[int, int]:
    profiles = pd.read_csv(profile_counts_csv, usecols=["num_audios"])
    counts = pd.to_numeric(profiles["num_audios"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    return {k: int(np.sum(counts >= 2 * k)) for k in ks}


def normalized_condition(row: pd.Series | dict[str, Any]) -> dict[str, str]:
    return {
        field: normalize_characteristic_value(field, row.get(field, "unknown"))
        for field in FIELDS
    }


def build_gmm_lookup(metadata: pd.DataFrame) -> tuple[dict[tuple[str, ...], pd.Series], list[tuple[int, tuple[str, ...], pd.Series]]]:
    exact_lookup: dict[tuple[str, ...], pd.Series] = {}
    wildcard_rows: list[tuple[int, tuple[str, ...], pd.Series]] = []
    for _, row in metadata.iterrows():
        key = condition_key(row, FIELDS)
        exact_lookup.setdefault(key, row)
        specificity = sum(value != "unknown" for value in key)
        wildcard_rows.append((specificity, key, row))
    wildcard_rows.sort(key=lambda item: item[0], reverse=True)
    return exact_lookup, wildcard_rows


def key_matches_profile(profile_key: tuple[str, ...], utterance_key: tuple[str, ...]) -> bool:
    return all(profile_value == "unknown" or profile_value == utterance_value for profile_value, utterance_value in zip(profile_key, utterance_key))


def find_profile_gmm(
    utterance: pd.Series,
    exact_lookup: dict[tuple[str, ...], pd.Series],
    wildcard_rows: list[tuple[int, tuple[str, ...], pd.Series]],
) -> tuple[pd.Series | None, str]:
    key = condition_key(utterance, FIELDS)
    exact = exact_lookup.get(key)
    if exact is not None:
        return exact, "exact"
    for _specificity, profile_key, row in wildcard_rows:
        if key_matches_profile(profile_key, key):
            return row, "wildcard"
    return None, "missing"




def cache_path_for(cache_dir: Path, dataset_name: str, split_name: str, k: int) -> Path:
    return cache_dir / f"{dataset_name}_{split_name}_K={k}_nll.npz"


def save_nll_cache(
    path: Path,
    *,
    split_name: str,
    dataset_name: str,
    k: int,
    metadata_csv: Path,
    summary: dict[str, object],
    nll_rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(
            f,
            split=np.asarray(split_name),
            dataset_name=np.asarray(dataset_name),
            K=np.asarray(k, dtype=np.int64),
            metadata_csv=np.asarray(str(metadata_csv)),
            nll=np.asarray([row["nll"] for row in nll_rows], dtype=np.float64),
            row_index=np.asarray([row["row_index"] for row in nll_rows], dtype=np.int64),
            utterance_id=np.asarray([row["utterance_id"] for row in nll_rows], dtype=str),
            match_type=np.asarray([row["match_type"] for row in nll_rows], dtype=str),
            gmm_path=np.asarray([row["gmm_path"] for row in nll_rows], dtype=str),
            num_total_xvectors=np.asarray(summary["num_total_xvectors"], dtype=np.int64),
            num_missing_gmm=np.asarray(summary["num_missing_gmm"], dtype=np.int64),
            num_exact_matches=np.asarray(summary["num_exact_matches"], dtype=np.int64),
            num_wildcard_matches=np.asarray(summary["num_wildcard_matches"], dtype=np.int64),
            num_loaded_gmms=np.asarray(summary["num_loaded_gmms"], dtype=np.int64),
        )
    tmp_path.replace(path)


def load_nll_cache(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    with np.load(path) as data:
        split_name = str(data["split"].item())
        nlls = np.asarray(data["nll"], dtype=np.float64)
        row_indices = np.asarray(data["row_index"], dtype=np.int64)
        utterance_ids = np.asarray(data["utterance_id"]).astype(str)
        match_types = np.asarray(data["match_type"]).astype(str)
        gmm_paths = np.asarray(data["gmm_path"]).astype(str)
        summary = {
            "split": split_name,
            "mean_nll": float(np.mean(nlls)) if len(nlls) else np.nan,
            "std_nll": float(np.std(nlls, ddof=1)) if len(nlls) > 1 else 0.0,
            "num_scored_xvectors": int(len(nlls)),
            "num_total_xvectors": int(data["num_total_xvectors"].item()),
            "num_missing_gmm": int(data["num_missing_gmm"].item()),
            "num_exact_matches": int(data["num_exact_matches"].item()),
            "num_wildcard_matches": int(data["num_wildcard_matches"].item()),
            "num_loaded_gmms": int(data["num_loaded_gmms"].item()),
        }
        nll_rows = [
            {
                "split": split_name,
                "row_index": int(row_index),
                "utterance_id": utterance_id,
                "nll": float(nll),
                "match_type": match_type,
                "gmm_path": gmm_path,
            }
            for row_index, utterance_id, nll, match_type, gmm_path in zip(
                row_indices,
                utterance_ids,
                nlls,
                match_types,
                gmm_paths,
            )
        ]
    return summary, nll_rows


def score_split(
    split_name: str,
    dataset_name: str,
    k: int,
    metadata: pd.DataFrame,
    metadata_path: Path,
    xvector_root: Path,
    num_load_workers: int,
    cache_dir: Path,
    overwrite: bool,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    cache_path = cache_path_for(cache_dir, dataset_name, split_name, k)
    if cache_path.exists() and not overwrite:
        print(f"K={k} {split_name}: loading cached NLL scores from {cache_path}", flush=True)
        return load_nll_cache(cache_path)

    utterances, xvectors = load_xvectors(
        [f"{dataset_name}_{split_name}"],
        xvector_root,
        num_workers=num_load_workers,
    )
    if len(xvectors) == 0:
        raise ValueError(f"No xvectors loaded for {dataset_name}_{split_name}")

    exact_lookup, wildcard_rows = build_gmm_lookup(metadata)
    loaded_gmms: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    missing_gmm_paths: set[str] = set()
    nlls: list[float] = []
    nll_rows: list[dict[str, object]] = []
    exact_matches = 0
    wildcard_matches = 0
    missing_matches = 0

    for row_idx, ((_, utterance), xvector) in enumerate(zip(utterances.iterrows(), xvectors)):
        gmm_row, match_type = find_profile_gmm(utterance, exact_lookup, wildcard_rows)
        if gmm_row is None:
            missing_matches += 1
            continue
        gmm_path = str(gmm_row["gmm_path"])
        if gmm_path not in loaded_gmms:
            try:
                loaded_gmms[gmm_path] = load_gmm(Path(gmm_path))
            except FileNotFoundError:
                missing_matches += 1
                if gmm_path not in missing_gmm_paths:
                    missing_gmm_paths.add(gmm_path)
                    print(
                        f"WARNING: K={k} {split_name}: matched metadata row but GMM file was not found: {gmm_path}",
                        flush=True,
                    )
                continue
        if match_type == "exact":
            exact_matches += 1
        else:
            wildcard_matches += 1

        _pi_logits, pi, mu, sigma = loaded_gmms[gmm_path]
        nll = -float(gmm_log_likelihood(xvector.reshape(1, -1), pi, mu, sigma)[0])
        nlls.append(nll)
        nll_rows.append(
            {
                "split": split_name,
                "row_index": row_idx,
                "utterance_id": str(utterance.get("id", row_idx)),
                "nll": nll,
                "match_type": match_type,
                "gmm_path": gmm_path,
            }
        )

    values = np.asarray(nlls, dtype=np.float64)
    summary = {
        "split": split_name,
        "mean_nll": float(np.mean(values)) if len(values) else np.nan,
        "std_nll": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "num_scored_xvectors": int(len(values)),
        "num_total_xvectors": int(len(xvectors)),
        "num_missing_gmm": int(missing_matches),
        "num_exact_matches": int(exact_matches),
        "num_wildcard_matches": int(wildcard_matches),
        "num_loaded_gmms": int(len(loaded_gmms)),
    }
    save_nll_cache(
        cache_path,
        split_name=split_name,
        dataset_name=dataset_name,
        k=k,
        metadata_csv=metadata_path,
        summary=summary,
        nll_rows=nll_rows,
    )
    print(f"K={k} {split_name}: saved NLL scores to {cache_path}", flush=True)
    return summary, nll_rows


def plot_results(
    nll_rows: list[dict[str, object]],
    ks: list[int],
    profile_counts_by_k: dict[int, int],
    output_path: Path,
) -> None:
    df = pd.DataFrame(nll_rows)
    df['nll'] = -df['nll']
    ks = sorted(ks)
    positions = np.arange(len(ks), dtype=float)
    colors = {"dev": "#1f77b4", "test": "#ff7f0e"}

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    legend_handles = []
    offsets = {"dev": -0.18, "test": 0.18}
    for split in ["dev", "test"]:
        split_data = []
        split_positions = []
        for position, k in zip(positions, ks):
            if df.empty:
                values = np.asarray([], dtype=float)
            else:
                values = df[(df["K"] == k) & (df["split"] == split)]["nll"].to_numpy(dtype=float)
            if len(values):
                split_data.append(values)
                split_positions.append(position + offsets[split])
        if not split_data:
            continue
        boxplot = ax.boxplot(
            split_data,
            positions=split_positions,
            widths=0.32,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(colors[split])
            patch.set_edgecolor(colors[split])
            patch.set_alpha(0.5)
            patch.set_linewidth(1.1)
        for key in ["whiskers", "caps"]:
            for artist in boxplot[key]:
                artist.set_color(colors[split])
                artist.set_alpha(0.85)
                artist.set_linewidth(1.0)
        legend_handles.append(plt.Line2D([0], [0], color=colors[split], linewidth=8, alpha=0.5, label=split))

    ax.set_xticks(positions)
    ax.set_xticklabels([str(int(k)) for k in ks])
    ax.set_xlabel("Number of GMM components (K)", fontsize=18)
    ax.set_ylabel("Negative Log-likelihood score", fontsize=18)
    ax.grid(True, axis="y", alpha=0.3)

    ax_counts = ax.twinx()
    count_color = "#2ca02c"
    count_values = [profile_counts_by_k.get(int(k), 0) for k in ks]
    ax_counts.plot(
        positions,
        count_values,
        color=count_color,
        marker="D",
        linewidth=2.0,
        zorder=6,
    )
    ax_counts.set_ylabel("Profiles with at least 2K xvectors", color=count_color, fontsize=16)
    ax_counts.tick_params(axis="y", colors=count_color)
    ax_counts.spines["right"].set_color(count_color)
    y_offset = max(count_values) * 0.015 if count_values else 0.0
    for position, count in zip(positions, count_values):
        ax_counts.text(
            position,
            count + y_offset,
            str(count),
            color=count_color,
            fontsize=7,
            ha="center",
            va="bottom",
        )

    ax.legend(handles=legend_handles, title="Split", loc="lower left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    ks = parse_ks(args.ks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    profile_counts_csv = resolve_profile_counts_csv(args.dataset_name, args.profile_counts_csv)
    profile_counts_by_k = compute_profile_counts_by_k(profile_counts_csv, ks)
    print(f"Loaded profile counts from {profile_counts_csv}", flush=True)
    for k in ks:
        print(f"K={k}: {profile_counts_by_k[k]} profiles have at least {2 * k} xvectors", flush=True)

    results: list[dict[str, object]] = []
    nll_rows: list[dict[str, object]] = []
    for k in ks:
        metadata_path = args.gmm_root / f"{k}_components_precomputed" / "metadata.csv"
        if not metadata_path.exists():
            print(f"WARNING: Missing precomputed metadata for K={k}: {metadata_path}", flush=True)
            for split in ["dev", "test"]:
                results.append(
                    {
                        "K": k,
                        "split": split,
                        "mean_nll": np.nan,
                        "std_nll": 0.0,
                        "num_scored_xvectors": 0,
                        "num_total_xvectors": 0,
                        "num_missing_gmm": 0,
                        "num_exact_matches": 0,
                        "num_wildcard_matches": 0,
                        "num_loaded_gmms": 0,
                        "num_profiles_with_at_least_2k_xvectors": profile_counts_by_k[k],
                        "profile_counts_csv": str(profile_counts_csv),
                        "nll_cache_path": str(cache_path_for(args.cache_dir, args.dataset_name, split, k)),
                        "metadata_csv": str(metadata_path),
                    }
                )
            continue
        metadata = load_gmm_metadata(metadata_path)
        available_gmm_mask = metadata["gmm_path"].map(lambda value: Path(str(value)).exists())
        num_available_gmms = int(available_gmm_mask.sum())
        if num_available_gmms == 0:
            print(
                f"WARNING: No available precomputed GMM files for K={k} from {metadata_path}; "
                "skipping xvector loading",
                flush=True,
            )
            for split in ["dev", "test"]:
                results.append(
                    {
                        "K": k,
                        "split": split,
                        "mean_nll": np.nan,
                        "std_nll": 0.0,
                        "num_scored_xvectors": 0,
                        "num_total_xvectors": 0,
                        "num_missing_gmm": 0,
                        "num_exact_matches": 0,
                        "num_wildcard_matches": 0,
                        "num_loaded_gmms": 0,
                        "num_profiles_with_at_least_2k_xvectors": profile_counts_by_k[k],
                        "profile_counts_csv": str(profile_counts_csv),
                        "nll_cache_path": str(cache_path_for(args.cache_dir, args.dataset_name, split, k)),
                        "metadata_csv": str(metadata_path),
                    }
                )
            continue
        if num_available_gmms < len(metadata):
            print(
                f"WARNING: K={k}: ignoring {len(metadata) - num_available_gmms} metadata rows "
                "whose GMM files are missing",
                flush=True,
            )
        metadata = metadata.loc[available_gmm_mask].reset_index(drop=True)
        print(
            f"Loaded {num_available_gmms} available precomputed K={k} GMMs from {metadata_path}",
            flush=True,
        )
        for split in ["dev", "test"]:
            row, split_nll_rows = score_split(
                split,
                args.dataset_name,
                k,
                metadata,
                metadata_path,
                args.xvector_root,
                args.num_load_workers,
                args.cache_dir,
                args.overwrite,
            )
            row["K"] = k
            row["num_profiles_with_at_least_2k_xvectors"] = profile_counts_by_k[k]
            row["profile_counts_csv"] = str(profile_counts_csv)
            row["nll_cache_path"] = str(cache_path_for(args.cache_dir, args.dataset_name, split, k))
            row["metadata_csv"] = str(metadata_path)
            for nll_row in split_nll_rows:
                nll_row["K"] = k
            results.append(row)
            nll_rows.extend(split_nll_rows)
            print(
                f"K={k} {split}: mean NLL={row['mean_nll']:.4f}, std={row['std_nll']:.4f}, "
                f"scored={row['num_scored_xvectors']}/{row['num_total_xvectors']}, "
                f"missing_gmm={row['num_missing_gmm']}",
                flush=True,
            )

    csv_path = args.output_dir / "precomputed_gmm_nll.csv"
    write_csv(
        csv_path,
        results,
        [
            "K",
            "split",
            "mean_nll",
            "std_nll",
            "num_scored_xvectors",
            "num_total_xvectors",
            "num_missing_gmm",
            "num_exact_matches",
            "num_wildcard_matches",
            "num_loaded_gmms",
            "num_profiles_with_at_least_2k_xvectors",
            "profile_counts_csv",
            "nll_cache_path",
            "metadata_csv",
        ],
    )
    nll_csv_path = args.output_dir / "precomputed_gmm_nll_values.csv"
    write_csv(
        nll_csv_path,
        nll_rows,
        ["K", "split", "row_index", "utterance_id", "nll", "match_type", "gmm_path"],
    )
    plot_path = args.output_dir / "precomputed_gmm_nll.png"
    plot_results(nll_rows, ks, profile_counts_by_k, plot_path)
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {nll_csv_path}", flush=True)
    print(f"Wrote {plot_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
