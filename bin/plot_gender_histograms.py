#!/usr/bin/env python3
"""Plot male/female real and generated x-vector histograms on one axis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from plot_profile_gmms import (
    build_lda_projection,
    build_pca_projection,
    condition_label,
    fit_ground_truth_gmm,
    parse_field_values,
    parse_test_csv_paths,
    prefixed_output_path,
)
from utils import (
    DEFAULT_TEST_DATASETS,
    FIELDS,
    condition_mask,
    generated_gmm_matches,
    load_gmm_metadata,
    load_xvectors,
    normalize_value,
    sample_gmm_rows,
)


DEFAULT_OUTPUT = Path("graphs/GMMs_visuals/gender_histograms.png")
FONT_SIZE = 15
GENDER_ORDER = ["male", "female"]
LINE_STYLES = {
    "real": ":",
    "generated": "-",
}
COLORS = {
    ("male", "real"): "#6baed6",
    ("male", "generated"): "#08519c",
    ("female", "real"): "#fdae6b",
    ("female", "generated"): "#d94801",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot overlaid male/female real and generated histograms.")
    parser.add_argument(
        "--test-csv-paths",
        type=parse_test_csv_paths,
        default=[(dataset_name, None) for dataset_name in DEFAULT_TEST_DATASETS],
    )
    parser.add_argument("--gmm-metadata-csv", type=Path, default=Path("exp/GMMs/metadata.csv"))
    parser.add_argument("--xvector-root", type=Path, default=Path("exp/xvectors/ecapa_tdnn"))
    parser.add_argument("--pitch", default="unknown")
    parser.add_argument("--age", default="unknown")
    parser.add_argument("--gender", default="male,female")
    parser.add_argument("--speaking-rate", default="unknown")
    parser.add_argument("--speech-monotony", default="unknown")
    parser.add_argument("--accent", default="unknown")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--ground-truth-components", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-load-workers", type=int, default=16)
    parser.add_argument("--lda", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_gender_only_args(args: argparse.Namespace) -> None:
    genders = parse_field_values(args.gender)
    if genders != GENDER_ORDER:
        raise ValueError("--gender must be exactly 'male,female' for this plotter")

    other_fields = {
        "pitch": args.pitch,
        "age": args.age,
        "speaking_rate": args.speaking_rate,
        "speech_monotony": args.speech_monotony,
        "accent": args.accent,
    }
    non_unknown = {
        field: value
        for field, value in other_fields.items()
        if parse_field_values(value) != ["unknown"]
    }
    if non_unknown:
        formatted = ", ".join(f"{field}={value}" for field, value in non_unknown.items())
        raise ValueError(f"This plotter only supports gender with all other fields unknown; got {formatted}")


def gender_conditions() -> list[pd.Series]:
    return [
        pd.Series(
            {
                "pitch": "unknown",
                "age": "unknown",
                "gender": gender,
                "speaking_rate": "unknown",
                "speech_monotony": "unknown",
                "accent": "unknown",
            }
        )
        for gender in GENDER_ORDER
    ]


def build_profile_data(
    condition: pd.Series,
    utterances: pd.DataFrame,
    xvectors: np.ndarray,
    gmm_metadata: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, Any]:
    condition_dict = {field: normalize_value(condition[field]) for field in FIELDS}
    mask = condition_mask(utterances, condition, ["gender"])
    selected_xvectors = xvectors[mask]
    if len(selected_xvectors) == 0:
        raise ValueError(f"No matching test xvectors for {condition_dict}")

    gmm_rows = generated_gmm_matches(
        gmm_metadata,
        condition_dict,
        strict_unknown_other_fields=False,
        unknown_is_wildcard=True,
    )
    if gmm_rows.empty:
        raise KeyError(f"No generated GMM metadata found for condition={condition_dict}")

    generated_points, sampled_gmm_rows = sample_gmm_rows(
        gmm_rows,
        args.samples,
        rng,
        prune_components=True,
    )
    first_gmm_path = Path(str(sampled_gmm_rows.iloc[0]["gmm_path"]))
    if not first_gmm_path.exists():
        raise FileNotFoundError(f"Missing generated GMM file: {first_gmm_path}")

    ground_truth_gmm = fit_ground_truth_gmm(
        selected_xvectors,
        num_components=args.ground_truth_components,
        seed=args.seed,
    )
    ground_truth_points, _ = ground_truth_gmm.sample(args.samples)

    print(f"Profile {condition_label(condition)}: matched {len(selected_xvectors)} test xvectors", flush=True)
    print(f"Sampled N={args.samples} from {len(gmm_rows)} generated GMM(s)", flush=True)
    return {
        "label": condition_dict["gender"],
        "condition": condition,
        "generated_points": generated_points,
        "ground_truth_points": ground_truth_points,
        "test_xvectors": selected_xvectors,
        "num_test_xvectors": len(selected_xvectors),
        "gmm_path": first_gmm_path,
        "num_generated_gmms": len(gmm_rows),
    }


def render_overlay_histogram(
    projected_profiles: list[dict[str, Any]],
    output_path: Path,
    x_label: str,
    title: str,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(9.5, 4))
    plt.tight_layout()
    all_values = []
    for profile in projected_profiles:
        all_values.extend(np.asarray(profile["test_xvectors"])[:, 0])
        all_values.extend(np.asarray(profile["generated_points"])[:, 0])
    bins = np.histogram_bin_edges(np.asarray(all_values), bins=45)

    for profile in projected_profiles:
        gender = profile["label"]
        real_values = np.asarray(profile["test_xvectors"])[:, 0]
        generated_values = np.asarray(profile["generated_points"])[:, 0]
        for source, values in [("real", real_values), ("generated", generated_values)]:
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2.1,
                linestyle=LINE_STYLES[source],
                color=COLORS[(gender, source)],
                label=f"{gender} {source}",
            )

    ax.set_title(title, fontsize=FONT_SIZE)
    ax.set_xlabel(x_label, fontsize=FONT_SIZE)
    ax.set_ylabel("Density", fontsize=FONT_SIZE)
    ax.grid(True, alpha=0.25)
    handles = [
        Line2D([0], [0], color=COLORS[("male", "real")], linestyle=":", linewidth=2.1, label="male real"),
        Line2D([0], [0], color=COLORS[("male", "generated")], linestyle="-", linewidth=2.1, label="male generated"),
        Line2D([0], [0], color=COLORS[("female", "real")], linestyle=":", linewidth=2.1, label="female real"),
        Line2D([0], [0], color=COLORS[("female", "generated")], linestyle="-", linewidth=2.1, label="female generated"),
    ]
    ax.legend(handles=handles, ncol=2, frameon=True, fontsize=FONT_SIZE, title_fontsize=FONT_SIZE)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_histograms(profiles: list[dict[str, Any]], output_path: Path, include_lda: bool) -> list[Path]:
    projected_pca, pca_x_label, _ = build_pca_projection(profiles)
    pca_output = prefixed_output_path(output_path, "PCA")
    render_overlay_histogram(projected_pca, pca_output, pca_x_label, "PCA gender distributions")

    output_paths = [pca_output]
    if include_lda:
        projected_lda, lda_x_label, _ = build_lda_projection(profiles)
        lda_output = prefixed_output_path(output_path, "LDA")
        render_overlay_histogram(projected_lda, lda_output, lda_x_label, "LDA of the generated and real x-vectors for male and females")
        output_paths.append(lda_output)
    return output_paths


def main() -> int:
    args = parse_args()
    validate_gender_only_args(args)
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.ground_truth_components <= 0:
        raise ValueError("--ground-truth-components must be positive")

    rng = np.random.default_rng(args.seed)
    selection_rng = np.random.default_rng(args.seed)
    utterances, xvectors = load_xvectors(
        args.test_csv_paths,
        args.xvector_root,
        profile_fields=FIELDS,
        num_workers=args.num_load_workers,
        rng=selection_rng,
    )
    if len(xvectors) == 0:
        raise ValueError(f"No xvectors loaded from {args.test_csv_paths}")

    gmm_metadata = load_gmm_metadata(args.gmm_metadata_csv)
    profiles = [
        build_profile_data(condition, utterances, xvectors, gmm_metadata, args, rng)
        for condition in gender_conditions()
    ]
    output_paths = plot_histograms(profiles, args.output, include_lda=args.lda)
    for output_path in output_paths:
        print(f"Wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
