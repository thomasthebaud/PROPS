#!/usr/bin/env python3
"""Plot generated and ground-truth profile GMMs with PCA."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture

from utils import (
    DEFAULT_TEST_DATASETS,
    condition_key,
    condition_mask,
    load_gmm,
    load_gmm_metadata,
    load_xvectors,
    normalize_value,
    prune_gmm_components,
    sample_gmm,
)


PLOT_FIELDS = ["gender", "age", "accent"]
DEFAULT_OUTPUT = Path("graphs/GMMs_visuals/age_gender_accent.png")


def parse_test_csv_paths(value: str) -> list[tuple[str, float | None]]:
    specs: list[tuple[str, float | None]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            specs.append((item, None))
            continue

        dataset_name, proportion_text = (part.strip() for part in item.split("=", 1))
        if not dataset_name:
            raise argparse.ArgumentTypeError(f"Missing dataset name in {item!r}")
        try:
            proportion = float(proportion_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid sample proportion in {item!r}") from exc
        if not 0 < proportion <= 1:
            raise argparse.ArgumentTypeError(
                f"Sample proportion for {dataset_name!r} must be > 0 and <= 1; got {proportion}"
            )
        specs.append((dataset_name, proportion))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize generated and ground-truth profile GMMs.")
    parser.add_argument(
        "--test-csv-paths",
        type=parse_test_csv_paths,
        default=[(dataset_name, None) for dataset_name in DEFAULT_TEST_DATASETS],
        help=(
            "Comma-separated dataset names under data/ to load xvectors from. "
            "Use DATASET=PROPORTION, e.g. CommonVoice_train=0.05, to load a random subset."
        ),
    )
    parser.add_argument(
        "--gmm-metadata-csv",
        type=Path,
        default=Path("exp/GMMs/metadata.csv"),
        help="Metadata CSV written by step 06.",
    )
    parser.add_argument(
        "--xvector-root",
        type=Path,
        default=Path("exp/xvectors/ecapa_tdnn"),
        help="Root directory containing test-set xvectors.",
    )
    parser.add_argument("--pitch", default="unknown")
    parser.add_argument("--age", default="young adult")
    parser.add_argument("--gender", default="male")
    parser.add_argument("--speaking-rate", default="unknown")
    parser.add_argument("--speech-monotony", default="unknown")
    parser.add_argument("--accent", default="England")
    parser.add_argument("--samples", type=int, default=5000, help="Samples to draw from each GMM for PCA/KDE.")
    parser.add_argument(
        "--ground-truth-components",
        type=int,
        default=128,
        help="Number of components for the test-set ground-truth GMM.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-load-workers", type=int, default=16, help="Number of threads for xvector loading.")
    parser.add_argument("--lda", action="store_true", help="Also save an LDA projection plot using requested profiles as classes.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()



def parse_field_values(value: str) -> list[str]:
    values = [normalize_value(item) for item in str(value).split(",") if str(item).strip()]
    return values or ["unknown"]


def requested_conditions(args: argparse.Namespace) -> list[pd.Series]:
    values_by_field = {
        "age": parse_field_values(args.age),
        "gender": parse_field_values(args.gender),
        "accent": parse_field_values(args.accent),
    }
    conditions = []
    for values in product(*(values_by_field[field] for field in PLOT_FIELDS)):
        conditions.append(pd.Series(dict(zip(PLOT_FIELDS, values))))
    return conditions


def condition_label(condition: pd.Series) -> str:
    known = [
        f"{field.replace('_', ' ')}={normalize_value(condition[field])}"
        for field in PLOT_FIELDS
        if normalize_value(condition[field]) != "unknown"
    ]
    return ", ".join(known) or "all profiles"



def fit_ground_truth_gmm(xvectors: np.ndarray, num_components: int, seed: int) -> GaussianMixture:
    if len(xvectors) < 2:
        raise ValueError(f"Need at least 2 matching xvectors to fit a ground-truth GMM, got {len(xvectors)}")
    fitted_components = min(num_components, len(xvectors))
    return GaussianMixture(
        n_components=fitted_components,
        covariance_type="diag",
        reg_covar=1e-6,
        random_state=seed,
    ).fit(xvectors)



def prefixed_output_path(output_path: Path, prefix: str) -> Path:
    return output_path.with_name(f"{prefix}_{output_path.name}")


def projection_blocks(profiles: list[dict[str, Any]], test_key: str) -> list[tuple[int, str, np.ndarray]]:
    blocks: list[tuple[int, str, np.ndarray]] = []
    for profile_idx, profile in enumerate(profiles):
        blocks.extend(
            [
                (profile_idx, "generated_points", profile["generated_points"]),
                (profile_idx, "ground_truth_points", profile["ground_truth_points"]),
                (profile_idx, "test_xvectors", profile[test_key]),
                (profile_idx, "generated_centers", profile["generated_centers"]),
                (profile_idx, "ground_truth_centers", profile["ground_truth_centers"]),
            ]
        )
    return blocks


def split_projection(
    projected: np.ndarray,
    blocks: list[tuple[int, str, np.ndarray]],
    profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    projected_profiles = [
        {"label": profile["label"], "condition": profile["condition"], "num_test_xvectors": profile["num_test_xvectors"]}
        for profile in profiles
    ]
    cursor = 0
    for profile_idx, name, values in blocks:
        next_cursor = cursor + len(values)
        projected_profiles[profile_idx][name] = projected[cursor:next_cursor]
        cursor = next_cursor
    return projected_profiles


def build_pca_projection(profiles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, str]:
    blocks = projection_blocks(profiles, test_key="test_xvectors")
    train_on_xvectors_only=True
    if train_on_xvectors_only: pca_training_xvectors = np.vstack([profile["test_xvectors"] for profile in profiles])
    combined = np.vstack([block[2] for block in blocks])

    pca = PCA(n_components=2, random_state=0)
    if train_on_xvectors_only:
        pca.fit(pca_training_xvectors)
        projected = pca.transform(combined)
        print(f"PCA fit on test xvectors: {pca_training_xvectors.shape}; transformed: {projected.shape}")
    else:
        projected = pca.fit_transform(combined)
        print(f"PCA fit on all xvectors, transformed shape: {projected.shape}")

    x_label = f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)"
    y_label = f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)"
    return split_projection(projected, blocks, profiles), x_label, y_label


def build_lda_projection(profiles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, str]:
    if len(profiles) < 2:
        raise ValueError("--lda requires at least two requested profiles/classes")

    blocks = projection_blocks(profiles, test_key="test_xvectors")
    training_xvectors = np.vstack([profile["test_xvectors"] for profile in profiles])
    training_labels = np.concatenate(
        [np.full(len(profile["test_xvectors"]), profile_idx) for profile_idx, profile in enumerate(profiles)]
    )
    combined = np.vstack([block[2] for block in blocks])

    n_components = min(2, len(profiles) - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(training_xvectors, training_labels)
    projected = lda.transform(combined)
    if projected.shape[1] == 1:
        projected = np.column_stack([projected[:, 0], np.zeros(len(projected), dtype=projected.dtype)])

    print(
        f"LDA fit on test xvectors: {training_xvectors.shape}, classes={len(profiles)}; "
        f"transformed: {projected.shape}"
    )
    y_label = "LD2" if n_components > 1 else "0"
    return split_projection(projected, blocks, profiles), "LD1", y_label



def sample_frame(projected_profiles: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for profile in projected_profiles:
        for distribution, points in [
            ("Generated GMM", profile["generated_points"]),
            ("Test-set GMM", profile["ground_truth_points"]),
        ]:
            rows.append(
                pd.DataFrame(
                    {
                        "X": points[:, 0],
                        "Y": points[:, 1],
                        "Distribution": distribution,
                        "Profile": profile["label"],
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def compute_axis_limits(projected_profiles: list[dict[str, Any]], padding_fraction: float = 0.1) -> tuple[float, float, float, float]:
    combined = np.vstack([profile["test_xvectors"] for profile in projected_profiles])
    xmin, ymin = np.min(combined, axis=0)
    xmax, ymax = np.max(combined, axis=0)
    x_pad = max((xmax - xmin) * padding_fraction, 1e-3)
    y_pad = max((ymax - ymin) * padding_fraction, 1e-3)
    return xmin - x_pad, xmax + x_pad, ymin - y_pad, ymax + y_pad


def draw_profile_layers(
    ax: plt.Axes,
    profile: dict[str, Any],
    color: tuple[float, float, float],
    show_test_kde: bool = False,
    show_test_xvectors: bool = False,
    show_test_centers: bool = False,
    show_generated_kde: bool = False,
    show_generated_centers: bool = False,
) -> None:
    generated_points = profile["generated_points"]
    ground_truth_points = profile["ground_truth_points"]

    if show_generated_kde and len(generated_points) >= 3:
        generated_df = pd.DataFrame({"X": generated_points[:, 0], "Y": generated_points[:, 1]})
        sns.kdeplot(
            data=generated_df,
            x="X",
            y="Y",
            color=color,
            fill=True,
            common_norm=False,
            alpha=0.20,
            levels=8,
            thresh=0.04,
            ax=ax,
            zorder=1,
        )
    if show_test_kde and len(ground_truth_points) >= 3:
        ground_truth_df = pd.DataFrame({"X": ground_truth_points[:, 0], "Y": ground_truth_points[:, 1]})
        sns.kdeplot(
            data=ground_truth_df,
            x="X",
            y="Y",
            color=color,
            fill=False,
            levels=5,
            thresh=0.02,
            linewidths=1.9,
            linestyles="--",
            ax=ax,
            zorder=4,
        )
    if show_test_xvectors:
        ax.scatter(
            profile["test_xvectors"][:, 0],
            profile["test_xvectors"][:, 1],
            marker=".",
            s=18,
            color=color,
            alpha=0.55,
            linewidths=0,
            zorder=5,
        )
    if show_generated_centers:
        ax.scatter(
            profile["generated_centers"][:, 0],
            profile["generated_centers"][:, 1],
            marker="o",
            s=48,
            facecolors="none",
            edgecolors=color,
            linewidths=1.3,
            alpha=0.98,
            zorder=6,
        )
    if show_test_centers:
        ax.scatter(
            profile["ground_truth_centers"][:, 0],
            profile["ground_truth_centers"][:, 1],
            marker="^",
            s=52,
            facecolors="none",
            edgecolors=color,
            linewidths=1.4,
            alpha=0.98,
            zorder=6,
        )


def render_panel(
    ax: plt.Axes,
    projected_profiles: list[dict[str, Any]],
    palette: list[tuple[float, float, float]],
    title: str,
    limits: tuple[float, float, float, float],
    x_label: str,
    y_label: str,
    **layer_flags: bool,
) -> None:
    for profile, color in zip(projected_profiles, palette):
        draw_profile_layers(ax, profile, color, **layer_flags)
    xmin, xmax, ymin, ymax = limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)


def render_distribution_plot(
    projected_profiles: list[dict[str, Any]],
    output_path: Path,
    title_prefix: str,
    x_label: str,
    y_label: str,
) -> None:
    palette = sns.color_palette("tab10", n_colors=max(1, len(projected_profiles)))
    limits = compute_axis_limits(projected_profiles)
    title_profiles = "; ".join(profile["label"] for profile in projected_profiles)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    fig.suptitle(f"{title_prefix}: generated vs test-set GMM ({title_profiles})", fontsize=14)

    render_panel(
        axes[0, 0],
        projected_profiles,
        palette,
        "Test-set KDE, x-vectors, centers",
        limits,
        x_label,
        y_label,
        show_test_kde=True,
        show_test_xvectors=True,
        show_test_centers=True,
    )
    render_panel(
        axes[0, 1],
        projected_profiles,
        palette,
        "Generated KDE and centers",
        limits,
        x_label,
        y_label,
        show_generated_kde=True,
        show_generated_centers=True,
    )
    render_panel(
        axes[1, 0],
        projected_profiles,
        palette,
        "All layers",
        limits,
        x_label,
        y_label,
        show_test_kde=True,
        show_test_xvectors=True,
        show_test_centers=True,
        show_generated_kde=True,
        show_generated_centers=True,
    )
    render_panel(
        axes[1, 1],
        projected_profiles,
        palette,
        "Test-set x-vectors only",
        limits,
        x_label,
        y_label,
        show_test_xvectors=True,
    )

    profile_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=color, label=profile["label"])
        for profile, color in zip(projected_profiles, palette)
    ]
    layer_handles = [
        Line2D([0], [0], color="black", linewidth=6, alpha=0.20, label="Generated KDE fill"),
        Line2D([0], [0], color="black", linestyle="--", label="Test-set KDE contours"),
        Line2D([0], [0], marker=".", linestyle="None", color="black", label="Test x-vectors"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none", color="black", label="Generated centers"),
        Line2D([0], [0], marker="^", linestyle="None", markerfacecolor="none", color="black", label="Test-set centers"),
    ]
    fig.legend(handles=profile_handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=True, title="Profiles")
    fig.legend(handles=layer_handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=True, title="Layers")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_kdes(profiles: list[dict[str, Any]], output_path: Path, include_lda: bool = False) -> list[Path]:
    projected_pca, pca_x_label, pca_y_label = build_pca_projection(profiles)
    pca_output = prefixed_output_path(output_path, "PCA")
    render_distribution_plot(projected_pca, pca_output, "PCA", pca_x_label, pca_y_label)

    output_paths = [pca_output]
    if include_lda:
        projected_lda, lda_x_label, lda_y_label = build_lda_projection(profiles)
        lda_output = prefixed_output_path(output_path, "LDA")
        render_distribution_plot(projected_lda, lda_output, "LDA", lda_x_label, lda_y_label)
        output_paths.append(lda_output)
    return output_paths


def build_profile_data(
    condition: pd.Series,
    utterances: pd.DataFrame,
    xvectors: np.ndarray,
    gmm_lookup: dict[tuple[str, ...], pd.Series],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    key = condition_key(condition, PLOT_FIELDS)
    mask = condition_mask(utterances, condition, PLOT_FIELDS)
    selected_xvectors = xvectors[mask]
    if len(selected_xvectors) == 0:
        print(f"Skipping {dict(zip(PLOT_FIELDS, key))}: no matching test xvectors", flush=True)
        return None
    
    gmm_row = gmm_lookup.get(key)
    if gmm_row is None:
        raise KeyError(f"No generated GMM metadata found for condition={dict(zip(PLOT_FIELDS, key))}")

    gmm_path = Path(str(gmm_row["gmm_path"]))
    if not gmm_path.exists():
        raise FileNotFoundError(f"Missing generated GMM file: {gmm_path}")

    _, generated_pi, generated_mu, generated_sigma = load_gmm(gmm_path)
    generated_pi, generated_mu, generated_sigma = prune_gmm_components(
        generated_pi,
        generated_mu,
        generated_sigma,
    )
    generated_points = sample_gmm(
        (generated_pi, generated_mu, generated_sigma),
        args.samples,
        rng,
    )

    ground_truth_gmm = fit_ground_truth_gmm(
        selected_xvectors,
        num_components=args.ground_truth_components,
        seed=args.seed,
    )
    ground_truth_points, _ = ground_truth_gmm.sample(args.samples)

    print(f"Profile {condition_label(condition)}: matched {len(selected_xvectors)} test xvectors", flush=True)
    # print(f"Weights from the K gaussians = {list(generated_pi)}")
    print(f"Loaded generated GMM from {gmm_path}", flush=True)
    return {
        "label": condition_label(condition),
        "condition": condition,
        "generated_points": generated_points,
        "ground_truth_points": ground_truth_points,
        "generated_centers": generated_mu,
        "ground_truth_centers": ground_truth_gmm.means_,
        "test_xvectors": selected_xvectors,
        "num_test_xvectors": len(selected_xvectors),
        "gmm_path": gmm_path,
    }


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.ground_truth_components <= 0:
        raise ValueError("--ground-truth-components must be positive")
    rng = np.random.default_rng(args.seed)
    selection_rng = np.random.default_rng(args.seed)
    conditions = requested_conditions(args)
    utterances, xvectors = load_xvectors(
        args.test_csv_paths,
        args.xvector_root,
        profile_fields=PLOT_FIELDS,
        num_workers=args.num_load_workers,
        rng=selection_rng,
    )
    if len(xvectors) == 0:
        raise ValueError(f"No xvectors loaded from {args.test_csv_paths}")

    gmm_metadata = load_gmm_metadata(args.gmm_metadata_csv)
    gmm_lookup = {condition_key(row, PLOT_FIELDS): row for _, row in gmm_metadata.iterrows()}

    profiles = [
        profile
        for condition in conditions
        if (profile := build_profile_data(condition, utterances, xvectors, gmm_lookup, args, rng)) is not None
    ]
    if not profiles:
        raise ValueError("No requested profiles had matching test xvectors")
    print("Plotting KDEs")
    output_paths = plot_kdes(profiles, args.output, include_lda=args.lda)
    for output_path in output_paths:
        print(f"Wrote {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
