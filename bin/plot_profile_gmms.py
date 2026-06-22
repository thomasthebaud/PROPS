#!/usr/bin/env python3
"""Plot generated and ground-truth profile GMMs with PCA."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture

from utils import (
    DEFAULT_TEST_DATASETS,
    condition_mask,
    generated_gmm_matches,
    load_gmm_metadata,
    load_xvectors,
    normalize_value,
    sample_gmm_rows,
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
        normalize_value(condition[field])
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
    y_label = "LDA - dimension 2"
    return split_projection(projected, blocks, profiles), "LDA - dimension 1", y_label



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
    show_generated_kde: bool = False,
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

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)

    render_panel(
        axes[0, 0],
        projected_profiles,
        palette,
        "Test-set KDE",
        limits,
        x_label,
        y_label,
        show_test_kde=True,
    )
    render_panel(
        axes[0, 1],
        projected_profiles,
        palette,
        "Generated KDE",
        limits,
        x_label,
        y_label,
        show_generated_kde=True,
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
        show_generated_kde=True,
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
    fig.legend(
        handles=profile_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=max(1, len(profile_handles)),
        frameon=True,
        title="Profiles",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    render_all_layers_plot(projected_profiles, output_path, x_label, y_label)


def all_layers_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_all_layers{output_path.suffix}")


def render_all_layers_plot(
    projected_profiles: list[dict[str, Any]],
    output_path: Path,
    x_label: str,
    y_label: str,
) -> Path:
    palette = sns.color_palette("tab10", n_colors=max(1, len(projected_profiles)))
    limits = compute_axis_limits(projected_profiles)
    all_layers_path = all_layers_output_path(output_path)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(8.8, 7.0))
    render_panel(
        ax,
        projected_profiles,
        palette,
        "All layers",
        limits,
        x_label,
        y_label,
        show_test_kde=True,
        show_test_xvectors=True,
        show_generated_kde=True,
    )

    profile_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=color, label=profile["label"])
        for profile, color in zip(projected_profiles, palette)
    ]
    layer_handles = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.9, label="Test-set KDE"),
        Patch(facecolor="black", alpha=0.20, label="Generated KDE"),
        Line2D([0], [0], marker=".", linestyle="None", color="black", markersize=9, label="Test-set x-vectors"),
    ]
    layer_legend = ax.legend(handles=layer_handles, loc="upper right", frameon=True, title="Layers")
    ax.add_artist(layer_legend)
    ax.legend(
        handles=profile_handles,
        loc="lower right",
        ncol=1,
        frameon=True,
        title="Profiles",
    )
    fig.tight_layout()

    all_layers_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(all_layers_path, dpi=220)
    plt.close(fig)
    print(f"Wrote {all_layers_path}", flush=True)
    return all_layers_path


def draw_histogram_layers(
    ax: plt.Axes,
    projected_profiles: list[dict[str, Any]],
    palette: list[tuple[float, float, float]],
    value_key: str,
    title: str,
    x_label: str,
) -> None:
    for profile, color in zip(projected_profiles, palette):
        values = np.asarray(profile[value_key])[:, 0]
        ax.hist(
            values,
            bins=40,
            density=True,
            alpha=0.28,
            color=color,
            label=f"{profile['label']} (n={len(values)})",
        )
        if len(values) >= 3:
            sns.kdeplot(x=values, ax=ax, color=color, linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend()


def render_two_class_histogram_plot(
    projected_profiles: list[dict[str, Any]],
    output_path: Path,
    x_label: str,
    title_prefix: str,
) -> None:
    palette = sns.color_palette("tab10", n_colors=2)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.0), sharex=True)

    draw_histogram_layers(
        axes[0],
        projected_profiles,
        palette,
        "test_xvectors",
        "Test-set x-vector distributions",
        x_label,
    )
    draw_histogram_layers(
        axes[1],
        projected_profiles,
        palette,
        "generated_points",
        "Generated GMM sample distributions",
        x_label,
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_kdes(profiles: list[dict[str, Any]], output_path: Path, include_lda: bool = False) -> list[Path]:
    if include_lda:
        projected_lda, lda_x_label, lda_y_label = build_lda_projection(profiles)
        if len(projected_lda) == 2:
            render_two_class_histogram_plot(projected_lda, output_path, lda_x_label, "LDA")
        else:
            render_distribution_plot(projected_lda, output_path, "LDA", lda_x_label, lda_y_label)
        output_paths = [output_path]

        projected_pca, pca_x_label, pca_y_label = build_pca_projection(profiles)
        pca_output = prefixed_output_path(output_path, "PCA")
        if len(projected_pca) == 2:
            render_two_class_histogram_plot(projected_pca, pca_output, pca_x_label, "PCA")
        else:
            render_distribution_plot(projected_pca, pca_output, "PCA", pca_x_label, pca_y_label)
        output_paths.append(pca_output)
        return output_paths

    projected_pca, pca_x_label, pca_y_label = build_pca_projection(profiles)
    if len(projected_pca) == 2:
        render_two_class_histogram_plot(projected_pca, output_path, pca_x_label, "PCA")
    else:
        render_distribution_plot(projected_pca, output_path, "PCA", pca_x_label, pca_y_label)
    return [output_path]


def build_profile_data(
    condition: pd.Series,
    utterances: pd.DataFrame,
    xvectors: np.ndarray,
    gmm_metadata: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    condition_dict = {field: normalize_value(condition[field]) for field in PLOT_FIELDS}
    mask = condition_mask(utterances, condition, PLOT_FIELDS)
    selected_xvectors = xvectors[mask]
    if len(selected_xvectors) == 0:
        print(f"Skipping {condition_dict}: no matching test xvectors", flush=True)
        return None

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
    print(f"Sampled N={args.samples} from {len(gmm_rows)} matching generated GMMs (first sampled: {first_gmm_path})", flush=True)
    return {
        "label": condition_label(condition),
        "condition": condition,
        "generated_points": generated_points,
        "ground_truth_points": ground_truth_points,
        "test_xvectors": selected_xvectors,
        "num_test_xvectors": len(selected_xvectors),
        "gmm_path": first_gmm_path,
        "num_generated_gmms": len(gmm_rows),
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

    profiles = [
        profile
        for condition in conditions
        if (profile := build_profile_data(condition, utterances, xvectors, gmm_metadata, args, rng)) is not None
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
