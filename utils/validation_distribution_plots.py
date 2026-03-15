from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde


@dataclass
class DistributionPlotConfig:
    enabled: bool = True
    splits: tuple[str, ...] = ("val",)
    every_n_epochs: int = 1
    max_classes: int = 6
    max_points_per_class: int = 10000
    projection: str = "pca"
    dim0: int = 0
    dim1: int = 1
    kde_levels: int = 70
    kde_thresh: float = 0.02
    min_points_for_kde: int = 32
    density_cmap: str = "Blues"
    scatter_alpha: float = 0.35
    dpi: int = 200
    seed: int = 0
    grid_size: int = 160
    shared_density_scale: bool = True

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "DistributionPlotConfig":
        if mapping is None:
            return cls()

        payload = dict(mapping)
        splits = payload.get("splits", cls.splits)
        if isinstance(splits, str):
            payload["splits"] = (splits,)
        else:
            payload["splits"] = tuple(str(split) for split in splits)
        return cls(**payload)

    def supports_split(self, split: str) -> bool:
        return self.enabled and split in set(self.splits)


def _subsample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[np.sort(indices)]


def _project_points(
    arrays: list[np.ndarray],
    cfg: DistributionPlotConfig,
) -> list[np.ndarray]:
    if not arrays:
        return []

    feature_dim = int(arrays[0].shape[1])
    if feature_dim < 2:
        padded = [np.pad(arr, ((0, 0), (0, 2 - feature_dim))) for arr in arrays]
        return padded

    if cfg.projection == "dims":
        dims = [int(cfg.dim0), int(cfg.dim1)]
        return [arr[:, dims] for arr in arrays]

    if cfg.projection != "pca":
        raise ValueError(f"Unsupported projection mode: {cfg.projection}")

    joined = np.concatenate(arrays, axis=0)
    centered = joined - joined.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2].T

    projected: list[np.ndarray] = []
    offset = 0
    for arr in arrays:
        chunk = centered[offset : offset + arr.shape[0]]
        projected.append(chunk @ basis)
        offset += arr.shape[0]
    return projected


def _format_class_summary(class_id: int, class_params: Any | None) -> str:
    if class_params is None:
        return f"class {class_id}"

    family = getattr(class_params, "family", "unknown")
    intrinsic_dim = getattr(class_params, "d", "?")
    ambient_dim = getattr(class_params, "D", "?")
    thickness = getattr(class_params, "thickness", "?")
    anisotropy_cfg = getattr(class_params, "anisotropy", None)
    anis_max = getattr(anisotropy_cfg, "max_scale", 1.0) if anisotropy_cfg is not None else 1.0
    return (
        f"class {class_id} | fam={family} | d={intrinsic_dim}, D={ambient_dim}\n"
        f"thickness={thickness}, anis_max={anis_max}"
    )


def _plot_density_panel(
    ax: Any,
    points: np.ndarray,
    *,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    density: np.ndarray | None,
    levels: np.ndarray | None,
    cfg: DistributionPlotConfig,
    title: str,
):
    x = points[:, 0]
    y = points[:, 1]

    if density is not None and levels is not None:
        ax.contourf(
            grid_x,
            grid_y,
            density,
            levels=levels,
            cmap=cfg.density_cmap,
            antialiased=True,
        )
    else:
        ax.scatter(x, y, s=10, alpha=cfg.scatter_alpha, color=sns.color_palette(cfg.density_cmap, 3)[-1])

    ax.set_title(title)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_aspect("equal", adjustable="box")


def _estimate_density_on_grid(
    points: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    cfg: DistributionPlotConfig,
) -> np.ndarray | None:
    if (
        points.shape[0] < cfg.min_points_for_kde
        or float(np.std(points[:, 0])) <= 0.0
        or float(np.std(points[:, 1])) <= 0.0
    ):
        return None

    try:
        kde = gaussian_kde(points.T)
    except (np.linalg.LinAlgError, ValueError):
        return None

    stacked = np.vstack([grid_x.ravel(), grid_y.ravel()])
    return kde(stacked).reshape(grid_x.shape)


def _build_pair_density_grids(
    real_proj: np.ndarray,
    gen_proj: np.ndarray,
    cfg: DistributionPlotConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    combined = np.concatenate([real_proj, gen_proj], axis=0)
    xmin, ymin = combined.min(axis=0)
    xmax, ymax = combined.max(axis=0)
    xpad = max((xmax - xmin) * 0.08, 1e-3)
    ypad = max((ymax - ymin) * 0.08, 1e-3)

    xs = np.linspace(xmin - xpad, xmax + xpad, cfg.grid_size)
    ys = np.linspace(ymin - ypad, ymax + ypad, cfg.grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys)

    real_density = _estimate_density_on_grid(real_proj, grid_x, grid_y, cfg)
    gen_density = _estimate_density_on_grid(gen_proj, grid_x, grid_y, cfg)
    density_candidates = [density for density in (real_density, gen_density) if density is not None]

    if not density_candidates:
        return grid_x, grid_y, None, None, None, None

    if cfg.shared_density_scale:
        max_density = max(float(density.max()) for density in density_candidates)
        min_density = max(max_density * cfg.kde_thresh, 1e-12)
        levels = np.linspace(min_density, max_density, cfg.kde_levels)
        return grid_x, grid_y, real_density, gen_density, levels, levels

    real_levels = None
    if real_density is not None:
        real_max = float(real_density.max())
        real_min = max(real_max * cfg.kde_thresh, 1e-12)
        real_levels = np.linspace(real_min, real_max, cfg.kde_levels)

    gen_levels = None
    if gen_density is not None:
        gen_max = float(gen_density.max())
        gen_min = max(gen_max * cfg.kde_thresh, 1e-12)
        gen_levels = np.linspace(gen_min, gen_max, cfg.kde_levels)

    return grid_x, grid_y, real_density, gen_density, real_levels, gen_levels


def save_distribution_comparison_plot(
    *,
    real_samples: dict[int, np.ndarray],
    generated_samples: dict[int, np.ndarray],
    class_params_by_id: dict[int, Any] | None,
    output_path: str | Path,
    split: str,
    step: int,
    epoch: int,
    cfg: DistributionPlotConfig,
) -> dict[str, Any]:
    class_ids = sorted(set(real_samples.keys()) & set(generated_samples.keys()))
    class_ids = class_ids[: cfg.max_classes]
    if not class_ids:
        raise ValueError("No overlapping class ids between real and generated samples.")

    rng = np.random.default_rng(cfg.seed + step)
    sampled_real: dict[int, np.ndarray] = {}
    sampled_generated: dict[int, np.ndarray] = {}
    arrays_for_projection: list[np.ndarray] = []

    for class_id in class_ids:
        sampled_real[class_id] = _subsample_points(real_samples[class_id], cfg.max_points_per_class, rng)
        sampled_generated[class_id] = _subsample_points(
            generated_samples[class_id],
            cfg.max_points_per_class,
            rng,
        )
        arrays_for_projection.extend([sampled_real[class_id], sampled_generated[class_id]])

    projected_arrays = _project_points(arrays_for_projection, cfg)
    projected_real: dict[int, np.ndarray] = {}
    projected_generated: dict[int, np.ndarray] = {}
    for idx, class_id in enumerate(class_ids):
        projected_real[class_id] = projected_arrays[2 * idx]
        projected_generated[class_id] = projected_arrays[2 * idx + 1]

    sns.set_theme(style="white")
    fig, axes = plt.subplots(
        nrows=len(class_ids),
        ncols=2,
        figsize=(11, max(4.2 * len(class_ids), 4.2)),
        squeeze=False,
    )

    for row_idx, class_id in enumerate(class_ids):
        real_ax = axes[row_idx][0]
        gen_ax = axes[row_idx][1]
        real_proj = projected_real[class_id]
        gen_proj = projected_generated[class_id]

        class_params = None if class_params_by_id is None else class_params_by_id.get(class_id)
        summary = _format_class_summary(class_id, class_params)
        grid_x, grid_y, real_density, gen_density, real_levels, gen_levels = _build_pair_density_grids(
            real_proj,
            gen_proj,
            cfg,
        )

        _plot_density_panel(
            real_ax,
            real_proj,
            grid_x=grid_x,
            grid_y=grid_y,
            density=real_density,
            levels=real_levels,
            cfg=cfg,
            title=f"Ground Truth\n{summary}",
        )
        _plot_density_panel(
            gen_ax,
            gen_proj,
            grid_x=grid_x,
            grid_y=grid_y,
            density=gen_density,
            levels=gen_levels,
            cfg=cfg,
            title=f"Generated\n{summary}",
        )

        for ax in (real_ax, gen_ax):
            ax.set_xlim(float(grid_x.min()), float(grid_x.max()))
            ax.set_ylim(float(grid_y.min()), float(grid_y.max()))

    fig.suptitle(
        f"{split.upper()} distribution comparison | step={step} | epoch={epoch}",
        y=0.995,
        fontsize=14,
    )
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "output_path": str(output_path),
        "class_ids": class_ids,
        "num_classes_plotted": len(class_ids),
        "projection": cfg.projection,
        "step": int(step),
        "epoch": int(epoch),
        "split": split,
    }
