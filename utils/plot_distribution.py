# plot_synth_pc.py
# Usage:
#   python plot_synth_pc.py dataset=synth_pc plot.max_points_per_class=20000
#
# Expects your dataset config under conf/data/synth_pc.yaml
# and a conf/plot.yaml with defaults including data=synth_pc.

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader
try:
    from utils.colorful_logger import *
except ModuleNotFoundError:
    from colorful_logger import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datamodules.synthetic_pointclouds import collate_pointclouds


@dataclass
class PlotConfig:
    # How many points (total) to aggregate per class before plotting KDE
    max_points_per_class: int = 20000

    # How many point-cloud samples to scan at most (safety stop)
    max_batches: int = 2000

    # Projection to 2D: "pca" uses global PCA across all collected points.
    # "dims" uses the first two coordinates (dim0, dim1).
    projection: str = "pca"  # "pca" or "dims"
    dim0: int = 0
    dim1: int = 1

    # KDE rendering
    kde_levels: int = 80
    kde_thresh: float = 0.0  # show all
    cmap: str = "rocket"
    num_workers: int = 0

    # Output
    out_name: str = "synth_pointclouds_kde.png"
    dpi: int = 200


def pca_2d_global(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X: [P, D] float tensor on CPU.
    Returns: (Z, W) where Z is [P,2], W is [D,2].
    Deterministic given X.
    """
    Xc = X - X.mean(dim=0, keepdim=True)
    # SVD: Xc = U S V^T, principal directions are V[:, :2]
    # torch.linalg.svd is deterministic for fixed input on CPU.
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    W = Vt.T[:, :2]  # [D,2]
    Z = Xc @ W       # [P,2]
    return Z, W


def ensure_plot_cfg(cfg: DictConfig) -> PlotConfig:
    if "plot" not in cfg:
        return PlotConfig()
    # Merge OmegaConf -> dataclass defaults
    base = OmegaConf.structured(PlotConfig())
    merged = OmegaConf.merge(base, cfg.plot)
    return PlotConfig(**OmegaConf.to_container(merged, resolve=True))


def _plot_classes(
    ds,
    plot_cfg: PlotConfig,
    class_ids,
    out_name: str,
):
    # Collect points per class
    points_by_class: Dict[int, List[torch.Tensor]] = {y: [] for y in class_ids}
    counts: Dict[int, int] = {y: 0 for y in class_ids}

    def class_full(y: int) -> bool:
        return counts[y] >= plot_cfg.max_points_per_class

    all_full = lambda: all(class_full(y) for y in class_ids)

    dl = DataLoader(
        ds,
        batch_size=32,
        shuffle=True,
        num_workers=plot_cfg.num_workers,
        collate_fn=collate_pointclouds,
        drop_last=False,
    )

    device = "cpu"  # plotting pipeline stays on CPU
    batches_seen = 0

    for xb, yb in dl:
        batches_seen += 1
        if batches_seen > plot_cfg.max_batches or all_full():
            break

        xb = xb.to(device)  # [B, N, D]
        yb = yb.to(device)  # [B]

        B, N, D = xb.shape
        for i in range(B):
            y = int(yb[i].item())
            if y not in points_by_class or class_full(y):
                continue

            x_i = xb[i]  # [N, D]
            remaining = plot_cfg.max_points_per_class - counts[y]
            take = min(remaining, N)
            points_by_class[y].append(x_i[:take].contiguous())
            counts[y] += take

        if batches_seen % 50 == 0:
            info(f"Collected points per class: {counts}")

    # Concatenate per class
    Xc: Dict[int, torch.Tensor] = {}
    for y in class_ids:
        if len(points_by_class[y]) == 0:
            raise RuntimeError(f"No points collected for class {y}. Check your dataset sampling / labels.")
        Xc[y] = torch.cat(points_by_class[y], dim=0).to(torch.float32)  # [P, D]

    # Build a single global matrix for PCA (same axes across classes)
    if plot_cfg.projection == "pca":
        X_all = torch.cat([Xc[y] for y in class_ids], dim=0)
        Z_all, W = pca_2d_global(X_all)

        # Split back per class
        Zc: Dict[int, np.ndarray] = {}
        offset = 0
        for y in class_ids:
            P = Xc[y].shape[0]
            Z = Z_all[offset: offset + P]
            Zc[y] = Z.numpy()
            offset += P
    elif plot_cfg.projection == "dims":
        Zc = {}
        for y in class_ids:
            Z = Xc[y][:, [plot_cfg.dim0, plot_cfg.dim1]]
            Zc[y] = Z.numpy()
    else:
        raise ValueError("plot.projection must be 'pca' or 'dims'")

    # Plot: one subplot per class, KDE colored by density (viridis)
    sns.set_theme(style="white")
    C = len(class_ids)
    ncols = min(3, C)
    nrows = int(math.ceil(C / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 4.6 * nrows), squeeze=False)

    for idx, y in enumerate(class_ids):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        xy = Zc[y]
        x = xy[:, 0]
        y2 = xy[:, 1]

        sns.kdeplot(
            x=x,
            y=y2,
            fill=True,
            levels=plot_cfg.kde_levels,
            thresh=plot_cfg.kde_thresh,
            cmap=plot_cfg.cmap,
            ax=ax,
        )

        # Title with key knobs for that class (if available)
        params = ds.class_params[int(class_ids[idx])]
        ax.set_title(
            f"class {class_ids[idx]} | fam={params.family} | d={params.d}, D={params.D}, K={params.K}\n"
            f"sep={params.separation}, thick={params.thickness}, tail={params.tail.kind}, "
            f"anis={params.anisotropy.enabled}, curv={params.curvature.enabled}"
        )
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_aspect("equal", adjustable="box")

    # Hide unused axes
    for j in range(C, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    out_dir = os.path.dirname(out_name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.tight_layout()
    fig.savefig(out_name, dpi=plot_cfg.dpi, bbox_inches="tight")
    ok(f"Saved: {out_name}")


@hydra.main(config_path="../conf", config_name="plot", version_base=None)
def main(cfg: DictConfig):
    info(OmegaConf.to_yaml(cfg))

    plot_cfg = ensure_plot_cfg(cfg)

    ds = hydra.utils.instantiate(cfg.data)

    # Collect points per class
    class_ids = sorted(ds.class_params.keys()) if hasattr(ds, "class_params") else None
    if class_ids is None:
        raise RuntimeError("Dataset must expose .class_params (dict y->params) for per-class plotting.")

    splits = getattr(ds, "class_splits", {})
    if splits:
        base, ext = plot_cfg.out_name.rsplit(".", 1)
        for name, ids in splits.items():
            _plot_classes(ds, plot_cfg, ids, f"{base}_{name}.{ext}")
    else:
        _plot_classes(ds, plot_cfg, class_ids, plot_cfg.out_name)


if __name__ == "__main__":
    main()
