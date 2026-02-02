from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import ot
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import pairwise_distances


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Symmetric Chamfer distance between two point clouds.
    a, b: [N, D] arrays.
    Returns mean squared distance (a->b + b->a).
    """
    tree_a = cKDTree(a)
    tree_b = cKDTree(b)
    dist_a, _ = tree_b.query(a, k=1)
    dist_b, _ = tree_a.query(b, k=1)
    return float(np.mean(dist_a ** 2) + np.mean(dist_b ** 2))


def pairwise_chamfer_matrix(clouds_a: np.ndarray, clouds_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Chamfer distance matrix between two sets of clouds.
    clouds_a: [Na, N, D]
    clouds_b: [Nb, N, D]
    Returns: [Na, Nb] matrix.
    """
    if clouds_a.ndim != 3 or clouds_b.ndim != 3:
        raise ValueError("clouds_a/clouds_b must have shape [Nclouds, Npoints, D].")

    n_points = clouds_a.shape[1]
    d = clouds_a.shape[2]

    def _metric(x_flat: np.ndarray, y_flat: np.ndarray) -> float:
        x = x_flat.reshape(n_points, d)
        y = y_flat.reshape(n_points, d)
        return chamfer_distance(x, y)

    a_flat = clouds_a.reshape(clouds_a.shape[0], -1)
    b_flat = clouds_b.reshape(clouds_b.shape[0], -1)
    return pairwise_distances(a_flat, b_flat, metric=_metric)


def sliced_wasserstein_distance(
    real_points: np.ndarray,
    gen_points: np.ndarray,
    num_projections: int = 256,
    seed: int = 0,
) -> float:
    """
    SWD over point distributions using random projections.
    real_points/gen_points: [P, D] arrays.
    """
    return float(
        ot.sliced.sliced_wasserstein_distance(
            real_points,
            gen_points,
            n_projections=num_projections,
            seed=seed,
        )
    )


def mmd_rbf_from_distance_matrices(
    D_xx: np.ndarray,
    D_yy: np.ndarray,
    D_xy: np.ndarray,
    gamma: float,
) -> float:
    # Exact RBF on distances: K = exp(-gamma * D)
    K_xx = np.exp(-gamma * D_xx)
    K_yy = np.exp(-gamma * D_yy)
    K_xy = np.exp(-gamma * D_xy)
    return float(K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean())


def mmd_rbf_from_chamfer(
    real_clouds: np.ndarray,
    gen_clouds: np.ndarray,
    gamma: float,
) -> float:
    """
    MMD with an RBF kernel applied to Chamfer distances between point clouds.
    real_clouds/gen_clouds: [Nclouds, Npoints, D]
    """
    D_xx = pairwise_chamfer_matrix(real_clouds, real_clouds)
    D_yy = pairwise_chamfer_matrix(gen_clouds, gen_clouds)
    D_xy = pairwise_chamfer_matrix(real_clouds, gen_clouds)
    return mmd_rbf_from_distance_matrices(D_xx, D_yy, D_xy, gamma)
