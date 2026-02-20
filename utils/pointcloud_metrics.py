from __future__ import annotations

"""Point cloud metrics utilities.

Implemented metrics:
- Sliced Wasserstein Distance (SWD) on flattened points
- Energy Distance (U-statistic) on pairwise cloud distances
- RBF-MMD on invariant per-cloud feature vectors
"""

from typing import Optional

import numpy as np
import ot


def _as_f32_contig(x: np.ndarray) -> np.ndarray:
    """Convert to contiguous float32 array."""
    x = np.asarray(x, dtype=np.float32)
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    return x


def _validate_clouds(clouds: np.ndarray, name: str) -> np.ndarray:
    clouds = _as_f32_contig(clouds)
    if clouds.ndim != 3:
        raise ValueError(f"{name} must have shape [N_clouds, N_points, D], got {clouds.shape}.")
    if clouds.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one cloud.")
    if clouds.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two points per cloud.")
    return clouds


def _subsample_clouds(clouds: np.ndarray, max_clouds: Optional[int], seed: int) -> np.ndarray:
    clouds = _validate_clouds(clouds, "clouds")
    if max_clouds is None or max_clouds >= clouds.shape[0]:
        return clouds
    if max_clouds <= 0:
        raise ValueError("max_clouds must be > 0 when provided.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(clouds.shape[0], size=max_clouds, replace=False)
    return clouds[idx]


def _downsample_points(cloud: np.ndarray, num_points: Optional[int], rng: np.random.Generator) -> np.ndarray:
    if num_points is None or num_points >= cloud.shape[0]:
        return cloud
    if num_points <= 1:
        raise ValueError("downsample_points must be > 1 when provided.")
    idx = rng.choice(cloud.shape[0], size=num_points, replace=False)
    return cloud[idx]


def _pairwise_sqeuclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    out = x2 + y2 - 2.0 * (x @ y.T)
    return np.maximum(out, 0.0)


def chamfer_distance_l2(a: np.ndarray, b: np.ndarray, *, squared: bool = False, eps: float = 1e-12) -> float:
    """Symmetric Chamfer distance between two clouds.

    Args:
        a: Cloud with shape [Na, D]
        b: Cloud with shape [Nb, D]
        squared: If True, uses squared L2 costs. Otherwise uses L2 costs.
        eps: Numerical epsilon for sqrt stabilization.
    """
    a = _as_f32_contig(a)
    b = _as_f32_contig(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must have shape [N_points, D].")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Dimension mismatch: a D={a.shape[1]} vs b D={b.shape[1]}")

    dist2 = _pairwise_sqeuclidean(a, b)
    if squared:
        dist = dist2
    else:
        dist = np.sqrt(dist2 + eps)
    return float(np.min(dist, axis=1).mean() + np.min(dist, axis=0).mean())


def pairwise_cloud_distance_matrix(
    clouds_a: np.ndarray,
    clouds_b: np.ndarray,
    *,
    distance: str = "chamfer_l2",
    downsample_points: Optional[int] = None,
    seed: int = 0,
    symmetric: bool = False,
) -> np.ndarray:
    """Pairwise cloud-distance matrix.

    Currently supported distances:
    - ``chamfer_l2``
    - ``chamfer_l2_squared``
    """
    clouds_a = _validate_clouds(clouds_a, "clouds_a")
    clouds_b = _validate_clouds(clouds_b, "clouds_b")
    if clouds_a.shape[2] != clouds_b.shape[2]:
        raise ValueError(f"D mismatch: clouds_a D={clouds_a.shape[2]} vs clouds_b D={clouds_b.shape[2]}")

    if distance not in {"chamfer_l2", "chamfer_l2_squared"}:
        raise ValueError(f"Unsupported cloud distance: {distance}")

    rng = np.random.default_rng(seed)
    na, nb = clouds_a.shape[0], clouds_b.shape[0]
    if symmetric and na != nb:
        raise ValueError("symmetric=True requires clouds_a and clouds_b to have the same number of clouds.")
    out = np.zeros((na, nb), dtype=np.float32)

    a_ds = [_downsample_points(clouds_a[i], downsample_points, rng) for i in range(na)]
    if symmetric:
        b_ds = a_ds
    else:
        b_ds = [_downsample_points(clouds_b[j], downsample_points, rng) for j in range(nb)]

    squared = distance == "chamfer_l2_squared"

    if symmetric:
        for i in range(na):
            out[i, i] = 0.0
            for j in range(i + 1, nb):
                d = chamfer_distance_l2(a_ds[i], b_ds[j], squared=squared)
                out[i, j] = d
                out[j, i] = d
        return out

    for i in range(na):
        for j in range(nb):
            out[i, j] = chamfer_distance_l2(a_ds[i], b_ds[j], squared=squared)
    return out


def energy_distance_u_statistic_from_matrices(
    D_xx: np.ndarray,
    D_yy: np.ndarray,
    D_xy: np.ndarray,
) -> float:
    """Unbiased U-statistic estimator for Energy Distance.

    Computes:
        2 E[d(X,Y)] - E[d(X,X')] - E[d(Y,Y')]
    where within-sample expectations exclude diagonal terms.
    """
    D_xx = _as_f32_contig(D_xx)
    D_yy = _as_f32_contig(D_yy)
    D_xy = _as_f32_contig(D_xy)

    n = D_xx.shape[0]
    m = D_yy.shape[0]
    if D_xx.shape != (n, n) or D_yy.shape != (m, m) or D_xy.shape != (n, m):
        raise ValueError("Invalid matrix shapes for energy distance.")
    if n < 2 or m < 2:
        raise ValueError("U-statistic energy distance requires at least 2 samples in each set.")

    sum_xx_offdiag = float(D_xx.sum() - np.trace(D_xx))
    sum_yy_offdiag = float(D_yy.sum() - np.trace(D_yy))
    mean_xy = float(D_xy.mean())
    mean_xx_offdiag = sum_xx_offdiag / float(n * (n - 1))
    mean_yy_offdiag = sum_yy_offdiag / float(m * (m - 1))

    return float(2.0 * mean_xy - mean_xx_offdiag - mean_yy_offdiag)


def energy_distance_u_statistic_clouds(
    real_clouds: np.ndarray,
    gen_clouds: np.ndarray,
    *,
    distance: str = "chamfer_l2",
    max_clouds: Optional[int] = 256,
    downsample_points: Optional[int] = None,
    seed: int = 0,
) -> float:
    """Energy Distance (U-statistic) between two cloud sets."""
    real_clouds = _subsample_clouds(real_clouds, max_clouds=max_clouds, seed=seed)
    gen_clouds = _subsample_clouds(gen_clouds, max_clouds=max_clouds, seed=seed + 1)

    D_xx = pairwise_cloud_distance_matrix(
        real_clouds,
        real_clouds,
        distance=distance,
        downsample_points=downsample_points,
        seed=seed,
        symmetric=True,
    )
    D_yy = pairwise_cloud_distance_matrix(
        gen_clouds,
        gen_clouds,
        distance=distance,
        downsample_points=downsample_points,
        seed=seed + 1,
        symmetric=True,
    )
    D_xy = pairwise_cloud_distance_matrix(
        real_clouds,
        gen_clouds,
        distance=distance,
        downsample_points=downsample_points,
        seed=seed + 2,
        symmetric=False,
    )
    return energy_distance_u_statistic_from_matrices(D_xx, D_yy, D_xy)


def _radial_kurtosis_excess(x_centered: np.ndarray, eps: float = 1e-8) -> float:
    r = np.linalg.norm(x_centered, axis=1)
    r_std = float(r.std())
    if r_std < eps:
        return 0.0
    z = (r - r.mean()) / (r_std + eps)
    return float(np.mean(z**4) - 3.0)


def _local_surface_variation_curvature(cloud: np.ndarray, k_neighbors: int, eps: float = 1e-8) -> float:
    """Average local surface-variation curvature from kNN covariance spectra."""
    n, _ = cloud.shape
    if n <= 2:
        return 0.0

    k = int(max(2, min(k_neighbors, n - 1)))
    values = np.zeros(n, dtype=np.float32)
    for i in range(n):
        d2 = np.sum((cloud - cloud[i]) ** 2, axis=1)
        nn = np.argpartition(d2, k)[: (k + 1)]
        nn = nn[nn != i]
        if nn.size > k:
            nn = nn[:k]
        neigh = cloud[nn]
        yc = neigh - neigh.mean(axis=0, keepdims=True)
        cov = (yc.T @ yc) / float(max(yc.shape[0] - 1, 1))
        eigvals = np.linalg.eigvalsh(cov)
        tr = float(max(eigvals.sum(), 0.0))
        if tr <= eps:
            values[i] = 0.0
        else:
            values[i] = float(max(eigvals[0], 0.0) / (tr + eps))
    return float(values.mean())


def pointcloud_invariant_features(
    clouds: np.ndarray,
    *,
    include_centroid: bool = True,
    include_log_eigvals: bool = True,
    include_participation_ratio: bool = True,
    include_thickness: bool = True,
    include_kurtosis: bool = True,
    include_curvature: bool = True,
    thickness_tail_fraction: float = 0.25,
    curvature_k_neighbors: int = 16,
    eig_eps: float = 1e-8,
) -> np.ndarray:
    """Extract per-cloud feature vectors.

    Feature blocks (toggle-able):
    - centroid [D]
    - log-eigenvalue spectrum of covariance [D]
    - participation ratio [1]
    - thickness from covariance tail [1]
    - radial kurtosis excess [1]
    - local curvature statistic [1]
    """
    clouds = _validate_clouds(clouds, "clouds")
    if not (0.0 < thickness_tail_fraction <= 1.0):
        raise ValueError("thickness_tail_fraction must be in (0, 1].")
    if curvature_k_neighbors < 2:
        raise ValueError("curvature_k_neighbors must be >= 2.")

    feats = []
    for i in range(clouds.shape[0]):
        x = clouds[i]
        mu = x.mean(axis=0)
        xc = x - mu
        cov = (xc.T @ xc) / float(max(x.shape[0] - 1, 1))
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals[::-1], 0.0)  # descending

        blocks = []
        if include_centroid:
            blocks.append(mu.astype(np.float32, copy=False))
        if include_log_eigvals:
            blocks.append(np.log(eigvals + eig_eps).astype(np.float32, copy=False))
        if include_participation_ratio:
            num = float(eigvals.sum() ** 2)
            den = float(np.sum(eigvals**2) + eig_eps)
            pr = num / den
            blocks.append(np.asarray([pr], dtype=np.float32))
        if include_thickness:
            tail_k = max(1, int(round(eigvals.shape[0] * thickness_tail_fraction)))
            thickness = float(np.sqrt(np.mean(eigvals[-tail_k:]) + eig_eps))
            blocks.append(np.asarray([thickness], dtype=np.float32))
        if include_kurtosis:
            blocks.append(np.asarray([_radial_kurtosis_excess(xc, eps=eig_eps)], dtype=np.float32))
        if include_curvature:
            curv = _local_surface_variation_curvature(x, k_neighbors=curvature_k_neighbors, eps=eig_eps)
            blocks.append(np.asarray([curv], dtype=np.float32))

        if not blocks:
            raise ValueError("At least one feature block must be enabled for invariant feature extraction.")
        feats.append(np.concatenate(blocks, axis=0))

    return _as_f32_contig(np.stack(feats, axis=0))


def mmd_rbf_from_features(
    features_x: np.ndarray,
    features_y: np.ndarray,
    *,
    gamma: Optional[float] = None,
    gamma_scale: float = 1.0,
    unbiased: bool = True,
    eps: float = 1e-12,
) -> float:
    """Compute RBF-MMD on feature vectors."""
    features_x = _as_f32_contig(features_x)
    features_y = _as_f32_contig(features_y)
    if features_x.ndim != 2 or features_y.ndim != 2:
        raise ValueError("features_x and features_y must have shape [N, F] and [M, F].")
    if features_x.shape[1] != features_y.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: x F={features_x.shape[1]} vs y F={features_y.shape[1]}"
        )
    if gamma_scale <= 0:
        raise ValueError("gamma_scale must be > 0.")

    D_xx = _pairwise_sqeuclidean(features_x, features_x)
    D_yy = _pairwise_sqeuclidean(features_y, features_y)
    D_xy = _pairwise_sqeuclidean(features_x, features_y)

    if gamma is None:
        positive = D_xy[D_xy > 0]
        if positive.size == 0:
            gamma_eff = 1.0
        else:
            gamma_eff = 1.0 / (float(np.median(positive)) + eps)
    else:
        gamma_eff = float(gamma)
    gamma_eff *= float(gamma_scale)

    K_xx = np.exp(-gamma_eff * D_xx)
    K_yy = np.exp(-gamma_eff * D_yy)
    K_xy = np.exp(-gamma_eff * D_xy)

    n = features_x.shape[0]
    m = features_y.shape[0]

    if unbiased:
        if n < 2 or m < 2:
            raise ValueError("Unbiased MMD requires at least 2 samples per set.")
        mean_xx = float((K_xx.sum() - np.trace(K_xx)) / (n * (n - 1)))
        mean_yy = float((K_yy.sum() - np.trace(K_yy)) / (m * (m - 1)))
        mean_xy = float(K_xy.mean())
        return float(mean_xx + mean_yy - 2.0 * mean_xy)

    return float(K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean())


def mmd_rbf_invariant_features(
    real_clouds: np.ndarray,
    gen_clouds: np.ndarray,
    *,
    max_clouds: Optional[int] = 512,
    seed: int = 0,
    gamma: Optional[float] = None,
    gamma_scale: float = 1.0,
    unbiased: bool = True,
    standardize_features: bool = True,
    include_centroid: bool = True,
    include_log_eigvals: bool = True,
    include_participation_ratio: bool = True,
    include_thickness: bool = True,
    include_kurtosis: bool = True,
    include_curvature: bool = True,
    thickness_tail_fraction: float = 0.25,
    curvature_k_neighbors: int = 16,
    eig_eps: float = 1e-8,
) -> float:
    """RBF-MMD on invariant per-cloud features."""
    real_clouds = _subsample_clouds(real_clouds, max_clouds=max_clouds, seed=seed)
    gen_clouds = _subsample_clouds(gen_clouds, max_clouds=max_clouds, seed=seed + 1)

    feat_real = pointcloud_invariant_features(
        real_clouds,
        include_centroid=include_centroid,
        include_log_eigvals=include_log_eigvals,
        include_participation_ratio=include_participation_ratio,
        include_thickness=include_thickness,
        include_kurtosis=include_kurtosis,
        include_curvature=include_curvature,
        thickness_tail_fraction=thickness_tail_fraction,
        curvature_k_neighbors=curvature_k_neighbors,
        eig_eps=eig_eps,
    )
    feat_gen = pointcloud_invariant_features(
        gen_clouds,
        include_centroid=include_centroid,
        include_log_eigvals=include_log_eigvals,
        include_participation_ratio=include_participation_ratio,
        include_thickness=include_thickness,
        include_kurtosis=include_kurtosis,
        include_curvature=include_curvature,
        thickness_tail_fraction=thickness_tail_fraction,
        curvature_k_neighbors=curvature_k_neighbors,
        eig_eps=eig_eps,
    )

    if standardize_features:
        both = np.concatenate([feat_real, feat_gen], axis=0)
        mu = both.mean(axis=0, keepdims=True)
        sigma = both.std(axis=0, keepdims=True)
        sigma = np.where(sigma < eig_eps, 1.0, sigma)
        feat_real = (feat_real - mu) / sigma
        feat_gen = (feat_gen - mu) / sigma

    return mmd_rbf_from_features(
        feat_real,
        feat_gen,
        gamma=gamma,
        gamma_scale=gamma_scale,
        unbiased=unbiased,
        eps=eig_eps,
    )


def sliced_wasserstein_distance(
    real_points: np.ndarray,
    gen_points: np.ndarray,
    num_projections: int = 256,
    seed: int = 0,
) -> float:
    """Compute SWD between two flattened point sets.

    Args:
        real_points: Real points with shape [N, D]
        gen_points: Generated points with shape [M, D]
        num_projections: Number of random 1D projections
        seed: Random seed for projection sampling

    Returns:
        Sliced Wasserstein distance.
    """
    real_points = _as_f32_contig(real_points)
    gen_points = _as_f32_contig(gen_points)

    if real_points.ndim != 2 or gen_points.ndim != 2:
        raise ValueError("real_points and gen_points must have shape [N, D] and [M, D].")
    if real_points.shape[1] != gen_points.shape[1]:
        raise ValueError(
            f"Dimension mismatch: real D={real_points.shape[1]} vs gen D={gen_points.shape[1]}"
        )
    if num_projections <= 0:
        raise ValueError("num_projections must be > 0")

    return float(
        ot.sliced.sliced_wasserstein_distance(
            real_points,
            gen_points,
            n_projections=num_projections,
            seed=seed,
        )
    )


__all__ = [
    "sliced_wasserstein_distance",
    "chamfer_distance_l2",
    "pairwise_cloud_distance_matrix",
    "energy_distance_u_statistic_from_matrices",
    "energy_distance_u_statistic_clouds",
    "pointcloud_invariant_features",
    "mmd_rbf_from_features",
    "mmd_rbf_invariant_features",
]
