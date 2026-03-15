from __future__ import annotations

"""Distribution metrics for vector-valued samples.

Implemented metrics:
- Sliced Wasserstein Distance (SWD) on samples in R^D
- Energy Distance (U-statistic) on sample pairs
- RBF-MMD on raw vector samples
"""

from typing import Literal, Optional

import numpy as np
import ot


def _as_f32_contig(x: np.ndarray) -> np.ndarray:
    """Convert to contiguous float32 array."""
    x = np.asarray(x, dtype=np.float32)
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    return x


def _as_f64_contig(x: np.ndarray) -> np.ndarray:
    """Convert to contiguous float64 array."""
    x = np.asarray(x, dtype=np.float64)
    if not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    return x


def _validate_samples(samples: np.ndarray, name: str) -> np.ndarray:
    samples = _as_f32_contig(samples)
    if samples.ndim != 2:
        raise ValueError(f"{name} must have shape [N_samples, D], got {samples.shape}.")
    if samples.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    return samples


def _subsample_samples(samples: np.ndarray, max_samples: Optional[int], seed: int) -> np.ndarray:
    samples = _validate_samples(samples, "samples")
    if max_samples is None or max_samples >= samples.shape[0]:
        return samples
    if max_samples <= 0:
        raise ValueError("max_samples must be > 0 when provided.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(samples.shape[0], size=max_samples, replace=False)
    return samples[idx]


def _pairwise_sqeuclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    out = x2 + y2 - 2.0 * (x @ y.T)
    return np.maximum(out, 0.0)


def pairwise_sample_distance_matrix(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    *,
    distance: str = "l2",
    symmetric: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    samples_a = _validate_samples(samples_a, "samples_a")
    samples_b = _validate_samples(samples_b, "samples_b")
    if samples_a.shape[1] != samples_b.shape[1]:
        raise ValueError(f"D mismatch: samples_a D={samples_a.shape[1]} vs samples_b D={samples_b.shape[1]}")
    if distance not in {"l2", "l2_squared"}:
        raise ValueError(f"Unsupported sample distance: {distance}")

    na, nb = samples_a.shape[0], samples_b.shape[0]
    if symmetric and na != nb:
        raise ValueError("symmetric=True requires samples_a and samples_b to have the same number of samples.")

    dist2 = _pairwise_sqeuclidean(samples_a, samples_b)
    if distance == "l2_squared":
        out = dist2
    else:
        out = np.sqrt(dist2 + eps)

    if symmetric:
        np.fill_diagonal(out, 0.0)
    return out.astype(np.float32, copy=False)


def energy_distance_u_statistic_from_matrices(
    D_xx: np.ndarray,
    D_yy: np.ndarray,
    D_xy: np.ndarray,
) -> float:
    """Unbiased U-statistic estimator for Energy Distance."""
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


def energy_distance_u_statistic_samples(
    real_samples: np.ndarray,
    gen_samples: np.ndarray,
    *,
    distance: str = "l2",
    max_samples: Optional[int] = 256,
    seed: int = 0,
) -> float:
    """Energy Distance (U-statistic) between two vector-sample sets."""
    real_samples = _subsample_samples(real_samples, max_samples=max_samples, seed=seed)
    gen_samples = _subsample_samples(gen_samples, max_samples=max_samples, seed=seed + 1)

    D_xx = pairwise_sample_distance_matrix(
        real_samples,
        real_samples,
        distance=distance,
        symmetric=True,
    )
    D_yy = pairwise_sample_distance_matrix(
        gen_samples,
        gen_samples,
        distance=distance,
        symmetric=True,
    )
    D_xy = pairwise_sample_distance_matrix(
        real_samples,
        gen_samples,
        distance=distance,
        symmetric=False,
    )
    return energy_distance_u_statistic_from_matrices(D_xx, D_yy, D_xy)


def _resolve_standardize_mode(standardize_features: bool | str) -> Literal["none", "pooled", "reference"]:
    if isinstance(standardize_features, bool):
        return "pooled" if standardize_features else "none"

    mode = str(standardize_features).strip().lower()
    aliases = {
        "none": "none",
        "false": "none",
        "off": "none",
        "pooled": "pooled",
        "joint": "pooled",
        "true": "pooled",
        "reference": "reference",
        "real": "reference",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported standardize_features mode: {standardize_features}")
    return aliases[mode]


def _standardize_feature_pair(
    feat_real: np.ndarray,
    feat_gen: np.ndarray,
    *,
    mode: Literal["none", "pooled", "reference"],
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    feat_real = _as_f64_contig(feat_real)
    feat_gen = _as_f64_contig(feat_gen)

    if mode == "none":
        return feat_real, feat_gen

    stats_source = feat_real if mode == "reference" else np.concatenate([feat_real, feat_gen], axis=0)
    mu = stats_source.mean(axis=0, keepdims=True)
    sigma = stats_source.std(axis=0, keepdims=True)
    sigma = np.where(sigma < eps, 1.0, sigma)
    return (feat_real - mu) / sigma, (feat_gen - mu) / sigma


def _resolve_gamma(
    D_xx: np.ndarray,
    D_xy: np.ndarray,
    *,
    gamma: Optional[float],
    gamma_mode: str,
    feature_dim: int,
    eps: float,
) -> float:
    if gamma is not None:
        return float(gamma)

    mode = str(gamma_mode).strip().lower()
    if mode == "median_cross":
        positive = D_xy[D_xy > 0]
        if positive.size == 0:
            return 1.0
        return 1.0 / (float(np.median(positive)) + eps)

    if mode == "median_reference":
        upper = D_xx[np.triu_indices_from(D_xx, k=1)]
        positive = upper[upper > 0]
        if positive.size == 0:
            return 1.0
        return 1.0 / (float(np.median(positive)) + eps)

    if mode == "feature_dim":
        return 1.0 / float(max(feature_dim, 1))

    raise ValueError(f"Unsupported gamma_mode: {gamma_mode}")


def mmd_rbf_from_features(
    features_x: np.ndarray,
    features_y: np.ndarray,
    *,
    gamma: Optional[float] = None,
    gamma_mode: str = "median_cross",
    gamma_scale: float = 1.0,
    unbiased: bool = True,
    eps: float = 1e-12,
) -> float:
    """Compute RBF-MMD on feature vectors."""
    features_x = _as_f64_contig(features_x)
    features_y = _as_f64_contig(features_y)
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

    gamma_eff = _resolve_gamma(
        D_xx,
        D_xy,
        gamma=gamma,
        gamma_mode=gamma_mode,
        feature_dim=features_x.shape[1],
        eps=eps,
    )
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


def mmd_rbf_samples(
    real_samples: np.ndarray,
    gen_samples: np.ndarray,
    *,
    max_samples: Optional[int] = 512,
    seed: int = 0,
    gamma: Optional[float] = None,
    gamma_mode: str = "median_cross",
    gamma_scale: float = 1.0,
    unbiased: bool = True,
    standardize_features: bool | str = True,
    eig_eps: float = 1e-8,
) -> float:
    """RBF-MMD on raw vector samples."""
    real_samples = _subsample_samples(real_samples, max_samples=max_samples, seed=seed)
    gen_samples = _subsample_samples(gen_samples, max_samples=max_samples, seed=seed + 1)

    feat_real, feat_gen = _standardize_feature_pair(
        real_samples,
        gen_samples,
        mode=_resolve_standardize_mode(standardize_features),
        eps=eig_eps,
    )
    return mmd_rbf_from_features(
        feat_real,
        feat_gen,
        gamma=gamma,
        gamma_mode=gamma_mode,
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
    """Compute SWD between two point sets in R^D."""
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


def exact_discrete_w2_distance(
    real_samples: np.ndarray,
    gen_samples: np.ndarray,
    *,
    max_samples: Optional[int] = 1024,
    seed: int = 0,
) -> float:
    """Exact empirical W2 distance between two equally weighted sample sets."""
    real_samples = _subsample_samples(real_samples, max_samples=max_samples, seed=seed)
    gen_samples = _subsample_samples(gen_samples, max_samples=max_samples, seed=seed + 1)
    real_samples = _validate_samples(real_samples, "real_samples")
    gen_samples = _validate_samples(gen_samples, "gen_samples")

    if real_samples.shape[1] != gen_samples.shape[1]:
        raise ValueError(
            f"Dimension mismatch: real D={real_samples.shape[1]} vs gen D={gen_samples.shape[1]}"
        )

    n = real_samples.shape[0]
    m = gen_samples.shape[0]
    a = np.full(n, 1.0 / float(n), dtype=np.float64)
    b = np.full(m, 1.0 / float(m), dtype=np.float64)
    cost = _pairwise_sqeuclidean(_as_f64_contig(real_samples), _as_f64_contig(gen_samples))
    w2_sq = float(ot.emd2(a, b, cost))
    return float(np.sqrt(max(w2_sq, 0.0)))


__all__ = [
    "sliced_wasserstein_distance",
    "pairwise_sample_distance_matrix",
    "energy_distance_u_statistic_from_matrices",
    "energy_distance_u_statistic_samples",
    "mmd_rbf_from_features",
    "mmd_rbf_samples",
    "exact_discrete_w2_distance",
]
