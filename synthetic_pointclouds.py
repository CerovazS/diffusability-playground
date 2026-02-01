from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal, Tuple, Any

import math
import torch
from torch.utils.data import Dataset


# Config dataclasses (Hydra)

BaseDist = Literal["gauss", "laplace", "student_t", "cauchy_trunc"]
GeomFamily = Literal["affine_subspace", "sine_warp_subspace", "mog"]


@dataclass
class TailConfig:
    """Controls base distribution in intrinsic coordinates z (tail-heaviness)."""
    kind: BaseDist = "gauss"
    # student-t
    student_df: float = 4.0
    # cauchy_trunc: we sample Cauchy and clamp to [-cauchy_clip, cauchy_clip]
    cauchy_clip: float = 10.0


@dataclass
class AnisotropyConfig:
    """Controls per-intrinsic-dimension scaling s (independent of d and D)."""
    enabled: bool = False
    # If enabled, scales are log-spaced between min_scale and max_scale across d dims.
    min_scale: float = 0.5
    max_scale: float = 2.0
    # Optional: if True, randomly permute scales for each component k (keeps distribution but changes alignment).
    permute_per_mode: bool = False


@dataclass
class CurvatureConfig:
    """Controls curvature for sine-warp family."""
    enabled: bool = False
    alpha: float = 0.5   # amplitude
    freq: float = 2.0    # frequency


@dataclass
class ClassParams:
    """
    Per-class parameters (label y has its own set of knobs).
    Keep them independent: d, D, K are allowed to differ per class.
    """
    family: GeomFamily = "affine_subspace"

    # Geometry
    d: int = 4                 # intrinsic dimension
    D: int = 16                # ambient dimension
    K: int = 4                 # number of modes (mixture components)
    separation: float = 6.0    # controls inter-mode separation magnitude

    # Noise / manifold thickness
    thickness: float = 0.05    # sigma_data (absolute continuity)

    # Tail + anisotropy + curvature
    tail: TailConfig = field(default_factory=TailConfig)
    anisotropy: AnisotropyConfig = field(default_factory=AnisotropyConfig)
    curvature: CurvatureConfig = field(default_factory=CurvatureConfig)

    # Optional: mode weights (if None -> uniform)
    mode_weights: Optional[List[float]] = None

    # Optional: control orientation randomness
    # If False, each class has a fixed base orientation and each mode reuses it.
    # If True, each mode has its own random orientation.
    orientation_per_mode: bool = True

    # For MoG family
    mog_diag_cov: float = 1.0  # baseline diagonal covariance (before thickness addition)


@dataclass
class DatasetConfig:
    """
    Dataset-level controls.
    Determinism: (base_seed, idx) -> unique RNG state; stable across workers.
    """
    num_samples: int = 50_000
    points_per_cloud: int = 512
    num_classes: int = 4
    base_seed: int = 1234
    device: str = "cpu"  # usually keep cpu for dataset generation; move batch to GPU in training

    # Map y -> ClassParams. If empty, will be auto-filled with defaults (same for all classes).
    classes: Dict[int, ClassParams] = field(default_factory=dict)


# -------------------------
# Low-level RNG + sampling
# -------------------------

def _make_generator(device: str, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed) % (2**63 - 1))
    return g


def _randn(shape: Tuple[int, ...], g: torch.Generator, device: str) -> torch.Tensor:
    return torch.randn(shape, generator=g, device=device)


def _rand(shape: Tuple[int, ...], g: torch.Generator, device: str) -> torch.Tensor:
    return torch.rand(shape, generator=g, device=device)


def _sample_tail(kind: BaseDist, n: int, d: int, g: torch.Generator, device: str, tail: TailConfig) -> torch.Tensor:
    if kind == "gauss":
        return _randn((n, d), g, device)
    if kind == "laplace":
        # inverse-CDF Laplace(0,1)
        u = _rand((n, d), g, device) - 0.5
        return -torch.sign(u) * torch.log1p(-2.0 * torch.abs(u))
    if kind == "student_t":
        # z = N(0,1) / sqrt(Chi2/df)
        df = float(tail.student_df)
        if df <= 0:
            raise ValueError("student_df must be > 0")
        normal = _randn((n, d), g, device)
        # Chi2 sampling: use torch.distributions with manual seed control via generator by sampling base uniforms
        # To keep it simple and deterministic, approximate Chi2 via Gamma(k=df/2, theta=2).
        # torch.distributions.Gamma currently uses global RNG; we avoid it by reparameterization from uniforms.
        # Use Marsaglia-Tsang would be longer; here we accept Gamma using global RNG is NOT ideal.
        # Therefore: we implement a simple (deterministic) approximation using normal^2 sum:
        # Chi2(df) = sum_{i=1..df} N(0,1)^2 for integer df. For non-integer, we fallback to integer rounding.
        df_int = max(1, int(round(df)))
        chi2 = (_randn((n, d, df_int), g, device) ** 2).sum(dim=-1)
        return normal / torch.sqrt(chi2 / float(df_int))
    if kind == "cauchy_trunc":
        # Cauchy via tan(pi(u-0.5)), then clamp
        u = _rand((n, d), g, device)
        z = torch.tan(math.pi * (u - 0.5))
        return z.clamp(min=-tail.cauchy_clip, max=tail.cauchy_clip)
    raise ValueError(f"Unknown tail kind: {kind}")


def _logspace_scales(d: int, min_s: float, max_s: float, device: str) -> torch.Tensor:
    if d == 1:
        return torch.tensor([max_s], device=device)
    return torch.exp(torch.linspace(math.log(min_s), math.log(max_s), steps=d, device=device))


def _random_orthonormal(D: int, d: int, g: torch.Generator, device: str) -> torch.Tensor:
    # QR on random matrix gives orthonormal columns. Deterministic w.r.t. generator.
    A = _randn((D, d), g, device)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    return Q[:, :d]


def _choose_mode(K: int, weights: Optional[List[float]], g: torch.Generator, device: str) -> int:
    if K <= 1:
        return 0
    if weights is None:
        # uniform
        u = _rand((1,), g, device).item()
        return int(min(K - 1, math.floor(u * K)))
    w = torch.tensor(weights, device=device, dtype=torch.float32)
    w = w / w.sum()
    u = _rand((1,), g, device).item()
    cdf = torch.cumsum(w, dim=0)
    return int(torch.searchsorted(cdf, torch.tensor(u, device=device)).item())


# -------------------------
# Families (generators)
# -------------------------

class PointCloudFamily:
    def sample(self, *, N: int, params: ClassParams, g: torch.Generator, device: str) -> torch.Tensor:
        raise NotImplementedError


class AffineSubspaceMixture(PointCloudFamily):
    """
    x = mu_k + (z * s) @ U^T + thickness * eps
    where U is D x d orthonormal, z in R^d sampled from tail dist.
    """
    def sample(self, *, N: int, params: ClassParams, g: torch.Generator, device: str) -> torch.Tensor:
        d, D, K = params.d, params.D, params.K
        if d > D:
            raise ValueError(f"Need d <= D, got d={d}, D={D}")

        # Choose mode
        k = _choose_mode(K, params.mode_weights, g, device)

        # Orientation
        if params.orientation_per_mode:
            U = _random_orthonormal(D, d, g, device)
        else:
            # fixed per-class orientation: derive from generator but keep stable within sample
            U = _random_orthonormal(D, d, g, device)

        # Intrinsic sample
        z = _sample_tail(params.tail.kind, N, d, g, device, params.tail)

        # Anisotropy in intrinsic coords
        if params.anisotropy.enabled:
            s = _logspace_scales(d, params.anisotropy.min_scale, params.anisotropy.max_scale, device)
            if params.anisotropy.permute_per_mode and d > 1:
                perm = torch.randperm(d, generator=g, device=device)
                s = s[perm]
            z = z * s

        # Mean shift controlling mode separation
        # Put separation along a random direction in ambient space for each mode.
        # This keeps "separation" independent of U choice.
        dir_vec = _randn((D,), g, device)
        dir_vec = dir_vec / (dir_vec.norm() + 1e-8)

        # Center the modes around 0 for stability:
        # offsets = (k - (K-1)/2) * separation
        offset = (float(k) - (float(K) - 1.0) / 2.0) * float(params.separation)
        mu = offset * dir_vec  # [D]

        x = z @ U.T + mu  # [N, D]
        x = x + float(params.thickness) * _randn((N, D), g, device)
        return x


class SineWarpSubspaceMixture(PointCloudFamily):
    """
    Same as affine subspace, but warps intrinsic coordinates:
        z' = z + alpha * sin(freq * z)
    """
    def sample(self, *, N: int, params: ClassParams, g: torch.Generator, device: str) -> torch.Tensor:
        if not params.curvature.enabled:
            # still allow using this family with alpha=0
            pass

        d, D, K = params.d, params.D, params.K
        if d > D:
            raise ValueError(f"Need d <= D, got d={d}, D={D}")

        k = _choose_mode(K, params.mode_weights, g, device)

        U = _random_orthonormal(D, d, g, device) if params.orientation_per_mode else _random_orthonormal(D, d, g, device)

        z = _sample_tail(params.tail.kind, N, d, g, device, params.tail)

        if params.anisotropy.enabled:
            s = _logspace_scales(d, params.anisotropy.min_scale, params.anisotropy.max_scale, device)
            if params.anisotropy.permute_per_mode and d > 1:
                perm = torch.randperm(d, generator=g, device=device)
                s = s[perm]
            z = z * s

        alpha = float(params.curvature.alpha) if params.curvature.enabled else 0.0
        freq = float(params.curvature.freq)
        z = z + alpha * torch.sin(freq * z)

        dir_vec = _randn((D,), g, device)
        dir_vec = dir_vec / (dir_vec.norm() + 1e-8)
        offset = (float(k) - (float(K) - 1.0) / 2.0) * float(params.separation)
        mu = offset * dir_vec

        x = z @ U.T + mu
        x = x + float(params.thickness) * _randn((N, D), g, device)
        return x


class MoGFamily(PointCloudFamily):
    """
    A plain mixture of Gaussians in R^D (controls K, separation, cov).
    Useful as phase-0 sanity check.
    """
    def sample(self, *, N: int, params: ClassParams, g: torch.Generator, device: str) -> torch.Tensor:
        D, K = params.D, params.K
        k = _choose_mode(K, params.mode_weights, g, device)

        # Means arranged along a random direction, centered
        dir_vec = _randn((D,), g, device)
        dir_vec = dir_vec / (dir_vec.norm() + 1e-8)
        offset = (float(k) - (float(K) - 1.0) / 2.0) * float(params.separation)
        mu = offset * dir_vec  # [D]

        # Diagonal covariance controlled by mog_diag_cov (then add thickness as extra noise)
        base_sigma = float(params.mog_diag_cov)
        x = mu + base_sigma * _randn((N, D), g, device)
        x = x + float(params.thickness) * _randn((N, D), g, device)
        return x


_FAMILY_REGISTRY: Dict[GeomFamily, PointCloudFamily] = {
    "affine_subspace": AffineSubspaceMixture(),
    "sine_warp_subspace": SineWarpSubspaceMixture(),
    "mog": MoGFamily(),
}


# -------------------------
# Dataset (deterministic)
# -------------------------

class ConditionalPointCloudDataset(Dataset):
    """
    Determinism strategy:
      - base_seed fixed
      - each idx uses seed = base_seed + idx
      - within idx, we sample class y using that generator => stable across workers
    """
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.N = int(cfg.points_per_cloud)
        self.num_samples = int(cfg.num_samples)
        self.device = cfg.device

        if not cfg.classes:
            # Auto-fill identical classes
            cfg.classes = {i: ClassParams() for i in range(cfg.num_classes)}
        else:
            # Ensure num_classes matches if user specifies
            cfg.num_classes = len(cfg.classes)

        self.class_ids = sorted(cfg.classes.keys())
        self.class_params: Dict[int, ClassParams] = cfg.classes

    def __len__(self) -> int:
        return self.num_samples

    def _sample_label(self, g: torch.Generator) -> int:
        # Uniform over classes (you can extend with class priors)
        C = len(self.class_ids)
        u = torch.rand((1,), generator=g, device=self.device).item()
        return self.class_ids[int(min(C - 1, math.floor(u * C)))]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        seed = int(self.cfg.base_seed) + int(idx)
        g = _make_generator(self.device, seed)

        y = self._sample_label(g)
        params = self.class_params[y]
        family = _FAMILY_REGISTRY[params.family]

        x = family.sample(N=self.N, params=params, g=g, device=self.device)  # [N, D]
        # Return label as python int for standard collate
        return x, int(y)


# -------------------------
# Optional: collate helper
# -------------------------

def collate_pointclouds(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # [B, N, D] (D may differ across classes; keep D consistent in configs)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y
