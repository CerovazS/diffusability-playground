"""
Synthetic point-cloud dataset and Lightning DataModule.

Overview
--------
This module generates labeled point-cloud samples with controllable geometry.
Each dataset item is a tuple (x, y) where:
  - x is a point cloud of shape [N, D] (N points in D-dimensional space)
  - y is an integer class label in [0, num_classes-1]

Key properties
--------------
- Permutation invariant: each sample is an unordered set of points; no positional
  encoding or ordering is assumed.
- Deterministic by index: for a fixed base_seed, each sample index maps to a
  deterministic RNG stream, making the dataset stable across workers and runs.
- Multi-family geometry: each class chooses a geometric family with its own
  parameters, such as affine subspaces, sine-warped subspaces, or mixtures of
  Gaussians.

Config structure
----------------
The dataset is parameterized by dataclasses:
  - DatasetConfig: global controls (num_samples, points_per_cloud, num_classes,
    base_seed, device, classes)
  - ClassParams: per-class geometry parameters, including family, intrinsic/ambient
    dimensions (d, D), mixture components K, separation, thickness, and optional
    tail/anisotropy/curvature settings.

Lightning integration
---------------------
SyntheticPointCloudDataModule wraps the dataset and provides a train dataloader.
It expects a DatasetConfig instance created by Hydra (via hydra.utils.instantiate),
and exposes typical DataLoader knobs (batch_size, num_workers, pin_memory, drop_last).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Literal, Tuple, Any

import math
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from datamodules.metrics_protocol import BaseMetricsDataModule, EvalConfig


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
    # If set, enforce an equal number of samples per class.
    samples_per_class: Optional[int] = None

    # Map y -> ClassParams. If empty, will be auto-filled with defaults (same for all classes).
    classes: Dict[int, ClassParams] = field(default_factory=dict)
    # Optional: expand additional classes from one or more sweep definitions.
    # Each sweep entry is a dict with:
    #   - name: optional string label for plotting
    #   - base: dict of class params
    #   - sweep: dict of param -> list of values (cartesian product)
    class_sweeps: List[Dict[str, Any]] = field(default_factory=list)


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
        self.samples_per_class = cfg.samples_per_class
        self.class_splits: Dict[str, List[int]] = {}

        if cfg.class_sweeps:
            _expand_class_sweeps(cfg, self.class_splits)

        if not cfg.classes:
            # Auto-fill identical classes
            cfg.classes = {i: ClassParams() for i in range(cfg.num_classes)}
        else:
            # Ensure num_classes matches if user specifies
            cfg.num_classes = len(cfg.classes)
        if cfg.class_sweeps:
            cfg.num_classes = len(cfg.classes)

        self.class_ids = sorted(cfg.classes.keys())
        self.class_params: Dict[int, ClassParams] = cfg.classes

    def __len__(self) -> int:
        if self.samples_per_class is not None:
            return int(self.samples_per_class) * len(self.class_ids)
        return self.num_samples

    def _sample_label(self, g: torch.Generator) -> int:
        # Uniform over classes (you can extend with class priors)
        C = len(self.class_ids)
        u = torch.rand((1,), generator=g, device=self.device).item()
        return self.class_ids[int(min(C - 1, math.floor(u * C)))]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.samples_per_class is not None:
            class_index = int(idx) // int(self.samples_per_class)
            sample_index = int(idx) % int(self.samples_per_class)
            if class_index >= len(self.class_ids):
                raise IndexError("Sample index out of range.")
            y = self.class_ids[class_index]
            seed = int(self.cfg.base_seed) + class_index * int(self.samples_per_class) + sample_index
        else:
            seed = int(self.cfg.base_seed) + int(idx)
            g = _make_generator(self.device, seed)
            y = self._sample_label(g)

        g = _make_generator(self.device, seed)
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


def _set_nested(d: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def _dict_to_class_params(d: dict) -> ClassParams:
    """Convert a dict (possibly from Hydra/YAML) to ClassParams, handling nested dataclasses."""
    d = dict(d)  # shallow copy
    # Convert nested dicts to their respective dataclasses
    if "tail" in d and isinstance(d["tail"], dict):
        d["tail"] = TailConfig(**d["tail"])
    if "anisotropy" in d and isinstance(d["anisotropy"], dict):
        d["anisotropy"] = AnisotropyConfig(**d["anisotropy"])
    if "curvature" in d and isinstance(d["curvature"], dict):
        d["curvature"] = CurvatureConfig(**d["curvature"])
    return ClassParams(**d)


def _expand_class_sweeps(cfg: DatasetConfig, splits_out: Dict[str, List[int]]) -> None:
    next_id = max(cfg.classes.keys(), default=-1) + 1
    for idx, sweep_def in enumerate(cfg.class_sweeps):
        if not isinstance(sweep_def, dict) or "base" not in sweep_def or "sweep" not in sweep_def:
            raise ValueError("Each class_sweeps entry must be a dict with keys: base, sweep.")

        name = sweep_def.get("name", f"sweep_{idx}")
        base = sweep_def["base"]
        sweep = sweep_def["sweep"]

        if isinstance(base, ClassParams):
            base_dict = asdict(base)
        elif isinstance(base, dict):
            base_dict = dict(base)  # shallow copy
        else:
            raise TypeError("class_sweeps.base must be a dict or ClassParams.")

        if not isinstance(sweep, dict):
            raise TypeError("class_sweeps.sweep must be a dict of param -> list.")

        keys = list(sweep.keys())
        values = []
        for key in keys:
            v = sweep[key]
            if isinstance(v, list):
                values.append(v)
            else:
                values.append([v])

        import itertools
        combo_ids: List[int] = []
        for combo in itertools.product(*values):
            cfg_dict = {k: v for k, v in base_dict.items()}
            for k, v in zip(keys, combo):
                _set_nested(cfg_dict, k, v)
            cfg.classes[next_id] = _dict_to_class_params(cfg_dict)
            combo_ids.append(next_id)
            next_id += 1
        splits_out[name] = combo_ids


# -------------------------
# Lightning DataModule
# -------------------------

class SyntheticPointCloudDataModule(LightningDataModule, BaseMetricsDataModule):
    def __init__(
        self,
        cfg: Optional[DatasetConfig] = None,
        dataset: Optional[ConditionalPointCloudDataset] = None,
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        shuffle: bool = True,
        train_seed: Optional[int] = None,
        val_seed: Optional[int] = None,
        test_seed: Optional[int] = None,
        train_samples_per_class: Optional[int] = None,
        val_samples_per_class: Optional[int] = None,
        test_samples_per_class: Optional[int] = None,
        # Extra config fields for Hydra interpolations (not used directly)
        in_channels: Optional[int] = None,
        use_vae: bool = False,
    ):
        super().__init__()
        if cfg is None and dataset is None:
            raise TypeError("Provide either cfg (DatasetConfig) or dataset (ConditionalPointCloudDataset).")
        if cfg is not None and not isinstance(cfg, DatasetConfig):
            raise TypeError("cfg must be a DatasetConfig instance.")
        if dataset is not None and not isinstance(dataset, ConditionalPointCloudDataset):
            raise TypeError("dataset must be a ConditionalPointCloudDataset instance.")
        self.cfg = cfg if cfg is not None else dataset.cfg
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.train_seed = train_seed
        self.val_seed = val_seed
        self.test_seed = test_seed
        self.train_samples_per_class = train_samples_per_class
        self.val_samples_per_class = val_samples_per_class
        self.test_samples_per_class = test_samples_per_class
        self.train_dataset: Optional[ConditionalPointCloudDataset] = None
        self.val_dataset: Optional[ConditionalPointCloudDataset] = None
        self.test_dataset: Optional[ConditionalPointCloudDataset] = None
        self.is_pointcloud = True
        self.is_metrics_capable = True

    def setup(self, stage: Optional[str] = None):
        if self.dataset is not None:
            self.train_dataset = self.dataset
            return

        from copy import deepcopy
        base_cfg = deepcopy(self.cfg)
        train_seed = self.train_seed if self.train_seed is not None else base_cfg.base_seed
        val_seed = self.val_seed if self.val_seed is not None else base_cfg.base_seed + 1
        test_seed = self.test_seed if self.test_seed is not None else base_cfg.base_seed + 2

        train_spc = self.train_samples_per_class if self.train_samples_per_class is not None else base_cfg.samples_per_class
        val_spc = self.val_samples_per_class if self.val_samples_per_class is not None else base_cfg.samples_per_class
        test_spc = self.test_samples_per_class if self.test_samples_per_class is not None else base_cfg.samples_per_class

        train_cfg = replace(base_cfg, base_seed=train_seed, samples_per_class=train_spc)
        val_cfg = replace(base_cfg, base_seed=val_seed, samples_per_class=val_spc)
        test_cfg = replace(base_cfg, base_seed=test_seed, samples_per_class=test_spc)

        self.train_dataset = ConditionalPointCloudDataset(train_cfg)
        self.val_dataset = ConditionalPointCloudDataset(val_cfg)
        self.test_dataset = ConditionalPointCloudDataset(test_cfg)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=collate_pointclouds,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        batch_size = self.val_batch_size or self.batch_size
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_pointclouds,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        batch_size = self.test_batch_size or self.batch_size
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=collate_pointclouds,
        )

    # -------------------------
    # Metrics Interface (BaseMetricsDataModule)
    # -------------------------

    def get_eval_config(self) -> EvalConfig:
        """Return evaluation configuration for point cloud metrics."""
        dataset = self.val_dataset or self.train_dataset
        if dataset is None:
            raise RuntimeError("Call setup() before requesting eval config.")
        
        samples_per_class = dataset.samples_per_class
        if samples_per_class is None:
            raise ValueError("samples_per_class must be set for point-cloud metrics.")
        
        return EvalConfig(
            class_ids=dataset.class_ids,
            samples_per_class=samples_per_class,
            sample_shape=(dataset.N, self.cfg.classes[dataset.class_ids[0]].D),
            num_classes=len(dataset.class_ids),
            needs_decoding=False,
        )

    def collect_real_samples_by_class(
        self,
        split: str,
        samples_per_class: int,
        batch_size: int = 64,
    ) -> Dict[int, np.ndarray]:
        """
        Collect real point cloud samples from the dataset, organized by class.
        
        Args:
            split: "train", "val", or "test"
            samples_per_class: number of samples to collect per class
            batch_size: batch size for loading
            
        Returns:
            Dict mapping class_id -> numpy array of shape [N_samples, N_points, D]
        """
        dataset = getattr(self, f"{split}_dataset", None)
        if dataset is None:
            raise ValueError(f"No {split} dataset available.")
        
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, samples_per_class),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_pointclouds,
            drop_last=False,
        )

        clouds: Dict[int, List[np.ndarray]] = {cid: [] for cid in dataset.class_ids}
        counts: Dict[int, int] = {cid: 0 for cid in dataset.class_ids}

        for xb, yb in loader:
            for i in range(xb.size(0)):
                y = int(yb[i].item())
                if counts[y] >= samples_per_class:
                    continue
                clouds[y].append(xb[i].cpu().numpy())
                counts[y] += 1
            if all(c >= samples_per_class for c in counts.values()):
                break

        return {k: np.stack(v, axis=0) for k, v in clouds.items()}

    def compute_metrics(
        self,
        real_samples: Dict[int, np.ndarray],
        generated_samples: Dict[int, np.ndarray],
        split: str,
    ) -> Dict[str, float]:
        """
        Compute point cloud metrics: SWD and MMD for each class.
        
        Args:
            real_samples: Dict mapping class_id -> real point clouds [N, points, D]
            generated_samples: Dict mapping class_id -> generated point clouds [N, points, D]
            split: "val" or "test" for metric naming
            
        Returns:
            Dict of metric_name -> metric_value
        """
        from utils.pointcloud_metrics import (
            mmd_rbf_from_distance_matrices,
            pairwise_chamfer_matrix,
            sliced_wasserstein_distance,
        )
        
        metrics: Dict[str, float] = {}
        
        for class_id in real_samples.keys():
            if class_id not in generated_samples:
                continue
                
            real = real_samples[class_id]
            gen = generated_samples[class_id]

            # Flatten for SWD (all points from all clouds)
            real_points = real.reshape(-1, real.shape[-1])
            gen_points = gen.reshape(-1, gen.shape[-1])

            swd = sliced_wasserstein_distance(
                real_points,
                gen_points,
                num_projections=256,
                seed=0,
            )

            # Chamfer-based MMD
            D_xx = pairwise_chamfer_matrix(real, real)
            D_yy = pairwise_chamfer_matrix(gen, gen)
            D_xy = pairwise_chamfer_matrix(real, gen)
            gamma = 1.0 / (float(np.median(D_xy)) + 1e-8)
            mmd = mmd_rbf_from_distance_matrices(D_xx, D_yy, D_xy, gamma=gamma)

            metrics[f"{split}/swd/class_{class_id}"] = float(swd)
            metrics[f"{split}/mmd/class_{class_id}"] = float(mmd)

        return metrics
