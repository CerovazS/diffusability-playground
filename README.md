# Diffusability Playground

> **Research goal:** study diffusability of latent spaces by starting from synthetic datasets with controllable geometry, evaluating them with DiT models, and later moving to real audio and vision datasets.

> [!NOTE]
> This repository is under active construction.

## Quick Context
- **Env/deps:** use Astral `uv` (`uv add` only, no `pip install`).
- **Training/config:** Hydra-driven experiments; main training config is `conf/config.yaml` with groups under `conf/data/`, `conf/model/`, `conf/trainer/`.
- **Plotting:** `utils/plot_distribution.py` uses `conf/plot.yaml` and `conf/data/synth_pc.yaml`.
- **DataModules:** `datamodules/synthetic_pointclouds.py` provides a Lightning DataModule around the synthetic point cloud dataset.
- **SiT model flags:** `use_pos_embed` and `use_patch_embed` allow disabling positional embeddings and patchifying (e.g. for permutation-invariant sequences).
- **Synthetic sweeps:** `conf/data/synth_pc.yaml` supports `class_sweeps` to expand a base class into multiple classes via a parameter grid.
- **Point-cloud training:** use `conf/data/synth_pc_datamodule.yaml` (wraps `conf/data/synth_pc.yaml`) and set `conf/model/mini_sit.yaml` for non-patchified, no-PE models.
- **Metrics system:** generic metrics interface in `datamodules/metrics_protocol.py`; each DataModule implements `compute_metrics()` for dataset-specific evaluation.
- **Point-cloud metrics:** SWD and MMD-RBF (Chamfer) computed via `SyntheticPointCloudDataModule.compute_metrics()`, logged to wandb per class.
- **Sampling:** supports both **ODE** (dopri5, euler, heun) and **SDE** (Euler, Heun with configurable diffusion) via `model.sampling` config.
- **Docs:** `https://docs.astral.sh/uv/` and `https://hydra.cc/docs/intro/`.
- **Maintenance:** after each operation, update this README to reflect new changes or fix outdated info.

## Sampling Configuration

The model supports both ODE and SDE sampling methods. Configure via `conf/model/*.yaml`:

```yaml
sampling:
  mode: ODE  # or SDE
  ode:
    method: dopri5  # dopri5, euler, heun
    num_steps: 50
    atol: 1.0e-6
    rtol: 1.0e-3
  sde:
    method: Euler  # Euler, Heun
    num_steps: 250
    diffusion_form: SBDM  # constant, SBDM, sigma, linear, decreasing, increasing-decreasing
    diffusion_norm: 1.0
    last_step: Mean  # null, Mean, Tweedie, Euler
    last_step_size: 0.04
```

## Synthetic Point Clouds (brief)
`synthetic_pointclouds.py` implements a deterministic, Hydra-friendly dataset generator.  
Each sample is a point cloud `x ∈ R^{N×D}` with label `y`, produced by selecting a class and a mixture component, sampling intrinsic coordinates `z` from a configurable tail distribution, mapping them into ambient space, and adding thickness noise.  
It supports multiple geometric families (affine subspaces, sine-warped subspaces, and MoG) with per-class controls for intrinsic dimension, ambient dimension, number of modes, separation, anisotropy, curvature, and tail heaviness.

## Architecture Overview

```
train.py (LightningModule)
    │
    ├── _run_metrics()          # Generic metrics orchestration
    │       │
    │       └── dm.compute_metrics()   # Delegates to DataModule
    │
    └── _generate_samples_by_class()   # Generic sampling (ODE/SDE)
            │
            └── _get_sample_fn()       # Returns ODE or SDE sampler

datamodules/
    ├── metrics_protocol.py     # Interface: MetricsCapableDataModule, EvalConfig
    ├── synthetic_pointclouds.py  # Implements: collect_real_samples, compute_metrics (SWD/MMD)
    └── image_datamodule.py       # Implements: collect_real_samples, compute_metrics (placeholder for FID)
```
