# Diffusability Playground

> **Research goal:** study diffusability of latent spaces by starting from synthetic datasets with controllable geometry, evaluating them with DiT models, and later moving to real audio and vision datasets.

> [!NOTE]
> This repository is under active construction.

## Quick Context
- **Env/deps:** use Astral `uv` (`uv add` only, no `pip install`).
- **Training/config:** Hydra-driven experiments; main training config is `conf/config.yaml` with groups under `conf/data/`, `conf/model/`, `conf/trainer/`.
- **Plotting:** `utils/plot_distribution.py` uses `conf/plot.yaml` and `conf/data/synth_pc.yaml`.
- **Dataset docs:** detailed synthetic dataset documentation is available at `docs/synthetic_pointcloud_dataset.md` (attributes, sweep strategies, commands, and image gallery).
- **Plot config ambient dim:** `conf/plot.yaml` defines `ambient_dim` so `${ambient_dim}` interpolations in `conf/data/synth_pc.yaml` resolve correctly during plotting.
- **Anisotropy control:** `conf/config.yaml` defines `anisotropy_max_scale`, used by `conf/data/synth_pc.yaml` for single-class anisotropy configuration and easy multirun sweeps.
- **Plot output dirs:** `utils/plot_distribution.py` now creates parent directories for `plot.out_name` automatically (e.g. `media_outputs/`).
- **Doc plot configs:** use `conf/plot_dataset_docs.yaml` and `conf/plot_dataset_docs_anis.yaml` to generate reproducible sweep figures for documentation.
- **DataModules:** `datamodules/synthetic_pointclouds.py` provides a Lightning DataModule around the synthetic point cloud dataset.
- **SiT model flags:** `use_pos_embed` and `use_patch_embed` allow disabling positional embeddings and patchifying (e.g. for permutation-invariant sequences).
- **Synthetic sweeps:** `conf/data/synth_pc.yaml` supports `class_sweeps` to expand a base class into multiple classes via a parameter grid.
- **Current default synthetic setup:** `conf/data/synth_pc.yaml` is set to an anisotropy-focused ablation (`sweep_affine_anis`) with `K=1`, `separation=0.0`, and a single `anisotropy.max_scale` value by default (use Hydra multirun override to sweep).
- **Anisotropy sweep override:** use `anisotropy_max_scale=...` in Hydra CLI multirun to launch one run per anisotropy value.
- **Point-cloud training:** use `conf/data/synth_pc_datamodule.yaml` (wraps `conf/data/synth_pc.yaml`) and set `conf/model/mini_sit.yaml` for non-patchified, no-PE models.
- **Class count auto-resolve:** `model.num_classes` is auto-resolved at runtime from the instantiated datamodule/dataset before model construction.
- **Validation:** periodic validation with metrics computation; configure `trainer.check_val_every_n_epoch` (epochs, default `1`) or `trainer.val_check_interval` (steps).
- **Validation reproducibility:** set `trainer.val_metrics_seed` to use fixed-noise validation sampling for stable metric curves (`null` keeps stochastic sampling).
- **Metrics system:** generic metrics interface in `datamodules/metrics_protocol.py`; each DataModule implements `compute_metrics()` for dataset-specific evaluation.
- **Point-cloud metrics:** SWD + Energy Distance + Feature-MMD are available; current point-cloud training defaults focus on Feature-MMD (`swd.enabled=false`, `energy_distance.enabled=false`, `feature_mmd.enabled=true`).
- **Early stopping:** `trainer.early_stopping` monitors `val/feature_mmd_mean` (mean across class-wise Feature-MMD metrics) and can stop if no improvement for a configured patience.
- **Metric hyperparameters:** controlled in `conf/data/synth_pc_datamodule.yaml` under `metrics` (enable flags, SWD projections, Energy cloud-distance settings, Feature-MMD feature toggles/kernel params).
- **Feature-MMD tracking:** keep `metrics.feature_mmd.gamma` fixed across validation epochs for comparable curves (avoid `gamma: null` when monitoring convergence over time).
- **Sampling:** supports both **ODE** (dopri5, euler, heun) and **SDE** (Euler, Heun with configurable diffusion) via `model.sampling` config.
- **Per-class validation loss tracking:** each run writes `results/<run>/metrics/class_registry.json` (class id -> sweep/params mapping) and appends `results/<run>/metrics/val_loss_by_class.jsonl` (time series by step/epoch).
- **W&B metrics artifacts:** metric files are uploaded as artifact entries under `metrics/class_registry.json` and `metrics/val_loss_by_class.jsonl`.
- **W&B run IDs:** run IDs are now left to W&B auto-generation (no deterministic `id` passed in `wandb.init`).
- **Validation generation logging:** validation/test sample generation now prints class-by-class progress lines (no tqdm hash/progress bar output).
- **Point-cloud class params:** `orientation_per_mode` was removed from active configs/code path (old configs are ignored safely if the key is still present).
- **Logger naming:** `utils/colorfull_logger.py` was renamed to `utils/colorful_logger.py`.
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
SiT/train.py                 # Hydra entrypoint + Trainer wiring
SiT/lightning_module.py      # SiTLightningModule (train/val/test hooks)
SiT/eval_runner.py           # Metric orchestration + eval sampling/logging
SiT/class_registry.py        # Class registry serialization helpers

datamodules/
    ├── metrics_protocol.py       # Interface: MetricsCapableDataModule, EvalConfig
    ├── synthetic_pointclouds.py  # collect_real_samples + compute_metrics (SWD/Energy-U/Feature-MMD)
    └── image_datamodule.py       # collect_real_samples + compute_metrics (placeholder for FID)
```
