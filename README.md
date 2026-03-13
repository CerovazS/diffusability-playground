# Diffusability Playground

Study diffusability of latent spaces starting from synthetic datasets with controllable geometry, then move toward real audio and vision datasets.

This repo is Hydra-driven and uses Astral `uv` for environment and dependency management. Use `uv add` for dependencies. For config-driven construction, use `hydra.utils.instantiate`; do not wire instantiation manually with OmegaConf.

## Setup

```bash
uv sync
```

Main references:
- `https://docs.astral.sh/uv/`
- `https://hydra.cc/docs/intro/`

## Repo Layout

```text
conf/config.yaml                 # main experiment entrypoint
conf/data/*.yaml                 # dataset / datamodule configs
conf/model/*.yaml                # model configs
conf/trainer/*.yaml              # trainer configs

SiT/train.py                     # Hydra entrypoint + Lightning Trainer wiring
SiT/lightning_module.py          # training / validation / test module
SiT/eval_runner.py               # evaluation sampling + metric orchestration

datamodules/synthetic_pointclouds.py  # synthetic vector dataset + datamodule
utils/plot_distribution.py            # synthetic dataset visualization
utils/plot_anisotropy_intrinsic_sweep.py  # aggregate sweep plots from saved runs

docs/synthetic_pointcloud_dataset.md
docs/synthetic_pointcloud_math_foundations.md
```

## Current Default Experiment

`conf/config.yaml` is the main training config. Its defaults are:
- `data: synth_pc_datamodule`
- `model: mini_mlp`
- `trainer: sit_trainer`

The current synthetic setup is a single-class affine-subspace dataset with:
- one sample = one vector in `R^D`
- `ambient_dim` defined once in `conf/config.yaml`
- `intrinsic_dim` defined once in `conf/config.yaml`
- `anisotropy_max_scale` defined once in `conf/config.yaml`

Those shared parameters are propagated via Hydra interpolations:
- `ambient_dim -> data.in_channels`
- `ambient_dim -> model.in_channels`
- `ambient_dim -> class_sweeps[*].base.D`
- `intrinsic_dim -> class_sweeps[*].base.d`
- `anisotropy_max_scale -> class_sweeps[*].sweep.anisotropy.max_scale`

`conf/data/synth_pc.yaml` uses `class_sweeps` with a single base class and a sweepable anisotropy value. In practice, Hydra multirun produces one training job per anisotropy setting.

## Synthetic Dataset Notes

`datamodules/synthetic_pointclouds.py` now works in the point-wise setting:
- each sample is a single vector `[D]`, not a cloud `[N, D]`
- class geometry is sampled once per class
- per-sample randomness only resamples latent coordinates, component choice, and additive noise

Supported geometric families:
- `affine_subspace`
- `sine_warp_subspace`
- `mog`

The default datamodule computes:
- SWD
- Energy-U
- Feature-MMD

These metrics are computed directly on vector samples `[N, D]`; the legacy cloud-level metric path has been removed.

Per-run artifacts are written under `results/<experiment>/metrics/`:
- `class_registry.json`
- `val_loss_by_class.jsonl`

## Training Commands

Single run with the current defaults:

```bash
uv run python SiT/train.py
```

Single run with explicit synthetic controls:

```bash
uv run python SiT/train.py \
  ambient_dim=8 \
  intrinsic_dim=6 \
  anisotropy_max_scale=4.0 \
  trainer.strategy=auto
```

Ambient-8 anisotropy sweep over 5 levels:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python SiT/train.py -m \
  ambient_dim=8 \
  intrinsic_dim=6 \
  anisotropy_max_scale=1.0,2.0,4.0,8.0,16.0 \
  trainer.results_dir=results/anisotropy_sweep_ambient8_d6 \
  trainer.strategy=auto
```

Ambient-16 anisotropy sweep over the same 5 levels:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python SiT/train.py -m \
  ambient_dim=16 \
  intrinsic_dim=6 \
  anisotropy_max_scale=1.0,2.0,4.0,8.0,16.0 \
  trainer.results_dir=results/anisotropy_sweep_ambient16_d6 \
  trainer.strategy=auto
```

Notes:
- `trainer.strategy=auto` is the safest override for single-GPU runs.
- `model.num_classes` is resolved automatically at runtime from the instantiated datamodule.
- W&B is enabled by default through `conf/trainer/sit_trainer.yaml`.

## Plotting

Visualize the current synthetic dataset:

```bash
uv run python utils/plot_distribution.py
```

Regenerate the documentation figures:

```bash
uv run python utils/plot_distribution.py --config-name plot_dataset_docs
uv run python utils/plot_distribution.py --config-name plot_dataset_docs_anis
```

Aggregate anisotropy sweep results:

```bash
uv run python utils/plot_anisotropy_intrinsic_sweep.py \
  --results-root results/anisotropy_sweep_ambient8_d6

uv run python utils/plot_anisotropy_intrinsic_sweep.py \
  --results-root results/anisotropy_sweep_ambient16_d6
```

The sweep plotting utility reads local training artifacts from `results/...` and matches them with local W&B logs under `wandb/`.

## Sampling Configuration

Sampling is configured in `conf/model/*.yaml`:

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
    diffusion_form: SBDM
    diffusion_norm: 1.0
    last_step: Mean
    last_step_size: 0.04
```

## Maintenance

When repo behavior changes, update this README so commands, config names, and outputs remain accurate.
