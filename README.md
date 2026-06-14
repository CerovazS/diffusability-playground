# Diffusability Playground

Diffusability Playground is a research codebase for studying when a latent space is easy or hard for diffusion and flow-matching models to learn.

The central idea is to separate the geometry of the data distribution from the decoder, dataset, and representation-learning confounders that usually appear in latent diffusion experiments. The repository starts from synthetic vector distributions with controlled intrinsic dimension, ambient dimension, anisotropy, curvature, multimodality, tail behavior, and manifold thickness. These controlled settings make it possible to ask sharper questions about latent-space diffusability before moving the same evaluation protocol to real vision and audio latents.

![Affine-subspace anisotropy sweep](docs/assets/pointcloud_dataset/sweep_anis_single_cloud_sweep_affine_anis.png)

## Scientific Goal

Latent diffusion performance is often discussed in terms of semantic alignment, compression quality, or reconstruction fidelity, but these explanations do not fully isolate the geometry of the latent distribution itself. This project treats diffusability as an empirical property of a distribution: how efficiently a generative dynamics model can match it under fixed modeling, sampling, and compute budgets.

The current experiments are designed to test questions such as:

- Does increasing anisotropy make a distribution systematically harder to model?
- How do intrinsic dimension and ambient dimension interact with diffusion/flow-matching quality?
- Which distributional metrics are stable enough to compare synthetic latent geometries?
- Can synthetic geometric probes predict failure modes that later appear in real autoencoder latents?

The long-term aim is to build a measurable bridge between latent-space geometry and generative-model trainability.

## What Is Included

- Synthetic vector datasets with class-conditional geometric controls.
- SiT-style flow/diffusion training on vector data and image-folder data.
- Per-class evaluation with sliced Wasserstein distance, exact Wasserstein-2, energy distance, and feature-space MMD.
- Validation and test artifacts written locally for reproducible comparison across runs.
- Plotting utilities for visualizing distributions, validation samples, metric stability, and anisotropy sweeps.

The conceptual project note is available in [`docs/flywheel/on_the_diffusability_of_latent_spaces.md`](docs/flywheel/on_the_diffusability_of_latent_spaces.md).

## Repository Layout

```text
conf/                         Experiment and tool configurations
SiT/train.py                  Main training entrypoint
SiT/lightning_module.py       Lightning module for training, validation, and test
SiT/eval_runner.py            Sampling and metric orchestration
datamodules/synthetic_pointclouds.py
                              Controlled synthetic vector distributions
datamodules/image_datamodule.py
                              Image-folder datamodule for later real-data experiments
utils/pointcloud_metrics.py   Distributional metrics for vector samples
utils/plot_distribution.py    Synthetic distribution visualization
utils/plot_anisotropy_intrinsic_sweep.py
                              Sweep aggregation and plotting
utils/evaluate_checkpoint_metrics.py
                              Checkpoint-by-checkpoint metric evaluation
```

## Setup

The project uses Python 3.13 and GPU-enabled PyTorch. Install the environment with:

```bash
uv sync
```

Training currently expects a CUDA-capable GPU. For a single-GPU local run, override the distributed strategy with `trainer.strategy=auto`.

## Quick Start

Run the default synthetic experiment:

```bash
uv run python SiT/train.py trainer.strategy=auto
```

Run a small smoke test without W&B:

```bash
uv run python SiT/train.py \
  trainer.strategy=auto \
  trainer.use_wandb=false \
  trainer.epochs=1 \
  data.train_samples_per_class=512 \
  data.val_samples_per_class=512 \
  data.test_samples_per_class=256
```

The default configuration trains a vector model on a single-class affine-subspace distribution. The main geometric controls live in [`conf/config.yaml`](conf/config.yaml):

```yaml
ambient_dim: 16
intrinsic_dim: 6
anisotropy_max_scale: 1.0
data_thickness: 0.02
```

These values are propagated into the datamodule and model configuration, so changing them at the command line is usually enough to define a new synthetic condition.

## Example Experiments

Single run with explicit geometry:

```bash
uv run python SiT/train.py \
  ambient_dim=8 \
  intrinsic_dim=6 \
  anisotropy_max_scale=4.0 \
  trainer.strategy=auto
```

Anisotropy sweep in ambient dimension 8:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python SiT/train.py -m \
  ambient_dim=8 \
  intrinsic_dim=6 \
  anisotropy_max_scale=1.0,2.0,4.0,8.0,16.0 \
  trainer.results_dir=results/anisotropy_sweep_ambient8_d6 \
  trainer.strategy=auto
```

Anisotropy sweep in ambient dimension 16:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python SiT/train.py -m \
  ambient_dim=16 \
  intrinsic_dim=6 \
  anisotropy_max_scale=1.0,2.0,4.0,8.0,16.0 \
  trainer.results_dir=results/anisotropy_sweep_ambient16_d6 \
  trainer.strategy=auto
```

Evaluate saved checkpoints every 5 epochs:

```bash
uv run python utils/evaluate_checkpoint_metrics.py
```

With an explicit result root and stride:

```bash
uv run python utils/evaluate_checkpoint_metrics.py \
  roots='[results/anisotropy_sweep_ambient8_d6]' \
  epoch_stride=10
```

## Synthetic Data Controls

Synthetic samples are individual vectors in `R^D`, not point clouds. Each class has a fixed geometry, and each sample resamples latent coordinates, mode choice, and additive noise.

Supported families:

- `affine_subspace`
- `sine_warp_subspace`
- `mog`

Controllable factors include:

- intrinsic dimension `d`
- ambient dimension `D`
- number and separation of mixture modes
- anisotropy of intrinsic coordinates
- manifold thickness
- tail behavior: Gaussian, Laplace, Student-t, or truncated Cauchy
- sine-warp curvature

## Outputs

Each training run writes a separate experiment directory under `results/`:

```text
results/<run>/
  checkpoints/
  metrics/
    class_registry.json
    val_loss_by_class.jsonl
    test_loss_by_class.json
  plots/
    val/
      distribution_comparison_epochXXX_stepXXXXXXX.png
      manifest.jsonl
```

The metric files are intended to be consumed directly by plotting and aggregation scripts. They contain both optimization losses and generative distribution metrics, including per-class values and aggregate summaries when available.

## Plotting

Visualize the configured synthetic dataset:

```bash
uv run python utils/plot_distribution.py
```

Regenerate the documentation-style dataset figures:

```bash
uv run python utils/plot_distribution.py --config-name plot_dataset_docs
uv run python utils/plot_distribution.py --config-name plot_dataset_docs_anis
```

Aggregate anisotropy sweep results:

```bash
uv run python utils/plot_anisotropy_intrinsic_sweep.py \
  results_root=results/anisotropy_sweep_ambient8_d6

uv run python utils/plot_anisotropy_intrinsic_sweep.py \
  results_root=results/anisotropy_sweep_ambient16_d6
```

The aggregation script reads local metric artifacts and, when present, local W&B logs under `wandb/`. It summarizes validation loss, sliced Wasserstein distance, and feature-MMD trends across sweep conditions.

## Sampling

Sampling behavior is configured in the model YAML files under [`conf/model/`](conf/model/). The vector experiments currently support ODE and SDE sampling modes:

```yaml
sampling:
  mode: ODE
  ode:
    method: dopri5
    num_steps: 50
    atol: 1.0e-6
    rtol: 1.0e-3
  sde:
    method: Euler
    num_steps: 250
    diffusion_form: SBDM
    diffusion_norm: 1.0
    last_step: Mean
    last_step_size: 0.04
```

## Status

This is an active research playground. The synthetic-vector path is the most developed part of the repository; image-folder support exists as a bridge toward real latent experiments, but the core scientific protocol is still centered on controlled synthetic geometry.
