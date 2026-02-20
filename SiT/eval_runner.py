from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.distributed as dist

import SiT.wandb_utils as wandb_utils
from utils.colorful_logger import info


def get_sample_fn(module: Any):
    mode = module.sampling_mode.upper()

    if mode == "ODE":
        ode_cfg = module.sampling_cfg.get("ode", {})
        return module.transport_sampler.sample_ode(
            sampling_method=ode_cfg.get("method", "dopri5"),
            num_steps=ode_cfg.get("num_steps", 50),
            atol=ode_cfg.get("atol", 1e-6),
            rtol=ode_cfg.get("rtol", 1e-3),
            reverse=ode_cfg.get("reverse", False),
        )

    if mode == "SDE":
        sde_cfg = module.sampling_cfg.get("sde", {})
        return module.transport_sampler.sample_sde(
            sampling_method=sde_cfg.get("method", "Euler"),
            num_steps=sde_cfg.get("num_steps", 250),
            diffusion_form=sde_cfg.get("diffusion_form", "SBDM"),
            diffusion_norm=sde_cfg.get("diffusion_norm", 1.0),
            last_step=sde_cfg.get("last_step", "Mean"),
            last_step_size=sde_cfg.get("last_step_size", 0.04),
        )

    raise ValueError(f"Unknown sampling mode: {mode}. Use 'ODE' or 'SDE'.")


def generate_samples_by_class(
    module: Any,
    *,
    class_ids: list[int],
    samples_per_class: int,
    sample_shape: tuple[int, ...],
    needs_decoding: bool = False,
    batch_size: int = 64,
    seed: int | None = None,
) -> dict[int, np.ndarray]:
    generated: dict[int, list[np.ndarray]] = {class_id: [] for class_id in class_ids}
    sample_fn = get_sample_fn(module)

    for idx, class_id in enumerate(class_ids):
        if module.is_main_process():
            info(
                f"Generating samples ({module.sampling_mode}) "
                f"class={class_id} [{idx + 1}/{len(class_ids)}], target={samples_per_class}"
            )

        remaining = samples_per_class
        class_rng = None
        if seed is not None:
            class_rng = torch.Generator(device=module.device)
            class_rng.manual_seed(int(seed) + int(class_id))

        while remaining > 0:
            batch = min(remaining, batch_size)
            y = torch.full((batch,), class_id, device=module.device, dtype=torch.long)
            z = torch.randn(batch, *sample_shape, device=module.device, generator=class_rng)

            if module.use_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([module.cfg.model.num_classes] * batch, device=module.device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=module.cfg.model.cfg_scale)
                model_fn = module.ema.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                model_fn = module.ema.forward

            with torch.no_grad():
                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                if module.use_cfg:
                    samples, _ = samples.chunk(2, dim=0)

                if needs_decoding and module.vae is not None:
                    samples = module.vae.decode(samples / 0.18215).sample

            generated[class_id].append(samples.cpu().numpy())
            remaining -= batch

    return {key: np.concatenate(value, axis=0) for key, value in generated.items()}


def run_metrics(module: Any, split: str) -> None:
    module._ensure_metrics_storage()
    module._write_class_registry_file()
    datamodule = module.trainer.datamodule

    if datamodule is None or not getattr(datamodule, "is_metrics_capable", False):
        return

    if not module.is_main_process():
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return

    try:
        eval_config = datamodule.get_eval_config()
    except (AttributeError, RuntimeError, ValueError) as exc:
        if module.is_main_process():
            info(f"Skipping metrics for {split}: {exc}")
        return

    samples_per_class = eval_config.samples_per_class

    if module.is_main_process():
        info(f"Running {split} metrics (samples_per_class={samples_per_class}, mode={module.sampling_mode})...")

    real_samples = datamodule.collect_real_samples_by_class(
        split=split,
        samples_per_class=samples_per_class,
    )

    metrics_seed = getattr(module.cfg.trainer, "val_metrics_seed", None) if split == "val" else None
    generated_samples = generate_samples_by_class(
        module,
        class_ids=eval_config.class_ids,
        samples_per_class=samples_per_class,
        sample_shape=eval_config.sample_shape,
        needs_decoding=eval_config.needs_decoding,
        seed=metrics_seed,
    )

    if module.is_main_process():
        info("Computing metrics (SWD, Energy-U, Feature-MMD)...")
    metrics = datamodule.compute_metrics(real_samples, generated_samples, split)

    if module.cfg.trainer.use_wandb:
        wandb_utils.log(metrics, step=module.global_step)

    if module.is_main_process():
        for name, value in metrics.items():
            info(f"(step={module.global_step:07d}) {name}: {value:.6f}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def sample_and_log_ema(module: Any, step: int) -> None:
    if module.is_main_process():
        info(f"Generating EMA samples (mode={module.sampling_mode})...")

    with torch.no_grad():
        sample_fn = get_sample_fn(module)
        samples = sample_fn(module._sample_zs, module._sample_model_fn, **module._sample_model_kwargs)[-1]

        if module.use_cfg:
            samples, _ = samples.chunk(2, dim=0)
        samples = module.vae.decode(samples / 0.18215).sample

        if module.trainer.world_size > 1:
            gathered = module.all_gather(samples)
            samples = gathered.reshape(-1, *samples.shape[1:])

    if module.cfg.trainer.use_wandb and module.is_main_process():
        wandb_utils.log_image(samples, step)
    if module.is_main_process():
        info("Generating EMA samples done.")
