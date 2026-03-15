from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

import SiT.wandb_utils as wandb_utils
from utils.colorful_logger import info
from utils.validation_distribution_plots import (
    DistributionPlotConfig,
    save_distribution_comparison_plot,
)


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


def _get_distribution_plot_cfg(module: Any) -> DistributionPlotConfig:
    trainer_cfg = getattr(module.cfg, "trainer", None)
    raw_cfg = None if trainer_cfg is None else getattr(trainer_cfg, "distribution_plots", None)
    if raw_cfg is None:
        return DistributionPlotConfig(enabled=False)

    if isinstance(raw_cfg, dict):
        mapping = raw_cfg
    else:
        mapping = {key: raw_cfg[key] for key in raw_cfg.keys()}
    return DistributionPlotConfig.from_mapping(mapping)


def _get_class_params_by_id(datamodule: Any, split: str) -> dict[int, Any]:
    dataset = getattr(datamodule, f"{split}_dataset", None)
    if dataset is None:
        dataset = getattr(datamodule, "train_dataset", None)
    if dataset is None or not hasattr(dataset, "class_params"):
        return {}
    return {int(class_id): params for class_id, params in dataset.class_params.items()}


def _resolve_eval_step_epoch(module: Any) -> tuple[int, int]:
    step_value = int(getattr(module, "_regen_step_override", module.global_step))
    epoch_value = int(getattr(module, "_regen_epoch_override", module.current_epoch))
    return step_value, epoch_value


def _nest_split_metrics(split: str, metrics: dict[str, float]) -> dict[str, Any]:
    nested: dict[str, dict[str, Any]] = {}
    prefix = f"{split}/"
    for key, value in metrics.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        value_f = float(value)

        if "/class_" in suffix:
            metric_name, class_label = suffix.split("/class_", 1)
            bucket = nested.setdefault(metric_name, {"class": {}, "aggregate": {}})
            bucket["class"][class_label] = value_f
            continue

        if suffix.endswith("_mean"):
            metric_name = suffix[: -len("_mean")]
            bucket = nested.setdefault(metric_name, {"class": {}, "aggregate": {}})
            bucket["aggregate"]["mean"] = value_f
            continue

        bucket = nested.setdefault(suffix, {"class": {}, "aggregate": {}})
        bucket["aggregate"]["value"] = value_f

    return nested


def _attach_split_metrics_to_loss_records(module: Any, *, split: str, metrics: dict[str, float]) -> None:
    if split == "val":
        path_attr = "val_loss_by_class_path"
        file_format = "jsonl"
    elif split == "test":
        path_attr = "test_loss_by_class_path"
        file_format = "json"
    else:
        return

    if not hasattr(module, path_attr):
        return

    path = Path(getattr(module, path_attr))
    if not path.exists():
        return

    step_value, epoch_value = _resolve_eval_step_epoch(module)
    if file_format == "jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()
        records: list[dict[str, Any]] = [json.loads(line) for line in lines if line.strip()]
    else:
        raw = path.read_text(encoding="utf-8").strip()
        records = json.loads(raw) if raw else []
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON list of records.")

    target_idx = None
    for idx in range(len(records) - 1, -1, -1):
        row = records[idx]
        if (
            str(row.get("split", "")) == split
            and int(row.get("step", -1)) == step_value
            and int(row.get("epoch", -1)) == epoch_value
        ):
            target_idx = idx
            break

    flat_metrics = {key: float(value) for key, value in metrics.items() if key.startswith(f"{split}/")}
    payload = {
        "distribution_metrics_flat": dict(sorted(flat_metrics.items())),
        "distribution_metrics": _nest_split_metrics(split, metrics),
        "metrics_enriched_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if target_idx is None:
        records.append(
            {
                "schema_version": "1.1",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "step": step_value,
                "epoch": epoch_value,
                "split": split,
                **payload,
            }
        )
    else:
        records[target_idx].update(payload)

    with path.open("w", encoding="utf-8") as handle:
        if file_format == "jsonl":
            for row in records:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        else:
            json.dump(records, handle, indent=2, sort_keys=True)


def _log_distribution_comparison(
    module: Any,
    *,
    datamodule: Any,
    split: str,
    real_samples: dict[int, np.ndarray],
    generated_samples: dict[int, np.ndarray],
) -> None:
    cfg = _get_distribution_plot_cfg(module)
    if not cfg.supports_split(split):
        return

    step_value, epoch_value = _resolve_eval_step_epoch(module)
    is_regeneration = hasattr(module, "_regen_epoch_override") or hasattr(module, "_regen_step_override")
    if not is_regeneration and cfg.every_n_epochs > 1 and (epoch_value + 1) % int(cfg.every_n_epochs) != 0:
        return

    plots_dir = Path(module.experiment_dir) / "plots" / split
    output_path = plots_dir / f"distribution_comparison_epoch{epoch_value:03d}_step{step_value:07d}.png"

    payload = save_distribution_comparison_plot(
        real_samples=real_samples,
        generated_samples=generated_samples,
        class_params_by_id=_get_class_params_by_id(datamodule, split),
        output_path=output_path,
        split=split,
        step=step_value,
        epoch=epoch_value,
        cfg=cfg,
    )

    manifest_path = plots_dir / "manifest.jsonl"
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    info(f"Saved {split} distribution comparison plot: {output_path}")

    if module.cfg.trainer.use_wandb:
        wandb_utils.log_image_file(f"{split}/distribution_comparison", str(output_path), step=module.global_step)


def run_metrics(module: Any, split: str) -> None:
    module._ensure_metrics_storage()
    module._write_class_registry_file()
    datamodule = module.trainer.datamodule

    if datamodule is None or not getattr(datamodule, "is_metrics_capable", False):
        return

    if split == "val":
        cadence = int(getattr(module.cfg.trainer, "val_generative_metrics_every_n_epoch", 1))
        is_regeneration = hasattr(module, "_regen_epoch_override") or hasattr(module, "_regen_step_override")
        if not is_regeneration and cadence > 1 and (int(module.current_epoch) + 1) % cadence != 0:
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

    _log_distribution_comparison(
        module,
        datamodule=datamodule,
        split=split,
        real_samples=real_samples,
        generated_samples=generated_samples,
    )

    if module.is_main_process():
        info("Computing metrics (SWD, Exact-W2, Energy-U, Feature-MMD)...")
    metrics = datamodule.compute_metrics(real_samples, generated_samples, split)

    # Aggregate class metrics into a single scalar so summaries/plots can track sweep trends.
    swd_keys = [k for k in metrics.keys() if k.startswith(f"{split}/swd/class_")]
    if swd_keys:
        swd_mean = float(np.mean([float(metrics[k]) for k in swd_keys]))
        metrics[f"{split}/swd_mean"] = swd_mean
        module.log(
            f"{split}/swd_mean",
            swd_mean,
            on_step=False,
            on_epoch=True,
            prog_bar=(split == "val"),
            logger=False,
            sync_dist=False,
        )

    # Aggregate feature-MMD across classes so Trainer callbacks can monitor a single scalar.
    feature_mmd_keys = [k for k in metrics.keys() if k.startswith(f"{split}/feature_mmd/class_")]
    if feature_mmd_keys:
        feature_mmd_mean = float(np.mean([float(metrics[k]) for k in feature_mmd_keys]))
        metrics[f"{split}/feature_mmd_mean"] = feature_mmd_mean
        module.log(
            f"{split}/feature_mmd_mean",
            feature_mmd_mean,
            on_step=False,
            on_epoch=True,
            prog_bar=(split == "val"),
            logger=False,
            sync_dist=False,
        )

    _attach_split_metrics_to_loss_records(module, split=split, metrics=metrics)

    step_value, _ = _resolve_eval_step_epoch(module)
    if module.cfg.trainer.use_wandb:
        wandb_utils.log(metrics, step=step_value)

    if module.is_main_process():
        for name, value in metrics.items():
            info(f"(step={step_value:07d}) {name}: {value:.6f}")

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
