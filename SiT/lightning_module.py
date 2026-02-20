from __future__ import annotations

import json
import os
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
from time import time

import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

import SiT.wandb_utils as wandb_utils
from SiT.class_registry import build_class_registry
from SiT.eval_runner import run_metrics, sample_and_log_ema
from SiT.models import SiT_models
from SiT.transport import Sampler, create_transport
from utils.colorful_logger import info


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    for parameter in model.parameters():
        parameter.requires_grad = flag


def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


class SiTLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig, experiment_name: str, experiment_dir: str):
        super().__init__()
        self.cfg = cfg
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        self.save_hyperparameters({"hydra": OmegaConf.to_container(cfg, resolve=True)})

        self.use_vae = bool(getattr(cfg.data, "use_vae", True))

        if self.use_vae:
            assert cfg.model.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
            latent_size = cfg.model.image_size // 8
        else:
            latent_size = int(getattr(cfg.model, "input_size", 1))
        self.latent_size = latent_size

        if self.is_main_process():
            info(f"Model config: num_classes={cfg.model.num_classes}, in_channels={cfg.model.in_channels}")

        self.model = SiT_models[cfg.model.name](
            input_size=latent_size,
            num_classes=cfg.model.num_classes,
            in_channels=cfg.model.in_channels,
        )
        self.ema = deepcopy(self.model)
        requires_grad(self.ema, False)

        self.transport = create_transport(
            cfg.model.transport.path_type,
            cfg.model.transport.prediction,
            cfg.model.transport.loss_weight,
            cfg.model.transport.train_eps,
            cfg.model.transport.sample_eps,
        )
        self.transport_sampler = Sampler(self.transport)

        if self.use_vae:
            self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cfg.model.vae}")
            self.vae.requires_grad_(False)
        else:
            self.vae = None

        self.use_cfg = cfg.model.cfg_scale > 1.0
        self._running_loss = None
        self._log_steps = 0
        self._log_start_time = None

        self._sample_zs = None
        self._sample_model_kwargs = None
        self._sample_model_fn = None

        sampling_cfg = getattr(cfg.model, "sampling", None)
        if sampling_cfg is not None:
            self.sampling_mode = sampling_cfg.get("mode", "ODE")
            self.sampling_cfg = OmegaConf.to_container(sampling_cfg, resolve=True)
        else:
            self.sampling_mode = "ODE"
            self.sampling_cfg = {}

        self.metrics_dir = os.path.join(self.experiment_dir, "metrics")
        self.class_registry_path = os.path.join(self.metrics_dir, "class_registry.json")
        self.val_loss_by_class_path = os.path.join(self.metrics_dir, "val_loss_by_class.jsonl")
        self._class_registry_written = False
        self._class_registry_cache: dict[str, dict] = {}
        self._val_class_ids: list[int] = []
        self._val_class_id_to_idx: dict[int, int] = {}
        self._val_loss_sums: torch.Tensor | None = None
        self._val_loss_counts: torch.Tensor | None = None

    def is_main_process(self) -> bool:
        return is_main_process()

    def on_fit_start(self):
        self._ensure_metrics_storage()
        self.ema.eval()
        update_ema(self.ema, self.model, decay=0)
        self._running_loss = torch.tensor(0.0, device=self.device)
        self._log_steps = 0
        self._log_start_time = time()

        world_size = getattr(self.trainer, "world_size", 1)
        if self.cfg.trainer.global_batch_size % world_size != 0:
            raise ValueError("global_batch_size must be divisible by world size.")
        local_batch_size = int(self.cfg.trainer.global_batch_size // world_size)

        ys = torch.randint(self.cfg.model.num_classes, size=(local_batch_size,), device=self.device)
        if self.use_vae:
            zs = torch.randn(local_batch_size, 4, self.latent_size, self.latent_size, device=self.device)
        else:
            zs = torch.randn(local_batch_size, self.latent_size, self.cfg.model.in_channels, device=self.device)

        if self.use_cfg:
            zs = torch.cat([zs, zs], 0)
            y_null = torch.tensor([self.cfg.model.num_classes] * local_batch_size, device=self.device)
            ys = torch.cat([ys, y_null], 0)
            self._sample_model_kwargs = dict(y=ys, cfg_scale=self.cfg.model.cfg_scale)
            self._sample_model_fn = self.ema.forward_with_cfg
        else:
            self._sample_model_kwargs = dict(y=ys)
            self._sample_model_fn = self.ema.forward

        self._sample_zs = zs

        if self.is_main_process():
            info(f"SiT Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.trainer.lr,
            weight_decay=self.cfg.trainer.weight_decay,
        )

    def _ensure_metrics_storage(self):
        os.makedirs(self.metrics_dir, exist_ok=True)

    def _build_class_registry(self) -> dict:
        return build_class_registry(
            cfg=self.cfg,
            datamodule=self.trainer.datamodule,
            val_class_ids=self._val_class_ids,
            experiment_name=self.experiment_name,
            experiment_dir=self.experiment_dir,
        )

    def _write_class_registry_file(self):
        if self._class_registry_written or not self.is_main_process():
            return

        self._ensure_metrics_storage()
        registry = self._build_class_registry()
        with open(self.class_registry_path, "w", encoding="utf-8") as handle:
            json.dump(registry, handle, indent=2, sort_keys=True)
        self._class_registry_cache = registry.get("classes", {})
        self._class_registry_written = True
        info(f"Wrote class registry: {self.class_registry_path}")

        if self.cfg.trainer.use_wandb:
            self._log_metrics_artifact(aliases=["latest", "registry"])

    def _log_metrics_artifact(self, aliases: list[str] | None = None):
        if not self.cfg.trainer.use_wandb or not self.is_main_process():
            return

        artifact_name = f"{self.experiment_name}-metrics".replace("/", "-")
        metadata = {
            "experiment_name": self.experiment_name,
            "global_step": int(self.global_step),
            "epoch": int(self.current_epoch),
        }
        logged = wandb_utils.log_artifact_files(
            name=artifact_name,
            artifact_type="metrics",
            files=[
                (self.class_registry_path, "metrics/class_registry.json"),
                (self.val_loss_by_class_path, "metrics/val_loss_by_class.jsonl"),
            ],
            metadata=metadata,
            aliases=aliases or ["latest"],
        )
        if logged:
            info(f"Logged metrics artifact to W&B: {artifact_name}")

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.use_vae:
            with torch.no_grad():
                x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        loss_dict = self.transport.training_losses(self.model, x, dict(y=y))
        loss = loss_dict["loss"].mean()
        self._running_loss += loss.detach()
        self._log_steps += 1
        return loss

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        update_ema(self.ema, self.model)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        step = self.global_step
        if step > 0 and step % self.cfg.trainer.log_every == 0:
            avg_loss = self._running_loss / max(self._log_steps, 1)
            if self.trainer.world_size > 1:
                avg_loss = self.all_gather(avg_loss).mean()
            avg_loss_value = avg_loss.item()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = max(time() - self._log_start_time, 1e-8)
            steps_per_sec = self._log_steps / elapsed

            if self.is_main_process():
                info(
                    f"(step={step:07d}) Train Loss: {avg_loss_value:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}"
                )
                if self.cfg.trainer.use_wandb:
                    wandb_utils.log(
                        {"train loss": avg_loss_value, "train steps/sec": steps_per_sec},
                        step=step,
                    )

            self._running_loss = torch.tensor(0.0, device=self.device)
            self._log_steps = 0
            self._log_start_time = time()

        if step > 0 and step % self.cfg.trainer.sample_every == 0 and self.use_vae:
            sample_and_log_ema(self, step)

    def on_validation_epoch_start(self):
        self._ensure_metrics_storage()

        class_ids = None
        datamodule = self.trainer.datamodule
        if datamodule is not None and getattr(datamodule, "is_metrics_capable", False):
            try:
                class_ids = [int(class_id) for class_id in datamodule.get_eval_config().class_ids]
            except (AttributeError, RuntimeError, ValueError):
                class_ids = None
        if class_ids is None:
            class_ids = list(range(int(self.cfg.model.num_classes)))

        self._val_class_ids = class_ids
        self._val_class_id_to_idx = {class_id: idx for idx, class_id in enumerate(self._val_class_ids)}
        self._val_loss_sums = torch.zeros(len(self._val_class_ids), device=self.device, dtype=torch.float64)
        self._val_loss_counts = torch.zeros(len(self._val_class_ids), device=self.device, dtype=torch.float64)
        self._write_class_registry_file()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if y.max() >= self.cfg.model.num_classes:
            raise ValueError(
                f"Label out of range: max(y)={y.max().item()}, "
                f"num_classes={self.cfg.model.num_classes}"
            )
        if self.use_vae:
            with torch.no_grad():
                x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        loss_dict = self.transport.training_losses(self.model, x, dict(y=y))
        per_sample_loss = loss_dict["loss"]

        if self._val_loss_sums is not None and self._val_loss_counts is not None:
            idx_list = [self._val_class_id_to_idx.get(int(class_id), -1) for class_id in y.detach().cpu().tolist()]
            valid_pos = [i for i, idx in enumerate(idx_list) if idx >= 0]
            if valid_pos:
                idx_tensor = torch.tensor([idx_list[i] for i in valid_pos], device=self.device, dtype=torch.long)
                loss_tensor = per_sample_loss[valid_pos].detach().to(torch.float64)
                count_tensor = torch.ones_like(loss_tensor, dtype=torch.float64)
                self._val_loss_sums.scatter_add_(0, idx_tensor, loss_tensor)
                self._val_loss_counts.scatter_add_(0, idx_tensor, count_tensor)

        loss = per_sample_loss.mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        self._log_validation_loss_by_class()
        run_metrics(self, split="val")

    def _log_validation_loss_by_class(self):
        if self._val_loss_sums is None or self._val_loss_counts is None:
            return

        loss_sums = self._val_loss_sums.clone()
        loss_counts = self._val_loss_counts.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_counts, op=dist.ReduceOp.SUM)

        if not self.is_main_process():
            return

        class_loss_values: dict[str, float] = {}
        class_count_values: dict[str, int] = {}
        wandb_stats: dict[str, float] = {}

        total_sum = 0.0
        total_count = 0.0
        for idx, class_id in enumerate(self._val_class_ids):
            count = float(loss_counts[idx].item())
            if count <= 0:
                continue
            class_loss = float(loss_sums[idx].item() / count)
            class_key = str(class_id)
            class_loss_values[class_key] = class_loss
            class_count_values[class_key] = int(count)
            total_sum += float(loss_sums[idx].item())
            total_count += count
            wandb_stats[f"val_loss/class_{class_id}"] = class_loss

            class_meta = self._class_registry_cache.get(class_key, {})
            sweep_names = class_meta.get("sweeps", [])
            sweep_str = ",".join(sweep_names) if sweep_names else "none"
            info(
                f"(step={self.global_step:07d}) val_loss/class_{class_id}: "
                f"{class_loss:.6f} (n={int(count)}, sweeps={sweep_str})"
            )

        overall_val_loss = (total_sum / total_count) if total_count > 0 else float("nan")
        info(f"(step={self.global_step:07d}) val_loss/overall_from_class: {overall_val_loss:.6f}")
        wandb_stats["val_loss/overall_from_class"] = overall_val_loss

        if self.cfg.trainer.use_wandb and wandb_stats:
            wandb_utils.log(wandb_stats, step=self.global_step)

        record = {
            "schema_version": "1.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "step": int(self.global_step),
            "epoch": int(self.current_epoch),
            "split": "val",
            "overall_val_loss": overall_val_loss,
            "class_val_loss": class_loss_values,
            "class_counts": class_count_values,
        }
        with open(self.val_loss_by_class_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        info(f"Appended class-wise validation loss: {self.val_loss_by_class_path}")

        if self.cfg.trainer.use_wandb:
            self._log_metrics_artifact(aliases=["latest", "val"])

    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.use_vae:
            with torch.no_grad():
                x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        loss_dict = self.transport.training_losses(self.model, x, dict(y=y))
        loss = loss_dict["loss"].mean()
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_test_epoch_end(self):
        run_metrics(self, split="test")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()
        checkpoint["hydra_config"] = OmegaConf.to_container(self.cfg, resolve=True)
        checkpoint["experiment_name"] = self.experiment_name
        checkpoint["experiment_dir"] = self.experiment_dir

    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"], strict=True)
