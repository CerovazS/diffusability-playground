# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Training script for SiT using PyTorch Lightning + Hydra.
"""
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hydra
import numpy as np
import torch
import torch.distributed as dist
from lightning import LightningModule, Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from diffusers.models import AutoencoderKL
from models import SiT_models
from transport import Sampler, create_transport
import wandb_utils
from utils.colorfull_logger import info
from datamodules.metrics_protocol import MetricsCapableDataModule

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def is_main_process():
    return int(os.environ.get("LOCAL_RANK", "0")) == 0




#################################################################################
#                                   Lightning Module                           #
#################################################################################

class SiTLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig, experiment_name: str, experiment_dir: str):
        super().__init__()
        self.cfg = cfg
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        self.save_hyperparameters({"hydra": OmegaConf.to_container(cfg, resolve=True)})

        use_vae = bool(getattr(cfg.data, "use_vae", True))
        self.use_vae = use_vae

        if self.use_vae:
            assert cfg.model.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
            latent_size = cfg.model.image_size // 8
        else:
            latent_size = int(getattr(cfg.model, "input_size", 1))
        self.latent_size = latent_size

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

        # Sampling configuration (with backward compatibility)
        sampling_cfg = getattr(cfg.model, "sampling", None)
        if sampling_cfg is not None:
            self.sampling_mode = sampling_cfg.get("mode", "ODE")
            self.sampling_cfg = OmegaConf.to_container(sampling_cfg, resolve=True)
        else:
            self.sampling_mode = "ODE"
            self.sampling_cfg = {}

    def on_fit_start(self):
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

        if is_main_process():
            info(f"SiT Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.trainer.lr,
            weight_decay=self.cfg.trainer.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.use_vae:
            with torch.no_grad():
                x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        model_kwargs = dict(y=y)
        loss_dict = self.transport.training_losses(self.model, x, model_kwargs)
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
            end_time = time()
            steps_per_sec = self._log_steps / max(end_time - self._log_start_time, 1e-8)

            if is_main_process():
                info(
                    f"(step={step:07d}) Train Loss: {avg_loss_value:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}"
                )
                if self.cfg.trainer.wandb:
                    wandb_utils.log(
                        {"train loss": avg_loss_value, "train steps/sec": steps_per_sec},
                        step=step,
                    )

            self._running_loss = torch.tensor(0.0, device=self.device)
            self._log_steps = 0
            self._log_start_time = time()

        if step > 0 and step % self.cfg.trainer.sample_every == 0 and self.use_vae:
            self._sample_and_log(step)

    def on_validation_epoch_end(self):
        self._run_metrics(split="val")

    def on_test_epoch_end(self):
        self._run_metrics(split="test")

    def _run_metrics(self, split: str):
        """
        Generic metrics computation using the datamodule's metrics interface.
        
        This method delegates all dataset-specific metric computation to the
        datamodule, keeping train.py generic across different data types.
        """
        dm = self.trainer.datamodule
        
        # Check if datamodule supports metrics
        if dm is None or not getattr(dm, "is_metrics_capable", False):
            return
        
        # Only main process computes metrics
        if not is_main_process():
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            return
        
        try:
            eval_config = dm.get_eval_config()
        except (AttributeError, RuntimeError, ValueError) as e:
            if is_main_process():
                info(f"Skipping metrics for {split}: {e}")
            return
        
        # Collect real samples using the datamodule's method
        real_samples = dm.collect_real_samples_by_class(
            split=split,
            samples_per_class=eval_config.samples_per_class,
        )
        
        # Generate samples using the generic method
        gen_samples = self._generate_samples_by_class(
            class_ids=eval_config.class_ids,
            samples_per_class=eval_config.samples_per_class,
            sample_shape=eval_config.sample_shape,
            needs_decoding=eval_config.needs_decoding,
        )
        
        # Compute metrics using the datamodule's method
        metrics = dm.compute_metrics(real_samples, gen_samples, split)
        
        if self.cfg.trainer.wandb:
            wandb_utils.log(metrics, step=self.global_step)
        
        if is_main_process():
            for name, value in metrics.items():
                info(f"(step={self.global_step:07d}) {name}: {value:.6f}")
        
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _get_sample_fn(self):
        """
        Get the appropriate sampling function based on configuration.
        
        Supports both ODE and SDE sampling methods with configurable parameters.
        
        Returns:
            A sampling function that takes (z, model_fn, **model_kwargs)
        """
        mode = self.sampling_mode.upper()
        
        if mode == "ODE":
            ode_cfg = self.sampling_cfg.get("ode", {})
            return self.transport_sampler.sample_ode(
                sampling_method=ode_cfg.get("method", "dopri5"),
                num_steps=ode_cfg.get("num_steps", 50),
                atol=ode_cfg.get("atol", 1e-6),
                rtol=ode_cfg.get("rtol", 1e-3),
                reverse=ode_cfg.get("reverse", False),
            )
        elif mode == "SDE":
            sde_cfg = self.sampling_cfg.get("sde", {})
            return self.transport_sampler.sample_sde(
                sampling_method=sde_cfg.get("method", "Euler"),
                num_steps=sde_cfg.get("num_steps", 250),
                diffusion_form=sde_cfg.get("diffusion_form", "SBDM"),
                diffusion_norm=sde_cfg.get("diffusion_norm", 1.0),
                last_step=sde_cfg.get("last_step", "Mean"),
                last_step_size=sde_cfg.get("last_step_size", 0.04),
            )
        else:
            raise ValueError(f"Unknown sampling mode: {mode}. Use 'ODE' or 'SDE'.")

    def _generate_samples_by_class(
        self,
        class_ids: list,
        samples_per_class: int,
        sample_shape: tuple,
        needs_decoding: bool = False,
        batch_size: int = 64,
    ) -> dict:
        """
        Generate samples for each class using the EMA model.
        
        This is a generic sampling method that works for any data type.
        The sample_shape determines the noise tensor shape.
        Supports both ODE and SDE sampling based on configuration.
        
        Args:
            class_ids: List of class IDs to generate for
            samples_per_class: Number of samples per class
            sample_shape: Shape of each sample (e.g., (N_points, D) or (C, H, W))
            needs_decoding: Whether to decode with VAE (for images)
            batch_size: Batch size for generation
            
        Returns:
            Dict mapping class_id -> numpy array of generated samples
        """
        gen = {cid: [] for cid in class_ids}
        sample_fn = self._get_sample_fn()

        for class_id in class_ids:
            remaining = samples_per_class
            while remaining > 0:
                batch = min(remaining, batch_size)
                y = torch.full((batch,), class_id, device=self.device, dtype=torch.long)
                z = torch.randn(batch, *sample_shape, device=self.device)

                if self.use_cfg:
                    z = torch.cat([z, z], 0)
                    y_null = torch.tensor([self.cfg.model.num_classes] * batch, device=self.device)
                    y = torch.cat([y, y_null], 0)
                    model_kwargs = dict(y=y, cfg_scale=self.cfg.model.cfg_scale)
                    model_fn = self.ema.forward_with_cfg
                else:
                    model_kwargs = dict(y=y)
                    model_fn = self.ema.forward

                with torch.no_grad():
                    samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                    if self.use_cfg:
                        samples, _ = samples.chunk(2, dim=0)
                    
                    # Decode with VAE if needed (for images)
                    if needs_decoding and self.vae is not None:
                        samples = self.vae.decode(samples / 0.18215).sample

                gen[class_id].append(samples.cpu().numpy())
                remaining -= batch

        return {k: np.concatenate(v, axis=0) for k, v in gen.items()}

    def _sample_and_log(self, step):
        if is_main_process():
            info(f"Generating EMA samples (mode={self.sampling_mode})...")

        with torch.no_grad():
            sample_fn = self._get_sample_fn()
            samples = sample_fn(self._sample_zs, self._sample_model_fn, **self._sample_model_kwargs)[-1]

            if self.use_cfg:
                samples, _ = samples.chunk(2, dim=0)
            samples = self.vae.decode(samples / 0.18215).sample

            if self.trainer.world_size > 1:
                gathered = self.all_gather(samples)
                samples = gathered.reshape(-1, *samples.shape[1:])

        if self.cfg.trainer.wandb and is_main_process():
            wandb_utils.log_image(samples, step)
        if is_main_process():
            info("Generating EMA samples done.")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()
        checkpoint["hydra_config"] = OmegaConf.to_container(self.cfg, resolve=True)
        checkpoint["experiment_name"] = self.experiment_name
        checkpoint["experiment_dir"] = self.experiment_dir

    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema"], strict=True)


#################################################################################
#                                  Config helpers                              #
#################################################################################


def _load_checkpoint_config(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "hydra_config" in ckpt:
        return ckpt, OmegaConf.create(ckpt["hydra_config"])
    raise ValueError("Checkpoint missing hydra_config. Only Lightning checkpoints are supported.")


#################################################################################
#                                  Training Entry                              #
#################################################################################


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    ckpt_path = cfg.ckpt
    if ckpt_path is not None:
        ckpt_abs = hydra.utils.to_absolute_path(ckpt_path)
        _, ckpt_cfg = _load_checkpoint_config(ckpt_abs)
        cfg = ckpt_cfg
        cfg.ckpt = ckpt_abs

    seed_everything(cfg.trainer.global_seed, workers=True)

    results_dir = hydra.utils.to_absolute_path(cfg.trainer.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    model_string_name = cfg.model.name.replace("/", "-")
    experiment_index = len(glob(f"{results_dir}/*"))
    experiment_name = (
        f"{experiment_index:03d}-{model_string_name}-"
        f"{cfg.model.transport.path_type}-{cfg.model.transport.prediction}-{cfg.model.transport.loss_weight}"
    )
    experiment_dir = os.path.join(results_dir, experiment_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if is_main_process():
        info(f"Experiment directory created at {experiment_dir}")

    if cfg.trainer.wandb and is_main_process():
        entity = cfg.trainer.wandb_entity
        project = cfg.trainer.wandb_project
        wandb_utils.initialize(OmegaConf.to_container(cfg, resolve=True), entity, experiment_name, project)

    if cfg.data._target_ == "datamodules.image_datamodule.ImageFolderDataModule":
        datamodule = hydra.utils.instantiate(
            cfg.data,
            num_workers=cfg.trainer.num_workers,
            global_batch_size=cfg.trainer.global_batch_size,
        )
    else:
        datamodule = hydra.utils.instantiate(cfg.data)
    module = SiTLightningModule(cfg, experiment_name, experiment_dir)

    from lightning.pytorch.callbacks import ModelCheckpoint

    trainer = Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        enable_progress_bar=is_main_process(),
        logger=False,
        enable_checkpointing=True,
        default_root_dir=experiment_dir,
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="{step:07d}",
                every_n_train_steps=cfg.trainer.ckpt_every,
                save_top_k=-1,
                save_last=False,
            )
        ],
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.ckpt)

    if is_main_process():
        info("Done!")


if __name__ == "__main__":
    main()
