# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Training script for SiT using PyTorch Lightning + Hydra.
"""
import logging
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
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from diffusers.models import AutoencoderKL
from models import SiT_models
from transport import Sampler, create_transport
import wandb_utils

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


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logger = logging.getLogger(__name__)
    if is_main_process():
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                   Data Module                                #
#################################################################################

class ImageFolderDataModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        image_size,
        global_batch_size,
        num_workers,
        pin_memory=True,
        drop_last=True,
    ):
        super().__init__()
        self.data_path = data_path
        self.image_size = image_size
        self.num_workers = num_workers
        self.global_batch_size = global_batch_size
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.dataset = None
        self.local_batch_size = None

    def setup(self, stage=None):
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        data_path = hydra.utils.to_absolute_path(self.data_path)
        self.dataset = ImageFolder(data_path, transform=transform)

    def train_dataloader(self):
        world_size = getattr(self.trainer, "world_size", 1)
        if self.global_batch_size % world_size != 0:
            raise ValueError("global_batch_size must be divisible by world size.")
        self.local_batch_size = int(self.global_batch_size // world_size)
        return DataLoader(
            self.dataset,
            batch_size=self.local_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )


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
        self._logger = logging.getLogger(__name__)

        assert cfg.model.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        latent_size = cfg.model.image_size // 8
        self.latent_size = latent_size

        self.model = SiT_models[cfg.model.name](
            input_size=latent_size,
            num_classes=cfg.model.num_classes,
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

        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cfg.model.vae}")
        self.vae.requires_grad_(False)

        self.use_cfg = cfg.model.cfg_scale > 1.0
        self._running_loss = None
        self._log_steps = 0
        self._log_start_time = None

        self._sample_zs = None
        self._sample_model_kwargs = None
        self._sample_model_fn = None

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
        zs = torch.randn(local_batch_size, 4, self.latent_size, self.latent_size, device=self.device)

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
            self._logger.info(f"SiT Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.trainer.lr,
            weight_decay=self.cfg.trainer.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
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
                self._logger.info(
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

        if step > 0 and step % self.cfg.trainer.sample_every == 0:
            self._sample_and_log(step)

    def _sample_and_log(self, step):
        if is_main_process():
            self._logger.info("Generating EMA samples...")

        with torch.no_grad():
            sample_fn = self.transport_sampler.sample_ode()
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
            self._logger.info("Generating EMA samples done.")

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

    logger = create_logger(experiment_dir)
    if is_main_process():
        logger.info(f"Experiment directory created at {experiment_dir}")

    if cfg.trainer.wandb and is_main_process():
        entity = os.environ["ENTITY"]
        project = os.environ["PROJECT"]
        wandb_utils.initialize(OmegaConf.to_container(cfg, resolve=True), entity, experiment_name, project)

    datamodule = hydra.utils.instantiate(
        cfg.data,
        num_workers=cfg.trainer.num_workers,
        global_batch_size=cfg.trainer.global_batch_size,
    )
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
        logger.info("Done!")


if __name__ == "__main__":
    main()
