"""Hydra entrypoint and Trainer wiring for SiT experiments."""

from __future__ import annotations

import os
import sys
from glob import glob

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hydra
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf, open_dict

import SiT.wandb_utils as wandb_utils
from SiT.lightning_module import SiTLightningModule, is_main_process
from utils.colorful_logger import info

# the first flag below was False when we tested this script but True makes A100 training faster.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _load_checkpoint_config(ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "hydra_config" not in checkpoint:
        raise ValueError("Checkpoint missing hydra_config. Only Lightning checkpoints are supported.")
    return checkpoint, OmegaConf.create(checkpoint["hydra_config"])


def _instantiate_datamodule(cfg: DictConfig):
    if cfg.data._target_ == "datamodules.image_datamodule.ImageFolderDataModule":
        return hydra.utils.instantiate(
            cfg.data,
            num_workers=cfg.trainer.num_workers,
            global_batch_size=cfg.trainer.global_batch_size,
        )
    return hydra.utils.instantiate(cfg.data)


def _resolve_model_num_classes(cfg: DictConfig, datamodule) -> int:
    resolved: int | None = None

    get_eval_config = getattr(datamodule, "get_eval_config", None)
    if callable(get_eval_config):
        try:
            resolved = int(get_eval_config().num_classes)
        except (AttributeError, RuntimeError, ValueError, TypeError):
            resolved = None

    if resolved is None:
        dataset = None
        for attr in ("train_dataset", "val_dataset", "test_dataset", "dataset"):
            dataset = getattr(datamodule, attr, None)
            if dataset is not None:
                break
        if dataset is not None:
            if hasattr(dataset, "class_ids"):
                resolved = len(list(dataset.class_ids))
            elif hasattr(dataset, "classes"):
                resolved = len(list(dataset.classes))

    if resolved is None or resolved <= 0:
        raise ValueError("Could not auto-resolve num_classes from datamodule/dataset.")

    configured = getattr(cfg.model, "num_classes", None)
    if configured is not None and int(configured) != resolved and is_main_process():
        info(f"Overriding model.num_classes: config={int(configured)} -> resolved={resolved}")

    with open_dict(cfg.model):
        cfg.model.num_classes = int(resolved)
    return int(resolved)


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

    datamodule = _instantiate_datamodule(cfg)
    datamodule.setup(stage="fit")
    resolved_num_classes = _resolve_model_num_classes(cfg, datamodule)

    if is_main_process():
        info(f"Resolved num_classes={resolved_num_classes}")

    if cfg.trainer.use_wandb and is_main_process():
        entity = cfg.trainer.wandb_entity
        project = cfg.trainer.wandb_project
        wandb_utils.initialize(OmegaConf.to_container(cfg, resolve=True), entity, experiment_name, project)

    module = SiTLightningModule(cfg, experiment_name, experiment_dir)

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
        val_check_interval=getattr(cfg.trainer, "val_check_interval", None),
        num_sanity_val_steps=getattr(cfg.trainer, "num_sanity_val_steps", 0),
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
