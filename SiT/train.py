"""Hydra entrypoint and Trainer wiring for SiT experiments."""

from __future__ import annotations

import os
import signal
import sys
from glob import glob

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hydra
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
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

    wandb_enabled = bool(cfg.trainer.use_wandb and is_main_process())
    signal_handlers_backup = {}
    signal_exit_code = 1

    def _on_termination(signum, _frame):
        nonlocal signal_exit_code
        signal_exit_code = 128 + int(signum)
        if wandb_enabled:
            info(f"Received signal {signum}; closing W&B run...")
            wandb_utils.finish(exit_code=signal_exit_code)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(signal_exit_code)

    if wandb_enabled:
        entity = cfg.trainer.wandb_entity
        project = cfg.trainer.wandb_project
        wandb_utils.initialize(OmegaConf.to_container(cfg, resolve=True), entity, experiment_name, project)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal_handlers_backup[sig] = signal.getsignal(sig)
            signal.signal(sig, _on_termination)

    module = SiTLightningModule(cfg, experiment_name, experiment_dir)

    best_ckpt_callback = None
    callbacks = []

    ckpt_every_steps = getattr(cfg.trainer, "ckpt_every", None)
    ckpt_every_epochs = getattr(cfg.trainer, "ckpt_every_n_epochs", None)
    if ckpt_every_steps is not None or ckpt_every_epochs is not None:
        checkpoint_kwargs = {
            "dirpath": checkpoint_dir,
            "save_top_k": -1,
            "save_last": False,
        }
        if ckpt_every_epochs is not None:
            checkpoint_kwargs.update(
                filename="{epoch:03d}-{step:07d}",
                every_n_epochs=int(ckpt_every_epochs),
            )
        else:
            checkpoint_kwargs.update(
                filename="{step:07d}",
                every_n_train_steps=int(ckpt_every_steps),
            )
        callbacks.append(ModelCheckpoint(**checkpoint_kwargs))

    best_ckpt_cfg = getattr(cfg.trainer, "best_checkpoint", None)
    if best_ckpt_cfg is not None and bool(best_ckpt_cfg.get("enabled", True)):
        best_ckpt_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=str(best_ckpt_cfg.get("filename", "best-{epoch:03d}-{step:07d}")),
            monitor=str(best_ckpt_cfg.get("monitor", "val_loss")),
            mode=str(best_ckpt_cfg.get("mode", "min")),
            save_top_k=int(best_ckpt_cfg.get("save_top_k", 1)),
            save_last=bool(best_ckpt_cfg.get("save_last", False)),
            auto_insert_metric_name=bool(best_ckpt_cfg.get("auto_insert_metric_name", False)),
        )
        callbacks.append(best_ckpt_callback)

    early_stopping_cfg = getattr(cfg.trainer, "early_stopping", None)
    if early_stopping_cfg is not None and bool(early_stopping_cfg.get("enabled", False)):
        callbacks.append(
            EarlyStopping(
                monitor=str(early_stopping_cfg.get("monitor", "val_loss")),
                mode=str(early_stopping_cfg.get("mode", "min")),
                patience=int(early_stopping_cfg.get("patience", 3)),
                min_delta=float(early_stopping_cfg.get("min_delta", 0.0)),
                strict=bool(early_stopping_cfg.get("strict", False)),
                verbose=bool(early_stopping_cfg.get("verbose", True)),
            )
        )

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
        check_val_every_n_epoch=getattr(cfg.trainer, "check_val_every_n_epoch", 1),
        num_sanity_val_steps=getattr(cfg.trainer, "num_sanity_val_steps", 0),
        callbacks=callbacks,
    )

    test_after_fit_cfg = getattr(cfg.trainer, "test_after_fit", None)
    run_test_after_fit = bool(test_after_fit_cfg is not None and test_after_fit_cfg.get("enabled", False))
    use_best_ckpt_for_test = bool(test_after_fit_cfg is not None and test_after_fit_cfg.get("use_best_checkpoint", True))
    strict_test_ckpt = bool(test_after_fit_cfg is not None and test_after_fit_cfg.get("strict", True))

    fit_succeeded = False
    test_succeeded = False
    run_succeeded = False
    try:
        trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.ckpt)
        fit_succeeded = True

        if run_test_after_fit:
            test_ckpt_path = None
            if use_best_ckpt_for_test:
                if best_ckpt_callback is None:
                    raise ValueError(
                        "test_after_fit.use_best_checkpoint=true requires trainer.best_checkpoint.enabled=true."
                    )
                best_model_path = str(best_ckpt_callback.best_model_path or "").strip()
                if not best_model_path:
                    monitor_name = str(getattr(best_ckpt_callback, "monitor", "unknown"))
                    message = (
                        "Best checkpoint path is empty after training. "
                        f"Monitor='{monitor_name}'. Ensure the monitored metric is logged during validation."
                    )
                    if strict_test_ckpt:
                        raise RuntimeError(message)
                    if is_main_process():
                        info(f"{message} Falling back to current in-memory weights for test.")
                else:
                    test_ckpt_path = best_model_path
                    if is_main_process():
                        info(
                            "Running test with best checkpoint "
                            f"(monitor={best_ckpt_callback.monitor}, path={best_model_path})"
                        )
            else:
                if is_main_process():
                    info("Running test with current in-memory weights.")

            trainer.test(model=module, datamodule=datamodule, ckpt_path=test_ckpt_path)
            test_succeeded = True
        else:
            test_succeeded = True

        run_succeeded = fit_succeeded and test_succeeded
    except KeyboardInterrupt:
        if is_main_process():
            info("Training interrupted by user.")
        raise
    finally:
        if wandb_enabled:
            if signal_handlers_backup:
                for sig, handler in signal_handlers_backup.items():
                    signal.signal(sig, handler)
            wandb_utils.finish(exit_code=0 if run_succeeded else signal_exit_code)

    if is_main_process():
        info("Done!")


if __name__ == "__main__":
    main()
