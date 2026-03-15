#!/usr/bin/env python

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import hydra
from lightning import Trainer, seed_everything
from omegaconf import DictConfig, open_dict

from SiT.lightning_module import SiTLightningModule
from SiT.train import _instantiate_datamodule, _load_checkpoint_config, _resolve_model_num_classes

CKPT_RE = re.compile(r"^epoch=(\d+)-step=(\d+)\.ckpt$")


@dataclass(frozen=True)
class CkptMeta:
    path: Path
    epoch: int
    step: int


def _collect_target_checkpoints(checkpoint_dir: Path, epoch_stride: int) -> list[CkptMeta]:
    selected: list[CkptMeta] = []
    for ckpt in sorted(checkpoint_dir.glob("epoch=*-step=*.ckpt")):
        match = CKPT_RE.match(ckpt.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        step = int(match.group(2))
        if (epoch + 1) % epoch_stride == 0:
            selected.append(CkptMeta(path=ckpt, epoch=epoch, step=step))
    return selected


def _evaluate_checkpoint(
    *,
    ckpt_path: Path,
    run_dir: Path,
    epoch: int,
    step: int,
    seed: int,
    w2_samples_per_class: int,
    other_samples_per_class: int,
    num_projections: int,
) -> None:
    _, cfg = _load_checkpoint_config(str(ckpt_path))
    seed_everything(seed, workers=True)

    with open_dict(cfg):
        cfg.ckpt = str(ckpt_path)
        cfg.trainer.use_wandb = False
        cfg.trainer.devices = 1
        cfg.trainer.strategy = "auto"
        cfg.trainer.num_workers = 0
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.data.val_samples_per_class = int(other_samples_per_class)
        cfg.data.test_samples_per_class = int(other_samples_per_class)

        if "exact_w2" not in cfg.data.metrics:
            cfg.data.metrics.exact_w2 = {}
        if "energy_distance" not in cfg.data.metrics:
            cfg.data.metrics.energy_distance = {}
        if "feature_mmd" not in cfg.data.metrics:
            cfg.data.metrics.feature_mmd = {}
        if "swd" not in cfg.data.metrics:
            cfg.data.metrics.swd = {}

        cfg.data.metrics.swd.enabled = True
        cfg.data.metrics.swd.num_projections = int(num_projections)
        cfg.data.metrics.exact_w2.enabled = True
        cfg.data.metrics.exact_w2.max_samples = int(w2_samples_per_class)
        cfg.data.metrics.energy_distance.enabled = False
        cfg.data.metrics.feature_mmd.enabled = True
        cfg.data.metrics.feature_mmd.max_samples = int(other_samples_per_class)

    datamodule = _instantiate_datamodule(cfg)
    datamodule.setup(stage="fit")
    _resolve_model_num_classes(cfg, datamodule)

    module = SiTLightningModule(cfg=cfg, experiment_name=run_dir.name, experiment_dir=str(run_dir))
    module._regen_epoch_override = int(epoch)
    module._regen_step_override = int(step)

    trainer = Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision=cfg.trainer.precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.test(model=module, datamodule=datamodule, ckpt_path=str(ckpt_path), verbose=False)


def _process_run(
    *,
    run_dir: Path,
    epoch_stride: int,
    w2_samples_per_class: int,
    other_samples_per_class: int,
    num_projections: int,
    seed: int,
    reset_test_loss: bool,
) -> None:
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"[skip] {run_dir.name}: missing checkpoints/")
        return

    selected = _collect_target_checkpoints(checkpoint_dir, epoch_stride=epoch_stride)
    if not selected:
        print(f"[skip] {run_dir.name}: no checkpoints matching stride={epoch_stride}")
        return

    test_path = run_dir / "metrics" / "test_loss_by_class.json"
    if reset_test_loss and test_path.exists():
        test_path.unlink()

    print(f"[run ] {run_dir.name}: {len(selected)} checkpoints (stride={epoch_stride})")
    for meta in selected:
        print(f"       - test {meta.path.name}")
        _evaluate_checkpoint(
            ckpt_path=meta.path,
            run_dir=run_dir,
            epoch=meta.epoch,
            step=meta.step,
            seed=seed + meta.epoch,
            w2_samples_per_class=w2_samples_per_class,
            other_samples_per_class=other_samples_per_class,
            num_projections=num_projections,
        )


def _iter_run_dirs(root: Path) -> list[Path]:
    if (root / "checkpoints").exists():
        return [root]
    return sorted(path for path in root.iterdir() if path.is_dir())


def _resolve_roots(raw_roots: list[str]) -> list[Path]:
    return [Path(hydra.utils.to_absolute_path(str(root))) for root in raw_roots]


@hydra.main(config_path="../conf/tools", config_name="evaluate_checkpoint_metrics", version_base=None)
def main(cfg: DictConfig) -> None:
    roots = _resolve_roots(list(cfg.roots))

    run_dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"[skip] missing root: {root}")
            continue
        run_dirs.extend(_iter_run_dirs(root))

    if not run_dirs:
        raise RuntimeError("No run directories found under provided roots.")

    for run_dir in run_dirs:
        _process_run(
            run_dir=run_dir,
            epoch_stride=int(cfg.epoch_stride),
            w2_samples_per_class=int(cfg.w2_samples_per_class),
            other_samples_per_class=int(cfg.other_samples_per_class),
            num_projections=int(cfg.num_projections),
            seed=int(cfg.seed),
            reset_test_loss=bool(cfg.reset_test_loss),
        )

    print("[done] Checkpoint metric evaluation complete.")


if __name__ == "__main__":
    main()
