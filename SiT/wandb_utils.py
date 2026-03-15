import os
import argparse
from collections.abc import Mapping
import math
from typing import Iterable

import torch
import torch.distributed as dist
import wandb
from torchvision.utils import make_grid

_WANDB_FINISHED = True


def is_main_process():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True

def namespace_to_dict(namespace):
    if isinstance(namespace, Mapping):
        return {k: namespace_to_dict(v) for k, v in namespace.items()}
    if isinstance(namespace, argparse.Namespace):
        return {
            k: namespace_to_dict(v) if isinstance(v, (argparse.Namespace, Mapping)) else v
            for k, v in vars(namespace).items()
        }
    return namespace


def initialize(
    args,
    entity,
    exp_name,
    project_name,
    *,
    group=None,
    job_type=None,
    tags=None,
):
    global _WANDB_FINISHED
    config_dict = namespace_to_dict(args)
    # Login: try with env var first, fallback to interactive
    wandb_key = os.environ.get("WANDB_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
    # else wandb will prompt for login interactively
    _WANDB_FINISHED = False
    return wandb.init(
        entity=entity,
        project=project_name,
        name=exp_name,
        config=config_dict,
        group=group,
        job_type=job_type,
        tags=tags,
        reinit=True,
    )


def finish(exit_code: int = 0):
    global _WANDB_FINISHED
    if _WANDB_FINISHED:
        return
    _WANDB_FINISHED = True

    try:
        if wandb.run is not None:
            wandb.finish(exit_code=exit_code)
    finally:
        # Ensure background services are torn down on interrupts (Ctrl+C/SIGTERM).
        try:
            wandb.teardown()
        except Exception:
            pass


def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(sample, step=None):
    if is_main_process():
        sample = array2grid(sample)
        wandb.log({f"samples": wandb.Image(sample), "train_step": step})


def log_image_file(key, image_path, step=None):
    if is_main_process():
        wandb.log({key: wandb.Image(image_path), "train_step": step}, step=step)


def log_artifact_files(
    name: str,
    artifact_type: str,
    files: Iterable[tuple[str, str]],
    metadata: dict | None = None,
    aliases: list[str] | None = None,
):
    if not is_main_process():
        return False
    if wandb.run is None:
        return False

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
    added_files = 0
    for local_path, artifact_path in files:
        if os.path.exists(local_path):
            artifact.add_file(local_path, name=artifact_path)
            added_files += 1
    if added_files == 0:
        return False
    wandb.log_artifact(artifact, aliases=aliases or [])
    return True


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1,1))
    x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x
