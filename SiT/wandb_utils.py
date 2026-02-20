import wandb
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import os
import argparse
from collections.abc import Mapping
import hashlib
import math
from typing import Iterable


def is_main_process():
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    if isinstance(namespace, Mapping):
        return {k: namespace_to_dict(v) for k, v in namespace.items()}
    if isinstance(namespace, argparse.Namespace):
        return {
            k: namespace_to_dict(v) if isinstance(v, (argparse.Namespace, Mapping)) else v
            for k, v in vars(namespace).items()
        }
    return namespace


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(args, entity, exp_name, project_name):
    config_dict = namespace_to_dict(args)
    # Login: try with env var first, fallback to interactive
    wandb_key = os.environ.get("WANDB_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
    # else wandb will prompt for login interactively
    wandb.init(
        entity=entity,
        project=project_name,
        name=exp_name,
        config=config_dict,
        id=generate_run_id(exp_name),
        resume="allow",
    )


def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(sample, step=None):
    if is_main_process():
        sample = array2grid(sample)
        wandb.log({f"samples": wandb.Image(sample), "train_step": step})


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
