from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time

import hydra
from hydra.utils import instantiate
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, TensorDataset

from .console import info, ok
from .posterior import covariance_profiles, draw, exact_velocity, generator, heun, make_centers, path_batch, sliced_w2


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate(cfg: DictConfig) -> dict:
    OmegaConf.resolve(cfg)
    if cfg.geometry not in cfg.radii or cfg.profile not in ("low", "mid", "high"):
        raise ValueError("Unknown geometry or profile.")
    positive = ["max_steps", "validation_interval", "batch_size", "eval_samples", "test_samples", "sample_count", "projections", "solver_steps", "cpu_threads", "log_interval"]
    if any(int(cfg[k]) <= 0 for k in positive) or not 0 < cfg.training_minutes <= 25:
        raise ValueError("Invalid positive counts or training time cap.")
    if cfg.max_steps % cfg.validation_interval:
        raise ValueError("max_steps must be divisible by validation_interval.")
    if cfg.components < 2 or cfg.learning_rate <= 0:
        raise ValueError("Invalid components or learning rate.")
    if set(cfg.suite.geometries) - set(cfg.radii) or set(cfg.suite.profiles) - {"low", "mid", "high"}:
        raise ValueError("Invalid suite conditions.")
    if not 60 * cfg.training_minutes < cfg.suite.watchdog_seconds <= 1800:
        raise ValueError("Process watchdog must leave evaluation headroom and not exceed 30 minutes.")
    target_logdet = cfg.dim * math.log(cfg.trace / cfg.dim) - cfg.logdet_gap
    profiles = covariance_profiles(cfg.dim, cfg.trace, target_logdet, list(cfg.multiplicities))
    return profiles


class PosteriorFlow(L.LightningModule):
    def __init__(self, cfg: DictConfig, variances, out: Path):
        super().__init__()
        self.cfg = cfg
        self.out = out
        self.network = instantiate(cfg.model)
        mu = make_centers(cfg.dim, cfg.components, cfg.radii[cfg.geometry], cfg.geometry_seed)
        self.register_buffer("mu", mu.float())
        self.register_buffer("variances", torch.as_tensor(variances, dtype=torch.float32))
        self.started = time.monotonic()

    def forward(self, x, t):
        return self.network(x, t.flatten(), torch.zeros(len(x), device=x.device, dtype=torch.long))

    def on_fit_start(self):
        self.train_rng = generator(self.cfg.seed, "train", self.device)
        self.started = time.monotonic()

    def training_step(self, batch, batch_idx):
        x, t, target = path_batch(self.mu, self.variances, self.cfg.batch_size, self.train_rng)
        loss = (self(x, t) - target).square().mean()
        self.log("train/fm_loss", loss, on_step=True, on_epoch=False, batch_size=self.cfg.batch_size)
        if (self.global_step + 1) % self.cfg.log_interval == 0:
            self.record("train", {"fm_loss": float(loss.detach())}, self.global_step + 1)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.learning_rate, weight_decay=0)

    def record(self, split, metrics, step=None):
        path = self.out / "metrics" / f"{split}.csv"
        row = {"step": self.global_step if step is None else step, "seconds": time.monotonic() - self.started, **metrics}
        exists = path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    @torch.no_grad()
    def velocity_metrics(self, split: str, count: int):
        rng = generator(self.cfg.seed, split + ":velocity", self.device)
        total = torch.zeros(3, device=self.device, dtype=torch.float64)
        for offset in range(0, count, 512):
            x, t, target = path_batch(self.mu, self.variances, min(512, count-offset), rng)
            truth = exact_velocity(x, t, self.mu, self.variances)
            prediction = self(x, t)
            total += torch.stack([(prediction-truth).square().sum(), truth.square().sum(), (prediction-target).square().sum()]).double()
        mse, energy, fm_loss = (total / (count * self.cfg.dim)).tolist()
        return {"oracle_mse": mse, "relative_oracle_mse": mse / max(energy, 1e-12), "oracle_energy": energy, "fm_loss": fm_loss}

    def validation_step(self, batch, batch_idx):
        metrics = self.velocity_metrics("validation", self.cfg.eval_samples)
        self.log("val_oracle_mse", metrics["oracle_mse"], batch_size=self.cfg.eval_samples)
        self.record("validation", metrics)
        info(f"step={self.global_step} validation oracle MSE={metrics['oracle_mse']:.6g}")

    @torch.no_grad()
    def sample_metrics(self):
        n = self.cfg.sample_count
        rng = generator(self.cfg.seed, "test:noise", self.device)
        noise = torch.randn(n, self.cfg.dim, device=self.device, generator=rng)
        generated = heun(self, noise, self.cfg.solver_steps)
        oracle = heun(lambda x, t: exact_velocity(x, t, self.mu, self.variances), noise, self.cfg.solver_steps)
        real = draw(self.mu, self.variances, n, generator(self.cfg.seed, "test:reference", self.device))
        real2 = draw(self.mu, self.variances, n, generator(self.cfg.seed, "test:reference2", self.device))
        directions = torch.randn(self.cfg.dim, self.cfg.projections, device=self.device, generator=generator(self.cfg.seed, "test:projections", self.device))
        directions /= directions.norm(dim=0, keepdim=True)
        torch.save({"generated": generated.cpu(), "reference": real.cpu(), "oracle": oracle.cpu()}, self.out / "artifacts" / "samples.pt")
        return {"swd": sliced_w2(generated, real, directions), "real_real_swd": sliced_w2(real2, real, directions), "oracle_sampler_swd": sliced_w2(oracle, real, directions), "solver_nfe": 2*self.cfg.solver_steps}


def plot_history(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor="white")
    for ax, split, key, color in zip(axes, ["train", "validation"], ["fm_loss", "oracle_mse"], ["#E07A5F", "#3D405B"]):
        rows = list(csv.DictReader((out / "metrics" / f"{split}.csv").open()))
        ax.plot([int(r["step"]) for r in rows], [float(r[key]) for r in rows], color=color)
        ax.set(xlabel="Optimizer steps", ylabel=key, title=split)
        ax.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(out / "plots" / "learning_curves.png", dpi=160)
    plt.close(fig)


def train(cfg: DictConfig, profiles: dict) -> None:
    if not cfg.run_id:
        raise ValueError("Explicit unique run_id is required.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or "3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Exactly one visible RTX 3090 is required.")
    out = Path(cfg.output_root).resolve() / cfg.run_id
    out.mkdir(parents=True, exist_ok=False)
    for folder in ["metrics", "plots", "reports", "artifacts", "checkpoints"]:
        (out / folder).mkdir()
    OmegaConf.save(cfg, out / "config.yaml")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if dirty:
        raise RuntimeError("Refusing training with modified tracked source.")
    mu = make_centers(cfg.dim, cfg.components, cfg.radii[cfg.geometry], cfg.geometry_seed)
    s = torch.as_tensor(profiles[cfg.profile], dtype=torch.float64)
    diagnostics = {"profile": cfg.profile, "geometry": cfg.geometry, "anisotropy": float(s.log().var(unbiased=False)), "trace": float(s.sum()), "logdet": float(s.log().sum()), "rate": float(0.5*(mu.square().sum(1).mean()+s.sum()-cfg.dim-s.log().sum())), "aggregate_trace": float(mu.square().sum(1).mean()+s.sum()), "variances": s.tolist(), "centers": mu.tolist()}
    write_json(out / "artifacts" / "distribution.json", diagnostics)
    write_json(out / "artifacts" / "provenance.json", {"git_sha": sha, "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "packages": {p:importlib.metadata.version(p) for p in ["torch", "lightning", "hydra-core", "scipy"]}})
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision("highest")
    module = PosteriorFlow(cfg, profiles[cfg.profile], out)
    checkpoint = ModelCheckpoint(dirpath=out / "checkpoints", filename="best-{step}", monitor="val_oracle_mse", mode="min", save_top_k=1, save_last=True, auto_insert_metric_name=False)
    trainer = L.Trainer(accelerator="gpu", devices=1, precision="32-true", max_steps=cfg.max_steps, max_epochs=1, max_time={"minutes": cfg.training_minutes}, callbacks=[checkpoint], logger=CSVLogger(str(out / "metrics"), name="lightning"), num_sanity_val_steps=0, val_check_interval=cfg.validation_interval, log_every_n_steps=cfg.log_interval, enable_progress_bar=False, enable_model_summary=False, deterministic=True)
    started = time.monotonic()
    train_loader = DataLoader(TensorDataset(torch.arange(cfg.max_steps)), batch_size=1, num_workers=0)
    val_loader = DataLoader(TensorDataset(torch.arange(1)), batch_size=1, num_workers=0)
    trainer.fit(module, train_loader, val_loader)
    training_seconds = time.monotonic()-started
    if trainer.global_step != cfg.max_steps:
        write_json(out / "status.json", {"state": "incomplete", "steps": trainer.global_step, "training_seconds": training_seconds})
        raise RuntimeError("Training time limit reached before fixed budget; stopping suite.")
    # Lightning returns the module to CPU after teardown; evaluation stays on 3090.
    module.to("cuda").eval()
    result = {"run_id": cfg.run_id, "geometry": cfg.geometry, "profile": cfg.profile, "seed": cfg.seed, "steps": trainer.global_step, "training_seconds": training_seconds, "checkpoint_policy": "final fixed-step checkpoint; test never selects checkpoint", **module.velocity_metrics("test", cfg.test_samples), **module.sample_metrics()}
    write_json(out / "metrics" / "test.json", result)
    plot_history(out)
    (out / "reports" / "summary.md").write_text(f"# {cfg.run_id}\n\nFinal fixed-step evaluation, {cfg.max_steps} updates.\n\n```json\n{json.dumps(result, indent=2)}\n```\n\nInterpret within a geometry; this does not establish downstream VAE benefits.\n")
    write_json(out / "status.json", {"state": "completed", "training_seconds": training_seconds})
    ok(f"Completed {cfg.run_id}: test oracle MSE={result['oracle_mse']:.6g}, SWD={result['swd']:.6g}")


@hydra.main(version_base=None, config_path="../../conf", config_name="posterior")
def main(cfg: DictConfig) -> None:
    torch.set_num_threads(cfg.cpu_threads)
    torch.set_num_interop_threads(2)
    profiles = validate(cfg)
    if cfg.action == "validate":
        ok("Configuration resolved; covariance constraints passed. No model or dataset loaded.")
    elif cfg.action == "train":
        train(cfg, profiles)
    elif cfg.action == "suite":
        from .suite import run_suite
        run_suite(cfg)
    else:
        raise ValueError(f"Unknown action: {cfg.action}")


if __name__ == "__main__":
    main()
