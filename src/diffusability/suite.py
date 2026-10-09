"""Sequential, fail-closed experiment queue and paired descriptive report."""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

from omegaconf import OmegaConf

from .console import info, ok
from .experiment import write_json


def summarize(root: Path, completed: list[str]) -> None:
    rows = []
    for name in completed:
        row = json.loads((root / name / "metrics/test.json").read_text())
        distribution = json.loads((root / name / "artifacts/distribution.json").read_text())
        rows.append({**row, "anisotropy": distribution["anisotropy"]})
    if not rows:
        return
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    effects = {}
    for geometry in sorted({r["geometry"] for r in rows}):
        paired = []
        low, high = [], []
        seeds = sorted({r["seed"] for r in rows if r["geometry"] == geometry})
        for seed in seeds:
            pair = {r["profile"]: r for r in rows if r["geometry"] == geometry and r["seed"] == seed}
            if "low" in pair and "high" in pair:
                lo, hi = pair["low"]["oracle_mse"], pair["high"]["oracle_mse"]
                paired.append({"seed": seed, "log_high_over_low": math.log(hi / lo)})
                low.append(lo)
                high.append(hi)
        if paired:
            values = [p["log_high_over_low"] for p in paired]
            mean = sum(values) / len(values)
            effects[geometry] = {"pairs": paired, "mean_log_effect": mean,
                "sd_log_effect": (sum((v-mean)**2 for v in values)/(len(values)-1))**0.5 if len(values)>1 else None,
                "relative_mean_error_reduction": 1-sum(low)/sum(high),
                "provisional_directional_support": len(values)>=3 and min(values)>0 and 1-sum(low)/sum(high)>=0.1}
    write_json(root / "paired_effects.json", effects)
    (root / "summary.md").write_text(
        f"# Posterior covariance experiment\n\nCompleted runs: {len(rows)}. "
        "Paired effects are descriptive; three seeds do not provide a powered significance test. "
        "Positive log(high/low error) favors lower anisotropy. Compare within each geometry.\n\n"
        f"```json\n{json.dumps(effects, indent=2)}\n```\n\n"
        "Interpret the full profile intervention, not a universal law about a scalar. "
        "No real-VAE downstream claim follows from this toy experiment.\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")
    for ax, geometry in zip(axes, ["overlap", "separated"]):
        for seed, color in zip(sorted({r["seed"] for r in rows}), ["#E07A5F", "#3D405B", "#81B29A"]):
            points = sorted([r for r in rows if r["geometry"]==geometry and r["seed"]==seed], key=lambda r:r["anisotropy"])
            ax.plot([r["anisotropy"] for r in points], [r["oracle_mse"] for r in points], "o-", color=color, label=f"seed {seed}")
        ax.set(xlabel="Posterior Var(log variance)", ylabel="Test oracle velocity MSE / coordinate", title=geometry)
        ax.set_facecolor("white")
        ax.legend()
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)


def run_suite(cfg) -> None:
    if not cfg.run_id:
        raise ValueError("An explicit unique suite run_id is required.")
    root = Path(cfg.output_root).resolve() / cfg.run_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    base = OmegaConf.to_container(cfg, resolve=True)
    base["defaults"] = [{"override hydra/job_logging": "disabled"}, {"override hydra/hydra_logging": "disabled"}, "_self_"]
    base["hydra"] = {"run": {"dir": "."}, "output_subdir": None, "job": {"chdir": False}}
    OmegaConf.save(OmegaConf.create(base), root / "base.yaml")
    matrix = list(itertools.product(cfg.suite.geometries, cfg.suite.seeds, cfg.suite.profiles))
    completed = []
    state = {"state": "running", "pid": os.getpid(), "total": len(matrix), "completed": completed}
    for geometry, seed, profile in matrix:
        name = f"{geometry}-{profile}-seed{seed}"
        command = [sys.executable, "-m", "diffusability.experiment", "--config-path", str(root), "--config-name", "base",
                   "action=train", f"run_id={cfg.run_id}/{name}", f"geometry={geometry}", f"profile={profile}", f"seed={seed}"]
        state.update(current=name, command=command)
        write_json(root / "status.json", state)
        info(f"Starting {len(completed)+1}/{len(matrix)}: {name}")
        with (root / "logs" / f"{name}.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state["child_pid"] = child.pid
            write_json(root / "status.json", state)
            try:
                result = child.wait(timeout=cfg.suite.watchdog_seconds)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                result = -1
        if result != 0 or not (root / name / "metrics/test.json").exists():
            state.update(state="failed", returncode=result)
            write_json(root / "status.json", state)
            summarize(root, completed)
            raise RuntimeError(f"Queue stopped at {name}; inspect {root / 'logs' / (name+'.log')}")
        completed.append(name)
        write_json(root / "status.json", state)
        summarize(root, completed)
    state.update(state="completed", current=None)
    write_json(root / "status.json", state)
    ok(f"All {len(completed)} runs completed: {root}")
