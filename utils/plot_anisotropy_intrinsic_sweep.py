from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
MMD_RE = re.compile(r"step=(\d+).+?val/feature_mmd_mean:\s*([0-9eE+\-.]+)")


@dataclass
class SweepRun:
    experiment_dir: Path
    experiment_name: str
    intrinsic_dim: int
    anisotropy_max_scale: float
    created_at_utc: str
    val_series: list[dict[str, Any]]
    mmd_by_step: dict[int, float]
    final_val_loss: float
    final_feature_mmd: float
    wandb_run_dir: Path | None
    is_partial_run: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot validation loss and Feature-MMD for the anisotropy/intrinsic-dimension DiT sweep."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/anisotropy_intrinsic_sweep"),
        help="Root directory containing per-run results folders.",
    )
    parser.add_argument(
        "--wandb-root",
        type=Path,
        default=Path("wandb"),
        help="Root directory containing local W&B runs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots and summary CSV. Defaults to <results-root>/plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text).replace("\r", "\n")


def parse_mmd_series(output_log: Path) -> dict[int, float]:
    text = strip_ansi(output_log.read_text(encoding="utf-8", errors="ignore"))
    mmd_by_step: dict[int, float] = {}
    for step_text, value_text in MMD_RE.findall(text):
        mmd_by_step[int(step_text)] = float(value_text)
    return mmd_by_step


def load_wandb_config(config_path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    intrinsic_dim = cfg.get("intrinsic_dim", {}).get("value")
    anisotropy = cfg.get("anisotropy_max_scale", {}).get("value")
    return {
        "intrinsic_dim": int(intrinsic_dim) if intrinsic_dim is not None else None,
        "anisotropy_max_scale": float(anisotropy) if anisotropy is not None else None,
    }


def discover_wandb_runs(wandb_root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not wandb_root.exists():
        return runs

    for run_dir in sorted(wandb_root.glob("run-*")):
        files_dir = run_dir / "files"
        output_log = files_dir / "output.log"
        if not output_log.exists():
            continue

        run_info: dict[str, Any] = {
            "run_dir": run_dir,
            "output_log": output_log,
            "summary_path": files_dir / "wandb-summary.json",
            "config_path": files_dir / "config.yaml",
        }

        if run_info["config_path"].exists():
            run_info.update(load_wandb_config(run_info["config_path"]))
        else:
            run_info.update({"intrinsic_dim": None, "anisotropy_max_scale": None})

        runs.append(run_info)
    return runs


def find_matching_wandb_run(
    experiment_name: str,
    intrinsic_dim: int,
    anisotropy_max_scale: float,
    wandb_runs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    name_matches: list[dict[str, Any]] = []
    exact_name_matches: list[dict[str, Any]] = []
    param_matches: list[dict[str, Any]] = []
    for run in wandb_runs:
        output_text = run["output_log"].read_text(encoding="utf-8", errors="ignore")
        if experiment_name in output_text:
            name_matches.append(run)

        run_intrinsic = run["intrinsic_dim"]
        run_anisotropy = run["anisotropy_max_scale"]
        if run_intrinsic == intrinsic_dim and run_anisotropy is not None and math.isclose(
            float(run_anisotropy),
            float(anisotropy_max_scale),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            param_matches.append(run)
            if experiment_name in output_text:
                exact_name_matches.append(run)

    if exact_name_matches:
        return sorted(exact_name_matches, key=lambda item: item["run_dir"].name)[-1]
    if name_matches:
        return sorted(name_matches, key=lambda item: item["run_dir"].name)[-1]
    if param_matches:
        return sorted(param_matches, key=lambda item: item["run_dir"].name)[-1]
    return None


def load_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_sweep_runs(results_root: Path, wandb_root: Path) -> list[SweepRun]:
    wandb_runs = discover_wandb_runs(wandb_root)
    runs: list[SweepRun] = []

    for experiment_dir in sorted(results_root.glob("*")):
        if not experiment_dir.is_dir():
            continue

        class_registry_path = experiment_dir / "metrics" / "class_registry.json"
        val_loss_path = experiment_dir / "metrics" / "val_loss_by_class.jsonl"
        if not class_registry_path.exists() or not val_loss_path.exists():
            continue

        class_registry = load_json(class_registry_path)
        class_zero = class_registry["classes"]["0"]["params"]
        intrinsic_dim = int(class_zero["d"])
        anisotropy_max_scale = float(class_zero["anisotropy"]["max_scale"])
        val_series = sorted(load_jsonl(val_loss_path), key=lambda row: int(row["step"]))
        if not val_series:
            continue

        wandb_run = find_matching_wandb_run(
            experiment_name=experiment_dir.name,
            intrinsic_dim=intrinsic_dim,
            anisotropy_max_scale=anisotropy_max_scale,
            wandb_runs=wandb_runs,
        )

        mmd_by_step: dict[int, float] = {}
        summary: dict[str, Any] = {}
        wandb_run_dir: Path | None = None
        if wandb_run is not None:
            wandb_run_dir = Path(wandb_run["run_dir"])
            mmd_by_step = parse_mmd_series(Path(wandb_run["output_log"]))
            summary = load_summary(Path(wandb_run["summary_path"]))

        last_val = float(val_series[-1]["overall_val_loss"])
        final_mmd = float(summary.get("val/feature_mmd_mean", np.nan))
        if math.isnan(final_mmd) and mmd_by_step:
            final_mmd = float(mmd_by_step[max(mmd_by_step)])

        runs.append(
            SweepRun(
                experiment_dir=experiment_dir,
                experiment_name=experiment_dir.name,
                intrinsic_dim=intrinsic_dim,
                anisotropy_max_scale=anisotropy_max_scale,
                created_at_utc=str(class_registry.get("created_at_utc", "")),
                val_series=val_series,
                mmd_by_step=mmd_by_step,
                final_val_loss=last_val,
                final_feature_mmd=final_mmd,
                wandb_run_dir=wandb_run_dir,
            )
        )

    return deduplicate_runs(runs)


def deduplicate_runs(runs: list[SweepRun]) -> list[SweepRun]:
    by_key: dict[tuple[int, float], SweepRun] = {}
    for run in runs:
        key = (run.intrinsic_dim, run.anisotropy_max_scale)
        current = by_key.get(key)
        if current is None:
            by_key[key] = run
            continue

        run_score = (
            len(run.val_series),
            0 if math.isnan(run.final_feature_mmd) else 1,
            run.created_at_utc,
            run.experiment_name,
        )
        current_score = (
            len(current.val_series),
            0 if math.isnan(current.final_feature_mmd) else 1,
            current.created_at_utc,
            current.experiment_name,
        )
        if run_score > current_score:
            by_key[key] = run

    return sorted(by_key.values(), key=lambda item: (item.intrinsic_dim, item.anisotropy_max_scale))


def build_summary_rows(runs: list[SweepRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        best_loss_record = min(run.val_series, key=lambda row: float(row["overall_val_loss"]))
        best_mmd = min(run.mmd_by_step.values()) if run.mmd_by_step else np.nan
        rows.append(
            {
                "experiment_name": run.experiment_name,
                "intrinsic_dim": run.intrinsic_dim,
                "anisotropy_max_scale": run.anisotropy_max_scale,
                "num_val_points": len(run.val_series),
                "final_epoch": int(run.val_series[-1]["epoch"]),
                "final_step": int(run.val_series[-1]["step"]),
                "final_val_loss": run.final_val_loss,
                "final_feature_mmd": run.final_feature_mmd,
                "best_val_loss": float(best_loss_record["overall_val_loss"]),
                "best_val_loss_epoch": int(best_loss_record["epoch"]),
                "best_feature_mmd": float(best_mmd) if not math.isnan(best_mmd) else np.nan,
                "is_partial_run": run.is_partial_run,
                "has_wandb_match": run.wandb_run_dir is not None,
                "wandb_run_dir": str(run.wandb_run_dir) if run.wandb_run_dir else "",
                "experiment_dir": str(run.experiment_dir),
            }
        )
    return rows


def save_summary_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    headers = list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        values = []
        for key in headers:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.10g}")
            else:
                text = str(value)
                if "," in text or '"' in text:
                    text = '"' + text.replace('"', '""') + '"'
                values.append(text)
        lines.append(",".join(values))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_matrix(
    runs: list[SweepRun],
    metric_name: str,
) -> tuple[np.ndarray, list[int], list[float]]:
    intrinsic_values = sorted({run.intrinsic_dim for run in runs})
    anisotropy_values = sorted({run.anisotropy_max_scale for run in runs})
    matrix = np.full((len(intrinsic_values), len(anisotropy_values)), np.nan, dtype=float)
    d_to_idx = {value: index for index, value in enumerate(intrinsic_values)}
    a_to_idx = {value: index for index, value in enumerate(anisotropy_values)}

    for run in runs:
        row_idx = d_to_idx[run.intrinsic_dim]
        col_idx = a_to_idx[run.anisotropy_max_scale]
        matrix[row_idx, col_idx] = float(getattr(run, metric_name))

    return matrix, intrinsic_values, anisotropy_values


def annotate_heatmap(ax: plt.Axes, matrix: np.ndarray) -> None:
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = "NA" if np.isnan(value) else f"{value:.3f}"
            color = "white" if not np.isnan(value) and value > np.nanmedian(matrix) else "black"
            ax.text(col, row, text, ha="center", va="center", color=color, fontsize=8)


def plot_heatmap(
    runs: list[SweepRun],
    metric_name: str,
    title: str,
    colorbar_label: str,
    out_path: Path,
    dpi: int,
) -> None:
    matrix, intrinsic_values, anisotropy_values = metric_matrix(runs, metric_name)

    fig, ax = plt.subplots(figsize=(1.8 * len(anisotropy_values) + 1.5, 1.4 * len(intrinsic_values) + 1.5))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#dddddd")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    annotate_heatmap(ax, matrix)

    ax.set_xticks(np.arange(len(anisotropy_values)), labels=[f"{value:g}" for value in anisotropy_values])
    ax.set_yticks(np.arange(len(intrinsic_values)), labels=[str(value) for value in intrinsic_values])
    ax.set_xlabel("Anisotropy max_scale")
    ax.set_ylabel("Intrinsic dimension d")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_loss_vs_mmd_scatter(runs: list[SweepRun], out_path: Path, dpi: int) -> None:
    intrinsic_values = sorted({run.intrinsic_dim for run in runs})
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(intrinsic_values), 1)))
    color_map = {d: colors[idx] for idx, d in enumerate(intrinsic_values)}

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for run in sorted(runs, key=lambda item: (item.intrinsic_dim, item.anisotropy_max_scale)):
        label = f"d={run.intrinsic_dim}, a={run.anisotropy_max_scale:g}"
        ax.scatter(
            run.final_val_loss,
            run.final_feature_mmd,
            s=40 + 15 * math.log2(max(run.anisotropy_max_scale, 1.0)),
            color=color_map[run.intrinsic_dim],
            alpha=0.9,
        )
        ax.annotate(label, (run.final_val_loss, run.final_feature_mmd), xytext=(5, 5), textcoords="offset points", fontsize=8)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color_map[d], label=f"d={d}", markersize=8)
        for d in intrinsic_values
    ]
    ax.legend(handles=handles, title="Intrinsic dim", loc="best")
    ax.set_xlabel("Final validation loss")
    ax.set_ylabel("Final val/feature_mmd_mean")
    ax.set_title("Validation loss vs Feature-MMD")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_trajectories(runs: list[SweepRun], out_path: Path, dpi: int) -> None:
    intrinsic_values = sorted({run.intrinsic_dim for run in runs})
    fig, axes = plt.subplots(
        nrows=len(intrinsic_values),
        ncols=2,
        figsize=(12, max(3.6 * len(intrinsic_values), 4.2)),
        squeeze=False,
        sharex=False,
    )

    cmap = plt.get_cmap("plasma")
    all_anis = sorted({run.anisotropy_max_scale for run in runs})
    color_map = {a: cmap(index) for a, index in zip(all_anis, np.linspace(0.15, 0.95, len(all_anis)))}

    for row_idx, intrinsic_dim in enumerate(intrinsic_values):
        subset = sorted(
            [run for run in runs if run.intrinsic_dim == intrinsic_dim],
            key=lambda item: item.anisotropy_max_scale,
        )
        loss_ax = axes[row_idx][0]
        mmd_ax = axes[row_idx][1]

        for run in subset:
            epochs = [int(record["epoch"]) for record in run.val_series]
            losses = [float(record["overall_val_loss"]) for record in run.val_series]
            mmds = [run.mmd_by_step.get(int(record["step"]), np.nan) for record in run.val_series]
            label = f"a={run.anisotropy_max_scale:g}"
            color = color_map[run.anisotropy_max_scale]

            loss_ax.plot(epochs, losses, marker="o", ms=3, lw=1.8, color=color, label=label)
            mmd_ax.plot(epochs, mmds, marker="o", ms=3, lw=1.8, color=color, label=label)

        loss_ax.set_title(f"d={intrinsic_dim} | val loss")
        loss_ax.set_ylabel("Validation loss")
        loss_ax.grid(alpha=0.25)

        mmd_ax.set_title(f"d={intrinsic_dim} | Feature-MMD")
        mmd_ax.grid(alpha=0.25)

        if row_idx == len(intrinsic_values) - 1:
            loss_ax.set_xlabel("Epoch")
            mmd_ax.set_xlabel("Epoch")

        if subset:
            loss_ax.legend(loc="best", title="Anisotropy")

    fig.suptitle("Validation trajectories across anisotropy/intrinsic-dimension sweep", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def print_console_summary(runs: list[SweepRun]) -> None:
    intrinsic_values = sorted({run.intrinsic_dim for run in runs})
    anisotropy_values = sorted({run.anisotropy_max_scale for run in runs})
    existing_pairs = {(run.intrinsic_dim, run.anisotropy_max_scale) for run in runs}
    missing_pairs = [
        (intrinsic_dim, anisotropy)
        for intrinsic_dim in intrinsic_values
        for anisotropy in anisotropy_values
        if (intrinsic_dim, anisotropy) not in existing_pairs
    ]

    print(f"Loaded {len(runs)} available sweep runs from {len(intrinsic_values)} intrinsic-dim values and {len(anisotropy_values)} anisotropy values.")
    if missing_pairs:
        print("Missing grid points:")
        for intrinsic_dim, anisotropy in missing_pairs:
            print(f"  - d={intrinsic_dim}, anisotropy_max_scale={anisotropy:g}")

    print("Final metrics by run:")
    for run in sorted(runs, key=lambda item: (item.intrinsic_dim, item.anisotropy_max_scale)):
        status = "partial" if run.is_partial_run else "final"
        print(
            "  - "
            f"d={run.intrinsic_dim}, a={run.anisotropy_max_scale:g}: "
            f"val_loss={run.final_val_loss:.4f}, "
            f"val/feature_mmd_mean={run.final_feature_mmd:.4f} "
            f"[{status}]"
        )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.results_root / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_sweep_runs(args.results_root, args.wandb_root)
    if not runs:
        raise SystemExit("No sweep runs found.")

    max_final_epoch = max(int(run.val_series[-1]["epoch"]) for run in runs)
    for run in runs:
        run.is_partial_run = int(run.val_series[-1]["epoch"]) < max_final_epoch

    rows = build_summary_rows(runs)
    save_summary_csv(rows, out_dir / "anisotropy_intrinsic_sweep_summary.csv")

    plot_heatmap(
        runs=runs,
        metric_name="final_val_loss",
        title="Final validation loss",
        colorbar_label="val_loss",
        out_path=out_dir / "final_val_loss_heatmap.png",
        dpi=args.dpi,
    )
    plot_heatmap(
        runs=runs,
        metric_name="final_feature_mmd",
        title="Final validation Feature-MMD",
        colorbar_label="val/feature_mmd_mean",
        out_path=out_dir / "final_feature_mmd_heatmap.png",
        dpi=args.dpi,
    )
    plot_loss_vs_mmd_scatter(
        runs=runs,
        out_path=out_dir / "val_loss_vs_feature_mmd_scatter.png",
        dpi=args.dpi,
    )
    plot_trajectories(
        runs=runs,
        out_path=out_dir / "validation_trajectories.png",
        dpi=args.dpi,
    )

    print_console_summary(runs)
    print(f"Saved plots and summary to {out_dir}")


if __name__ == "__main__":
    main()
