#!/usr/bin/env python

from __future__ import annotations

import csv
import itertools
import json
import warnings
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from utils.pointcloud_metrics import (
    exact_discrete_w2_distance,
    mmd_rbf_samples,
    sliced_wasserstein_distance,
)


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "variance": float("nan"), "std": float("nan"), "cv": float("nan")}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1) if arr.size > 1 else 0.0)
    var = float(arr.var(ddof=1) if arr.size > 1 else 0.0)
    cv = float(std / abs(mean)) if abs(mean) > 1e-12 else float("nan")
    return {"mean": mean, "variance": var, "std": std, "cv": cv}


def _make_points(seed: int, n: int, d: int) -> np.ndarray:
    return np.random.default_rng(int(seed)).standard_normal(size=(n, d), dtype=np.float32)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _is_solver_warning(w: WarningMessage) -> bool:
    text = str(w.message)
    return "numItermax reached before optimality" in text


@hydra.main(config_path="../conf/tools", config_name="metric_stability_seedstudy", version_base=None)
def main(cfg: DictConfig) -> None:
    sample_sizes = [int(x) for x in cfg.sample_sizes]
    seeds = [int(x) for x in cfg.seeds]
    if len(seeds) != 3:
        raise ValueError("This study expects exactly 3 seeds.")

    dim = int(cfg.dimension)
    num_projections = int(cfg.num_projections)
    output_root = Path(hydra.utils.to_absolute_path(str(cfg.output_root)))
    w2_stop_on_warning = bool(cfg.w2_stop_on_warning)

    seed_pairs = list(itertools.combinations(seeds, 2))
    raw_rows: list[dict[str, Any]] = []
    per_size: dict[str, Any] = {}
    w2_active = True

    for n in sample_sizes:
        datasets = {seed: _make_points(seed=seed, n=n, d=dim) for seed in seeds}

        swd_vals: list[float] = []
        mmd_vals: list[float] = []
        w2_vals: list[float] = []
        w2_warn_count = 0

        for pair_idx, (seed_i, seed_j) in enumerate(seed_pairs):
            points_i = datasets[seed_i]
            points_j = datasets[seed_j]

            swd = sliced_wasserstein_distance(
                points_i,
                points_j,
                num_projections=num_projections,
                seed=pair_idx,
            )
            mmd = mmd_rbf_samples(
                points_i,
                points_j,
                max_samples=None,
                seed=pair_idx,
                gamma=None,
                gamma_mode=str(cfg.mmd_gamma_mode),
                gamma_scale=float(cfg.mmd_gamma_scale),
                unbiased=bool(cfg.mmd_unbiased),
                standardize_features=cfg.mmd_standardize_features,
                eig_eps=float(cfg.mmd_eig_eps),
            )

            w2 = float("nan")
            w2_warning = False
            if w2_active:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    w2 = exact_discrete_w2_distance(
                        points_i,
                        points_j,
                        max_samples=None,
                        seed=pair_idx,
                    )
                w2_warning = any(_is_solver_warning(w) for w in caught)
                if w2_warning:
                    w2_warn_count += 1

            swd_vals.append(float(swd))
            mmd_vals.append(float(mmd))
            if w2_active:
                w2_vals.append(float(w2))

            raw_rows.append(
                {
                    "sample_size": n,
                    "seed_i": seed_i,
                    "seed_j": seed_j,
                    "swd": float(swd),
                    "mmd": float(mmd),
                    "w2": float(w2),
                    "w2_solver_warning": int(w2_warning),
                }
            )

        if w2_active and w2_stop_on_warning and w2_warn_count > 0:
            w2_active = False

        per_size[str(n)] = {
            "pair_count": len(seed_pairs),
            "swd": _stats(swd_vals),
            "mmd": _stats(mmd_vals),
            "w2": _stats(w2_vals) if len(w2_vals) > 0 else None,
            "w2_solver_warnings": int(w2_warn_count),
        }

    w2_sizes_no_warning = [
        int(n)
        for n, payload in per_size.items()
        if payload.get("w2") is not None and int(payload.get("w2_solver_warnings", 0)) == 0
    ]
    w2_solver_limit = max(w2_sizes_no_warning) if w2_sizes_no_warning else None

    summary = {
        "goal": "Stability comparison of SWD, MMD, and W2 across point-cloud sizes and seed changes.",
        "setting": {
            "distribution": "standard_normal_vs_standard_normal",
            "sample_sizes": sample_sizes,
            "seeds": seeds,
            "seed_pairs": [list(p) for p in seed_pairs],
            "dimension": dim,
            "num_projections_swd": num_projections,
            "mmd": {
                "gamma_mode": str(cfg.mmd_gamma_mode),
                "gamma_scale": float(cfg.mmd_gamma_scale),
                "unbiased": bool(cfg.mmd_unbiased),
                "standardize_features": str(cfg.mmd_standardize_features),
                "eig_eps": float(cfg.mmd_eig_eps),
            },
            "w2_stop_on_warning": w2_stop_on_warning,
        },
        "results_by_size": per_size,
        "w2_solver_limit_no_warning": w2_solver_limit,
    }

    raw_csv = output_root / "seed_stability_raw.csv"
    summary_csv = output_root / "seed_stability_report.csv"
    summary_json = output_root / "seed_stability_summary.json"
    _ensure_parent(raw_csv)
    _ensure_parent(summary_csv)
    _ensure_parent(summary_json)

    with raw_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_size",
                "seed_i",
                "seed_j",
                "swd",
                "mmd",
                "w2",
                "w2_solver_warning",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    report_rows: list[dict[str, Any]] = [
        {
            "section": "goal",
            "item": "research_question",
            "value": summary["goal"],
            "notes": "Compare metric stability under seed changes for identical Gaussian laws.",
        },
        {
            "section": "setting",
            "item": "sample_sizes",
            "value": ",".join(str(x) for x in sample_sizes),
            "notes": "SWD and MMD run on all sizes; W2 run until solver warning threshold.",
        },
        {
            "section": "setting",
            "item": "seeds",
            "value": ",".join(str(x) for x in seeds),
            "notes": "Three independent seeds form three pairwise comparisons per size.",
        },
        {
            "section": "method",
            "item": "statistics",
            "value": "mean,variance,std,cv",
            "notes": "CV is std/|mean|; can be unstable when mean is near zero (notably MMD).",
        },
    ]

    for n in sample_sizes:
        payload = per_size[str(n)]
        for metric in ("swd", "mmd", "w2"):
            metric_stats = payload.get(metric)
            if metric_stats is None:
                continue
            report_rows.append(
                {
                    "section": "results",
                    "item": f"N={n}:{metric}",
                    "value": (
                        f"mean={metric_stats['mean']:.8g}; var={metric_stats['variance']:.8g}; "
                        f"std={metric_stats['std']:.8g}; cv={metric_stats['cv']:.8g}"
                    ),
                    "notes": f"w2_solver_warnings={payload['w2_solver_warnings']}" if metric == "w2" else "",
                }
            )

    report_rows.append(
        {
            "section": "results",
            "item": "w2_solver_limit_no_warning",
            "value": str(w2_solver_limit),
            "notes": "Largest N with zero W2 solver warnings across all seed pairs.",
        }
    )

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "item", "value", "notes"])
        writer.writeheader()
        writer.writerows(report_rows)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] raw rows: {raw_csv}")
    print(f"[done] report csv: {summary_csv}")
    print(f"[done] summary json: {summary_json}")
    print(f"[done] W2 solver limit (no warning): {w2_solver_limit}")


if __name__ == "__main__":
    main()
