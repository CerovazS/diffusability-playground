#!/usr/bin/env python

from __future__ import annotations

import csv
import json
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


def _float_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "variance": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
        }
    return {
        "mean": float(arr.mean()),
        "variance": float(arr.var(ddof=1) if arr.size > 1 else 0.0),
        "std": float(arr.std(ddof=1) if arr.size > 1 else 0.0),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def _seed_for(*, base: int, sample_size: int, repeat_idx: int) -> int:
    return int(base + sample_size * 1009 + repeat_idx * 9176)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@hydra.main(config_path="../conf/tools", config_name="metric_stability_sweep", version_base=None)
def main(cfg: DictConfig) -> None:
    sample_sizes = [int(x) for x in cfg.sample_sizes]
    repeats = int(cfg.repeats)
    dim = int(cfg.dimension)
    num_projections = int(cfg.num_projections)
    seed_a = int(cfg.seed_a)
    seed_b = int(cfg.seed_b)
    output_root = Path(hydra.utils.to_absolute_path(str(cfg.output_root)))

    raw_rows: list[dict[str, Any]] = []
    summary_by_n: dict[str, Any] = {}

    for sample_size in sample_sizes:
        swd_values: list[float] = []
        w2_values: list[float] = []
        mmd_values: list[float] = []
        swd_sym_err: list[float] = []
        w2_sym_err: list[float] = []
        mmd_sym_err: list[float] = []

        for repeat_idx in range(repeats):
            rng_a = np.random.default_rng(_seed_for(base=seed_a, sample_size=sample_size, repeat_idx=repeat_idx))
            rng_b = np.random.default_rng(_seed_for(base=seed_b, sample_size=sample_size, repeat_idx=repeat_idx))

            points_a = rng_a.standard_normal(size=(sample_size, dim), dtype=np.float32)
            points_b = rng_b.standard_normal(size=(sample_size, dim), dtype=np.float32)

            swd_ab = sliced_wasserstein_distance(
                points_a,
                points_b,
                num_projections=num_projections,
                seed=repeat_idx,
            )
            swd_ba = sliced_wasserstein_distance(
                points_b,
                points_a,
                num_projections=num_projections,
                seed=repeat_idx,
            )

            w2_ab = exact_discrete_w2_distance(points_a, points_b, max_samples=None, seed=repeat_idx)
            w2_ba = exact_discrete_w2_distance(points_b, points_a, max_samples=None, seed=repeat_idx)

            mmd_ab = mmd_rbf_samples(
                points_a,
                points_b,
                max_samples=None,
                seed=repeat_idx,
                gamma=None,
                gamma_mode=str(cfg.mmd_gamma_mode),
                gamma_scale=float(cfg.mmd_gamma_scale),
                unbiased=bool(cfg.mmd_unbiased),
                standardize_features=cfg.mmd_standardize_features,
                eig_eps=float(cfg.mmd_eig_eps),
            )
            mmd_ba = mmd_rbf_samples(
                points_b,
                points_a,
                max_samples=None,
                seed=repeat_idx,
                gamma=None,
                gamma_mode=str(cfg.mmd_gamma_mode),
                gamma_scale=float(cfg.mmd_gamma_scale),
                unbiased=bool(cfg.mmd_unbiased),
                standardize_features=cfg.mmd_standardize_features,
                eig_eps=float(cfg.mmd_eig_eps),
            )

            swd_values.append(float(swd_ab))
            w2_values.append(float(w2_ab))
            mmd_values.append(float(mmd_ab))
            swd_sym_err.append(float(abs(swd_ab - swd_ba)))
            w2_sym_err.append(float(abs(w2_ab - w2_ba)))
            mmd_sym_err.append(float(abs(mmd_ab - mmd_ba)))

            raw_rows.append(
                {
                    "sample_size": sample_size,
                    "repeat": repeat_idx,
                    "swd": float(swd_ab),
                    "w2": float(w2_ab),
                    "mmd": float(mmd_ab),
                    "swd_symmetry_abs_error": float(abs(swd_ab - swd_ba)),
                    "w2_symmetry_abs_error": float(abs(w2_ab - w2_ba)),
                    "mmd_symmetry_abs_error": float(abs(mmd_ab - mmd_ba)),
                }
            )

        summary_by_n[str(sample_size)] = {
            "swd": _float_stats(swd_values),
            "w2": _float_stats(w2_values),
            "mmd": _float_stats(mmd_values),
            "symmetry_abs_error": {
                "swd": _float_stats(swd_sym_err),
                "w2": _float_stats(w2_sym_err),
                "mmd": _float_stats(mmd_sym_err),
            },
        }

    summary = {
        "experiment": {
            "distribution": "standard_normal_vs_standard_normal",
            "sample_sizes": sample_sizes,
            "repeats": repeats,
            "dimension": dim,
            "num_projections": num_projections,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "notes": "Both sets are i.i.d. Gaussian draws with different RNG seeds.",
        },
        "metrics": summary_by_n,
    }

    json_path = output_root / "metrics_stability_summary.json"
    csv_path = output_root / "metrics_stability_raw.csv"
    _ensure_parent(json_path)
    _ensure_parent(csv_path)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fieldnames = [
        "sample_size",
        "repeat",
        "swd",
        "w2",
        "mmd",
        "swd_symmetry_abs_error",
        "w2_symmetry_abs_error",
        "mmd_symmetry_abs_error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    print(f"[done] Wrote summary JSON to: {json_path}")
    print(f"[done] Wrote raw CSV to: {csv_path}")


if __name__ == "__main__":
    main()
