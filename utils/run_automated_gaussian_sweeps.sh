#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ANIS_VALUES=(1.0 2.0 4.0 8.0 16.0)
GPU_ID="${GPU_ID:-1}"
START_AT="${START_AT:-stage1}"

run_one() {
  local ambient="$1"
  local intrinsic="$2"
  local anis="$3"
  local results_dir="$4"
  local run_name="$5"
  shift 5

  echo "[$(date -Is)] START ${run_name}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run python SiT/train.py \
    ambient_dim="${ambient}" \
    intrinsic_dim="${intrinsic}" \
    anisotropy_max_scale="${anis}" \
    trainer.results_dir="${results_dir}" \
    trainer.run_name="${run_name}" \
    trainer.wandb_run_name="${run_name}" \
    trainer.use_wandb=true \
    trainer.strategy=auto \
    trainer.check_val_every_n_epoch=1 \
    trainer.val_generative_metrics_every_n_epoch=5 \
    trainer.ckpt_every_n_epochs=5 \
    "$@"
  echo "[$(date -Is)] DONE  ${run_name}"
}

if [[ "${START_AT}" == "stage1" ]]; then
  echo "[$(date -Is)] ===== Stage 1: ambient_dim=4 sweep ====="
  for anis in "${ANIS_VALUES[@]}"; do
    run_one 4 4 "${anis}" \
      results/sweep_ambient4_anisotropy \
      "ambient4-anisotropy-a${anis}"
  done
fi

if [[ "${START_AT}" == "stage1" || "${START_AT}" == "stage2a" ]]; then
  echo "[$(date -Is)] ===== Stage 2A: ambient16 anisotropy x model width (4) ====="
  for width in 96 128 192 256; do
    for anis in "${ANIS_VALUES[@]}"; do
      run_one 16 16 "${anis}" \
        results/sweep_ambient16_width \
        "ambient16-anisotropy-a${anis}-width${width}-depth6" \
        +model.hidden_size="${width}" \
        +model.depth=6
    done
  done
fi

if [[ "${START_AT}" == "stage1" || "${START_AT}" == "stage2a" || "${START_AT}" == "stage2b" ]]; then
  echo "[$(date -Is)] ===== Stage 2B: ambient16 anisotropy x model depth (4) ====="
  for depth in 4 6 8 10; do
    for anis in "${ANIS_VALUES[@]}"; do
      run_one 16 16 "${anis}" \
        results/sweep_ambient16_depth \
        "ambient16-anisotropy-a${anis}-width128-depth${depth}" \
        +model.hidden_size=128 \
        +model.depth="${depth}"
    done
  done
fi

echo "[$(date -Is)] All scheduled sweeps completed."
