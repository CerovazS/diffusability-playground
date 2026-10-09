#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=GPU-8eed0e25-daa2-baa4-5598-85f907abfaee
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export UV_PROJECT_ENVIRONMENT=/home/cerovaz/repos/diffusability-playground/.venv
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
# One suite at a time on the authorized GPU, including all evaluation children.
exec 9>"/tmp/posterior-3090-${UID}.lock"
flock -n 9 || { echo 'Another posterior suite owns the 3090 lock.' >&2; exit 1; }
exec uv run --frozen --no-sync python -m diffusability.experiment "$@"
