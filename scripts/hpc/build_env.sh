#!/usr/bin/env bash
# One-off environment build. Run on a login node (or an interactive GPU job to check CUDA).
#   bash scripts/hpc/build_env.sh [--sam3]
set -euo pipefail
export VENV="${VENV:-/scratch/prj/CHANGE_ME/camtrap-loop/venv}"
module load python/3.12 2>/dev/null || module load python 2>/dev/null || true
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.7.1 torchvision==0.22.1
pip install -r requirements.txt
if [ "${1:-}" = "--sam3" ]; then pip install -r requirements-sam3.txt; fi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
