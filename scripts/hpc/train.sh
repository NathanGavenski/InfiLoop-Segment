#!/usr/bin/env bash
#SBATCH --job-name=camtrap-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
# Train one baseline. Usage: sbatch scripts/hpc/train.sh configs/b1_dinov3.yaml [a.b=c ...]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/hpc/env.sh"  # sbatch runs a copy, so not $0
CONFIG="$1"; shift
run nvidia-smi --query-gpu=name,memory.total --format=csv
run python train.py --config "$CONFIG" --set "$@"
