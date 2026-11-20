#!/usr/bin/env bash
#SBATCH --job-name=camtrap-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# Not b200: the image's torch (cu126) has no sm_100 kernels.
#SBATCH --constraint=a100|h200|l40s
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nathan.schneider_gavenski@kcl.ac.uk
# Predict on the test split at batch size 1 (per-frame latency), then score with all strata.
# Usage: sbatch scripts/hpc/eval.sh configs/b1_dinov3.yaml [split=test]
# Pin the GPU type (e.g. --constraint) so latency is comparable across methods.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/hpc/env.sh"  # sbatch runs a copy, so not $0
CONFIG="$1"; SPLIT="${2:-test}"
NAME="$(basename "$CONFIG" .yaml)"
RUN_DIR="$WORK_DIR/outputs/$NAME"
run nvidia-smi --query-gpu=name --format=csv
run python eval.py predict --config "$CONFIG" --ckpt "$RUN_DIR/best.ckpt" --split "$SPLIT" \
  --out "$RUN_DIR/preds_$SPLIT.parquet"
RUN_ID=$(run python -c "import torch; print(torch.load('$RUN_DIR/best.ckpt', map_location='cpu', weights_only=False)['mlflow_run_id'])")
FLOCKS=(); [ -f "$WORK_DIR/flocks_test.json" ] && FLOCKS=(--flocks "$WORK_DIR/flocks_test.json")
run python eval.py score --pred "$RUN_DIR/preds_$SPLIT.parquet" --index-dir "$WORK_DIR/index" \
  --split "$SPLIT" --workers 16 "${FLOCKS[@]}" --out "$RUN_DIR/metrics_$SPLIT.json" --mlflow-run-id "$RUN_ID"
