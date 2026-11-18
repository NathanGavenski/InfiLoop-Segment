#!/usr/bin/env bash
#SBATCH --job-name=b2-sam3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
# Zero-shot SAM 3 baseline, sharded over videos.
#   sbatch scripts/hpc/b2_sam3.sh            # prompt "animal"
#   sbatch scripts/hpc/b2_sam3.sh --oracle   # GT species prompts, report separately
# When every shard has finished:
#   python eval.py score --pred $WORK_DIR/outputs/b2_sam3/preds_test.shard*.parquet \
#     --index-dir $WORK_DIR/index --out $WORK_DIR/outputs/b2_sam3/metrics_test.json
#   python scripts/flag_flocks.py --pred $WORK_DIR/outputs/b2_sam3/preds_test.shard*.parquet ...
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/hpc/env.sh"  # sbatch runs a copy, so not $0
TAG="b2_sam3"; [ "${1:-}" = "--oracle" ] && TAG="b2_sam3_oracle"
run python scripts/b2_sam3_zero_shot.py --config configs/b2_sam3.yaml --split test "$@" \
  --shard "$SLURM_ARRAY_TASK_ID" --num-shards "${SLURM_ARRAY_TASK_COUNT:-8}" \
  --out "$WORK_DIR/outputs/$TAG/preds_test.parquet"
