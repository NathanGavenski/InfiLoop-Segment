#!/usr/bin/env bash
#SBATCH --job-name=safari-index
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Build frame tables and instance indexes for train (with the committed val split) and test,
# then run the resolution study. Download first, interactively on a login node:
#   run python scripts/download_safari.py --root $SAFARI_ROOT --annotations
#   run python scripts/download_safari.py --root $SAFARI_ROOT --frames test train --extract
set -euo pipefail
source "$(dirname "$0")/env.sh"
run python -m data.sa_fari --ann "$SAFARI_ROOT/annotation/sa_fari_train_ext.json" --split train --out "$WORK_DIR/index"
run python -m data.sa_fari --ann "$SAFARI_ROOT/annotation/sa_fari_test_ext.json" --split test --out "$WORK_DIR/index"
run python scripts/resolution_study.py "$WORK_DIR/index/instances_train.parquet" \
  "$WORK_DIR/index/instances_test.parquet" --out "$WORK_DIR/outputs/resolution_study.csv"
