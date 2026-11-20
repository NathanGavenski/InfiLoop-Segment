#!/usr/bin/env bash
#SBATCH --job-name=safari-index
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=nathan.schneider_gavenski@kcl.ac.uk
# Indexing, not downloading: builds frame tables and instance indexes for train (with the
# committed val split) and test from the annotation files, then runs the resolution study.
# Needs only the annotations, not the frames, and runs outside the image (run_data, no torch).
# Needs tens of GB of RAM, so run it on a compute node: sbatch this file, or in
# `srun -p interactive_cpu -c 4 --mem=48G -t 4:00:00 --pty bash` run `bash scripts/hpc/prepare_data.sh`.
#
# Download the annotations first, without SLURM (login node or interactive session):
#   run_data python scripts/download_safari.py --root $SAFARI_ROOT --annotations
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/hpc/env.sh"  # sbatch runs a copy, so not $0

for split in train test; do
  f="$SAFARI_ROOT/annotation/sa_fari_${split}_ext.json"
  [ -f "$f" ] || { echo "Missing $f. Download the annotations first:" >&2
    echo "  run_data python scripts/download_safari.py --root \$SAFARI_ROOT --annotations" >&2; exit 1; }
done
run_data python -m data.sa_fari --ann "$SAFARI_ROOT/annotation/sa_fari_train_ext.json" --split train --out "$WORK_DIR/index"
run_data python -m data.sa_fari --ann "$SAFARI_ROOT/annotation/sa_fari_test_ext.json" --split test --out "$WORK_DIR/index"
run_data python scripts/resolution_study.py "$WORK_DIR/index/instances_train.parquet" \
  "$WORK_DIR/index/instances_test.parquet" --out "$WORK_DIR/outputs/resolution_study.csv"
