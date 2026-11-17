# Sourced by every job script. Edit the paths for your KCL HPC (CREATE) project space.
# SLURM directives live in the job scripts; this file only sets up the environment.

export PROJECT_DIR="${PROJECT_DIR:-$HOME/camtrap-loop}"           # this repository
export SAFARI_ROOT="${SAFARI_ROOT:-/scratch/prj/CHANGE_ME/sa_fari}" # dataset root (shared storage)
export WORK_DIR="${WORK_DIR:-/scratch/prj/CHANGE_ME/camtrap-loop}"  # outputs, checkpoints, caches
export VENV="${VENV:-$WORK_DIR/venv}"

# MLflow file store on shared storage, one store for the whole team.
export MLFLOW_TRACKING_URI="file://$WORK_DIR/mlruns"
export HF_HOME="$WORK_DIR/hf_cache"
export TORCH_HOME="$WORK_DIR/torch_cache"

module load python/3.12 2>/dev/null || module load python 2>/dev/null || true
module load cuda/12.6 2>/dev/null || true
# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$PROJECT_DIR"
