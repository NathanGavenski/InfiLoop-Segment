# Sourced by every job script. Edit the paths for your KCL HPC (CREATE) project space.
# SLURM directives live in the job scripts; this file sets paths and defines `run`, which
# executes a command inside the Apptainer/Singularity image with the GPU and data mounted.

export PROJECT_DIR="${PROJECT_DIR:-$HOME/camtrap-loop}"           # this repository
export SAFARI_ROOT="${SAFARI_ROOT:-/scratch/users/k21158663/sa_fari}" # dataset root
export WORK_DIR="${WORK_DIR:-/scratch/users/k21158663/camtrap-loop}"  # outputs, checkpoints, caches
export SIF="${SIF:-$WORK_DIR/containers/camtrap.sif}"               # built by container/build.sh

# MLflow file store on scratch. A plain directory: safe with many SLURM jobs writing at once.
export MLFLOW_TRACKING_URI="file://$WORK_DIR/mlruns"
export MLFLOW_ALLOW_FILE_STORE=true   # needed only if MLflow is ever upgraded past 3.x
export HF_HOME="$WORK_DIR/hf_cache"   # model weights cache on scratch, not in $HOME
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"  # token from `huggingface-cli login`
export TORCH_HOME="$WORK_DIR/torch_cache"

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
CONTAINER_CMD="$(command -v apptainer || command -v singularity)"

run() {
  # Host environment variables (above) pass through; PYTHONNOUSERSITE in the image keeps
  # ~/.local packages out.
  "$CONTAINER_CMD" exec --nv --pwd "$PROJECT_DIR" \
    --bind "$PROJECT_DIR,$SAFARI_ROOT,$WORK_DIR" "$SIF" "$@"
}

# Download and indexing need no torch or CUDA, so they run outside the image in a small uv
# environment (scripts/hpc/requirements-data.txt). Usage: run_data python scripts/...
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK_DIR/uv_cache}"  # keep the cache off $HOME
export UV_LINK_MODE=copy                                   # cache and venvs may sit on different filesystems
run_data() {
  command -v uv >/dev/null || {
    echo "uv not found; install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    return 1
  }
  uv run --isolated --no-project --python 3.12 \
    --with-requirements "$PROJECT_DIR/scripts/hpc/requirements-data.txt" "$@"
}

# Warn when the image was built from a different lock file than the checked-out code.
if [ -f "$SIF" ]; then
  IMG_LOCK="$("$CONTAINER_CMD" exec "$SIF" cat /opt/camtrap/uv.lock.sha256 2>/dev/null || true)"
  REPO_LOCK="$(sha256sum "$PROJECT_DIR/uv.lock" | cut -d' ' -f1)"
  [ "$IMG_LOCK" = "$REPO_LOCK" ] || echo "WARNING: $SIF was built from a different uv.lock; rebuild it." >&2
else
  echo "WARNING: no image at $SIF; build it with container/build.sh" >&2
fi
cd "$PROJECT_DIR"
