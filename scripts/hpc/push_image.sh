#!/usr/bin/env bash
# Copy the Apptainer image from this machine to the HPC. Code goes over git (git pull on the
# HPC); this is only for files git ignores, which so far is just the image.
#
#   bash scripts/hpc/push_image.sh                 # ./camtrap.sif -> create:$WORK_DIR/containers/
#   bash scripts/hpc/push_image.sh other.sif       # a different local image
#   HPC_HOST=otherhost bash scripts/hpc/push_image.sh
#
# Resumable: rerun after a dropped connection and rsync carries on from the partial file.
set -euo pipefail

HPC_HOST="${HPC_HOST:-create}"   # ssh alias in ~/.ssh/config
# Must match WORK_DIR in scripts/hpc/env.sh, since env.sh expects $WORK_DIR/containers/camtrap.sif.
REMOTE_WORK_DIR="${REMOTE_WORK_DIR:-/scratch/users/k21158663/camtrap-loop}"
SRC="${1:-camtrap.sif}"
DEST_DIR="$REMOTE_WORK_DIR/containers"

[ -f "$SRC" ] || { echo "No image at $SRC; build it with container/build.sh" >&2; exit 1; }

# env.sh looks for camtrap.sif, so the remote name is fixed whatever the local one is.
echo "Pushing $SRC ($(du -h "$SRC" | cut -f1)) to $HPC_HOST:$DEST_DIR/camtrap.sif"
rsync -ah --partial --info=progress2 \
  --rsync-path="mkdir -p '$DEST_DIR' && rsync" \
  "$SRC" "$HPC_HOST:$DEST_DIR/camtrap.sif"

# Compare checksums so a truncated copy is caught here rather than at job time.
LOCAL_SUM="$(sha256sum "$SRC" | cut -d' ' -f1)"
REMOTE_SUM="$(ssh "$HPC_HOST" "sha256sum '$DEST_DIR/camtrap.sif'" | cut -d' ' -f1)"
if [ "$LOCAL_SUM" = "$REMOTE_SUM" ]; then
  echo "OK: checksums match ($LOCAL_SUM)"
else
  echo "ERROR: checksum mismatch (local $LOCAL_SUM, remote $REMOTE_SUM); rerun to resume" >&2
  exit 1
fi
