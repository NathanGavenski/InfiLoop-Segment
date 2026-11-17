#!/usr/bin/env bash
# Build the Apptainer/Singularity image from pyproject.toml + uv.lock.
#   bash container/build.sh [output.sif]          # tries --fakeroot
#   sudo -E bash container/build.sh [output.sif]  # where fakeroot is not allowed
# If the HPC does not allow building, build on a Linux machine with Apptainer and copy the
# image over (about 7 GB):  rsync -P camtrap.sif <hpc>:<work dir>/containers/
# Rebuild whenever uv.lock changes; jobs warn when the image's lock differs from the repo's.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-camtrap.sif}"
CMD="$(command -v apptainer || command -v singularity)"
# Build temp space: /tmp is often a small tmpfs, so default to a folder next to the output.
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$(dirname "$(realpath -m "$OUT")")/.apptainer-tmp}"
export SINGULARITY_TMPDIR="$APPTAINER_TMPDIR"
mkdir -p "$APPTAINER_TMPDIR"
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT
if command -v uv >/dev/null; then
  uv lock --check   # refuse to build from a lock that is out of date with pyproject.toml
else
  echo "uv not on PATH (e.g. under sudo); skipping the lock freshness check" >&2
fi
FAKEROOT=(); [ "$(id -u)" -ne 0 ] && FAKEROOT=(--fakeroot)
if ! "$CMD" build "${FAKEROOT[@]}" "$OUT" container/camtrap.def; then
  [ "$(id -u)" -ne 0 ] && echo "Build failed. If the error mentions namespaces or privileges," \
    "fakeroot is not allowed here: run 'sudo -E bash container/build.sh $OUT' instead." >&2
  exit 1
fi
echo "lock sha256: $(sha256sum uv.lock | cut -c1-16)  ->  $OUT"
