#!/usr/bin/env bash
# Clone the upstream InfiLoop and TRM codebases into third_party/ as read-only references.
# Only the recursive block is later copied (Phase 3) into models/infiloop_block.py, with
# the source commit recorded in that file's header. third_party/ is git-ignored.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party

clone_at() {  # url dir commit
  if [ ! -d "third_party/$2/.git" ]; then git clone --quiet "$1" "third_party/$2"; fi
  git -C "third_party/$2" fetch --quiet origin
  git -C "third_party/$2" checkout --quiet "$3"
  echo "$2 @ $(git -C "third_party/$2" rev-parse --short HEAD)"
}

# Commits were upstream HEAD on 2026-11-17. Bump deliberately, never implicitly.
clone_at https://github.com/pixeli99/InfiLoop.git InfiLoop 63716a9c46723d3592c8ad4fc5819855f22aaa47
clone_at https://github.com/SamsungSAILMontreal/TinyRecursiveModels.git TinyRecursiveModels c01103738605ba39d1430519b1ee0c62f4c707f8
