#!/bin/bash
# Step 2: fetch the checkpoint bundle (Git LFS), unpack it to models/ and verify every file.   Usage: bash reproduce/02_weights.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
Z=checkpoints/fidip_checkpoints.zip
if [ "$(stat -c %s "$Z")" -lt 1000 ]; then          # still an LFS pointer (not fetched yet)
  command -v git-lfs >/dev/null || { echo "install Git LFS (https://git-lfs.com), then rerun" >&2; exit 1; }
  git lfs install --local >/dev/null && git lfs pull --include "$Z"
fi
unzip -o -q "$Z" -d .
sha256sum -c checkpoints/SHA256SUMS
ln -sfn w48_384x288.pth "models/coco/w48_384×288.pth"   # name used by the upstream config default
echo "weights OK: $(ls models/*/final_state.pth | wc -l) of our checkpoints + released/COCO-init weights in models/"
