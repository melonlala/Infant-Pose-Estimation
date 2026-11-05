#!/bin/bash
# Step 0: build the conda env (Python 3.12, PyTorch 2.3.1 + CUDA 12.1) at envs/fidip and compile lib/nms.
# Needs conda and an NVIDIA driver for CUDA 12.1. ~10 min, ~6 GB.   Usage: bash reproduce/00_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
ENV=${ENV:-$ROOT/envs/fidip}
command -v conda >/dev/null || { echo "conda not found" >&2; exit 1; }
[ -x "$ENV/bin/python" ] || conda create -y -q -p "$ENV" python=3.12
export PATH="$ENV/bin:$PATH"
pip install -q --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -q --no-cache-dir -r reproduce/requirements.txt
# lib/nms: cpu_nms (required) + gpu_nms (optional, needs nvcc). Compile for the local GPU's architecture.
export NMS_CUDA_ARCH=${NMS_CUDA_ARCH:-$(python -c "import torch; c = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (7, 0); print(f'sm_{c[0]}{c[1]}')")}
(cd lib && make >/dev/null 2>&1) || true
python -c "import sys; sys.path.insert(0, 'lib/nms'); import cpu_nms" || { echo "lib/nms build failed: run 'cd lib && make' to see the error" >&2; exit 1; }
python - <<'PY'
import sys, torch
sys.path.insert(0, 'lib')
from nms.nms import oks_nms, gpu_nms  # noqa: F401  (compiled extension)
import json_tricks, yacs, pycocotools, cv2  # noqa: F401
print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda,
      '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (eval/train need a GPU)')
print('gpu_nms:', 'built' if gpu_nms else 'not built (nvcc missing/incompatible; not needed)')
print('env OK:', sys.prefix)
PY
