#!/bin/bash
#SBATCH --job-name=fidip-pip
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
# Usage: sbatch slurm/install_pkgs.sh <pkg> [pkg...]
# pip-install extra packages into envs/fidip (the login node kills long pip runs).
set -euo pipefail
cd /home/hou.yuchen/courses/eece7373
module load anaconda3/2024.06
eval "$(conda shell.bash hook)"
conda activate /home/hou.yuchen/courses/eece7373/envs/fidip
pip install --no-cache-dir "$@"
python - <<'PY'
import torch, tensorboard, wandb
from torch.utils.tensorboard import SummaryWriter
print("torch", torch.__version__, "tensorboard", tensorboard.__version__, "wandb", wandb.__version__)
import sys; sys.path.insert(0, "tools"); sys.argv = ["x"]
import importlib.util
spec = importlib.util.spec_from_file_location("t", "tools/train_adaptive_model_hrnet.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("train script imports OK")
PY
