#!/bin/bash
#SBATCH --job-name=fidip-setup
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
# Build the FiDIP conda env (python 3.12 / torch 2.3.1+cu121, matching fidip_env.yml),
# compile lib/nms, fetch pretrained models, and run a GPU sanity check.
set -euo pipefail
cd /home/hou.yuchen/courses/eece7373
ENV=/home/hou.yuchen/courses/eece7373/envs/fidip

module load anaconda3/2024.06 cuda/12.1.1
eval "$(conda shell.bash hook)"

if [ ! -x "$ENV/bin/python" ]; then
  conda create -y -p "$ENV" python=3.12
fi
conda activate "$ENV"
pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir numpy==1.26.4 opencv-python-headless==4.10.0.84 pycocotools==2.0.8 \
  json-tricks==3.17.3 yacs==0.1.8 easydict==1.13 cython==3.0.10 scipy==1.14.0 shapely==2.0.4 \
  tensorboardx==2.6.2.2 tensorboard==2.21.0 wandb==0.31.0 matplotlib==3.9.1 dsntnn==0.5.3 pyyaml==6.0.1 setuptools gdown

(cd lib && make)

mkdir -p models/coco
[ -f models/hrnet_fidip.pth ] || gdown --folder 108P-1SnTqaj3xNtjYZ1o7T8z6UvUYuiC -O models/fidip_dl || true
[ -f models/coco/w48_384x288.pth ] || gdown --folder 14kAA1zXuKODYgrRiQmKnVcipbY7RedVV -O models/tga_dl || true
find models -name '*.pth' -exec ls -la {} \;

python - <<'PY'
import torch, cv2, pycocotools, sys
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
sys.path.insert(0, "lib")
from nms.nms import oks_nms; print("nms import ok")
PY
