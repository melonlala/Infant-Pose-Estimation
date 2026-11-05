#!/bin/bash
#SBATCH --job-name=fidip-train-mobile
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# MobileNet variant of slurm/train.sh: tools/train_adaptive_model_mobile_wb.py, 224x224 config, init from COCO posemobile.pth.
# Usage: sbatch slurm/train_mobile.sh [extra cfg overrides...]   (normally via slurm/submit_mobile.sh)
set -euo pipefail
cd /home/hou.yuchen/courses/eece7373
module load anaconda3/2024.06 cuda/12.1.1
eval "$(conda shell.bash hook)"
conda activate /home/hou.yuchen/courses/eece7373/envs/fidip
if [ -f .env ]; then set -a; . ./.env; set +a; fi
nvidia-smi --query-gpu=name,memory.total --format=csv
python tools/train_adaptive_model_mobile_wb.py \
  --cfg experiments/coco/mobilenet/mobile_224x224_adam_lr1e-3_infant.yaml \
  TRAIN.CHECKPOINT models/coco/posemobile.pth WORKERS 8 "$@"
