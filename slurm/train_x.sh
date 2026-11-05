#!/bin/bash
#SBATCH --job-name=fidip-train-x
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# Variant of slurm/train.sh running tools/train_adaptive_model_hrnet_x.py (OHKM option). Usage: sbatch slurm/train_x.sh [extra cfg overrides...]
# AUTO_RESUME is on in the config, so resubmitting continues from output/.../checkpoint.pth.
set -euo pipefail
cd /home/hou.yuchen/courses/eece7373
module load anaconda3/2024.06 cuda/12.1.1
eval "$(conda shell.bash hook)"
conda activate /home/hou.yuchen/courses/eece7373/envs/fidip
# secrets (WANDB_API_KEY, ...) from the git-ignored .env, if present
if [ -f .env ]; then set -a; . ./.env; set +a; fi
nvidia-smi --query-gpu=name,memory.total --format=csv
python tools/train_adaptive_model_hrnet_x.py \
  --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
  TRAIN.CHECKPOINT models/coco/w48_384x288.pth WORKERS 8 "$@"
