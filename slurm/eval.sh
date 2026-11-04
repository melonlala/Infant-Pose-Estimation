#!/bin/bash
#SBATCH --job-name=fidip-eval
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.out
# Usage: sbatch slurm/eval.sh <model.pth> [extra cfg overrides...]
set -euo pipefail
cd /home/hou.yuchen/courses/eece7373
module load anaconda3/2024.06 cuda/12.1.1
eval "$(conda shell.bash hook)"
conda activate /home/hou.yuchen/courses/eece7373/envs/fidip
# secrets (WANDB_API_KEY, ...) from the git-ignored .env, if present
if [ -f .env ]; then set -a; . ./.env; set +a; fi
MODEL=${1:?model path}; shift
nvidia-smi --query-gpu=name,memory.total --format=csv
python tools/test_adaptive_model.py \
  --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
  TEST.MODEL_FILE "$MODEL" TEST.USE_GT_BBOX True WORKERS 4 "$@"
