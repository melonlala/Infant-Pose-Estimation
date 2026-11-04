#!/bin/bash
#SBATCH --job-name=fidip-ens
#SBATCH --partition=gpu,courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x-%j.out
# Ensemble/TTA eval (tools/eval_ensemble_tta.py). Usage:
#   sbatch --job-name=fidip-ENS_A1-t100 slurm/eval_ens.sh <models,comma,sep> <scales> <rots> <out_dir> [cfg overrides, e.g. DATASET.ROOT data/syrip_fixed/]
set -euo pipefail
cd /home/hou.yuchen/courses/eece7373
module load anaconda3/2024.06 cuda/12.1.1
eval "$(conda shell.bash hook)"
conda activate /home/hou.yuchen/courses/eece7373/envs/fidip
nvidia-smi --query-gpu=name,memory.total --format=csv
M=${1:?models}; SC=${2:?scales}; RO=${3:?rots}; OUT=${4:?out}; shift 4
python tools/eval_ensemble_tta.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
  --models "$M" --scales "$SC" --rots "$RO" --out "$OUT" TEST.USE_GT_BBOX True WORKERS 4 "$@"
