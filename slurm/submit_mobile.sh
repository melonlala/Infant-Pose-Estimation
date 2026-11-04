#!/bin/bash
# MobileNet version of slurm/submit_run.sh: train (slurm/train_mobile.sh) + chained Test500f evals of best and final. Run ON THE CLUSTER.
#   bash slurm/submit_mobile.sh <NAME> <TIME> [cfg overrides...]
# Separate file (not an ARCH switch in submit_run.sh) so `slurm/sync.sh push` from the main checkout cannot revert it.
set -euo pipefail
cd /home/hou.yuchen/courses/eece7373
NAME=${1:?name}; TIME=${2:?time limit}; shift 2
ROOT=${ROOT:-data/syrip_fixed/}
[ -e "output/$NAME" ] && { echo "output/$NAME exists - pick a new name" >&2; exit 1; }
mkdir -p output_test500f
T=$(sbatch --parsable --partition=courses-gpu --time="$TIME" --job-name="fidip-$NAME" --export=ALL,WANDB_NAME="$NAME" \
      slurm/train_mobile.sh OUTPUT_DIR "output/$NAME" LOG_DIR "log/$NAME" DATASET.ROOT "$ROOT" "$@")
M=output/$NAME/syrip/adaptive_pose_mobile/mobile_224x224_adam_lr1e-3_infant
B=$(sbatch --parsable --dependency=afterok:$T --job-name="fidip-$NAME-t5b" \
      slurm/eval_mobile.sh "$M/model_best.pth" DATASET.ROOT data/syrip500_fixed/ OUTPUT_DIR "output_test500f/$NAME")
F=$(sbatch --parsable --dependency=afterok:$T --job-name="fidip-$NAME-t5f" \
      slurm/eval_mobile.sh "$M/final_state.pth" DATASET.ROOT data/syrip500_fixed/ OUTPUT_DIR "output_test500f/${NAME}_final")
echo "$NAME arch=mobile train=$T test500f_best=$B test500f_final=$F root=$ROOT overrides=[$*]"
