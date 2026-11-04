#!/bin/bash
# Submit one experiment = train job + two chained Test500(fixed-area) evals (best + final checkpoint). Run ON THE CLUSTER.
#   bash slurm/submit_run.sh <NAME> <TIME> [cfg overrides...]
#   e.g. bash slurm/submit_run.sh F2 00:50:00 TRAIN.LAMBDA 0.0005 TRAIN.PRE_EPOCH 1
# Data root defaults to data/syrip_fixed/ (override with ROOT=...). Refuses to reuse an existing output/<NAME>
# (AUTO_RESUME would silently continue another run). Prints all job ids on one line; log that line in AGENT.md.
set -euo pipefail
cd /home/hou.yuchen/courses/eece7373
NAME=${1:?name}; TIME=${2:?time limit}; shift 2
ROOT=${ROOT:-data/syrip_fixed/}
[ -e "output/$NAME" ] && { echo "output/$NAME exists - pick a new name" >&2; exit 1; }
mkdir -p output_test500f
T=$(sbatch --parsable --partition=courses-gpu --time="$TIME" --job-name="fidip-$NAME" --export=ALL,WANDB_NAME="$NAME" \
      slurm/train.sh OUTPUT_DIR "output/$NAME" LOG_DIR "log/$NAME" DATASET.ROOT "$ROOT" "$@")
M=output/$NAME/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant
B=$(sbatch --parsable --dependency=afterok:$T --job-name="fidip-$NAME-t5b" \
      slurm/eval.sh "$M/model_best.pth" DATASET.ROOT data/syrip500_fixed/ OUTPUT_DIR "output_test500f/$NAME")
F=$(sbatch --parsable --dependency=afterok:$T --job-name="fidip-$NAME-t5f" \
      slurm/eval.sh "$M/final_state.pth" DATASET.ROOT data/syrip500_fixed/ OUTPUT_DIR "output_test500f/${NAME}_final")
echo "$NAME train=$T test500f_best=$B test500f_final=$F root=$ROOT overrides=[$*]"
