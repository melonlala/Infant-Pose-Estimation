#!/bin/bash
# Step 6 (1 GPU, 16 GB+; HRNet 20 ep ~20 min, 100 ep ~1.5 h on a V100): retrain any run of REPORT §6 from the COCO init,
# then evaluate its final checkpoint on Test100 (done by the training loop) and Test500f, and score it.
#   Usage: bash reproduce/06_train.sh <ID> [extra cfg overrides]     IDs: reproduce/experiments.tsv   e.g.  bash reproduce/06_train.sh F4
#   On the Explorer cluster the same run was launched with:  bash slurm/submit_run.sh <ID> <TIME> <overrides>   (see SKILL.md)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
ID=${1:?run ID from reproduce/experiments.tsv}; shift
LINE=$(awk -F'\t' -v id="$ID" '$1 == id' reproduce/experiments.tsv)
[ -n "$LINE" ] || { echo "unknown ID $ID; see reproduce/experiments.tsv" >&2; exit 1; }
IFS=$'\t' read -r _ ARCH DROOT OVR DESC <<<"$LINE"; [ "$OVR" = - ] && OVR=""
case $ARCH in
  hrnet)   SCRIPT=tools/train_adaptive_model_hrnet.py;     INIT=$INIT_HRNET;  SUB=adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant ;;
  hrnet_x) SCRIPT=tools/train_adaptive_model_hrnet_x.py;   INIT=$INIT_HRNET;  SUB=adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant ;;
  mobile)  SCRIPT=tools/train_adaptive_model_mobile_wb.py; INIT=$INIT_MOBILE; SUB=adaptive_pose_mobile/mobile_224x224_adam_lr1e-3_infant ;;
esac
need "$INIT" "$DROOT"
D=$OUT/output/$ID
[ -e "$D" ] && { echo "$D exists (AUTO_RESUME would continue it): remove it or use OUT=<other dir>" >&2; exit 1; }
mkdir -p "$OUT/output" "$OUT/log" "$OUT/output_test500f"
say "$ID ($DESC): $SCRIPT  root=$DROOT  overrides=[$OVR $*]"
# shellcheck disable=SC2086
"$PY" "$SCRIPT" --cfg "$(cfg_of $ARCH)" TRAIN.CHECKPOINT "$INIT" WORKERS "${WORKERS:-8}" \
  OUTPUT_DIR "$D" LOG_DIR "$OUT/log/$ID" DATASET.ROOT "$DROOT" $OVR "$@"
say "$ID final_state on Test500f"
evaluate $ARCH "$D/syrip/$SUB/final_state.pth" data/syrip500_fixed/ "$OUT/output_test500f/${ID}_final"
say "$ID: this run vs the reported run"
"$PY" tools/eval_subsets.py --pred "$OUT/output" --pred500 "$OUT/output_test500f" "$ID"
"$PY" tools/eval_subsets.py "$ID"
echo "Single training runs differ by ~0.1-0.2 AP (REPORT §2.2); compare with the replicate pairs in REPORT §6.1."
