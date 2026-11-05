#!/bin/bash
# Step 5 (1 GPU, ~20 min): the test-time-augmentation / ensemble systems of REPORT §5 (A0-A3), from the bundled checkpoints.
#   A0 = F4 flip only (must equal the official eval)   A1 = {F4,H1,S1,S2} ensemble, flip only
#   A2 = F4 + flip x scale {0.9,1,1.1} x rotation {-30,0,30}   A3 = ensemble + the same TTA
#   Usage: bash reproduce/05_tta.sh [A0 A1 A2 A3]   (default: A0 A2, the sanity check and the recommended system)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
F4=models/F4/final_state.pth; ENS=models/F4/final_state.pth,models/H1/final_state.pth,models/S1/final_state.pth,models/S2/final_state.pth
need models/F4/final_state.pth models/H1/final_state.pth models/S1/final_state.pth models/S2/final_state.pth
RUNS=("$@"); [ ${#RUNS[@]} -gt 0 ] || RUNS=(A0 A2)
for a in "${RUNS[@]}"; do
  case $a in
    A0) M=$F4;  SC=1.0;          RO=0 ;;
    A1) M=$ENS; SC=1.0;          RO=0 ;;
    A2) M=$F4;  SC=0.9,1.0,1.1;  RO=-30,0,30 ;;
    A3) M=$ENS; SC=0.9,1.0,1.1;  RO=-30,0,30 ;;
    *) echo "unknown $a" >&2; exit 1 ;;
  esac
  for set in "syrip_fixed output" "syrip500_fixed output_test500f"; do
    read -r root dst <<<"$set"
    say "$a on $root (models=$M scales=$SC rots=$RO)"
    "$PY" tools/eval_ensemble_tta.py --cfg "$CFG_HRNET" --models "$M" --scales="$SC" --rots="$RO" \
      --out "$OUT/$dst/ENS_$a/syrip/ens/x" TEST.USE_GT_BBOX True WORKERS "${WORKERS:-4}" DATASET.ROOT "data/$root/" | tail -3
  done
done
say "Re-inferred (this machine)";  "$PY" tools/eval_subsets.py --pred "$OUT/output" --pred500 "$OUT/output_test500f" $(printf 'ENS_%s ' "${RUNS[@]}")
say "Reported";                    "$PY" tools/eval_subsets.py $(printf 'ENS_%s ' "${RUNS[@]}")
