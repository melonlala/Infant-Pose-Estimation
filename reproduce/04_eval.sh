#!/bin/bash
# Step 4 (1 GPU, ~10 min): re-run inference with the released and our bundled checkpoints on Test100 and Test500f,
# then score them exactly like step 3. Outputs go to $OUT (default repro/); the tracked output*/ are not touched.
#   Usage: bash reproduce/04_eval.sh [RUN ...]      RUN in: E0 D1 F4 H1 S1 S2 M0 M2   (default: all)
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
declare -A CKPT=([E0]=models/hrnet_fidip.pth [D1]=models/D1/final_state.pth [F4]=models/F4/final_state.pth [H1]=models/H1/final_state.pth
                 [S1]=models/S1/final_state.pth [S2]=models/S2/final_state.pth [M0]=models/mobile_fidip.pth [M2]=models/M2/final_state.pth)
RUNS=("$@"); [ ${#RUNS[@]} -gt 0 ] || RUNS=(E0 D1 F4 H1 S1 S2 M0 M2)
for r in "${RUNS[@]}"; do
  [ -n "${CKPT[$r]:-}" ] || { echo "unknown run $r" >&2; exit 1; }
  arch=hrnet; [[ $r == M* ]] && arch=mobile
  say "$r: ${CKPT[$r]} on Test100";   evaluate $arch "${CKPT[$r]}" data/syrip_fixed/    "$OUT/output/$r"
  say "$r: ${CKPT[$r]} on Test500f";  evaluate $arch "${CKPT[$r]}" data/syrip500_fixed/ "$OUT/output_test500f/$r"
done
say "Re-inferred (this machine)"
"$PY" tools/eval_subsets.py --pred "$OUT/output" --pred500 "$OUT/output_test500f" "${RUNS[@]}"
say "Reported (tracked predictions; E0 = D0_E0, M0 = M0_E0)"
"$PY" tools/eval_subsets.py $(for r in "${RUNS[@]}"; do case $r in E0) echo D0_E0:E0;; M0) echo M0_E0;; *) echo $r;; esac; done)
echo "Expect agreement within ~0.1 AP (GPU / cuDNN differences)."
