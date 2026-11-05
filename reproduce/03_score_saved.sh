#!/bin/bash
# Step 3 (no GPU, ~1 min): recompute every AP in REPORT.md from the prediction files tracked in output*/.
# Needs only step 1 (for the ground truth) and a python with pycocotools/numpy/matplotlib (any env: PY=...).
#   Usage: bash reproduce/03_score_saved.sh [bootstrap resamples, default 1000]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need data/syrip_fixed/annotations/person_keypoints_validate_infant.json data/syrip500_fixed/annotations/person_keypoints_validate_infant.json
say "REPORT §4-6: all runs, final checkpoints (Test100-family = output/<run>, Test500-family = output_test500f/<run>[_final])"
"$PY" tools/eval_subsets.py D0_E0:E0 D0b_E1b:E1b D1 F1 F2 F3 F4 G1 G2 G3 H1 H2 H3 L1 L2 L3 S1 S2 B3a B3b B4a B4b SY1 SY2 \
                            M0_E0 M1 M2 ENS_A0 ENS_A1 ENS_A2 ENS_A3
say "REPORT §5.1: full metrics + paired bootstrap vs the released model"
"$PY" tools/report_stats_bootstrap.py "${1:-1000}"
say "REPORT Fig. 1"
"$PY" tools/plot_report_ablation.py && echo "wrote figures/report/ablation_overview.png"
