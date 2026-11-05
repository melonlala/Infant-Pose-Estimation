#!/bin/bash
# Step 1: download SyRIP (public, ~1 GB), verify sha256, unpack to data/, and build the corrected eval/train roots.
# Re-runnable: skips anything already present.   Usage: bash reproduce/01_data.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
URL=https://coe.northeastern.edu/Research/AClab/SyRIP
mkdir -p data/raw
for f in syrip_for_train.zip SyRIP.zip; do
  [ -s data/raw/$f ] || curl -fL --retry 3 -o data/raw/$f "$URL/$f"
done
sha256sum -c - <<'SUMS'
1f267878732fb81776c5c7f153bad45d652cadf5bfd83761971390b64d6cee1b  data/raw/syrip_for_train.zip
1d13e816375988be9bbdab8bf5be6562a5b9bfa83684d57ccbfec9dd619e1692  data/raw/SyRIP.zip
SUMS
[ -d data/syrip ] || unzip -q data/raw/syrip_for_train.zip -d data     # -> data/syrip (FiDIP training layout)
[ -d data/SyRIP ] || unzip -q data/raw/SyRIP.zip -d data               # -> data/SyRIP (original release, correct labels)
test "$(ls data/syrip/images/train_infant | wc -l)" -eq 1500
test "$(ls data/syrip/images/train_pre_infant | wc -l)" -eq 4057
test "$(ls data/SyRIP/images/validate_infant | wc -l)" -eq 500

# Corrected roots (symlink trees; REPORT §1.2 explains each defect)
bash tools/make_syrip_fixed.sh                 # data/syrip_fixed: correct synthetic train labels + correct test11.jpg (all runs from D1 on)
python3 tools/make_syrip500_fixed.py           # data/syrip500_fixed: Test500 with area = bbox area
bash slurm/make_real_only_data.sh              # data/syrip_fixed_200R: real-only ablation (G2)
bash slurm/make_plus_syn_data.sh               # data/syrip_fixed_plus522: + our 522 SMIL renders in data/syrip_syn_new (SY1/SY2)
echo "data OK"
