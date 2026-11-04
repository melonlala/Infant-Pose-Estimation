#!/bin/bash
# Build data/syrip_clean/ (symlinks only, ~0 bytes) from the two official archives. Run from the project root on either machine.
# Fixes two defects found in syrip_for_train.zip (see AGENT.md 2026-11-03 "data audit"):
#   1. train_infant synthetic labels (1000 images) do not match the images -> use SyRIP.zip annotations/200R_1000S
#      (real 200 identical, synthetic = 1000S labels that match the images).
#   2. validate_infant/test11.jpg is a different 2048x2048 picture -> use SyRIP.zip images/validate_infant (identical otherwise).
# Usage: bash slurm/make_clean_data.sh      then train/eval with  DATASET.ROOT data/syrip_clean/
set -euo pipefail
D=data/syrip_clean
test -f data/SyRIP/annotations/200R_1000S/person_keypoints_train_infant.json
test -f data/SyRIP/images/validate_infant/test11.jpg
mkdir -p $D/annotations $D/images
ln -sfn ../../SyRIP/annotations/200R_1000S/person_keypoints_train_infant.json $D/annotations/person_keypoints_train_infant.json
ln -sfn ../../syrip/annotations/person_keypoints_train_pre_infant.json        $D/annotations/person_keypoints_train_pre_infant.json
ln -sfn ../../syrip/annotations/person_keypoints_validate_infant.json         $D/annotations/person_keypoints_validate_infant.json
ln -sfn ../../syrip/images/train_infant      $D/images/train_infant
ln -sfn ../../syrip/images/train_pre_infant  $D/images/train_pre_infant
ln -sfn ../../SyRIP/images/validate_infant   $D/images/validate_infant
python3 - <<'PY'
import json, os
d = 'data/syrip_clean'
for a in ['person_keypoints_train_infant', 'person_keypoints_train_pre_infant', 'person_keypoints_validate_infant']:
    j = json.load(open(f'{d}/annotations/{a}.json')); print(f'{a}: {len(j["images"])} images, {len(j["annotations"])} anns')
for s in ['train_infant', 'train_pre_infant', 'validate_infant']:
    print(f'images/{s}: {len(os.listdir(f"{d}/images/{s}"))} files')
PY
