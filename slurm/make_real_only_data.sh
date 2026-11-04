#!/bin/bash
# data/syrip_fixed_200R/: like data/syrip_fixed but train_infant = the 200 real images only (SyRIP annotations/200R).
# For the real-only vs real+synthetic ablation (paper Table III; tells whether more synthetic data (7.1) is worth generating).
set -euo pipefail
D=data/syrip_fixed_200R; S=data/syrip_fixed
test -f data/SyRIP/annotations/200R/person_keypoints_train_infant.json && test -d $S
mkdir -p $D/annotations $D/images
ln -sfn ../../SyRIP/annotations/200R/person_keypoints_train_infant.json $D/annotations/person_keypoints_train_infant.json
for a in train_pre_infant validate_infant; do ln -sfn "$(readlink -f $S/annotations/person_keypoints_$a.json)" $D/annotations/person_keypoints_$a.json; done
for i in train_infant train_pre_infant validate_infant; do ln -sfn "$(readlink -f $S/images/$i)" $D/images/$i; done
python3 -c "import json;[print(a, len(json.load(open(f'$D/annotations/person_keypoints_{a}.json'))['images'])) for a in ['train_infant','train_pre_infant','validate_infant']]"
