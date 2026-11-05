#!/bin/bash
# Build data/syrip_fixed/: the FiDIP training layout (data/syrip/) with two data bugs fixed.
#   1. train labels: data/syrip's person_keypoints_train_infant.json carries MINI-RGBD poses for the
#      1000 synthetic images (they do not match the images). Use SyRIP/annotations/200R_1000S instead
#      (same 1200 images, same 200 real labels, correct synthetic labels).
#   2. data/syrip/images/validate_infant/test11.jpg is a copy of test24.jpg; use SyRIP's test11.jpg.
# Everything else is symlinked to data/syrip. Nothing under data/syrip or data/SyRIP is modified.
# Run from the project root (server or cluster): bash tools/make_syrip_fixed.sh
set -euo pipefail
cd "${1:-.}"
test -d data/syrip -a -d data/SyRIP || { echo "run from the project root" >&2; exit 1; }
F=data/syrip_fixed
mkdir -p $F/annotations $F/images/validate_infant
ln -sfn ../../SyRIP/annotations/200R_1000S/person_keypoints_train_infant.json $F/annotations/person_keypoints_train_infant.json
ln -sfn ../../syrip/annotations/person_keypoints_train_pre_infant.json      $F/annotations/person_keypoints_train_pre_infant.json
ln -sfn ../../syrip/annotations/person_keypoints_validate_infant.json       $F/annotations/person_keypoints_validate_infant.json
ln -sfn ../../syrip/images/train_infant     $F/images/train_infant
ln -sfn ../../syrip/images/train_pre_infant $F/images/train_pre_infant
for f in data/syrip/images/validate_infant/*.jpg; do
  b=$(basename "$f")
  ln -sfn ../../../syrip/images/validate_infant/$b $F/images/validate_infant/$b
done
ln -sfn ../../../SyRIP/images/validate_infant/test11.jpg $F/images/validate_infant/test11.jpg
# checks
n=$(ls $F/images/validate_infant | wc -l); [ "$n" = 100 ] || { echo "expected 100 val images, got $n" >&2; exit 1; }
cmp -s $F/images/validate_infant/test11.jpg data/SyRIP/images/validate_infant/test11.jpg
cmp -s $F/annotations/person_keypoints_train_infant.json data/SyRIP/annotations/200R_1000S/person_keypoints_train_infant.json
echo "built $F: $(ls $F/images/train_infant/ | wc -l) train files, $n val images, train json = SyRIP/200R_1000S"
