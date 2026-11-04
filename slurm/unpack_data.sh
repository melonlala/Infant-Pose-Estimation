#!/bin/bash
#SBATCH --job-name=fidip-unpack
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
# Extract syrip_for_train.zip (downloaded to data/raw/) into data/syrip and verify counts.
set -euo pipefail
cd /home/hou.yuchen/courses/eece7373/data
rm -rf syrip
unzip -q raw/syrip_for_train.zip -d .
for d in syrip/images/*; do echo "$d $(ls "$d" | wc -l)"; done
test "$(ls syrip/images/train_pre_infant | wc -l)" -eq 4057
test "$(ls syrip/images/train_infant | wc -l)" -eq 1500
test "$(ls syrip/images/validate_infant | wc -l)" -eq 100
echo "data OK"
