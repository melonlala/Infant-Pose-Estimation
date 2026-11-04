#!/bin/bash
# Run on the cluster LOGIN node (not via sbatch): commit Slurm logs in logs/ and push them
# so the server can `git pull` them.  Usage: bash slurm/push_results.sh "E1 epoch 0-7"
# Only logs/ is committed here; docs (AGENT.md, MANIFEST.md, SKILL.md) are edited on the server.
set -euo pipefail
cd /home/hou.yuchen/courses/eece7373
git pull --rebase --autostash -q origin main
git add logs/
if git diff --cached --quiet; then echo "no new results"; exit 0; fi
git commit -q -m "results: ${1:-sync Slurm logs} [cluster]"
git push -q origin main
git log --oneline -1
