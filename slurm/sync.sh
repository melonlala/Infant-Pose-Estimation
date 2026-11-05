#!/bin/bash
# Server <-> cluster sync by plain copy (rsync over ssh to the transfer node xfer.discovery.neu.edu). Run on the SERVER.
#   bash slurm/sync.sh push   # server code/scripts/docs -> cluster (do this before every sbatch)
#   bash slurm/sync.sh pull   # cluster results (Slurm logs, train/eval logs, result json, TB events) -> server
#   bash slurm/sync.sh env    # copy only WANDB_* lines of the server's .env -> cluster .env (chmod 600)
# GitHub holds code only; the cluster never talks to GitHub. No --delete: nothing is ever removed on either side.
set -euo pipefail
SERVER=/home/yuchen/courses/EECE7373/hands_on_project
# File copies go through the data-transfer node (same /home as the login node), not the login node, which drops big rsyncs
# and kills heavy processes. Slurm commands (sbatch/squeue/sacct) still use `ssh explorer`.
XFER=hou.yuchen@xfer.discovery.neu.edu
XSSH="ssh -o BatchMode=yes -o ServerAliveInterval=60"
CLUSTER=$XFER:/home/hou.yuchen/courses/eece7373
cd "$SERVER"
# the Explorer login node sometimes drops rsync connections ("connection unexpectedly closed"); rsync is incremental, so retry
rs() { local i; for i in 1 2 3; do rsync -e "$XSSH" --partial "$@" && return 0; echo "rsync failed (try $i), retrying" >&2; sleep 5; done; return 1; }

case "${1:-}" in
push)
  # Never overwrite cluster-only state: data, models, env, outputs, results, secrets, nms build products, cluster .git
  # .gitignore rules are applied too (`--filter=':- .gitignore'`), so anything git ignores (venvs, figures, models...) never goes up
  rs -az --itemize-changes --filter=':- .gitignore' \
    --exclude='.git/' --exclude='.claude/' --exclude='.codex/' --exclude='.venv*/' --exclude='figures/' \
    --exclude='data/' --exclude='models/' --exclude='envs/' \
    --exclude='output/' --exclude='output_*/' --exclude='log/' --exclude='logs/' --exclude='wandb/' \
    --exclude='.env' --exclude='__pycache__/' --exclude='*.pth' --exclude='*.npy' \
    --exclude='lib/nms/*.so' --exclude='lib/nms/cpu_nms.c' --exclude='lib/nms/gpu_nms.cpp' \
    ./ "$CLUSTER/"
  ;;
pull)
  rs -az --itemize-changes "$CLUSTER/logs/" logs/
  rs -az --itemize-changes "$CLUSTER/log/" log/
  # text results only from output dirs (no checkpoints / feature.npy / debug images)
  for d in $($XSSH $XFER 'cd /home/hou.yuchen/courses/eece7373 && ls -d output output_* 2>/dev/null'); do
    rs -az --itemize-changes --prune-empty-dirs \
      --include='*/' --include='*.log' --include='*.json' --include='*.txt' --exclude='*' \
      "$CLUSTER/$d/" "$d/"
  done
  ;;
env)
  [ -f .env ] || { echo "no .env in $SERVER" >&2; exit 1; }
  grep -E '^WANDB_' .env | $XSSH $XFER 'umask 077; cat > /home/hou.yuchen/courses/eece7373/.env'
  $XSSH $XFER 'test -s /home/hou.yuchen/courses/eece7373/.env && echo "cluster .env updated ($(wc -l < /home/hou.yuchen/courses/eece7373/.env) lines)"'
  ;;
*) echo "usage: $0 push|pull|env" >&2; exit 2;;
esac
