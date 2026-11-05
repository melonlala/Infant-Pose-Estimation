
## Explorer cluster instructions

### Access
- login node login.explorer.northeastern.edu, user hou.yuchen.
- file transfer: use the data-transfer node xfer.discovery.neu.edu (same /home), e.g. `rsync -e ssh <src> hou.yuchen@xfer.discovery.neu.edu:<path>`. Do not copy files through the login node.
- all operations on the explorer cluster are executed through `ssh explorer '<command>'`, SSH is configured for non-interactive and connection reuse, do not attempt to enter a password.
- if a command cannot be found (such as module, conda), use `ssh explorer 'bash -lc "<command>"'` instead.

### Data and project location
- default working and data directory on the cluster is `/home/hou.yuchen/courses/eece7373`.
- use github to synchronize code between the local machine (A) and the cluster

### Slurm
- explorer cluster uses the Slurm scheduling system. Do not run heavy computation tasks (training, large-scale data processing) directly on the login node; they must be submitted through Slurm.
- before submitting a job, use `ssh explorer 'sinfo'` to check available partitions and computing resources, and choose the appropriate partition.
  - detailed node information: `ssh explorer 'sinfo -N -o "%N %P %T %c %m %G"'` 
- Prefer several short jobs chained with --dependency=afterok:<jobid> over one long job -- shorter requests are scheduled sooner, and a crash only costs you one stage.
- poll squeue/sacct and tail logs to check progress
- submit batch job: `ssh explorer 'cd /home/hou.yuchen/courses/eece7373 && sbatch job.sh'`
- check jobs: `ssh explorer 'squeue -u $USER'`
- check completed jobs: `ssh explorer 'sacct -u $USER --starttime today'`
- cancel job: `ssh explorer 'scancel <jobid>'`
- write `--partition`、`--time`、`--cpus-per-task`、`--mem` explicitly in the sbatch script; if GPU is needed, write `--gres`; output logs to the project's `logs/` directory (`#SBATCH --output=logs/%x-%j.out`).
- report job ID after submitting the job; when waiting for results, use squeue to poll, do not refresh frequently (at least 30 seconds apart).

### Safety
- don't delete any file outside of `/home/hou.yuchen/courses/eece7373` ；ask me before doing so.


