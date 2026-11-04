---
name: fidip-explorer
description: Operate the FiDIP (infant pose) project on the Northeastern Explorer cluster - sync code/results via GitHub, submit and monitor Slurm train/eval jobs, log to W&B. Use for any cluster run, job status check, or result sync in this project.
---

# FiDIP on Explorer: agent operating guide

General cluster rules (access, Slurm etiquette, safety) are in
`.claude/skills/use_explorer_cluster/SKILL.md`. This file covers only this project.
The experiment log is in `AGENT.md`, and the inventory of artifacts is in `MANIFEST.md`.

## Hosts and paths

| | Server (where the agent runs) | Cluster |
|--|--|--|
| Host | local workstation, user `yuchen` | `ssh explorer` -> `login.explorer.northeastern.edu`, user `hou.yuchen` |
| Project root | `/home/yuchen/courses/EECE7373/hands_on_project` | `/home/hou.yuchen/courses/eece7373` |
| Data / models | `data/` (raw zips + extracted, local copy) | `data/`, `models/` (authoritative) |
| Env | none (no GPU work here) | conda prefix env `envs/fidip` (py 3.12, torch 2.3.1+cu121) |

- Always run through `ssh explorer '<cmd>'`. If `module`/`conda` are not found, use
  `ssh explorer 'bash -lc "<cmd>"'`. SSH is key-only (`BatchMode yes`, `ControlPersist 4h`); never type passwords.
- Code: fork of `ostadabbas/Infant-Pose-Estimation` @ `a76b4f4` (remote `upstream`).
  The guide's `gpu_test.py`, `demo.py`, `streamlit_demo.py`, `fidip_simple.yml` do not exist in any public fork.

## Sync model (GitHub)

One repo, `origin` = `git@github.com:melonlala/Infant-Pose-Estimation.git` (private), branch `main`, on both machines.

- **Server -> cluster (code, scripts, docs):** edit and commit on the server, `git push`, then
  `ssh explorer 'cd /home/hou.yuchen/courses/eece7373 && git pull --ff-only'` **before every `sbatch`**.
  Do not edit tracked files on the cluster.
- **Cluster -> server (results):** Slurm stdout goes to `logs/<job>-<id>.out` (tracked, a few KB each). After a job
  finishes: `ssh explorer 'cd /home/hou.yuchen/courses/eece7373 && bash slurm/push_results.sh "E1 done"'`,
  then `git pull` on the server. The script commits **only `logs/`**.
- Never commit `data/`, `models/`, `envs/`, `output/`, `log/`, `*.pth`, `*.npy`, `wandb/`, or `.env`
  (see `.gitignore`). Checkpoints stay on the cluster; record their paths in `MANIFEST.md`.
- `lib/nms` build products (`*.so`, `cpu_nms.c`, `gpu_nms.cpp`) are untracked; rebuild with `cd lib && make` (inside a job).
- Training curves: W&B project `fidip` (live), plus TensorBoard event files in `log/syrip/...` on the cluster.

## Secrets

- `.env` sits at the project root on each machine (git-ignored, `chmod 600`). Template: `.env.example`.
- The cluster's `.env` holds only `WANDB_API_KEY` (+ `WANDB_PROJECT`). Never put `GITHUB_TOKEN` on the shared cluster:
  the cluster pushes with its own SSH deploy key (`~/.ssh/id_ed25519_github`).
- `slurm/train.sh` and `slurm/eval.sh` source `.env`. Never print, `cat`, or commit it. Check for it with `test -f .env`.

## Slurm scripts (`slurm/`, run from the cluster project root)

| Script | What it does | Resources | Submit |
|--|--|--|--|
| `setup_env.sh` | (re)build `envs/fidip`, `make` nms, gdown models, GPU check | courses-gpu, 1 GPU, 1:30 | `sbatch slurm/setup_env.sh` |
| `unpack_data.sh` | unzip `syrip_for_train.zip`, assert image counts | short, CPU, 0:30 | `sbatch slurm/unpack_data.sh` |
| `eval.sh <model> [cfg overrides]` | `test_adaptive_model.py` on Test100, GT bbox | courses-gpu, 1 GPU, 0:45 | `sbatch slurm/eval.sh models/hrnet_fidip.pth` |
| `train.sh [cfg overrides]` | `train_adaptive_model_hrnet.py`, init DarkPose W48, W&B if key present | courses-gpu, V100-SXM2, 8:00 | `sbatch slurm/train.sh` |
| `push_results.sh "<msg>"` | commit + push `logs/` (login node, **not** sbatch) | - | `bash slurm/push_results.sh "E1 done"` |

- Name the W&B run per experiment: `sbatch --export=ALL,WANDB_NAME=E2-lr5e-4 slurm/train.sh TRAIN.LR 0.0005`.
- Different experiments must use different `OUTPUT_DIR`s (cfg override). Otherwise AUTO_RESUME picks up another run's `checkpoint.pth`.
- `AUTO_RESUME` is on: resubmitting the same command continues from `output/.../checkpoint.pth`
  and the same W&B run (id stored in `output/.../wandb_run_id.txt`). Chain stages with
  `sbatch --dependency=afterok:<id> ...`.

## Monitoring

```bash
ssh explorer 'squeue -u $USER -o "%.10i %.20j %.8T %.10M %.20R"'
ssh explorer 'sacct -j <id> --format=JobID,JobName%20,State,Elapsed,ExitCode -X'
ssh explorer 'tail -n 40 /home/hou.yuchen/courses/eece7373/logs/fidip-train-<id>.out'
ssh explorer 'grep -h "| adaptive" /home/hou.yuchen/courses/eece7373/output/syrip/adaptive_pose_hrnet/*/*train.log | tail'
```

Poll at most every 30 s. A training epoch prints an AP table; `model_best.pth` tracks the best Test100 AP.

## Active runs

| ID | Slurm job | Status | Notes |
|--|--|--|--|
| E1 | see `AGENT.md` summary table | - | reproduce FiDIP training |

Keep this table and the `AGENT.md` summary table in sync whenever you submit, finish, or cancel a job.

## Guardrails

- **Confirm with the user before `scancel`**, before resubmitting a job that could overwrite an `output/` dir,
  and before deleting anything (never delete outside `/home/hou.yuchen/courses/eece7373`).
- No heavy compute on the login node: no training, no unzip of the big archives (the login node kills it). Use `sbatch`.
- Report the job ID right after `sbatch`. Log every experiment (including failures) in `AGENT.md`.
- Paper target: Test100 AP 93.6. Released checkpoint in our env: AP 92.1 (E0).
