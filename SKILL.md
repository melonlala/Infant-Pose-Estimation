---
name: fidip-explorer
description: Operate the FiDIP (infant pose) project on the Northeastern Explorer cluster - sync code/results with rsync (code to GitHub), submit and monitor Slurm train/eval jobs, log to W&B. Use for any cluster run, job status check, or result sync in this project.
---

# FiDIP on Explorer: agent operating guide

General cluster rules (access, Slurm etiquette, safety) are in
`.claude/skills/use_explorer_cluster/SKILL.md`. This file covers only this project.
The experiment log is in `AGENT.md`, and the inventory of artifacts is in `MANIFEST.md`.

## Hosts and paths

| | Server (where the agent runs) | Cluster |
|--|--|--|
| Host | local workstation, user `yuchen` | `ssh explorer` -> `login.explorer.northeastern.edu`, user `hou.yuchen` (Slurm commands) |
| File transfer | — | **`hou.yuchen@xfer.discovery.neu.edu`** (data-transfer node, same `/home` as the login node) |
| Project root | `/home/yuchen/courses/EECE7373/hands_on_project` | `/home/hou.yuchen/courses/eece7373` |
| Data / models | `data/` (raw zips + extracted, local copy) | `data/`, `models/` (authoritative) |
| Env | none (no GPU work here) | conda prefix env `envs/fidip` (py 3.12, torch 2.3.1+cu121) |

- **Copy files to/from the cluster through the transfer node `xfer.discovery.neu.edu`, never through the login node.** Use
  `rsync -e ssh <src> hou.yuchen@xfer.discovery.neu.edu:/home/hou.yuchen/courses/eece7373/<dst>` (or `scp`), and `slurm/sync.sh`, which already does this.
  This applies to one-off copies too (scripts, checkpoints, figures). The login node drops big rsyncs (`Broken pipe`) and is slow when busy.
  The xfer node shares `/home` with the login node, so paths are identical. Key login works (host key accepted 2026-11-04).
- Always run Slurm/shell commands through `ssh explorer '<cmd>'`. If `module`/`conda` are not found, use
  `ssh explorer 'bash -lc "<cmd>"'`. SSH is key-only (`BatchMode yes`, `ControlPersist 4h`); never type passwords.
- Code: fork of `ostadabbas/Infant-Pose-Estimation` @ `a76b4f4` (remote `upstream`).
  The guide's `gpu_test.py`, `demo.py`, `streamlit_demo.py`, `fidip_simple.yml` do not exist in any public fork.

## Sync model

- **GitHub = code + results**: `origin` = `git@github.com:melonlala/Infant-Pose-Estimation.git` (private, `main`), pushed
  from the **server only**. `upstream` = ostadabbas. Commit code, scripts, docs **and** the text results on the server: since 2026-11-04 `logs/`, `log/` (TB),
  `output*/` (json/log/txt), `figures/`, `temp_uploads/`, `data/syrip_syn_new/` are tracked; commit them after each `sync.sh pull`.
  Never commit SyRIP data (`data/*` except `data/syrip_syn_new/`), `models/`, `envs/`, `*.pth`, `*.npy`, `wandb/`, `.env`, or the licensed SMIL files (see `.gitignore`).
  Checkpoints go only into `checkpoints/fidip_checkpoints.zip` through **Git LFS** (`.gitattributes`); rebuild it and `checkpoints/SHA256SUMS` when a reported checkpoint changes.
- **Server <-> cluster = plain copy** (`rsync` over ssh to `xfer.discovery.neu.edu`), via `slurm/sync.sh` run on the server. The cluster never uses GitHub.
  - `bash slurm/sync.sh push`: server code/scripts/docs -> cluster. **Run before every `sbatch`.** Excludes data, models,
    env, outputs, results, `.env`, the cluster's `.git`, and the `lib/nms` build products.
  - `bash slurm/sync.sh pull`: cluster -> server for `logs/` (Slurm stdout), `log/` (TensorBoard events), and only the
    `*.log`/`*.json`/`*.txt` files from `output*/`. Checkpoints stay on the cluster; record their paths in `MANIFEST.md`.
  - `bash slurm/sync.sh env`: copies only the `WANDB_*` lines of the server `.env` to the cluster `.env` (mode 600).
  - No `--delete` in either direction. Never edit files on the cluster: the next `push` overwrites them.
- The cluster still has the upstream `.git` from the original clone. Ignore it: `git status` there shows our changes as modifications.
- `lib/nms` build products (`*.so`, `cpu_nms.c`, `gpu_nms.cpp`) are untracked; rebuild with `cd lib && make` (inside a job).
- Training curves: W&B project `fidip` (when `.env` is on the cluster), plus TensorBoard events in `log/syrip/...`.

## Secrets

- `.env` sits at the project root on the server (git-ignored). Template: `.env.example`.
- The cluster `.env` gets only the `WANDB_*` lines, via `slurm/sync.sh env`. `GITHUB_TOKEN` never goes to the shared cluster.
- `slurm/train.sh` and `slurm/eval.sh` source `.env`. Never print, `cat`, or commit it. Check for it with `test -f .env`.

## Slurm scripts (`slurm/`, run from the cluster project root)

| Script | What it does | Resources | Submit |
|--|--|--|--|
| `setup_env.sh` | (re)build `envs/fidip`, `make` nms, gdown models, GPU check | courses-gpu, 1 GPU, 1:30 | `sbatch slurm/setup_env.sh` |
| `unpack_data.sh` | unzip `syrip_for_train.zip`, assert image counts | short, CPU, 0:30 | `sbatch slurm/unpack_data.sh` |
| `eval.sh <model> [cfg overrides]` | `test_adaptive_model.py` on Test100, GT bbox | courses-gpu, 1 GPU, 0:45 | `sbatch slurm/eval.sh models/hrnet_fidip.pth` |
| `install_pkgs.sh <pkg...>` | pip-install into `envs/fidip` + import smoke test (login node kills pip) | short, CPU, 0:30 | `sbatch slurm/install_pkgs.sh wandb` |
| `train.sh [cfg overrides]` | `train_adaptive_model_hrnet.py`, init DarkPose W48, W&B if key present | courses-gpu, V100-SXM2, 8:00 | `sbatch slurm/train.sh` |
| `sync.sh push\|pull\|env` | rsync code to / results from the cluster (run on the **server**) | - | `bash slurm/sync.sh push` |

- **Preferred for new experiments:** `ssh explorer 'bash /home/hou.yuchen/courses/eece7373/slurm/submit_run.sh <NAME> <TIME> [cfg overrides]'`
  (train on `data/syrip_fixed` + Test500f evals of best/final; refuses existing `output/<NAME>`). Use `timeout 280`.
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
| — | none from this session | SY1/SY2 done 2026-11-04 (no effect; F4 stays final) | S*/L*/M* jobs in the queue belong to another session; don't touch them |

Keep this table and the `AGENT.md` summary table in sync whenever you submit, finish, or cancel a job.

## Server-side demos and analysis (no cluster needed)

- The server has 2× RTX 6000 Ada. Use `.venv-demo/bin/python` (base python 3.13 + torch, plus yacs/streamlit/dsntnn/pycocotools).
- Weights: `rsync hou.yuchen@xfer.discovery.neu.edu:/home/hou.yuchen/courses/eece7373/<ckpt> models/...` (git-ignored).
- `tools/demo.py --image temp_uploads/own05.jpg --model models/F4/final_state.pth --output figures/demo/x.jpg` (person detector by default; `--person K`, `--box-mode full`, `--bbox`)
- `.venv-demo/bin/streamlit run tools/streamlit_demo.py --server.headless true` -> http://localhost:8501 (deep link `?src=temp_uploads/&img=own05.jpg`); stop it afterwards, it holds GPU memory
- `.venv-demo/bin/python tools/video_infer.py --source <video|folder|0> --out figures/video/ [--no-flip] [--show]` (7.3; no webcam on the server)
- 7.1: `cd syn_generation && ../.venv-smil/bin/python tools/<step>.py ...` (fit on CPU `--jobs 20`, render with `EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1`); SMIL files are licensed: never commit them (`syn_generation/.gitignore`)
- `python3 tools/plot_curves.py log/<RUN>/syrip/adaptive_pose_hrnet/<dir> --name <RUN>` after `sync.sh pull`
- `.venv-demo/bin/python tools/visualize_preds.py --models E0=models/hrnet_fidip.pth ...` (add `--ann`/`--images` for `data/syrip_clean`)
- `.venv-demo/bin/python tools/eval_test96.py D1 F4 ...`: Test100 vs Test96 (minus 4 leaked images) from saved final-epoch prediction JSONs (CPU, no checkpoint needed)

## Data

- Train on **`DATASET.ROOT data/syrip_clean/`** (or the user's `data/syrip_fixed/`), never the raw `data/syrip/`: its synthetic labels are wrong.
- GPU memory: V100 peak 12.6 GiB (no DA) / 17.2 GiB (DA). P100 16 GB OOMs. courses-gpu QoS allows 2 running jobs per user.

## Portable reproduction (graders, any GPU machine)

`REPRODUCE.md` + `reproduce/0*_*.sh` reproduce everything without Slurm (outputs to `repro/`). When a new reported run is added, add its row to
`reproduce/experiments.tsv` (overrides from its train-log Namespace line) and, if its checkpoint is reported, to the bundle (`checkpoints/README.md`).

## Guardrails

- **Confirm with the user before `scancel`**, before resubmitting a job that could overwrite an `output/` dir,
  and before deleting anything (never delete outside `/home/hou.yuchen/courses/eece7373`).
- No heavy compute on the login node: no training, no unzip of the big archives (the login node kills it). Use `sbatch`.
- Report the job ID right after `sbatch`. Log every experiment (including failures) in `AGENT.md`.
- Paper target: Test100 AP 93.6. Released checkpoint in our env: AP 92.1 (E0). **Test100 is the primary metric**: Test500 is saturated (~98.4 for every model).
- Test500 eval: `mkdir -p output_test500 && sbatch slurm/eval.sh <model> DATASET.ROOT data/syrip500/ OUTPUT_DIR output_test500/<run>`.
