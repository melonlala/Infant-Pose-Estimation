# AGENT.md — Agent Collaboration Log

Running lab notebook for the FiDIP (Fine-tuned Domain-adapted Infant Pose) hands-on project,
written collaboratively with the AI coding agent (Claude Code / Codex). Each entry records what
was asked, what the agent did, what was learned, and why direction changed.

**Project goal:** reproduce the published FiDIP baseline on SyRIP, then fine-tune for higher
accuracy than the paper's numbers (see `STUDENT_GUIDE.md`, Part 7).

**Related files:**
- `SKILL.md` (root) — project operating guide: paths, GitHub sync, Slurm scripts, active runs, guardrails
- `.claude/skills/use_explorer_cluster/SKILL.md` — general Explorer cluster rules
- `MANIFEST.md` — inventory of datasets, configs, checkpoints, and the run behind final numbers

---

## Entry template

```
### YYYY-MM-DD — <short title>
- **Asked:** what I asked the agent to do
- **Did:** what the agent actually did (commands, files changed, job IDs)
- **Hypothesis:** (for experiments) what we expected and why
- **Result:** outcome / metrics (AP, AP50, AP75, AR) — link to log path
- **Decision / next:** what we concluded and why direction changed (or didn't)
```

---

## Experiment summary

| ID | Date | Description | Config | Slurm job(s) | AP | AP50 | AP75 | AR | Status |
|----|------|-------------|--------|--------------|----|------|------|----|--------|
| E0 | 2026-10-30 | Pretrained FiDIP eval (`hrnet_fidip.pth`, GT bbox) | `w48_384x288_adam_lr1e-3_infant.yaml` | 11509270 | 92.1 | 97.1 | 97.1 | 93.6 | done |
| E1 | 2026-10-30 | Reproduce FiDIP training (init: DarkPose `coco/w48_384x288.pth`) | `w48_384x288_adam_lr1e-3_infant.yaml` + `TRAIN.CHECKPOINT` | 11509271 | — | — | — | — | **failed** (no `tensorboard`, 9 s) |
| E1b | 2026-11-03 | E1 resubmitted after env fix, W&B logging | same as E1 | 11601446 (pip) → 11601467 (train, afterok) | 91.1 | 97.0 | 97.0 | 92.7 | done (best @ epoch 6 of 20; final 90.3) |

---

## Log

### 2026-10-30 — Project initialization
- **Asked:** initialize `AGENT.md` and `MANIFEST.md` according to `STUDENT_GUIDE.md`.
- **Did:** read `STUDENT_GUIDE.md` and the Explorer cluster skill; created this log and
  `MANIFEST.md` with templates and placeholders. No code, data, or jobs exist yet.
- **Decision / next:**
  1. Part 0: confirm passwordless `ssh explorer` and set up `.env` (+ `.gitignore`) for secrets.
  2. Part 1: clone the FiDIP repo into `/home/hou.yuchen/courses/eece7373`, build conda env, `make` in `lib/`.
  3. Part 2: obtain SyRIP (password from authors) and pretrained models (DarkPose `w48_384x288.pth`, `hrnet_fidip.pth`, `mobile_fidip.pth`).
  4. Run E0 (pretrained eval) as the reference point before any training.

### 2026-10-30 — Part 0 setup + Part 2 data download
- **Asked:** follow the guide and start; download SyRIP from the website.
- **Did:**
  - Added `.gitignore` (`.env`, `data/`, `models/`, `output/`, `log/`, `logs/`, `*.pth`).
  - Added `Host explorer` to `~/.ssh/config` on the Linux workstation. `ssh explorer` still fails with
    `Permission denied (publickey)`: this machine's `~/.ssh/id_ed25519` is not authorized on Explorer
    (the Mac's key is). **Blocked** until the key is added with `ssh-copy-id`.
  - SyRIP is public (open directory listing, no password needed). Downloaded `syrip_for_train.zip` and
    `SyRIP.zip`, ran `unzip -t` on both (no errors), extracted them to `data/syrip/` and `data/SyRIP/`. Counts and sha256 are in MANIFEST §3.
  - Found upstream code at `ostadabbas/Infant-Pose-Estimation`. It does not include the guide's `gpu_test.py`,
    `demo.py`, `streamlit_demo.py`, or `fidip_simple.yml`, so the course may use a fork.
- **Observation:** `train_infant/` holds 1500 images but only 1200 are annotated (200R+1000S); val in
  `syrip_for_train` is Test100. Paper also reports Test500 (`data/SyRIP/annotations/validate500`).
- **Decision / next:** authorize key → rsync `data/` to the cluster → clone code → build env → E0.

### 2026-10-30 — Explorer setup + first experiments queued
- **Asked:** add the Explorer setting now and start experiments.
- **Did:**
  - `ssh explorer` now works (key authorized). Added `BatchMode yes` + `ControlMaster auto` /
    `ControlPersist 4h` to `~/.ssh/config` (backup at `~/.ssh/config.bak.*`).
  - Cloned upstream `ostadabbas/Infant-Pose-Estimation` @ `a76b4f4` into the cluster root; untracked
    dirs listed in `.git/info/exclude`.
  - Slurm scripts in `slurm/` (rsync'd to cluster): `setup_env.sh`, `unpack_data.sh`, `eval.sh`, `train.sh`.
  - Job 11509124 (`setup_env.sh`, 7 min): conda env at `envs/fidip` (py 3.12.16, torch 2.3.1+cu121),
    `lib/nms` built, GPU check OK on V100-SXM2-32GB. gdown fetched `hrnet_fidip.pth`, `mobile_fidip.pth`,
    `coco/posemobile.pth`. Downloaded DarkPose TGA folder separately; files use `×` in names, so stored as
    `models/coco/w48_384x288.pth` with a `w48_384×288.pth` symlink (the config's default MODEL_FILE).
  - Queued: unpack 11509265 → E0 eval 11509270 → E1 train 11509271 (each `afterok`).
- **Observation:** the provided config has `MODEL.PRETRAINED: ''` and `TRAIN.CHECKPOINT: ''`, so plain
  `train_adaptive_model_hrnet.py` would start HRNet from random init. `train.sh` passes
  `TRAIN.CHECKPOINT models/coco/w48_384x288.pth`.
- **Result (E0, job 11509270, 16 s):** Test100, GT bbox, flip test: AP 92.1 / AP50 97.1 / AP75 97.1 / AR 93.6
  (log `logs/fidip-eval-11509270.out`). Paper reports AP 93.6 / 98.5 / 98.5 / AR 94.6, so the released
  checkpoint is 1.5 AP below the paper in our env (torch 2.3.1, cv2 4.10). The pipeline works end to end.
- **Decision / next:** treat **92.1** as the practical baseline to beat, with 93.6 as the paper target.
  E1 (11509271) was released by `afterok` and is waiting for a V100.

### 2026-11-03 — Progress check: E1 had failed
- **Asked:** follow the student guide and report current progress.
- **Did:** checked `sacct`. E1 (11509271) **FAILED after 9 s**: `ModuleNotFoundError: No module named 'tensorboard'`.
  `train_adaptive_model_hrnet.py` imports `torch.utils.tensorboard`, but `setup_env.sh` installed only `tensorboardx`.
  The earlier entry wrongly recorded E1 as "queued", because nobody polled the job after `afterok` released it.
- **Also checked:** the code is the official upstream `ostadabbas/Infant-Pose-Estimation`. None of its 19 public forks has the guide's
  `gpu_test.py`/`demo.py`/`streamlit_demo.py`/`fidip_simple.yml`. The core train/test scripts and config match the guide.
- **Also checked (labels):** `syrip/annotations/person_keypoints_validate_infant.json` and `SyRIP/annotations/validate100/` are
  identical (100 images, same ids/files, keypoints identical, `num_keypoints` consistent). So the 92.1 vs 93.6 E0 gap
  is **not** a GT-annotation problem. More likely causes are environment/version differences (torch 2.3.1, cv2 4.10, numpy 1.26 vs the paper's stack).

### 2026-11-03 — GitHub sync, W&B, E1 resubmit
- **Asked:** sync the cluster and the server through GitHub, unblock E1, write the root `SKILL.md`, and use W&B.
- **Did:**
  - Made the server project dir a git repo on top of upstream history (`upstream` = ostadabbas, `a76b4f4`), with private
    `origin` = `melonlala/Infant-Pose-Estimation`. Tracked: code, `slurm/`, docs, `logs/` (Slurm stdout = results).
    Ignored: data, models, envs, output, log, `*.pth`, `.env`, and the `lib/nms` build products (untracked so the cluster's py3.12 build is not clobbered).
  - `slurm/push_results.sh`: the cluster commits and pushes only `logs/`. Docs are edited on the server only, which avoids merge conflicts.
  - Installed `tensorboard` + `wandb` into `envs/fidip`. Added both to `setup_env.sh` and removed the bad gdown `--remaining-ok`.
  - `train_adaptive_model_hrnet.py`: optional W&B (`sync_tensorboard=True`, per-epoch `val/AP`, `best_AP` summary). It is enabled only
    when `WANDB_API_KEY` is set. The run id is stored in the output dir so AUTO_RESUME continues the same W&B run.
    `train.sh`/`eval.sh` source `.env`.
  - The first `pip install` ran on the login node and was killed (SIGTERM, exit 143). Added `slurm/install_pkgs.sh` (short partition)
    and installed through job 11601446, which also smoke-tests the training script's imports.
  - Submitted E1b: `sbatch --dependency=afterok:11601446 --export=ALL,WANDB_NAME=E1b-reproduce slurm/train.sh` -> **11601467**.
- **Decision / next:** once E1b trains, compare its best Test100 AP to E0 (92.1) and the paper (93.6). W&B is active only if
  `.env` with `WANDB_API_KEY` is on the cluster before the job starts. Otherwise sync later with `wandb sync --sync-tensorboard log/...`.

### 2026-11-03 — E1b result: reproduction is ~1 AP below the released checkpoint
- **Did:** E1b (11601467) ran on V100 d1004: COMPLETED in 18 min 30 s (20 epochs, ~55 s each). The cluster had no `.env`, so there was **no W&B** logging.
  TensorBoard events are in `log/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant_2026-11-03-19-28/`.
- **Result (Test100, GT bbox, flip):** AP by epoch: 88.4, 89.1, 89.7, 90.0, 90.0, 90.1, **91.1** (ep 6), then flat at 90.0–90.8 to the end (final ep 19: 90.3).
  Best = AP 91.1 / AP50 97.0 / AP75 97.0 / AR 92.7 → `model_best.pth`. That is **−1.0 vs E0 (92.1)** and **−2.5 vs paper (93.6)**.
  Train log: `output/.../w48_384x288_adam_lr1e-3_infant_2026-11-03-19-28_train.log`.
- **Observations:**
  - AP plateaus after ~6 epochs. Training `Loss_p` is ~8.5e-4 and flat, and the domain classifier reaches `Accuracy_d` ≈ 0.98–1.0, so it separates real from
    synthetic features almost perfectly. The adversarial term is probably not yielding domain-invariant features in this config
    (`LAMBDA: 0.0`, `GAMMA2: 0.0`, `PRE_EPOCH: 0`). Check this against the paper's training schedule before tuning.
  - "Best epoch" is selected on Test100, which is also the reported test set, so `model_best` is optimistically biased.
    Report Test500 (`data/SyRIP/annotations/validate500`) as well for a fairer number.
- **Decision / next:** (1) eval `model_best.pth` on Test500 and the released `hrnet_fidip.pth` on Test500. (2) Compare the yaml with the
  paper's FiDIP schedule (pre-train epochs, λ for the domain loss, LR). (3) Every new run needs its own `OUTPUT_DIR`, because this dir now
  holds `checkpoint.pth` at epoch 20 and AUTO_RESUME would end any rerun immediately.

---

## Dead ends & lessons learned

_(Record approaches that didn't work and why, so they aren't repeated.)_

- **rsync of `data/` to Explorer** broke with `Broken pipe` twice (~1 GB). `| tail` hid the failure
  the first time. Fix: `wget` the public zip on the cluster, check sha256, then unzip in a Slurm job,
  because the login node kills `unzip` for exceeding resource limits.
- `gdown` has no `--remaining-ok` flag (folder download failed in job 11509124).
- **`pip install` on the login node gets killed** (exit 143). Run installs with `sbatch slurm/install_pkgs.sh <pkgs>`.
- **E1 (11509271) died at import**: the env had `tensorboardx`, but the script imports `torch.utils.tensorboard`. Lesson: smoke-test the
  *training* script's imports in `setup_env.sh`, not just the eval path. Also poll `sacct` after `afterok` chains, and don't log a job as running unchecked.
