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
| E0-500 | 2026-11-03 | E0 model on **Test500** | baseline + `DATASET.ROOT data/syrip500/` | 11602004 | 98.4 | 98.9 | 98.7 | 99.3 | done |
| E1b-500 | 2026-11-03 | E1b `model_best` on **Test500** | same | 11602005 | 98.5 | 98.9 | 98.8 | 99.3 | done |
| E1b-F-500 | 2026-11-03 | E1b `final_state` (ep 20) on **Test500** | same | 11602006 | 98.3 | 98.9 | 98.8 | 99.1 | done |
| E1b | 2026-11-03 | E1 resubmitted after env fix, W&B logging | same as E1 | 11601446 (pip) → 11601467 (train, afterok) | 91.1 | 97.0 | 97.0 | 92.7 | done (best @ epoch 6 of 20; final 90.3) |
| E1c | 2026-11-03 | E1b replicate (run-to-run noise) | same as E1b, `OUTPUT_DIR output/E1c` | 11602172 | 90.8 | — | — | — | done (best @ epoch 18, 0-indexed; final 89.6) |
| E2 | 2026-11-03 | E1b + domain adaptation on (`LAMBDA 0.0005`, `PRE_EPOCH 1`) | `output/E2` | 11602173 | 91.1 | — | — | — | done (best @ epoch 9; final 90.1); **trained on broken synthetic labels** |
| E3 | 2026-11-03 | E2 + LR 1e-3 + 100 ep | `output/E3` | 11602174 | — | — | — | — | **failed** at W&B init (`wandb.util.generate_id`), superseded |
| E4 | 2026-11-03 | E1b + 100 ep | `output/E4` | 11602175 | — | — | — | — | **failed** (same W&B bug), superseded |
| E5–E8 | 2026-11-03 | clean-data runs (`data/syrip_clean`): E5 plain, E6 +DA, E7 paper-like, E8 100 ep | `output/E5..E8` | 11602485, 11602498/11602574 (dup), 11602575, 11602580 | — | — | — | — | **cancelled** by request: the user is retraining on their own fix (D-track, `data/syrip_fixed`) |
| P100 probe | 2026-11-03 | 1 epoch on P100-16GB, E2 settings | `output/probe_p100` | 11602176 | — | — | — | — | **CUDA OOM** → P100 not usable at batch 20 (V100 peak 12.6 GiB plain / 17.2 GiB with DA) |
| D0 | 2026-11-03 | *(user)* E0 `hrnet_fidip.pth` on **fixed Test100** (`data/syrip_fixed`, correct test11) | `DATASET.ROOT data/syrip_fixed/` | 11602447 | **93.6** | 98.5 | 98.5 | 94.6 | done; = paper's 93.6 |
| D0b | 2026-11-03 | *(user)* E1b `model_best` on fixed Test100 | same | 11602449 | 92.6 | 98.5 | 98.5 | 93.7 | done |
| D1 | 2026-11-03 | *(user)* retrain E1b config on **fixed labels** (`200R_1000S`), DA off, 20 ep | `train.sh DATASET.ROOT data/syrip_fixed/ OUTPUT_DIR output/D1` | 11602450 | **93.8** | 98.5 | 98.5 | 94.9 | done (best @ epoch 16, 0-idx; final 93.6); W&B run 0997f45c |
| F1 | 2026-11-03 | D1 replicate (run-to-run noise on fixed data) | `submit_run.sh F1` (D1 config, `data/syrip_fixed`) | 11603301 → t5b 11603312, t5f 11603317 | 93.7 | 98.5 | 98.5 | 94.8 | done (best @ 15, final 93.5); **Test500f 91.2 best / 91.2 final** |
| F2 | 2026-11-03 | D1 + DA (`LAMBDA 0.0005`, `PRE_EPOCH 1`) | `submit_run.sh F2 ...` | 11603328 → 11603329, 11603330 | 93.8 | 98.5 | 98.5 | 94.7 | done (best @ 17, final 93.5); **Test500f 90.9 / 91.0** |
| F3 | 2026-11-03 | D1 + LR 1e-3 (paper LR) | `submit_run.sh F3 ... TRAIN.LR 0.001` | 11603335 → 11603336, 11603337 | 93.3 | 98.4 | 98.4 | 94.7 | done (best @ 13, final 92.9); **Test500f 89.9 / 90.6** |
| F4 | 2026-11-03 | D1 + 100 epochs (LR drop ×0.1 at ep 40) | `submit_run.sh F4 ... TRAIN.END_EPOCH 100` | 11603343 → 11603344, 11603345 | 94.1 | — | — | 95.1 | done (best @ 20, final 94.0); **Test500f 91.0 best / 91.7 final** |
| G1 | 2026-11-03 | D1 + `TRAIN.BALANCED_SHUFFLE True` (sampler fix) | `submit_run.sh G1 ...` | 11606776 → 11606777, 11606778 | 94.2 | — | — | — | done (best @ 12, final 94.0); **Test500f 91.4 / 91.5** → adopted |
| G2 | 2026-11-03 | **real-only** 200R, 200 ep (= 2000 iters like D1), no LR drop | `ROOT=data/syrip_fixed_200R/ submit_run.sh G2 ... TRAIN.END_EPOCH 200 TRAIN.LR_STEP [400,500]` | 11606811 → 11606812, 11606813 | 93.9 | — | — | — | done (best = final @ 199); **Test500f 91.3 / 91.3** |
| G3 | 2026-11-03 | D1 + LR 5e-5 | `submit_run.sh G3 ... TRAIN.LR 0.00005` | 11606822 → 11606823, 11606824 | 93.4 | — | — | — | done (best @ 17, final 93.3); **Test500f 90.9 / 91.0** |
| H1 | 2026-11-04 | F4 replicate (100 ep, no shuffle) | `submit_run.sh H1 ... TRAIN.END_EPOCH 100` | 11607648 → 11607652, 11607654 | 94.3 | — | — | — | done (best @ 30, final 93.9); **Test500f 91.4 / 91.6** → with F4: confirmed |
| H2 | 2026-11-04 | B (= D1 + shuffle) + stronger aug (`ROT_FACTOR 60`, `SCALE_FACTOR 0.45`) | `submit_run.sh H2 ...` | 11607656 → 11607658, 11607659 | 94.0 | — | — | — | done (best @ 14, final 93.9); **Test500f 91.4 / 91.6** (single run, not confirmable) |
| H3 | 2026-11-04 | G1 replicate (D1 + shuffle); *extra confirmation run, not in the rule* | `submit_run.sh H3 ... TRAIN.BALANCED_SHUFFLE True` | 11607662 → 11607664, 11607666 | 94.2 | — | — | — | done (best @ 12, final 93.9); **Test500f 91.3 / 91.3** → shuffle pair mean 91.4 < 91.5, not confirmed |
| SY1 | 2026-11-04 | F4 config (100 ep) + **522 new SMIL renders** (7.1) → train = 200R + 1000S + 522S | `ROOT=data/syrip_fixed_plus522/ submit_run.sh SY1 03:30:00 TRAIN.END_EPOCH 100` | 11618211 → 11618213, 11618215 | 93.8 | — | — | — | done (best @ Test100 93.8, final 93.2, Test96 92.6); **Test500f 91.4 / 91.7** (best / final) |
| SY2 | 2026-11-04 | SY1 replicate | same, `SY2` | 11618217 → 11618218, 11618220 | 93.5 | — | — | — | done (best 93.5, final 93.1, Test96 92.5); **Test500f 90.9 / 91.5** → with SY1: no effect (rule) |
| L1 | 2026-11-04 | D1 + DA λ 1e-4 (`PRE_EPOCH 1`) | `submit_run.sh L1 ... TRAIN.LAMBDA 0.0001 TRAIN.PRE_EPOCH 1` | 11617341 → 11617348, 11617351 | 93.7 | — | — | — | done (final); **Test99 92.8 / Test499f 91.1** |
| L2 | 2026-11-04 | D1 + DA λ 2e-3 | same, `TRAIN.LAMBDA 0.002` | 11617353 → 11617376, 11617492 | 92.4 | — | — | — | done; Test99 91.4 / Test499f 89.6 |
| L3 | 2026-11-04 | D1 + DA λ 5e-3 | same, `TRAIN.LAMBDA 0.005` | 11617418 → 11617494, 11617495 | 89.7 | — | — | — | done; Test99 88.8 / Test499f 87.2 → **DA never helps** |
| S1 | 2026-11-04 | F4 (100 ep) + `BALANCED_SHUFFLE True` | `submit_run.sh S1 02:45:00 TRAIN.END_EPOCH 100 TRAIN.BALANCED_SHUFFLE True` | 11617384 → 11617393, 11617423 | 94.3 | — | — | — | done; Test99 93.3 / Test499f 91.4 |
| S2 | 2026-11-04 | S1 replicate | same | 11617427 → 11617435, 11617437 | 94.2 | — | — | — | done; Test99 93.2 / Test499f 91.6 → pair mean 91.5 < 91.7, **not confirmed** |
| M0 | 2026-11-04 | released `mobile_fidip.pth`, fixed data | `CFG=<mobile yaml> eval.sh` | 11617333, 11617336 | 79.2 | 99.0 | 89.4 | 84.1 | done; Test99 78.2 / Test499f 69.7 |
| M1 | 2026-11-04 | MobileNet on fixed labels, own yaml, 20 ep | `submit_mobile.sh M1 01:00:00` | 11618020 → 11618092, 11618105 | 74.7 | — | — | — | done; Test99 74.0 / Test499f 69.0 (first attempt 11617422 ran as HRNet by mistake, cancelled) |
| M2 | 2026-11-04 | M1 + 100 ep | `submit_mobile.sh M2 03:00:00 TRAIN.END_EPOCH 100` | 11618140 → 11618148, 11618151 | 79.4 | — | — | — | done; Test99 78.6 / Test499f 73.1 (+3.4 vs released MobileNet) |
| T5f-E0 | 2026-11-03 | Released `hrnet_fidip.pth` on **Test500f** (Test500, `area` = bbox w*h) | `DATASET.ROOT data/syrip500_fixed/`, `OUTPUT_DIR output_test500f/E0` | 11602599 | 90.9 | 98.7 | 96.7 | 92.3 | done |
| T5f-E1b | 2026-11-03 | E1b `model_best` on Test500f | same, `output_test500f/E1b` | 11602600 | 89.7 | 98.8 | 95.7 | 91.4 | done |
| T5f-D1 | 2026-11-03 | D1 `model_best` on Test500f | same, `output_test500f/D1` | 11602601 (afterok 11602450) | 91.4 | 98.8 | 96.7 | 92.6 | done |

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
  *(Superseded the same day: see "Sync redesign" below. The cluster no longer uses git.)*
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

### 2026-11-03 — Sync redesign: GitHub for code only, rsync for the cluster
- **Asked:** after creating the private repo, push only the server's code to it, and sync the cluster with the server by simple copy.
- **Did:** removed `slurm/push_results.sh` and the plan for a cluster deploy key. Added `slurm/sync.sh push|pull|env` (rsync, no `--delete`).
  `push` copies code/scripts/docs to the cluster and excludes data/models/env/outputs/results/`.env`/nms build products. `pull` brings
  back `logs/`, `log/` (TB), and `*.log|*.json|*.txt` from `output*/`. `env` copies only `WANDB_*` lines. `logs/` and `output_*/` are now git-ignored. *(Reversed 2026-11-04: results are tracked for submission, see the last log entry.)*
  Pushed `main` to `origin` (private `melonlala/Infant-Pose-Estimation`).
- **Why:** simpler. The cluster needs no GitHub credentials and git history stays code-only. The trade-off is that the cluster copy has no
  version info, so the commit used by a run must be noted in AGENT.md when submitting.

### 2026-11-03 — Test500 evals: first attempt failed
- **Did:** built `data/syrip500/` (symlinks to `data/SyRIP/annotations/validate500/...json` and `data/SyRIP/images/validate_infant`;
  ids verified: 500 images `test{id}.jpg`, all present). Submitted evals of E0 / E1b best / E1b final with `DATASET.ROOT data/syrip500/`.
- **Result:** 11601956-58 all FAILED at start: `FileNotFoundError: 'output_test500/E0'`. `create_logger` calls `mkdir()` without
  `parents=True` for `OUTPUT_DIR`, so the parent must already exist. Created `output_test500/` and resubmitted as 11602004 (E0), 11602005 (E1b best), 11602006 (E1b final).

### 2026-11-03 — Test500 results: saturated, so not useful for comparing models
- **Result (Test500, GT bbox, flip):** E0 98.4 / E1b best 98.5 / E1b final 98.3 AP (AR 99.1–99.3). On Test100 the same models score
  92.1 / 91.1 / 90.3. All three are within 0.2 AP on Test500.
- **Checked with md5 of image bytes (server `data/`):** Test100 ⊂ Test500 (99 unique of 100). The other ~400 images score ≈ 100 AP.
  Exact overlap with **annotated** training images: only 4 (test20/42/46/50, all in Test100), so exact leakage does not explain the saturation.
  203 Test500 images are byte-identical to *unannotated* `train_infant/train00xxx.jpg` files, which are unused in training. Test500 also has internal
  duplicates (e.g. test345 = test349). Near-duplicate video frames between Test500 and the real training images are possible but unchecked.
- **Decision:** use **Test100** as the primary metric (hard and discriminative). Report Test500 only as a secondary sanity number.
  The 1-AP gap of E1b vs E0 on Test100 does not show up on Test500, so the gap is in the hard images.

  *(Update, from the "原始数据标注检查" session: the saturation is an annotation artifact. Every one of the 500 `validate500` anns
  has the same `area` = 2443298, which inflates OKS. With area = bbox w·h, the old Test500 predictions rescore to E0 90.9 / E1b 89.7 / E1b-final 90.0.
  See their T5f entries.)*

### 2026-11-03 — Part 7 start: config vs paper; first ablations (E1c, E2; E3/E4 failed)
- **Asked:** start Part 7 by comparing the training config with the paper.
- **Paper (arXiv 2010.06100, HTML v5; some table values came from a summary, so verify against the PDF):** init session pre-trains the domain
  classifier for 1 epoch (batch 128, feature extractor frozen). Then 100 epochs (batch 64) alternate Stage I (update the DC with the pose net locked) and
  Stage II (update the pose net with the DC locked). Loss L = L_P − λ·L_D, λ = 0.0005. Adam, LR 1e-3, schedule not stated. First 3 ResNet stages frozen.
  Table IV (Test100): DarkPose 88.5 → DarkPose+Finetune 92.7 → DarkPose+FiDIP 93.6 AP.
- **Our yaml vs paper:** `LAMBDA 0.0` (DA **off**: `loss_p = MSE − LAMBDA·BCE`, `lib/core/function.py:140`), `PRE_EPOCH 0` (no DC init),
  LR 1e-4, 20 epochs, batch 20 (16 for pre). `GAMMA1/2` are unused in code. Real data is up-weighted via `BalancedBatchSampler` (50/50 real/syn batches).
  Code freezes the first 109 parameter tensors (`idx <= 108`). => **E1b is effectively "DarkPose + Finetune"**, not FiDIP.
- **Did:** single-factor runs vs E1b (commit b3d0be0): E1c = replicate, E2 = DA on, E3 = DA + LR 1e-3 + 100 ep, E4 = 100 ep.
  Added a per-epoch `val AP` / `peak GPU mem` log line + W&B fields.
- **Result:** E1c best 90.8 (vs E1b 91.1) → **run-to-run noise ≈ ±0.3 AP on best-epoch Test100**; differences below ~0.5 AP are not meaningful from
  one run. E2 best 91.1 (no gain), but its DC accuracy stays at 0.84–0.96 instead of 1.0, so the adversarial term does act on the features.
  E3/E4 crashed at W&B init (`wandb.util.generate_id` is gone in wandb 0.31); fixed in 59e4756 (`uuid`) and checked with `tools/wandb_smoke.py`.
  E1c/E2 ran before `.env` reached the cluster, so they have no W&B (TB only).

### 2026-11-03 — Part 4.5: over/underfitting analysis (E1b; E1c/E2 look the same)
- **Did:** `tools/plot_curves.py <tb_run_dir> --name <run>` → `figures/<run>_curves.png` (pose loss train vs val, PCK accuracy, val AP/AR, DC).
- **Observations (E1b, 20 ep):** train MSE falls 34% (1.02e-3 → 0.67e-3) and train PCK rises 0.51 → 0.65. Val MSE drops only over epochs 1–4
  (4.35e-4 → 3.9e-4) and then stays flat; val PCK stays flat at ~0.905 and val AP plateaus at 0.90–0.91 after epoch ~5. Val loss < train loss throughout,
  because training uses heavy augmentation (rot ±45°, scale ±35%, half-body 0.3) and half of each batch is synthetic. The val/train loss ratio climbs from
  0.43 to 0.59.
- **Reading:** not underfitting (val AP 0.90+ from epoch 1, starting from a strong COCO init). There is **mild overfitting onset**: train keeps improving
  and val does not, but val loss and AP do not degrade, so it is a plateau, not classic overfitting. Best-epoch selection on Test100 picks noise
  (best epochs 6, 18, 9 across E1b/E1c/E2). Report best and final. The plateau is partly explained by the label defect below: half of every batch
  carried wrong targets.
- **Superseded (2026-11-04, see "Part 4.5 corrected" below):** this reading was made on broken data and is wrong in its cause. The
  plateau after epoch 6 and the slowly falling, high train loss come from the **wrong synthetic labels** (half of every batch had unlearnable targets)
  plus the swapped `test11.jpg` capping val AP. With correct labels (D1) train loss is 2.5× lower and val AP keeps rising to epoch ~17.

### 2026-11-03 — Part 4 visualisation + data audit: synthetic training labels in syrip_for_train are wrong
- **Did:** `tools/fidip_infer.py` (shared inference = eval path), `tools/visualize_preds.py` (GT vs pred grids for worst/best OKS, per-keypoint
  similarity bars, COCO AP recomputed with pycocotools). Ran on the server GPU (RTX 6000 Ada) in `.venv-demo` with weights copied to `models/`.
  The recomputed Test100 AP is E0 91.8 / E1b 91.0 (official 92.1 / 91.1), so the pipeline agrees within 0.3.
- **Finding 1 (Test100):** `test11.jpg` in `syrip_for_train.zip` is a different 2048×2048 picture (baby lying sideways). Its GT (1536×2048, upright baby)
  matches `SyRIP.zip`'s `test11.jpg`. Every model scores OKS ≈ 0.1 on it. All other 99 Test100 files are byte-identical to SyRIP's.
- **Finding 2 (training labels, major):** for the **1000 synthetic** images in `syrip/annotations/person_keypoints_train_infant.json`, the keypoints do
  not match the images. 46 have `width/height` 640×480 but the files are 1920×1080. The other 954 have consistent sizes but wrong poses
  (figures/viz/synthetic_label_compare.jpg). `SyRIP.zip` `annotations/1000S` (= synthetic part of `200R_1000S`) has the right labels.
  Independent check: E0 predictions (whole-image box) vs labels on all 1000 synthetic images give OKS mean **0.21** vs for_train labels and **0.62**
  vs 1000S labels; 1000S is better on 955/1000. The 200 real labels are identical in both files. Every run so far (E1b, E1c, E2) trained with
  wrong targets on half of each batch.
- **Also:** 4 Test100 images are byte-identical to annotated real training images (test20=train00073, test42=train00315, test46=train00272,
  test50=train00340). This leakage is not fixed by `syrip_clean`.
- **Fix built:** `slurm/make_clean_data.sh` → `data/syrip_clean/` (symlinks): train ann = `SyRIP/annotations/200R_1000S`, val images = SyRIP's (fixes test11).
  Built on both machines. The user independently built `data/syrip_fixed/` and is retraining (D-track: D0, D0b, D1), so the clean-data runs E5–E8 were cancelled.
- **Per-keypoint (Test100, OKS-similarity):** ears are the weakest joints (left_ear 0.89 E0 / 0.91 E1b), then wrists and ankles; E1b is worse than E0
  mainly on nose, elbows and right_knee/ankle. Worst images (both models): test11 (bad GT), test33, test41, test98 (small/occluded infants).

### 2026-11-03 — Part 6 demos
- `tools/demo.py` (6.1, guide's CLI: `--cfg --image --model --output --conf-threshold --output-size`, plus `--bbox`, `--no-flip`).
  It prints a per-keypoint table and draws left = blue, right = orange.
- `tools/streamlit_demo.py` (6.2): upload, `temp_uploads/`, or validation picker; model picker; threshold slider; flip toggle; optional box;
  confidence table; download button. Uploads are saved to `temp_uploads/` (6.3). Headless-tested with `streamlit.testing` (no exceptions, 17-row table).
- Run on the server: `.venv-demo/bin/streamlit run tools/streamlit_demo.py` (venv = base python + yacs, streamlit, dsntnn, pycocotools).
- 6.3 still needs the user's own infant images in `temp_uploads/`.

### 2026-11-03 — Raw-data annotation audit (`tools/check_annotations.py`)
- **Asked:** check the raw data for wrong or duplicate annotations.
- **Did:** wrote `tools/check_annotations.py` (structure, keypoints, bbox/area, is_synthetic, identical labels, md5 + dHash
  duplicates, and label agreement between copies), ran it on all 8 JSONs in `data/`, drew overlays of the flagged cases, and
  re-scored the saved Test500 predictions with pycocotools.
- **Result (confirmed by overlays):**
  1. **Training labels for all 1000 synthetic images are wrong** in `data/syrip/annotations/person_keypoints_train_infant.json`
     (the file FiDIP trains on). Its synthetic entries carry MINI-RGBD source poses (`image_source: MINI_dataset`,
     `original_file_name: 01/syn_00000.png`, 13 kps, always 640x480), not labels for the SyRIP renders in `images/train_infant/`.
     bbox IoU against `SyRIP/annotations/200R_1000S` is a median of 0.31, and 0% of images agree within 5% of bbox size. 46 entries also have the wrong size
     (json 640x480, file 1920x1080). The images are byte-identical in both copies. `SyRIP/annotations/200R_1000S` matches them, and its 200 real labels are identical.
  2. **Test500 `area` is the constant 2443298** for all 500 anns (`validate500` json). COCOeval uses gt area as the OKS scale, so
     Test500 AP is inflated. With area = bbox w*h: E0 98.4 → **90.9**, E1b 98.5 → **89.7**, E1b final 98.3 → **90.0**. This explains
     the "saturation" logged earlier. The train JSONs also use a constant area of 14233, which is harmless because the loader only checks `area > 0`.
  3. **Test100 `test11.jpg` is the wrong image** in `data/syrip/images/validate_infant/`: it is byte-identical to `test24.jpg`. The labels
     and the `SyRIP/` copy are correct. Scoring Test100 GT with predictions made on the correct images gives E0 92.1 → **92.7** and E1b 91.1 → **92.7**.
     So E1b's "−1 AP vs E0" was an artifact, and the gap to the paper (93.6) is 0.9.
  4. **Duplicates / leakage:** 4 images are in both train and Test100 (train00073=test20, train00272=test46, train00315=test42,
     train00340=test50), each labelled twice with disagreements of 32–87 px. Three Test500 images appear twice (test56=test217, test7=test222,
     test345=test349), also labelled inconsistently. Near-duplicates (dHash≤12), real images only: test162~test435, test328~test365,
     test207~test47, test139~test166, train00044~train00464. There are no near-duplicates between train and test beyond the 4 exact ones.
  5. **Copy-pasted label:** test149 has exactly test118's keypoints, on a different baby (Test500 only).
  6. Test100 and Test500 disagree on test83 and test84 (two annotation passes; both look plausible).
  7. Weak heuristic flags only. These need manual review and are mostly real twisted or side-view poses: 17 L/R shoulder–hip disagreements in Test500,
     7 in 200R, and coincident joints such as train00133 with l_hip = r_hip.
  8. `train_pre_infant`: keypoints are in original COCO coordinates on resized images, and the 2000+ SURREAL entries share one dummy pose.
     This is **harmless**: that loader feeds only the domain classifier (`meta['synthetic']`).
- **Decision / next:** train with `SyRIP/annotations/200R_1000S` labels, restore the correct `test11.jpg`, and evaluate Test500 with area = bbox area.
  Then rerun E0 and E1. Report Test100 numbers from both before and after the fix.

### 2026-11-03 — Part 7 track D: retrain on fixed labels
- **Asked:** to beat the paper (Part 7), retrain with the corrected annotations.
- **Did:** `tools/make_syrip_fixed.sh` builds `data/syrip_fixed/` on the server and the cluster. All files are symlinks, so `data/syrip` and `data/SyRIP` stay untouched:
  - train json → `SyRIP/annotations/200R_1000S`
  - `test11.jpg` → `SyRIP` copy
  - everything else → `data/syrip`
  Checks: ids ↔ file names match, json sizes match the files, and `is_synthetic` counts are 1000 synthetic / 200 real.
  Submitted (cluster code as already pushed, no code change):
  - D0 11602447 (released ckpt, fixed Test100)
  - D0b 11602449 (E1b `model_best`, fixed Test100)
  - D1 11602450 (E1b config + `DATASET.ROOT data/syrip_fixed/ OUTPUT_DIR output/D1 LOG_DIR log/D1`, W&B `D1-fixed-labels`, `--time 1:00`)
- **Note:** a parallel sweep (E1c/E2/E3/E4/p100probe, submitted 20:12 from another session) still trains on `data/syrip/`,
  i.e. with the broken synthetic labels. Its results are not comparable with track D. The QOS allows 2 running jobs per user, so D jobs queue behind it.
- **Hypothesis:** about half of every batch had wrong pose targets. With correct targets, the synthetic half should help instead of hurting,
  so D1 should beat E1b (92.6 on fixed Test100, D0b) and may reach or exceed the paper's 93.6. *(Result: D1 93.8 best / 93.6 final; see the next entries.)*
- **Compare against:** D0 / D0b (same fixed Test100), not the old 92.1 / 91.1.

### 2026-11-03 — Test500 with a valid `area` (third data fix)
- **Did:** `tools/make_syrip500_fixed.py` builds `data/syrip500_fixed/` on the server and the cluster. It is the validate500 json with `area` = bbox w*h
  (the original uses a constant 2443298). Images symlink to `SyRIP/images/validate_infant`. Copied the script and `tools/check_annotations.py` to the cluster
  with a one-off rsync (not via `sync.sh`, which pushes from the main checkout). Submitted T5f-E0 11602599, T5f-E1b 11602600, and T5f-D1 11602601
  (afterok on D1).
- **Note:** the offline Test100 estimate (92.7) was too low. The real D0 eval on `data/syrip_fixed` gave **93.6 / 98.5 / 98.5 / AR 94.6**, which matches the paper.
  The whole E0 gap (92.1 vs 93.6) was the wrong `test11.jpg`.
- **Next:** Test500 (area fixed) is now usable as a second metric. It is not used for model selection, so it avoids the Test100 `model_best` bias.

### 2026-11-03 — Check of the user's data fix (`data/syrip_fixed`) and D-track results
- **Asked:** check the updated fixed data and the model trained on it.
- **Audit of `data/syrip_fixed` (server and cluster):** train ann -> `SyRIP/annotations/200R_1000S` (the correct synthetic labels). `train_pre`
  and val ann are unchanged. `validate_infant/test11.jpg` -> SyRIP's 1536×2048 file (md5 0161fbed on both machines). 0 size mismatches in train/val.
  It is functionally identical to `data/syrip_clean`. Still present: the 4 Test100 <-> annotated-train duplicates (test20/42/46/50).
- **Results (Test100 fixed):** released E0 **93.6** (exactly the paper's DarkPose+FiDIP number; the earlier 92.1 came entirely from the swapped
  test11, not the environment). E1b 92.6. **D1 93.8** best / 93.6 final. Fixing the labels is worth **+1.2 AP** over E1b, and D1 matches or slightly beats
  the released checkpoint (+0.2, within the ±0.3 run noise from E1b vs E1c).
- **Results (Test500 with fixed area, T5f jobs of the 原始数据标注检查 session, not used for model selection):** E0 90.9, E1b 89.7, **D1 91.4**
  (AP50 98.8, AP75 96.7, AR 92.6). D1 beats the released checkpoint by **+0.5** on the larger set.
- **Curves (`figures/D1_curves.png` vs `figures/E1b_curves.png`):** with correct targets the train MSE ends at 2.7e-4 (E1b 6.7e-4) and train PCK at
  0.92 (E1b 0.65). Val AP keeps rising to epoch ~16 instead of plateauing at ~5, so the E1b "plateau" was mostly label noise. Now val loss > train loss
  (ratio 1.16), a normal small generalisation gap with no sign of over/underfitting in 20 epochs. The domain classifier only reaches 0.6–0.83
  accuracy (E1b 1.0) even with DA off; with correct synthetic poses, real and synthetic features are less separable.
- **Decision / next:** D1 is the new reproduction baseline (93.8 Test100 / 91.4 Test500f). Part 7 ablations should restart from D1 on
  `data/syrip_fixed`: replicate D1 (noise), +DA (λ 0.0005, PRE_EPOCH 1), LR 1e-3, 100 epochs. Report Test500f alongside Test100, and consider
  removing the 4 leaked Test100 images (or reporting Test96) for the final numbers.

### 2026-11-03 — Part 7 restarted from D1 on fixed data (F-track)
- **Asked:** restart the Part 7 experiments from D1 on `data/syrip_fixed`.
- **Did:** `slurm/submit_run.sh <NAME> <TIME> [overrides]` (commit cfc5da2) submits the train job plus two `afterok` Test500-fixed-area evals
  (`model_best` -> `output_test500f/<NAME>`, `final_state` -> `output_test500f/<NAME>_final`), and refuses a reused `output/<NAME>`.
  Each run changes one factor vs D1: F1 replicate, F2 +DA, F3 LR 1e-3, F4 100 epochs. W&B is on (project `fidip`, run names = F1..F4).
- **Hypotheses:** F1 measures noise on clean data (E1b vs E1c gave ±0.3 on broken data). F2: with correct labels the DC is no longer trivially
  separating domains (D1 acc_D 0.6–0.83), so λ = 0.0005 should give a small gain, as in paper Table IV (+0.9 Finetune -> FiDIP). F3: the paper's 1e-3 may
  destabilise fine-tuning from COCO weights. F4: D1 was still improving at epoch ~16, so longer training plus the LR drop at 40 should help.
- **Note:** F1's first `submit_run.sh` call hit the local 100 s SSH timeout but kept running on the cluster and submitted its evals (11603312/17).
  My manual re-submits (11603316/23) were duplicates and were cancelled.

### 2026-11-03 — F1–F3 results (fixed data; F4 still running)
| Run | change vs D1 | Test100 best / final | Test500f best / final |
|---|---|---|---|
| D1 | — | 93.8 / 93.6 | 91.4 / — |
| F1 | none (replicate) | 93.7 / 93.5 | 91.2 / 91.2 |
| F2 | DA on (λ 0.0005, PRE_EPOCH 1) | 93.8 / 93.5 | 90.9 / 91.0 |
| F3 | LR 1e-3 | 93.3 / 92.9 | 89.9 / 90.6 |
- **Noise:** D1 vs F1 differ by 0.1–0.2 AP on both test sets. Changes ≤ ~0.3 are not significant from single runs.
- **DA (F2): no gain, maybe a slight loss on Test500f (−0.2…−0.5).** It does what it is meant to do: domain-classifier accuracy stays at 0.58–0.67 vs
  0.46→0.82 in F1, so features become more domain-invariant. But invariance does not translate into keypoint accuracy here.
  The paper's Finetune→FiDIP +0.9 (Table IV) is **not reproduced** on correct labels. Possible reasons: a single λ (paper value), only 20 epochs, and
  stage I/II here alternate every iteration, not as in the paper (not stated). Logged train loss for DA runs is `MSE − λ·BCE` (goes negative), so
  it is not comparable with the other runs (`plot_curves.py` now labels this).
- **LR 1e-3 (F3): worse (−0.5 Test100, −1.5 Test500f best) and noisier.** It also has the lowest train MSE (2.1e-4) and highest train PCK (0.947)
  while val is worse. This is **clear overfitting**: the val/train loss ratio is 1.56 vs 1.25 for F1. The paper's LR is too high for fine-tuning from COCO weights
  in this code. Keep 1e-4.
- **Best so far:** D1/F1/F2 are tied within noise. No ablation beats D1 yet.
- **Next:** wait for F4 (100 ep). Then candidates: lower LR (5e-5) or cosine schedule, stronger augmentation for the 200 real images, λ sweep (1e-4,
  2e-3) if DA is to be argued at all, real-only vs real+syn ablation (paper Table III), MobileNet (`mobile_fidip.pth`) as a backbone comparison,
  and Test96 (drop the 4 leaked images) for final numbers.

### 2026-11-03 — Part 3 notes, 7.3 video/webcam app, 7.1 assessment, sampler finding
- **Asked:** write the Part 3 notes, try 7.1 (synthetic generation) and 7.3 (webcam / video batch). After Part 7's current batch, evaluate
  automatically and choose the next direction.
- **Part 3:** `docs/part3_model_notes.md` covers the HRNet-W48 stages and params, the DC architecture, the alternating stage I/II with `−λ·BCE`, augmentation,
  targets, DarkPose decoding and OKS/AP. Measured: pose net 63.6 M params; frozen `idx <= 108` = stem..stage2 + 1 conv = **1.72 M (2.7 %)**; DC 0.75 M;
  feature tap 48×96×72.
- **New finding:** `BalancedBatchSampler` **never shuffles**. Every epoch uses the same order and the same 100 batches, and the real oversampling is drawn once.
  Added opt-in `TRAIN.BALANCED_SHUFFLE` (default False = identical to before; verified the default sequence is unchanged and the shuffled one keeps the
  real/syn alternation and full coverage).
- **7.3:** `tools/video_infer.py` (file / folder batch / webcam; keypoint-hull tracking; jsonl + csv + annotated mp4). On 3 synthetic pan/zoom clips with
  known GT, tracking beats the whole-frame box by +0.010…+0.024 mean OKS. ~20 fps with flip, 32 fps without (RTX 6000 Ada). Details in
  `docs/part7_synthetic_and_apps.md`.
- **7.1:** blocked. `smil_web.pkl` and the pose prior are licence-gated (user must register). OpenDR 0.78 is not installable on our pythons, and the paths are hard-coded.
  Documented the steps and a pyrender port. A cheap proxy experiment (real-only vs real+syn) decides whether more synthetic data is worth it.
- **Pre-registered rule for the next batch (G-track), applied automatically when F4 finishes.** Metric: Test500f AP of `model_best`.
  Significant = beats D1 (91.4) by ≥ 0.3 (noise D1 vs F1 = 0.2).
  - Base config B = F4's 100-epoch setting if F4 is significant, else D1 (20 ep).
  - Always run **G1 = B + `TRAIN.BALANCED_SHUFFLE True`** (sampler fix) and **G2 = real-only 200R** (`data/syrip_fixed_200R`, matched 2000 iters:
    200 ep with LR_STEP 400,500; ×5 if B is 100-ep) → value of synthetic data / 7.1.
  - **G3 = B + LR 5e-5** (F3 showed 1e-3 overfits; check the other side).
  - Afterwards: adopt whatever is significant into B and continue (augmentation strength, λ sweep only if DA ever helps, MobileNet comparison).

### 2026-11-03 — F4 result and automatic decision -> G-track
- **F4 (100 ep):** Test100 rises to 94.1 at epoch 20 (0-idx) and then sits flat at 93.8–94.1 for 80 epochs (final 94.0). The LR drop at 40 changes nothing on val.
  Val loss keeps creeping down (3.10e-4 -> 3.02e-4) while train loss falls faster (2.8e-4 -> 1.9e-4; ratio 1.11 -> 1.58): a growing generalisation gap, but no
  val degradation (`figures/F4_curves.png`).
  Test500f: **model_best 91.0, final_state 91.7**.
- **Rule applied (pre-registered above):** the metric is Test500f of `model_best`. 91.0 < 91.4 + 0.3, so **not significant → base B = D1 (20 ep)**. Submitted
  G1 (shuffle), G2 (real-only), G3 (LR 5e-5) unchanged from the rule.
- **Observation that changes future rules:** F4's *final* checkpoint (91.7) beats its Test100-selected *best* (91.0) on Test500f, and beats F1-final (91.2) by +0.5.
  Picking the best epoch on Test100 (which is also part of Test500) is a biased, noisy selector. For later decisions both checkpoints are reported, and
  adoption needs **≥ 0.3 on final_state Test500f** (no test-set selection). This is stated before seeing the G results.
- **Pre-registered rule for the H-track (applied automatically when G1–G3 finish):**
  - Adopt into B every G change whose final_state Test500f ≥ B's final + 0.3. B's final is F1-final 91.2 for the 20-ep D1 config.
  - **H1 = F4 replicate (100 ep)**, to check whether the +0.5 final-checkpoint gain is real (one run can't distinguish it from noise).
  - **H2 = B + stronger augmentation** (`DATASET.ROT_FACTOR 60 DATASET.SCALE_FACTOR 0.45`): the 200 real images are seen 5×/epoch, and the train/val gap grows.
  - G2 interpretation: if real-only is within 0.3 of D1 (final), synthetic data adds ~nothing → 7.1 (generate more synthetic) deprioritised. If clearly worse,
    synthetic data matters → 7.1 is worth the SMIL setup.

### 2026-11-04 — G-track results and automatic decision -> H-track
| Run | change vs D1 | Test100 best / final | Test500f best / **final** |
|---|---|---|---|
| F1 (D1 replicate, reference) | — | 93.7 / 93.5 | 91.2 / **91.2** |
| G1 | sampler reshuffle per epoch | **94.2 / 94.0** | 91.4 / **91.5** |
| G2 | real-only 200R (matched 2000 iters) | 93.9 / 93.9 | 91.3 / **91.3** |
| G3 | LR 5e-5 | 93.4 / 93.3 | 90.9 / **91.0** |
- **Rule (adopt if final Test500f ≥ 91.2 + 0.3 = 91.5):** **G1 adopted** (91.5, exactly at the bar; it also has the best Test100 of any run, +0.5 vs F1).
  It is borderline from a single run, so H3 replicates it (extra run, outside the rule). New base **B = D1 + `TRAIN.BALANCED_SHUFFLE True`**. G3 is not adopted;
  together with F3 this means LR 1e-4 is a sweet spot (5e-5 underfits slightly, 1e-3 overfits).
- **G2 (value of synthetic data): real-only reaches 91.3, within 0.1 of real+syn (F1 91.2).** At matched iterations the 1000 synthetic images add
  nothing measurable for a COCO-pretrained HRNet-W48. That is unlike the paper's Table III (SimpleBaseline-50: 200 real 87.1 vs 1200 R+S 90.1).
  Per the rule, **7.1 (generating more synthetic data with SMIL) is deprioritised**. G2's best epoch is its last, so real-only may not have saturated yet.
- **H-track submitted per rule:** H1 = F4 replicate (does 100 ep really help on the final checkpoint?), H2 = B + stronger augmentation, plus H3 = G1 replicate.
- **Pre-registered wrap-up after H (no more automatic batches after that):** a config counts as *confirmed* if its replicate pair has mean final
  Test500f ≥ 91.5 and both runs ≥ 91.2. Pick the final model = best mean final Test500f among confirmed configs (else D1). Report Test100,
  Test500f and Test96 (Test100 minus the 4 leaked images), then stop and ask the user before launching anything else.
- **H-track results / wrap-up applied (2026-11-04).** Final-checkpoint Test500f: H1 91.6, H2 91.6, H3 91.3.
  - **100 ep (F4/H1): 91.7 / 91.6, mean 91.65, both ≥ 91.2 → confirmed.** Only confirmed config.
  - Shuffle (G1/H3): 91.5 / 91.3, mean 91.4 → **not confirmed**; G1's gain was mostly noise. Base reverts to D1.
  - H2 (shuffle + strong aug): 91.6 from one run; promising but unconfirmed and on the unconfirmed shuffle base.
  - **Final model = F4 `final_state.pth`** (higher of the pair). Test100 94.0 / Test500f 91.7 / Test96 93.2, vs D1 93.6 / 91.4 / 92.9 and
    released E0 93.6 / 90.9 / 92.9 (Test96 via `tools/eval_test96.py`, all final-epoch predictions).
  - Lesson: for 100-ep runs the Test100-selected `model_best` is *worse* on Test500f (F4 91.0, H1 91.4) than `final_state`; selecting on the
    100-image test set overfits it. Report final checkpoints.
  - Test96 is ~0.7 below Test100 for every run (the 4 leaked images are easy); ranking unchanged.
  - Automatic batches stopped here, per rule; next steps need the user's go-ahead.

### 2026-11-04: Part 4 and Part 6 redone with the final model (user: "you didn't really do 6. update 4")
- **Part 4:** checkpoints are now copied from the cluster (`models/F4`, `models/D1`, both `final_state.pth`).
  - `tools/visualize_preds.py` compares E0/D1/F4 on fixed Test100 → `figures/viz_final/`.
  - F4 gains on hips, left ankle and left elbow. test33 and test41 (self-occlusion) remain the worst images for every model.
  - The 4.5 table now covers all runs and H1–H3 curves are drawn. Write-up: `docs/part4_results_and_fitting.md`.
  - Legend fixes in both plot tools.
- **Part 6:** earlier, 6.1/6.2 had only been tested on a SyRIP image (6.2 only through AppTest), and 6.3 had no images.
  - Added a person detector (`fidip_infer.detect_people`, torchvision Faster R-CNN) with `--box-mode/--person` in the CLI and a box/person picker
    in Streamlit. F4 is the default model.
  - 6.3 uses 12 Commons photos that are not in SyRIP (`temp_uploads/`, licences in `SOURCES.md`).
  - Streamlit was actually served and driven with headless Chromium (3 screenshots).
  - Finding: the box matters more than the model (whole image 0.77 → detected box 0.86 mean confidence). F4 is more confident than E0 on 11/12.
  - Failure modes: picking an adult as the "largest person", swaddling, foreshortening, side-view occlusion. Write-up: `docs/part6_demos.md`.

### 2026-11-04 — test11 dropped from eval, Part 4.5 corrected, Part 7 target fixed at 93.6 on fixed data
- **Asked:** test11's GT is wrong, drop it. 6.3 with images from the web. Correct the 4.5 analysis. Update AGENT.md and MANIFEST.md.
  Part 7 must beat 93.6, compared on the fixed data.
- **test11 check before dropping:** the GT (1536×2048, supine, 17 kps v=2) lines up with SyRIP's `test11.jpg` for head, arms and left leg;
  left ear, both hips and the right knee/ankle are off by 50–90 px (`left_ear` is drawn on the face). Every fixed-data model scores it OKS ≈ 1 at all
  thresholds, so it is an *easy* image under this loose GT. **Dropping it lowers every model's AP by ~1.0** (one perfect image out of 100).
  Decision (user): drop it from all reported numbers.
- **New eval subsets** (`tools/eval_subsets.py`, rescoring saved prediction JSONs with pycocotools, no re-inference):
  Test99 = fixed Test100 − test11; Test95 = Test99 − 4 train-leaked images (test20/42/46/50); Test499f = Test500f − test11.
  Test100-family predictions are each run's last training-loop validation (= `final_state`), Test500-family are `<run>_final` evals
  (D1, E0, E1b: their only Test500f eval, `model_best` for D1/E1b).

| run | what | Test100 | **Test99** | Test95 | Test500f | **Test499f** |
|---|---|---|---|---|---|---|
| E0 (D0) | released `hrnet_fidip.pth` = paper's model | 93.6 | **92.6** | 92.8 | 90.9 | **90.7** |
| E1b (D0b) | our repro on broken labels | 92.6 | 91.6 | 91.8 | 89.7 | 89.6 |
| D1 | repro on fixed labels, 20 ep | 93.6 | 92.7 | 92.9 | 91.4* | 91.2* |
| F1 | D1 replicate | 93.5 | 92.5 | 92.7 | 91.2 | 91.1 |
| F2 | + DA | 93.5 | 92.5 | 92.8 | 91.0 | 90.9 |
| F3 | LR 1e-3 | 92.9 | 92.0 | 92.1 | 90.6 | 90.6 |
| **F4** | 100 ep (final model) | **94.0** | **93.0** | **93.2** | **91.7** | **91.4** |
| G1 | shuffle | 94.0 | 93.0 | 93.2 | 91.5 | 91.2 |
| G2 | real-only 200R | 93.9 | 93.0 | 93.1 | 91.3 | 91.1 |
| G3 | LR 5e-5 | 93.3 | 92.3 | 92.5 | 91.0 | 90.9 |
| H1 | F4 replicate | 93.9 | 93.0 | 93.2 | 91.6 | 91.4 |
| H2 | shuffle + strong aug | 93.9 | 92.9 | 93.0 | 91.6 | 91.5 |
| H3 | G1 replicate | 93.9 | 92.9 | 93.1 | 91.3 | 91.3 |

  \* D1 Test500f is `model_best` (no final-state Test500f eval).
- **Part 7 target:** the paper's **93.6** is DarkPose+FiDIP on Test100. On the fixed Test100 the released checkpoint reproduces it exactly (93.6), so the
  fixed data is the right place to compare. **F4 final = 94.0 > 93.6 (+0.4).** Once test11 is dropped, 93.6 is no longer the comparable number (it
  includes test11). The equivalent bar is the released model on the same subset: **Test99 92.6 → F4 93.0 (+0.4)**, Test95 92.8 → 93.2 (+0.4),
  Test499f 90.7 → 91.4 (+0.7). The F4/H1 replicate pair agrees to 0.0 on Test99 and Test499f, so the gain is not single-run noise (noise ≈ 0.2).
  The ranking of all runs is unchanged by dropping test11 (every run loses 0.9–1.0 on Test100→Test99).
- **Part 4.5 corrected** (`tools/plot_compare.py` → `figures/compare_4_5.png`; E1b vs D1 vs F3 vs F4 on one set of axes; write-up in
  `docs/part4_results_and_fitting.md` §4.5.0):
  - The original observation (E1b: "AP stalls after epoch 6, train loss stays flat") is real, but the earlier explanation (mild overfitting onset) was wrong.
    E1b's train loss starts at 1.02e-3 and ends at 6.7e-4, 2.5× above D1 (4.2e-4 → 2.7e-4). E1b's val loss is *below* its train loss (ratio 0.59),
    which is impossible for a healthy fit. Both are signatures of **label noise**: 1000 of the 1200 training targets were MINI-RGBD poses that do not match the
    images, so half of every batch is unlearnable. Its val AP (on the swapped-test11 Test100) peaks at 0.911 @ ep 7 and then wanders.
  - With correct labels (D1) val AP keeps rising to ep 17 (0.938) and val loss falls monotonically; val/train = 1.16, a normal small gap.
  - Real fitting regimes, on fixed data: LR 1e-3 (F3) **overfits** (lowest train loss, val loss rises after ep 6, ratio 1.56, AP 0.929).
    100 ep (F4) widens the gap to 1.58 but val loss keeps creeping *down* (3.02e-4), and AP holds at 0.94 → **benign plateau, not harmful overfitting**.
    LR 5e-5 (G3) mildly underfits. Nothing at LR 1e-4 underfits.
- **6.3:** already done earlier on 2026-11-04 (12 Wikimedia Commons infant photos, not in SyRIP, in `temp_uploads/` with licences in
  `temp_uploads/SOURCES.md`; results in `docs/part6_demos.md`). Nothing re-run.

### 2026-11-04 — L/M/S-track: λ sweep, MobileNet comparison, 100 ep + shuffle (rules pre-registered before results)
- **Asked:** sweep λ, compare against MobileNet, and run long training with the reshuffled sampler. Monitor progress every 10 min.
- **Code (new files only, see the incident below):** `slurm/submit_mobile.sh` → `slurm/train_mobile.sh` (`tools/train_adaptive_model_mobile_wb.py` =
  upstream mobile script + W&B + per-epoch `val AP` line; 224×224 yaml; init `models/coco/posemobile.pth`) and `slurm/eval_mobile.sh`.
  `tools/eval_subsets.py` now also finds `adaptive_pose_mobile` outputs. These were pushed to the cluster by one-off rsync from this worktree (branch
  `claude/raw-data-annotation-check-23431c`).
- **Runs** (all `data/syrip_fixed`, single factor vs the named base):
  - **L1 / L2 / L3**: D1 + DA with `PRE_EPOCH 1`, λ = 1e-4 / 2e-3 / 5e-3, 20 ep. Together with F1 (λ = 0) and F2 (λ = 5e-4) this gives a 5-point sweep.
  - **M0**: released `mobile_fidip.pth` on fixed Test100 and Test500f (eval only).
  - **M1**: MobileNet on fixed labels, its own yaml (LR 1e-3, 20 ep, DA off). **M2**: M1 + 100 ep (LR drop at 90).
  - **S1 / S2**: F4 (100 ep) + `TRAIN.BALANCED_SHUFFLE True`, two runs (pair), so the rule can be applied without another round.
- **Metric:** final_state AP on **Test499f**, with Test99 as a secondary metric (`tools/eval_subsets.py`). Noise ≈ 0.2, so the significance bar is +0.3.
- **Pre-registered rules:**
  - **λ:** baseline F1 final Test499f 91.1. A λ is *promising* if it scores ≥ 91.4. Then it gets one replicate, and it is adopted only if the pair mean is ≥ 91.4
    and both runs are ≥ 91.1. If no λ is promising, the conclusion is that **DA does not help on fixed labels at any λ in [1e-4, 5e-3]**. Separately, report
    whether the domain-classifier accuracy falls toward 0.5 as λ grows (a sanity check that the adversarial term works).
  - **Shuffle at 100 ep:** baseline F4/H1 pair mean 91.4 (91.4 / 91.4). *Confirmed* if the S1/S2 mean is ≥ 91.7 and both runs are ≥ 91.4. If confirmed, the final model
    becomes the better of S1/S2. Otherwise F4 stays the final model.
  - **MobileNet:** a comparison, not a candidate for the final model. Report M0 vs M1 (does the label fix help MobileNet as it did HRNet, +1.2?),
    M1 vs M2 (does 100 ep help, ≥ +0.3?), and the gap between HRNet and MobileNet at matched settings (D1 vs M1, F4 vs M2).
  - After these runs, stop and report. Nothing new is launched without the user.
- **Job ids:** M0 t100 11617333, t500f 11617336. L1 11617341 → t5b 11617348, t5f 11617351. L2 11617353 → 11617376, 11617492.
  L3 11617418 → 11617494, 11617495. S1 11617384 → 11617393, 11617423. S2 11617427 → 11617435, 11617437.
  M1 11617422 and M2 11617442 (→ 11617447/49) were wrong (HRNet, see the incident): held, then cancelled with the user's OK and resubmitted with
  `slurm/submit_mobile.sh`: **M1 11618020 → 11618092, 11618105; M2 11618140 → 11618148, 11618151** (arch=mobile).
- **Incident (11:05):** in the middle of the submit loop, the cluster's `slurm/submit_run.sh`, `slurm/eval.sh` and `tools/train_adaptive_model_mobile.py` reverted to
  main's versions, most likely because another session ran `slurm/sync.sh push` from the main checkout. So `ARCH=mobile` was ignored: **M1 and M2 were submitted as HRNet
  runs (`slurm/train.sh`)**. Both were held (`scontrol hold`) before starting. Three sbatch calls also failed with "Unexpected message received"
  (L2-t5f, L3-t5b/t5f); they were resubmitted by hand. L1–L3 and S1/S2 are HRNet by design, so they are unaffected. M0 is unaffected too, because sbatch snapshots the
  batch script at submit time. Fix: the MobileNet path now lives only in new files that main does not have, so a push cannot revert them.
- **M0 (released `mobile_fidip.pth`, fixed data):** Test100 **79.2** (AP50 99.0, AP75 89.4, AR 84.1). Test500f **69.8** (96.2 / 79.9 / AR 78.1).
  That is 14–21 AP below HRNet (93.6 / 90.9).
- **λ sweep result (done 2026-11-04 16:20; all final_state, 20 ep, `PRE_EPOCH 1` for λ > 0):**

| run | λ | Test100 | Test99 | Test95 | Test500f | **Test499f** | DC acc (mean of last 5 ep) |
|---|---|---|---|---|---|---|---|
| F1 | 0 | 93.5 | 92.5 | 92.7 | 91.2 | **91.1** | 0.78 |
| L1 | 1e-4 | 93.7 | 92.8 | 92.9 | 91.4 | **91.1** | 0.71 |
| F2 | 5e-4 (paper) | 93.5 | 92.5 | 92.8 | 91.0 | **90.9** | 0.69 |
| L2 | 2e-3 | 92.4 | 91.4 | 91.7 | 89.6 | **89.6** | 0.60 |
| L3 | 5e-3 | 89.7 | 88.8 | 89.3 | 87.3 | **87.2** | 0.54 |

  - **Rule applied:** no λ reaches 91.4 (the best is L1 at 91.1, equal to F1), so none is promising. **DA does not help on fixed labels at any λ in [1e-4, 5e-3].** Replicates are not needed.
  - **Sanity check passes:** domain-classifier accuracy falls steadily toward chance (0.78 → 0.54) as λ grows, so the adversarial term does make the
    features domain-invariant. Pose AP falls with it (−0.2 at the paper's λ, −1.5 at 2e-3, −3.9 at 5e-3). On correct labels, real-vs-synthetic invariance
    costs keypoint accuracy instead of helping. The paper's Finetune→FiDIP +0.9 (Table IV) does not reproduce in this code at any λ.
- **100 ep + shuffle result (S1/S2, final_state):**

| run | Test100 | Test99 | Test95 | Test500f | **Test499f** |
|---|---|---|---|---|---|
| F4 / H1 (100 ep, no shuffle) | 94.0 / 93.9 | 93.0 / 93.0 | 93.2 / 93.2 | 91.7 / 91.6 | **91.4 / 91.4** |
| S1 / S2 (100 ep + shuffle) | 94.3 / 94.2 | 93.3 / 93.2 | 93.5 / 93.4 | 91.5 / 91.7 | **91.4 / 91.6** |

  - **Rule applied:** the pair mean is 91.5, below the 91.7 bar (both runs are ≥ 91.4) → **shuffle not confirmed**, **F4 stays the final model**. This is the same outcome as G1/H3 at 20 ep.
    The shuffle pair is consistently a little higher (+0.25 Test99, +0.1 Test499f), but the gain is below the pre-registered significance bar.
  - **Shuffle, pooled over both schedules (report wording):** *shuffle shows a consistent small positive trend (about +0.1–0.3) that does not reach the
    significance bar.* The 4 shuffle runs vs their matched no-shuffle runs, final_state:

    | schedule | no shuffle (Test99 / Test499f) | shuffle (Test99 / Test499f) | Δ Test99 | Δ Test499f |
    |---|---|---|---|---|
    | 20 ep (D1 base) | D1 92.7 / 91.2, F1 92.5 / 91.1 | G1 93.0 / 91.2, H3 92.9 / 91.3 | +0.35 | +0.1 |
    | 100 ep (F4 base) | F4 93.0 / 91.4, H1 93.0 / 91.4 | S1 93.3 / 91.4, S2 93.2 / 91.6 | +0.25 | +0.1 |

    All 4 shuffle runs are ≥ their no-shuffle counterparts on Test99, and none is lower on Test499f. The effect is too small to confirm with pairs
    (noise ≈ 0.2, bar 0.3 on Test499f), so it is not adopted and the final model stays F4. Shuffle is a bug fix (the stock `BalancedBatchSampler` uses
    the same batch order and the same real-image oversampling every epoch), so a small, non-negative effect is what one would expect.
    *Decision (user, 2026-11-04): report it as a trend, no further replicates.*
- **MobileNet result (final_state, `mobile_224x224` yaml: LR 1e-3, LR drop at ep 90, DA off):**

| run | what | Test100 | Test99 | Test95 | Test500f | **Test499f** |
|---|---|---|---|---|---|---|
| M0 | released `mobile_fidip.pth` | 79.2 | 78.2 | 77.9 | 69.8 | **69.7** |
| M1 | fixed labels, 20 ep | 74.7 | 74.0 | 73.9 | 69.2 | **69.0** |
| M2 | fixed labels, 100 ep | 79.4 | 78.6 | 78.3 | 73.2 | **73.1** |

  - **The label fix alone does not help MobileNet at 20 ep** (M1 vs M0: −4.2 Test99, −0.7 Test499f). For HRNet it was worth +1.2. A 224×224 MobileNet initialised from
    COCO is still far from converged after 20 epochs (val AP is still rising at ep 19: best 76.4).
  - **100 epochs is worth +4.6 Test99 / +4.1 Test499f** (M2 vs M1, ≫ 0.3), and M2 beats the released MobileNet by +0.4 Test99 and **+3.4 Test499f**. MobileNet needs the long schedule. HRNet needs it much less (F4 vs D1: +0.3 / +0.2).
  - **HRNet vs MobileNet at matched settings:** D1 vs M1 = 92.7 vs 74.0 Test99 (−18.7), 91.2 vs 69.0 Test499f (−22.2). F4 vs M2 = 93.0 vs 78.6 (−14.4), 91.4 vs 73.1 (−18.3).
    The gap is larger on Test500, where poses are more varied. AP50 stays high for MobileNet (≥ 96), so the loss comes from localisation precision
    (224 input, 56×56 heatmaps vs 384×288 / 96×72), not from missed detections.
- **Wrap-up (per the pre-registered stop rule):** no change to the final model (**F4 `final_state`**). Nothing new launched. The P100 test job 11619062 is still queued (it only checks that a P100 can run CUDA).

### 2026-11-04 — A/B-track: ensemble, TTA, OHKM, stronger rotation (rules pre-registered before any result)
- **Asked:** try A1 (model ensemble), A2 (test-time augmentation), B3 (OHKM) and B4 (stronger rotation augmentation).
- **Code (new files only, so `sync.sh push` from main cannot revert them):**
  - `tools/eval_ensemble_tta.py` + `slurm/eval_ens.sh`: official eval path.
    - For each transform (scale, rotation), crop → each model + flip (as in `validate_adaptive`) → warp the heatmaps back to the reference frame → average
      over transforms and models → the same DARK decoding (`get_final_preds`) → `dataset.evaluate`.
    - With one model and the identity transform it equals the official eval. Sanity run **A0** must reproduce F4 (Test100 94.0 / Test500f 91.7).
  - `tools/train_adaptive_model_hrnet_x.py` + `slurm/train_x.sh` + `slurm/submit_x.sh`: stock training script, plus `LOSS.USE_OHKM` used for the
    *training* pose loss (`JointsOHKMMSELoss`, top-8 of 17 joints). Validation stays plain MSE. `submit_x.sh` uses partitions `gpu,courses-gpu` (V100-SXM2, same GPU type as F4).
- **Runs** (all on fixed data, final_state, base = F4 = D1 + 100 ep):
  - **A0** = F4 alone, flip only (sanity check).
  - **A1** = ensemble {F4, H1, S1, S2}, flip only.
  - **A2** = F4 alone + TTA: flip × scales {0.9, 1.0, 1.1} × rotations {−30°, 0°, +30°} (9 transforms).
  - **A3** = A1 + A2 (4 models × 9 transforms).
  - **B3a / B3b** = F4 + `LOSS.USE_OHKM True` (pair).
  - **B4a / B4b** = F4 + `DATASET.ROT_FACTOR 60` (default 45; pair; scale aug unchanged, unlike H2).
- **Metric:** Test499f (Test99 secondary), via `tools/eval_subsets.py`.
- **Pre-registered rules:**
  - **A (deterministic, no replicate):** a config is adopted if its Test499f ≥ **91.7** (F4 91.4 + 0.3). If several pass, the highest wins. The final *reported
    system* then becomes "F4-based ensemble/TTA". Its inference cost (number of forward passes) is reported alongside. If A0 does not reproduce F4 within 0.1,
    every A result is void until the tool is fixed.
  - **B (training, pair):** confirmed if the pair mean Test499f is ≥ **91.7** and both runs are ≥ 91.4 (same bar as S1/S2). A confirmed B change replaces F4 as the final
    *single* model (the better of its pair). It is not combined with A unless the user asks.
  - Stop and report afterwards. Nothing new without the user.
- **Job ids:** ENS_A0 t100 11625786 / t500 11625787, A1 11625790 / 11625795, A2 11625799 / 11625800, A3 11625802 / 11625804 (`slurm/eval_ens.sh`).
  B3a 11625832 → 11625833, 11625835. B3b 11625837 → 11625839, 11625840. B4a 11625841 → 11625842, 11625844. B4b 11625846 → 11625847, 11625859
  (`PART=courses-gpu slurm/submit_x.sh`: the gpu QOS caps *submitted* jobs at 8 per user and the 8 evals filled it).
- **A0 sanity check passed:** Test100 93.99 (official F4 94.0), Test500f 91.73 (official 91.7). The tool matches the official eval, so A1–A3 are valid.
- **Bug, fixed:** `slurm/eval_ens.sh` passed `--rots -30,0,30`, and argparse read `-30,...` as an option flag. A3-t500 (11625804) failed at argument parsing.
  The other first-round A2/A3 jobs were left to fail on their own (no scancel). The fix is `--rots=...`/`--scales=...`, resubmitted on `gpu-short`:
  A2 t100 11625924, t500 11625943; A3 t100 11625950, t500 11625953.
- **Interim (17:50):** A1 (4-model ensemble) Test100 94.2 / Test99 93.2 / Test500f 91.8 / **Test499f 91.5** (F4 91.4) → +0.1, below the 91.7 bar.
  A2 (F4 + 9-way TTA) Test100 94.4 / Test99 93.4; Test500f pending. B3a and B3b are training; B4a and B4b are queued.
- **A result (all done 18:05; 3-decimal values from the same rescoring):**

| config | models × transforms | forward passes / person | Test100 | Test99 | Test95 | Test500f | **Test499f** |
|---|---|---|---|---|---|---|---|
| F4 = A0 (flip only) | 1 × 1 | 2 | 93.99 | 92.99 | 93.20 | 91.73 | **91.43** |
| A1 ensemble {F4,H1,S1,S2} | 4 × 1 | 8 | 94.23 | 93.22 | 93.39 | 91.79 | **91.49** |
| A2 F4 + TTA (s 0.9/1/1.1 × r ±30°/0) | 1 × 9 | 18 | 94.42 | 93.42 | 93.62 | 92.16 | **91.96** |
| A3 ensemble + TTA | 4 × 9 | 72 | 94.45 | 93.45 | 93.60 | 92.24 | **92.04** |

  - **Rule applied:** A2 (91.96) and A3 (92.04) both pass the 91.7 bar. A1 (+0.06) does not. **Highest wins → A3 is adopted as the final reported
    system**: Test99 **93.4** / Test499f **92.0**, vs F4 93.0 / 91.4 and vs the released checkpoint 92.6 / 90.7.
  - **TTA does almost all the work.** On top of TTA, the 4-model ensemble adds only +0.08 Test499f at 4× the cost. A2 (single F4 model, 18 passes) is the practical choice
    and is within 0.1 of A3. The ensemble members are near-copies (same data, same config family, plateaued), so their errors are correlated. TTA's
    scale/rotation views see the infant differently, which suits the arbitrary orientations of lying infants.
  - Caveat: the TTA grid was fixed in advance and not tuned, so it is not selected on the test set. Only the A1–A3 choice among 3 pre-listed configs touches
    the test set, and the winner clears the bar by 0.3.
- **B3 (OHKM) result (final_state):**

| run | Test100 | Test99 | Test95 | Test500f | **Test499f** |
|---|---|---|---|---|---|
| F4 / H1 | 93.99 / 93.93 | 92.99 / 93.03 | 93.20 / 93.16 | 91.73 / 91.60 | **91.43 / 91.39** |
| B3a / B3b (OHKM top-8) | 94.00 / 94.14 | 93.01 / 93.14 | 93.20 / 93.25 | 91.51 / 91.17 | **91.40 / 91.06** |

  - **Rule applied:** the pair mean is 91.23, below 91.7, and B3b (91.06) is below 91.4 → **OHKM not confirmed**. It is flat on Test99 (+0.07) and slightly worse on Test500f (−0.3).
    Mining the 8 hardest joints does not help here: the remaining error sits on a few self-occluded images, not on a fixed set of joints.
- **B4a / B4b FAILED (19:04, both at once on different nodes):** `RuntimeError: File output/B4x/.../checkpoint.pth cannot be opened`. Cause: the cluster
  **/home quota is full** (78.9 GB used / 76 GB limit). The big items are 50 `feature.npy` files (28.3 GB; written by `test_adaptive_model.py` on every eval,
  never used) and 28 resume-only `checkpoint.pth` files (19.6 GB; contain optimizer state, not needed once a run is finished). `/scratch/hou.yuchen` (16 TB quota) is unused.
  B4 got to epoch 3/1. Nothing was deleted or resubmitted until the user decided.
- **Cleanup (user approved "删并重跑"):** deleted every `feature.npy` (53 files, 30.2 GB) and every `checkpoint.pth` (28 files, 19.6 GB) under `output*`
  (queue was empty, so all were finished runs; `final_state.pth` / `model_best.pth` / results json untouched). Removed the broken `output/B4a`, `output/B4b`,
  `log/B4a`, `log/B4b`. /home usage 78.9 GB → **31.4 GB** (quota updates within a minute).
- **B4 rerun** (same names, same overrides `TRAIN.END_EPOCH 100 DATASET.ROT_FACTOR 60`, partitions `gpu,courses-gpu`, same rule):
  B4a 11631221 → 11631227, 11631230. B4b 11631233 → 11631234, 11631235.
  Both trains sat in PENDING (Priority) because `train_x.sh` pins `--gres=gpu:v100-sxm2:1`. On the user's request ("use hou.yuchen's gpu resource or imani group")
  they were moved with `scontrol update jobid=… Partition=imanilab Account=m.imani TresPerNode=gres/gpu:a100:1` and started at once on **d3233 (A100 80GB PCIe)**.
  The 4 evals got `Partition=imanilab,gpu,courses-gpu Account=m.imani`. **Caveat:** F4/H1/B3 trained on V100-SXM2. On A100, PyTorch runs convolutions in TF32
  by default, so B4 differs from them in GPU type as well as ROT_FACTOR. That is expected to be small next to the ~0.2 run-to-run noise, but note it if B4 lands near the bar.
- **B4 result** (all 6 jobs COMPLETED; train 61 min each on d3233, best val AP 0.9424 / 0.9417; scored with `tools/eval_subsets.py`, final_state):

| run | Test100 | Test99 | Test95 | Test500f | **Test499f** |
|---|---|---|---|---|---|
| F4 / H1 | 93.99 / 93.93 | 92.99 / 93.03 | 93.20 / 93.16 | 91.73 / 91.60 | **91.43 / 91.39** |
| B4a / B4b (ROT_FACTOR 60, A100) | 93.84 / 93.96 | 92.85 / 92.96 | 93.01 / 93.11 | 91.38 / 91.60 | **91.28 / 91.42** |

  - **Rule applied:** the pair mean is 91.35, below 91.7, and B4a (91.28) is below 91.4 → **stronger rotation not confirmed**. It is flat to −0.1 on every subset.
    The ±45° default already covers the infant poses, so widening it to ±60° adds no useful variety. Being this far from the bar, the TF32 caveat does not change the verdict.
  - **A/B track closed.** A3 (ensemble + TTA) stays the final reported system and F4 the final single model. TTA (A2) was the only change that passed the bar; both training-side
    changes (B3 OHKM, B4 rotation) did nothing. Stopping as pre-registered; nothing new without the user.

### 2026-11-04 — File transfers go through the xfer node
- **Asked:** copy files between the server and the cluster through `xfer.discovery.neu.edu`, and add this to SKILL.md.
- **Did:** checked key login (`hou.yuchen@xfer.discovery.neu.edu`, host `xfer-00`, host key added with `accept-new`). It mounts the same `/discovery/home`, so
  paths are identical. `slurm/sync.sh` now runs every rsync and its helper ssh calls (listing `output*`, writing `.env`) against the xfer node. Slurm commands still use
  `ssh explorer`. Tested `sync.sh pull`: 14.5 s. Rule added to `SKILL.md` (Hosts table, Sync model, weights copy) and `.claude/skills/use_explorer_cluster/SKILL.md`.
- **Note:** the main checkout's `slurm/sync.sh` is still the old login-node version until this branch is merged. Until then, run this branch's copy
  (`bash .claude/worktrees/raw-data-annotation-check-23431c/slurm/sync.sh pull`; it `cd`s to the main checkout anyway).
### 2026-11-04: Part 7.1 synthetic generation with SMIL (user downloaded `syn_generation/models/smil.zip`)
- Delegated to a subagent; I checked the result (figures, json bounds, git staging).
- **Model conversion:** chumpy pickles → numpy npz through a restricted unpickler, so chumpy isn't needed. Fitting is in float64 with SMIL's own
  1-Gaussian prior and small torch-2 fixes. The renderer is ported from OpenDR to pyrender EGL. Everything runs in `.venv-smil`.
- **Fits:** SMIL fitted to all 200 real training images. Reprojection error: mean 3.0%, median 2.1% of the box diagonal.
  - 174 fits at ≤ 5% are kept; some fits between 5% and 8% were wrong in 3D.
  - About 3 s per image with 20 CPU jobs.
- **New finding, hips:** SMIL hip joints sit inside the pelvis, only 0.39× the annotated hip width.
  - A calibrated outward offset of 2.8 cm brings the ratio to 0.89 and cuts cross-validated hip error from 6.0% to 3.8%.
  - The original 1000S set has narrow hips too (hip/shoulder 0.64 vs real 0.75), which may be one reason it helps little (G2).
- **New set `data/syrip_syn_new/`:** 522 images (174 poses × 3 variants). COCO json with the same fields as `syrip_fixed`; ids start at 20001.
- **Label checks:** 0 keypoints out of bounds (checked again independently); left/right agrees with the 3D body direction in 445/445.
  Visual grid: `figures/syn/syn_label_check.jpg`.
- **Limitations:** no LSUN backgrounds (generated textures and blurred crops of real scenes instead); 12 diaper textures; top-down views only;
  poses are the real poses plus jitter, so it adds appearance and viewpoint, not new poses.
- **Not trained yet.** Proposed experiment: F4 config with the new set added, run as a replicated pair and compared with F4/H1 on final Test500f.
  Awaiting the user's go-ahead per the stop-and-ask rule.
- Licensed SMIL files are kept out of git by `syn_generation/.gitignore`. Write-up: `docs/part7_synthetic_and_apps.md` §7.1.

### 2026-11-04: SY-track submitted (user: "Yes, train it")
- **Data:** `data/syrip_fixed_plus522/`, built by `slurm/make_plus_syn_data.sh` on both machines. Train json = syrip_fixed train (1200 = 200R + 1000S)
  + 522 new; 1722 images, 1522 synthetic. Validate and pre-train are symlinks to syrip_fixed.
- **Iterations:** the balanced sampler oversamples real images to the synthetic count, so an epoch is 3044 samples vs 2000 for F4 (1.52× the
  iterations at the same 100 epochs, LR drop at 40).
- **Names:** SY1/SY2, because another session already has S1/S2 (and L*, M*) queued on the cluster. Those are not ours; leave them alone.
- **Pre-registered reading:** a gain counts only if the SY mean final Test500f ≥ 91.95 (F4/H1 mean 91.65 + 0.3) and both runs ≥ 91.6.
  - If so: SY becomes the final model.
  - If within ±0.3 of 91.65: no measurable effect, consistent with G2 (synthetic data adds nothing here).
  - If ≤ 91.35: it hurts (likely the synthetic/real appearance gap).
  - Report Test100 / Test96 / Test500f either way, and stop and ask before more runs.

### 2026-11-04: SY-track result: no effect; F4 stays final
| Run | Test100 best | Test100 final | Test96 final | Test500f best | Test500f final |
|--|--|--|--|--|--|
| F4 | 94.1 | 94.0 | 93.2 | 91.0 | 91.7 |
| H1 | 94.3 | 93.9 | 93.3 | 91.4 | 91.6 |
| SY1 | 93.8 | 93.2 | 92.6 | 91.4 | 91.7 |
| SY2 | 93.5 | 93.1 | 92.5 | 90.9 | 91.5 |
- **Pre-registered verdict:** the SY mean final Test500f is 91.6. That is within ±0.3 of 91.65, so **no measurable effect**.
  - The 522 new SMIL renders add nothing on the large test set, which matches G2.
- **Secondary signal (not the decision metric):** both replicates are about 0.8 lower on Test100 final and 0.7 lower on Test96.
  - The drop is consistent (the F4/H1 spread is 0.1, the SY spread 0.1), so on the small test set extra synthetic data mildly hurts.
- **Confounds:**
  - SY took 1.52× more iterations per epoch.
  - Real images were oversampled 7.6× instead of 5×, so each real image is seen more often.
  - Synthetic diversity is limited: 174 poses × 3 variants.
- **Final model unchanged: F4 final_state.** No further runs without the user's go-ahead.

### 2026-11-04: Grading-rubric pass and report draft (`REPORT.md`)
- **Asked:** organise the existing results against the course rubric (50 pts + 5 bonus), list unfinished items, draft the report.
- **Did:** rescored every run on Test100/99/95/500f/499f with the annotation branch's `tools/eval_subsets.py` (numbers match the logs). New on
  this branch: SY1/SY2 on Test99 = 92.4 / 92.3 (F4/H1 93.0), Test499f 91.5 / 91.2. Paired image bootstrap vs the released model
  (`tools/report_stats_bootstrap.py`, B = 1000): F4 Test99 +0.5 [−0.1, +1.2], Test499f +0.7 [+0.2, +1.3]; A2 (F4 + TTA) +1.0 [+0.2, +1.8] / +1.3 [+0.7, +1.9].
  Fig. 1 `figures/report/ablation_overview.png` (`tools/plot_report_ablation.py`).
- **Open:** B4a/B4b still training (A100).

### 2026-11-04: Results and generated data now tracked in git (user: "sync all data with repo, update the path in report as in repo")
- **Did:** merged `claude/raw-data-annotation-check-23431c` into this branch (`claude/fidip-grading-rubric-86bb65`; AGENT/MANIFEST conflicts resolved by keeping
  both sides). Changed `.gitignore` so these are tracked, copied from the main checkout as of 2026-11-04 ~22:00:
  `logs/` (104 Slurm stdout files), `log/` (TensorBoard events, 309 files), `output/`, `output_test500/`, `output_test500f/` (prediction JSONs + train/eval logs, no
  checkpoints), `figures/` (92 files incl. 10 mp4), `temp_uploads/` (12 CC photos + `SOURCES.md`), `data/syrip_syn_new/` (522 SMIL renders + COCO json). ~227 MB total, largest file 7 MB.
- **Still not in git (size/licence):** SyRIP images and annotations (`data/syrip`, `data/SyRIP`, raw zips; re-download per MANIFEST §3), the symlink roots built from them
  (`data/syrip_fixed`, `data/syrip500_fixed`, ...; rebuilt by scripts), all `*.pth` checkpoints, `*.npy`, and the licensed SMIL model.
- **Sync safety:** `slurm/sync.sh push` already excluded `output*/`, `log/`, `logs/`, `figures/`, `data/` explicitly; added `temp_uploads/` and `REPORT.md`, so a push still
  cannot overwrite cluster results. `sync.sh pull` now updates tracked files: commit after each pull.
- REPORT.md figure links point at `figures/...`; the duplicate `docs/report_figures/` copies were removed. B4 logs in the snapshot are partial (run in progress).

### 2026-11-04: Merged into main; checkpoint bundle (user: "zip all necessary .pth checkpoints and .npy files to reproduce and sync with git")
- **Merge:** `claude/fidip-grading-rubric-86bb65` (incl. the annotation-audit branch up to the B4 result) merged into `main` in the main checkout. The untracked result
  dirs were moved aside, the merge was run, and the current files were rsynced back. The only differences were the completed B4 files, committed separately.
- **Bundle:** copied H1/S1/S2/M2 `final_state.pth`, `mobile_fidip.pth` and the COCO inits from the cluster (xfer node) to `models/`, then zipped the 11 files a
  reproduction needs into `checkpoints/fidip_checkpoints.zip` (1.70 GB, `unzip -t` OK; per-file sha256 in `checkpoints/SHA256SUMS`, table in `checkpoints/README.md`).
  Left out: `model_best`/`checkpoint.pth`, `feature.npy`, and the licensed SMIL npz/pkl.
- **Git:** every `.pth` is 255 MB, over GitHub's 100 MB file limit, so the zip must go through Git LFS (`.gitattributes` added). git-lfs is not installed on the server
  and installing the release binary was first blocked by the sandbox. After the user allowed it ("Allow you to install"), git-lfs 3.5.1 was installed to `~/.local/bin`
  (sha256 checked against the release's `sha256sums.asc`; the GPG signature was not checked) and `git lfs install` was run. The zip is committed as an LFS pointer
  (oid = the zip's sha256), and the `.gitignore` guard was removed. **Not pushed:** check the GitHub account's LFS storage quota first (1.7 GB).

### 2026-11-04: Portable reproduction scripts (user: "organize codes for simple reproduce", "update reproduce and ref path in report")
- **Did:** added `REPRODUCE.md` and `reproduce/` with one script per step:
  - `00_env.sh`: conda env with the cluster pins, in `reproduce/requirements.txt`.
  - `01_data.sh`: download, sha256 check, unpack, build the corrected roots.
  - `02_weights.sh`: LFS pull, unzip, `sha256sum -c`.
  - `03_score_saved.sh`: all report APs, the bootstrap and Fig. 1 from tracked predictions.
  - `04_eval.sh`, `05_tta.sh`: re-inference into `repro/`, which is git-ignored.
  - `06_train.sh <ID>`: retrain any run. `reproduce/experiments.tsv` lists all 24 training runs, with overrides parsed from each run's train-log Namespace line.

  Also added `tools/README.md` (index of every tool), `--pred/--pred500` options for `tools/eval_subsets.py`, and a README pointer. REPORT §1.1, §3 and §8 were
  rewritten around these scripts. This also fixed §8's old TTA command, whose `--out` must end in `<run>/syrip/ens/x`.
- **Portability bug fixed:** `lib/nms/setup_linux.py` hard-coded `-arch=sm_61`, which CUDA 13's nvcc rejects. The arch now comes from `NMS_CUDA_ARCH`; `00_env.sh`
  sets it from the local GPU, and the default `sm_61` is unchanged for the cluster. `lib/nms/nms.py` now tolerates a missing `gpu_nms`; only `oks_nms` is used.
- **Clean-checkout test** (this worktree, RTX 6000 Ada, CUDA 13 driver):
  - 00 built the env in ~3 min (py 3.12.16, torch 2.3.1+cu121, gpu_nms for sm_89).
  - 01 rebuilt every data root byte-identically (md5 vs the main checkout).
  - 02: all 11 files OK.
  - 03 reprinted every report number.
  - 04 (3.6 min): E0/D1/F4/H1/S1/S2 identical on all 5 subsets, M0 within 0.1.
  - 05 (1.5 min): A0/A2 identical.
  - 06 smoke test: D1 for 1 epoch, 91.0 Test100, 12.65 GiB peak, eval and scoring OK (output deleted).

---

## Dead ends & lessons learned

_(Record approaches that didn't work and why, so they aren't repeated.)_

- **rsync of `data/` to Explorer** broke with `Broken pipe` twice (~1 GB). `| tail` hid the failure
  the first time. Fix: `wget` the public zip on the cluster, check sha256, then unzip in a Slurm job,
  because the login node kills `unzip` for exceeding resource limits.
- `gdown` has no `--remaining-ok` flag (folder download failed in job 11509124).
- **A nested `OUTPUT_DIR` must already exist** (`lib/utils/utils.py:27` uses `mkdir()` without parents). `mkdir -p` the parent before `sbatch`.
- **Val loss below train loss for a whole run is a red flag for label noise**, not just augmentation (E1b: ratio 0.59 with broken labels vs 1.16 with fixed).
- **One-off rsync of a *tracked* file to the cluster is not durable:** any `slurm/sync.sh push` from the main checkout reverts it, even in the middle of a submit
  loop (M1/M2 ran as HRNet). Put experimental cluster code in new files, or commit it to main first. sbatch snapshots the batch script, but not the Python it calls.
- **Negative numbers as CLI values:** `--rots -30,0,30` is parsed as an option by argparse. Always pass list arguments as `--rots=-30,0,30`.
- **Cluster /home quota is 76 GB.** Every eval writes a ~0.57 GB `feature.npy` and every run leaves a ~0.75 GB `checkpoint.pth`; ~50 evals filled the quota and killed B4a/B4b.
  Keep bulky outputs on `/scratch/hou.yuchen`, or clean `feature.npy` after evals, and check `quota -s` before a batch.
  After deleting, the login node can show a removed dir as present for a few seconds (NFS cache): `submit_x.sh` refused once, then worked.
- **Extra GPUs:** hou.yuchen is in group `ImaniLab` → partition `imanilab` (node d3233, 3× A100 80GB, 14-day limit, account `m.imani`) is usually idle.
  A pending job can be moved there with `scontrol update jobid=<id> Partition=imanilab Account=m.imani TresPerNode=gres/gpu:a100:1` (needed when the script pins a GPU type).
- **Copy files through `xfer.discovery.neu.edu`, not the login node** (user rule, 2026-11-04). `slurm/sync.sh` now does this; `pull` took 14.5 s vs minutes before.
- **Don't trust `syrip_for_train.zip` labels blindly**: its synthetic keypoints are wrong and test11.jpg is swapped. Audit images vs annotations
  (size check, overlay a few, score a trusted model against each label set) before training.
- **`wandb.util.generate_id` no longer exists** (wandb 0.31) → E3/E4 died at start. Run `tools/wandb_smoke.py` after any W&B code change.
- **P100 (16 GB) OOMs** at batch 20 for W48 384×288 (V100 peak 12.6 GiB without DA, 17.2 GiB with DA pre-stage). Stay on V100/A100 or lower the batch.
- **courses-gpu QoS: max 2 running jobs and 2 GPUs per user.** Submitting 4 jobs does not parallelise; plan chains accordingly.
- **The login node is slow at `sbatch` (~30–60 s each)**: use `timeout 280` for `ssh explorer 'bash slurm/submit_run.sh ...'`. A local timeout
  does not stop the remote script.
- **An SSH timeout after `sbatch` does not mean the job was not submitted** (E6 got submitted twice). Check `sacct -S now-10minutes` before resubmitting.
- `sync.sh push` once copied `.venv-demo/` to the cluster. It now applies `.gitignore` rules (`--filter=':- .gitignore'`).
- **`pip install` on the login node gets killed** (exit 143). Run installs with `sbatch slurm/install_pkgs.sh <pkgs>`.
- **E1 (11509271) died at import**: the env had `tensorboardx`, but the script imports `torch.utils.tensorboard`. Lesson: smoke-test the
  *training* script's imports in `setup_env.sh`, not just the eval path. Also poll `sacct` after `afterok` chains, and don't log a job as running unchecked.
