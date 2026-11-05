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
  back `logs/`, `log/` (TB), and `*.log|*.json|*.txt` from `output*/`. `env` copies only `WANDB_*` lines. `logs/` and `output_*/` are now git-ignored.
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
  Fig. 1 `docs/report_figures/ablation_overview.png` (`tools/plot_report_ablation.py`); curated figures copied to `docs/report_figures/` (figures/ is git-ignored).
- **Open:** B4a/B4b still training (A100); branch `claude/raw-data-annotation-check-23431c` not merged into main yet, so REPORT.md cites tools that only
  exist there (`eval_subsets.py`, `eval_ensemble_tta.py`, `eval_ens.sh`, `make_syrip_fixed.sh`, `make_syrip500_fixed.py`, `plot_compare.py`).

---

## Dead ends & lessons learned

_(Record approaches that didn't work and why, so they aren't repeated.)_

- **rsync of `data/` to Explorer** broke with `Broken pipe` twice (~1 GB). `| tail` hid the failure
  the first time. Fix: `wget` the public zip on the cluster, check sha256, then unzip in a Slurm job,
  because the login node kills `unzip` for exceeding resource limits.
- `gdown` has no `--remaining-ok` flag (folder download failed in job 11509124).
- **A nested `OUTPUT_DIR` must already exist** (`lib/utils/utils.py:27` uses `mkdir()` without parents). `mkdir -p` the parent before `sbatch`.
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
