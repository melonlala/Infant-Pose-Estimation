# MANIFEST.md — Project Inventory

Inventory of everything produced or used in the FiDIP project. Goal: someone should be able to
reproduce the final reported result from this file alone.

- **Local repo:** `/home/yuchen/courses/EECE7373/hands_on_project`
- **Cluster project root:** `explorer:/home/hou.yuchen/courses/eece7373`
- **Code:** GitHub `git@github.com:melonlala/Infant-Pose-Estimation.git` (private, branch `main`; code only, pushed from the server). `upstream` = ostadabbas
- **Server <-> cluster:** `rsync` via `slurm/sync.sh push|pull|env` (cluster does not use git). See `SKILL.md` § Sync model
- **Status legend:** `planned` · `present` · `missing` · `deprecated`

---

## 1. Code

| Item | Source | Commit | Location | Notes |
|------|--------|--------|----------|-------|
| FiDIP codebase | https://github.com/ostadabbas/Infant-Pose-Estimation (upstream; lacks `gpu_test.py`, `demo.py`, `streamlit_demo.py`, `fidip_simple.yml` referenced by guide; no public fork has them either) | `a76b4f4` + our commits on `main` | server: `/home/yuchen/courses/EECE7373/hands_on_project`; cluster: `/home/hou.yuchen/courses/eece7373/` | `lib/` built with `make` (build products untracked) |
| Local changes | `tools/train_adaptive_model_hrnet.py` (optional W&B), `slurm/*`, `SKILL.md`, `AGENT.md`, `MANIFEST.md`, `.env.example` | `main` | server (git) -> cluster (rsync) | |

## 2. Environment

| Item | Value | Notes |
|------|-------|-------|
| Conda env | `/home/hou.yuchen/courses/eece7373/envs/fidip` (prefix env) | built by `slurm/setup_env.sh`; pip pins mirror `fidip_env.yml` |
| Python / PyTorch / CUDA | 3.12.16 / 2.3.1+cu121 / 12.1 | verified on V100-SXM2-32GB (job 11509124) |
| Cluster modules | `anaconda3/2024.06`, `cuda/12.1.1` | |
| Logging | `tensorboard` 2.21.0 + `wandb` 0.31.0 (added 2026-11-03 after E1 failure, job 11601446) | W&B project `fidip`; key from `.env` |

## 3. Datasets

Downloaded 2026-10-30 from https://coe.northeastern.edu/Research/AClab/SyRIP/ (public, no password).
Local copy under `data/` (git-ignored); cluster copy: `data/SyRIP` via rsync (1500 + 500 images OK); `data/syrip` from `wget` of the zip (sha256 matches) + `slurm/unpack_data.sh`.

| Archive | Size | sha256 | Local path |
|---------|------|--------|------------|
| `syrip_for_train.zip` | 637,172,858 B | `1f267878732fb81776c5c7f153bad45d652cadf5bfd83761971390b64d6cee1b` | `data/raw/` |
| `SyRIP.zip` | 327,912,103 B | `1d13e816375988be9bbdab8bf5be6562a5b9bfa83684d57ccbfec9dd619e1692` | `data/raw/` |

| Split (path) | Images | Annotations | Status | Notes |
|--------------|--------|-------------|--------|-------|
| `data/syrip/annotations/person_keypoints_train_pre_infant.json` | 4057 (`images/train_pre_infant/`) | 4057 | present | used by FiDIP training (pre-infant set) |
| `data/syrip/annotations/person_keypoints_train_infant.json` | 1200 annotated / 1500 files (`images/train_infant/`) | 1200 | present | 200 real + 1000 synthetic |
| `data/syrip/annotations/person_keypoints_validate_infant.json` | 100 (`images/validate_infant/`) | 100 | present | = Test100 in paper; default val set |
| `data/SyRIP/annotations/{200R,1000S,200R_1000S}/` | — | 200 / 1000 / 1200 | present | original SyRIP train subsets |
| `data/SyRIP/annotations/validate500/` | 500 (`data/SyRIP/images/validate_infant/`) | 500 | present | = Test500 in paper; ⊃ Test100; constant `area` → inflated ~98.4 AP; **use `syrip500_fixed`** |
| `data/syrip_fixed/` (server + cluster) | symlinks | 1200 / 100 | present | **use for training/eval from track D on.** `data/syrip` layout with train json = `SyRIP/200R_1000S` (the `syrip_for_train` json has wrong labels on all 1000 synthetic images) and the correct `test11.jpg`. Built by `tools/make_syrip_fixed.sh` |
| `data/syrip500/` (cluster only) | symlinks | — | deprecated | eval root for Test500 (constant `area` → inflated AP; use `syrip500_fixed`): `DATASET.ROOT data/syrip500/` |
| `data/syrip500_fixed/` (server + cluster) | 500 (symlink to `SyRIP/images/validate_infant`) | 500 | present | Test500 json with `area` = bbox w*h (original is constant 2443298). Built by `tools/make_syrip500_fixed.py`. Eval: `DATASET.ROOT data/syrip500_fixed/` |
| `data/SyRIP/annotations/validate100/` | — | 100 | present | = Test100 |
| `data/syrip_clean/` (server + cluster) | symlinks | 1200 / 4057 / 100 | present (unused) | built by `slurm/make_clean_data.sh`; functionally identical to `data/syrip_fixed` (same train ann, correct test11). All reported runs use `syrip_fixed` |
| Expanded SyRIP (optional) | https://coe.northeastern.edu/Research/AClab/Expanded_SyRIP/ | — | — | for extensions |

All annotations are COCO format, 17 keypoints.

**Reported eval subsets (2026-11-04, user decision: drop test11).** Rescored from saved predictions by `tools/eval_subsets.py`; no separate data root.

| Subset | Definition | Role |
|--------|------------|------|
| Test100 | `data/syrip_fixed` val (correct test11) | paper comparison only (paper's 93.6 includes test11) |
| **Test99** | Test100 − test11 | **primary** |
| Test95 | Test99 − test20/42/46/50 (duplicates of annotated train images) | leak-free check |
| Test500f | `data/syrip500_fixed` | secondary (larger) |
| **Test499f** | Test500f − test11 | **primary secondary** |

**Known defects (2026-11-03 audit, see AGENT.md):** `syrip_for_train.zip` has (1) wrong keypoints for all 1000 synthetic training images and
(2) a swapped `validate_infant/test11.jpg` (its GT also has loose hips/right leg/left ear, so test11 is dropped from reported numbers). Test100 also contains 4 images identical to annotated training images (test20/42/46/50), and
`validate500` has a constant `area` on every annotation (found by the 原始数据标注检查 session). The real 200R labels are fine.

## 4. Pretrained models

| Model | Source | Location (cluster) | Status | Notes |
|-------|--------|--------------------|--------|-------|
| `coco/w48_384x288.pth` | DarkPose [TGA_models](https://drive.google.com/drive/folders/14kAA1zXuKODYgrRiQmKnVcipbY7RedVV) | `models/coco/w48_384x288.pth` (+ `w48_384×288.pth` symlink) | present | adult-pose init |
| `hrnet_fidip.pth` | [FiDIP_models](https://drive.google.com/drive/folders/108P-1SnTqaj3xNtjYZ1o7T8z6UvUYuiC) | `models/hrnet_fidip.pth` | present | published baseline |
| `mobile_fidip.pth` | FiDIP_models | `models/mobile_fidip.pth` | present | MobileNet variant |

## 5. Configs

| Config | Purpose | Derived from | Used by |
|--------|---------|--------------|---------|
| `experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml` | baseline FiDIP train/test | upstream | E0, E1 |

## 6. Slurm scripts

| Script | Purpose | Partition / GPU / time | Notes |
|--------|---------|------------------------|-------|
| `slurm/setup_env.sh` | build env, `make` nms, gdown models, GPU check | courses-gpu / 1 GPU / 1:30 | logs → `logs/%x-%j.out` |
| `slurm/unpack_data.sh` | unzip `syrip_for_train.zip`, assert image counts | short / CPU / 0:30 | |
| `slurm/eval.sh <model> [overrides]` | `test_adaptive_model.py`, GT bbox | courses-gpu / 1 GPU / 0:45 | |
| `slurm/train.sh [overrides]` | `train_adaptive_model_hrnet.py`, init DarkPose W48 | courses-gpu / v100-sxm2 / 8:00 | AUTO_RESUME on; resubmit to continue; sources `.env` for W&B |
| `slurm/install_pkgs.sh <pkgs>` | pip install into env + import smoke test | short / CPU / 0:30 | login node kills pip |
| `slurm/submit_run.sh <NAME> <TIME> [overrides]` | train + chained Test500f evals (best/final) | cluster | default root `data/syrip_fixed/` |
| `slurm/make_real_only_data.sh` | build `data/syrip_fixed_200R/` (real-only train set) | either machine | G2 ablation |
| `slurm/make_clean_data.sh` | build `data/syrip_clean/` symlink root | either machine | no copying |
| `tools/make_syrip_fixed.sh` / `tools/make_syrip500_fixed.py` | build `data/syrip_fixed/` / `data/syrip500_fixed/` | either machine, from project root | no copying of images |
| `tools/check_annotations.py` | annotation audit (`--data data --out <dir>`) | server | CPU, ~30 s |
| `slurm/sync.sh push\|pull\|env` | rsync code -> cluster, results -> server, WANDB key -> cluster | server | no `--delete` |

## 7. Runs & checkpoints

| Run ID | Description | Config + overrides | Slurm job(s) | Checkpoint path | Log path | AP |
|--------|-------------|--------------------|--------------|-----------------|----------|----|
| E0 | Pretrained FiDIP eval | baseline, `TEST.USE_GT_BBOX True` | 11509270 | `models/hrnet_fidip.pth` | `logs/fidip-eval-11509270.out` | 92.1 |
| E1 | Reproduce FiDIP training | baseline + `TRAIN.CHECKPOINT models/coco/w48_384x288.pth WORKERS 8` | 11509271 (**failed**: no tensorboard) | — | `logs/fidip-train-11509271.out` | — |
| E0/E1b on Test500 | evals with `DATASET.ROOT data/syrip500/` (symlink root -> `data/SyRIP/annotations/validate500`, `data/SyRIP/images/validate_infant`) | `OUTPUT_DIR output_test500/{E0,E1b,E1b_final}` | 11602004 / 11602005 / 11602006 | — | `logs/fidip-eval500-*-1160200[4-6].out` | 98.4 / 98.5 / 98.3 |
| D0 / D0b | released E0 / E1b `model_best` on fixed Test100 | `DATASET.ROOT data/syrip_fixed/` | 11602447 / 11602449 | — | `output/D0_E0`, `output/D0b_E1b` | 93.6 / 92.6 |
| D1 | E1b config on fixed labels, 20 ep | `train.sh DATASET.ROOT data/syrip_fixed/ OUTPUT_DIR output/D1` | 11602450 | `output/D1/.../{model_best,final_state}.pth` (server copy `models/D1/final_state.pth`) | TB `log/D1/`, W&B 0997f45c | 93.8 best / 93.6 final |
| F1–F4, G1–G3, H1–H3 | single-factor ablations from D1 (see AGENT.md table) | `slurm/submit_run.sh <NAME> <TIME> [overrides]` | 11603301 … 11607666 | `output/<NAME>/.../final_state.pth` | TB `log/<NAME>/`, W&B project `fidip` | see AGENT.md |
| E1b | E1 resubmitted after env fix | same as E1, `WANDB_NAME=E1b-reproduce` | 11601446 → 11601467 | `output/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant/model_best.pth` (ep 6; also `final_state.pth` ep 20, `checkpoint.pth`) | `logs/fidip-train-11601467.out`, `output/.../w48_384x288_adam_lr1e-3_infant_2026-11-03-19-28_train.log`, TB `log/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant_2026-11-03-19-28/` (no W&B) | 91.1 |

## 7b. Tools (server-side, `.venv-demo` = base python + yacs/streamlit/dsntnn/pycocotools)

| Tool | Purpose | Output |
|------|---------|--------|
| `tools/fidip_infer.py` | shared single-image inference (same path as eval) | — |
| `tools/demo.py` | Part 6.1 CLI demo | annotated image + keypoint table |
| `tools/streamlit_demo.py` | Part 6.2 web demo (`.venv-demo/bin/streamlit run tools/streamlit_demo.py`) | browser UI; uploads to `temp_uploads/` |
| `tools/plot_curves.py` | Part 4.5 curves from TB logs | `figures/<run>_curves.png` |
| `tools/visualize_preds.py` | Part 4 GT-vs-pred grids, per-keypoint similarity, pycocotools AP | `figures/viz/` |
| `docs/part4_results_and_fitting.md` | Part 4 viz (E0/D1/F4) + 4.5 fit diagnosis for all runs | — |
| `docs/part6_demos.md` | Part 6.1–6.3 commands, Streamlit test, own-image results | — |
| `figures/viz_final/`, `figures/demo/`, `temp_uploads/` (server only) | final-model viz; demo outputs + Streamlit screenshots; 12 CC infant photos (`SOURCES.md`) | — |
| `models/F4/final_state.pth`, `models/D1/final_state.pth` (server only) | copies of the cluster checkpoints for demos/viz | — |
| `tools/eval_subsets.py` | Test100 / **Test99** / Test95 / Test500f / **Test499f** AP from saved prediction JSONs (`run[:test500f_run]` args) | stdout |
| `tools/plot_compare.py` | Part 4.5 overlay of several runs (train loss, val loss, val AP) | `figures/compare_4_5.png` |
| `tools/eval_test96.py` | (superseded by `eval_subsets.py`) | Test100 vs Test96 AP from final-epoch prediction JSONs in `output/<run>/` | stdout |
| `tools/wandb_smoke.py` | offline check of the W&B code path | — |
| `tools/video_infer.py` | 7.3 video file / folder batch / webcam inference with tracking | `<out>/<name>_pose.mp4`, `_keypoints.jsonl/.csv` |
| `docs/part3_model_notes.md`, `docs/part7_synthetic_and_apps.md` | Part 3 notes; 7.1 status + 7.3 results | — |

Server copies of weights (git-ignored): `models/hrnet_fidip.pth`, `models/E1b/model_best.pth`. Figures are git-ignored (`figures/`).

## 8. Final reported result

**Part 7 target:** beat the paper's 93.6 (DarkPose+FiDIP, Test100), compared on the fixed data only. Because test11 is dropped, the like-for-like
bar on each subset is the released checkpoint (= the paper's model, which reproduces 93.6 on fixed Test100). All numbers are `final_state`.

| | Test100 | **Test99** | Test95 | Test500f | **Test499f** |
|--|--|--|--|--|--|
| Paper (DarkPose+FiDIP) | 93.6 | — | — | — | — |
| Released `hrnet_fidip.pth` (E0/D0), same eval | 93.6 | **92.6** | 92.8 | 90.9 | **90.7** |
| Reproduced baseline **D1** (fixed labels, 20 ep) | 93.6 | 92.7 | 92.9 | 91.4 (`model_best`) | 91.2 |
| **Best improved: F4** (D1 + `TRAIN.END_EPOCH 100`) | **94.0** | **93.0** | **93.2** | **91.7** | **91.4** |
| F4 replicate H1 | 93.9 | 93.0 | 93.2 | 91.6 | 91.4 |
| F4 vs released | +0.4 | **+0.4** | +0.4 | +0.8 | **+0.7** |

| | Value |
|--|-------|
| Checkpoint | cluster `output/F4/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant/final_state.pth` (server copy `models/F4/final_state.pth`; use `final_state`, not `model_best`) |
| Exact reproduce command | `bash slurm/submit_run.sh F4 02:45:00 TRAIN.END_EPOCH 100` (train on `data/syrip_fixed/` + chained Test500f evals), then `.venv-demo/bin/python tools/eval_subsets.py D0_E0:E0 F4` on the server after `slurm/sync.sh pull` |
| Superseded | E1b 91.1 / E0 92.1 (broken labels / swapped test11); Test500 98.x (constant area) |

## 9. Secrets

Secrets (`WANDB_API_KEY`, `GITHUB_TOKEN`) live in `.env` (git-ignored). Never listed here.
