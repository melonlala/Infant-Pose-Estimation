# MANIFEST.md — Project Inventory

Inventory of everything produced or used in the FiDIP project. Goal: someone should be able to
reproduce the final reported result from this file alone.

- **Local repo:** `/home/yuchen/courses/EECE7373/hands_on_project`
- **Cluster project root:** `explorer:/home/hou.yuchen/courses/eece7373`
- **Code sync:** GitHub `git@github.com:melonlala/Infant-Pose-Estimation.git` (private, branch `main`). `upstream` = ostadabbas. See `SKILL.md` § Sync model
- **Status legend:** `planned` · `present` · `missing` · `deprecated`

---

## 1. Code

| Item | Source | Commit | Location | Notes |
|------|--------|--------|----------|-------|
| FiDIP codebase | https://github.com/ostadabbas/Infant-Pose-Estimation (upstream; lacks `gpu_test.py`, `demo.py`, `streamlit_demo.py`, `fidip_simple.yml` referenced by guide; no public fork has them either) | `a76b4f4` + our commits on `main` | server: `/home/yuchen/courses/EECE7373/hands_on_project`; cluster: `/home/hou.yuchen/courses/eece7373/` | `lib/` built with `make` (build products untracked) |
| Local changes | `tools/train_adaptive_model_hrnet.py` (optional W&B), `slurm/*`, `SKILL.md`, `AGENT.md`, `MANIFEST.md`, `.env.example` | `main` | both | |

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
| `data/SyRIP/annotations/validate500/` | 500 (`data/SyRIP/images/validate_infant/`) | 500 | present | = Test500 in paper |
| `data/SyRIP/annotations/validate100/` | — | 100 | present | = Test100 |
| Expanded SyRIP (optional) | https://coe.northeastern.edu/Research/AClab/Expanded_SyRIP/ | — | — | for extensions |

All annotations are COCO format, 17 keypoints.

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
| `slurm/push_results.sh "<msg>"` | commit + push `logs/` from the cluster | login node | results sync to server |

## 7. Runs & checkpoints

| Run ID | Description | Config + overrides | Slurm job(s) | Checkpoint path | Log path | AP |
|--------|-------------|--------------------|--------------|-----------------|----------|----|
| E0 | Pretrained FiDIP eval | baseline, `TEST.USE_GT_BBOX True` | 11509270 | `models/hrnet_fidip.pth` | `logs/fidip-eval-11509270.out` | 92.1 |
| E1 | Reproduce FiDIP training | baseline + `TRAIN.CHECKPOINT models/coco/w48_384x288.pth WORKERS 8` | 11509271 (**failed**: no tensorboard) | — | `logs/fidip-train-11509271.out` | — |
| E1b | E1 resubmitted after env fix | same as E1, `WANDB_NAME=E1b-reproduce` | 11601446 → 11601467 | `output/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant/model_best.pth` | `logs/fidip-train-<id>.out`, `log/syrip/adaptive_pose_hrnet/`, W&B `fidip` | — |

## 8. Final reported result

| | Value |
|--|-------|
| Published FiDIP baseline (paper) | DarkPose+FiDIP, Test100: AP 93.6 / AP50 98.5 / AP75 98.5 / AR 94.6 |
| Reproduced baseline run | _TBD_ |
| Best improved run | _TBD_ |
| Checkpoint | _TBD_ |
| Exact reproduce command | _TBD_ |

## 9. Secrets

Secrets (`WANDB_API_KEY`, `GITHUB_TOKEN`) live in `.env` (git-ignored). Never listed here.
