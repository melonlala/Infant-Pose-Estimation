# Reproducing the results in REPORT.md

Every step is one script in `reproduce/`. Run the scripts from any directory: each one `cd`s to the repo root. They need no Slurm and no cluster
paths. Fresh outputs go to `repro/` (override with `OUT=`), so the tracked results in `output*/` are never overwritten.

| Step | Command | Needs | Time* | Result |
|---|---|---|---|---|
| 0 | `bash reproduce/00_env.sh` | conda, NVIDIA driver | ~3 min | conda env `envs/fidip` (Python 3.12, PyTorch 2.3.1 + CUDA 12.1), `lib/nms` compiled for your GPU |
| 1 | `bash reproduce/01_data.sh` | internet (1 GB) | ~1 min + download | SyRIP in `data/` (sha256-checked) plus the corrected roots `data/syrip_fixed`, `data/syrip500_fixed`, `data/syrip_fixed_200R`, `data/syrip_fixed_plus522` |
| 2 | `bash reproduce/02_weights.sh` | Git LFS | ~20 s + 1.7 GB LFS download | `checkpoints/fidip_checkpoints.zip` unpacked to `models/`; every file sha256-verified |
| 3 | `bash reproduce/03_score_saved.sh` | steps 0–1 (no GPU) | ~3 min | **every AP in REPORT §4–6**, the bootstrap CIs (§5.1) and Fig. 1, recomputed from the tracked predictions |
| 4 | `bash reproduce/04_eval.sh [RUN…]` | steps 0–2, 1 GPU | ~4 min | re-inference of the released model and our checkpoints (E0 D1 F4 H1 S1 S2 M0 M2) on Test100 and Test500f, side by side with the reported values |
| 5 | `bash reproduce/05_tta.sh [A0 A1 A2 A3]` | steps 0–2, 1 GPU | ~1.5 min (A0 + A2) | the TTA / ensemble systems of REPORT §5 |
| 6 | `bash reproduce/06_train.sh <ID>` | steps 0–2, 1 GPU ≥ 16 GB | ~40 s/epoch: 20 ep ≈ 15 min, 100 ep ≈ 1.2 h | retrains any run of REPORT §6 (IDs and exact overrides: `reproduce/experiments.tsv`), then evaluates and scores its final checkpoint |

\* Measured on one RTX 6000 Ada (2026-11-04, clean checkout). On that machine step 4 reproduced every reported HRNet AP exactly and MobileNet to within 0.1; step 5 reproduced A0/A2 exactly.

**Fastest check (no GPU, ~5 min):** steps 0, 1, 3. Step 3 needs only a Python with `pycocotools`, `numpy` and `matplotlib`, so
`PY=/usr/bin/python3 bash reproduce/03_score_saved.sh` works without step 0 if those packages are installed.

**Final model:** `bash reproduce/04_eval.sh E0 F4` (released vs ours) and `bash reproduce/05_tta.sh A0 A2` (F4 alone vs F4 + TTA).
Retraining it from the COCO init: `bash reproduce/06_train.sh F4`.

## What to expect

- Step 3 prints exactly the numbers in REPORT §4–6, because it scores the same prediction files with the same code.
- Steps 4–5 re-run inference. They matched exactly in our clean-checkout test (MobileNet to within 0.1). Other GPUs or cuDNN versions may differ by ~0.1.
- Step 6 is training. Two runs of the same config differ by 0.1–0.2 AP (REPORT §2.2), so compare with the replicate pairs in REPORT §6.1, not a single number.

## Evaluation protocol used by every step

GT person boxes, flip test, COCO keypoint OKS AP, last-epoch checkpoint (`final_state.pth`). Five test subsets are scored from the same predictions
(REPORT §3): Test100 (paper protocol, corrected `test11.jpg`), Test99 (primary), Test95, Test500f and Test499f (decision metric).

## Repository layout

| Path | Contents |
|---|---|
| `reproduce/` | the step scripts above, `experiments.tsv` (every training run and its overrides), `requirements.txt` |
| `tools/` | training, evaluation, analysis and demo scripts; index in `tools/README.md` |
| `lib/`, `experiments/` | model, dataset and training code; configs (upstream FiDIP, configs unchanged) |
| `slurm/` | the Explorer-cluster versions of the same steps, which produced every reported number (`slurm/submit_run.sh <ID> <TIME> <overrides>`; see `SKILL.md`) |
| `syn_generation/` | SMIL fitting and rendering pipeline (Part 7.1; the licensed SMIL model is not included) |
| `output/`, `output_test500f/`, `output_test500/` | tracked predictions and train/eval logs of every reported run |
| `logs/`, `log/` | Slurm stdout and TensorBoard events of every run |
| `figures/` | every figure in the report (`figures/report/` = Fig. 1) |
| `checkpoints/` | the weight bundle (Git LFS) with `README.md` and `SHA256SUMS` |
| `data/syrip_syn_new/` | our 522 SMIL renders; the rest of `data/` is created by step 1 |
| `docs/` | Part 3/4/6/7 write-ups |
| `REPORT.md`, `AGENT.md`, `MANIFEST.md`, `SKILL.md` | report, lab notebook, artifact inventory, cluster operating guide |

## Troubleshooting

- **`lib/nms` build:** `00_env.sh` compiles the CUDA NMS for the local GPU (`NMS_CUDA_ARCH`, e.g. `sm_89`). If `nvcc` is missing or too new, only the
  CPU NMS is built. That is enough, because the keypoint pipeline uses `oks_nms` (CPU) with GT boxes.
- **Another env:** `PY=/path/to/python bash reproduce/04_eval.sh` uses any Python that has the packages in `reproduce/requirements.txt`.
- **Out of GPU memory in training:** add `TRAIN.BATCH_SIZE_PER_GPU 10` to step 6. This changes the run, so the result is not directly comparable.
- **`repro/output/<ID>` exists:** training resumes automatically (AUTO_RESUME), so `06_train.sh` refuses to reuse a directory. Delete it or set `OUT=`.
