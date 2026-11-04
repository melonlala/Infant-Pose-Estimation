# Part 3 — FiDIP model, domain adaptation and data pipeline (notes)

All numbers below were measured on our checkout (`a76b4f4` + our commits). Shapes and parameter counts come from building
`adaptive_pose_hrnet` with `experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml` (2026-11-03). Code references are
`file:line` in this repo.

## 1. Big picture

```
                       ┌──────────────────────── pose net P (HRNet-W48, 63.6 M params) ─────────────────────────┐
 image crop 3×384×288 ─▶ stem (2× conv s2) ─▶ layer1 ─▶ stage2 ─▶ stage3 ─▶ stage4 ─▶ hi-res branch y0 48×96×72 ─▶ final 1×1 conv ─▶ 17 heatmaps 17×96×72
                       └───────────────────────────────────────────────────────────────────┬──────────────────────┘
                                                                                           │ "feature_output"
                                                                    domain classifier D (0.75 M) ─▶ 1 logit: synthetic?
```

* **Task:** top-down single-person 2D pose. Given an infant bounding box, predict 17 COCO keypoints as heatmaps.
* **FiDIP idea:** fine-tune an adult-pose model (DarkPose HRNet-W48, COCO) on a small infant set of 200 real and 1000 synthetic images.
  A domain classifier on the pose features is trained to tell real from synthetic, and the pose net is trained to *fool* it. The goal is
  features that are invariant to the real/synthetic gap, so the synthetic data helps the real-infant task.

## 2. HRNet-W48 backbone (`lib/models/adaptive_pose_hrnet.py`)

* **Stem:** two 3×3 stride-2 convs (`conv1`, `conv2`) bring the input to 1/4 resolution, followed by `layer1` (4 Bottleneck blocks, 256 ch).
* **Multi-resolution stages:** HRNet keeps a **high-resolution branch the whole way** (1/4 scale, 48 ch) and adds lower-resolution branches
  (96, 192, 384 ch at 1/8, 1/16, 1/32). After each stage the branches **exchange information** (fuse layers: up-/down-sample and sum).
  | stage | branches | modules | blocks/branch | params |
  |---|---|---|---|---|
  | stage2 | 2 (48, 96) | 1 | 4 BASIC | 0.88 M |
  | stage3 | 3 (+192) | 4 | 4 BASIC | 15.34 M |
  | stage4 | 4 (+384) | 3 | 4 BASIC | 45.89 M |
* **Head:** only the highest-resolution output `y_list[0]` (48×96×72) is used: a 1×1 conv to 17 heatmaps (`final_layer`).
  The same tensor is returned as `feature_output` for the domain classifier (`forward`, ~line 461-500).
* **Why HRNet for pose:** keypoints need precise localisation. Keeping high resolution, instead of encoding down and then up-sampling, preserves
  spatial detail, and the repeated fusion gives the hi-res branch semantic context.
* **Initialisation:** `MODEL.PRETRAINED ''` → normal init, then `slurm/train.sh` loads the DarkPose COCO weights via `TRAIN.CHECKPOINT
  models/coco/w48_384x288.pth` (the stock yaml would otherwise train from scratch).
* **Frozen layers:** `tools/train_adaptive_model_hrnet.py` sets `requires_grad=False` on the first 109 parameter tensors (`idx <= 108`).
  That is the stem, layer1, transition1, stage2, transition2 **and the first conv of stage3** (it stops mid-block, at
  `stage3.0.branches.0.0.conv1.weight`). It covers only **1.72 M params (2.7 %)**. The frozen BatchNorm layers still update their running
  statistics, because the model is in `train()` mode. The paper says it freezes "Res1-3" of a ResNet; this HRNet mapping is the authors' code choice.

## 3. Domain classifier D (`DomainClassifier`, line ~276)

* Input: `feature_output` 48×96×72.
  Conv3×3 (48→32, no pad) → BN → ReLU → MaxPool2 → Conv3×3 (32→8) → BN → ReLU → MaxPool2 → **8×22×16** → FC 2816→256 → BN → ReLU → FC 256→64 → BN →
  ReLU → FC 64→1 (logit). 0.75 M params. Loss `BCEWithLogitsLoss`; label = `is_synthetic`.

## 4. Training procedure (`train_adaptive`, `lib/core/function.py:104`)

1. **Initialisation session (`TRAIN.PRE_EPOCH` epochs):** train only D on `train_pre_infant` (4057 adult real/synthetic images;
   per the paper, COCO val + SURREAL). P features are `detach()`ed. The infant yaml sets `PRE_EPOCH 0`; the paper uses 1 epoch.
2. **Formal session, every iteration:**
   * **Stage I:** `loss_d = BCE(D(feat.detach()), y)` → update **D only**.
   * **Stage II:** `loss_p = MSE(heatmaps, target) − λ·BCE(D(feat), y)` → update **P only** (D's weights are not stepped here).
     Minimising `−λ·BCE` = maximising D's error: an adversarial objective written directly as a negative term, not a gradient-reversal layer.
     It is equivalent to GRL with coefficient λ for P's gradient.
   * `TRAIN.LAMBDA` = λ. The yaml ships **λ = 0** (DA off); the paper uses 0.0005. `GAMMA1/GAMMA2` exist in the config but are unused.
3. Separate Adam optimisers for P and D (same LR), `MultiStepLR` at `LR_STEP` (yaml: 40, 200; i.e. no decay within 20 epochs).
4. After each epoch: validate on Test100 → AP; keep `model_best.pth` (highest Test100 AP), `checkpoint.pth`, `final_state.pth`.

## 5. Data pipeline

* **Datasets** (`lib/dataset/syrip.py`, COCO json): `TRAIN_SET train_infant` (200 real + 1000 synthetic, use the fixed labels
  `200R_1000S`), `TRAIN_PRE_SET train_pre_infant`, `TEST_SET validate_infant` (Test100). The image path is built from the image id
  (`train%05d.jpg`, `test%d.jpg`), not from `file_name`.
* **Box → crop:** box `(x,y,w,h)` → center + scale, padded to the 288:384 aspect ratio, ×1.25 margin, `pixel_std` 200 (`_xywh2cs`).
  An affine warp then produces the 384×288 crop.
* **Augmentation (train only, `JointsDataset.__getitem__`):**
  * half-body crop with p = 0.3 (needs > 8 visible joints);
  * scale ×N(1, 0.35) clipped to [0.65, 1.35];
  * rotation N(0, 45°) clipped to ±90°, applied with p = 0.6;
  * horizontal flip with p = 0.5 (with left/right joint swap);
  * ImageNet mean/std normalisation, RGB.
* **Targets:** one 2D Gaussian (σ = 3 px on the 96×72 heatmap) per joint. `target_weight` = visibility, and joints that fall outside
  the heatmap get weight 0.
* **Pose loss:** `JointsMSELoss` = Σ_joints 0.5·MSE(pred·w, gt·w) (`lib/core/loss.py:15`).
* **Real/synthetic balancing:** `BalancedBatchSampler` (`lib/utils/sampler.py`) oversamples the 200 real images to 1000 (random picks
  fixed **once** at construction) and then alternates real, synthetic, real, … So each batch of 20 is 10 real + 10 synthetic, and each epoch
  is 2000 samples = 100 iterations; every real image is seen about 5× per epoch. **Caveat found 2026-11-03:** the sampler never shuffles, so
  every epoch visits the samples in the same order with the same batch composition (verified by iterating it twice). This is a candidate fix in Part 7.

## 6. Inference and evaluation

* **Flip test:** average the heatmaps of the image and of its mirror (flipped back with joint swap).
* **DarkPose post-processing** (`core/inference.py:get_final_preds`): Gaussian-blur each heatmap (`BLUR_KERNEL 11`), take the log, then a
  2nd-order **Taylor expansion** around the arg-max gives sub-pixel coordinates (distribution-aware decoding). The coordinates are mapped back
  to the image by the inverse affine transform.
* **Metric:** COCO keypoint **OKS** = mean over visible joints of exp(−d² / (2·s²·k²)), with s² = object area and k = per-joint
  sigma (`fidip_infer.SIGMAS`). AP is averaged over OKS thresholds 0.50:0.05:0.95. AP50/AP75 are single thresholds, and AR is the
  matching recall. With GT boxes and one infant per image, AP is mostly about localisation precision; detection does not play a role.
* Our reference numbers (fixed data): released FiDIP 93.6 Test100 (= paper); our D1 retrain 93.8 Test100 / 91.4 Test500 (fixed area).

## 7. Things that surprised us (useful for the report)

1. The shipped `syrip_for_train.zip` had wrong synthetic training labels and a swapped `test11.jpg`. Fixing them gives +1.2 AP and closes the
   gap to the paper completely (AGENT.md, data audit).
2. Domain adaptation (λ = 0.0005) makes the features measurably more domain-invariant (D accuracy 0.6 vs 0.8), but it does not improve AP on
   correct labels (F2).
3. Only 2.7 % of the weights are frozen, not "most of the early network". The domain-classifier tap is the final hi-res feature, the deepest possible point.
4. The balanced sampler does not shuffle.
