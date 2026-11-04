# Part 4: prediction visualisation and over/underfitting analysis (final model)

All numbers come from the fixed data: `data/syrip_fixed` (Test100 with the corrected test11) and `data/syrip500_fixed` (Test500 with real areas).
The scripts run on the server with `.venv-demo/bin/python`.

## 4. Prediction visualisation: released (E0) vs reproduced (D1) vs final (F4)

```bash
CUDA_VISIBLE_DEVICES=0 .venv-demo/bin/python tools/visualize_preds.py \
  --models E0=models/hrnet_fidip.pth D1=models/D1/final_state.pth F4=models/F4/final_state.pth \
  --ann data/syrip_fixed/annotations/person_keypoints_validate_infant.json \
  --images data/syrip_fixed/images/validate_infant --out figures/viz_final
```

The checkpoints are last-epoch copies from the cluster (`output/<run>/.../final_state.pth`).

| model | Test100 AP (this tool) | AP50 = AP75 | AR | mean per-image OKS | images with OKS < 0.9 |
|---|---|---|---|---|---|
| E0 (released `hrnet_fidip.pth`) | 0.933 | 0.985 | 0.944 | 0.954 | 10 |
| D1 (user retrain, 20 ep) | 0.937 | 0.985 | 0.948 | 0.957 | 8 |
| **F4 (100 ep, final model)** | **0.940** | 0.984 | **0.951** | **0.958** | 9 |

These figures come from the tool's own inference, which crops to the GT box and uses flip test. The official numbers come from `tools/test.py` and the
train-loop eval, and are 0.936 / 0.936 / 0.940; the small differences come from box handling. The ranking is the same.

**Figures:**
- `figures/viz_final/{E0,D1,F4}_worst.png`: the 8 worst images by OKS, GT next to the prediction.
- `figures/viz_final/{E0,D1,F4}_best.png`: the 8 best.
- `figures/viz_final/per_keypoint.png`: mean OKS similarity per joint.
- `figures/viz_final/*_per_image_oks.json`: per-image scores.

**What the figures show:**
- **The same two images are worst for every model:** test33 (side-lying infant, limbs overlapping) and test41 (sitting, seen from behind, arms hidden
  by a toy). F4 raises both a little (0.75/0.76 → 0.77/0.79), but they stay the hardest. Both are **self-occlusion** cases, which none of the
  training changes address.
- **Where F4 gains (per joint):**
  - Hips: left 0.945 → 0.964, right 0.936 → 0.969 vs E0.
  - Left ankle: 0.940 → 0.957.
  - Left elbow: 0.953 → 0.963.
  These are the joints that are ambiguous on a chubby infant body and are hidden under clothes or a blanket. Training longer on the 200 real images
  mostly sharpens them.
- **Where F4 loses:** eyes. Left 0.981 → 0.974, right 0.970 → 0.963. Face joints are already near saturation, so this is within noise.
- **Ears stay the weakest joint** (0.90–0.93) for every model, the same as on COCO.
- **AP50 = AP75 = 0.98 for every model.** All the remaining error is fine localisation on a few hard images, not gross failures.

## 4.5 Over/underfitting: curves and diagnosis for every run

Make the curves with `tools/plot_curves.py log/<run>/syrip/adaptive_pose_hrnet/<timestamp>/ --name <run>`. They are written to
`figures/<run>_curves.png` and have four panels: pose loss, heatmap accuracy, AP/AR, and the domain classifier.

Columns are last-epoch values from the TensorBoard logs:

| run | epochs | train loss | val loss | val/train ratio | train acc | val acc | val AP | reading |
|---|---|---|---|---|---|---|---|---|
| D1 | 20 | 2.69e-4 | 3.12e-4 | 1.16 | 0.918 | 0.933 | 0.936 | small gap, healthy |
| F1 | 20 | 2.51e-4 | 3.13e-4 | 1.25 | 0.932 | 0.930 | 0.935 | same as D1 (replicate) |
| G1 / H3 (shuffle) | 20 | 2.7–3.0e-4 | 3.05e-4 | 1.03–1.12 | 0.92–0.93 | 0.93 | 0.940 / 0.939 | smallest gap |
| G3 (LR 5e-5) | 20 | 3.12e-4 | 3.15e-4 | 1.01 | 0.899 | 0.924 | 0.933 | **slight underfit**: train ≈ val, both higher, AP lower |
| F3 (LR 1e-3) | 20 | 2.12e-4 | 3.31e-4 | 1.56 | 0.947 | 0.920 | 0.929 | **overfit**: lowest train loss, highest val loss, lowest AP |
| F4 / H1 (100 ep) | 100 | 1.9–2.0e-4 | 3.02–3.07e-4 | 1.55–1.58 | 0.95 | 0.93 | 0.940 / 0.939 | gap widens, but val does **not** get worse |
| G2 (real-only, 200 ep) | 200 | 1.33e-4 | 3.16e-4 | 2.38 | 0.981 | 0.933 | 0.939 | largest gap (memorises 200 images) without a val penalty |
| F2 (DA on) | 20 | MSE − λ·BCE | 3.21e-4 | n/a | 0.907 | 0.920 | 0.935 | train loss includes −λ·BCE, so the ratio is meaningless |

**Diagnosis:**
1. **Nothing underfits at LR 1e-4.** Val AP is above 0.90 from epoch 1 because the COCO-pretrained HRNet-W48 already transfers well. Only LR 5e-5
   (G3) shows mild underfitting: train and val losses are equal and both are higher than at 1e-4.
2. **Longer training widens the train/val gap** (ratio 1.16 at 20 ep vs about 1.6 at 100 ep, 2.4 for real-only at 200 ep). **Val loss and val AP do
   not degrade, though:**
   - In F4/H1, val loss stays flat at about 3.0e-4 after epoch 20, and AP stays at 0.939–0.941.
   - On the held-out Test500 the 100-epoch final checkpoints are the best we have (91.7 / 91.6).
   This is a benign plateau, not harmful overfitting. The LR drop at epoch 40 (×0.1) helps hold it there.
3. **The harmful case is a high LR (F3, the paper's 1e-3).** Train loss drops fastest while val loss rises (ratio 1.56 after only 20 epochs), and
   Test500 drops to 89.9 / 90.6. Use 1e-4 to fine-tune from COCO weights.
4. **Model selection matters more than the curves suggest.** The epoch with the best Test100 AP is not the best on Test500:
   - F4 `model_best`: 91.0 on Test500, vs `final_state` 91.7.
   - H1 `model_best`: 91.4 on Test500, vs `final_state` 91.6.

   Picking the peak of a noisy curve measured on 100 images overfits those 100 images, so we report final checkpoints.
5. **Domain classifier panel (DA off, λ = 0):** the classifier still trains in stage I and reaches 0.8–0.9 accuracy by the end of a run. Real and
   synthetic features stay separable, because nothing pushes the backbone to hide the domain. With DA on (F2) it stays at 0.6–0.75, closer to
   chance, so the adversarial term works as intended. It still gives no accuracy gain (Test500 91.0 vs 91.2).
