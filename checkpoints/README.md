# Checkpoints and binary artifacts

`fidip_checkpoints.zip` (1,703,029,973 B, sha256 `9ab25d537b07b50f30c03b8cbf88098d7d15fbb2235b1cd5c83ca0bdc14a7c66`) holds every weight file needed to
reproduce the numbers in `REPORT.md`. It is stored with **Git LFS** (`.gitattributes`). After cloning:

```bash
git lfs install && git lfs pull                  # fetch the zip
unzip checkpoints/fidip_checkpoints.zip -d .     # extracts to models/... and figures/... at the repo root
sha256sum -c checkpoints/SHA256SUMS              # verify every file
```

| File in the zip | What | Produced by | Used for |
|---|---|---|---|
| `models/F4/final_state.pth` | **final single model** (D1 config + 100 ep, last epoch) | Slurm 11603343 | REPORT §5; A2 = F4 + TTA |
| `models/H1/final_state.pth` | F4 replicate | 11607648 | replicate evidence; A1/A3 ensemble member |
| `models/S1/final_state.pth`, `models/S2/final_state.pth` | 100 ep + sampler shuffle pair | 11617384, 11617427 | A1/A3 ensemble members |
| `models/D1/final_state.pth` | reproduction baseline (corrected labels, 20 ep) | 11602450 | REPORT §4, Part 4 visualisation |
| `models/M2/final_state.pth` | best MobileNetV2 (224², 100 ep) | 11618140 | backbone comparison |
| `models/hrnet_fidip.pth` | released FiDIP HRNet-W48 (third party, [FiDIP_models](https://drive.google.com/drive/folders/108P-1SnTqaj3xNtjYZ1o7T8z6UvUYuiC)) | authors | published baseline (D0) |
| `models/mobile_fidip.pth` | released FiDIP MobileNet (third party, same folder) | authors | M0 |
| `models/coco/w48_384x288.pth` | DarkPose COCO HRNet-W48 init (third party, [TGA_models](https://drive.google.com/drive/folders/14kAA1zXuKODYgrRiQmKnVcipbY7RedVV)) | DarkPose authors | training init for every HRNet run (the configs reference `w48_384×288.pth`; symlink it) |
| `models/coco/posemobile.pth` | Pose-MobileNet COCO init (third party) | authors | MobileNet training init |
| `figures/viz/synthetic_oks_check.npy` | per-image OKS of the released model vs both synthetic label sets (data audit, AGENT.md 2026-11-03) | `tools/visualize_preds.py` audit | evidence for the label defect |

Not included, on purpose:
- `model_best.pth` files: they were selected on the test set and are not used for any reported number.
- `checkpoint.pth` resume files (optimizer state).
- `feature.npy` eval dumps: unused, and deleted on the cluster.
- The **licensed SMIL model** (`syn_generation/**/smil_*.npz`, `.pkl`): Fraunhofer IOSB licence, register and download it yourself.

The cluster keeps the originals under `/home/hou.yuchen/courses/eece7373/output/<run>/syrip/adaptive_pose_*/<cfg>/final_state.pth` and `models/`.
