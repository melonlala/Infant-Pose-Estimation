# Part 7.1 (synthetic data generation) and 7.3 (applications)

## 7.1 Synthetic infant data generation — status: runs on the server (2026-11-04); trained on in SY1/SY2: no effect

**What the pipeline does** (`syn_generation/`):
1. **Fit SMIL to 2D poses:** `smplifyx/main.py` (SMPLify-X adapted to the infant SMIL model). It optimises SMIL shape (20 betas) and pose (69 = 23 joints × 3)
   plus the camera so that the projected SMIL joints match the 2D keypoints of a real infant image. Output: `<out>/results/<id>/000.pkl`
   (`betas, body_pose, global_orient, camera_*`) and the posed mesh `<out>/meshes/<id>/000.obj`.
2. **Render:** pose SMIL with those parameters (plus jitter), apply a random texture (12 MINI-RGBD infant textures in `render/textures/`), light it and
   composite it on a background. The 2D labels are the SMIL joints projected with the render camera, so they are exact by construction
   (except for the hip definition, see below).

**Set-up (server, done once).** The user downloaded SMIL (licence accepted). `syn_generation/models/smil.zip` was unzipped to `syn_generation/smil/`
and the files copied to `models/smil_web.pkl`, `render/smil_web.pkl` and `priors/smil_pose_prior.pkl` (all git-ignored: `syn_generation/.gitignore`).
Separate env so `.venv-demo` is untouched: `python3 -m venv --system-site-packages .venv-smil` + `pip install configargparse trimesh pyrender
PyOpenGL pycocotools smplx` and `pip install -U "PyOpenGL>=3.1.7"` (pyrender pins 3.1.0, which has no working EGL). torch / numpy / opencv come from the base python.

**Porting problems and fixes.**
| problem | fix |
|---|---|
| `smil_web.pkl` / `smil_pose_prior.pkl` are Python-2 pickles with chumpy objects (chumpy does not run on py3.13 / numpy 2) | `tools/smil_pickle.py`: restricted unpickler (only numpy/scipy classes allowed, chumpy `Ch`/`Select` and `Mahalanobis` replaced by stubs). `tools/convert_smil.py` writes plain `models/smil_web_np.npz` (+ copy in `render/`) and `priors/smil_pose_prior_np.npz`; check: `J_regressor @ v_template == J` (max err 1e-16). `smil_model.py` loads the npz when present; `prior.py` reads the prior mean from the npz |
| `gmm_08.pkl` (the adult SMPLify GMM pose prior the config used) does not come with SMIL | SMIL's own Gaussian pose prior converted to a 1-component GMM `priors/gmm_01.pkl`; `fit_smil.yaml`: `num_gaussians: 1` |
| float32 fit diverges (NaN in the LBFGS line search; the SMIL prior is very stiff, condition number 5e6) | `fit_smil.yaml`: `float_dtype: float64` |
| torch 2: `Tensor.add_(scalar, tensor)` removed | `optimizers/lbfgs_ls.py`: `add_(t, alpha=s)` |
| `human_body_prior` (VPoser) imported at module level | import moved inside `if use_vposer` |
| OpenDR 0.78 renderer + `/home/faye/...` paths | new pyrender (EGL, headless) renderer `tools/smil_render.py`; all folders are CLI arguments. Geometry stays in the fitter's camera frame (OpenCV convention) and is converted to OpenGL inside, so projected labels and rendered pixels use the same K |
| interactive viewer (`--visualize True`) needs a display | run with `--visualize False`; overlays are made by `tools/fit_overlay.py` |
`render/image_generation.py` (OpenDR) is left as the authors' reference and is no longer used.

**Commands** (from `syn_generation/`; `EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1` for the rendering steps):
```
../.venv-smil/bin/python tools/convert_smil.py                                   # once: chumpy pickles -> npz
# (a) the 2 shipped examples (GPU, 56 s/image)
CUDA_VISIBLE_DEVICES=1 ../.venv-smil/bin/python smplifyx/main.py --config cfg_files/fit_smil.yaml \
    --data_folder data --output_folder output_examples --model_folder models --visualize False
../.venv-smil/bin/python tools/fit_overlay.py --data_folder data --fit_folder output_examples --out ../figures/syn/fit_examples.png --cols 1
# (b) the 200 real training images with their GT keypoints
../.venv-smil/bin/python tools/prepare_fit_data.py --out data_real --num 200       # COCO-17 -> OpenPose COCO-19, images symlinked
../.venv-smil/bin/python tools/fit_parallel.py --data_folder data_real --output_folder output_real --jobs 20 --device cpu
../.venv-smil/bin/python tools/fit_overlay.py --data_folder data_real --fit_folder output_real --out ../figures/syn/fit_real_overlays.jpg --cols 4
../.venv-smil/bin/python tools/fit_hip_offset.py --fit_folder output_real          # -> models/coco_hip_offset.json
# (c) render + label checks
../.venv-smil/bin/python tools/make_backgrounds.py --out render/bg_pool
../.venv-smil/bin/python tools/render_synthetic.py --fit_folders output_real --variants 3 \
    --out_root ../data/syrip_syn_new --image_set train_infant --start_id 20001
../.venv-smil/bin/python tools/check_syn_labels.py --root ../data/syrip_syn_new --grid ../figures/syn/syn_label_check.jpg --fit_folder output_real
../.venv-smil/bin/python tools/pipeline_figure.py --fit_folder output_real --data_folder data_real \
    --syn_root ../data/syrip_syn_new --out ../figures/syn/pipeline_examples.jpg
```
COCO-17 → OpenPose-19 (the fitter's format, `utils.smpl_to_openpose('smil', 'coco19')`): neck = mean of the shoulders, mid-hip = mean of the hips,
confidence = COCO visibility (0/1/2, the convention of the shipped examples). Neck and both hips are ignored by the fitter (`joints_to_ign: 1, 9, 12`);
mid-hip is used.

**Speed.** One fit = 4 stages × ≤30 LBFGS iterations, ~7 000 model evaluations. float64, batch 1: 56 s on the RTX 6000 Ada, 57 s on one CPU thread.
The GPU does not parallelise: 10 jobs on GPU 1 took 600 s each (no gain). CPU jobs do: `--jobs 20 --device cpu` fitted 180 images in 580 s
(about 3 s/image overall). Rendering: 107 ms/image (pyrender EGL on GPU 1, plus the numpy SMIL forward pass and compositing).

**Fit results.**
* Examples 46 / 247: mean reprojection error 0.6 % / 1.8 % of the keypoint-box diagonal. The camera translation matches the authors' shipped fit
  (`output/results/46/000.pkl`: t = (0.001, −0.155, 0.674) vs ours (0.001, −0.155, 0.678)). The pose differs a little because of the infant prior
  (vs. their adult GMM). Figure: `figures/syn/fit_examples.png`.
* 200 real images: mean error 3.0 %, median 2.1 %. 142 fits ≤ 3 %, 174 ≤ 5 %, 8 > 10 %. The bad ones are mostly strongly self-occluded or prone poses
  (mesh flipped or pushed out of view). Figures: `figures/syn/fit_real_overlays.jpg` (first 24), `figures/syn/fit_real_worst.jpg` (worst 8).
  Per-image numbers: `syn_generation/output_real/fit_quality.json`. Only fits ≤ 5 % are rendered: visual inspection showed that several 5–8 % fits
  are already wrong in 3D (e.g. train00085, head and legs swapped).
* **Hip-definition mismatch (found by the label check, fixed).** SMIL hip *joints* sit inside the pelvis. On the 174 good fits the projected SMIL
  hip width is only **0.39×** the annotated hip width (median), and hip error is 6.0 % vs ≤ 2 % for other joints. With the raw joints the synthetic
  hips would follow a different definition than the real labels. `tools/fit_hip_offset.py` fits one mirrored 3D offset in the pelvis frame
  (2.8 cm lateral, 0.2 cm up, 0.4 cm forward; the two cross-validation folds agree to 1 mm). Cross-validated hip error drops 6.0 % → 3.8 % and the width
  ratio goes to 0.89. The renderer uses this calibrated hip (`--raw_hips` turns it off). (For reference, the existing 1000S labels have
  hip/shoulder width 0.64 vs 0.75 for real images, so they are also narrower.)

**Rendering choices** (`tools/render_synthetic.py`, 3 variants per fit, seed 0):
* pose jitter: N(0, min(0.3·prior std, 0.08 rad)) per joint; shape jitter: betas + N(0, 0.25)
* view jitter: roll ±40°, pitch/yaw ±20° about the body centre; focal length 500–1000 px; body covers 45–85 % of the 640×480 frame; random position.
  The whole body must stay inside the image (no truncation)
* texture: one of 12 MINI-RGBD textures with mild HSV jitter; 1–2 directional lights (random direction, intensity, slight tint) + ambient
* background: `render/bg_pool/` (`tools/make_backgrounds.py`). 300 procedural "bedding / mat / floor" textures (stripes, checks, plaid, dots,
  wood grain, noise, with lighting gradients) plus 300 crops of the 200 real *training* images, taken only from regions outside the infant
  (keypoint box ×1.6 + 20 % of the image) and Gaussian-blurred σ = 2–5 px. Then a soft contact shadow, feathered alpha, noise, blur, and JPEG q 70–95
* visibility: face points v=1 if > 1.5 cm behind the visible surface (depth buffer). Body joints v=1 if the body part rendered at that pixel
  (part-label pass, dominant skinning joint) is not adjacent to the joint. Otherwise v=2

**New dataset:** `data/syrip_syn_new/` (laid out like `data/syrip_fixed`):
`images/train_infant/train20001.jpg … train20522.jpg` (522 images, 640×480, 174 source poses × 3) and
`annotations/person_keypoints_train_infant.json`. The image/annotation fields are identical to the SyRIP json (`is_synthetic: true`,
`image_source: "smil_render:<source real image>"`, ids 20001+ so file name == `train%05d.jpg` as `lib/dataset/syrip.py` expects, no clash with
existing ids ≤ 12028). bbox = tight silhouette box, area = silhouette pixels, `num_keypoints` = 17. Visibility counts: 7 901 × v2, 973 × v1.
Extra provenance (source fit, texture, background, camera, 3D keypoints): `render_meta.json`. To train on it, merge its `images`/`annotations` into
a copy of the train json and copy/symlink the images next to the others (not done here; no training was launched).

**Label sanity checks** (`tools/check_syn_labels.py` → `data/syrip_syn_new/label_check.json`; grid `figures/syn/syn_label_check.jpg`, left = blue,
right = red, nose = green, hollow = v1; pipeline figure `figures/syn/pipeline_examples.jpg`):
* structure: pycocotools loads it; keys and categories identical to `syrip_fixed`; ids unique; all `is_synthetic`. **OK**
* per label: 0 / 522 with a keypoint outside the image or outside the bbox (+2 px), a wrong `num_keypoints`, or a bad area. **OK**
* left/right: the 2D shoulder/hip orientation agrees with the 3D facing direction of the body in **445 / 445** decidable cases. The nose is v2 when
  the head faces the camera and v1 when it faces away in 367 / 372. The 5 exceptions were inspected: a hand or arm in front of the face, which is correct.
* mapping check against **human** labels (not circular): projecting the fitted SMIL COCO-17 points onto the 200 real images gives median
  errors of 0.1–2.1 % of the box diagonal for all joints except the hips (3.8–4.0 %, after calibration; that number is in-sample, CV value above).
  The L/R-swapped assignment fits better in **0 / 200** images, so the joint map and left/right are correct.

**Limitations.** (1) Backgrounds are not LSUN: procedural textures plus blurred crops of the real training images. The crops can contain unlabelled
objects or parts of other people, and the crops reuse the 200 training scenes, so background diversity is low. (2) Only 12 textures (MINI-RGBD,
mostly a diaper), no clothing; renders look like top-down views on a flat surface, even when the source pose is sitting or crawling. (3) The poses are
the 174 real training poses plus small jitter, so new pose diversity is small; the set mostly adds appearance and viewpoint variation.
(4) The fits use SMIL's single-Gaussian prior, not the adult GMM the authors used, and some fits that pass the 5 % error filter are still
implausible in 3D (2D error cannot detect depth ambiguities). (5) The hip offset is one global 3D vector learned from 174 fits. (6) No truncated bodies.

**Is it worth it? Test first, cheaply:** G2 = **real-only (200R)** training vs D1 (200R + 1000S), at matched iterations (2000). If the
synthetic data adds little, generating more of the same kind will add little too, and Part 7 effort should go elsewhere. Result: see AGENT.md G-track.
**Training test (SY1/SY2, 2026-11-04):** F4 config (100 epochs) on 200R + 1000S + 522 new, as two replicates; compared with F4/H1.

| Run | Test100 final | Test96 final | Test500f final |
|--|--|--|--|
| F4 / H1 (no new data) | 94.0 / 93.9 | 93.2 / 93.3 | 91.7 / 91.6 |
| SY1 / SY2 (+522 new) | 93.2 / 93.1 | 92.6 / 92.5 | 91.7 / 91.5 |

- **Verdict:** no effect on Test500f. The mean is 91.6 against 91.65, inside the pre-registered ±0.3 band.
- **Smaller sets:** both replicates are about 0.8 lower on Test100 and Test96.
- **Consistent with G2:** more synthetic data of this kind does not help.
- **Confounds:**
  - The balanced sampler gives 1.52× more iterations.
  - Real images are oversampled 7.6× instead of 5×.
  - The set covers only 174 distinct poses.
- **What might help instead:** more diverse poses, realistic textures and backgrounds, or a closer match to real hip geometry.

## 7.3 Applications — video batch processing and webcam

`tools/video_infer.py` (shared inference with the demos; `.venv-demo/bin/python tools/video_infer.py ...`):
* `--source clip.mp4` (one video), `--source folder/` (**batch**: every video, plus `batch_summary.json`), `--source 0` (**webcam**; `--show` opens a
  window, `q` quits).
* **Tracking:** the box for frame t+1 is the hull of frame t's confident keypoints × `--box-margin` (1.4). If fewer than `--min-joints` (6) are confident,
  it falls back to `--init-box` or the whole frame. `--smooth` adds an EMA on keypoints.
* **Outputs:** `<name>_pose.mp4` (skeleton, box, live fps), `<name>_keypoints.jsonl` (per-frame box + 17 × [x, y, conf]), `<name>_keypoints.csv`
  (one row per frame, for movement analysis, e.g. General Movements Assessment-style features).

**Test (server, RTX 6000 Ada, released FiDIP weights):** no real infant videos are available, so 3 clips (120 frames, 960×540) were made by
smoothly panning, zooming and rolling a camera over Test100 images (test0, test33, test41). Ground-truth keypoints are known in every frame, so we can score them.
| clip | tracking box: mean OKS (min) | whole-frame box: mean OKS (min) |
|---|---|---|
| test0 | **0.935** (0.885) | 0.925 (0.837) |
| test33 | **0.751** (0.728) | 0.727 (0.698) |
| test41 | **0.771** (0.740) | 0.761 (0.728) |
Speed: model 22–24 fps / end-to-end ~20 fps with flip test; **38 fps / 32 fps with `--no-flip`**, so real-time at 30 fps is feasible without flip.
**Limitations:** the clips are camera motion over still images, not real infant motion. There is no webcam on the server, so `--source 0` is untested here
(same code path as files). There is no person detector, so the infant must be the main subject. Try real videos and a webcam on a desktop.
