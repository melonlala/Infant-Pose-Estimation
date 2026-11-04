# Part 7.1 (synthetic data generation) and 7.3 (applications)

## 7.1 Synthetic infant data generation — status: blocked on the SMIL licence

**What the pipeline does** (`syn_generation/`):
1. **Fit SMIL to 2D poses:** `smplifyx/main.py` (SMPLify-X adapted to the infant SMIL model). It optimises SMIL shape (20 betas) and pose (69 = 23 joints × 3)
   plus the camera so that the projected SMIL joints match the 2D keypoints of a real infant image. Output: `output/results/<id>/000.pkl`
   (`betas, body_pose, global_orient, camera_*`) and the posed mesh `output/meshes/<id>/000.obj`. Two examples (46, 247) are included.
2. **Render:** `render/image_generation.py` poses SMIL with those parameters (OpenDR), applies a random texture (12 MINI-RGBD infant textures
   in `render/textures/`, optionally SMPL adult clothing), lights it and composites it on a random background (LSUN). The 2D keypoints of the
   rendered infant come from SMIL's joint regressor, which is why the synthetic labels are exact by construction.

**Why it cannot run here yet (checked 2026-11-03):**
| needed | status |
|---|---|
| `smil_web.pkl` (model: template, blend shapes, skinning weights, joint regressor) | **missing** on server and cluster. Download needs registration and acceptance of the SMIL licence (Fraunhofer IOSB, MINI-RGBD page). Only the user can accept it |
| `smil_pose_prior` (for fitting) | missing, comes with SMIL |
| OpenDR 0.78 | not installed. It is a Python-3.6-era C extension and hard to build on python 3.12/3.13. Port the renderer to `pyrender` or `pytorch3d` instead |
| backgrounds | `bg_folder`/`txt_folder` are hard-coded to the authors' machine (`/home/faye/...`). LSUN download or any image folder works |
| `template.obj` | present (static template mesh only; it cannot be posed without the SMIL skinning weights) |

**Steps once SMIL is available:** put `smil_web.pkl` in `syn_generation/models/` and `syn_generation/render/`, and `smil_pose_prior` in `priors/`.
Make the background and texture paths arguments. Replace OpenDR with pyrender (`TexturedRenderer` → `pyrender.Mesh.from_trimesh` +
`OffscreenRenderer`; `ProjectPoints` → the same pinhole camera). Fit on the 200 real training images (their GT keypoints) to get new poses.
Render N variants per pose (texture × background × camera jitter), write COCO json with `is_synthetic: true`, sanity-check with
`tools/visualize_preds.py`-style overlays (lesson from the label audit), and train on 200R + 1000S + new.

**Is it worth it? Test first, cheaply:** G2 = **real-only (200R)** training vs D1 (200R + 1000S), at matched iterations (2000). If the
synthetic data adds little, generating more of the same kind will add little too, and Part 7 effort should go elsewhere. Result: see AGENT.md G-track.

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
