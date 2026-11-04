# Part 6: demos (6.1 CLI, 6.2 Streamlit, 6.3 own images)

All demos share one inference path, `tools/fidip_infer.py`. It does the same affine crop, flip test and DarkPose decoding as the evaluation, and
loads any of our checkpoints. **The default model is the final model `models/F4/final_state.pth`** (a copy of the cluster's F4 `final_state.pth`).

## Change made for real photos: find the infant first

FiDIP is a **top-down** model: it estimates one pose inside one box. In SyRIP the box is given. In a real photo the infant often fills only part of
the frame, and there may be adults too.

`detect_people()` therefore runs a torchvision Faster R-CNN (ResNet-50-FPN v2, COCO) person detector and returns the person boxes, largest first.
- **CLI:** `--box-mode detect` (default; falls back to the whole image), `--box-mode full`, `--person K` to choose a box, or `--bbox x,y,w,h`.
- **Streamlit:** a *Box* radio (Detect person / Whole image / Manual) and a *Detected people* dropdown to pick the infant.

## 6.1 CLI demo

```bash
CUDA_VISIBLE_DEVICES=0 .venv-demo/bin/python tools/demo.py \
    --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
    --image temp_uploads/own05.jpg --model models/F4/final_state.pth \
    --output figures/demo/own05_fidip.jpg --conf-threshold 0.3 --output-size 512,512
```

The command prints a 17-row table (x, y, confidence; keypoints below the threshold are marked *hidden*), then the box source, the mean confidence
and the number of keypoints shown. The output image has the skeleton (blue = left, orange = right, aqua = centre) and the box.

## 6.2 Streamlit demo (run and used in a real browser)

```bash
CUDA_VISIBLE_DEVICES=0 .venv-demo/bin/streamlit run tools/streamlit_demo.py --server.headless true --server.port 8501
# open http://localhost:8501 (or http://<server>:8501); deep link: ?src=temp_uploads/&img=own05.jpg
```

**Features:**
- Choose the model (F4 first).
- Confidence threshold slider and a flip-test toggle.
- Image source: upload (a copy is kept in `temp_uploads/` and existing files are never overwritten), `temp_uploads/`, or Test100.
- Box mode, plus choosing among the detected people.
- Shows the keypoints shown, the mean confidence and a per-keypoint confidence table, with a download button for the annotated image.

**Test (2026-11-04):** the server ran, and a headless Chromium (Playwright) drove the real UI.

| screenshot | what it tests |
|---|---|
| `figures/demo/streamlit/1_temp_uploads_detect.png` | `temp_uploads/` source, detected box, default threshold |
| `figures/demo/streamlit/2_pick_infant_among_people.png` | 5 people detected; the infant (#5) chosen in the dropdown; 17/17 keypoints, mean conf 0.823 |
| `figures/demo/streamlit/3_upload_threshold_0.6.png` | file-upload path and slider moved to 0.60; the largest person is the mother, which shows why the dropdown is needed |

## 6.3 Own infant images

SyRIP is our only infant data and we have no photos of our own. We used **12 photos from Wikimedia Commons that are not in SyRIP**: crawling,
tummy time, sitting, lying on a bed, an archival photo, and a swaddled newborn held by an adult. They are in `temp_uploads/own00–11.jpg`, with
source, author and licence (CC0/CC BY/CC BY-SA/no restrictions) in `temp_uploads/SOURCES.md`.

- Results: `figures/demo/own*_fidip.jpg`
- Contact sheet: `figures/demo/own_grid.jpg`
- Chosen infant box for multi-person photos: `figures/demo/own08_p4_fidip.jpg`, `own11_p1_fidip.jpg`

**Mean keypoint confidence:**
- Box columns: *whole* = the whole image is the box; *det* = the detected infant box (own08: person #4, own11: #1).
- Confidence is not accuracy: these photos have no ground truth.
- The last column is for det with F4: keypoints with confidence ≥ 0.3.

| image | content | whole E0 | whole F4 | det E0 | det F4 | det F4 shown |
|---|---|---|---|---|---|---|
| own00 | crawling on grass | 0.876 | 0.913 | 0.862 | 0.905 | 17/17 |
| own01 | lying on side, close-up | 0.807 | 0.835 | 0.791 | 0.821 | 17/17 |
| own02 | crawling, B/W | 0.885 | 0.906 | 0.840 | 0.882 | 17/17 |
| own03 | sitting with toys | 0.925 | 0.951 | 0.953 | 0.969 | 17/17 |
| own04 | crawling toward camera (strong foreshortening) | 0.792 | 0.798 | 0.752 | 0.759 | 15/17 |
| own05 | prone on bed | 0.870 | 0.907 | 0.876 | 0.939 | 17/17 |
| own06 | prone on bed, head up | 0.838 | 0.915 | 0.896 | 0.946 | 17/17 |
| own07 | small infant lying, far from camera | 0.267 | 0.321 | 0.746 | 0.773 | 17/17 |
| own08 | archival photo, infant small, 4 other people | 0.329 | 0.276 | 0.788 | 0.823 | 17/17 |
| own09 | prone on grass, side view | 0.907 | 0.934 | 0.927 | 0.949 | 17/17 |
| own10 | crawling, far | 0.858 | 0.899 | 0.873 | 0.904 | 17/17 |
| own11 | swaddled newborn held by an adult | 0.530 | 0.534 | 0.642 | 0.676 | 14/17 |
| **mean** | | 0.740 | 0.766 | 0.829 | **0.862** | |

**Analysis:**
1. **The box matters more than the model.** With the whole image as the box, small infants fail (own07 0.32, own08 0.28): the skeleton lands on
   background or on an adult. With a detected box they reach 0.77–0.82. The mean rises from 0.77 to 0.86 for F4.
2. **F4 is more confident than the released model on 11 of 12 images** (detected box; mean 0.862 vs 0.829). Its largest gains are on the prone
   infants (own05 +0.06, own06 +0.05), the pose type that dominates SyRIP. This matches the per-joint gains on Test100 (hips, ankles), but here it
   is not a measured accuracy gain.
3. **Visual check (contact sheet):** skeletons are plausible on own00, 02, 03, 05, 06, 07, 09 and 10.
4. **Failure modes:**
   - (a) **Choosing the person:** in own08 and own11, "largest person" is an adult. An infant-vs-adult classifier (or box size and aspect
     priors) would be needed for automatic use.
   - (b) **Swaddling:** own11 has almost no visible limbs. The model invents arms and legs inside the blanket at low confidence (0.4–0.6), and the
     threshold hides some of them; raising it to 0.6 leaves 5/17.
   - (c) **Strong foreshortening** (own04, crawling straight toward the camera) gives the lowest confidence among the uncluttered images, and 2
     keypoints are hidden.
   - (d) **Left/right on side views** (own01, own09): the far-side limbs are mostly occluded and their confidences are the lowest in those images.
     This is the same self-occlusion failure as test33/test41 in Part 4.
