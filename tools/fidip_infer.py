"""Shared single-image inference for FiDIP (used by demo.py, streamlit_demo.py, visualize_preds.py).

Mirrors the evaluation path in lib/core/function.py::validate_adaptive: bbox -> center/scale (x1.25 padding),
affine crop to cfg.MODEL.IMAGE_SIZE, ImageNet normalisation, optional flip test (average with flipped heatmaps),
then DarkPose post-processing (core.inference.get_final_preds: gaussian blur + Taylor refinement).
"""
import os
import sys

import cv2
import numpy as np
import torch

_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib')
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

from config import cfg as _base_cfg  # noqa: E402
from core.inference import get_final_preds  # noqa: E402
from utils.transforms import get_affine_transform, flip_back  # noqa: E402
import models  # noqa: E402,F401

KEYPOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
             'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
             'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
FLIP_PAIRS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
            (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
# COCO OKS per-keypoint sigmas
SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
# validated categorical palette (BGR for OpenCV): left = blue, right = orange, centre = aqua
_LEFT, _RIGHT, _MID = (214, 120, 42), (52, 104, 235), (122, 175, 27)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _side(j):
    name = KEYPOINTS[j]
    return _LEFT if name.startswith('left') else _RIGHT if name.startswith('right') else _MID


def load_model(cfg_file, model_file, device='cuda'):
    """Build adaptive_pose_hrnet from the yaml and load weights; returns (model, cfg). Fails loudly on key mismatch."""
    cfg = _base_cfg.clone()
    cfg.defrost(); cfg.merge_from_file(cfg_file); cfg.TEST.MODEL_FILE = model_file; cfg.freeze()
    model_p, _ = eval('models.' + cfg.MODEL.NAME + '.get_adaptive_pose_net')(cfg, is_train=False)
    state = torch.load(model_file, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'best_state_dict' in state:       # a checkpoint.pth
        state = state['best_state_dict']
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    missing, unexpected = model_p.load_state_dict(state, strict=False)
    missing = [k for k in missing if 'num_batches_tracked' not in k]
    if missing:
        raise RuntimeError(f'{len(missing)} weights missing from {model_file}, e.g. {missing[:3]}')
    model_p.to(device).eval()
    return model_p, cfg


def box_to_cs(box, cfg):
    """(x, y, w, h) -> center, scale exactly as lib/dataset/syrip.py::_xywh2cs (pixel_std 200, x1.25)."""
    x, y, w, h = box
    aspect = cfg.MODEL.IMAGE_SIZE[0] * 1.0 / cfg.MODEL.IMAGE_SIZE[1]
    center = np.array([x + (w - 1) * 0.5, y + (h - 1) * 0.5], dtype=np.float32)
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect
    return center, np.array([w / 200.0, h / 200.0], dtype=np.float32) * 1.25


@torch.no_grad()
def predict(model, cfg, img_bgr, box=None, flip=True, device='cuda'):
    """Return (keypoints[17, 2] in image pixels, scores[17]) for one person in `box` (default: whole image)."""
    if box is None:
        box = (0, 0, img_bgr.shape[1], img_bgr.shape[0])
    center, scale = box_to_cs(box, cfg)
    w, h = int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])
    trans = get_affine_transform(center, scale, 0, (w, h))
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if cfg.DATASET.COLOR_RGB else img_bgr
    crop = cv2.warpAffine(img, trans, (w, h), flags=cv2.INTER_LINEAR)
    x = torch.from_numpy(((crop.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]).to(device)
    _, hm = model(x)
    if flip:
        _, hm_f = model(torch.flip(x, dims=[3]))
        hm = (hm + torch.from_numpy(flip_back(hm_f.cpu().numpy(), FLIP_PAIRS).copy()).to(device)) * 0.5
    preds, maxvals = get_final_preds(cfg, hm.cpu().numpy(), np.array([center]), np.array([scale]))
    return preds[0], maxvals[0, :, 0]


def draw(img_bgr, kps, scores=None, thr=0.3, radius=None, thickness=None):
    """Draw skeleton + joints; joints with score < thr (and their limbs) are skipped."""
    out = img_bgr.copy()
    s = max(out.shape[:2]) / 512.0
    r = radius or max(2, int(round(4 * s))); t = thickness or max(1, int(round(2 * s)))
    ok = np.ones(len(kps), bool) if scores is None else scores >= thr
    for a, b in SKELETON:
        if ok[a] and ok[b]:
            col = _side(a) if _side(a) == _side(b) else _MID
            cv2.line(out, tuple(int(v) for v in kps[a]), tuple(int(v) for v in kps[b]), col, t, cv2.LINE_AA)
    for j, (px, py) in enumerate(kps):
        if ok[j]:
            cv2.circle(out, (int(px), int(py)), r + 1, (251, 252, 252), -1, cv2.LINE_AA)   # surface ring
            cv2.circle(out, (int(px), int(py)), r, _side(j), -1, cv2.LINE_AA)
    return out


def oks(pred, gt, gt_vis, area):
    """COCO object keypoint similarity between one prediction and one GT (visible joints only)."""
    v = gt_vis > 0
    if not v.any():
        return np.nan
    d2 = ((pred - gt) ** 2).sum(1)
    e = d2 / (2 * (area + np.spacing(1)) * (2 * SIGMAS) ** 2)
    return float(np.exp(-e[v]).mean())
