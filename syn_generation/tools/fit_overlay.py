"""Overlay SMIL fits on their input images and measure 2D reprojection error.

    cd syn_generation
    CUDA_VISIBLE_DEVICES=1 ../.venv-smil/bin/python tools/fit_overlay.py \
        --data_folder data --fit_folder output_examples --out ../figures/syn/fit_examples.png

For every <fit_folder>/results/<id>/000.pkl: render the fitted mesh (pyrender, same camera as the
fitter), blend it over the image, draw the GT 2D keypoints (green) and the projected SMIL joints
(red, OpenPose-19 map of the fitter) and write a grid figure + <fit_folder>/fit_quality.json
(per-image mean/max reprojection error in px and as a fraction of the keypoint-bbox diagonal).
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys

import cv2
import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from smil_common import FOCAL, OP19_NAMES, SMILNumpy, load_fit, project  # noqa: E402
from smil_render import Renderer  # noqa: E402
from utils import smpl_to_openpose  # noqa: E402  (smplifyx)

OP19_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
              (10, 11), (8, 12), (12, 13), (13, 14), (0, 15), (0, 16), (15, 17), (16, 18)]
IGNORED = {1, 9, 12}  # joints_to_ign in cfg_files/fit_smil.yaml (neck, hips)


def overlay_one(img, fit, kp, sm, rend, model2op):
    H, W = img.shape[:2]
    verts, joints = sm(fit['betas'], fit['global_orient'], fit['body_pose'])
    R, t = fit['camera_rotation'][0], fit['camera_translation'][0]
    center = (W / 2.0, H / 2.0)
    rend.resize(W, H)
    vc = verts @ R.T + t
    rgb, mask, _ = rend.render(vc, sm.f, FOCAL, center, color=(0.55, 0.75, 1.0),
                               lights=[dict(dir=(0.3, 0.5, 1.0), intensity=3.0)])
    left = img.copy()
    m = mask[..., None]
    out = np.where(m, 0.2 * img + 0.8 * rgb[..., ::-1], img).astype(np.uint8)
    j2d, _ = project(np.concatenate([joints, verts[:0]])[model2op], R, t, FOCAL, center)
    used = [i for i in range(19) if kp[i, 2] > 0 and i not in IGNORED]
    err = np.linalg.norm(j2d[used] - kp[used, :2], axis=1)
    lo, hi = kp[used, :2].min(0), kp[used, :2].max(0)
    diag = float(np.linalg.norm(hi - lo))
    s = max(2, int(round(max(H, W) / 300)))
    for a, b in OP19_EDGES:
        if kp[a, 2] > 0 and kp[b, 2] > 0:
            cv2.line(left, tuple(map(int, kp[a, :2])), tuple(map(int, kp[b, :2])), (0, 200, 0), s)
    for i in range(19):
        if kp[i, 2] > 0:
            cv2.circle(left, tuple(map(int, kp[i, :2])), 2 * s, (0, 255, 0), -1)
            cv2.circle(out, tuple(map(int, kp[i, :2])), 2 * s, (0, 255, 0), -1)
        if i not in IGNORED:
            cv2.circle(out, tuple(map(int, j2d[i])), 2 * s, (0, 0, 255), -1)
            if kp[i, 2] > 0:
                cv2.line(out, tuple(map(int, kp[i, :2])), tuple(map(int, j2d[i])), (0, 0, 255), max(1, s // 2))
    out = np.hstack([left, out])
    return out, dict(mean_px=float(err.mean()), max_px=float(err.max()), kp_diag_px=diag,
                     mean_rel=float(err.mean() / diag), max_rel=float(err.max() / diag),
                     worst_joint=OP19_NAMES[used[int(err.argmax())]])


def grid(tiles, cols, tile_w=720, tile_h=300):
    """Letterbox every tile into a fixed tile_w x tile_h box so mixed aspect ratios line up."""
    rs = []
    for t in tiles:
        s = min(tile_w / t.shape[1], tile_h / t.shape[0])
        r = cv2.resize(t, (int(t.shape[1] * s), int(t.shape[0] * s)), interpolation=cv2.INTER_AREA)
        box = np.full((tile_h, tile_w, 3), 255, np.uint8)
        y0, x0 = (tile_h - r.shape[0]) // 2, (tile_w - r.shape[1]) // 2
        box[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
        rs.append(cv2.copyMakeBorder(box, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    while len(rs) % cols:
        rs.append(np.full_like(rs[0], 255))
    return np.vstack([np.hstack(rs[i:i + cols]) for i in range(0, len(rs), cols)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_folder', required=True)
    ap.add_argument('--fit_folder', required=True)
    ap.add_argument('--out', required=True, help='grid png')
    ap.add_argument('--max_tiles', type=int, default=24)
    ap.add_argument('--cols', type=int, default=6)
    ap.add_argument('--ids', nargs='*', help='only these ids in the figure')
    args = ap.parse_args()

    sm = SMILNumpy()
    rend = Renderer(64, 64)
    model2op = smpl_to_openpose('smil', use_hands=False, use_face=False, openpose_format='coco19')
    res = sorted(glob.glob(osp.join(args.fit_folder, 'results', '*', '000.pkl')))
    quality, tiles = {}, []
    for p in res:
        fid = osp.basename(osp.dirname(p))
        img_p = [q for q in glob.glob(osp.join(args.data_folder, 'images', fid + '.*'))][0]
        img = cv2.imread(img_p)
        kp = np.array(json.load(open(osp.join(args.data_folder, 'keypoints', fid + '_keypoints.json')))['annotations']).reshape(-1, 3)
        out, q = overlay_one(img, load_fit(p), kp, sm, rend, model2op)
        quality[fid] = q
        if (args.ids and fid in args.ids) or (not args.ids and len(tiles) < args.max_tiles):
            cv2.putText(out, f'{fid}  err {100 * q["mean_rel"]:.1f}%', (10, int(out.shape[0] * 0.06)),
                        cv2.FONT_HERSHEY_SIMPLEX, out.shape[0] / 700, (255, 255, 255), max(2, out.shape[0] // 300))
            cv2.putText(out, 'GT 2D kp', (10, int(out.shape[0] * 0.97)), cv2.FONT_HERSHEY_SIMPLEX, out.shape[0] / 900, (0, 255, 0), max(2, out.shape[0] // 300))
            cv2.putText(out, 'SMIL fit (red = projected joints)', (out.shape[1] // 2 + 10, int(out.shape[0] * 0.97)), cv2.FONT_HERSHEY_SIMPLEX, out.shape[0] / 900, (0, 0, 255), max(2, out.shape[0] // 300))
            tiles.append(out)
    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, grid(tiles, min(args.cols, len(tiles))))
    rel = np.array([q['mean_rel'] for q in quality.values()])
    summary = dict(n=len(rel), mean_rel_err=float(rel.mean()), median_rel_err=float(np.median(rel)),
                   n_over_10pct=int((rel > 0.10).sum()), per_image=quality)
    json.dump(summary, open(osp.join(args.fit_folder, 'fit_quality.json'), 'w'), indent=1)
    print(f'{len(rel)} fits: mean reprojection error {100 * rel.mean():.2f}% of kp-bbox diagonal '
          f'(median {100 * np.median(rel):.2f}%, >10%: {(rel > 0.10).sum()}) -> {args.out}')


if __name__ == '__main__':
    main()
