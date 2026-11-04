"""Sanity checks for the synthetic COCO keypoint set (lesson of the label audit: check labels).

    cd syn_generation
    ../.venv-smil/bin/python tools/check_syn_labels.py --root ../data/syrip_syn_new \
        --grid ../figures/syn/syn_label_check.jpg [--fit_folder output_real]

1. structure: pycocotools loads it; image/annotation keys identical to data/syrip_fixed; ids unique;
   file names follow lib/dataset/syrip.py (train%05d.jpg == id); is_synthetic true.
2. per label: image exists with the json size; all 17 keypoints inside the image and inside the bbox
   (+2 px); num_keypoints == #(v>0); 0 < area <= bbox area.
3. left/right: the 2D orientation of (left_shoulder - right_shoulder) vs. (shoulder centre - hip centre)
   must agree with the 3D facing direction of the posed SMIL body (render_meta.json kp_cam); and the
   nose should be visible (v=2) when the head faces the camera, occluded (v=1) when it faces away
   (remaining mismatches are listed: hand in front of the face etc.).
4. (--fit_folder) mapping check on the *real* images: project the fitted SMIL COCO-17 joints and
   compare with the human GT labels (independent of the renderer): per-joint error, and whether the
   left/right-swapped assignment would fit better (it must not).
5. a grid of rendered images with the json keypoints drawn (left = blue, right = red, nose = green;
   hollow = v1).
"""
import argparse
import json
import os.path as osp
import sys

import cv2
import numpy as np
from pycocotools.coco import COCO

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from smil_common import (COCO17_NAMES, COCO17_SKELETON, FOCAL, LR_PAIRS, PROJECT,  # noqa: E402
                         SMILNumpy, coco17_points, load_fit, load_hip_offset, project)

REF = osp.join(PROJECT, 'data/syrip_fixed/annotations/person_keypoints_train_infant.json')


def draw(img, kp, bbox=None):
    for s, e in COCO17_SKELETON:
        if kp[s - 1, 2] > 0 and kp[e - 1, 2] > 0:
            cv2.line(img, tuple(map(int, kp[s - 1, :2])), tuple(map(int, kp[e - 1, :2])), (230, 230, 230), 1, cv2.LINE_AA)
    for i, (x, y, v) in enumerate(kp):
        if v <= 0:
            continue
        col = (0, 220, 0) if i == 0 else ((255, 80, 0) if i % 2 == 1 else (0, 0, 255))  # odd idx = left
        cv2.circle(img, (int(round(x)), int(round(y))), 4, col, -1 if v == 2 else 1, cv2.LINE_AA)
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 1)
    return img


def check_real_mapping(fit_folder, real_ann):
    d = json.load(open(real_ann))
    gt = {osp.splitext(im['file_name'])[0]: None for im in d['images'] if not im['is_synthetic']}
    byid = {a['image_id']: a for a in d['annotations']}
    for im in d['images']:
        if not im['is_synthetic']:
            gt[osp.splitext(im['file_name'])[0]] = (np.array(byid[im['id']]['keypoints']).reshape(17, 3), im)
    sm = SMILNumpy()
    hip_offset = load_hip_offset()
    errs, swaps, n = [], 0, 0
    for fid, (kp, im) in sorted(gt.items()):
        p = osp.join(fit_folder, 'results', fid, '000.pkl')
        if not osp.exists(p):
            continue
        f = load_fit(p)
        _, j = sm(f['betas'], f['global_orient'], f['body_pose'])
        uv, _ = project(coco17_points(j, f['global_orient'], hip_offset), f['camera_rotation'][0], f['camera_translation'][0], FOCAL,
                        (im['width'] / 2, im['height'] / 2))
        lab = kp[:, 2] > 0
        diag = np.linalg.norm(kp[lab, :2].max(0) - kp[lab, :2].min(0))
        e = np.where(lab, np.linalg.norm(uv - kp[:, :2], axis=1) / diag, np.nan)
        errs.append(e)
        perm = np.arange(17)
        for a, b in LR_PAIRS:
            perm[a], perm[b] = b, a
        es = np.where(lab[perm], np.linalg.norm(uv - kp[perm, :2], axis=1) / diag, np.nan)
        swaps += int(np.nansum(es) < np.nansum(e))
        n += 1
    errs = np.array(errs)
    med = np.nanmedian(errs, 0)
    print(f'\n[4] SMIL->COCO mapping on {n} fitted real images (error / GT-kp-bbox diagonal; hips '
          f'{"calibrated" if hip_offset is not None else "raw joints"}):')
    for k in range(17):
        print(f'    {COCO17_NAMES[k]:15s} median {100 * med[k]:5.1f}%   p90 {100 * np.nanpercentile(errs[:, k], 90):5.1f}%')
    print(f'    L/R-swapped labels fit better in {swaps}/{n} images (should be ~0)')
    return dict(n=n, median_rel_err=dict(zip(COCO17_NAMES, med.round(4).tolist())), lr_swap_better=swaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--image_set', default='train_infant')
    ap.add_argument('--grid', required=True)
    ap.add_argument('--n_grid', type=int, default=24)
    ap.add_argument('--fit_folder')
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    ann_fn = osp.join(args.root, 'annotations', f'person_keypoints_{args.image_set}.json')
    coco = COCO(ann_fn)
    d = json.load(open(ann_fn))
    ref = json.load(open(REF))
    report, bad = {}, []
    same_img = set(d['images'][0]) == set(ref['images'][0])
    same_ann = set(d['annotations'][0]) == set(ref['annotations'][0])
    ids = [im['id'] for im in d['images']]
    report['structure'] = dict(n_images=len(d['images']), n_anns=len(d['annotations']), image_keys_match=same_img,
                               ann_keys_match=same_ann, unique_ids=len(set(ids)) == len(ids),
                               names_follow_id=all(im['file_name'] == f'train{im["id"]:05d}.jpg' for im in d['images']),
                               all_synthetic=all(im['is_synthetic'] is True for im in d['images']),
                               categories_match=d['categories'] == ref['categories'])
    print('[1] structure:', report['structure'])

    imgs = {im['id']: im for im in d['images']}
    vis_count = np.zeros(3, int)
    for a in d['annotations']:
        im = imgs[a['image_id']]
        kp = np.array(a['keypoints']).reshape(17, 3)
        x, y, w, h = a['bbox']
        img = cv2.imread(osp.join(args.root, 'images', args.image_set, im['file_name']))
        probs = []
        if img is None or img.shape[:2] != (im['height'], im['width']):
            probs.append('image missing / size mismatch')
        if (kp[:, 0] < 0).any() or (kp[:, 1] < 0).any() or (kp[:, 0] > im['width'] - 1).any() or (kp[:, 1] > im['height'] - 1).any():
            probs.append('kp outside image')
        if ((kp[:, 0] < x - 2) | (kp[:, 0] > x + w + 2) | (kp[:, 1] < y - 2) | (kp[:, 1] > y + h + 2)).any():
            probs.append('kp outside bbox')
        if a['num_keypoints'] != int((kp[:, 2] > 0).sum()):
            probs.append('num_keypoints mismatch')
        if not (0 < a['area'] <= w * h):
            probs.append('bad area')
        for v in kp[:, 2].astype(int):
            vis_count[v] += 1
        if probs:
            bad.append((a['id'], probs))
    report['per_label'] = dict(n_bad=len(bad), examples=bad[:10], visibility_counts=dict(v0=int(vis_count[0]),
                               v1=int(vis_count[1]), v2=int(vis_count[2])))
    print('[2] per-label problems:', len(bad), bad[:5], ' visibility counts v0/v1/v2 =', vis_count.tolist())

    meta_fn = osp.join(args.root, 'render_meta.json')
    if osp.exists(meta_fn):
        meta = {m['id']: m for m in json.load(open(meta_fn))}
        agree = n_lr = 0
        nose_ok = n_nose = 0
        disagree_ids, nose_bad = [], []
        for a in d['annotations']:
            kp = np.array(a['keypoints']).reshape(17, 3)
            P = np.array(meta[a['id']]['kp_cam'])
            left3, up3 = P[5] - P[6], (P[5] + P[6]) / 2 - (P[11] + P[12]) / 2
            fwd = np.cross(left3, up3)
            fz = fwd[2] / (np.linalg.norm(fwd) + 1e-9)
            l2, u2 = kp[5, :2] - kp[6, :2], (kp[5, :2] + kp[6, :2]) / 2 - (kp[11, :2] + kp[12, :2]) / 2
            c2 = l2[0] * u2[1] - l2[1] * u2[0]
            if abs(fz) > 0.3 and abs(c2) > 1e-3 * np.linalg.norm(l2) * np.linalg.norm(u2):
                n_lr += 1
                ok = (fz < 0) == (c2 < 0)
                agree += int(ok)
                if not ok:
                    disagree_ids.append(a['id'])
            # head facing = nose relative to the ear midpoint (the head can face the camera while the
            # torso faces away, e.g. a prone infant lifting its head)
            hf = P[0] - (P[3] + P[4]) / 2
            hz = hf[2] / (np.linalg.norm(hf) + 1e-9)
            if abs(hz) > 0.6:
                n_nose += 1
                ok = (kp[0, 2] == 2) == (hz < 0)
                nose_ok += int(ok)
                if not ok:
                    nose_bad.append(a['id'])
        report['left_right'] = dict(n_checked=n_lr, consistent=agree, inconsistent_ids=disagree_ids[:10],
                                    nose_vis_vs_head_facing_n=n_nose, nose_vis_vs_head_facing_ok=nose_ok,
                                    nose_mismatch_ids=nose_bad[:20])
        print(f'[3] L/R 2D-vs-3D orientation consistent in {agree}/{n_lr}; nose visibility matches head facing in '
              f'{nose_ok}/{n_nose} (mismatches: {nose_bad[:8]})')

    if args.fit_folder:
        report['real_mapping'] = check_real_mapping(args.fit_folder, REF)

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(d['annotations']), min(args.n_grid, len(d['annotations'])), replace=False)
    tiles = []
    for i in sorted(pick):
        a = d['annotations'][i]
        im = imgs[a['image_id']]
        img = cv2.imread(osp.join(args.root, 'images', args.image_set, im['file_name']))
        img = draw(img, np.array(a['keypoints']).reshape(17, 3), a['bbox'])
        cv2.putText(img, im['file_name'], (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        tiles.append(cv2.resize(img, (400, 300), interpolation=cv2.INTER_AREA))
    while len(tiles) % 6:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[k:k + 6]) for k in range(0, len(tiles), 6)])
    cv2.imwrite(args.grid, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    json.dump(report, open(osp.join(args.root, 'label_check.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('grid ->', args.grid, '; report ->', osp.join(args.root, 'label_check.json'))


if __name__ == '__main__':
    main()
