"""Audit SyRIP / pre-infant COCO-keypoint annotations for label errors and duplicates.

Usage:
    python tools/check_annotations.py --data data [--out data_audit]

Checks (per annotation file):
  structure : duplicate image/ann ids, duplicate file_name, >1 ann per image,
              orphan anns, images without anns, missing image files,
              width/height vs. real image size
  keypoints : visibility not in {0,1,2}, num_keypoints != #(v>0),
              labelled points outside image / far outside bbox,
              two different joints at (almost) the same pixel,
              left/right swap heuristics, limb-length outliers
  bbox/area : bbox outside image or degenerate, area vs bbox area
              (COCOeval uses gt 'area' as the OKS scale!)
  meta      : is_synthetic flag vs. id range / image_source
  duplicates: identical keypoint arrays across anns, byte-identical and
              near-identical (dHash) images within and across splits, and
              whether duplicated images carry consistent labels
"""
import argparse
import hashlib
import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
from PIL import Image

KP = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_sho', 'r_sho', 'l_elb', 'r_elb',
      'l_wri', 'r_wri', 'l_hip', 'r_hip', 'l_kne', 'r_kne', 'l_ank', 'r_ank']
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
LIMBS = {'upper_arm': [(5, 7), (6, 8)], 'forearm': [(7, 9), (8, 10)],
         'thigh': [(11, 13), (12, 14)], 'shin': [(13, 15), (14, 16)]}
COCO_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
                        1.07, 1.07, .87, .87, .89, .89]) / 10.0

SPLITS = {
    # name: (annotation json, image dir) relative to --data
    'syrip/train_infant': ('syrip/annotations/person_keypoints_train_infant.json', 'syrip/images/train_infant'),
    'syrip/val100': ('syrip/annotations/person_keypoints_validate_infant.json', 'syrip/images/validate_infant'),
    'syrip/train_pre_infant': ('syrip/annotations/person_keypoints_train_pre_infant.json', 'syrip/images/train_pre_infant'),
    'SyRIP/200R_1000S': ('SyRIP/annotations/200R_1000S/person_keypoints_train_infant.json', 'SyRIP/images/train_infant'),
    'SyRIP/200R': ('SyRIP/annotations/200R/person_keypoints_train_infant.json', 'SyRIP/images/train_infant'),
    'SyRIP/1000S': ('SyRIP/annotations/1000S/person_keypoints_train_infant.json', 'SyRIP/images/train_infant'),
    'SyRIP/val100': ('SyRIP/annotations/validate100/person_keypoints_validate_infant.json', 'SyRIP/images/validate_infant'),
    'SyRIP/val500': ('SyRIP/annotations/validate500/person_keypoints_validate_infant.json', 'SyRIP/images/validate_infant'),
}
INFANT_SPLITS = [s for s in SPLITS if 'pre_infant' not in s]


def kps(ann):
    return np.asarray(ann['keypoints'], dtype=float).reshape(17, 3)


def dhash(path, size=16):
    im = Image.open(path).convert('L').resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


def hamming(a, b):
    return int(np.unpackbits(a ^ b).sum())


def side_sign(p_l, p_r, up):
    """Sign of the 2D cross product: which side of the torso axis the left joint is on."""
    d = p_l - p_r
    return np.sign(up[0] * d[1] - up[1] * d[0])


class Auditor:
    def __init__(self, data):
        self.data = data
        self.issues = []  # dict(split, image, ann_id, check, severity, detail)
        self.splits = {}
        self.img_md5 = {}
        self.img_dhash = {}

    def add(self, split, check, sev, detail, image='', ann_id=''):
        self.issues.append(dict(split=split, check=check, severity=sev, image=image,
                                ann_id=ann_id, detail=detail))

    # ------------------------------------------------------------------ per split
    def audit_split(self, name, ann_path, img_dir):
        d = json.load(open(os.path.join(self.data, ann_path)))
        img_dir = os.path.join(self.data, img_dir)
        self.splits[name] = (d, img_dir)
        imgs, anns = d['images'], d['annotations']
        add = lambda *a, **k: self.add(name, *a, **k)

        # ---- structure
        ids = defaultdict(list)
        for im in imgs:
            ids[im['id']].append(im)
        for i, l in ids.items():
            if len(l) > 1:
                add('dup_image_id', 'error', f'image id {i} appears {len(l)}x: {[x["file_name"] for x in l]}')
        fn = defaultdict(list)
        for im in imgs:
            fn[im['file_name']].append(im['id'])
        for f, l in fn.items():
            if len(l) > 1:
                add('dup_file_name', 'error', f'{f} listed {len(l)}x (ids {l})', image=f)
        aid = defaultdict(int)
        for a in anns:
            aid[a['id']] += 1
        for i, c in aid.items():
            if c > 1:
                add('dup_ann_id', 'error', f'annotation id {i} appears {c}x', ann_id=i)
        by_img = defaultdict(list)
        for a in anns:
            by_img[a['image_id']].append(a)
        id2img = {im['id']: im for im in imgs}
        for i, l in by_img.items():
            if i not in id2img:
                add('orphan_ann', 'error', f'ann(s) {[a["id"] for a in l]} reference missing image id {i}')
            if len(l) > 1 and 'pre_infant' not in name:
                same = all(np.allclose(kps(l[0]), kps(x)) for x in l[1:])
                add('multi_ann_per_image', 'error' if same else 'warn',
                    f'{len(l)} anns on one image (ann ids {[a["id"] for a in l]}); keypoints identical={same}',
                    image=id2img.get(i, {}).get('file_name', i))
        for im in imgs:
            if im['id'] not in by_img:
                add('image_without_ann', 'warn', 'image has no annotation', image=im['file_name'])

        # ---- image files
        for im in imgs:
            p = os.path.join(img_dir, im['file_name'])
            if not os.path.exists(p):
                add('missing_image_file', 'error', p, image=im['file_name'])
                continue
            if p not in self.img_md5:
                self.img_md5[p] = hashlib.md5(open(p, 'rb').read()).hexdigest()
                self.img_dhash[p] = dhash(p)
            with Image.open(p) as I:
                w, h = I.size
            if (w, h) != (im['width'], im['height']):
                add('wrong_image_size', 'error', f'json {im["width"]}x{im["height"]} vs file {w}x{h}', image=im['file_name'])

        # ---- meta: synthetic flag
        for im in imgs:
            if 'pre_infant' in name or 'is_synthetic' not in im:
                continue
            syn_by_id = im['id'] >= 10000
            if bool(im['is_synthetic']) != syn_by_id:
                add('is_synthetic_mismatch', 'error',
                    f'is_synthetic={im["is_synthetic"]} but id={im["id"]} source={im.get("image_source")!r}',
                    image=im['file_name'])
            if 'is_labeled' in im and not im['is_labeled']:
                add('is_labeled_false', 'warn', 'image flagged is_labeled=False but has an entry', image=im['file_name'])

        # ---- constant 'area' (COCOeval OKS uses gt area as scale -> metric is wrong)
        areas = {round(a.get('area', 0), 2) for a in anns}
        if len(anns) > 10 and len(areas) == 1:
            add('constant_area', 'error',
                f'all {len(anns)} anns have area={areas.pop()}; OKS/AP computed on this file is invalid')

        # ---- per-annotation
        kp_index = defaultdict(list)
        limb_rows = []
        for a in anns:
            im = id2img.get(a['image_id'])
            if im is None:
                continue
            f = im['file_name']
            W, H = im['width'], im['height']
            k = kps(a)
            v = k[:, 2]
            lab = v > 0
            add_a = lambda c, s, dt: add(c, s, dt, image=f, ann_id=a['id'])

            bad_v = set(v.tolist()) - {0.0, 1.0, 2.0}
            if bad_v:
                add_a('bad_visibility', 'error', f'visibility values {sorted(bad_v)}')
            if a.get('num_keypoints') != int(lab.sum()):
                add_a('num_keypoints_mismatch', 'warn', f'num_keypoints={a.get("num_keypoints")} but {int(lab.sum())} labelled')
            if lab.sum() == 0:
                add_a('no_keypoints', 'error' if 'pre_infant' not in name else 'info', 'annotation has 0 labelled keypoints')
            unl_nonzero = (~lab) & (np.abs(k[:, :2]).sum(1) > 0)
            if unl_nonzero.any():
                add_a('unlabelled_with_coords', 'info',
                      'v=0 but coords set: ' + ', '.join(f'{KP[j]}({k[j,0]:.0f},{k[j,1]:.0f})' for j in np.where(unl_nonzero)[0]))
            # out of image
            tol = 2.0
            out = lab & ((k[:, 0] < -tol) | (k[:, 1] < -tol) | (k[:, 0] > W - 1 + tol) | (k[:, 1] > H - 1 + tol))
            if out.any():
                add_a('kp_outside_image', 'error',
                      ', '.join(f'{KP[j]}({k[j,0]:.1f},{k[j,1]:.1f}) img {W}x{H}' for j in np.where(out)[0]))

            # bbox
            x, y, bw, bh = a['bbox']
            if bw <= 1 or bh <= 1:
                add_a('degenerate_bbox', 'error', f'bbox {a["bbox"]}')
            if x < -tol or y < -tol or x + bw > W + tol or y + bh > H + tol:
                add_a('bbox_outside_image', 'warn', f'bbox {[round(t,1) for t in a["bbox"]]} img {W}x{H}')
            if lab.any() and bw > 1 and bh > 1:
                m = 0.15 * max(bw, bh)
                far = lab & ((k[:, 0] < x - m) | (k[:, 0] > x + bw + m) | (k[:, 1] < y - m) | (k[:, 1] > y + bh + m))
                if far.any():
                    add_a('kp_outside_bbox', 'warn',
                          ', '.join(f'{KP[j]}({k[j,0]:.0f},{k[j,1]:.0f})' for j in np.where(far)[0]) + f' bbox {[round(t) for t in a["bbox"]]}')
                # tightness: bbox much larger than labelled keypoint extent
                if lab.sum() >= 8:
                    ex = (k[lab, 0].max() - k[lab, 0].min()) * (k[lab, 1].max() - k[lab, 1].min())
                    if ex > 0 and bw * bh / ex > 6:
                        add_a('bbox_loose', 'info', f'bbox area {bw*bh:.0f} = {bw*bh/ex:.1f}x keypoint extent')

            # area
            area = a.get('area', 0)
            ba = max(bw, 0) * max(bh, 0)
            if ba > 0:
                r = area / ba
                if r > 1.05 or r < 0.2:
                    add_a('area_vs_bbox', 'error' if (r > 2 or r < 0.1) else 'warn',
                          f'area={area:.0f}, bbox w*h={ba:.0f}, ratio={r:.3f}')

            # duplicate joints: two different labelled joints at ~same position
            if lab.sum() >= 2:
                scale = max(np.sqrt(max(ba, 1)), 1)
                pts = np.where(lab)[0]
                for i, j in combinations(pts, 2):
                    if {i, j} <= {0, 1, 2, 3, 4}:
                        continue  # face points can legitimately be close
                    dist = np.linalg.norm(k[i, :2] - k[j, :2])
                    if dist < 0.01 * scale:
                        add_a('coincident_joints', 'warn', f'{KP[i]} and {KP[j]} {dist:.1f}px apart (bbox scale {scale:.0f})')

            # left/right consistency (torso-axis cross product)
            if lab[[5, 6, 11, 12]].all():
                sho_c = (k[5, :2] + k[6, :2]) / 2
                hip_c = (k[11, :2] + k[12, :2]) / 2
                up = sho_c - hip_c
                torso = np.linalg.norm(up)
                if torso > 1:
                    s_sho = side_sign(k[5, :2], k[6, :2], up)
                    s_hip = side_sign(k[11, :2], k[12, :2], up)
                    # only meaningful when both widths are non-trivial
                    w_sho = np.linalg.norm(k[5, :2] - k[6, :2]) / torso
                    w_hip = np.linalg.norm(k[11, :2] - k[12, :2]) / torso
                    if s_sho != s_hip and w_sho > 0.25 and w_hip > 0.25:
                        add_a('lr_shoulder_hip_disagree', 'warn',
                              f'shoulders and hips on opposite L/R sides (sho width {w_sho:.2f}, hip width {w_hip:.2f} x torso)')
                    # left chain should stay on left side: elbow/wrist/knee/ankle closer to own-side root
                    for (jl, jr, rl, rr) in [(7, 8, 5, 6), (13, 14, 11, 12)]:
                        if lab[[jl, jr]].all():
                            dll = np.linalg.norm(k[jl, :2] - k[rl, :2]) + np.linalg.norm(k[jr, :2] - k[rr, :2])
                            dlr = np.linalg.norm(k[jl, :2] - k[rr, :2]) + np.linalg.norm(k[jr, :2] - k[rl, :2])
                            if dlr < 0.6 * dll:
                                add_a('lr_limb_crossed', 'warn',
                                      f'{KP[jl]}/{KP[jr]} fit the opposite {KP[rl]}/{KP[rr]} better (cost {dll:.0f} vs swapped {dlr:.0f})')
                    for name_l, pairs in LIMBS.items():
                        for side, (p, q) in zip('LR', pairs):
                            if lab[[p, q]].all():
                                limb_rows.append((a['id'], f, name_l, side,
                                                  np.linalg.norm(k[p, :2] - k[q, :2]) / torso))
                    # L/R asymmetry of the same limb
                    for name_l, ((p1, q1), (p2, q2)) in LIMBS.items():
                        if lab[[p1, q1, p2, q2]].all():
                            l1 = np.linalg.norm(k[p1, :2] - k[q1, :2])
                            l2 = np.linalg.norm(k[p2, :2] - k[q2, :2])
                            if min(l1, l2) > 0 and max(l1, l2) / min(l1, l2) > 4 and max(l1, l2) / torso > 0.4:
                                add_a('limb_asymmetry', 'info',
                                      f'{name_l} L={l1/torso:.2f} R={l2/torso:.2f} x torso (ratio {max(l1,l2)/min(l1,l2):.1f}; foreshortening possible)')

            key = tuple(np.round(k, 2).flatten().tolist())
            if lab.sum() > 0:
                kp_index[key].append((a['id'], f))

        # identical keypoint arrays = copy-pasted labels
        for key, l in kp_index.items():
            if len(l) > 1:
                add('identical_keypoints', 'error' if 'pre_infant' not in name else 'warn',
                    f'{len(l)} anns share byte-identical keypoints: {l}')

        # limb-length outliers (robust z on log ratio to torso)
        if limb_rows:
            arr = defaultdict(list)
            for r in limb_rows:
                arr[r[2]].append(r)
            for limb, rows in arr.items():
                vals = np.log(np.array([r[4] for r in rows]) + 1e-6)
                med = np.median(vals)
                mad = np.median(np.abs(vals - med)) * 1.4826 + 1e-6
                for r, vv in zip(rows, vals):
                    z = (vv - med) / mad
                    if z > 4.5:
                        add('limb_too_long', 'warn',
                            f'{r[3]} {limb} = {r[4]:.2f} x torso (median {np.exp(med):.2f}, z={z:.1f})',
                            image=r[1], ann_id=r[0])

    # ------------------------------------------------------------------ cross split
    def cross(self):
        # same name, different label between the two copies of a split
        pairs = [('syrip/val100', 'SyRIP/val100'), ('SyRIP/val100', 'SyRIP/val500'),
                 ('syrip/train_infant', 'SyRIP/200R_1000S'), ('SyRIP/200R', 'SyRIP/200R_1000S'),
                 ('SyRIP/1000S', 'SyRIP/200R_1000S')]
        self.pair_summary = []
        for A, B in pairs:
            da, db = self.splits[A][0], self.splits[B][0]
            fa = {im['file_name']: im for im in da['images']}
            fb = {im['file_name']: im for im in db['images']}
            aa = {a['image_id']: a for a in da['annotations']}
            ab = {a['image_id']: a for a in db['annotations']}
            common = sorted(set(fa) & set(fb))
            ndiff = 0
            for f in common:
                ka, kb = kps(aa[fa[f]['id']]), kps(ab[fb[f]['id']])
                if not np.allclose(ka, kb, atol=0.5):
                    ndiff += 1
                    dj = np.where(np.abs(ka - kb).max(1) > 0.5)[0]
                    self.add(f'{A} vs {B}', 'label_differs_between_copies', 'warn',
                             f'joints differ: {[KP[j] for j in dj]}', image=f)
            self.pair_summary.append((A, B, len(fa), len(fb), len(common), ndiff))

        # byte-identical / near-identical images across all infant splits (dedupe file paths)
        uniq = {}
        for s in INFANT_SPLITS:
            d, img_dir = self.splits[s]
            id2a = defaultdict(list)
            for a in d['annotations']:
                id2a[a['image_id']].append(a)
            for im in d['images']:
                p = os.path.join(img_dir, im['file_name'])
                if p not in self.img_md5:
                    continue
                uniq.setdefault(p, []).append((s, im, id2a[im['id']]))
        # role of each physical file: train or test
        def role(p):
            return 'test' if 'validate' in p else 'train'
        paths = sorted(uniq)
        by_md5 = defaultdict(list)
        for p in paths:
            by_md5[self.img_md5[p]].append(p)
        # syrip/ and SyRIP/ mirror the same files under the same name: only a group
        # with >1 distinct (role, basename) is a real duplicate
        self.exact_groups = [g for g in by_md5.values()
                             if len({(role(p), os.path.basename(p)) for p in g}) > 1]
        for g in self.exact_groups:
            self._compare_group(g, uniq, 'exact_duplicate_image', role)

        # near duplicates via dHash (skip exact ones); only consider labelled files
        lab_paths = [p for p in paths if any(x[2] for x in uniq[p])]
        H = np.stack([np.unpackbits(self.img_dhash[p]) for p in lab_paths]).astype(np.uint8)
        md5s = [self.img_md5[p] for p in lab_paths]
        self.near_pairs = []
        for i in range(len(lab_paths)):
            dist = (H[i + 1:] != H[i]).sum(1)
            for j in np.where(dist <= 12)[0] + i + 1:
                if md5s[i] == md5s[j]:
                    continue
                self.near_pairs.append((lab_paths[i], lab_paths[j], int(dist[j - i - 1])))
        for p, q, dd in self.near_pairs:
            self._compare_group([p, q], uniq, f'near_duplicate_image(dhash={dd})', role)

    def _compare_group(self, g, uniq, check, role):
        rel = lambda p: os.path.relpath(p, self.data)
        labelled = [(p, x) for p in g for x in uniq[p] if x[2]]
        roles = sorted({role(p) for p, _ in labelled})
        leak = 'train+test' if len(roles) > 1 else (roles[0] if roles else 'unlabelled')
        # label consistency among labelled copies (only meaningful for exact dupes: same pixels)
        cons = ''
        if 'exact' in check and len({p for p, _ in labelled}) > 1:
            ks = [(p, kps(x[2][0])) for p, x in labelled]
            base_p, base = ks[0]
            diffs = []
            for p, k in ks[1:]:
                if p == base_p:
                    continue
                both = (base[:, 2] > 0) & (k[:, 2] > 0)
                err = np.linalg.norm(base[:, :2] - k[:, :2], axis=1)
                vis_diff = int(((base[:, 2] > 0) != (k[:, 2] > 0)).sum())
                diffs.append(f'{rel(p)}: max {err[both].max() if both.any() else 0:.1f}px, '
                             f'mean {err[both].mean() if both.any() else 0:.1f}px, vis-mismatch {vis_diff}')
            cons = '; label diff vs ' + rel(base_p) + ' -> ' + ' | '.join(diffs) if diffs else ''
        sev = 'error' if leak == 'train+test' else ('warn' if len({p for p, _ in labelled}) > 1 else 'info')
        self.add('cross', check, sev,
                 f'[{leak}] ' + ', '.join(sorted({rel(p) for p in g})) + cons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--out', default='data_audit')
    args = ap.parse_args()
    au = Auditor(args.data)
    for name, (ann, img) in SPLITS.items():
        print('auditing', name)
        au.audit_split(name, ann, img)
    au.cross()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'issues.json'), 'w') as f:
        json.dump(au.issues, f, indent=1, default=str)
    with open(os.path.join(args.out, 'pairs.json'), 'w') as f:
        json.dump(dict(pair_summary=au.pair_summary, exact_groups=au.exact_groups,
                       near_pairs=au.near_pairs), f, indent=1, default=str)
    cnt = defaultdict(lambda: defaultdict(int))
    for i in au.issues:
        cnt[(i['split'] if i['split'] != 'cross' else 'cross')][(i['check'].split('(')[0], i['severity'])] += 1
    for s in cnt:
        print(f'\n== {s}')
        for (c, sev), n in sorted(cnt[s].items()):
            print(f'  {sev:5s} {c:32s} {n}')
    print('\ncopies compared (A, B, |A|, |B|, common, differing):')
    for r in au.pair_summary:
        print(' ', r)
    print(f'exact duplicate image groups: {len(au.exact_groups)}; near-duplicate pairs: {len(au.near_pairs)}')


if __name__ == '__main__':
    main()
