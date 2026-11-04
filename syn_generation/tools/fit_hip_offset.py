"""Calibrate a COCO-style hip keypoint for SMIL from the real fits.

SMIL/SMPL hip *joints* sit deep inside the pelvis, much closer together than the hip keypoints
human annotators mark in SyRIP/COCO (the fitter ignores the 2D hips for this reason, joints_to_ign).
Measured on the fitted real images, the projected SMIL hip width is only ~0.36x the annotated one, so
using the raw joints would give synthetic labels with a different hip definition than the real data.
This script fits one 3D offset o (pelvis/root frame, metres, mirrored in x for the right hip):
    hip_coco = J_hip + R(global_orient) @ o
by robust least squares of the 2D reprojection error against the GT hips (2-fold cross-validated),
and writes models/coco_hip_offset.json, which tools/render_synthetic.py uses.

    cd syn_generation
    ../.venv-smil/bin/python tools/fit_hip_offset.py --fit_folder output_real
"""
import argparse
import json
import os.path as osp
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from smil_common import FOCAL, PROJECT, ROOT, SMIL_FOR_COCO17, SMILNumpy, load_fit, project  # noqa: E402


def collect(fit_folder, max_err):
    d = json.load(open(osp.join(PROJECT, 'data/syrip_fixed/annotations/person_keypoints_train_infant.json')))
    q = json.load(open(osp.join(fit_folder, 'fit_quality.json')))['per_image']
    byid = {a['image_id']: a for a in d['annotations']}
    sm = SMILNumpy()
    items = []
    for im in d['images']:
        fid = osp.splitext(im['file_name'])[0]
        p = osp.join(fit_folder, 'results', fid, '000.pkl')
        if im['is_synthetic'] or not osp.exists(p) or q.get(fid, {}).get('mean_rel', 1) > max_err:
            continue
        kp = np.array(byid[im['id']]['keypoints']).reshape(17, 3)
        if kp[11, 2] == 0 or kp[12, 2] == 0 or np.allclose(kp[11, :2], kp[12, :2]):
            continue
        f = load_fit(p)
        _, j = sm(f['betas'], f['global_orient'], f['body_pose'])
        R = sm.rodrigues(np.ravel(f['global_orient']))[0]
        lab = kp[:, 2] > 0
        diag = np.linalg.norm(kp[lab, :2].max(0) - kp[lab, :2].min(0))
        items.append(dict(J=j, R=R, Rc=f['camera_rotation'][0], t=f['camera_translation'][0],
                          c=(im['width'] / 2, im['height'] / 2), gt=kp[[11, 12], :2], diag=diag,
                          uv=project(j[SMIL_FOR_COCO17], f['camera_rotation'][0], f['camera_translation'][0],
                                     FOCAL, (im['width'] / 2, im['height'] / 2))[0]))
    return items


def hips(it, o):
    oL, oR = np.asarray(o), np.asarray(o) * [-1, 1, 1]
    P = np.stack([it['J'][1] + it['R'] @ oL, it['J'][2] + it['R'] @ oR])
    return project(P, it['Rc'], it['t'], FOCAL, it['c'])[0]


def resid(o, items):
    return np.concatenate([((hips(it, o) - it['gt']) / it['diag']).ravel() for it in items])


def err(o, items):
    return np.array([np.linalg.norm(hips(it, o) - it['gt'], axis=1).mean() / it['diag'] for it in items])


def width_ratio(o, items):
    return np.median([np.linalg.norm(np.subtract(*hips(it, o))) / np.linalg.norm(np.subtract(*it['gt'])) for it in items])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit_folder', default='output_real')
    ap.add_argument('--max_fit_err', type=float, default=0.05)
    args = ap.parse_args()
    items = collect(args.fit_folder, args.max_fit_err)
    fit = lambda its: least_squares(resid, np.zeros(3), args=(its,), loss='soft_l1', f_scale=0.05).x  # noqa: E731
    a, b = items[0::2], items[1::2]
    oa, ob = fit(a), fit(b)
    o = fit(items)
    z = np.zeros(3)
    rep = dict(n_images=len(items), offset_left_m=o.tolist(), offset_fold_a=oa.tolist(), offset_fold_b=ob.tolist(),
               raw_joint_err_median=float(np.median(err(z, items))),
               cv_err_median=float(np.median(np.concatenate([err(oa, b), err(ob, a)]))),
               raw_hip_width_ratio=float(width_ratio(z, items)), cv_hip_width_ratio=float(
                   np.median([width_ratio(oa, b), width_ratio(ob, a)])))
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(osp.join(ROOT, 'models', 'coco_hip_offset.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
