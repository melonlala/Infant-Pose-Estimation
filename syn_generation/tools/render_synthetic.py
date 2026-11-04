"""Render synthetic infant images + COCO keypoint labels from SMIL fits (port of
render/image_generation.py from OpenDR to pyrender/EGL; all folders are arguments).

    cd syn_generation
    CUDA_VISIBLE_DEVICES=1 ../.venv-smil/bin/python tools/render_synthetic.py \
        --fit_folders output_real --variants 3 \
        --out_root ../data/syrip_syn_new --image_set train_infant --start_id 20001

For each fit <fit_folder>/results/<id>/000.pkl (skipped if its reprojection error in
<fit_folder>/fit_quality.json is above --max_fit_err), render --variants images:
  * pose / shape jitter: body_pose += N(0, (0.3*prior_std, capped 0.08 rad)), betas += N(0, 0.25)
  * view jitter: extra rotation of the body about its centre (roll +-40 deg, pitch/yaw +-20 deg),
    random focal length (500-1000 px), random scale (body 45-85% of the image) and position
  * appearance: one of the 12 MINI-RGBD textures (render/textures) with mild colour jitter,
    1-2 directional lights with random direction / intensity / tint + ambient
  * background: random image of --bg_folder (tools/make_backgrounds.py), random flip / brightness,
    soft contact shadow, feathered alpha, sensor noise, blur and JPEG re-compression
Labels: the 17 COCO keypoints are the SMIL joints (smil_common.SMIL_FOR_COCO17) projected with the
render camera; the hips use the calibrated COCO-style hip point (tools/fit_hip_offset.py) unless
--raw_hips. Visibility: 2 if the keypoint is visible, 1 if self-occluded (a face point behind
the surface by > 1.5 cm, or a body joint whose pixel shows a body part not adjacent to it in the
part-label render). All keypoints must lie inside the image (otherwise the view is re-sampled).
bbox = tight box of the rendered silhouette, area = silhouette pixel count.

Writes <out_root>/images/<image_set>/train<id>.jpg and
<out_root>/annotations/person_keypoints_<image_set>.json (same fields as data/syrip_fixed,
is_synthetic: true, image_source: "smil_render:<real source image>"), plus render_meta.json.
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from smil_common import (PROJECT, ROOT, SURFACE_COCO17, SMILNumpy, coco17_points,  # noqa: E402
                         load_fit, load_hip_offset, project)
from smil_render import Renderer, load_template_uv, render_labels  # noqa: E402

W, H = 640, 480
# body parts (dominant SMIL skinning joint) that may legitimately be in front of each COCO joint
ADJ_PARTS = {5: {13, 16, 9, 6}, 6: {14, 17, 9, 6}, 7: {16, 18}, 8: {17, 19}, 9: {18, 20, 22},
             10: {19, 21, 23}, 11: {0, 1, 3}, 12: {0, 2, 3}, 13: {1, 4}, 14: {2, 5},
             15: {4, 7, 10}, 16: {5, 8, 11}}


def rot(axis, deg):
    r = np.zeros(3)
    r['xyz'.index(axis)] = np.radians(deg)
    return cv2.Rodrigues(r)[0]


def sample_lights(rng):
    lights = []
    for _ in range(rng.integers(1, 3)):
        d = np.array([rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), 1.0])  # mostly from the camera side
        tint = np.clip(1 + rng.normal(0, 0.06, 3), 0.85, 1.0)
        lights.append(dict(dir=d, intensity=rng.uniform(2.5, 5.0), color=tint))
    return lights


def jitter_texture(tex, rng):
    hsv = cv2.cvtColor(tex, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6)) % 180
    hsv[..., 1] *= rng.uniform(0.8, 1.15)
    hsv[..., 2] *= rng.uniform(0.85, 1.15)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def place_camera(verts, rng, focal):
    """Translation putting the (already rotated, centred) body at a random scale / position."""
    for frac in np.linspace(rng.uniform(0.45, 0.85), 0.35, 6):
        ext = verts[:, :2].max(0) - verts[:, :2].min(0)
        depth_ext = verts[:, 2].max() - verts[:, 2].min()
        # projected size ~ focal * ext / z ; the larger side gets `frac` of its image side
        z = max(focal * ext[0] / (frac * W), focal * ext[1] / (frac * H)) + depth_ext / 2
        sw, sh = focal * ext[0] / z, focal * ext[1] / z
        cx = W / 2 + rng.uniform(-1, 1) * max(0, (W - sw) / 2 - 10) * 0.8
        cy = H / 2 + rng.uniform(-1, 1) * max(0, (H - sh) / 2 - 10) * 0.8
        mid = (verts[:, :2].max(0) + verts[:, :2].min(0)) / 2
        t = np.array([(cx - W / 2) * z / focal - mid[0], (cy - H / 2) * z / focal - mid[1], z])
        uv, _ = project(verts, t=t, focal=focal, center=(W / 2, H / 2))
        if uv[:, 0].min() >= 2 and uv[:, 1].min() >= 2 and uv[:, 0].max() <= W - 3 and uv[:, 1].max() <= H - 3:
            return t
    return None


def composite(rgb, mask, bg, rng, light_dir):
    m = mask.astype(np.float32)
    # soft contact shadow, offset away from the main light
    off = -np.asarray(light_dir[:2]) / (abs(light_dir[2]) + 1e-6) * rng.uniform(4, 12)
    M = np.float32([[1, 0, off[0]], [0, 1, off[1]]])
    sh = cv2.warpAffine(m, M, (W, H))
    sh = cv2.GaussianBlur(sh, (0, 0), rng.uniform(5, 14))
    bgf = bg.astype(np.float32) * (1 - rng.uniform(0.25, 0.5) * sh[..., None])
    a = cv2.GaussianBlur(m, (0, 0), 0.7)[..., None]
    out = a * rgb.astype(np.float32) + (1 - a) * bgf
    out *= rng.uniform(0.85, 1.15)
    out += rng.normal(0, rng.uniform(1, 5), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.0))
    q = int(rng.integers(70, 96))
    return cv2.imdecode(cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def visibility(kp_uv, kp_z, depth, labels):
    vis = np.full(17, 2, int)
    for k in range(17):
        x, y = int(round(kp_uv[k, 0])), int(round(kp_uv[k, 1]))
        if SURFACE_COCO17[k]:
            patch = depth[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            patch = patch[patch > 0]
            if patch.size and kp_z[k] - patch.min() > 0.015:
                vis[k] = 1
        else:
            patch = labels[max(0, y - 2):y + 3, max(0, x - 2):x + 3].ravel()
            patch = patch[patch >= 0]
            if patch.size:
                vals, cnt = np.unique(patch, return_counts=True)
                if vals[cnt.argmax()] not in ADJ_PARTS[k]:
                    vis[k] = 1
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit_folders', nargs='+', required=True)
    ap.add_argument('--variants', type=int, default=3)
    ap.add_argument('--max_fit_err', type=float, default=0.05,
                    help='skip fits whose mean reprojection error > this fraction of the kp-bbox diagonal')
    ap.add_argument('--bg_folder', default=osp.join(ROOT, 'render', 'bg_pool'))
    ap.add_argument('--txt_folder', default=osp.join(ROOT, 'render', 'textures'))
    ap.add_argument('--out_root', default=osp.join(PROJECT, 'data', 'syrip_syn_new'))
    ap.add_argument('--image_set', default='train_infant')
    ap.add_argument('--start_id', type=int, default=20001)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--raw_hips', action='store_true',
                    help='label hips with the raw SMIL hip joints instead of models/coco_hip_offset.json')
    ap.add_argument('--real_ann', default=osp.join(PROJECT, 'data/syrip_fixed/annotations/person_keypoints_train_infant.json'))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sm = SMILNumpy()
    hip_offset = None if args.raw_hips else load_hip_offset()
    print('hip labels:', 'raw SMIL joints' if hip_offset is None else f'calibrated offset {np.round(hip_offset, 4)} m')
    vt, ft = load_template_uv()
    # (OBJ UVs are used as-is: checked visually, face + diaper land in the right place)
    part_of_vert = sm.weights.argmax(1)
    face_part = np.array([np.bincount(part_of_vert[f], minlength=24).argmax() for f in sm.f])
    pr = np.load(osp.join(ROOT, 'priors', 'smil_pose_prior_np.npz'))
    prior_std = np.sqrt(np.diag(np.linalg.inv(pr['prec'] @ pr['prec'].T)))
    pose_sigma = np.minimum(0.3 * prior_std, 0.08)

    txt_files = sorted(glob.glob(osp.join(args.txt_folder, '**', '*.png'), recursive=True))
    textures = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in txt_files]
    bg_files = sorted(glob.glob(osp.join(args.bg_folder, '*.jpg')))
    assert textures and bg_files, 'no textures / backgrounds found'

    fits = []
    for ff in args.fit_folders:
        q = json.load(open(osp.join(ff, 'fit_quality.json')))['per_image']
        for p in sorted(glob.glob(osp.join(ff, 'results', '*', '000.pkl'))):
            fid = osp.basename(osp.dirname(p))
            if fid in q and q[fid]['mean_rel'] <= args.max_fit_err:
                fits.append((fid, p))
    print(f'{len(fits)} fits pass max_fit_err={args.max_fit_err}; {args.variants} variants each')

    real = json.load(open(args.real_ann))
    img_dir = osp.join(args.out_root, 'images', args.image_set)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(osp.join(args.out_root, 'annotations'), exist_ok=True)
    rend = Renderer(W, H)
    images, anns, meta = [], [], []
    next_id = args.start_id
    t0 = time.time()
    n_fail = 0
    for fid, p in fits:
        fit = load_fit(p)
        for k in range(args.variants):
            betas = np.clip(np.ravel(fit['betas']) + rng.normal(0, 0.25, 20), -3, 3)
            bp = np.ravel(fit['body_pose']) + rng.normal(0, 1, 69) * pose_sigma
            verts, joints = sm(betas, fit['global_orient'], bp)
            joints = coco17_points(joints, fit['global_orient'], hip_offset)  # (17,3) COCO order
            # body in the fitted camera frame (fitted camera rotation is identity)
            Rc = fit['camera_rotation'][0]
            verts, joints = verts @ Rc.T, joints @ Rc.T
            c = verts.mean(0)
            ang = (rng.uniform(-40, 40), rng.uniform(-20, 20), rng.uniform(-20, 20))
            Rx = rot('z', ang[0]) @ rot('x', ang[1]) @ rot('y', ang[2])
            verts, joints = (verts - c) @ Rx.T, (joints - c) @ Rx.T
            focal = rng.uniform(500, 1000)
            t = place_camera(verts, rng, focal)
            if t is None:
                n_fail += 1
                continue
            vc, jc = verts + t, joints + t
            lights = sample_lights(rng)
            ti = int(rng.integers(len(textures)))
            tex = jitter_texture(textures[ti], rng)
            rgb, mask, depth = rend.render(vc, sm.f, focal, (W / 2, H / 2), texture=tex, vt=vt, ft=ft,
                                           lights=lights, ambient=rng.uniform(0.25, 0.5))
            labels = render_labels(rend, vc, sm.f, face_part, focal, (W / 2, H / 2))
            kp_uv, kp_z = project(jc, focal=focal, center=(W / 2, H / 2))
            if (kp_uv < 0).any() or (kp_uv[:, 0] > W - 1).any() or (kp_uv[:, 1] > H - 1).any() or mask.sum() < 500:
                n_fail += 1
                continue
            vis = visibility(kp_uv, kp_z, depth, labels)
            bi = int(rng.integers(len(bg_files)))
            bg = cv2.imread(bg_files[bi])
            if rng.random() < 0.5:
                bg = cv2.flip(bg, int(rng.integers(-1, 2)))
            img = composite(rgb[..., ::-1], mask, bg, rng, lights[0]['dir'])
            ys, xs = np.nonzero(mask)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            bbox = [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]
            fn = f'train{next_id:05d}.jpg'
            cv2.imwrite(osp.join(img_dir, fn), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            kp = np.concatenate([kp_uv, vis[:, None]], 1).ravel().tolist()
            images.append({'file_name': fn, 'is_synthetic': True, 'frame_id': 0, 'height': H, 'id': next_id,
                           'is_labeled': True, 'nframes': 1, 'original_file_name': fn, 'width': W,
                           'image_source': f'smil_render:{fid}.jpg'})
            anns.append({'bbox': bbox, 'bbox_head': [], 'category_id': 1, 'id': next_id, 'image_id': next_id,
                         'keypoints': kp, 'scores': [], 'track_id': 0, 'num_keypoints': 17,
                         'segmentation': [], 'area': float(mask.sum()), 'iscrowd': 0})
            meta.append({'id': next_id, 'source_fit': p, 'variant': k, 'focal': focal, 'rot_zxy_deg': ang,
                         'texture': osp.basename(txt_files[ti]), 'background': osp.basename(bg_files[bi]),
                         'cam_t': t.tolist(), 'kp_cam': jc.tolist()})
            next_id += 1
    out = {'images': images, 'annotations': anns, 'categories': real['categories']}
    ann_fn = osp.join(args.out_root, 'annotations', f'person_keypoints_{args.image_set}.json')
    json.dump(out, open(ann_fn, 'w'))
    json.dump(meta, open(osp.join(args.out_root, 'render_meta.json'), 'w'))
    dt = time.time() - t0
    print(f'rendered {len(images)} images ({n_fail} view samples rejected) in {dt:.0f}s '
          f'({1000 * dt / max(1, len(images)):.0f} ms/image) -> {ann_fn}')


if __name__ == '__main__':
    main()
