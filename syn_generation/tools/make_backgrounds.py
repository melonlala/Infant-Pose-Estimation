"""Build a local background pool (LSUN is not available on this server).

    cd syn_generation
    ../.venv-smil/bin/python tools/make_backgrounds.py --out render/bg_pool --n_proc 300 --n_crop 300

Two kinds, 640x480 jpg:
  proc_*.jpg  procedural "bedding / floor / mat" textures: stripes, checks, plaid, dots, wood-like
              grain, value noise, in random colours, with a lighting gradient, blur and sensor noise.
  crop_*.jpg  crops of the *real* SyRIP training images taken from regions that do not overlap the
              infant (keypoint box x1.6 + 20% of the image size), resized and Gaussian-blurred (sigma 2-5 px) so
              blanket/floor/carpet appearance is kept but detail is removed. These can still contain
              parts of other people or objects (unlabelled) -- see the limitation note in the docs.
"""
import argparse
import json
import os
import os.path as osp
import sys

import cv2
import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from smil_common import PROJECT  # noqa: E402

W, H = 640, 480


def rand_color(rng, sat=(0.05, 0.6), val=(0.35, 0.98)):
    hsv = np.uint8([[[rng.integers(0, 180), int(255 * rng.uniform(*sat)), int(255 * rng.uniform(*val))]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)


def value_noise(rng, scale):
    small = rng.random((max(2, H // scale), max(2, W // scale))).astype(np.float32)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


def procedural(rng):
    kind = rng.choice(['stripes', 'checks', 'plaid', 'dots', 'grain', 'noise', 'plain'])
    c1, c2 = rand_color(rng), rand_color(rng)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = rng.uniform(0, np.pi)
    u = xx * np.cos(ang) + yy * np.sin(ang)
    v = -xx * np.sin(ang) + yy * np.cos(ang)
    p = rng.uniform(12, 70)
    if kind == 'stripes':
        m = (np.sin(2 * np.pi * u / p) > rng.uniform(-0.5, 0.5)).astype(np.float32)
    elif kind == 'checks':
        m = ((np.floor(u / p) + np.floor(v / p)) % 2).astype(np.float32)
    elif kind == 'plaid':
        m = 0.5 * (np.sin(2 * np.pi * u / p) > 0.3) + 0.5 * (np.sin(2 * np.pi * v / (p * rng.uniform(0.6, 1.6))) > 0.3)
    elif kind == 'dots':
        r = p * rng.uniform(0.12, 0.3)
        m = (((u % p) - p / 2) ** 2 + ((v % p) - p / 2) ** 2 < r * r).astype(np.float32)
    elif kind == 'grain':
        m = 0.5 + 0.5 * np.sin(2 * np.pi * (u / p + 3 * value_noise(rng, 60)))
        m = m * 0.6 + 0.4 * value_noise(rng, 8)
    elif kind == 'noise':
        m = 0.6 * value_noise(rng, int(rng.integers(20, 120))) + 0.4 * value_noise(rng, 6)
    else:
        m = 0.15 * value_noise(rng, 50)
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), rng.uniform(0.5, 2.0))
    img = c1[None, None] * (1 - m[..., None]) + c2[None, None] * m[..., None]
    # fabric micro-texture, illumination gradient, vignetting
    img *= (0.9 + 0.2 * value_noise(rng, 3))[..., None]
    g = rng.uniform(0.6, 1.0) + rng.uniform(0, 0.4) * (xx * np.cos(ang * 2) + yy * np.sin(ang * 2)) / max(W, H)
    img *= np.clip(g, 0.4, 1.3)[..., None]
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def real_crops(rng, n, ann, img_dir):
    d = json.load(open(ann))
    anns = {a['image_id']: a for a in d['annotations']}
    real = [im for im in d['images'] if not im['is_synthetic']]
    out = []
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        im = real[rng.integers(len(real))]
        kp = np.array(anns[im['id']]['keypoints']).reshape(-1, 3)
        kp = kp[kp[:, 2] > 0, :2]
        lo, hi = kp.min(0), kp.max(0)
        c, s = (lo + hi) / 2, (hi - lo) * 1.6 + 0.2 * max(im["width"], im["height"])
        bx0, by0, bx1, by1 = c[0] - s[0] / 2, c[1] - s[1] / 2, c[0] + s[0] / 2, c[1] + s[1] / 2
        cw = int(rng.uniform(0.15, 0.45) * im['width'])
        ch = int(cw * H / W)
        if ch >= im['height']:
            continue
        x0, y0 = rng.integers(0, im['width'] - cw), rng.integers(0, im['height'] - ch)
        if x0 < bx1 and x0 + cw > bx0 and y0 < by1 and y0 + ch > by0:
            continue  # overlaps the infant
        img = cv2.imread(osp.join(img_dir, im['file_name']))
        crop = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_AREA)
        crop = cv2.GaussianBlur(crop, (0, 0), rng.uniform(2, 5))
        out.append((crop, im['file_name']))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='render/bg_pool')
    ap.add_argument('--n_proc', type=int, default=300)
    ap.add_argument('--n_crop', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ann', default=osp.join(PROJECT, 'data/syrip_fixed/annotations/person_keypoints_train_infant.json'))
    ap.add_argument('--img_dir', default=osp.join(PROJECT, 'data/syrip_fixed/images/train_infant'))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.n_proc):
        cv2.imwrite(osp.join(args.out, f'proc_{i:04d}.jpg'), procedural(rng), [cv2.IMWRITE_JPEG_QUALITY, 92])
    crops = real_crops(rng, args.n_crop, args.ann, args.img_dir)
    src = {}
    for i, (c, fn) in enumerate(crops):
        cv2.imwrite(osp.join(args.out, f'crop_{i:04d}.jpg'), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
        src[f'crop_{i:04d}.jpg'] = fn
    json.dump(src, open(osp.join(args.out, 'crop_sources.json'), 'w'), indent=0)
    print(f'{args.n_proc} procedural + {len(crops)} real-crop backgrounds -> {args.out}')


if __name__ == '__main__':
    main()
