"""Figure: real image + GT | SMIL fit | the synthetic variants rendered from that fit.

    cd syn_generation
    ../.venv-smil/bin/python tools/pipeline_figure.py --fit_folder output_real --data_folder data_real \
        --syn_root ../data/syrip_syn_new --out ../figures/syn/pipeline_examples.jpg --n 4
"""
import argparse
import json
import os.path as osp
import sys

import cv2
import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from check_syn_labels import draw  # noqa: E402
from fit_overlay import overlay_one  # noqa: E402
from smil_common import SMILNumpy, load_fit  # noqa: E402
from smil_render import Renderer  # noqa: E402
from utils import smpl_to_openpose  # noqa: E402


def fit_box(img, w, h):
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    box = np.full((h, w, 3), 255, np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    box[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return cv2.copyMakeBorder(box, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit_folder', required=True)
    ap.add_argument('--data_folder', required=True)
    ap.add_argument('--syn_root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--seed', type=int, default=3)
    args = ap.parse_args()
    d = json.load(open(osp.join(args.syn_root, 'annotations', 'person_keypoints_train_infant.json')))
    by_src = {}
    for im, a in zip(d['images'], d['annotations']):
        by_src.setdefault(im['image_source'].split(':')[1][:-4], []).append((im, a))
    rng = np.random.default_rng(args.seed)
    srcs = sorted(by_src)
    srcs = [srcs[i] for i in sorted(rng.choice(len(srcs), args.n, replace=False))]
    sm, rend = SMILNumpy(), Renderer(64, 64)
    m2o = smpl_to_openpose('smil', use_hands=False, use_face=False, openpose_format='coco19')
    rows = []
    for s in srcs:
        img = cv2.imread(osp.join(args.data_folder, 'images', s + '.jpg'))
        kp = np.array(json.load(open(osp.join(args.data_folder, 'keypoints', s + '_keypoints.json')))['annotations']).reshape(-1, 3)
        pair, _ = overlay_one(img, load_fit(osp.join(args.fit_folder, 'results', s, '000.pkl')), kp, sm, rend, m2o)
        w = pair.shape[1] // 2
        tiles = [fit_box(pair[:, :w], 320, 240), fit_box(pair[:, w:], 320, 240)]
        for im, a in by_src[s]:
            syn = cv2.imread(osp.join(args.syn_root, 'images', 'train_infant', im['file_name']))
            tiles.append(fit_box(draw(syn, np.array(a['keypoints']).reshape(17, 3), a['bbox']), 320, 240))
        rows.append(np.hstack(tiles))
    wmax = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    cv2.imwrite(args.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print('->', args.out, 'sources:', srcs)


if __name__ == '__main__':
    main()
