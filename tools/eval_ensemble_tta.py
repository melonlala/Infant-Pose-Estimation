"""Model ensemble + test-time augmentation (TTA) on the official eval path (Part 7, A1/A2).

For every GT box (TEST.USE_GT_BBOX) and every transform t = (scale factor, rotation):
  crop with get_affine_transform(c, s*sf, rot)  ->  each model, plus its horizontal flip (flip_back, as validate_adaptive)
  -> heatmap in t's frame -> warp back to the reference frame (sf = 1, rot = 0) -> average over transforms and models
then decode with the same DARK post-processing (get_final_preds) and score with dataset.evaluate. With one model,
--scales 1 --rots 0 this is exactly validate_adaptive, so it must reproduce the official AP (sanity run A0).

Usage (cluster, via slurm/eval_ens.sh):
  python tools/eval_ensemble_tta.py --cfg <yaml> --models a.pth,b.pth --scales 0.9,1.0,1.1 --rots -30,0,30 \
      --out output/ENS_A1/syrip/ens/x DATASET.ROOT data/syrip_fixed/
Writes <out>/results/keypoints_validate_infant_results_0.json (+ prints the COCO AP table).
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _init_paths  # noqa: E402,F401
from config import cfg, update_config  # noqa: E402
from core.inference import get_final_preds  # noqa: E402
from utils.transforms import flip_back, get_affine_transform  # noqa: E402
import dataset  # noqa: E402
import models  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', required=True)
    p.add_argument('--models', required=True, help='comma-separated checkpoint paths')
    p.add_argument('--scales', default='1.0')
    p.add_argument('--rots', default='0')
    p.add_argument('--out', required=True)
    p.add_argument('opts', nargs=argparse.REMAINDER)
    a = p.parse_args()
    a.modelDir = a.logDir = a.dataDir = a.prevModelDir = ''
    return a


def to3(m):
    return np.vstack([m, [0, 0, 1]])


def main():
    args = parse_args()
    update_config(cfg, args)
    scales = [float(x) for x in args.scales.split(',')]
    rots = [float(x) for x in args.rots.split(',')]
    tfs = [(sf, r) for sf in scales for r in rots]
    print('transforms (scale, rot):', tfs, flush=True)

    nets = []
    for f in args.models.split(','):
        m, _ = eval('models.' + cfg.MODEL.NAME + '.get_adaptive_pose_net')(cfg, is_train=False)
        m.load_state_dict(torch.load(f, map_location='cpu'), strict=True)
        nets.append(m.cuda().eval())
        print('loaded', f, flush=True)

    ds = eval('dataset.' + cfg.DATASET.DATASET)(cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False, None)
    norm = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    iw, ih = int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])
    hw, hh = int(cfg.MODEL.HEATMAP_SIZE[0]), int(cfg.MODEL.HEATMAP_SIZE[1])
    nj = cfg.MODEL.NUM_JOINTS

    n = len(ds.db)
    all_preds = np.zeros((n, nj, 3), dtype=np.float32)
    all_boxes = np.zeros((n, 6))
    image_path = []
    with torch.no_grad():
        for i, rec in enumerate(ds.db):
            img = cv2.imread(rec['image'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if cfg.DATASET.COLOR_RGB:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            c, s = rec['center'], rec['scale']
            crops = []
            for sf, r in tfs:
                tr = get_affine_transform(c, s * sf, r, [iw, ih])
                crops.append(norm(cv2.warpAffine(img, tr, (iw, ih), flags=cv2.INTER_LINEAR)))
            x = torch.stack(crops).cuda()
            xf = torch.flip(x, [3])
            hm = np.zeros((len(tfs), nj, hh, hw), dtype=np.float32)
            for m in nets:
                _, o = m(x)
                _, of = m(xf)
                o = o[-1] if isinstance(o, list) else o
                of = of[-1] if isinstance(of, list) else of
                of = flip_back(of.cpu().numpy(), ds.flip_pairs)
                if cfg.TEST.FLIP_TEST:
                    hm += (o.cpu().numpy() + of) * 0.5
                else:
                    hm += o.cpu().numpy()
            hm /= len(nets)
            ref = to3(get_affine_transform(c, s, 0, [hw, hh]))            # image -> reference heatmap
            acc = np.zeros((nj, hh, hw), dtype=np.float32)
            for k, (sf, r) in enumerate(tfs):
                if sf == 1.0 and r == 0:
                    acc += hm[k]
                    continue
                inv = to3(get_affine_transform(c, s * sf, r, [hw, hh], inv=1))  # t heatmap -> image
                A = (ref @ inv)[:2]
                for j in range(nj):
                    acc[j] += cv2.warpAffine(hm[k, j], A, (hw, hh), flags=cv2.INTER_LINEAR)
            acc /= len(tfs)
            preds, maxvals = get_final_preds(cfg, acc[None].copy(), np.array([c]), np.array([s]))
            all_preds[i, :, 0:2] = preds[0, :, 0:2]
            all_preds[i, :, 2:3] = maxvals[0]
            all_boxes[i, 0:2] = c[0:2]
            all_boxes[i, 2:4] = s[0:2]
            all_boxes[i, 4] = np.prod(s * 200)
            all_boxes[i, 5] = rec.get('score', 1)
            image_path.append(rec['image'])
            if i % 50 == 0:
                print(f'{i}/{n}', flush=True)

    os.makedirs(args.out, exist_ok=True)
    name_values, perf = ds.evaluate(cfg, all_preds, args.out, all_boxes, image_path)
    print('AP', perf)
    print(name_values)


if __name__ == '__main__':
    main()
