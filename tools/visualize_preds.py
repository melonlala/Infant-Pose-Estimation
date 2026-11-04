"""Part 4/5 visualisation: GT vs prediction on a validation set, per-image OKS and per-keypoint error.

python tools/visualize_preds.py --models E0=models/hrnet_fidip.pth E1b=models/E1b/model_best.pth \
    [--ann data/syrip/annotations/person_keypoints_validate_infant.json --images data/syrip/images/validate_infant]
Writes figures/viz/<name>_worst.png, <name>_best.png, figures/viz/per_keypoint.png and prints COCO AP
(recomputed with pycocotools, same GT boxes + flip test) so the numbers can be checked against tools/test_adaptive_model.py.
"""
import argparse, contextlib, io, json, os, sys

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fidip_infer import KEYPOINTS, SIGMAS, draw, load_model, oks, predict  # noqa: E402

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def run(model, cfg, coco, img_dir, device):
    rows = []
    for ann in coco.loadAnns(coco.getAnnIds()):
        info = coco.loadImgs(ann['image_id'])[0]
        img = cv2.imread(os.path.join(img_dir, info['file_name']))
        x, y, w, h = ann['bbox']                           # clip box to image like syrip.py (clean_bbox)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(info['width'] - 1, x1 + max(0, w - 1)), min(info['height'] - 1, y1 + max(0, h - 1))
        kps, sc = predict(model, cfg, img, (x1, y1, x2 - x1, y2 - y1), flip=True, device=device)
        gt = np.array(ann['keypoints'], float).reshape(-1, 3)
        rows.append(dict(image_id=ann['image_id'], file=info['file_name'], kps=kps, scores=sc, gt=gt[:, :2],
                         vis=gt[:, 2], area=ann['area'], oks=oks(kps, gt[:, :2], gt[:, 2], ann['area'])))
    return rows


def coco_ap(coco, rows, in_vis_thre=0.2):
    res = []
    for r in rows:
        m = r['scores'] > in_vis_thre
        s = float(r['scores'][m].mean()) if m.any() else 0.0  # same rescoring as lib/dataset (box score = 1)
        k = np.concatenate([r['kps'], r['scores'][:, None]], 1).reshape(-1).tolist()
        res.append(dict(image_id=r['image_id'], category_id=1, keypoints=k, score=s))
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(coco, coco.loadRes(res), 'keypoints'); E.params.useSegm = None
        E.evaluate(); E.accumulate(); E.summarize()
    return E.stats  # AP, AP50, AP75, APm, APl, AR, AR50, AR75, ARm, ARl


def grid(rows, img_dir, title, out, n=8):
    fig, axs = plt.subplots(n // 4, 8, figsize=(22, 2.9 * (n // 4)), facecolor=SURF)
    for i, r in enumerate(rows[:n]):
        img = cv2.imread(os.path.join(img_dir, r['file']))
        vis = np.where(r['vis'] > 0, 1.0, 0.0)
        for j, (tag, im) in enumerate([('GT', draw(img, r['gt'], vis, 0.5)), ('pred', draw(img, r['kps'], r['scores'], 0.3))]):
            ax = axs.flat[2 * i + j]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            ax.set_title(f"{r['file']} {tag}" + (f"  OKS {r['oks']:.2f}" if tag == 'pred' else ''), fontsize=8, color=INK)
    fig.suptitle(title + '   (blue = left side, orange = right side, aqua = centre; pred hides conf < 0.3)',
                 x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=110, facecolor=SURF); plt.close(fig)


def per_kpt_error(rows):
    """Mean per-keypoint OKS similarity (visible joints) - 1.0 is perfect; shows which joints fail."""
    sims = np.full((len(rows), 17), np.nan)
    for i, r in enumerate(rows):
        d2 = ((r['kps'] - r['gt']) ** 2).sum(1)
        e = np.exp(-d2 / (2 * (r['area'] + np.spacing(1)) * (2 * SIGMAS) ** 2))
        sims[i, r['vis'] > 0] = e[r['vis'] > 0]
    return np.nanmean(sims, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml')
    p.add_argument('--models', nargs='+', required=True, help='name=path.pth ...')
    p.add_argument('--ann', default='data/syrip/annotations/person_keypoints_validate_infant.json')
    p.add_argument('--images', default='data/syrip/images/validate_infant')
    p.add_argument('--out', default='figures/viz'); p.add_argument('--n', type=int, default=8)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(a.ann)
    per_kpt, summary = {}, {}
    for spec in a.models:
        name, path = spec.split('=', 1)
        model, cfg = load_model(a.cfg, path, a.device)
        rows = run(model, cfg, coco, a.images, a.device)
        st = coco_ap(coco, rows); summary[name] = st
        rows.sort(key=lambda r: r['oks'])
        grid(rows, a.images, f'{name}: {a.n} worst images by OKS (Test set: {os.path.basename(a.ann)})', f'{a.out}/{name}_worst.png', a.n)
        grid(rows[::-1], a.images, f'{name}: {a.n} best images by OKS', f'{a.out}/{name}_best.png', a.n)
        per_kpt[name] = per_kpt_error(rows)
        json.dump([dict(file=r['file'], oks=r['oks']) for r in rows], open(f'{a.out}/{name}_per_image_oks.json', 'w'), indent=1)
        print(f"{name}: AP {st[0]:.3f} AP50 {st[1]:.3f} AP75 {st[2]:.3f} AR {st[5]:.3f} | mean OKS {np.nanmean([r['oks'] for r in rows]):.3f}"
              f" | worst: " + ', '.join(f"{r['file']}={r['oks']:.2f}" for r in rows[:5]))
        del model; torch.cuda.empty_cache()

    # per-keypoint comparison (grouped horizontal bars, one axis)
    names = list(per_kpt); y = np.arange(17); hgt = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=SURF); ax.set_facecolor(SURF)
    for i, n in enumerate(names):
        ax.barh(y + (i - (len(names) - 1) / 2) * hgt, per_kpt[n], height=hgt * 0.9, color=C[i], label=n)
    ax.set_yticks(y, KEYPOINTS, fontsize=8.5, color=INK2); ax.invert_yaxis()
    lo = min(v.min() for v in per_kpt.values()); ax.set_xlim(max(0, lo - 0.05), 1.0)
    ax.set_xlabel('mean per-keypoint OKS similarity on visible joints (1 = perfect)', fontsize=9, color=INK2)
    ax.set_title('Which joints are hardest? (Test100)', loc='left', fontsize=11, color=INK)
    ax.grid(axis='x', color=GRID); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=len(names), borderaxespad=0.2)
    fig.tight_layout(); fig.savefig(f'{a.out}/per_keypoint.png', dpi=140, facecolor=SURF)
    print('per-keypoint similarity:'); print('  ' + ' '.join(f'{n:>6}' for n in names))
    for j, k in enumerate(KEYPOINTS):
        print(f"  {k:<15}" + ' '.join(f'{per_kpt[n][j]:6.3f}' for n in names))


if __name__ == '__main__':
    main()
