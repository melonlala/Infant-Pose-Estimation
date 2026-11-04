"""Make a smplifyx data folder (images/ + keypoints/ in OpenPose COCO-19) from the real SyRIP
training images and their GT COCO-17 keypoints. data/syrip_fixed is only read; images are
symlinked.

    cd syn_generation
    ../.venv-smil/bin/python tools/prepare_fit_data.py --out data_real --num 200
"""
import argparse
import json
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from smil_common import PROJECT, coco17_to_op19  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', default=osp.join(PROJECT, 'data/syrip_fixed/annotations/person_keypoints_train_infant.json'))
    ap.add_argument('--img_dir', default=osp.join(PROJECT, 'data/syrip_fixed/images/train_infant'))
    ap.add_argument('--out', default='data_real')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--num', type=int, default=200)
    ap.add_argument('--min_kp', type=int, default=10, help='skip images with fewer labelled keypoints')
    args = ap.parse_args()

    d = json.load(open(args.ann))
    anns = {}
    for a in d['annotations']:
        anns.setdefault(a['image_id'], []).append(a)
    real = sorted([im for im in d['images'] if not im['is_synthetic']], key=lambda im: im['id'])
    print(f'{len(real)} real images in {args.ann}')
    os.makedirs(osp.join(args.out, 'images'), exist_ok=True)
    os.makedirs(osp.join(args.out, 'keypoints'), exist_ok=True)
    n = skipped = 0
    for im in real[args.start:args.start + args.num]:
        a = anns.get(im['id'], [])
        if len(a) != 1 or a[0]['num_keypoints'] < args.min_kp:
            skipped += 1
            continue
        stem = osp.splitext(im['file_name'])[0]
        dst = osp.join(args.out, 'images', stem + '.jpg')
        if not osp.lexists(dst):
            os.symlink(osp.abspath(osp.join(args.img_dir, im['file_name'])), dst)
        op19 = coco17_to_op19(a[0]['keypoints'])
        json.dump({'annotations': op19.ravel().tolist(), 'image_id': im['id'],
                   'source': 'syrip_fixed train_infant GT (COCO-17 -> OpenPose COCO-19)'},
                  open(osp.join(args.out, 'keypoints', stem + '_keypoints.json'), 'w'))
        n += 1
    print(f'wrote {n} images (+keypoints) to {args.out}, skipped {skipped}')


if __name__ == '__main__':
    main()
