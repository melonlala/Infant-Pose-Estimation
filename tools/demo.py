"""Part 6.1: run FiDIP pose estimation on a single image and save an annotated copy.

python tools/demo.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
    --image data/syrip/images/validate_infant/test0.jpg --model models/hrnet_fidip.pth --output demo_output.jpg
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fidip_infer import KEYPOINTS, draw, load_model, predict  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cfg', default='experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml')
    p.add_argument('--image', required=True)
    p.add_argument('--model', default='models/hrnet_fidip.pth')
    p.add_argument('--output', default='demo_output.jpg')
    p.add_argument('--conf-threshold', type=float, default=0.3, help='hide keypoints below this confidence')
    p.add_argument('--output-size', default='512,512', help='W,H box the output is fitted into (aspect kept)')
    p.add_argument('--bbox', default=None, help='x,y,w,h of the infant; default = whole image')
    p.add_argument('--no-flip', action='store_true', help='disable flip-test averaging')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()

    img = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f'cannot read image {a.image}')
    box = tuple(float(v) for v in a.bbox.split(',')) if a.bbox else None
    model, cfg = load_model(a.cfg, a.model, a.device)
    kps, scores = predict(model, cfg, img, box, flip=not a.no_flip, device=a.device)

    out = draw(img, kps, scores, a.conf_threshold)
    W, H = (int(v) for v in a.output_size.split(','))
    f = min(W / out.shape[1], H / out.shape[0])
    out = cv2.resize(out, (max(1, int(out.shape[1] * f)), max(1, int(out.shape[0] * f))), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    cv2.imwrite(a.output, out)

    print(f'{"keypoint":<16}{"x":>8}{"y":>8}{"conf":>7}')
    for n, (x, y), s in zip(KEYPOINTS, kps, scores):
        print(f'{n:<16}{x:8.1f}{y:8.1f}{s:7.3f}{"" if s >= a.conf_threshold else "  (hidden)"}')
    print(f'mean confidence {np.mean(scores):.3f}; {int((scores >= a.conf_threshold).sum())}/17 shown -> {a.output}')


if __name__ == '__main__':
    main()
