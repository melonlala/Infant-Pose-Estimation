"""Part 6.1: run FiDIP pose estimation on a single image and save an annotated copy.

python tools/demo.py --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml \
    --image temp_uploads/own00.jpg --model models/F4/final_state.pth --output figures/demo/own00_fidip.jpg
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fidip_infer import KEYPOINTS, detect_people, draw, load_model, predict  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cfg', default='experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml')
    p.add_argument('--image', required=True)
    p.add_argument('--model', default='models/hrnet_fidip.pth')
    p.add_argument('--output', default='demo_output.jpg')
    p.add_argument('--conf-threshold', type=float, default=0.3, help='hide keypoints below this confidence')
    p.add_argument('--output-size', default='512,512', help='W,H box the output is fitted into (aspect kept)')
    p.add_argument('--bbox', default=None, help='x,y,w,h of the infant (overrides --box-mode)')
    p.add_argument('--box-mode', choices=['detect', 'full'], default='detect',
                   help='detect = largest person found by Faster R-CNN (falls back to the whole image); full = whole image')
    p.add_argument('--person', type=int, default=0, help='with --box-mode detect: which detected person (0 = largest)')
    p.add_argument('--no-flip', action='store_true', help='disable flip-test averaging')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()

    img = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f'cannot read image {a.image}')
    box, src = None, 'whole image'
    if a.bbox:
        box, src = tuple(float(v) for v in a.bbox.split(',')), '--bbox'
    elif a.box_mode == 'detect':
        people = detect_people(img, a.device)
        if people:
            k = min(a.person, len(people) - 1)
            box, src = people[k][:4], f'detected person #{k} (score {people[k][4]:.2f}, {len(people)} found)'
    model, cfg = load_model(a.cfg, a.model, a.device)
    kps, scores = predict(model, cfg, img, box, flip=not a.no_flip, device=a.device)

    out = draw(img, kps, scores, a.conf_threshold)
    if box is not None:
        x, y, bw, bh = (int(v) for v in box)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (235, 235, 235), max(1, int(max(img.shape[:2]) / 400)))
    W, H = (int(v) for v in a.output_size.split(','))
    f = min(W / out.shape[1], H / out.shape[0])
    out = cv2.resize(out, (max(1, int(out.shape[1] * f)), max(1, int(out.shape[0] * f))), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    cv2.imwrite(a.output, out)

    print(f'{"keypoint":<16}{"x":>8}{"y":>8}{"conf":>7}')
    for n, (x, y), s in zip(KEYPOINTS, kps, scores):
        print(f'{n:<16}{x:8.1f}{y:8.1f}{s:7.3f}{"" if s >= a.conf_threshold else "  (hidden)"}')
    print(f'box: {src}; mean confidence {np.mean(scores):.3f}; {int((scores >= a.conf_threshold).sum())}/17 shown -> {a.output}')


if __name__ == '__main__':
    main()
