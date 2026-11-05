"""Rescore saved prediction JSONs on the cleaned test subsets (no GPU, no re-inference).

Subsets (all on the fixed data: data/syrip_fixed, data/syrip500_fixed):
  Test100   = fixed Test100 (correct test11 image)
  Test99    = Test100 minus test11 (GT dropped by the user's decision; it only matches
              the SyRIP.zip copy of the image, and a few joints are loose)
  Test95    = Test99 minus test20/42/46/50 (byte-identical to annotated training images)
  Test500f  = Test500 with area = bbox w*h
  Test499f  = Test500f minus test11

Predictions used:
  Test100-family: output/<run>/.../results/keypoints_validate_infant_results_0.json
                  (last validation of a training run = final_state; for eval-only runs = that model)
  Test500-family: output_test500f/<run>_final/... if present, else output_test500f/<run>/...

Usage (server, project root, .venv-demo):
  python tools/eval_subsets.py [--pred DIR] [--pred500 DIR] D0_E0:E0 D0b_E1b:E1b D1 F1 F4 ...
  Each arg is <output/ name>[:<output_test500f/ name>]. --pred / --pred500 replace the default prediction roots
  output/ and output_test500f/ (e.g. repro/output, used by reproduce/04_eval.sh).
"""
import contextlib
import copy
import glob
import io
import json
import sys

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DROP = {'test11.jpg'}
LEAK = {'test20.jpg', 'test42.jpg', 'test46.jpg', 'test50.jpg'}
REL = 'syrip/*/*/results/keypoints_validate_infant_results_0.json'   # hrnet or mobile


def subset(gt, drop_files):
    ids = {im['id'] for im in gt['images'] if im['file_name'] in drop_files}
    assert len(ids) == len(drop_files), (drop_files, ids)
    g = copy.deepcopy(gt)
    g['images'] = [i for i in gt['images'] if i['id'] not in ids]
    g['annotations'] = [a for a in gt['annotations'] if a['image_id'] not in ids]
    return g, ids


def ap(gt_dict, dets):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dict
        gt.createIndex()
        e = COCOeval(gt, gt.loadRes(dets), 'keypoints')
        e.evaluate()
        e.accumulate()
        e.summarize()
    return e.stats[0] * 100


def load(pattern):
    fs = sorted(glob.glob(pattern))
    return (json.load(open(fs[0])), fs[0]) if fs else (None, None)


ROOTS = {'--pred': 'output', '--pred500': 'output_test500f'}
args = sys.argv[1:]
while args and args[0] in ROOTS:
    ROOTS[args[0]] = args[1].rstrip('/')
    args = args[2:]
PRED, PRED500 = ROOTS['--pred'], ROOTS['--pred500']

t100 = json.load(open('data/syrip_fixed/annotations/person_keypoints_validate_infant.json'))
t500 = json.load(open('data/syrip500_fixed/annotations/person_keypoints_validate_infant.json'))
t99, d99 = subset(t100, DROP)
t95, d95 = subset(t100, DROP | LEAK)
t499, d499 = subset(t500, DROP)
assert (len(t99['images']), len(t95['images']), len(t499['images'])) == (99, 95, 499)

print(f"{'run':10s} {'Test100':>8s} {'Test99':>8s} {'Test95':>8s} {'Test500f':>9s} {'Test499f':>9s}  test500 source")
for arg in args:
    run, _, r500 = arg.partition(':')
    r500 = r500 or run
    row = [f'{run:10s}']
    d, _ = load(f'{PRED}/{run}/{REL}')
    if d is None:
        row += [f'{"—":>8s}'] * 3
    else:
        row += [f'{ap(t100, d):8.1f}',
                f'{ap(t99, [x for x in d if x["image_id"] not in d99]):8.1f}',
                f'{ap(t95, [x for x in d if x["image_id"] not in d95]):8.1f}']
    d5, src = load(f'{PRED500}/{r500}_final/{REL}')
    if d5 is None:
        d5, src = load(f'{PRED500}/{r500}/{REL}')
    if d5 is None:
        row += [f'{"—":>9s}'] * 2
        src = '-'
    else:
        row += [f'{ap(t500, d5):9.1f}', f'{ap(t499, [x for x in d5 if x["image_id"] not in d499]):9.1f}']
        src = src[len(PRED500) + 1:].split('/')[0]
    print(' '.join(row), ' ', src)
