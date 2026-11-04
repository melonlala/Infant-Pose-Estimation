"""Test100 vs Test96 AP from saved final-epoch prediction JSONs.

Test96 = Test100 minus test20/42/46/50, which duplicate annotated training images.
Usage (server, .venv-demo): python tools/eval_test96.py D1 F4 H1 ...  (run names under output/)
"""
import json, sys, glob, io, contextlib, copy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
ann = json.load(open('data/syrip_fixed/annotations/person_keypoints_validate_infant.json'))
leak = {'test20.jpg','test42.jpg','test46.jpg','test50.jpg'}
leak_ids = {im['id'] for im in ann['images'] if im['file_name'] in leak}
assert len(leak_ids)==4, leak_ids
def ap(gt_dict, dets):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(); gt.dataset = gt_dict; gt.createIndex()
        dt = gt.loadRes(dets) if dets else None
        e = COCOeval(gt, dt, 'keypoints'); e.evaluate(); e.accumulate(); e.summarize()
    return e.stats[0]
g96 = copy.deepcopy(ann)
g96['images'] = [i for i in ann['images'] if i['id'] not in leak_ids]
g96['annotations'] = [a for a in ann['annotations'] if a['image_id'] not in leak_ids]
for run in sys.argv[1:]:
    fs = glob.glob(f'output/{run}/*/*/*/results/keypoints_validate_infant_results_0.json')
    if not fs: print(run, 'no json'); continue
    d = json.load(open(fs[0]))
    d96 = [x for x in d if x['image_id'] not in leak_ids]
    print(f'{run:8s} Test100={ap(ann,d):.4f}  Test96={ap(g96,d96):.4f}')
