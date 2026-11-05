"""Report §5.1: full COCO stats on Test99/Test499f + paired image bootstrap of AP deltas vs the released model.
Usage (server, project root): .venv-demo/bin/python tools/report_stats_bootstrap.py [B=1000]"""
import contextlib, copy, glob, io, json, sys, random
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
REL = 'syrip/*/*/results/keypoints_validate_infant_results_0.json'
def load(p):
    fs = sorted(glob.glob(p)); return json.load(open(fs[0])) if fs else None
def mkcoco(d):
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO(); c.dataset = d; c.createIndex()
    return c
def stats(gt, dets, img_ids=None):
    with contextlib.redirect_stdout(io.StringIO()):
        e = COCOeval(gt, gt.loadRes(dets), 'keypoints')
        if img_ids is not None: e.params.imgIds = img_ids
        e.evaluate(); e.accumulate(); e.summarize()
    return e.stats
t100 = json.load(open('data/syrip_fixed/annotations/person_keypoints_validate_infant.json'))
t500 = json.load(open('data/syrip500_fixed/annotations/person_keypoints_validate_infant.json'))
def drop11(g):
    ids = {i['id'] for i in g['images'] if i['file_name'] == 'test11.jpg'}
    g = copy.deepcopy(g); g['images'] = [i for i in g['images'] if i['id'] not in ids]
    g['annotations'] = [a for a in g['annotations'] if a['image_id'] not in ids]; return g, ids
g99, d99 = drop11(t100); g499, d499 = drop11(t500)
C99, C499 = mkcoco(g99), mkcoco(g499)
runs = {'E0': ('D0_E0', 'E0'), 'D1': ('D1', 'D1'), 'F4': ('F4', 'F4_final'), 'H1': ('H1', 'H1_final'),
        'A2': ('ENS_A2', 'ENS_A2_final'), 'A3': ('ENS_A3', 'ENS_A3_final')}
P = {}
for k, (r100, r500) in runs.items():
    a = [x for x in load(f'output/{r100}/{REL}') if x['image_id'] not in d99]
    b = [x for x in load(f'output_test500f/{r500}/{REL}') if x['image_id'] not in d499]
    P[k] = (a, b)
    s1, s2 = stats(C99, a), stats(C499, b)
    print(f"{k:3s} Test99 AP {s1[0]*100:.1f} AP50 {s1[1]*100:.1f} AP75 {s1[2]*100:.1f} AR {s1[5]*100:.1f} | "
          f"Test499f AP {s2[0]*100:.1f} AP50 {s2[1]*100:.1f} AP75 {s2[2]*100:.1f} AR {s2[5]*100:.1f}")
sys.stdout.flush()
# paired bootstrap over images (resample image ids with replacement -> rebuild gt with duplicated images)
def boot(gt_dict, da, db, B, seed=0):
    rng = random.Random(seed); ids = [i['id'] for i in gt_dict['images']]
    ga = {}; [ga.setdefault(a['image_id'], []).append(a) for a in gt_dict['annotations']]
    pa = {}; [pa.setdefault(x['image_id'], []).append(x) for x in da]
    pb = {}; [pb.setdefault(x['image_id'], []).append(x) for x in db]
    imgs = {i['id']: i for i in gt_dict['images']}; out = []
    for _ in range(B):
        s = [rng.choice(ids) for _ in ids]; g = {'images': [], 'annotations': [], 'categories': gt_dict['categories']}
        A, Bd = [], []; aid = 1
        for n, i in enumerate(s):
            im = dict(imgs[i]); im['id'] = n + 1; g['images'].append(im)
            for a in ga.get(i, []):
                a = dict(a); a['image_id'] = n + 1; a['id'] = aid; aid += 1; g['annotations'].append(a)
            A += [dict(x, image_id=n + 1) for x in pa.get(i, [])]; Bd += [dict(x, image_id=n + 1) for x in pb.get(i, [])]
        c = mkcoco(g); out.append((stats(c, Bd)[0] - stats(c, A)[0]) * 100)
    o = np.array(out); return o.mean(), np.percentile(o, 2.5), np.percentile(o, 97.5), (o <= 0).mean()
B = int(sys.argv[1]) if len(sys.argv) > 1 else 200
for k in ['F4', 'A2']:
    for name, gd, idx in [('Test99', g99, 0), ('Test499f', g499, 1)]:
        m, lo, hi, p = boot(gd, P['E0'][idx], P[k][idx], B)
        print(f"bootstrap {k} - E0 on {name}: mean {m:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  P(delta<=0) {p:.3f}  (B={B})"); sys.stdout.flush()
