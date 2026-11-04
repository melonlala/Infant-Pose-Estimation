"""Build data/syrip500_fixed/: the Test500 eval root with a valid 'area' field.

SyRIP/annotations/validate500 gives every annotation the same area (2443298). COCOeval uses
the gt area as the OKS scale, so Test500 AP is inflated (~98.4 for every model). This writes
a copy where area = bbox w*h, which is what the Test100 json (syrip/.../validate_infant) uses.
Nothing else changes. Images are a symlink to SyRIP/images/validate_infant.

Run from the project root (server or cluster): python tools/make_syrip500_fixed.py
Eval: sbatch slurm/eval.sh <model> DATASET.ROOT data/syrip500_fixed/ OUTPUT_DIR output_test500f/<run>
"""
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else '.'
src = os.path.join(root, 'data/SyRIP/annotations/validate500/person_keypoints_validate_infant.json')
out = os.path.join(root, 'data/syrip500_fixed')
d = json.load(open(src))
assert len(d['annotations']) == 500, len(d['annotations'])
for a in d['annotations']:
    a['area'] = float(a['bbox'][2] * a['bbox'][3])
os.makedirs(os.path.join(out, 'annotations'), exist_ok=True)
os.makedirs(os.path.join(out, 'images'), exist_ok=True)
with open(os.path.join(out, 'annotations/person_keypoints_validate_infant.json'), 'w') as f:
    json.dump(d, f)
link = os.path.join(out, 'images/validate_infant')
if not os.path.islink(link):
    os.symlink('../../SyRIP/images/validate_infant', link)
n = len(os.listdir(link))
assert n == 500, f'expected 500 images, got {n}'
areas = sorted(a['area'] for a in d['annotations'])
print(f'built {out}: 500 anns, {n} images, area min/median/max = '
      f'{areas[0]:.0f}/{areas[250]:.0f}/{areas[-1]:.0f}')
