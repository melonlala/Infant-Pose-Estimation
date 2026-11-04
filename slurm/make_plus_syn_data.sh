#!/bin/bash
# data/syrip_fixed_plus522/: data/syrip_fixed + the 522 new SMIL renders from 7.1 (data/syrip_syn_new, ids 20001+).
# train_infant = 200R + 1000S (original) + 522S (new); validate/pre-train are the syrip_fixed ones.
# Needs data/syrip_syn_new on this machine (rsync it from the server: it is git-ignored and not in sync.sh push).
set -euo pipefail
D=data/syrip_fixed_plus522; S=data/syrip_fixed; N=data/syrip_syn_new
test -d $S && test -f $N/annotations/person_keypoints_train_infant.json
mkdir -p $D/annotations $D/images/train_infant
python3 - "$S" "$N" "$D" <<'PY'
import json, sys
s, n, d = sys.argv[1:]
a = json.load(open(f'{s}/annotations/person_keypoints_train_infant.json'))
b = json.load(open(f'{n}/annotations/person_keypoints_train_infant.json'))
assert a['categories'] == b['categories']
assert not {i['id'] for i in a['images']} & {i['id'] for i in b['images']}
assert not {x['id'] for x in a['annotations']} & {x['id'] for x in b['annotations']}
a['images'] += b['images']; a['annotations'] += b['annotations']
json.dump(a, open(f'{d}/annotations/person_keypoints_train_infant.json', 'w'))
print('train images', len(a['images']), 'synthetic', sum(bool(i.get('is_synthetic')) for i in a['images']))
PY
for a in train_pre_infant validate_infant; do ln -sfn "$(readlink -f $S/annotations/person_keypoints_$a.json)" $D/annotations/person_keypoints_$a.json; done
for i in train_pre_infant validate_infant; do ln -sfn "$(readlink -f $S/images/$i)" $D/images/$i; done
for src in "$(readlink -f $S/images/train_infant)" "$(readlink -f $N/images/train_infant)"; do
  for f in "$src"/*.jpg; do ln -sfn "$f" $D/images/train_infant/; done
done
echo "train_infant files: $(ls $D/images/train_infant | wc -l)"
