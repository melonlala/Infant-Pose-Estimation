"""Part 6.2: interactive FiDIP demo.   streamlit run tools/streamlit_demo.py   (then open http://localhost:8501)

Upload an infant image (or pick one from temp_uploads/ or the validation set), choose a model, adjust the
confidence threshold, and see the predicted skeleton plus a per-keypoint confidence table.
Inference is the same code path as tools/demo.py and the official evaluation (tools/fidip_infer.py).
"""
import glob
import os
import sys

import cv2
import numpy as np
import streamlit as st
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fidip_infer import KEYPOINTS, detect_people, draw, load_model, predict  # noqa: E402

CFG = os.path.join(ROOT, 'experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
UPLOADS = os.path.join(ROOT, 'temp_uploads')


@st.cache_resource(show_spinner='Loading model...')
def get_model(path):
    return load_model(CFG, path, DEVICE)


st.set_page_config(page_title='FiDIP infant pose', layout='wide')
st.title('FiDIP infant pose estimation')

models = sorted(glob.glob(os.path.join(ROOT, 'models', '**', '*.pth'), recursive=True))
models = [m for m in models if os.path.getsize(m) > 50e6 and 'coco' not in m]   # skip adult-only init weights
FINAL = os.path.join(ROOT, 'models', 'F4', 'final_state.pth')                      # final model (MANIFEST.md section 8)
models.sort(key=lambda m: (os.path.abspath(m) != os.path.abspath(FINAL), m))
if not models:
    st.error('No model found under models/. Copy one from the cluster, e.g. models/hrnet_fidip.pth (see SKILL.md).')
    st.stop()

with st.sidebar:
    model_path = st.selectbox('Model', models, format_func=lambda p: os.path.relpath(p, ROOT))
    thr = st.slider('Confidence threshold', 0.0, 1.0, 0.3, 0.05)
    flip = st.checkbox('Flip-test averaging (as in evaluation)', True)
    st.caption(f'device: {DEVICE}')
    st.markdown('**Infant box.** FiDIP is top-down: it estimates the pose inside one box. *Detect* uses a COCO '
                'Faster R-CNN person detector; pick the infant if adults are also in the picture.')
    box_mode = st.radio('Box', ['Detect person', 'Whole image', 'Manual (x, y, w, h)'])

qp = st.query_params                                  # e.g. ?src=temp_uploads/&img=own05.jpg (shareable views)
SOURCES = ['Upload', 'temp_uploads/', 'Validation set']
src = st.radio('Image source', SOURCES, horizontal=True,
               index=SOURCES.index(qp['src']) if qp.get('src') in SOURCES else 0)
img = name = None
if src == 'Upload':
    f = st.file_uploader('Infant image', type=['jpg', 'jpeg', 'png', 'bmp'])
    if f is not None:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR); name = f.name
        os.makedirs(UPLOADS, exist_ok=True)
        dst = os.path.join(UPLOADS, os.path.basename(f.name))
        if not os.path.exists(dst):                      # keep a copy for 6.3; never overwrite an existing file
            with open(dst, 'wb') as fh:
                fh.write(f.getvalue())
else:
    folder = UPLOADS if src == 'temp_uploads/' else os.path.join(ROOT, 'data/syrip_fixed/images/validate_infant')
    files = sorted(glob.glob(os.path.join(folder, '*.jp*g')) + glob.glob(os.path.join(folder, '*.png')))
    if not files:
        st.info(f'No images in {os.path.relpath(folder, ROOT)}.')
    else:
        names = [os.path.basename(f) for f in files]
        p = st.selectbox('Image', files, format_func=os.path.basename,
                         index=names.index(qp['img']) if qp.get('img') in names else 0)
        img = cv2.imread(p, cv2.IMREAD_COLOR); name = os.path.basename(p)

if img is not None:
    h, w = img.shape[:2]
    box = None
    if box_mode.startswith('Manual'):
        c = st.columns(4)
        box = (c[0].number_input('x', 0, w - 1, 0), c[1].number_input('y', 0, h - 1, 0),
               c[2].number_input('w', 1, w, w), c[3].number_input('h', 1, h, h))
    elif box_mode == 'Detect person':
        people = detect_people(img, DEVICE)
        if not people:
            st.warning('No person detected; using the whole image.')
        else:
            k = st.selectbox(f'Detected people ({len(people)}, largest first)', range(len(people)),
                             format_func=lambda i: f'#{i + 1}: score {people[i][4]:.2f}, '
                                                   f'{int(people[i][2])}x{int(people[i][3])} px')
            box = people[k][:4]
    model, cfg = get_model(model_path)
    kps, scores = predict(model, cfg, img, box, flip=flip, device=DEVICE)
    out = draw(img, kps, scores, thr)
    if box is not None:
        x, y, bw, bh = (int(v) for v in box); cv2.rectangle(out, (x, y), (x + bw, y + bh), (235, 235, 235), 2)
    left, right = st.columns([3, 2])
    left.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption=f'{name}: blue = left side, orange = right side',
               width='stretch')
    shown = int((scores >= thr).sum())
    right.metric('Keypoints shown', f'{shown} / 17', help=f'confidence >= {thr:.2f}')
    right.metric('Mean confidence', f'{scores.mean():.3f}')
    right.dataframe({'keypoint': KEYPOINTS, 'x': np.round(kps[:, 0], 1), 'y': np.round(kps[:, 1], 1),
                     'confidence': np.round(scores, 3), 'shown': scores >= thr},
                    hide_index=True, width='stretch', height=640)
    ok, buf = cv2.imencode('.jpg', out)
    right.download_button('Download annotated image', buf.tobytes(), file_name=f'fidip_{os.path.splitext(name)[0]}.jpg')
