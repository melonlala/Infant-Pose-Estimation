"""Part 7.3 application: FiDIP on videos (batch) or a live webcam.

  # one video, or every video in a folder (batch processing)
  python tools/video_infer.py --source clip.mp4 --out figures/video/
  python tools/video_infer.py --source videos/ --out figures/video/
  # live webcam (device 0); press q to quit. Use --no-show on a headless machine (still writes the annotated video).
  python tools/video_infer.py --source 0 --out figures/video/ --show

Per frame: the infant box is taken from the previous frame's confident keypoints (expanded by --box-margin). If there are fewer than
--min-joints confident joints, it falls back to --init-box or the whole frame. Keypoints can be smoothed with an EMA (--smooth).
Outputs per source: <name>_pose.mp4 (annotated), <name>_keypoints.jsonl (one JSON object per frame), <name>_keypoints.csv, and an FPS summary.
"""
import argparse, csv, glob, json, os, sys, time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fidip_infer import KEYPOINTS, draw, load_model, predict  # noqa: E402

VIDEO_EXT = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v')


def box_from_kps(kps, scores, thr, margin, w, h, min_joints):
    ok = scores >= thr
    if ok.sum() < min_joints:
        return None
    x1, y1 = kps[ok].min(0); x2, y2 = kps[ok].max(0)
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * margin, (y2 - y1) * margin
    bw, bh = max(bw, 0.15 * w), max(bh, 0.15 * h)          # never collapse onto a single limb
    x, y = max(0.0, cx - bw / 2), max(0.0, cy - bh / 2)
    return (x, y, min(w - x, bw), min(h - y, bh))


def run(src, model, cfg, a):
    cam = src.isdigit()
    cap = cv2.VideoCapture(int(src) if cam else src)
    if not cap.isOpened():
        print(f'cannot open {src}'); return None
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    name = f'webcam{src}' if cam else os.path.splitext(os.path.basename(src))[0]
    os.makedirs(a.out, exist_ok=True)
    writer = None
    jl = open(os.path.join(a.out, f'{name}_keypoints.jsonl'), 'w')
    cf = open(os.path.join(a.out, f'{name}_keypoints.csv'), 'w', newline='')
    cw = csv.writer(cf); cw.writerow(['frame', 'time_s'] + [f'{k}_{c}' for k in KEYPOINTS for c in ('x', 'y', 'conf')])
    init_box = tuple(float(v) for v in a.init_box.split(',')) if a.init_box else None
    box, prev, n, t_model, t0, tracked = init_box, None, 0, 0.0, time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok or (a.max_frames and n >= a.max_frames):
            break
        h, w = frame.shape[:2]
        t = time.time()
        kps, sc = predict(model, cfg, frame, box, flip=not a.no_flip, device=a.device)
        if a.device == 'cuda':
            torch.cuda.synchronize()
        t_model += time.time() - t
        if a.smooth > 0 and prev is not None:                 # EMA on confident joints only
            m = (sc >= a.thr) & (prev[1] >= a.thr)
            kps[m] = a.smooth * prev[0][m] + (1 - a.smooth) * kps[m]
        prev = (kps.copy(), sc.copy())
        nb = box_from_kps(kps, sc, a.thr, a.box_margin, w, h, a.min_joints)
        tracked += nb is not None
        box = nb if nb is not None else init_box               # lost -> fall back to init box / whole frame
        vis = draw(frame, kps, sc, a.thr)
        if box is not None:
            x, y, bw, bh = (int(v) for v in box); cv2.rectangle(vis, (x, y), (x + bw, y + bh), (200, 200, 200), 1)
        live_fps = (n + 1) / max(1e-6, time.time() - t0)
        cv2.putText(vis, f'{live_fps:4.1f} fps  conf {sc.mean():.2f}', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(os.path.join(a.out, f'{name}_pose.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (w, h))
        writer.write(vis)
        ts = n / fps_in
        jl.write(json.dumps({'frame': n, 'time_s': round(ts, 3), 'box': [round(float(v), 1) for v in box] if box else None,
                             'keypoints': {k: [round(float(x), 1), round(float(y), 1), round(float(s), 3)]
                                           for k, (x, y), s in zip(KEYPOINTS, kps, sc)}}) + '\n')
        cw.writerow([n, round(ts, 3)] + [round(float(v), 3) for x_, y_, s_ in zip(kps[:, 0], kps[:, 1], sc) for v in (x_, y_, s_)])
        if a.show:
            cv2.imshow('FiDIP', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        n += 1
    cap.release(); jl.close(); cf.close()
    if writer: writer.release()
    if a.show: cv2.destroyAllWindows()
    wall = time.time() - t0
    res = dict(source=src, frames=n, model_fps=round(n / max(t_model, 1e-6), 1), end_to_end_fps=round(n / max(wall, 1e-6), 1),
               tracked_frac=round(tracked / max(n, 1), 3), out=os.path.join(a.out, f'{name}_pose.mp4'))
    print(json.dumps(res)); return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True, help='video file, folder of videos, or webcam index (e.g. 0)')
    p.add_argument('--cfg', default='experiments/coco/hrnet/w48_384x288_adam_lr1e-3_infant.yaml')
    p.add_argument('--model', default='models/hrnet_fidip.pth')
    p.add_argument('--out', default='figures/video')
    p.add_argument('--thr', type=float, default=0.3, help='keypoint confidence threshold (draw + tracking)')
    p.add_argument('--init-box', default=None, help='x,y,w,h for the first frame / when tracking is lost (default whole frame)')
    p.add_argument('--box-margin', type=float, default=1.4, help='expand the keypoint hull by this factor for the next box')
    p.add_argument('--min-joints', type=int, default=6, help='confident joints needed to keep tracking')
    p.add_argument('--smooth', type=float, default=0.0, help='EMA weight on the previous frame (0 = off, 0.5 = strong)')
    p.add_argument('--no-flip', action='store_true', help='skip flip test (≈2× faster)')
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--show', action='store_true', help='display a window (webcam / desktop use)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    model, cfg = load_model(a.cfg, a.model, a.device)
    if os.path.isdir(a.source):
        srcs = sorted(f for f in glob.glob(os.path.join(a.source, '*')) if f.lower().endswith(VIDEO_EXT))
        print(f'batch: {len(srcs)} videos in {a.source}')
    else:
        srcs = [a.source]
    results = [r for s in srcs if (r := run(s, model, cfg, a))]
    if len(results) > 1:
        json.dump(results, open(os.path.join(a.out, 'batch_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
