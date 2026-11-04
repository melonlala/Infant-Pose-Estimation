"""Run smplifyx/main.py over a data folder in K parallel shards (all on the same GPU).

    cd syn_generation
    CUDA_VISIBLE_DEVICES=1 ../.venv-smil/bin/python tools/fit_parallel.py \
        --data_folder data_real --output_folder output_real --jobs 16 [--device cpu] [--limit 20]

Images that already have <output_folder>/results/<id>/000.pkl are skipped (resumable).
Each shard is a folder of symlinks under <output_folder>/_shards/k; logs in <output_folder>/logs.
"""
import argparse
import os
import os.path as osp
import subprocess
import sys
import time

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_folder', required=True)
    ap.add_argument('--output_folder', required=True)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--config', default='cfg_files/fit_smil.yaml')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                    help='cpu: one thread per job, scales with cores (a batch-1 float64 fit is latency-bound; '
                         '10 jobs on one GPU were no faster than 1)')
    ap.add_argument('--threads', type=int, default=1, help='torch/OMP threads per job')
    args = ap.parse_args()

    imgs = sorted(f for f in os.listdir(osp.join(args.data_folder, 'images')) if f.endswith(('.jpg', '.png')))
    todo = [f for f in imgs if not osp.exists(osp.join(args.output_folder, 'results', osp.splitext(f)[0], '000.pkl'))]
    if args.limit:
        todo = todo[:args.limit]
    print(f'{len(imgs)} images, {len(todo)} to fit, {args.jobs} jobs')
    os.makedirs(osp.join(args.output_folder, 'logs'), exist_ok=True)
    procs = []
    t0 = time.time()
    for k in range(args.jobs):
        part = todo[k::args.jobs]
        if not part:
            continue
        sd = osp.join(args.output_folder, '_shards', str(k))
        for sub in ('images', 'keypoints'):
            os.makedirs(osp.join(sd, sub), exist_ok=True)
            for f in os.listdir(osp.join(sd, sub)):
                os.remove(osp.join(sd, sub, f))
        for f in part:
            stem = osp.splitext(f)[0]
            os.symlink(osp.abspath(osp.join(args.data_folder, 'images', f)), osp.join(sd, 'images', f))
            os.symlink(osp.abspath(osp.join(args.data_folder, 'keypoints', stem + '_keypoints.json')),
                       osp.join(sd, 'keypoints', stem + '_keypoints.json'))
        cmd = [sys.executable, 'smplifyx/main.py', '--config', args.config, '--data_folder', sd,
               '--output_folder', args.output_folder, '--model_folder', 'models', '--visualize', 'False',
               '--interactive', 'True', '--use_cuda', str(args.device == 'cuda')]
        log = open(osp.join(args.output_folder, 'logs', f'shard{k}.log'), 'w')
        env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
        procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env))
    rc = [p.wait() for p in procs]
    done = sum(osp.exists(osp.join(args.output_folder, 'results', osp.splitext(f)[0], '000.pkl')) for f in todo)
    print(f'exit codes {rc}; fitted {done}/{len(todo)} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
