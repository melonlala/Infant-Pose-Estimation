"""Overlay several FiDIP runs (Part 4.5): train loss, val loss, val AP per epoch.

Usage: python tools/plot_compare.py NAME=<tb_run_dir> [NAME=<tb_run_dir> ...] [--out figures/compare_4_5.png]
  e.g. E1b=log/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant_2026-11-03-19-28 \
       D1=log/D1/syrip/adaptive_pose_hrnet/<ts> F3=... F4=...
Val = the training-loop validation on Test100. Prints the per-run summary used in docs/part4_results_and_fitting.md.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_curves import C, INK2, SURF, load, plt, style  # noqa: E402

LS = ['-', '-', '-', (0, (4, 2))]


def per_epoch(d, n_ep):
    x, y = d['train_loss_P']
    spe = x.max() / n_ep
    x = x / spe
    return np.array([y[(x > e) & (x <= e + 1)].mean() for e in range(n_ep)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out', default='figures/compare_4_5.png')
    a = ap.parse_args()
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), facecolor=SURF)
    style(axs[0], 'Train pose loss (epoch mean)', 'heatmap MSE')
    style(axs[1], 'Val pose loss (Test100)', 'heatmap MSE')
    style(axs[2], 'Val AP (Test100)', 'AP')
    print('run   | epochs | best AP @ep | final AP | train loss first->last | val loss first->min->last | last val/train')
    for i, spec in enumerate(a.runs):
        name, _, path = spec.partition('=')
        path = sorted(glob.glob(path))[-1]
        d = load(path)
        vl, vap = d['valid_loss'][1], d['valid_AP'][1]
        n = len(vl)
        tl = per_epoch(d, n)
        ep = np.arange(1, n + 1)
        kw = dict(color=C[i % len(C)], lw=2, ls=LS[i % len(LS)], label=name)
        axs[0].plot(ep, tl, **kw)
        axs[1].plot(ep, vl, **kw)
        axs[2].plot(ep, vap, **kw)
        b = int(np.argmax(vap))
        axs[2].plot(b + 1, vap[b], 'o', ms=5, color=C[i % len(C)], mec=SURF, mew=1.2)
        print(f'{name:5s} | {n:6d} | {vap[b]:.3f} @{b + 1:<3d} | {vap[-1]:.3f}    | '
              f'{tl[0]:.2e}->{tl[-1]:.2e}       | {vl[0]:.2e}->{vl.min():.2e}->{vl[-1]:.2e} | {vl[-1] / tl[-1]:.2f}')
    for ax in axs:
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
        ax.set_xticklabels(['1', '2', '5', '10', '20', '50', '100'])
        ax.legend(frameon=False, fontsize=8)
    axs[2].annotate('dot = best epoch', (0.02, 0.03), xycoords='axes fraction', fontsize=7.5, color=INK2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor=SURF)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
