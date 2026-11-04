"""Plot FiDIP training curves from a TensorBoard run dir (Part 4.4/4.5 of the student guide).

Usage: python tools/plot_curves.py <tb_run_dir> [--name E1b] [--out figures/]
       e.g. python tools/plot_curves.py log/syrip/adaptive_pose_hrnet/w48_384x288_adam_lr1e-3_infant_2026-11-03-19-28 --name E1b
Produces <out>/<name>_curves.png (4 small multiples, one y-axis each) and prints a per-epoch table.
"""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# validated categorical order (dataviz default palette, light mode): blue, orange, aqua, yellow
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def load(run_dir):
    """Return {tag: (steps, values)}; sub-dir runs like 'valid_AP .75' become their own tags."""
    out = {}
    for f in glob.glob(os.path.join(run_dir, '**', 'events.out.tfevents*'), recursive=True):
        ea = EventAccumulator(f, size_guidance={'scalars': 0}); ea.Reload()
        sub = os.path.relpath(os.path.dirname(f), run_dir)
        for t in ea.Tags()['scalars']:
            ev = ea.Scalars(t)
            key = t if sub == '.' else sub
            out[key] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def style(ax, title, ylabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc='left', fontsize=10, color=INK, pad=6)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
    ax.set_xlabel('epoch', fontsize=8.5, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def label_end(ax, x, y, text, color):
    ax.plot(x[-1], y[-1], 'o', ms=4, color=color, mec=SURF, mew=1.5, zorder=4)
    ax.annotate(text, (x[-1], y[-1]), xytext=(6, 0), textcoords='offset points',
                fontsize=8, color=INK, va='center')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir'); ap.add_argument('--name', default=None); ap.add_argument('--out', default='figures')
    a = ap.parse_args()
    name = a.name or os.path.basename(a.run_dir.rstrip('/'))
    d = load(a.run_dir)
    n_ep = len(d['valid_loss'][1])
    steps_per_ep = d['train_loss_P'][0].max() / n_ep if n_ep else 1
    tr_ep = lambda tag: (d[tag][0] / steps_per_ep)          # train step -> fractional epoch
    ep = np.arange(n_ep)

    # per-epoch mean of the train series (sampled every PRINT_FREQ iters)
    def per_epoch(tag):
        x, y = tr_ep(tag), d[tag][1]
        return np.array([y[(x > e) & (x <= e + 1)].mean() if ((x > e) & (x <= e + 1)).any() else np.nan for e in ep])

    fig, axs = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURF)
    fig.suptitle(f'{name}: FiDIP training on SyRIP (val = Test100)', x=0.01, ha='left', fontsize=12, color=INK)

    # 1. pose loss: train (per-epoch mean, raw samples faint) vs val
    da = d['train_loss_P'][1].min() < 0   # with DA the logged train loss is MSE - lambda*BCE, not pure MSE
    ax = axs[0, 0]; style(ax, 'Pose loss' + (' (train = MSE - λ·BCE, DA on; val = MSE)' if da else ' (heatmap MSE)'), 'loss')
    ax.plot(tr_ep('train_loss_P'), d['train_loss_P'][1], color=C[0], lw=0.8, alpha=0.25)
    tl, vl = per_epoch('train_loss_P'), d['valid_loss'][1]
    ax.plot(ep + 1, tl, color=C[0], lw=2, label='train (epoch mean)')
    ax.plot(ep + 1, vl, color=C[1], lw=2, label='val')
    label_end(ax, ep + 1, tl, 'train', C[0]); label_end(ax, ep + 1, vl, 'val', C[1])
    ax.legend(frameon=False, fontsize=8, loc='best')

    # 2. keypoint accuracy (PCK-style, from the training code) train vs val
    ax = axs[0, 1]; style(ax, 'Keypoint accuracy (PCK@0.5 heatmap)', 'accuracy')
    ta, va = per_epoch('train_acc_P'), d['valid_acc'][1]
    ax.plot(ep + 1, ta, color=C[0], lw=2, label='train (epoch mean)')
    ax.plot(ep + 1, va, color=C[1], lw=2, label='val')
    label_end(ax, ep + 1, ta, 'train', C[0]); label_end(ax, ep + 1, va, 'val', C[1])
    ax.legend(frameon=False, fontsize=8, loc='lower right')

    # 3. val COCO metrics
    ax = axs[1, 0]; style(ax, 'Validation keypoint AP / AR (Test100)', 'score')
    ends = []
    for i, (tag, lab) in enumerate([('valid_AP', 'AP'), ('valid_Ap .5', 'AP50'), ('valid_AP .75', 'AP75'), ('valid_AR', 'AR')]):
        if tag in d:
            y = d[tag][1]; ax.plot(ep + 1, y, color=C[i], lw=2, label=lab, marker='o', ms=3.5, mec=SURF, mew=1)
            ends.append([y[-1], lab, C[i], y])
    # merge end labels that would collide (e.g. AP50 == AP75) into one "AP50 = AP75 0.985" label
    span = max(e[0] for e in ends) - min(e[0] for e in ends) or 1.0
    ends.sort(key=lambda e: -e[0]); groups = []
    for e in ends:
        if groups and abs(groups[-1][-1][0] - e[0]) < 0.04 * span + 1e-3:
            groups[-1].append(e)
        else:
            groups.append([e])
    for g in groups:
        label_end(ax, ep + 1, g[0][3], ' = '.join(e[1] for e in g) + f' {g[0][0]:.3f}', g[0][2])
    b = int(np.argmax(d['valid_AP'][1]))
    ax.annotate(f'best AP {d["valid_AP"][1][b]:.3f} @ epoch {b + 1}', (b + 1, d['valid_AP'][1][b]),
                xytext=(28, -24), textcoords='offset points', ha='left', fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    ax.legend(frameon=False, fontsize=8, loc='lower right', ncol=4)
    ax.set_xlim(0.5, n_ep + 3.5)

    # 4. domain classifier
    ax = axs[1, 1]; style(ax, 'Domain classifier (real vs synthetic)', 'accuracy / BCE loss')
    da, dl = per_epoch('train_acc_D'), per_epoch('train_loss_D')
    ax.plot(ep + 1, da, color=C[2], lw=2, label='accuracy (epoch mean)')
    ax.plot(ep + 1, dl, color=C[3], lw=2, label='BCE loss (epoch mean)')
    ax.axhline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.annotate('chance = 0.5', (1, 0.5), xytext=(2, 3), textcoords='offset points', fontsize=7.5, color=INK2)
    label_end(ax, ep + 1, da, 'acc', C[2]); label_end(ax, ep + 1, dl, 'loss', C[3])
    ax.legend(frameon=False, fontsize=8, loc='best')

    for ax in axs.flat:
        ax.set_xticks(range(0, n_ep + 1, max(1, n_ep // 10)))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f'{name}_curves.png'); fig.savefig(out, dpi=150, facecolor=SURF)
    print('wrote', out)

    print('epoch | train_loss | val_loss | val/train | train_acc | val_acc | AP | acc_D')
    for e in ep:
        print(f'{e + 1:5d} | {tl[e]:.6f} | {vl[e]:.6f} | {vl[e] / tl[e]:.2f} | {ta[e]:.3f} | {va[e]:.3f} | {d["valid_AP"][1][e]:.3f} | {da[e]:.3f}')


if __name__ == '__main__':
    main()
