"""Report Fig. 1 (figures/report/ablation_overview.png). Values = tools/eval_subsets.py output, final_state. Run from project root."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
# (label, [(Test99, Test499f) per run], highlight)
rows = [
 ("Released hrnet_fidip.pth (paper model)", [(92.6, 90.7)], 'ref'),
 ("D1  reproduction, 20 ep (D1, F1)", [(92.7, 91.2), (92.5, 91.1)], ''),
 ("+ DA λ=1e-4 (L1)", [(92.8, 91.1)], ''),
 ("+ DA λ=5e-4, paper (F2)", [(92.5, 90.9)], ''),
 ("+ DA λ=2e-3 (L2)", [(91.4, 89.6)], ''),
 ("+ DA λ=5e-3 (L3)", [(88.8, 87.2)], ''),
 ("LR 1e-3, paper (F3)", [(92.0, 90.6)], ''),
 ("LR 5e-5 (G3)", [(92.3, 90.9)], ''),
 ("real-only 200R (G2)", [(93.0, 91.1)], ''),
 ("sampler shuffle (G1, H3)", [(93.0, 91.2), (92.9, 91.3)], ''),
 ("shuffle + strong aug (H2)", [(92.9, 91.5)], ''),
 ("100 ep (F4, H1)  ← final single model", [(93.0, 91.4), (93.0, 91.4)], 'final'),
 ("100 ep + shuffle (S1, S2)", [(93.3, 91.4), (93.2, 91.6)], ''),
 ("100 ep + OHKM (B3a, B3b)", [(93.0, 91.4), (93.1, 91.1)], ''),
 ("100 ep + 522 SMIL renders (SY1, SY2)", [(92.4, 91.5), (92.3, 91.2)], ''),
 ("F4 + TTA, 18 passes (A2)", [(93.4, 92.0)], 'final'),
 ("4-model ensemble + TTA (A3)", [(93.4, 92.0)], ''),
]
BLUE, ORANGE, GRAY, INK, INK2 = '#2a78d6', '#eb6834', '#8a8985', '#0b0b0b', '#52514e'
fig, axes = plt.subplots(1, 2, figsize=(11, 6.4), sharey=True)
y = list(range(len(rows)))[::-1]
for k, (ax, name) in enumerate(zip(axes, ['Test99 AP (fixed Test100 − test11)', 'Test499f AP (Test500, area fixed, − test11)'])):
    ref = rows[0][1][0][k]
    ax.axvline(ref, color=GRAY, lw=1.2, ls='--', zorder=1)
    ax.axvspan(ref - 0.0, ref + 0.3, color=GRAY, alpha=0.10, lw=0, zorder=0)
    for yi, (lab, vals, h) in zip(y, rows):
        c = ORANGE if h == 'final' else (GRAY if h == 'ref' else BLUE)
        xs = [v[k] for v in vals]
        if len(xs) > 1:
            ax.plot([min(xs), max(xs)], [yi, yi], color=c, lw=2, alpha=0.5, zorder=2)
        ax.scatter(xs, [yi] * len(xs), s=46, color=c, edgecolor='white', linewidth=1.5, zorder=3)
    ax.set_title(name, fontsize=10.5, color=INK, loc='left')
    ax.grid(axis='x', color='#e6e5e1', lw=0.8); ax.set_axisbelow(True)
    for s in ['top', 'right', 'left']: ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#c9c8c3'); ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_xlabel('COCO keypoint AP (final checkpoint, GT box, flip test)', fontsize=9, color=INK2)
axes[0].set_xlim(88.4, 93.8); axes[1].set_xlim(86.8, 92.4)
axes[0].set_yticks(y); axes[0].set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
fig.suptitle('Every ablation vs the released FiDIP checkpoint (dot = one training run, pairs joined; dashed = released model, shaded = +0.3 bar)', fontsize=11.5, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig('figures/report/ablation_overview.png', dpi=150, facecolor='white')
