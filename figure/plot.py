"""VES-2423 copy of the VES-2577 figure (legend without internal arm names): depth profile of predicted ink at held-out label sites, label J vs D vs the no-depth baseline,
with the raw-CT mean in its own panel. Reads profile_0139.json / profile_1667.json (depth_profile.py).

  python figure/plot.py      -> ../depth_profile.png (the copy the draft uses)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, GREY, INK, MUTED = '#2a78d6', '#eb6834', '#8a8983', '#1f1f1d', '#6b6a63'
SEG = {'0139': 'PHerc. 0139, held-out segment w040', '1667': 'PHerc. 1667, held-out segment w028'}
SCAN = {'0139': 'scan 20250728140407, 9.362 µm, 113 keV', '1667': 'scan 20251217075048 level 2, 9.596 µm, 78 keV'}
MM = 1 / 25.4


def main():
    P = {s: json.loads((HERE / f'profile_{s}.json').read_text()) for s in ('0139', '1667')}
    assert all(p['ok'] for p in P.values()), 'a profile failed P1/P2: figure VOID'
    plt.rcParams.update({'font.size': 7, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(2, 2, figsize=(180 * MM, 120 * MM), dpi=200, sharex='col',
                           gridspec_kw=dict(height_ratios=[1, 2.4], hspace=0.08, wspace=0.22))
    for c, (s, p) in enumerate(P.items()):
        t = np.asarray(p['t'])
        lo, hi = -p['half_pitch_behind'], p['half_pitch_front']
        for a in ax[:, c]:
            a.axvspan(t[0] - 0.5, lo, color=GREY, alpha=0.13, lw=0)
            a.axvspan(hi, t[-1] + 0.5, color=GREY, alpha=0.13, lw=0)
            a.axvline(0, color=MUTED, lw=0.6, ls=':')
            a.set_xlim(t[0] - 0.5, t[-1] + 0.5)
        a = ax[0, c]
        a.plot(t, p['ct_mean'], color=INK, lw=1.5)
        a.set_title(f'{SEG[s]}\n{SCAN[s]}\n{p["n_label_sites"]} label sites, {p["n_points"]} points', fontsize=7,
                    loc='left', color=INK)
        a.set_ylabel('raw CT\nmean (u8)')
        a = ax[1, c]
        for n, col, ls in (('J0', BLUE, '-'), ('J1', BLUE, (0, (3, 1.5))), ('D0', ORANGE, '-'),
                           ('D1', ORANGE, (0, (3, 1.5)))):
            a.plot(t, np.asarray(p['runs'][n]['share']) * 100, color=col, lw=2, ls=ls, marker='o', ms=2.5)
        b = np.asarray(p['baseline']) * 100
        a.plot(t, b, color=INK, lw=1.2, ls=(0, (1, 1.5)), zorder=5)
        a.text(t[-1] + 0.4, b[-1] + 2.2, f'no-depth baseline {b[-1]:.2f} %', ha='right', va='bottom',
               fontsize=6, color=INK)
        rj = [p['runs'][n]['L'] * 100 for n in ('J0', 'J1')]
        rd = [p['runs'][n]['L'] * 100 for n in ('D0', 'D1')]
        r0 = np.mean([p['runs'][n]['L0'] for n in p['runs']]) * 100
        ymax = 33
        a.set_ylim(0, ymax)
        a.text(t[-1] + 0.4, ymax * 0.97, f'ink beyond half a sheet spacing:\nshell label {rj[0]:.1f} / {rj[1]:.1f} %\n'
               f'extruded label {rd[0]:.1f} / {rd[1]:.1f} %\nsame ink spread evenly {r0:.1f} %', va='top', ha='right',
               fontsize=6.5, color=INK)
        a.set_xlabel(f'distance from the labelled sheet (vx of {p["vx_um"]} µm)')
        a.set_ylabel('share of predicted ink (p ≥ 0.5)\nper plane, %')
    h = [plt.Line2D([], [], color=BLUE, lw=2, marker='o', ms=2.5, label='shell label (seed 0)'),
         plt.Line2D([], [], color=BLUE, lw=2, ls=(0, (3, 1.5)), marker='o', ms=2.5, label='shell label (seed 1)'),
         plt.Line2D([], [], color=ORANGE, lw=2, marker='o', ms=2.5, label='extruded label (seed 0)'),
         plt.Line2D([], [], color=ORANGE, lw=2, ls=(0, (3, 1.5)), marker='o', ms=2.5, label='extruded label (seed 1)'),
         plt.Line2D([], [], color=INK, lw=1.2, ls=(0, (1, 1.5)), label='no-depth baseline (same ink, spread evenly)'),
         plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.25, lw=0, label='more than half a sheet spacing away (median)')]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.2)
    fig.savefig(HERE.parent / 'depth_profile.png', facecolor='white')


if __name__ == '__main__':
    main()
