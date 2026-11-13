#!/usr/bin/env python3
"""Figure of one held-out label site at 2.4 um: CT along the surface normal, the training label, and the outputs of
the D-label net and the J-label net trained without this segment, with each net's ink mass at p >= 0.5 at the site.
No per-site L is shown: at the picked site J has almost no ink at p >= 0.5 (VES-3977 audit defect 3), so the
off-sheet share is given for the whole held-out segment only.

  python l0/figure/site_figure.py [--fold w023] [--u 640] [--out l0/figure/site_w023.png]

Everything comes from stored files (LABELJ_DATA, the stored asset): the CT is the stored replay tile of the site
(ves3951/replay_l0.npz), the outputs are the scored outputs (ves3416/out/out_<net>_h<fold>.npz, mean of the 2 grid
phases, as the scorer), the points and sheet spacings are points.npz. Nothing is rerun. The panels show the centre row
of the site's 17 x 17 point grid (points 8 px apart; CT at full resolution between them). L = share of the predicted
ink (p >= 0.5) more than half a sheet spacing from the labelled sheet, over all defined points of the site
(run_t4.leak_sums), next to L0 = the same ink spread evenly over the column (no depth information).
Writes the PNG and a JSON with every number in the caption.
"""
import argparse
import textwrap
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
L0DIR = HERE.parent
ROOT = L0DIR.parent
VD = Path(os.environ.get('LABELJ_DATA', ROOT / 'data'))
os.environ.setdefault('VES2578_RUN', str(VD / 'ves3416/run'))
sys.path.insert(0, str(L0DIR / 'f485' / 'step1'))
import common as C  # noqa: E402

UM = 2.403   # um per level-0 voxel (scan 20251217075048 level 0; F-567 scope)

def thou(x):
    """integer with a space as thousands separator"""
    return f'{x:,.0f}'.replace(',', ' ')


def site_L(p, d, t4):
    k = d['dfn']
    s = t4.leak_sums(p[k].astype(np.float64), d['pm'][k].astype(np.float64), d['pp'][k].astype(np.float64))
    den = s['den'].sum()
    return dict(L=float(s['num'].sum() / den), L0=float(s['num0'].sum() / den), ink_mass=float(den),
                n_points=int(k.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', default='w023')
    ap.add_argument('--u', type=int, default=640)
    ap.add_argument('--out', default=str(HERE / 'site_w023.png'))
    a = ap.parse_args()
    t4 = C.f389()
    keys = json.loads((VD / 'ves3416/run/prep.json').read_text())['site_keys']
    key = keys[a.u]
    assert key[0] == a.fold and key[1] == 'label', key
    P = np.load(VD / 'ves3416/run/points.npz')
    d = {k: P[f'{k}_{a.u}'] for k in ('rr', 'cc', 'use', 'pm', 'pp', 'dfn')}
    nets = {'D0': 'label D (ink copied through the slab)', 'J0': 'label J (ink on the sheet only)'}
    out = {n: np.load(VD / f'ves3416/out/out_{n}_h{a.fold}.npz')[f'p_{a.u}'].mean(0) for n in nets}
    res = json.loads((L0DIR / 'f567/step2/results.json').read_text())['folds'][a.fold]['runs']
    nums = {n: dict(site=site_L(out[n], d, t4), fold=dict(L=res[n]['label']['L'], L0=res[n]['label']['L0']))
            for n in nets}
    rp = np.load(VD / 'ves3951/replay_l0.npz')
    i = [int(x) for x in rp['u']].index(a.u)
    assert np.array_equal(rp[f'rr_{i}'], d['rr']) and np.array_equal(rp[f'cc_{i}'], d['cc'])
    vol, ra, ca = rp['vol'][i], int(rp['ra'][i]), int(rp['ca'][i])
    # centre row of the 17 x 17 grid
    rows = np.unique(d['rr'])
    r = rows[len(rows) // 2]
    sel = np.flatnonzero(d['rr'] == r)
    sel = sel[np.argsort(d['cc'][sel])]
    cc = d['cc'][sel]
    planes = C.JIT + C.SCORE_K                                    # the 64 scored planes of the k = 0 window
    t = C.TWIN[C.SCORE_K]
    x0, x1 = int(cc.min()) - 4, int(cc.max()) + 4                  # CT columns covering the row's points
    ct = vol[planes][:, r - ra, x0 - ca:x1 - ca + 1].astype(np.float32)
    lo, hi = np.percentile(ct, [1, 99])
    # per-point columns 8 px wide, centred on each point
    def cols(v):                                                  # v: (n_sel, 64) -> (64, x1 - x0 + 1)
        img = np.full((len(t), x1 - x0 + 1), np.nan, np.float32)
        for j, c in enumerate(cc):
            img[:, c - 4 - x0:c + 4 - x0] = v[j][:, None]
        return img
    lk = t4.leak_mask(d['pm'][sel], d['pp'][sel])                  # (n_sel, 64): off-sheet planes
    lab = np.where(d['use'][sel][:, None] & (np.abs(t)[None, :] <= C.SHELL), 1.0, 0.0)
    ext = [0, (x1 - x0 + 1) * UM, t[-1] * UM + UM / 2, t[0] * UM - UM / 2]

    plt.rcParams.update({'font.size': 17, 'figure.facecolor': 'black', 'text.color': 'white',
                         'axes.edgecolor': '#888', 'axes.labelcolor': '#ddd', 'xtick.color': '#bbb',
                         'ytick.color': '#bbb'})
    fig, ax = plt.subplots(2, 2, figsize=(12, 9.0), dpi=100, sharex=True, sharey=True)
    ax = ax.ravel()
    ax[0].imshow(ct, cmap='gray', vmin=lo, vmax=hi, extent=ext, aspect='equal', interpolation='nearest')
    ax[0].set_title('CT along the surface normal', fontsize=19)
    ax[1].imshow(cols(lab), cmap='gray', vmin=0, vmax=1, extent=ext, aspect='equal', interpolation='nearest')
    ax[1].set_title('label: ink here (J trains only this shell)', fontsize=19)
    for k, (n, title) in enumerate(nets.items()):
        A = ax[2 + k]
        A.imshow(cols(out[n]), cmap='magma', vmin=0, vmax=1, extent=ext, aspect='equal', interpolation='nearest')
        A.set_title(title, fontsize=19)
        s = nums[n]['site']
        A.text(0.01, 0.04, f"ink at p ≥ 0.5 here: {thou(s['ink_mass'])}", transform=A.transAxes,
               fontsize=19, color='white', bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=2))
    for A in ax:                                                  # sheet boundary: half a sheet spacing
        for j, c in enumerate(cc):
            xs = np.array([c - 4 - x0, c + 4 - x0]) * UM
            for side, sp in ((-1, d['pm'][sel][j]), (1, d['pp'][sel][j])):
                if np.isfinite(sp) and sp / 2 <= abs(t[0]):
                    A.plot(xs, [side * sp / 2 * UM] * 2, color='#3cf', lw=1.6)
        A.set_facecolor('black')
    for A in ax[2:]:
        A.set_xlabel('along the sheet (µm)')
    for A in ax[::2]:
        A.set_ylabel('depth (µm)')
    fr = json.loads((L0DIR / 'f567/step2/results.json').read_text())['folds'][a.fold]
    auc = {n: fr['runs'][n]['ink_auc'] for n in nets}
    fl, sl = {n: nums[n]['fold'] for n in nets}, {n: nums[n]['site'] for n in nets}
    n_lab = json.loads((L0DIR / 'f567/step2/results.json').read_text())['n_sites'][a.fold]['label']
    cap = (f"PHerc. 1667, scan 20251217075048 at 2.4 µm (level 0). Segment {a.fold} is held out: neither net saw its "
           f"labels.\nOne row of one labelled ink site. Colour: predicted ink probability, 0 (black) to 1 (bright), "
           f"same scale for both nets.\nCyan: half way to the neighbouring sheet; predicted ink (p ≥ 0.5) beyond it "
           f"is off the sheet. At this site J has almost no ink at p ≥ 0.5 ({thou(sl['J0']['ink_mass'])} against D's "
           f"{thou(sl['D0']['ink_mass'])}, {100 * sl['J0']['ink_mass'] / sl['D0']['ink_mass']:.1f} %); its faint band is "
           f"below 0.5 and does not count. Off-sheet share L over all {n_lab} label sites of {a.fold}: "
           f"D {fl['D0']['L']:.2f} (no-depth baseline {fl['D0']['L0']:.2f}), J {fl['J0']['L']:.2f} "
           f"({fl['J0']['L0']:.2f}). "
           f"Ink AUC on {a.fold}: J {auc['J0']:.2f}, D {auc['D0']:.2f}.")
    cap = cap.replace('\n', ' ')
    fig.text(0.012, 0.012, textwrap.fill(cap, 106), fontsize=15, color='#ddd', va='bottom', ha='left',
             linespacing=1.3)
    fig.tight_layout(h_pad=0.6, w_pad=0.6, rect=(0, 0.21, 1, 1))
    fig.savefig(a.out, facecolor='black')
    js = dict(caption=cap, ink_auc_fold=auc, fold=a.fold, u=a.u, site_key=key, row=int(r), n_points_row=int(len(sel)), um_per_vx=UM,
              ct_percentiles_1_99=[float(lo), float(hi)], numbers=nums,
              note='site L over all defined points of the site (kept here, not drawn: J ink mass at the site is far below the '
                   'scorer floor); fold L from l0/f567/step2/results.json')
    Path(a.out).with_suffix('.json').write_text(json.dumps(js, indent=1) + '\n')
    print(json.dumps(nums))


if __name__ == '__main__':
    main()
