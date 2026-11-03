"""VES-2577: depth profile of predicted ink at the held-out label sites of one fold, label J vs D, from the stored
F-401 (0139) / F-418 (1667) outputs. CPU, no inference. One scroll per process (the two experiment folders use the
same module names).

  python depth_profile.py --scroll 0139      -> profile_0139.json
  python depth_profile.py --scroll 1667      -> profile_1667.json

Rules (README "1. Depth-profile figure"): fold = median stored L_J (upper median for an even count); per run the
share of label-site p >= 0.5 mass (mean of the 2 phases) by scored plane; P1 replay of L, P2 mass, P3 choice.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # release repo root
SCROLLS = {
    '0139': dict(code=ROOT / 'pherc0139', results='attempt4/results.json', env={},
                 fold='w040', scan='PHerc0139 20250728140407-9.362um-1.2m-113keV-masked', vx_um=9.362, kev=113),
    '1667': dict(code=ROOT / 'pherc1667', results='attempt2/results.json',
                 env={}, fold='w028',
                 scan='PHerc1667 20251217075048-2.399um-0.2m-78keV-masked, level 2', vx_um=9.596, kev=78),
}
RUNS = ('J0', 'J1', 'D0', 'D1')


def pick_fold(res):
    """P3: fold with the median stored L_J; even count -> the higher of the two middle folds."""
    order = sorted(res['folds'], key=lambda f: res['folds'][f]['L_J'])
    return order[len(order) // 2], {f: res['folds'][f]['L_J'] for f in order}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scroll', choices=sorted(SCROLLS), required=True)
    a = ap.parse_args()
    cfg = SCROLLS[a.scroll]
    os.environ.update(cfg['env'])
    sys.argv = [sys.argv[0]]
    spec = importlib.util.spec_from_file_location('score4', cfg['code'] / 'attempt4' / 'score4.py')
    s4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s4)
    C, S, t4 = s4.C, s4.S, s4.t4
    res = json.loads((cfg['code'] / cfg['results']).read_text())
    f, lj = pick_fold(res)
    assert f == cfg['fold'], f'P3 FAIL: rule picks {f}, README names {cfg["fold"]}'

    U = s4.Sites()
    O = s4.Out(U, f)
    lab = [u for u in U.fidx[f] if U.flags[u]]
    old_st = np.load(C.DATA / 'sites.npz')
    old44 = np.load(C.DATA / 'sites44.npz')['vol']
    new_st = np.load(s4.OUT / 'sites4.npz')
    new_pt = np.load(s4.OUT / 'points4.npz')
    planes = C.JIT + C.SCORE_K                        # the 16 scored planes inside the 44-plane render
    t = C.T44[planes]
    assert np.allclose(t, t4.T_PL), (t, t4.T_PL)

    ct_sum, ct_n, pm, pp = np.zeros(len(t)), 0, [], []
    for u in lab:
        kind, i = U.ids[u]
        d = U.pts[u]
        k = d['dfn']
        if kind == 'old':
            v, ra, ca, rr, cc = old44[i], old_st['ra'][i], old_st['ca'][i], old_st[f'rr_{i}'], old_st[f'cc_{i}']
        else:
            v, ra, ca, rr, cc = new_st['vol'][i], new_st['ra'][i], new_st['ca'][i], new_pt[f'rr_{i}'], new_pt[f'cc_{i}']
        assert len(rr) == len(k)
        ct = v[planes][:, rr - ra, cc - ca].T.astype(np.float64)       # (n, 16) raw-CT u8 at the points
        ct_sum += ct[k].sum(0)
        ct_n += int(k.sum())
        pm.append(d['pm'][k])
        pp.append(d['pp'][k])
    pm, pp = np.concatenate(pm), np.concatenate(pp)

    out = dict(scroll=a.scroll, scan=cfg['scan'], vx_um=cfg['vx_um'], kev=cfg['kev'], fold=f, L_J_by_fold=lj,
               n_label_sites=len(lab), n_points=ct_n, t=t.tolist(), ct_mean=(ct_sum / ct_n).tolist(),
               half_pitch_behind=float(np.median(pm[np.isfinite(pm)]) / 2),
               half_pitch_front=float(np.median(pp[np.isfinite(pp)]) / 2), baseline=[1 / len(t)] * len(t), runs={})
    ok = True
    for n in RUNS:
        mass = np.zeros(len(t))
        sums = []
        for u in lab:
            d = U.pts[u]
            p = O.site(n, u).mean(0)                                    # primary: mean of the 2 phases
            q = p[d['dfn']].astype(np.float64)
            mass += (q * (q >= t4.INK)).sum(0)
            sums.append(S.leak_site(p, d))
        L = sum(s['num'] for s in sums) / sum(s['den'] for s in sums)
        L0 = sum(s['num0'] for s in sums) / sum(s['den'] for s in sums)
        st = res['folds'][f]['runs'][n]['label']
        share = mass / mass.sum()
        p1 = abs(L - st['L']) <= 1e-9
        p2 = abs(share.sum() - 1) <= 1e-9 and abs(mass.sum() - st['ink_mass']) <= 1e-6 * st['ink_mass']
        ok &= p1 and p2
        out['runs'][n] = dict(share=share.tolist(), L=L, L0=L0, L_stored=st['L'], L0_stored=st['L0'],
                              ink_mass=float(mass.sum()), ink_mass_stored=st['ink_mass'], P1=bool(p1), P2=bool(p2))
        print(f'{a.scroll} {f} {n}: L {L:.4f} (stored {st["L"]:.4f}) L0 {L0:.4f} mass {mass.sum():.1f} '
              f'(stored {st["ink_mass"]:.1f}) P1 {p1} P2 {p2}', flush=True)
    out['P3'] = True
    out['ok'] = bool(ok)
    (HERE / f'profile_{a.scroll}.json').write_text(json.dumps(out, indent=1))
    print('PASS' if ok else 'VOID', flush=True)


if __name__ == '__main__':
    main()
