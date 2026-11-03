"""VES-2199 test 4c on PHerc0139: does hecate's 3D ink leak onto other sheets?

Criteria: README "Test 4" (committed dd3bd98d before any hecate output). Same sites as tests 2-3 (label and N1
sites of the dev five, seed 2199). Per site a 96 x 96 canvas-px tile, 16 planes at t = k - 7.5 own vx along the
grid normal of the 9.362 um mesh, run through hecate's own predict() (public scrollprize/hecate 9.6 um, fp32,
CPU). Sheets from villa PR #2368 band.bands on the 9.362 um CT profile (vendored, sha-checked). Writes
results_t4.json.

  python run_t4.py --check           scorer check on synthetic mass only
  python run_t4.py [--limit N]       full run (or the first N sites per kind, for a smoke test)
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage, stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # release repo root; data defaults live under ROOT / 'data'
HEC_DIR = Path(os.environ.get('HECATE_DIR', ROOT / 'data/models/hecate'))   # hecate.py + hecate_9.6um.pth
CKPT = HEC_DIR / 'hecate_9.6um.pth'

TILE = 48                   # half tile, canvas px (96 x 96)
NPL = 16
T_PL = np.arange(NPL) - 7.5  # plane offsets, own vx
INK = 0.5
PC_AUC = 0.70
MIN_DEF = 0.50
H_A2 = 24.0                 # attempt 2 profile half-length, own vx


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


t23 = load('ves2199_run_t23', HERE / 'run_t23.py')     # site draws, geometry, profiles, vendored band finder
band = t23.band


# ---------------------------------------------------------------- sheets and leak
def pitches(t, p, rule='a1'):
    """(P-, P+) from the own band (nearest t = 0, must be complete) to the next band on each side.
    a1 (committed dd3bd98d): centre-to-centre to the next complete band; nan if that band is incomplete or missing.
    a2 (attempt 2, registered after the smoke run): centre-to-centre for a complete neighbour; near face - own
    centre (a lower bound, can only raise L) for a neighbour that runs off the profile; inf if no band on that side."""
    bl, _, _ = band.bands(t, p)
    if not bl:
        return math.nan, math.nan
    t0, t1 = float(t[0]), float(t[-1])

    def dist(b):
        lo = b[0] if math.isfinite(b[0]) else t0
        hi = b[1] if math.isfinite(b[1]) else t1
        return 0.0 if lo <= 0.0 <= hi else min(abs(lo), abs(hi))

    k = int(np.argsort([dist(b) for b in bl], kind='stable')[0])
    cen = [0.5 * (a + b) if math.isfinite(a) and math.isfinite(b) else math.nan for a, b in bl]
    if not math.isfinite(cen[k]):
        return math.nan, math.nan
    if rule == 'a1':
        pm = cen[k] - cen[k - 1] if k > 0 else math.nan
        pp = cen[k + 1] - cen[k] if k + 1 < len(bl) else math.nan
        return float(pm), float(pp)
    if k == 0:
        pm = math.inf
    else:
        pm = cen[k] - cen[k - 1] if math.isfinite(cen[k - 1]) else cen[k] - bl[k - 1][1]
    if k + 1 == len(bl):
        pp = math.inf
    else:
        pp = cen[k + 1] - cen[k] if math.isfinite(cen[k + 1]) else bl[k + 1][0] - cen[k]
    return float(pm), float(pp)


def leak_mask(pm, pp):
    """(n, 16) bool: plane t_k is more than half a pitch from the mesh on its side."""
    t = T_PL[None, :]
    return np.where(t > 0, t > pp[:, None] / 2, -t > pm[:, None] / 2)


def leak_sums(prob, pm, pp):
    """Numerators and denominators of L (mass p >= 0.5), L0 (depth-blind) and L-soft (all p), per point."""
    lk = leak_mask(pm, pp)
    m = prob * (prob >= INK)
    col = m.sum(1)
    frac = lk.mean(1)
    return dict(num=(m * lk).sum(1), den=col, num0=col * frac,
                nums=(prob * lk).sum(1), dens=prob.sum(1))


def scorer_check():
    rng = np.random.default_rng(t23.SEED)
    pm, pp = rng.uniform(4, 14, 500), rng.uniform(4, 14, 500)
    k0 = np.argmin(np.abs(T_PL))                     # plane nearest t = 0 (t = -0.5 or +0.5)
    a = np.zeros((500, NPL)); a[:, k0] = 1
    b = np.zeros((500, NPL)); kk = np.argmin(np.abs(T_PL[None, :] - pp[:, None]), 1)
    inwin = pp <= T_PL[-1]
    b[np.arange(500), kk] = 1
    sa, sb = leak_sums(a, pm, pp), leak_sums(b[inwin], pm[inwin], pp[inwin])
    la, lb = sa['num'].sum() / sa['den'].sum(), sb['num'].sum() / sb['den'].sum()
    ok = (la == 0.0) and (lb == 1.0)
    print(f'scorer check: mass at t=0 -> L {la:.3f} (want 0); mass at t=+P+ -> L {lb:.3f} (want 1); '
          f'{"PASS" if ok else "FAIL"}')
    return dict(L_at_0=float(la), L_at_pitch=float(lb), n_in_window=int(inwin.sum()), ok=bool(ok))


# ---------------------------------------------------------------- render + hecate
def render_tile(g, nrm, v_pix, r0, c0):
    H, W = v_pix.shape
    ra, rb, ca, cb = max(r0 - TILE, 0), min(r0 + TILE, H), max(c0 - TILE, 0), min(c0 + TILE, W)
    rr, cc = np.mgrid[ra:rb, ca:cb]
    P, Nn = t23.at_pixels(g, nrm, rr.ravel(), cc.ravel())
    ok = v_pix[ra:rb, ca:cb].ravel() & np.all(np.isfinite(P), 1) & np.all(np.isfinite(Nn), 1)
    P, Nn = np.where(ok[:, None], P, 0.0), np.where(ok[:, None], Nn, 0.0)
    q = P[None] + T_PL[:, None, None] * Nn[None]                    # (16, n, xyz)
    qq = q[:, ok].reshape(-1, 3)
    lo = np.maximum(np.floor(qq.min(0)).astype(int) - 2, 0)
    hi = np.ceil(qq.max(0)).astype(int) + 3
    hi = np.minimum(hi, np.array(t23.vol(t23.VOL9).shape[::-1]))
    sub = t23.read_box(t23.VOL9, lo, hi)
    coords = np.stack([q[..., 2] - lo[2], q[..., 1] - lo[1], q[..., 0] - lo[0]]).reshape(3, -1)
    val = ndimage.map_coordinates(sub.astype(np.float32), coords, order=1, mode='constant', cval=0.0)
    val = val.reshape(NPL, -1) * ok[None]
    volu8 = np.clip(np.rint(val), 0, 255).astype(np.uint8).reshape(NPL, rb - ra, cb - ca)
    return volu8, (ra, ca), ok.reshape(rb - ra, cb - ca)


def run_hecate(hec, model, volu8):
    out = {}
    for rev in (False, True):
        o2 = np.zeros(volu8.shape[1:], np.uint8)
        hec.predict(model, volu8, o2, reverse=rev, batch_size=4)
        o3 = np.zeros(volu8.shape, np.uint8)
        hec.predict(model, volu8, o3, reverse=rev, batch_size=4, three_d=True)
        out[rev] = (o2, o3)
    return out


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return None
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ratio(per_site, a, b, nboot=2000, seed=t23.SEED):
    A = np.array([s[a] for s in per_site]); B = np.array([s[b] for s in per_site])
    if B.sum() <= 0:
        return None, [None, None]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nboot):
        i = rng.integers(0, len(A), len(A))
        if B[i].sum() > 0:
            bs.append(A[i].sum() / B[i].sum())
    return float(A.sum() / B.sum()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--threads', type=int, default=10)
    ap.add_argument('--out', default=str(HERE / 'results_t4.json'))
    a = ap.parse_args()
    chk = scorer_check()
    if a.check or not chk['ok']:
        return
    import torch
    torch.set_num_threads(a.threads)
    hec = load('hecate_ref', HEC_DIR / 'hecate.py')
    model = hec.load_model(str(CKPT), 'cpu')
    ck_sha = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    T0 = time.time()
    sites, pix = [], {False: ([], []), True: ([], [])}
    for wi, w in enumerate(t23.DEV):
        g, v = t23.load_grid(t23.REPO_DATA / 'ves170/tifxyz' / w)
        nrm = t23.grid_normals(g, v)
        lab = np.load(t23.REPO_DATA / 'ves170/labels' / f'{w}.npz')
        pos, neg = lab['pos'], lab['neg']
        valid = t23.pixel_valid(v, pos.shape)
        for si, (sname, m) in enumerate((('label', pos), ('N1_nonlabel', neg))):
            rng = np.random.default_rng([t23.SEED, wi, si])          # the test-2/3 draw, same seeds
            draws = t23.draw_sites(m, valid, rng, t23.N_SITES)
            for k, (r0, c0, rr, cc) in enumerate(draws[:a.limit]):
                if not len(rr):
                    continue
                volu8, (ra, ca), okt = render_tile(g, nrm, valid, r0, c0)
                hp = run_hecate(hec, model, volu8)
                # 2D AUC pixels (input check): every pos / neg pixel of the tile
                tp, tn = pos[ra:ra + okt.shape[0], ca:ca + okt.shape[1]] & okt, \
                    neg[ra:ra + okt.shape[0], ca:ca + okt.shape[1]] & okt
                for rev in (False, True):
                    pix[rev][0].append(hp[rev][0][tp].astype(np.float32) / 255)
                    pix[rev][1].append(hp[rev][0][tn].astype(np.float32) / 255)
                # leak at the site's scored points (mask pixels of the 33 x 33 box, every 2nd px)
                P, Nn = t23.at_pixels(g, nrm, rr, cc)
                okp = np.all(np.isfinite(P), 1) & np.all(np.isfinite(Nn), 1)
                rr, cc, P, Nn = rr[okp], cc[okp], P[okp], Nn[okp]
                t, pr = t23.profiles(t23.VOL9, P, Nn, H_A2)        # +-24 vx; the a1 profile is its +-12 slice
                use = (pr <= 0).mean(1) <= t23.ZERO_MAX
                s12 = np.abs(t) <= t23.H9 + 1e-9
                rec = dict(key=[w, sname, k, int(r0), int(c0)], n=int(len(rr)), n_use=int(use.sum()))
                for att in ('a1', 'a2'):
                    tt, pp_ = (t[s12], pr[:, s12]) if att == 'a1' else (t, pr)
                    pmpp = np.array([pitches(tt, p, att) for p in pp_])
                    pm, pp = pmpp[:, 0], pmpp[:, 1]
                    dfn = use & ~np.isnan(pm) & ~np.isnan(pp)
                    rec[f'n_def_{att}'] = int(dfn.sum())
                    rec[f'n_inwin_{att}'] = int((dfn & ((pm / 2 <= T_PL[-1]) | (pp / 2 <= T_PL[-1]))).sum())
                    fin = np.concatenate([pm[dfn], pp[dfn]])
                    fin = fin[np.isfinite(fin)]
                    rec[f'pitch_med_{att}'] = float(np.median(fin)) if len(fin) else None
                    for rev in (False, True):
                        prob = hp[rev][1][:, rr - ra, cc - ca].T.astype(np.float64) / 255     # (n, 16), t order
                        s = leak_sums(prob[dfn], pm[dfn], pp[dfn])
                        tag = ('rev' if rev else 'fwd') + '_' + att
                        for kk, vv in s.items():
                            rec[f'{kk}_{tag}'] = float(vv.sum())
                        rec[f'mass_plane_{tag}'] = (prob[dfn] * (prob[dfn] >= INK)).sum(0).round(4).tolist()
                sites.append(rec)
                print(f'  {len(sites)} {rec["key"]} n {rec["n"]} def a1 {rec["n_def_a1"]} a2 {rec["n_def_a2"]} '
                      f'{time.time() - T0:.0f} s', flush=True)
    aucs = {('rev' if rev else 'fwd'): auc(np.concatenate(pix[rev][0]), np.concatenate(pix[rev][1]))
            for rev in (False, True)}
    order = 'rev' if (aucs['rev'] or 0) > (aucs['fwd'] or 0) else 'fwd'
    other = 'fwd' if order == 'rev' else 'rev'
    pc_h = (aucs[order] or 0) >= PC_AUC
    out = dict(scorer_check=chk, checkpoint=CKPT.name, checkpoint_sha256=ck_sha, auc2d=aucs, order=order,
               pc_h=dict(auc=aucs[order], bar=PC_AUC, ok=bool(pc_h)), seconds=round(time.time() - T0, 1))
    for att in ('a1', 'a2'):
        out[att] = {}
        for sname in ('label', 'N1_nonlabel'):
            S = [s for s in sites if s['key'][1] == sname]
            n, nd = sum(s['n_use'] for s in S), sum(s[f'n_def_{att}'] for s in S)
            pms = [s[f'pitch_med_{att}'] for s in S if s[f'pitch_med_{att}'] is not None]
            res = dict(n_sites=len(S), n_points=n, n_def=nd, share_def=nd / n if n else None,
                       share_inwin=sum(s[f'n_inwin_{att}'] for s in S) / nd if nd else None,
                       pitch_median_of_sites=float(np.median(pms)) if pms else None)
            for o in (order, other):
                tag = f'{o}_{att}'
                L, ci = boot_ratio(S, f'num_{tag}', f'den_{tag}')
                L0, ci0 = boot_ratio(S, f'num0_{tag}', f'den_{tag}')
                Ls, cis = boot_ratio(S, f'nums_{tag}', f'dens_{tag}')
                res[o] = dict(L=L, L_ci=ci, L0=L0, L0_ci=ci0, L_over_L0=(L / L0) if (L is not None and L0) else None,
                              L_soft=Ls, L_soft_ci=cis, ink_mass=sum(s[f'den_{tag}'] for s in S),
                              mass_by_plane=np.sum([s[f'mass_plane_{tag}'] for s in S], 0).round(2).tolist()
                              if S else None)
            out[att][sname] = res
        lab = out[att]['label']
        out[att]['verdict'] = dict(pc_h=bool(pc_h), scorer=chk['ok'], defined_share=lab['share_def'],
                                   leak_arm=bool(pc_h and chk['ok'] and (lab['share_def'] or 0) >= MIN_DEF),
                                   L=lab[order]['L'], L0=lab[order]['L0'])
    out['sites'] = sites
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'sites'}, indent=1))


if __name__ == '__main__':
    main()
