"""VES-2232 step 0 on PHerc0139: does the leak L onto other sheets differ by ink model?

Criteria: README "Criteria" (committed before any per-site model value). Points, pitches (attempt-2 rule) and
hecate's path are F-389's run_t4.py / run_t23.py, imported unchanged. The ink96 fields (Forrest, ufsm) are read
with `ufsm read` in docker ($UFSM_IMAGE, CPU) and sampled trilinear at the same 3D positions. Writes results.json
and per-point arrays to $POINTS_0139.

  python run.py --check           scorer checks only
  python run.py [--limit N]       full run (or the first N sites per kind and segment)
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                          # release repo root; data defaults live under ROOT / 'data'
F389 = ROOT / 'shared'
OUT = Path(os.environ.get('POINTS_0139', ROOT / 'data/points0139'))
IMAGE = os.environ.get('UFSM_IMAGE', 'ufsm')   # Docker image of the ufsm reader
FORREST = 'https://dl.ash2txt.org/community-uploads/forrest/ink3d'
FIELDS = {'ink96_last': f'{FORREST}/ink96-im1-last/PHerc0139-20250728140407.zarr',
          'ink96_14000': f'{FORREST}/ink96-im1-14000/PHerc0139-20250728140407.zarr'}
KEY = '9.362'
SCAN_SHAPE = [20974, 6621, 6621]

PC_L, PC_L0, PC_TOL = 0.042, 0.184, 0.003
MIN_MASS = 1000.0
MIN_AUC = 0.60
RATIO = 2.0
RATIO_L0 = 1.5
AUC_TOL = 0.05
T_WIDE = np.arange(-24.0, 24.5, 1.0)        # reading: ink96 over its full 3D field
TOPQ = 90                                   # reading: own top-10 % threshold


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


t4 = load('ves2199_run_t4', F389 / 'run_t4.py')
t23 = t4.t23
T_PL, INK = t4.T_PL, t4.INK


# ---------------------------------------------------------------- leak on any plane set
def leak_sums_T(prob, pm, pp, T, thr=INK):
    """run_t4.leak_sums for planes at offsets T (own vx) and ink threshold thr."""
    t = T[None, :]
    lk = np.where(t > 0, t > pp[:, None] / 2, -t > pm[:, None] / 2)
    m = prob * (prob >= thr)
    col = m.sum(1)
    return dict(num=(m * lk).sum(1), den=col, num0=col * lk.mean(1),
                nums=(prob * lk).sum(1), dens=prob.sum(1))


def scorer_checks():
    chk = t4.scorer_check()
    rng = np.random.default_rng(2232)
    prob = rng.uniform(0, 1, (400, 16))
    pm, pp = rng.uniform(3, 30, 400), rng.uniform(3, 30, 400)
    pm[::7] = np.inf
    a, b = t4.leak_sums(prob, pm, pp), leak_sums_T(prob, pm, pp, T_PL)
    same = all(np.array_equal(a[k], b[k]) for k in a)
    print(f'generalised leak == run_t4.leak_sums: {"PASS" if same else "FAIL"}')
    return dict(t4=chk, generalised_equal=bool(same), ok=bool(chk['ok'] and same))


# ---------------------------------------------------------------- ink96 fields
def field_meta(root):
    j = json.loads(urllib.request.urlopen(f'{root}/{KEY}/zarr.json', timeout=60).read())
    u = j['attributes']['ufsm']
    return dict(shape=j['shape'], origin_zyx=u.get('origin_zyx'), checkpoint=u.get('checkpoint'),
                step=u.get('step'), codecs=json.dumps(j['codecs'])[:200])


def ufsm_read(root, lo_zyx, n_zyx, name, cache, tries=4):
    """One shard cache per field: ufsm's cache is keyed by the array path, not the root URL, so a shared cache
    returns the first field's shards for the second (smoke run 10-31: ink96 last == 14000 exactly)."""
    (OUT / 'cache' / cache).mkdir(parents=True, exist_ok=True)
    cmd = ['docker', 'run', '--rm', '--entrypoint', 'ufsm', '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'HOME=/tmp',
           '--cpus', '4', '--memory', '4g', '-v', f'{OUT}:/data', IMAGE, 'read', root, KEY,
           *map(str, lo_zyx), *map(str, n_zyx), f'/data/{name}', '--cache', f'/data/cache/{cache}', '--threads', '4']
    for k in range(tries):
        r = subprocess.run(cmd, capture_output=True, text=True)
        f = OUT / name
        if r.returncode == 0 and f.exists() and f.stat().st_size == int(np.prod(n_zyx)):
            a = np.fromfile(f, np.uint8).reshape(n_zyx)
            f.unlink()
            return a
        print(f'  ufsm read retry {k + 1}: rc {r.returncode} {r.stderr[-300:]}', flush=True)
        time.sleep(3 * 2 ** k)
    raise RuntimeError('ufsm read failed')


def field_box(P, Nn, T):
    q = P[:, None, :] + T[None, :, None] * Nn[:, None, :]           # (n, T, xyz)
    lo = np.maximum(np.floor(q.reshape(-1, 3).min(0)).astype(int) - 2, 0)
    hi = np.minimum(np.ceil(q.reshape(-1, 3).max(0)).astype(int) + 3, np.array(SCAN_SHAPE[::-1]))
    return lo, hi


def sample_field(sub, lo, P, Nn, T):
    q = P[:, None, :] + T[None, :, None] * Nn[:, None, :]
    coords = np.stack([q[..., 2] - lo[2], q[..., 1] - lo[1], q[..., 0] - lo[0]]).reshape(3, -1)
    v = ndimage.map_coordinates(sub.astype(np.float32), coords, order=1, mode='constant', cval=0.0)
    return v.reshape(len(P), len(T)) / 255.0


# ---------------------------------------------------------------- stats
def auc(pos, neg):
    return t4.auc(np.asarray(pos), np.asarray(neg))


def boot(S, a, b):
    return t4.boot_ratio(S, a, b)


def summarise(S, model, tag=''):
    L, ci = boot(S, f'num{tag}_{model}', f'den{tag}_{model}')
    L0, ci0 = boot(S, f'num0{tag}_{model}', f'den{tag}_{model}')
    Ls, cis = boot(S, f'nums{tag}_{model}', f'dens{tag}_{model}')
    return dict(L=L, L_ci=ci, L0=L0, L0_ci=ci0, L_over_L0=(L / L0) if (L is not None and L0) else None,
                L_soft=Ls, L_soft_ci=cis, ink_mass=float(sum(s[f'den{tag}_{model}'] for s in S)))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--threads', type=int, default=10)
    ap.add_argument('--out', default=str(HERE / 'results.json'))
    a = ap.parse_args()
    chk = scorer_checks()
    if a.check or not chk['ok']:
        return
    import torch
    torch.set_num_threads(a.threads)
    hec = t4.load('hecate_ref', t4.HEC_DIR / 'hecate.py')
    model = hec.load_model(str(t4.CKPT), 'cpu')
    ck_sha = hashlib.sha256(t4.CKPT.read_bytes()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)

    meta = {f: field_meta(r) for f, r in FIELDS.items()}
    frame = {f: dict(shape_ok=m['shape'] == SCAN_SHAPE, origin_ok=m['origin_zyx'] == [0, 0, 0], reread_identical=None) for f, m in meta.items()}
    print('field meta', json.dumps(meta), flush=True)

    T0 = time.time()
    sites, pix = [], {False: ([], []), True: ([], [])}
    pts = []                                                           # per-point arrays -> npz
    for wi, w in enumerate(t23.DEV):
        g, v = t23.load_grid(t23.REPO_DATA / 'ves170/tifxyz' / w)
        nrm = t23.grid_normals(g, v)
        lab = np.load(t23.REPO_DATA / 'ves170/labels' / f'{w}.npz')
        pos, neg = lab['pos'], lab['neg']
        valid = t23.pixel_valid(v, pos.shape)
        for si, (sname, msk) in enumerate((('label', pos), ('N1_nonlabel', neg))):
            rng = np.random.default_rng([t23.SEED, wi, si])
            draws = t23.draw_sites(msk, valid, rng, t23.N_SITES)
            for k, (r0, c0, rr, cc) in enumerate(draws[:a.limit]):
                if not len(rr):
                    continue
                volu8, (ra, ca), okt = t4.render_tile(g, nrm, valid, r0, c0)
                hp = t4.run_hecate(hec, model, volu8)
                tp = pos[ra:ra + okt.shape[0], ca:ca + okt.shape[1]] & okt
                tn = neg[ra:ra + okt.shape[0], ca:ca + okt.shape[1]] & okt
                for rev in (False, True):
                    pix[rev][0].append(hp[rev][0][tp].astype(np.float32) / 255)
                    pix[rev][1].append(hp[rev][0][tn].astype(np.float32) / 255)
                P, Nn = t23.at_pixels(g, nrm, rr, cc)
                okp = np.all(np.isfinite(P), 1) & np.all(np.isfinite(Nn), 1)
                rr, cc, P, Nn = rr[okp], cc[okp], P[okp], Nn[okp]
                t, pr = t23.profiles(t23.VOL9, P, Nn, t4.H_A2)
                use = (pr <= 0).mean(1) <= t23.ZERO_MAX
                pmpp = np.array([t4.pitches(t, p, 'a2') for p in pr])
                pm, pp = pmpp[:, 0], pmpp[:, 1]
                dfn = use & ~np.isnan(pm) & ~np.isnan(pp)
                rec = dict(key=[w, sname, k, int(r0), int(c0)], n=int(len(rr)), n_use=int(use.sum()),
                           n_def=int(dfn.sum()))
                probs = {}
                for rev in (False, True):
                    probs['hecate_' + ('rev' if rev else 'fwd')] = \
                        hp[rev][1][:, rr - ra, cc - ca].T.astype(np.float64) / 255
                wide = {}
                lo, hi = field_box(P, Nn, T_WIDE)
                for f, root in FIELDS.items():
                    n_zyx = (hi - lo)[::-1]
                    sub = ufsm_read(root, lo[::-1], n_zyx, f'{f}_site.raw', f)
                    if frame[f]['reread_identical'] is None:
                        frame[f]['reread_identical'] = bool(np.array_equal(
                            sub, ufsm_read(root, lo[::-1], n_zyx, f'{f}_site2.raw', f)))
                    sub_zyx = sub                                         # (z, y, x)
                    probs[f] = sample_field(sub_zyx, lo, P, Nn, T_PL)
                    wide[f] = sample_field(sub_zyx, lo, P, Nn, T_WIDE)
                if not sites:                                         # guard: the two checkpoints are two fields
                    frame['fields_differ'] = bool(not np.array_equal(probs['ink96_last'], probs['ink96_14000']))
                for mname, prob in probs.items():
                    s = leak_sums_T(prob[dfn], pm[dfn], pp[dfn], T_PL)
                    for kk, vv in s.items():
                        rec[f'{kk}_{mname}'] = float(vv.sum())
                    rec[f'mass_plane_{mname}'] = (prob[dfn] * (prob[dfn] >= INK)).sum(0).round(4).tolist()
                for f, prob in wide.items():
                    s = leak_sums_T(prob[dfn], pm[dfn], pp[dfn], T_WIDE)
                    for kk, vv in s.items():
                        rec[f'{kk}W_{f}'] = float(vv.sum())
                pts.append(dict(site=len(sites), label=sname == 'label', dfn=dfn, use=use, pm=pm, pp=pp,
                                **{f'p_{m}': probs[m].astype(np.float16) for m in probs},
                                **{f'pW_{f}': wide[f].astype(np.float16) for f in wide}))
                sites.append(rec)
                print(f'  {len(sites)} {rec["key"]} n {rec["n"]} def {rec["n_def"]} {time.time() - T0:.0f} s',
                      flush=True)

    # hecate order, as F-389 (PC-H)
    auc2d = {('rev' if rev else 'fwd'): auc(np.concatenate(pix[rev][0]), np.concatenate(pix[rev][1]))
             for rev in (False, True)}
    order = 'rev' if (auc2d['rev'] or 0) > (auc2d['fwd'] or 0) else 'fwd'
    pc_h = dict(auc=auc2d[order], order=order, bar=t4.PC_AUC, ok=bool((auc2d[order] or 0) >= t4.PC_AUC))
    hname = f'hecate_{order}'
    models = [hname, 'ink96_last', 'ink96_14000']
    readings_models = [f'hecate_{"rev" if order == "fwd" else "fwd"}']

    # per-point arrays: ink AUC (column max) and the own top-10 % threshold reading
    res = dict(scorer=chk, checkpoint_sha256=ck_sha, field_meta=meta, pc_frame=frame, pc_h=pc_h, auc2d_tile=auc2d,
               seconds=None)
    out = {}
    for mname in models + readings_models:
        lab = [s for s in sites if s['key'][1] == 'label']
        n1 = [s for s in sites if s['key'][1] == 'N1_nonlabel']
        r = dict(label=summarise(lab, mname), N1=summarise(n1, mname))
        cm_pos = np.concatenate([d[f'p_{mname}'][d['use']].astype(np.float32).max(1) for d in pts if d['label']])
        cm_neg = np.concatenate([d[f'p_{mname}'][d['use']].astype(np.float32).max(1) for d in pts if not d['label']])
        r['ink_auc_colmax'] = auc(cm_pos, cm_neg)
        n = sum(s['n_use'] for s in lab)
        r['share_def'] = sum(s['n_def'] for s in lab) / n if n else None
        r['mass_by_plane_label'] = np.sum([s[f'mass_plane_{mname}'] for s in lab], 0).round(2).tolist()
        # reading: own top-10 % threshold over label-site point-planes (defined points)
        allp = np.concatenate([d[f'p_{mname}'][d['dfn']].astype(np.float64).ravel() for d in pts if d['label']])
        thr = float(np.percentile(allp, TOPQ)) if len(allp) else None
        num = den = num0 = 0.0
        for d in pts:
            if not d['label'] or not d['dfn'].any():
                continue
            s = leak_sums_T(d[f'p_{mname}'][d['dfn']].astype(np.float64), d['pm'][d['dfn']], d['pp'][d['dfn']],
                            T_PL, thr=thr)
            num, den, num0 = num + s['num'].sum(), den + s['den'].sum(), num0 + s['num0'].sum()
        r['reading_topq'] = dict(thr=thr, L=num / den if den else None, L0=num0 / den if den else None)
        # reading: per-segment L
        r['per_segment'] = {w: summarise([s for s in lab if s['key'][0] == w], mname)['L'] for w in t23.DEV}
        r['measurable'] = bool((r['share_def'] or 0) >= t4.MIN_DEF and r['label']['ink_mass'] >= MIN_MASS and
                               (r['ink_auc_colmax'] or 0) >= MIN_AUC)
        out[mname] = r
    for f in FIELDS:
        lab = [s for s in sites if s['key'][1] == 'label']
        out[f]['reading_wide_24'] = summarise(lab, f, tag='W')
    res['models'] = out

    # verdict
    def sratio(hi, lo):  # R9 VES-2234: L = 0 is a value, not 'missing'; x/0 = inf, 0/0 = 1
        return (hi / lo) if lo > 0 else (float('inf') if hi > 0 else 1.0)

    H, I, J = out[hname], out['ink96_last'], out['ink96_14000']
    pc_repro = dict(L=H['label']['L'], L0=H['label']['L0'],
                    ok=bool(H['label']['L'] is not None and abs(H['label']['L'] - PC_L) <= PC_TOL + 1e-12 and
                            abs(H['label']['L0'] - PC_L0) <= PC_TOL + 1e-12))
    res['pc_repro'] = pc_repro
    v = dict(scorer=chk['ok'], pc_repro=pc_repro['ok'], pc_h=pc_h['ok'],
             pc_frame_last=all(x for x in frame['ink96_last'].values()),
             pc_frame_14000=all(x for x in frame['ink96_14000'].values()) and frame.get('fields_differ', False))
    void = not (v['scorer'] and v['pc_repro'] and v['pc_h'] and v['pc_frame_last'])
    hL, iL = H['label']['L'], I['label']['L']
    v['a_measurable'] = bool(H['measurable'] and I['measurable'])
    if v['a_measurable'] and hL is not None and iL is not None:
        lo_m, hi_m = (hname, 'ink96_last') if hL < iL else ('ink96_last', hname)
        Lo, Hi = out[lo_m], out[hi_m]
        v['lower_L_model'] = lo_m
        v['L_ratio'] = sratio(Hi['label']['L'], Lo['label']['L'])
        v['b_ratio_and_ci'] = bool(v['L_ratio'] >= RATIO and Lo['label']['L_ci'][1] < Hi['label']['L_ci'][0])
        v['L_over_L0_ratio'] = sratio(Hi['label']['L_over_L0'], Lo['label']['L_over_L0'])
        v['c_L_over_L0'] = bool(v['L_over_L0_ratio'] >= RATIO_L0)
        if J['measurable'] and v['pc_frame_14000'] and J['label']['L'] is not None:
            jr = sratio(max(iL, J['label']['L']), min(iL, J['label']['L']))
            v['within_design_ratio'] = jr
            v['d_within_design'] = bool(jr < RATIO)
        else:
            v['within_design_ratio'] = None
            v['d_within_design'] = 'not tested'
        v['e_equal_auc'] = bool((Lo['ink_auc_colmax'] or 0) >= (Hi['ink_auc_colmax'] or 0) - AUC_TOL)
        lN, hN = Lo['N1']['L'], Hi['N1']['L']
        v['f_N1_direction'] = bool(lN is not None and hN is not None and lN < hN)
        conds = [v['b_ratio_and_ci'], v['c_L_over_L0'], v['d_within_design'] in (True, 'not tested'),
                 v['e_equal_auc'], v['f_N1_direction']]
        v['verdict'] = 'VOID' if void else ('GO-STEP1' if all(conds) else 'NO-GO')
    else:
        v['verdict'] = 'VOID' if void else 'NO-GO'
    res['verdict'] = v
    res['seconds'] = round(time.time() - T0, 1)
    res['sites'] = sites
    np.savez_compressed(OUT / 'points.npz', **{f'{k}_{i}': d[k] for i, d in enumerate(pts) for k in d
                                                if isinstance(d[k], np.ndarray)})
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'sites'}, indent=1))


if __name__ == '__main__':
    main()
