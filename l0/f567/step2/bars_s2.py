"""VES-3416 (F-567 step 2) per-fold fit bars, inference only, before any step-2 net exists. The F-538 rule as used by
F-545 / F-560 / F-567 (score3226.bars): bar(f) = AUC of the frozen 6-segment net P0 on fold f's 5-segment fit draw
(P0's own input norm) - 0.03. The fit draw of fold f = ../../f538/draws.py (frng 22410,
64 patches, shell planes, kept blocks = every segment but f), which is the draw train_fold.py scores a held-out net on.
Controls (all must hold, else no bar is written and no step-2 training starts):
  B1  P0 on the w029 and w028 draws reproduces F-538 (0.7261 / 0.6583) within 1e-3 (bar code = F-538's);
  B2  the frozen held-out nets J0_hw029 (F-485) and D0_hw029 (F-567) scored on the w029 draw with their own norm
      reproduce their train json fit_auc_shell within 1e-3 (the bar draw is the draw the gate reads).
No held-out site is read: every fit patch is a training patch of P0 (all 6 segments).
  VES2578_RUN=.../ves3416/run python bars_s2.py [--device cuda]   -> bars.json (here) and $VES3416_OUT/bars_scores.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[3]   # release repo root (path-only change for the release)
_VD = os.environ.get('LABELJ_DATA', str(_ROOT / 'data'))   # stored runs: <LABELJ_DATA>/ves<id>/...

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
F538 = HERE.parent.parent / 'f538'
sys.path.insert(0, str(F538))
import draws as W  # noqa: E402

OUT = Path(os.environ.get('VES3416_OUT', _VD + '/ves3416/out'))
NEW = ('w013', 'w031', 'w018', 'w023')                    # step-2 folds, run order
BAR_TOL = 0.03                                            # F-538 / score3226.BAR_TOL
REPRO_TOL = 1e-3
F538_REF = {'w029': 0.7261308035548334, 'w028': 0.6582696527226487}   # F-538 results.json matrix P0_on_<f>
HO = W.C.DATA / 'heldout'
CHUNK = 4


def net_norm(pt, tj):
    net = W.T.UNet3D()
    net.load_state_dict(torch.load(pt, map_location='cpu'))
    return net, tuple(json.loads(Path(tj).read_text())['norm'][k] for k in ('mean', 'std'))


def score(net, dev, X, Y, M):
    """= F-538 score.py score(): sigmoid, eval mode, chunks of 4, supervised shell voxels, T.auc."""
    P = np.empty_like(X)
    with torch.no_grad():
        for a in range(0, len(X), CHUNK):
            P[a:a + CHUNK] = torch.sigmoid(net(torch.from_numpy(X[a:a + CHUNK]).to(dev))).cpu().numpy()
    m = M > 0
    return float(W.T.auc(P[m & (Y > 0.5)], P[m & (Y < 0.5)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.backends.cudnn.benchmark = False
    dev = torch.device(a.device)
    T0 = time.time()
    p0, p0n = net_norm(W.C.DATA / 'net_P0.pt', W.C.DATA / 'train_P0.json')
    p0 = p0.to(dev).eval()
    held = {n: net_norm(HO / f'net_{n}_hw029.pt', HO / f'train_{n}_hw029.json') for n in ('J0', 'D0')}
    res = dict(task='VES-3416', rule='bar = P0 on the fold 5-segment fit draw (P0 norm) - 0.03 (F-538)', folds={},
               B1={}, B2={})
    for f in ('w029', 'w028') + NEW:
        D, _, _, kept_w = W.data(f)
        assert f not in set(kept_w), f
        X, Y, M = W.draw(D, p0n)
        auc = score(p0, dev, X, Y, M)
        segs = W.patch_blocks(D, kept_w)
        res['folds'][f] = dict(P0_on_draw=auc, bar=auc - BAR_TOL, hX=W.h(X), hYM=W.h(Y, M),
                               kept_segments=sorted(set(kept_w)), patch_segments={s: segs.count(s) for s in sorted(set(segs))})
        if f in F538_REF:
            d = abs(auc - F538_REF[f])
            res['B1'][f] = dict(got=auc, want=F538_REF[f], abs_diff=d, ok=bool(d <= REPRO_TOL))
        if f == 'w029':
            for n, (net, nn) in held.items():
                Xn, Yn, Mn = W.draw(D, nn)
                net = net.to(dev).eval()
                got = score(net, dev, Xn, Yn, Mn)
                want = json.loads((HO / f'train_{n}_hw029.json').read_text())['fit_auc_shell']
                res['B2'][f'{n}_hw029'] = dict(got=got, want=want, abs_diff=abs(got - want), ok=bool(abs(got - want) <= REPRO_TOL))
                net.cpu()
        print(f, res['folds'][f]['P0_on_draw'], flush=True)
        del D, X, Y, M
    res['ok'] = bool(all(v['ok'] for v in res['B1'].values()) and all(v['ok'] for v in res['B2'].values())
                     and len(res['B1']) == 2 and len(res['B2']) == 2)
    res.update(gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None, seconds=round(time.time() - T0, 1),
               finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'bars_scores.json').write_text(json.dumps(res, indent=1))
    if res['ok']:
        (HERE / 'bars.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('B1', 'B2', 'ok')}, indent=1))
    print({f: round(v['bar'], 4) for f, v in res['folds'].items()})
    sys.exit(0 if res['ok'] else 4)


if __name__ == '__main__':
    main()
