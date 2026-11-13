"""VES-3199 (F-538) GPU score, inference only: every net in nets.json on every fit draw of draws.py, each net with its
own stored input norm, plus P0 on the w029 draw with P5 w029's norm (the exact input arrays P5 w029 was scored on).
AUC = train_a3.fit_auc's formula (sigmoid, eval mode, chunks of 4, supervised shell voxels, T.auc). Before a net runs,
each draw's blake2b hashes must equal check_draws.json (committed with the criteria), else the script stops.
No held-out site is read: every patch is a training patch of P0 (all 6 segments).
  VES2578_RUN=.../ves2578/step1a2 python score.py [--device cuda]   -> $OUT/scores.json, then make_results.py
"""
import argparse
import json
import os
import time
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]   # release repo root (path-only change for the release)
_VD = os.environ.get('LABELJ_DATA', str(_ROOT / 'data'))   # stored runs: <LABELJ_DATA>/ves<id>/...

import numpy as np
import torch

import draws as W

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('OUT', _VD + '/ves3199'))
CHUNK = 4


def score(net, dev, X, Y, M):
    P = np.empty_like(X)
    with torch.no_grad():
        for a in range(0, len(X), CHUNK):
            P[a:a + CHUNK] = torch.sigmoid(net(torch.from_numpy(X[a:a + CHUNK]).to(dev))).cpu().numpy()
    m = M > 0
    return W.T.auc(P[m & (Y > 0.5)], P[m & (Y < 0.5)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.backends.cudnn.benchmark = False
    T0 = time.time()
    dev = torch.device(a.device)
    nets_cfg = json.loads((HERE / 'nets.json').read_text())
    chk = json.loads((HERE / 'check_draws.json').read_text())
    assert chk['all_pass'] and chk['frng_seed'] == W.FRNG_SEED and chk['n_fit'] == W.T.N_FIT
    nets, norms = {}, {}
    for k, v in nets_cfg.items():
        net = W.T.UNet3D().to(dev)
        net.load_state_dict(torch.load(v['pt'], map_location=dev))
        net.eval()
        nets[k] = net
        norms[k] = tuple(json.loads(Path(v['train_json']).read_text())['norm'].values())
    jobs = [(s, nk, nk) for s in W.SETS for nk in nets] + [('w029', 'P0', 'P5_w029')]   # (set, net, norm of net)
    res = []
    for s in W.SETS:
        D = W.data(s)[0]
        for st, nk, normk in jobs:
            if st != s:
                continue
            X, Y, M = W.draw(D, norms[normk])
            ref = chk['sets'][s]['draws'][normk]
            assert (W.h(X), W.h(Y, M)) == (ref['hX'], ref['hYM']), (s, normk, 'draw hash differs from check_draws.json')
            fa = score(nets[nk], dev, X, Y, M)
            res.append(dict(set=s, net=nk, norm=normk, fit_auc_shell=fa, hX=ref['hX']))
            print(res[-1], flush=True)
        del D
    info = dict(scores=res, gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None,
                seconds=round(time.time() - T0, 1), chunk=CHUNK,
                peak_alloc_gb=round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if dev.type == 'cuda' else None,
                finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'scores.json').write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != 'scores'}), flush=True)


if __name__ == '__main__':
    main()
