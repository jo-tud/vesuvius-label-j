"""VES-3199 (F-538) CPU check before the GPU score (no net is run, nothing held-out is read): fixes the three 64-patch
fit draws (draws.py) by blake2b hash, with each set's own norm and with the norm of every net scored on it, and
records per draw the number of supervised shell voxels (pos / neg) and which segments the 64 patches come from.
Checks: (1) Y and M hashes do not depend on the norm (positions are norm-free); (2) the all6 norm equals P0's stored
norm (train_P0.json) and the w029 / w028 norms equal the P5 nets' stored norms; (3) the w029 / w028 draws hold no
patch from their own fold.
  VES2578_RUN=.../ves2578/step1a2 python check_draws.py   -> check_draws.json
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

import draws as W

HERE = Path(__file__).resolve().parent
NETS = json.loads((HERE / 'nets.json').read_text())


def main():
    norms = {k: tuple(json.loads(Path(v['train_json']).read_text())['norm'].values()) for k, v in NETS.items()}
    out, ok = dict(frng_seed=W.FRNG_SEED, n_fit=W.T.N_FIT, sets={}), True
    for s in W.SETS:
        D, mu, sd, kept_w = W.data(s)
        own = [k for k, v in NETS.items() if v['set'] == s]
        assert len(own) == 1
        norm_ok = bool(np.allclose((mu, sd), norms[own[0]], rtol=0, atol=1e-9))
        rec = dict(norm=[mu, sd], own_net=own[0], own_norm_matches_net=norm_ok, draws={})
        ym = set()
        for nk in NETS:
            X, Y, M = W.draw(D, norms[nk])
            m = M > 0
            rec['draws'][nk] = dict(hX=W.h(X), hYM=W.h(Y, M))
            ym.add(W.h(Y, M))
        X, Y, M = W.draw(D)
        m = M > 0
        rec['draws']['own_norm'] = dict(hX=W.h(X), hYM=W.h(Y, M))
        ym.add(W.h(Y, M))
        segs = W.patch_blocks(D, kept_w)
        rec.update(n_pos=int((m & (Y > 0.5)).sum()), n_neg=int((m & (Y < 0.5)).sum()),
                   segments=dict(sorted(Counter(segs).items())), ym_norm_free=len(ym) == 1,
                   no_fold_patch=s == 'all6' or s not in segs)
        ok &= norm_ok and rec['ym_norm_free'] and rec['no_fold_patch']
        out['sets'][s] = rec
        print(s, {k: v for k, v in rec.items() if k != 'draws'}, flush=True)
    out['all_pass'] = bool(ok)
    (HERE / 'check_draws.json').write_text(json.dumps(out, indent=1))
    print('all_pass', ok)


if __name__ == '__main__':
    main()
