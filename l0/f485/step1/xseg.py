"""VES-2241 reading only (no verdict role): sites with ANOTHER dev segment's training label within 48 vx in 3D
(R9 VES-2242 label-leakage note; adapted from its xseg.py). Reads prep.json + train.npz, no arm output.
VES-2578 step 1 (level 0): radius 192 L0 vx and training px every 16th (F-418: 48 vx, every 4th; both x 4), level-0
grids, all step-0 sites (old and spacing draw; F-418 split them over xseg.py and attempt4/sites4.xseg4).

  python xseg.py      -> DATA/xseg.json (drop_keys); score4.py reports L with these sites dropped
"""
import json

import numpy as np
from scipy.spatial import cKDTree

import common as C

R_VX = 48.0 * C.F
SUB = 4 * C.F
t23 = C.f389().t23
meta = json.loads((C.DATA / 'prep.json').read_text())
sup = np.load(C.DATA / 'train_sup.npy', mmap_mode='r')
G = {}
for w in t23.DEV:
    g, v, nrm = C.grid0(w)
    G[w] = (g, nrm)
TP = {w: [] for w in t23.DEV}                       # 3D points of supervised training px, every 16th px
for b, bl in enumerate(meta['blocks']):
    rr, cc = np.nonzero(sup[b][::SUB, ::SUB])
    P, _ = t23.at_pixels(*G[bl['w']], rr * SUB + bl['r'], cc * SUB + bl['c'])
    TP[bl['w']].append(P[np.all(np.isfinite(P), 1)])
TREE = {w: cKDTree(np.concatenate(v)) for w, v in TP.items() if v}
drop, dists = [], []
for k in meta['site_keys']:
    w, _, _, r0, c0 = k
    P, _ = t23.at_pixels(*G[w], np.array([r0]), np.array([c0]))
    if not np.all(np.isfinite(P)):
        continue
    d = min(TREE[x].query(P[0])[0] for x in TREE if x != w)
    dists.append(d)
    if d < R_VX:
        drop.append(k)
dists = np.array(dists)
out = dict(r_vx=R_VX, sub_px=SUB, n_sites=len(meta['site_keys']), n_drop=len(drop), n_lt64=int((dists < 64).sum()),
           min_vx=float(dists.min()), drop_keys=drop)
(C.DATA / 'xseg.json').write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != 'drop_keys'})
