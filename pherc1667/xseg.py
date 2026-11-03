"""VES-2241 reading only (no verdict role): sites with ANOTHER dev segment's training label within 48 vx in 3D
(R9 VES-2242 label-leakage note; adapted from its xseg.py). Reads prep.json + train.npz, no arm output.

  python xseg.py      -> $RUN_1667/xseg.json (drop_keys); score.py reports L with these sites dropped
"""
import json

import numpy as np
from scipy.spatial import cKDTree

import common as C

R_VX = 48.0
t23 = C.f389().t23
meta = json.loads((C.DATA / 'prep.json').read_text())
sup = np.load(C.DATA / 'train.npz')['sup']
G = {}
for w in t23.DEV:
    g, v = t23.load_grid(t23.REPO_DATA / 'ves170/tifxyz' / w)
    G[w] = (g, t23.grid_normals(g, v))
TP = {w: [] for w in t23.DEV}                       # 3D points of supervised training px, every 4th px
for b, bl in enumerate(meta['blocks']):
    rr, cc = np.nonzero(sup[b][::4, ::4])
    P, _ = t23.at_pixels(*G[bl['w']], rr * 4 + bl['r'], cc * 4 + bl['c'])
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
out = dict(r_vx=R_VX, n_sites=len(meta['site_keys']), n_drop=len(drop), n_lt16=int((dists < 16).sum()),
           min_vx=float(dists.min()), drop_keys=drop)
(C.DATA / 'xseg.json').write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != 'drop_keys'})
