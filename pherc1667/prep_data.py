"""VES-2321 data stage (CPU, no model): put the six labelled PHerc1667 segments into the F-401 input layout at
pyramid level 2 (9.596 um) of scan 20251217075048, so that the VES-2241 prep / train / score code runs unchanged.

Writes DATA/ves170/tifxyz/<w>/{x,y,z}.tif and DATA/ves170/labels/<w>.npz (`pos`, `neg`), the layout run_t23
reads under REPO_DATA (`ves170` is the layout name only; the data are 1667).

- tifxyz: the published 2.399 um mesh (`data/pherc1667_meshes/<w>`, `...-on-20251217075048-2.399um.tifxyz`,
  20 level-0 canvas px per node) keeps every 4th node in rows and columns, so a node is again 20 canvas px at
  level 2 (run_t23.STEP), and its xyz go to level-2 voxels with the pyramid voxel-centre rule
  p2 = (p0 - 1.5) / 4 (level-2 voxel k is centred on level-0 coordinate 4k + 1.5). Holes (-1) stay holes.
- labels: the published level-2 arrays `inklabels.zarr/2` and `supervision.zarr/2` (release 20260918);
  pos = ink & supervision, neg = supervision & ~ink. `validation` is not used (leave-one-segment-out).

  python prep_data.py            # ~6 x 150 MB of label reads over https
"""
import json
import os
from pathlib import Path

import numpy as np
import tifffile
import zarr

ROOT = Path(__file__).resolve().parent.parent   # release repo root
REPO_DATA = Path(os.environ.get('REPO_DATA', ROOT / 'data'))   # holds pherc1667_meshes/<w>
DATA = Path(os.environ.get('DATA_1667', ROOT / 'data/pherc1667'))
S3 = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667'
VOL_ID = '20251217075048'
VOL = f'{S3}/volumes/{VOL_ID}-2.399um-0.2m-78keV-masked.zarr'
LEVEL, F = 2, 4
SEGS = {   # w -> S3 segment dir (metadata.json 2026-10-20: the only 1667 segments with ink-labels)
    'w013': '20240304141531-w013_20240304141531_flatboi',
    'w018': '20240304144031-w018_20240304144031_flatboi',
    'w023': '20240304161941-w023_20240304161941_flatboi',
    'w028': '20251208130119-w028_20251208130119156_flatboi',
    'w029': '20251212185248-w029_20251212185248662_flatboi',
    'w031': '20251223230000-w031_2025122323_flatboi',
}


def to_level2(g0):
    """(3, R, C) level-0 xyz grid -> every 4th node, level-2 voxel coords; holes stay -1."""
    g = g0[:, ::F, ::F].astype(np.float64)
    hole = np.all(g == -1, 0) | ~np.all(np.isfinite(g), 0)
    out = (g - (F - 1) / 2) / F
    out[:, hole] = -1
    return out.astype(np.float32)


def main():
    zarr.config.set({'async.concurrency': 8})
    summary = {}
    for w, seg in SEGS.items():
        src = REPO_DATA / 'pherc1667_meshes' / w
        assert f'{seg[:14]}-on-{VOL_ID}-2.399um.tifxyz' in (src / 'source.txt').read_text(), w
        g0 = np.stack([tifffile.imread(src / f'{c}.tif') for c in 'xyz'])
        g2 = to_level2(g0)
        td = DATA / 'ves170/tifxyz' / w
        td.mkdir(parents=True, exist_ok=True)
        for k, c in enumerate('xyz'):
            tifffile.imwrite(td / f'{c}.tif', g2[k])
        lab = f'{S3}/segments/{seg}/ink-labels/2.399um-volume-{VOL_ID}/20260918'
        ink = np.asarray(zarr.open(f'{lab}/inklabels.zarr/{LEVEL}', mode='r')[:]) > 0
        sup = np.asarray(zarr.open(f'{lab}/supervision.zarr/{LEVEL}', mode='r')[:]) > 0
        canvas = (g2.shape[1] * 20, g2.shape[2] * 20)
        assert abs(canvas[0] - ink.shape[0]) <= 20 and abs(canvas[1] - ink.shape[1]) <= 20, (w, canvas, ink.shape)
        pos, neg = ink & sup, sup & ~ink
        ld = DATA / 'ves170/labels'
        ld.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ld / f'{w}.npz', pos=pos, neg=neg)
        summary[w] = dict(seg=seg, grid0=list(g0.shape[1:]), grid2=list(g2.shape[1:]), canvas2=list(ink.shape),
                          pos_px=int(pos.sum()), neg_px=int(neg.sum()), ink_outside_sup_px=int((ink & ~sup).sum()))
        print(w, summary[w], flush=True)
    (DATA / 'prep_data.json').write_text(json.dumps(dict(volume=VOL, level=LEVEL, segs=summary), indent=1))


if __name__ == '__main__':
    main()
