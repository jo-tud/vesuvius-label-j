"""PHerc0139 data stage (CPU, no model): put the five labelled 9.362 um segments into the input layout run_t23 reads
under REPO_DATA. Release-only script: it rebuilds the files the scored runs read (written there by an earlier study
with the same rule) and was checked against them (tifxyz byte-identical, label masks equal; PROVENANCE.md).

Writes REPO_DATA/ves170/tifxyz/<w>/{meta.json,x,y,z.tif} and REPO_DATA/ves170/labels/<w>.npz (`pos`, `neg`).

- tifxyz: the published mesh `<seg>/mesh/<id>-on-20250728140407-9.362um.tifxyz/`, copied as is.
- labels: the published level-0 arrays `inklabels.zarr/0` and `supervision.zarr/0` of
  `<seg>/ink-labels/9.362um-volume-20250728140407/20260918`; pos = ink & supervision, neg = supervision & ~ink.

  python pherc0139/prep_data.py
"""
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parent.parent   # release repo root
REPO_DATA = Path(os.environ.get('REPO_DATA', ROOT / 'data'))
S3 = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/segments'
VOL_ID = '20250728140407'
SEGS = {   # w -> S3 segment dir
    'w035': '20260317000000-w035_2026031718',
    'w039': '20260302000000-w039_2026030210',
    'w040': '20250831000000-w040_2025083102',
    'w041': '20260108000000-w041_2026010816',
    'w044': '20260115000000-w044_2026011522',
}


def get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as f:
                return f.read()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def main():
    summary = {}
    for w, seg in SEGS.items():
        td = REPO_DATA / 'ves170/tifxyz' / w
        td.mkdir(parents=True, exist_ok=True)
        mesh = f'{S3}/{seg}/mesh/{seg[:14]}-on-{VOL_ID}-9.362um.tifxyz'
        for f in ('meta.json', 'x.tif', 'y.tif', 'z.tif'):
            if not (td / f).exists():
                (td / f).write_bytes(get(f'{mesh}/{f}'))
        lab = f'{S3}/{seg}/ink-labels/9.362um-volume-{VOL_ID}/20260918'
        ink = np.asarray(zarr.open(f'{lab}/inklabels.zarr/0', mode='r')[:]) > 0
        sup = np.asarray(zarr.open(f'{lab}/supervision.zarr/0', mode='r')[:]) > 0
        pos, neg = ink & sup, sup & ~ink
        ld = REPO_DATA / 'ves170/labels'
        ld.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ld / f'{w}.npz', pos=pos, neg=neg)
        summary[w] = dict(seg=seg, canvas=list(ink.shape), pos_px=int(pos.sum()), neg_px=int(neg.sum()))
        print(w, summary[w], flush=True)
    (REPO_DATA / 'ves170/prep_data_0139.json').write_text(json.dumps(dict(volume=VOL_ID, level=0, segs=summary),
                                                                      indent=1))


if __name__ == '__main__':
    main()
