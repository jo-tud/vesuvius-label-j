"""Download the six published PHerc1667 2.399 um meshes that prep_data.py reads, into REPO_DATA/pherc1667_meshes/<w>
({meta.json,x,y,z.tif} plus source.txt, the S3 path prep_data.py checks). Release-only helper; the files it writes
were checked byte-identical to the ones the scored runs read (PROVENANCE.md).

  python pherc1667/fetch_meshes.py
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prep_data as PD  # noqa: E402

BUCKET = 'vesuvius-challenge-open-data'
HTTP = f'https://{BUCKET}.s3.amazonaws.com'


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
    for w, seg in PD.SEGS.items():
        rel = f'PHerc1667/segments/{seg}/mesh/{seg[:14]}-on-{PD.VOL_ID}-2.399um.tifxyz'
        d = PD.REPO_DATA / 'pherc1667_meshes' / w
        d.mkdir(parents=True, exist_ok=True)
        for f in ('meta.json', 'x.tif', 'y.tif', 'z.tif'):
            if not (d / f).exists():
                (d / f).write_bytes(get(f'{HTTP}/{rel}/{f}'))
        (d / 'source.txt').write_text(f'{BUCKET}/{rel}\n')
        print(w, d, flush=True)


if __name__ == '__main__':
    main()
