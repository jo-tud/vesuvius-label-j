"""Point the VES-2199 run_t23 module (used unchanged by the VES-2241 prep / score code) at the PHerc1667 level-2
data written by prep_data.py. Only four names change: REPO_DATA (input layout root), DEV (segments), VOL9 (CT
url) and vol() (opens pyramid level 2 instead of level 0; read_box reopens through vol(), so a retried read
stays on level 2)."""
import asyncio
import time

import aiohttp
import zarr

import prep_data as PD

DEV = tuple(PD.SEGS)

# zarr 3 (ves663-cpu venv) has no PathNotFoundError; run_t23's retry clauses name it and so raised AttributeError
# instead of retrying an S3 'Server disconnected' (VES-2332). Alias it to the zarr 3 name so run_t23 stays unchanged.
if not hasattr(zarr.errors, 'PathNotFoundError'):
    zarr.errors.PathNotFoundError = zarr.errors.NodeNotFoundError


def apply(t23):
    if getattr(t23, 'PATCHED_1667', False):
        return t23
    t23.REPO_DATA = PD.DATA
    t23.DEV = DEV
    t23.VOL9 = PD.VOL
    t23.UM9 = 2.399 * PD.F

    def vol(url, tries=6):
        assert url == PD.VOL, url
        for k in range(tries):
            if url in t23._VOLS:
                return t23._VOLS[url]
            try:
                t23._VOLS[url] = zarr.open(url, mode='r')[str(PD.LEVEL)]
            except (OSError, aiohttp.ClientError, asyncio.TimeoutError, zarr.errors.PathNotFoundError) as e:
                if k == tries - 1:
                    raise
                print(f'  open retry {k + 1}/{tries - 1} after {type(e).__name__}: {e}', flush=True)
                time.sleep(2.0 * 2 ** k)
        return t23._VOLS[url]

    t23.vol = vol
    t23._VOLS.pop(PD.VOL, None)
    t23.PATCHED_1667 = True
    return t23
