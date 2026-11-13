"""Point a VES-2199 run_t4 / run_t23 instance (used by the VES-2241 / VES-2321 prep and score code) at PHerc1667
LEVEL 0 (VES-2578 step 1, README "Level-0 recipe"). F-418's patch1667.py changed REPO_DATA, DEV, VOL9 and vol() for
level 2; here the data names change again and the length constants of the scorer go x 4:
  data     DEV (the six segments, F-418 order of prep_data.SEGS), VOL9 = the 2.399 um volume, UM9; run_t23.vol()
           already opens array '0' (level 0), so vol() is not replaced. Grids and labels come through common.py.
  profile  run_t23.DT 0.25 -> 1.0 and run_t4.H_A2 24 -> 96 L0 vx (pitch profiles t = -96 .. 96, 193 samples)
  bands    PR #2368 band.FACE_DEFAULTS sigma, fill, min_band, ambiguous x 4 (pitches by the attempt-2 rule)
  planes   run_t4.NPL 16 -> 64, T_PL = -31.5 .. 31.5 (leak_mask / leak_sums: the 64 scored slab planes)
  draws    run_t23.BOX, STRIDE 16, 2 -> 64, 8 (step0_labels.py's site rule; the draws themselves come from step 0)
run_t4.render_tile / TILE stay level-2 constructs (hecate's 16-plane 96 x 96 tile) and are disabled on this
instance; hecate runs on the unchanged F-418 level-2 instance (common.f418_l2)."""
import numpy as np
import zarr

# zarr 3 (ves663-cpu venv) has no PathNotFoundError; run_t23's retry clauses name it and so raised AttributeError
# instead of retrying an S3 'Server disconnected' (VES-2332). Alias it to the zarr 3 name so run_t23 stays unchanged.
if not hasattr(zarr.errors, 'PathNotFoundError'):
    zarr.errors.PathNotFoundError = zarr.errors.NodeNotFoundError

S3 = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667'
VOL = f'{S3}/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr'
DEV = ('w013', 'w018', 'w023', 'w028', 'w029', 'w031')     # F-418 prep_data.SEGS order (step-0 index wi)
F = 4
BAND_X4 = ('sigma', 'fill', 'min_band', 'ambiguous')


def apply(t4):
    t23 = t4.t23
    if getattr(t23, 'PATCHED_L0', False):
        return t4
    t23.REPO_DATA = None                                          # level-0 data via common.py only
    t23.DEV = DEV
    t23.VOL9 = VOL
    t23.UM9 = 2.399
    t23.DT = 0.25 * F
    t23.BOX, t23.STRIDE = 16 * F, 2 * F
    t4.H_A2 = 24.0 * F
    t4.NPL = 16 * F
    t4.T_PL = np.arange(t4.NPL) - (t4.NPL - 1) / 2
    for k in BAND_X4:
        t23.band.FACE_DEFAULTS[k] = t23.band.FACE_DEFAULTS[k] * F

    def no_tile(*a, **k):
        raise RuntimeError('run_t4.render_tile is level 2 (hecate); use common.f418_l2() for it')

    t4.render_tile = no_tile
    t4.TILE = None
    t23._VOLS.pop(VOL, None)
    t23.PATCHED_L0 = True
    return t4
