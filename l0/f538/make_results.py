"""VES-3199 (F-538) readings from $OUT/scores.json (criteria frozen in README before the score run).
  OUT=.../LABELJ_DATA/ves3199 python make_results.py   -> results.json
"""
import json
import os
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]   # release repo root (path-only change for the release)
_VD = os.environ.get('LABELJ_DATA', str(_ROOT / 'data'))   # stored runs: <LABELJ_DATA>/ves<id>/...

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('OUT', _VD + '/ves3199'))
PC_P0, PC_P5, PC_TOL = 0.8401817118430075, 0.6905341055562474, 0.01   # train_P0.json, train_J0_hw029_jit0.json
BAR_F, BAR_S = 0.75, 0.80
TEXT = {
    'F': 'P0 on the w029 5-segment fit draw <= 0.75: the gap to 0.80 is the fit-patch draw, not training on 5 '
         'segments; the 0.80 bar does not carry over to 5-segment fit sets. Bar that does: per fold, P0 on that '
         "fold's 5-segment draw (P0_on_w029 / P0_on_w028 below); a 5-segment net passes if within 0.03 below it.",
    'S': 'P0 on the w029 5-segment fit draw >= 0.80: the same patches are not harder for a 6-segment net; the gap is '
         'training on 5 segments. Next: P5 at 24 000 steps (F-502 row 117) or a 6-segment fold design, own GO.',
    'UNCLEAR': 'P0 on the w029 5-segment fit draw is between 0.75 and 0.80: unclear, no re-tune.',
    'VOID': 'a positive control failed (P0 own draw not 0.840 +- 0.01, or P5 w029 own draw not 0.691 +- 0.01), '
            'or check_draws.json does not pass: the w029 read does not count.',
}


def main():
    chk = json.loads((HERE / 'check_draws.json').read_text())
    sc = json.loads((OUT / 'scores.json').read_text())
    S = {(r['set'], r['net'], r['norm']): r['fit_auc_shell'] for r in sc['scores']}
    pc0, pc5 = S[('all6', 'P0', 'P0')], S[('w029', 'P5_w029', 'P5_w029')]
    prim = S[('w029', 'P0', 'P0')]
    g = dict(check_draws=bool(chk['all_pass']), pc_P0=abs(pc0 - PC_P0) <= PC_TOL, pc_P5_w029=abs(pc5 - PC_P5) <= PC_TOL,
             finite=prim == prim)
    if not all(g.values()):
        reading = 'VOID'
    else:
        reading = 'F' if prim <= BAR_F else 'S' if prim >= BAR_S else 'UNCLEAR'
    out = dict(task='VES-3199', finding='F-538', scan='PHerc1667 20251217075048 level 0 (2.403 um, 77 keV, 0.2 m)',
               metric='fit_auc_shell (voxel AUC, 64 training patches, frng 22410, shell planes)',
               gates=g, pc=dict(P0_all6=pc0, P0_ref=PC_P0, P5_w029_w029=pc5, P5_ref=PC_P5, tol=PC_TOL),
               primary=dict(P0_on_w029=prim, bar_F=BAR_F, bar_S=BAR_S), reading=reading, reading_text=TEXT[reading],
               matrix={f'{n}_on_{s}' + ('' if m == n else f'_norm_{m}'): v for (s, n, m), v in S.items()},
               descriptive=dict(
                   P0_on_w028=S[('w028', 'P0', 'P0')],
                   P0_all6_minus_P0_w029=round(pc0 - prim, 4),
                   P0_w029_minus_P5_w029=round(prim - pc5, 4),
                   P0_w028_minus_P5_w028=round(S[('w028', 'P0', 'P0')] - S[('w028', 'P5_w028', 'P5_w028')], 4),
                   P0_on_w029_norm_P5=S[('w029', 'P0', 'P5_w029')],
                   draw_pos_frac={s: round(v['n_pos'] / (v['n_pos'] + v['n_neg']), 4) for s, v in chk['sets'].items()},
                   draw_segments={s: v['segments'] for s, v in chk['sets'].items()},
                   note='descriptive only, one seed per net; every patch is a P0 training patch'),
               baselines=dict(chance=0.5, P5_w029_F526=PC_P5, P5_w028_F502=0.687706909025432, P0_F485=PC_P0,
                              shell_intensity_auc_L0_F485=0.525),
               gpu=sc['gpu'], gpu_hours=round(sc['seconds'] / 3600, 3), peak_alloc_gb=sc['peak_alloc_gb'],
               finished_utc=sc['finished_utc'], heldout_reads='none')
    (HERE / 'results.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('gates', 'pc', 'primary', 'reading')}, indent=1))


if __name__ == '__main__':
    main()
