"""VES-3663 (F-567 control) scorer, fold w029 only (README "Criteria"). CPU only.
  Q1 (primary, equal steps): L(D12s0) - L(J0) on w029; D12s0 = F-560's D0_hw029 (seed 0, 12 000 steps, = J0's steps).
  Q2 (second D seed): L(D24s1) - L(J0) on w029; D24s1 = this task's D1_hw029 (seed 1, 24 000 steps).
  Both: WIN if the gap > 2 x SEED_NOISE (0.0116, F-545 w029 |L(J0) - L(J1)|), LOSS if <= 0, UNCLEAR between.
Frozen rule functions are imported unchanged: F-485 step1/score.py (score_run via score4.score_set, frame_control,
checks), attempt4/score4.py (Sites, fold_controls, score_set, auc_ci), F-545 score3226.py (w029 bar, J loader,
p_shift), F-567 step 1 score3380.py (keep_keys). J0 / J1 outputs = F-545 (ves3226), D24s0 = F-567 step 1 (ves3380/out).

  python score3663.py --check   controls (C1 scorer, C2 pooled hecate PC, F1 hecate w029, F2 P0/P1 ARTEFACT, C4 frozen
                                J0 / J1 / D24s0 recomputed = F-545 / F-567, J0 FOLLOWS) + synthetic rule cases -> check.json
  python score3663.py           results.json (reads OUT/infer.json and the D12s0 / D24s1 outputs if present)
  python score3663.py --stub zero|nan|copy   main-path test on zeroed / NaN / copied D outputs -> OUT/../stub/<kind>/results.json
"""
import argparse
import json
import os
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]   # release repo root (path-only change for the release)
_VD = os.environ.get('LABELJ_DATA', str(_ROOT / 'data'))   # stored runs: <LABELJ_DATA>/ves<id>/...

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
STEP1 = EXP / 'f485' / 'step1'
F545 = EXP / 'f545'
F567 = EXP / 'f567'
for p in (STEP1, STEP1 / 'heldout', STEP1 / 'attempt4', F545, F567):
    sys.path.insert(0, str(p))
import common as C  # noqa: E402
import score as S  # noqa: E402
import score4 as S4  # noqa: E402
import score3226 as SC  # noqa: E402
import score3380 as S1  # noqa: E402

FOLD = 'w029'
VD = Path(_VD + '')
OUT = Path(os.environ.get('VES3663_OUT', str(VD / 'ves3663/out')))
HO = C.DATA / 'heldout'
SEED_NOISE = 0.011575894                                  # F-545 w029 |L(J0) - L(J1)| = 0.053863 - 0.042287 (F-567)
WIN_GAP = 2 * SEED_NOISE                                  # 0.02315
REOPEN_RATIO = 0.5                                        # F-567 reopen-if: a D run at equal steps or a 2nd D seed L/L0 < 0.5
GPU_CAP_H = 7.0
# name: (train json, site outputs npz, shift npz); J nets from F-545, D24s0 from F-567 step 1, new nets from OUT
FROZEN = {'J0': (HO / 'train_J0_hw029.json', SC.OUT / 'out_J0_hw029.npz', SC.OUT / 'out_J0_hw029_shift.npz'),
          'J1': (HO / 'train_J1_hw029.json', SC.OUT / 'out_J1_hw029.npz', SC.OUT / 'out_J1_hw029_shift.npz'),
          'D24s0': (VD / 'ves3380/run/heldout/train_D0_hw029.json', S1.OUT / 'out_D0_hw029.npz',
                    S1.OUT / 'out_D0_hw029_shift.npz')}
NEW = {'D12s0': (VD / 'ves3347/run/heldout/train_D0_hw029.json', OUT / 'out_D12s0_hw029.npz', OUT / 'out_D12s0_hw029_shift.npz'),
       'D24s1': (HO / 'train_D1_hw029.json', OUT / 'out_D24s1_hw029.npz', OUT / 'out_D24s1_hw029_shift.npz')}
REG = {'J0': ('J', 0, 12000), 'J1': ('J', 1, 12000), 'D24s0': ('D', 0, 24000), 'D12s0': ('D', 0, 12000),
       'D24s1': ('D', 1, 24000)}
QUESTIONS = {'Q1': 'D12s0', 'Q2': 'D24s1'}
RESULTS, CHECK = HERE / 'results.json', HERE / 'check.json'


def sub(xs, idx):
    return [xs[i] for i in idx]


def registered(n, tr):
    arm, seed, steps = REG[n]
    return bool((tr['fold'], tr['arm'], tr['seed'], tr['steps'], tr['patch'], tr['batch']) == (FOLD, arm, seed, steps, 128, 16)
                and tr['kept_segments'] == sorted(set(C.FOLDS) - {FOLD}) and tr['limit_blocks'] is None)


def infer_gates(inf, replay, want_nets):
    """G2 net replay (each read net finite and <= tol), G3 site replay (every replay point; a tile for every w029
    replay site), G4 registered nets (infer.json), the nets read = want_nets, not a smoke, 320 sites."""
    rs = inf['replay_sites']
    n_tiles = sum(r['key'][0] == FOLD for r in replay)
    return dict(
        replay_nets=bool(set(inf['replay']) == set(want_nets) and
                         all(np.isfinite(v) and v <= inf['replay_tol'] for v in inf['replay'].values())),
        replay_sites=bool(all(rs.values()) and sum(k.startswith('points_') for k in rs) == len(replay) and
                          sum(k.startswith('tile_') for k in rs) == n_tiles),
        nets_as_registered=bool(inf['fold'] == FOLD and inf['nets'] == list(want_nets) and not inf['smoke'] and
                                all(inf['registered'].get(n) is True for n in want_nets)),
        full_fold=bool(inf.get('sites') == 320))


def read_state(n, trained, inf, replay):
    """Per new net: NOT-RUN (not trained: kill at its cap; or the read's budget stop), VOID (no infer.json, crash,
    an inference gate fails), READ."""
    if not trained[n]:
        return 'NOT-RUN', 'net not trained'
    if inf is None:
        return 'VOID', 'no infer.json (read not run or crashed)'
    if inf.get('not_run'):
        return 'NOT-RUN', 'inference budget stop'
    if not inf.get('done'):
        return 'VOID', 'inference not done (crash)'
    want = [m for m in NEW if trained[m]]
    g = infer_gates(inf, replay, want)
    if not all(g.values()):
        return 'VOID', f'inference gate {g}'
    return 'READ', None


def q_status(controls_ok, state, run_j, run_d):
    """VOID (a control fails or the read is VOID) -> NOT-RUN -> NO-VERDICT (J0 or the D net not measurable: F-485
    floors) -> WIN (L_D - L_J0 > 2 x seed noise) / LOSS (<= 0) / UNCLEAR."""
    if not controls_ok or state == 'VOID':
        return dict(status='VOID')
    if state == 'NOT-RUN':
        return dict(status='NOT-RUN')
    if not (run_j['measurable'] and run_d['measurable']):
        return dict(status='NO-VERDICT')
    gap = float(run_d['label']['L'] - run_j['label']['L'])
    st = 'WIN' if gap > WIN_GAP else 'LOSS' if gap <= 0 else 'UNCLEAR'
    return dict(status=st, gap=gap, win_gap=WIN_GAP, L_J0=run_j['label']['L'], L_D=run_d['label']['L'])


def november_rule(q1, q2, nums):
    """Registered decision rule for the "D trained twice as long as J" sentence of the November N1 line (VES-2978)."""
    s1, s2 = q1['status'], q2['status']
    fitq = ('' if nums['D12s0_fit_ok'] else
            f' (that 12 000-step D net is under its fit bar, {nums["D12s0_fit"]:.3f} vs {nums["bar"]:.3f})')
    if s1 == 'WIN' and s2 == 'WIN':
        return ('M1', f'One seed per arm on four folds; on fold w029 a D net trained for the same 12 000 steps as J '
                      f'also leaks (L {q1["L_D"]:.3f} vs J {q1["L_J0"]:.3f}){fitq}, and a second D seed at 24 000 steps '
                      f'gives L {q2["L_D"]:.3f}.')
    if s1 == 'WIN':
        return ('M2', f'One seed per arm; D trained twice as long as J. On fold w029 a D net at J\'s 12 000 steps also '
                      f'leaks (L {q1["L_D"]:.3f} vs J {q1["L_J0"]:.3f}){fitq}; a second D seed at 24 000 steps did not '
                      f'confirm ({s2}).')
    if s1 in ('LOSS', 'UNCLEAR'):
        return ('M3', f'One seed per arm; D trained twice as long as J, and at equal steps (one fold, w029) J\'s lower '
                      f'leak is not shown (D L {q1["L_D"]:.3f} vs J {q1["L_J0"]:.3f}, {s1}).')
    return ('M4', 'One seed per arm; D trained twice as long as J. (Equal-step control not scored: ' + s1 + '.)')


def synthetic():
    """Rule cases: read_state, infer_gates, q_status, november_rule."""
    out = {}
    rep = [dict(key=['w029']), dict(key=['w028'])]
    good = dict(replay={'D12s0': 0.0, 'D24s1': 1e-4}, replay_tol=5e-3, fold='w029', nets=['D12s0', 'D24s1'],
                registered={'D12s0': True, 'D24s1': True}, smoke=0, done=True, not_run=False, sites=320,
                replay_sites={'points_0': True, 'points_1': True, 'tile_0': True})
    both, only12 = {'D12s0': True, 'D24s1': True}, {'D12s0': True, 'D24s1': False}
    one = dict(good, replay={'D12s0': 0.0}, nets=['D12s0'], registered={'D12s0': True})
    cases = {'read': ('D24s1', both, good, 'READ'), 'not_trained': ('D24s1', only12, one, 'NOT-RUN'),
             'only12_read': ('D12s0', only12, one, 'READ'), 'only12_extra_net': ('D12s0', only12, good, 'VOID'),
             'no_infer': ('D12s0', both, None, 'VOID'), 'budget_stop': ('D12s0', both, dict(good, done=False, not_run=True), 'NOT-RUN'),
             'not_done': ('D12s0', both, dict(good, done=False), 'VOID'),
             'replay_over': ('D12s0', both, dict(good, replay=dict(good['replay'], D24s1=6e-3)), 'VOID'),
             'replay_nan': ('D12s0', both, dict(good, replay=dict(good['replay'], D12s0=float('nan'))), 'VOID'),
             'net_missing': ('D12s0', both, dict(good, replay={'D12s0': 0.0}), 'VOID'),
             'tile_missing': ('D12s0', both, dict(good, replay_sites={'points_0': True, 'points_1': True}), 'VOID'),
             'tile_differs': ('D12s0', both, dict(good, replay_sites={'points_0': True, 'points_1': True, 'tile_0': False}), 'VOID'),
             'unregistered': ('D12s0', both, dict(good, registered={'D12s0': True, 'D24s1': False}), 'VOID'),
             'smoke': ('D12s0', both, dict(good, smoke=8), 'VOID'), 'wrong_fold': ('D12s0', both, dict(good, fold='w028'), 'VOID'),
             'few_sites': ('D12s0', both, dict(good, sites=16), 'VOID')}
    for k, (n, tr, inf, want) in cases.items():
        got = read_state(n, tr, inf, rep)[0]
        out[f'read_{k}'] = dict(want=want, got=got, ok=got == want)

    def run(L, meas=True):
        return dict(measurable=meas, label=dict(L=L))
    j = run(0.054)
    qc = {'win': (True, 'READ', j, run(0.28), 'WIN'), 'loss_equal': (True, 'READ', j, run(0.054), 'LOSS'),
          'loss_lower': (True, 'READ', j, run(0.03), 'LOSS'), 'unclear': (True, 'READ', j, run(0.054 + 0.02), 'UNCLEAR'),
          'win_edge': (True, 'READ', j, run(0.054 + 0.0232), 'WIN'), 'void_ctrl': (False, 'READ', j, run(0.28), 'VOID'),
          'void_read': (True, 'VOID', j, run(0.28), 'VOID'), 'not_run': (True, 'NOT-RUN', j, run(0.28), 'NOT-RUN'),
          'nv_d': (True, 'READ', j, run(0.28, False), 'NO-VERDICT'), 'nv_j': (True, 'READ', run(0.054, False), run(0.28), 'NO-VERDICT')}
    for k, (ck, st, rj, rd, want) in qc.items():
        got = q_status(ck, st, rj, rd)['status']
        out[f'q_{k}'] = dict(want=want, got=got, ok=got == want)
    W = dict(status='WIN', L_D=0.25, L_J0=0.054)
    nums = dict(D12s0_fit_ok=False, D12s0_fit=0.687, bar=0.696)
    nc = {'both': (W, W, 'M1'), 'q1_only': (W, dict(status='UNCLEAR'), 'M2'), 'q1_win_q2_notrun': (W, dict(status='NOT-RUN'), 'M2'),
          'q1_loss': (dict(W, status='LOSS'), W, 'M3'), 'q1_unclear': (dict(W, status='UNCLEAR'), W, 'M3'),
          'q1_void': (dict(status='VOID'), W, 'M4'), 'q1_nv': (dict(status='NO-VERDICT'), W, 'M4'),
          'q1_notrun': (dict(status='NOT-RUN'), W, 'M4')}
    for k, (a, b, want) in nc.items():
        got = november_rule(a, b, nums)[0]
        out[f'november_{k}'] = dict(want=want, got=got, ok=got == want)
    for k, v in out.items():
        print(f'synthetic {k}: {v["got"]} (want {v["want"]})')
    return dict(cases=out, ok=all(v['ok'] for v in out.values()))


def score_runs(U, idx, names, paths, bar):
    """score4.score_set on w029 for the named nets; S.MIN_FIT = the w029 bar (fit_ok is a reading here)."""
    z = {n: np.load(paths[n][1]) for n in names}
    trs = {n: json.loads(paths[n][0].read_text()) for n in names}
    keep, S.MIN_FIT = S.MIN_FIT, bar
    try:
        runs, floor = S4.score_set(list(names), lambda n, u: z[n][f'p_{u}'], trs, idx, U)
    finally:
        S.MIN_FIT = keep
    ci = {n: S4.auc_ci([z[n][f'p_{u}'].mean(0) for u in idx], sub(U.pts, idx), sub(U.flags, idx)) for n in names}
    const = {n: np.load(paths[n][2])['const'].tolist() for n in names}
    reg = {n: registered(n, trs[n]) for n in names}
    return runs, floor, ci, const, reg


def controls(U, idx, bar):
    """C1 + C2 (S.checks on the old-draw sites), F1 hecate + F2 P0/P1 ARTEFACT on w029 (score4.fold_controls), C4 the
    frozen J0 / J1 / D24s0 recomputed here = F-545 / F-567 step 1 (L, L0, ink AUC within 1e-9) and registered, J0
    FOLLOWS (frame test)."""
    chk = S.checks(sub(U.pts, U.old), sub(U.flags, U.old))
    fc = S4.fold_controls(U, FOLD, idx, SC.p_shift(U, [FOLD]))
    runs, floor, ci, const, reg = score_runs(U, idx, list(FROZEN), FROZEN, bar)
    r545 = json.loads((F545 / 'results.json').read_text())['folds'][FOLD]['runs']
    r567 = json.loads((F567 / 'results.json').read_text())['fold']['runs']
    ref = {'J0': r545['J0'], 'J1': r545['J1'], 'D24s0': r567['D0']}
    d = {n: max(abs(runs[n]['label']['L'] - ref[n]['label']['L']), abs(runs[n]['label']['L0'] - ref[n]['label']['L0']),
                abs(runs[n]['ink_auc'] - ref[n]['ink_auc'])) for n in FROZEN}
    jz = np.load(FROZEN['J0'][2])
    frame = S.frame_control(lambda k, j: jz[f'p{k}_{idx[j]}'], sub(U.pts, idx), sub(U.flags, idx))
    c4 = dict(max_abs_diff=d, registered=reg, ok=bool(all(v <= 1e-9 for v in d.values()) and all(reg.values())))
    seed_noise_ok = bool(abs(SEED_NOISE - abs(runs['J0']['label']['L'] - runs['J1']['label']['L'])) <= 1e-8)
    ok = bool(chk['ok'] and fc['ok'] and c4['ok'] and frame['status'] == 'FOLLOWS' and seed_noise_ok
              and runs['J0']['measurable'])
    return dict(checks=chk, fold_controls=fc, c4_frozen_equal=c4, J0_frame=frame['status'], J0_frame_e=frame['e'],
                seed_noise_equals_F545=seed_noise_ok, ok=ok), runs, floor, ci, const


def ledger():
    """GPU hours of this task: the inference smoke (if it ran on a GPU) + every unit.sh unit (OUT/ledger.jsonl)."""
    sm = OUT.parent / 'smoke' / 'infer.json'
    js = json.loads(sm.read_text()) if sm.exists() else {}
    h = js['total_seconds'] / 3600 if js.get('gpu') else 0.0
    lf = OUT / 'ledger.jsonl'
    rows = [json.loads(x) for x in lf.read_text().splitlines() if x.strip()] if lf.exists() else []
    return dict(smoke_h=round(h, 3), units=rows, gpu_h=round(h + sum(r['seconds'] for r in rows) / 3600, 3), cap_h=GPU_CAP_H)


def reading(n, r, ci):
    """Per-net reading row. A non-finite net (score_run: no label, not measurable) or an undefined L gives None fields."""
    lab = r.get('label') or {}
    return dict(fit=r['fit_auc_shell'], fit_ok_vs_bar=r['fit_ok'], finite=r.get('finite'), measurable=r.get('measurable'),
                L=lab.get('L'), L_ci=lab.get('L_ci'), L0=lab.get('L0'), L_over_L0=lab.get('L_over_L0'),
                L_N1=(r.get('N1') or {}).get('L'), phase_L=r.get('phase'), ink_auc=r.get('ink_auc'), ink_auc_ci=ci.get(n),
                steps=REG[n][2], seed=REG[n][1])


def minus(a, b):
    return None if a is None or b is None else a - b


def assemble(res, ctrl, runs, floor, ci, const, bar):
    """Questions, November rule and readings from the scored runs (res['read_state'] final). Returns the read names."""
    controls_ok = bool(all(res['gates'].values()) and ctrl['ok'])
    res['controls_ok'] = controls_ok
    q = {}
    for k, n in QUESTIONS.items():
        st = res['read_state'][n]['state']
        q[k] = dict(q_status(controls_ok, st, runs['J0'], runs.get(n)), net=n)
    res['questions'] = q
    nf = json.loads(NEW['D12s0'][0].read_text())['fit_auc_shell']
    res['november'] = november_rule(q['Q1'], q['Q2'], dict(D12s0_fit_ok=bool(nf >= bar), D12s0_fit=nf, bar=bar))
    names = [n for n in ('J0', 'J1', 'D24s0', 'D12s0', 'D24s1') if n in runs]
    per = {n: reading(n, runs[n], ci) for n in names}
    dl = {n: per[n]['L'] for n in names}
    final = [n for n in NEW if n in per and res['read_state'][n]['state'] == 'READ']   # reopen: final state READ only
    res['readings'] = dict(
        per_net=per,
        reopen_F567={n: (None if per[n]['L_over_L0'] is None else bool(per[n]['L_over_L0'] < REOPEN_RATIO)) for n in final},
        D_seed_spread_24k=(None if minus(dl.get('D24s1'), dl['D24s0']) is None else abs(dl['D24s1'] - dl['D24s0'])),
        J_seed_spread=SEED_NOISE,
        steps_effect_seed0=minus(dl['D24s0'], dl.get('D12s0')),
        const={n: const[n] for n in names if n[0] == 'D'},
        mass_floor=floor,
        baselines=dict(chance_auc=0.5, L0='each net\'s own L0 (no-information placement)', hecate_w029=ctrl['fold_controls']['pc_hecate'],
                       F567_w029=dict(J0=0.0539, J1=0.0423, D24s0=0.2805), F418_L2_D='~L0'))
    return names


def stub(kind, U, idx, bar, res, ctrl, runs, floor, ci, const):
    """--stub zero|nan|copy (R9 VES-3668): D24s0's w029 outputs zeroed (L undefined), NaN (non-finite) or unchanged
    stand in for BOTH new nets, scored by the real score_runs + assemble; read_state set to READ (read_state has its own
    synthetic cases). Writes OUT.parent/stub/<kind>/results.json only (never this folder).
    Want: zero / nan -> Q1, Q2 NO-VERDICT, November M4, reopen / spreads None; copy -> Q1, Q2 WIN, M1, reopen False,
    D seed spread 0, steps effect 0."""
    d = OUT.parent / 'stub' / kind
    d.mkdir(parents=True, exist_ok=True)
    z, zs = np.load(FROZEN['D24s0'][1]), np.load(FROZEN['D24s0'][2])
    fill = {'zero': np.zeros_like, 'nan': lambda a: np.full_like(a, np.nan), 'copy': lambda a: a}[kind]
    paths = {}
    for n in NEW:
        o, sh = d / f'out_{n}_hw029.npz', d / f'out_{n}_hw029_shift.npz'
        np.savez(o, **{k: fill(z[k]) for k in z.files})
        np.savez(sh, **{k: (zs[k] if k == 'const' else fill(zs[k])) for k in zs.files})
        paths[n] = (FROZEN['D24s0'][0], o, sh)                             # a registered-shape D train json
    r2, _, c2, k2, _ = score_runs(U, idx, list(NEW), paths, bar)
    runs.update(r2), ci.update(c2), const.update(k2)
    res['stub'] = kind
    res['read_state'] = {n: dict(state='READ', why=None) for n in NEW}
    names = assemble(res, ctrl, runs, floor, ci, const, bar)
    rd, qs = res['readings'], [res['questions'][k]['status'] for k in QUESTIONS]
    if kind == 'copy':
        ok = bool(qs == ['WIN', 'WIN'] and res['november'][0] == 'M1' and rd['reopen_F567'] == {n: False for n in NEW}
                  and rd['D_seed_spread_24k'] == 0 and rd['steps_effect_seed0'] == 0)
    else:
        ok = bool(qs == ['NO-VERDICT'] * 2 and res['november'][0] == 'M4' and rd['reopen_F567'] == {n: None for n in NEW}
                  and rd['D_seed_spread_24k'] is None and rd['steps_effect_seed0'] is None)
    res['stub_ok'] = ok
    (d / 'results.json').write_text(json.dumps(res, indent=1, default=float))
    show(res, names)
    print('stub', kind, 'ok' if ok else 'FAIL', d / 'results.json')
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--stub', choices=('zero', 'nan', 'copy'))
    a = ap.parse_args()
    U = S4.Sites()
    idx = U.fidx[FOLD]
    bar = SC.bars()[FOLD]['bar']
    res = dict(task='VES-3663', parent_finding='F-567', fold=FOLD,
               scan='PHerc1667 20251217075048 level 0 (2.403 um, 77 keV, 0.2 m)', normal='forward', depth='+-31.5 vx (64 planes)',
               fit_bar=bar, seed_noise=SEED_NOISE, win_gap=WIN_GAP)
    res['gates'] = dict(power=bool(U.info['power']['ok']),
                        full_draw=bool(U.info['limit'] is None and U.prep['limit_blocks'] is None and not U.prep['smoke']
                                       and not C.SMOKE))
    ctrl, runs, floor, ci, const = controls(U, idx, bar)
    res['controls'] = ctrl
    if a.check:
        res['synthetic'] = synthetic()
        res['ledger'] = ledger()
        res['check_ok'] = bool(all(res['gates'].values()) and ctrl['ok'] and res['synthetic']['ok'])
        CHECK.write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps({k: res[k] for k in ('gates', 'check_ok')}, indent=1),
              json.dumps({k: ctrl[k] for k in ('c4_frozen_equal', 'J0_frame', 'seed_noise_equals_F545', 'ok')}, default=float),
              'hecate w029', ctrl['fold_controls']['pc_hecate'], 'P', ctrl['fold_controls']['p_control'])
        return
    if a.stub:
        stub(a.stub, U, idx, bar, res, ctrl, runs, floor, ci, const)
    trained = {n: NEW[n][0].exists() for n in NEW}
    lf = OUT / 'infer.json'
    inf = json.loads(lf.read_text()) if lf.exists() else None
    res['trained'] = trained
    res['read_state'] = {n: dict(zip(('state', 'why'), read_state(n, trained, inf, U.prep['replay']))) for n in NEW}
    read = [n for n in NEW if res['read_state'][n]['state'] == 'READ']
    if read:
        r2, f2, c2, k2, g2 = score_runs(U, idx, read, NEW, bar)
        runs.update(r2), ci.update(c2), const.update(k2)
        for n in read:                                                     # the train json must be the registered run
            if not g2[n]:
                res['read_state'][n] = dict(state='VOID', why='train json not the registered run')
    names = assemble(res, ctrl, runs, floor, ci, const, bar)
    res['ledger'] = ledger()
    RESULTS.write_text(json.dumps(res, indent=1, default=float))
    show(res, names)


def show(res, names):
    per = res['readings']['per_net']
    print(json.dumps({k: res[k] for k in ('read_state', 'questions', 'november')}, indent=1, default=float))
    print(json.dumps({n: {k: per[n][k] for k in ('fit', 'L', 'L0', 'L_over_L0', 'ink_auc')} for n in names}, indent=1, default=float))


if __name__ == '__main__':
    main()
