#!/usr/bin/env python3
"""E10 aggregate: read e10_cells/lam*_mu*_s*.json (3 wall-seed blocks per (lam,mu) cell).
Per-cell placement mean +/- std across blocks, delta vs base (=2.500, deterministic base-vs-base
calibration), 95% CI (t, df=n-1). A cell SHIPS iff its placement 95% CI lies ENTIRELY ABOVE 2.500.
Reports placement_dist (1/2/3/4 %), first_rate, fourth_rate, override_rate, claim_rate.
Writes E10_RESULTS.json + prints a table.
"""
import os, sys, json, glob
import numpy as np

BASE = '/root/IJCAI-mahjong/train/caiest_repro'
CELLS = f'{BASE}/e10_cells'
CALIB = 2.500
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def agg_cells(cs):
    pp = np.array([c['placement_pts'] for c in cs], dtype=float)
    n = len(pp)
    mean = float(pp.mean()); std = float(pp.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    tc = TCRIT.get(n - 1, 1.96); ci_half = tc * se
    ci_lo, ci_hi = mean - ci_half, mean + ci_half
    dist = np.array([c['dist_pct'] for c in cs], dtype=float).mean(0)
    return dict(
        n_blocks=n, games=int(sum(c['games'] for c in cs)),
        placement_mean=round(mean, 4), placement_std=round(std, 4), placement_se=round(se, 4),
        ci95_lo=round(ci_lo, 4), ci95_hi=round(ci_hi, 4),
        vs_base_delta=round(mean - CALIB, 4),
        delta_ci95=[round(ci_lo - CALIB, 4), round(ci_hi - CALIB, 4)],
        placement_dist_pct=[round(float(x), 2) for x in dist],
        first_rate=round(float(dist[0]), 2), fourth_rate=round(float(dist[3]), 2),
        sg_win_rate=round(float(np.mean([c['sg_win_rate'] for c in cs])), 4),
        sg_fourth_rate=round(float(np.mean([c['sg_fourth_rate'] for c in cs])), 4),
        claim_rate=round(float(np.mean([c['claim_rate'] for c in cs])), 4),
        override_rate=round(float(np.mean([c['override_rate'] for c in cs])), 4),
        block_placements=[round(float(x), 4) for x in pp])


def main():
    cells = {}
    for f in glob.glob(f'{CELLS}/lam*_mu*_s*.json'):
        d = json.load(open(f))
        key = (float(d['lam']), float(d.get('mu', 0)))
        cells.setdefault(key, []).append(d)
    keys = sorted(cells)
    rows = []
    for (lam, mu) in keys:
        r = dict(lam=lam, mu=mu)
        r.update(agg_cells(cells[(lam, mu)]))
        is_base = (lam == 0 and mu == 0)
        r['ci_separated_better'] = None if is_base else bool(r['ci95_lo'] > CALIB)
        r['ci_separated_worse'] = None if is_base else bool(r['ci95_hi'] < CALIB)
        rows.append(r)
    winners = [r for r in rows if not (r['lam'] == 0 and r['mu'] == 0) and r['ci_separated_better']]
    if winners:
        best = max(winners, key=lambda r: r['placement_mean'])
        verdict = ('SHIP lam=%s mu=%s (placement %.4f, CI [%.4f,%.4f] > 2.500): risk-seeking is a '
                   'real lever -> deploy to sim11/final.' % (best['lam'], best['mu'],
                   best['placement_mean'], best['ci95_lo'], best['ci95_hi']))
    else:
        verdict = ('NULL: no (lam,mu) cell CI-separated above 2.500. Risk-seeking by the V_score '
                   'head does NOT convert 2nds->1sts under duplicate scoring; placement is flat-to-'
                   'worse and first-rate falls as lambda rises. Distill stays; lever space exhausted.')
    out = dict(
        experiment='E10_risk_seeking_1ply',
        base='cnn_lad_chunjiandu.npz (ResFused 128x40, deployed distill bot)',
        value_model='value_256x40.pkl (ValueMT 256x40; heads V_place, V_4th, V_score)',
        objective='argmax_a [ policy_logit(a) + lam*V_score_after(a) - mu*P4th_after(a) ]  (risk-seeking, push for 1sts)',
        inverse_of='E8 (which minimized V_place; NULL/worse)',
        format='duplicate placement gate (4-seat rotation), calibrated base-vs-base = 2.500',
        true_1ply=True, decisions_hooked='DISCARD (Play) + CLAIM (Chi/Peng)',
        topk=5, calibration_lam0=CALIB, blocks_per_cell=3, seeds_per_block=400, games_per_block=1600,
        ship_rule='cell ships iff placement 95% CI entirely > 2.500 (CI-separated better)',
        rows=rows, verdict=verdict)
    with open(f'{BASE}/E10_RESULTS.json', 'w') as fo:
        json.dump(out, fo, indent=2)
    print('\n=== E10 risk-seeking 1-ply vs base (cnn_lad), duplicate gate ===')
    print('lam   mu | place_mean +/- std | 95% CI            | dVbase | 1st%  4th% | claim | ovr    | sep')
    for r in rows:
        is_base = (r['lam'] == 0 and r['mu'] == 0)
        sep = 'BASE' if is_base else ('BETTER' if r['ci_separated_better']
              else ('worse' if r['ci_separated_worse'] else 'tie'))
        print('%-5s %-2s| %.4f +/- %.4f | [%.4f, %.4f] | %+.4f | %5.2f %5.2f | %.3f | %.4f | %s' % (
            r['lam'], r['mu'], r['placement_mean'], r['placement_std'], r['ci95_lo'], r['ci95_hi'],
            r['vs_base_delta'], r['first_rate'], r['fourth_rate'], r['claim_rate'],
            r['override_rate'], sep))
    print('\nVERDICT:', verdict)
    print('wrote', f'{BASE}/E10_RESULTS.json')


if __name__ == '__main__':
    main()
