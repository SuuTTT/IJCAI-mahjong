#!/usr/bin/env python3
"""E8 aggregate: read e8_cells/lam*_s*.json (3 wall-seed blocks per lambda), compute
per-lambda placement mean +/- std across blocks, delta vs base (=2.500, the deterministic
base-vs-base calibration constant), and a 95% CI on the placement (t, df=n-1).

A lambda SHIPS iff its placement 95% CI lies ENTIRELY ABOVE 2.500 (CI-separated better).
Also reports per-lambda claim_rate, single-game win/4th rate, override_rate.

Writes E8_RESULTS.json + prints a table.
"""
import os, sys, json, glob
sys.path.insert(0, '/root/IJCAI-mahjong/train/caiest_repro')
import numpy as np

BASE = '/root/IJCAI-mahjong/train/caiest_repro'
CELLS = f'{BASE}/e8_cells'
CALIB = 2.500   # base-vs-base duplicate calibration (verified: lam=0 == 2.500 exactly)

# t critical values (two-sided 95%) by df
TCRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def main():
    cells = {}
    for f in glob.glob(f'{CELLS}/lam*_s*.json'):
        d = json.load(open(f))
        lam = float(d['lam'])
        cells.setdefault(lam, []).append(d)
    lams = sorted(cells)
    rows = []
    for lam in lams:
        cs = cells[lam]
        pp = np.array([c['placement_pts'] for c in cs], dtype=float)
        n = len(pp)
        mean = float(pp.mean())
        std = float(pp.std(ddof=1)) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 1 else 0.0
        tc = TCRIT.get(n - 1, 1.96)
        ci_half = tc * se
        ci_lo, ci_hi = mean - ci_half, mean + ci_half
        delta = mean - CALIB
        # pooled context metrics (weighted equally across blocks)
        claim = float(np.mean([c['claim_rate'] for c in cs]))
        winr = float(np.mean([c['sg_win_rate'] for c in cs]))
        fourth = float(np.mean([c['sg_fourth_rate'] for c in cs]))
        ovr = float(np.mean([c['override_rate'] for c in cs]))
        games = int(sum(c['games'] for c in cs))
        ci_separated_better = bool(ci_lo > CALIB) if lam > 0 else None
        ci_separated_worse = bool(ci_hi < CALIB) if lam > 0 else None
        rows.append(dict(
            lam=lam, n_blocks=n, games=games,
            placement_mean=round(mean, 4), placement_std=round(std, 4),
            placement_se=round(se, 4),
            ci95_lo=round(ci_lo, 4), ci95_hi=round(ci_hi, 4),
            vs_base_delta=round(delta, 4),
            delta_ci95=[round(ci_lo - CALIB, 4), round(ci_hi - CALIB, 4)],
            ci_separated_better=ci_separated_better,
            ci_separated_worse=ci_separated_worse,
            claim_rate=round(claim, 4), win_rate=round(winr, 4),
            fourth_rate=round(fourth, 4), override_rate=round(ovr, 4),
            block_placements=[round(float(x), 4) for x in pp]))
    # verdict
    winners = [r for r in rows if r['lam'] > 0 and r['ci_separated_better']]
    verdict = ('SHIP lambda=%s (CI-separated better)' % winners[0]['lam']) if winners else \
              'NULL: no lambda CI-separated above 2.500; distill stays'
    out = dict(
        experiment='E8_value_guided_1ply',
        base='cnn_lad_chunjiandu.npz (ResFused 128x40, deployed bot)',
        value_model='value_256x40.pkl (ValueMT 256x40, held-out 4th-AUC 0.955)',
        format='duplicate placement gate (4-seat rotation), calibrated base-vs-base = 2.500',
        true_1ply=True, decisions_hooked='DISCARD (Play) + CLAIM (Chi/Peng)',
        topk=5, calibration_lam0=CALIB,
        blocks_per_lam=3, seeds_per_block=400, games_per_block=1600,
        ship_rule='lambda ships iff placement 95% CI entirely > 2.500',
        rows=rows, verdict=verdict)
    with open(f'{BASE}/E8_RESULTS.json', 'w') as fo:
        json.dump(out, fo, indent=2)
    # table
    print('\n=== E8 value-guided 1-ply vs base (cnn_lad), duplicate gate ===')
    print('lam | place_mean +/- std | 95% CI            | dVbase | claim | win  | 4th  | ovr   | sep')
    for r in rows:
        sep = 'BASE' if r['lam'] == 0 else ('BETTER' if r['ci_separated_better']
              else ('worse' if r['ci_separated_worse'] else 'tie'))
        print('%4s| %.4f +/- %.4f | [%.4f, %.4f] | %+.4f | %.3f | %.3f| %.3f| %.4f| %s' % (
            r['lam'], r['placement_mean'], r['placement_std'], r['ci95_lo'], r['ci95_hi'],
            r['vs_base_delta'], r['claim_rate'], r['win_rate'], r['fourth_rate'],
            r['override_rate'], sep))
    print('\nVERDICT:', verdict)
    print('wrote', f'{BASE}/E8_RESULTS.json')


if __name__ == '__main__':
    main()
