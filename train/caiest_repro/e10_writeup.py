#!/usr/bin/env python3
import json, os
BASE='/root/IJCAI-mahjong/train/caiest_repro'
d=json.load(open(f'{BASE}/E10_RESULTS.json'))
L=[]
L.append('# E10 - Risk-Seeking Value-Guided 1-ply (the inverse lever)\n')
L.append(f"**Base:** {d['base']}  ")
L.append(f"**Value model:** {d['value_model']}  ")
L.append(f"**Objective:** `{d['objective']}`  ")
L.append(f"**Inverse of:** {d['inverse_of']}  ")
L.append(f"**Format:** {d['format']}; topK={d['topk']}; {d['blocks_per_cell']} blocks x {d['seeds_per_block']} seeds = {d['games_per_block']} games/block/cell.  ")
L.append(f"**Ship rule:** {d['ship_rule']}.  ")
L.append(f"**Calibration:** lam=0,mu=0 must read {d['calibration_lam0']} exactly (base-vs-base).\n")
L.append('## Results\n')
L.append('| lam | mu | placement mean +/- std | 95% CI | dVbase | 1st% | 4th% | claim | override | sep |')
L.append('|---|---|---|---|---|---|---|---|---|---|')
for r in d['rows']:
    is_base = (r['lam']==0 and r['mu']==0)
    sep='BASE' if is_base else ('**BETTER**' if r['ci_separated_better'] else ('worse' if r['ci_separated_worse'] else 'tie'))
    L.append('| %s | %s | %.4f +/- %.4f | [%.4f, %.4f] | %+.4f | %.2f | %.2f | %.3f | %.4f | %s |' % (
        r['lam'], r['mu'], r['placement_mean'], r['placement_std'], r['ci95_lo'], r['ci95_hi'],
        r['vs_base_delta'], r['first_rate'], r['fourth_rate'], r['claim_rate'], r['override_rate'], sep))
L.append('\n*1st%/4th% = placement distribution (rounded-rank) averaged across blocks; override = fraction of candidate-seat decisions the risk-seeking objective changed.*\n')
L.append('## Verdict\n')
L.append(d['verdict']+'\n')
# narrative on the conversion question
rows=[r for r in d['rows'] if r['mu']==0]
rows=sorted(rows,key=lambda r:r['lam'])
if len(rows)>=2:
    base=rows[0]; hi=rows[-1]
    L.append('## Does risk-seeking convert 2nds -> 1sts?\n')
    L.append(f"As lambda rises 0 -> {hi['lam']} (mu=0), 1st-rate goes {base['first_rate']:.2f}% -> {hi['first_rate']:.2f}% and 4th-rate goes {base['fourth_rate']:.2f}% -> {hi['fourth_rate']:.2f}%, while placement goes {base['placement_mean']:.4f} -> {hi['placement_mean']:.4f}. ")
    if hi['first_rate']<=base['first_rate']:
        L.append("First-rate does NOT rise - pushing the V_score (upside) head trades away firsts rather than winning them. The risk trade does not pay under duplicate placement scoring.\n")
    else:
        L.append("First-rate rises; see whether the placement CI clears 2.500 above.\n")
open(f'{BASE}/E10_WRITEUP.md','w').write('\n'.join(L))
print('wrote', f'{BASE}/E10_WRITEUP.md')
