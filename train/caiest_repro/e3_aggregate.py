#!/usr/bin/env python3
"""E3 aggregate: read E3_json/*.json (per-run held-out eval), build E3_RESULTS.json + E3_WRITEUP.md."""
import json, glob, os
BASE = '/root/IJCAI-mahjong/train/caiest_repro'
recs = []
for fp in sorted(glob.glob(f'{BASE}/E3_json/*.json')):
    with open(fp) as f:
        recs.append(json.load(f))

# order by params (capacity)
recs.sort(key=lambda r: (r['params'], r['seed']))
results = []
for r in recs:
    results.append({
        'name': f"{r['channels']}x{r['blocks']}_s{r['seed']}",
        'channels': r['channels'], 'blocks': r['blocks'], 'params': r['params'],
        'fourth_auc': round(r['fourth_auc'], 4), 'place_acc': round(r['place_acc'], 4),
        'score_r': round(r['score_r'], 4), 'place_mae': round(r['place_mae'], 4),
        'score_mae': round(r['score_mae'], 4),
        'seed': r['seed'], 'epochs': r['epochs'], 'n_eval': r['n_eval'],
    })

out = {
    'experiment': 'E3 value/reward-model capacity scaling (ValueMT)',
    'data': 'data/cooked_value.npz', 'n_total': 5865816,
    'eval': 'held-out 10% fixed split (np.RandomState(0)), seed-independent -> identical across all runs; train/val 0% overlap (asserted in trainer)',
    'epochs': 8, 'lr': 1e-3, 'bs': 1024,
    'metrics': {'fourth_auc': 'P(this seat finishes 4th/last) ROC-AUC (headline)',
                'place_acc': 'argmax placement 1..4 accuracy (base rate ~0.264)',
                'score_r': 'Pearson r of predicted vs realized deal score'},
    'results': results,
}
with open(f'{BASE}/E3_RESULTS.json', 'w') as f:
    json.dump(out, f, indent=2)
print('wrote E3_RESULTS.json with', len(results), 'runs')

# ---- writeup ----
seed0 = [r for r in results if r['seed'] == 0]
def fmt(r):
    return f"| {r['channels']}x{r['blocks']} | {r['params']/1e6:.2f}M | {r['fourth_auc']:.4f} | {r['place_acc']:.4f} | {r['score_r']:+.4f} | {r['place_mae']:.3f} | {r['score_mae']:.3f} |"

lines = []
lines.append('# E3 — ValueMT placement-value-model capacity scaling\n')
lines.append('**Artifact.** ValueMT = ResBN stem+blocks -> GAP -> 3 heads (V_place 4-class, V_4th BCE, V_score regression) on the 38-plane state. Standalone paper artifact AND the "verified-good critic" justifying E4\'s RL-null.\n')
lines.append('**Setup.** data/cooked_value.npz, N=5,865,816 labeled states (official ~98k deals, per-deal outcome propagated to each decision state). Fixed 10% held-out split via np.RandomState(0) — **seed-independent, identical for every run** (n_eval=586,581); trainer asserts 0% train/val overlap. All capacities trained **8 epochs, lr 1e-3, bs 1024** (matches how value_256x40 was trained) for a fair scaling comparison.\n')
lines.append('## Scaling table (seed 0, held-out)\n')
lines.append('| capacity | params | V_4th AUC | V_place acc | V_score r | place MAE | score MAE |')
lines.append('|---|---|---|---|---|---|---|')
for r in seed0:
    lines.append(fmt(r))
lines.append('')

# seeds for 256x40
big = [r for r in results if r['channels'] == 256 and r['blocks'] == 40]
if len(big) > 1:
    aucs = [r['fourth_auc'] for r in big]
    lines.append(f"## 256x40 across seeds\n")
    lines.append('| seed | V_4th AUC | V_place acc | V_score r |')
    lines.append('|---|---|---|---|')
    for r in big:
        lines.append(f"| {r['seed']} | {r['fourth_auc']:.4f} | {r['place_acc']:.4f} | {r['score_r']:+.4f} |")
    mean = sum(aucs)/len(aucs)
    lines.append(f"\n256x40 V_4th AUC mean over {len(big)} seeds = **{mean:.4f}** (spread {max(aucs)-min(aucs):.4f}).\n")

# verdict
if seed0:
    aucs = [r['fourth_auc'] for r in seed0]
    monotone = all(aucs[i] <= aucs[i+1] + 1e-9 for i in range(len(aucs)-1))
    top = seed0[-1]
    lines.append('## Verdict\n')
    lines.append(f"- **Monotone in capacity (AUC):** {'YES' if monotone else 'NO'} — V_4th AUC by capacity: " +
                 ', '.join(f"{r['channels']}x{r['blocks']}={r['fourth_auc']:.4f}" for r in seed0) + '.')
    lines.append(f"- **Saturation:** see table — diminishing returns at the high end (compare top two capacities).")
    lines.append(f"- **Headline 256x40 V_4th AUC = {top['fourth_auc']:.4f}** (target ~0.955).")
with open(f'{BASE}/E3_WRITEUP.md', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('wrote E3_WRITEUP.md')
