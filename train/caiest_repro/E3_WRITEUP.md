# E3 — ValueMT placement-value-model capacity scaling

**Artifact.** ValueMT = ResBN stem+blocks -> GAP -> 3 heads (V_place 4-class, V_4th BCE, V_score regression) on the 38-plane state. Standalone paper artifact AND the "verified-good critic" justifying E4's RL-null.

**Setup.** data/cooked_value.npz, N=5,865,816 labeled states (official ~98k deals, per-deal outcome propagated to each decision state). Fixed 10% held-out split via np.RandomState(0) — **seed-independent, identical for every run** (n_eval=586,581); trainer asserts 0% train/val overlap. All capacities trained **8 epochs, lr 1e-3, bs 1024** (matches how value_256x40 was trained) for a fair scaling comparison.

## Scaling table (seed 0, held-out)

| capacity | params | V_4th AUC | V_place acc | V_score r | place MAE | score MAE |
|---|---|---|---|---|---|---|
| 64x6 | 0.49M | 0.8050 | 0.5360 | +0.4057 | 0.748 | 0.388 |
| 128x20 | 6.00M | 0.9035 | 0.6297 | +0.5314 | 0.617 | 0.345 |
| 128x40 | 11.91M | 0.9002 | 0.6228 | +0.5153 | 0.632 | 0.354 |
| 192x24 | 16.08M | 0.9382 | 0.6906 | +0.6018 | 0.524 | 0.324 |
| 256x40 | 47.41M | 0.9541 | 0.7352 | +0.6545 | 0.450 | 0.282 |

## 256x40 across seeds

| seed | V_4th AUC | V_place acc | V_score r |
|---|---|---|---|
| 0 | 0.9541 | 0.7352 | +0.6545 |
| 1 | 0.9525 | 0.7436 | +0.6717 |

256x40 V_4th AUC mean over 2 seeds = **0.9533** (spread 0.0016).

## Verdict

- **Monotone in capacity (AUC):** NO — V_4th AUC by capacity: 64x6=0.8050, 128x20=0.9035, 128x40=0.9002, 192x24=0.9382, 256x40=0.9541.
- **Saturation:** see table — diminishing returns at the high end (compare top two capacities).
- **Headline 256x40 V_4th AUC = 0.9541** (target ~0.955).
