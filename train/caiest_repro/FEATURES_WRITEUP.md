# +FEATURES (featplus ABC x e11 recipe) vs aug_s0 — results

parity_gate_plus (no-op-guarded; edge_per_game score edge, 0=tied). BEAT iff edge 95% CI lower bound > 0.  NOTE: this is the score-edge gate (native featplus gate), distinct from the arch campaign's calibrated placement gate (2.500=tied).

featplus ABC = base 38 + A(danger: opp-river/meld-commit/progress, +5) + B(shanten reg/7p/13o + useful-tile, +4) + C(genbutsu safe-tile per opp, +3) = 50 planes. Trained on the SAME deployable 128x40 CNN with the enhanced e11 recipe (suit x reflect x dragon aug, label-smoothing, EMA, warmup+cosine).

| seed | trained | done | val_acc | blocks | edge/game | 95% CI | beats aug_s0 | verdict | reads_planes |
|---|---|---|---|---|---|---|---|---|---|
| s0 | True | True | 0.8828 | 8 | 1.0075 | [-0.693, 2.708] | False | TIED_NOT_SEPARATED | True |
| s1 | True | True | 0.8829 | 8 | -0.4546 | [-1.4102, 0.501] | False | TIED_NOT_SEPARATED | True |

## Verdict
(auto-filled at completion — see JSON verdict fields per seed)