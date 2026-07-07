# CNN+GNN HYBRID vs aug_s0 - results

**Question:** does a GNN as a PARALLEL feature branch (concatenated with the CNN features before the heads, NOT a replacement) BEAT / TIE / WORSEN aug_s0?

## Graph spec
- nodes: 34 tile-types; node features: 38-plane per-tile-type feature vector from (38,4,9) obs (counts/ownership)
- edges: within-suit chi sequence-adjacency r+-1 & r+-2 + honor 7-clique + self-loop (peng/same-tile); sym-normalized
- 3 message-passing layers, hidden 128, emb 128, pool=mean over 34 nodes
- fusion: concat[cnn_512, gnn_emb128] -> Linear(640,235) (CNN backbone = aug_s0 arch, UNCHANGED)
- params: 14.43 M

## Gate
Calibrated duplicate placement, aug_s0-vs-aug_s0 = 2.500 (tied). HYBRID BEATS iff placement 95% CI lower bound > 2.500. 6 blocks x 500 seeds (t-CI). e11 enhanced aug recipe (suit x reflect x dragon, label-smoothing, EMA) - the SAME recipe as aug_s0.

| seed | val_acc | per_move_ms | TLE<=1s | blocks | placement | 95% CI | beats aug_s0 | verdict |
|---|---|---|---|---|---|---|---|---|
| s0 | 0.8825 | 26.73 | True | 6 | 2.5045 | [2.4952, 2.5137] | False | TIES |

## Context: the STANDALONE GNN (prior)
- kind gnn REPLACED the CNN on the fixed 34-tile graph: val 0.7697, placement 2.3053 -> WORSE (-0.212).
- Hypothesis for the hybrid: keeping the CNN strength and only ADDING relational info should NOT worsen (worst case ties aug_s0, since the head can zero the GNN branch).

## Verdict
HYBRID TIES aug_s0