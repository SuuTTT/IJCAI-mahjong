# Architecture exploration vs aug_s0 — results

Gate: calibrated duplicate placement, aug_s0-vs-aug_s0 = 2.5 (2.500=tied). A candidate BEATS aug_s0 iff its placement 95% CI lower bound > 2.500. Lower CNN/arch = same 38-plane caiest feature + the e11 enhanced aug recipe (suit x reflect x dragon, label-smoothing, EMA). val_acc uses the SAME rng-12345 val split as aug_s0 (0.887).

| arch | params(M) | val_acc | per_move_ms | TLE<=1s | blocks | placement | 95% CI | beats aug_s0 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| attn_s0 (d_model=256,layers=8,heads=8) | 6.59 | 0.8797 | 12.4 | True | 6 | 2.4968 | [2.479, 2.5145] | False | TIED_NOT_SEPARATED |
| cnnattn_s0 (channels=192,conv_blocks=8,layers=6,heads=8) | 8.28 | 0.8751 | 16.87 | True | 6 | 2.4845 | [2.4737, 2.4952] | False | WORSE |
| gnn_s0 (hidden=384,layers=6) | 7.71 | 0.7697 | 3.18 | True | 6 | 2.3053 | [2.288, 2.3227] | False | WORSE |
| temporal_s0 (channels=128,blocks=40,emb=64,gru=256) | 14.66 | 0.8833 | 31.27 | True | 6 | 2.5075 | [2.4909, 2.524] | False | TIED_NOT_SEPARATED |

## Deployability
- attn / cnn_attn / gnn: research-only for the numpy-fused Botzone bot (no BN-fold path; transformer/GNN ops not in the fused kernel). TLE-safe on 1 core if deployed via torch.
- temporal (CNN+GRU): research-only unless the deploy bot is extended to emit the ordered discard sequence; the CNN branch alone is BN-fuseable. TLE-safe (~CNN+small GRU).

## Context (already-run axes from the campaign, verified)
- Enhanced FEATURES (#3): 44-plane enh_192x40 / enh_384x40 + featA/B/C ablations -> all TIED (BESTNET_RESULTS.json). Richer features did not CI-beat aug_s0.
- CAPACITY (#5): raw192 / raw384 / big256 / big320 -> all TIED. Bigger CNN did not CI-beat aug_s0.

## Verdict
(auto-filled at completion — see JSON verdict fields per arch)