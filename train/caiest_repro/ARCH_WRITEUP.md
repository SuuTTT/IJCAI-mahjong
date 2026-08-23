# Architecture exploration vs aug_s0 — results

Gate: calibrated duplicate placement, aug_s0-vs-aug_s0 = 2.5 (2.500=tied). A candidate BEATS aug_s0 iff its placement 95% CI lower bound > 2.500. Lower CNN/arch = same 38-plane caiest feature + the e11 enhanced aug recipe (suit x reflect x dragon, label-smoothing, EMA). val_acc uses the SAME rng-12345 val split as aug_s0 (0.887).

| arch | params(M) | val_acc | per_move_ms | TLE<=1s | blocks | placement | 95% CI | beats aug_s0 | verdict |
|---|---|---|---|---|---|---|---|---|---|
| attn_s0 (d_model=256,layers=8,heads=8) | - | None | - | - | 0 | (gating/incomplete) | - | - | - |
| cnnattn_s0 (channels=192,conv_blocks=8,layers=6,heads=8) | - | None | - | - | 0 | (gating/incomplete) | - | - | - |
| gnn_s0 (hidden=384,layers=6) | - | None | - | - | 0 | (gating/incomplete) | - | - | - |
| temporal_s0 (channels=128,blocks=40,emb=64,gru=256) | - | None | - | - | 0 | (gating/incomplete) | - | - | - |
| resse_s0 (channels=128,blocks=40,se_r=8) | 14.51 | None | 34.0 | True | 18 | 2.4997 | [2.4926, 2.5068] | False | TIED_NOT_SEPARATED |
| meldw3_s0 (channels=128,blocks=40) | 14.33 | None | 25.8 | True | 18 | 2.5009 | [2.4951, 2.5067] | False | TIED_NOT_SEPARATED |
| val1_s0 (channels=128,blocks=40) | 14.33 | 0.8807 | 25.63 | True | 18 | 2.5017 | [2.4947, 2.5086] | False | TIED_NOT_SEPARATED |
| val2_s0 (channels=128,blocks=40) | 14.33 | 0.8814 | 25.08 | True | 18 | 2.4996 | [2.4944, 2.5049] | False | TIED_NOT_SEPARATED |
| seval_s0 (channels=128,blocks=40,se_r=8) | 14.51 | 0.8817 | 33.56 | True | 18 | 2.5 | [2.4922, 2.5078] | False | TIED_NOT_SEPARATED |
| rl1it30_s0 (channels=128,blocks=40) | 14.33 | None | 24.81 | True | 18 | 2.4937 | [2.4891, 2.4982] | False | WORSE |
| rl2it30_s0 (channels=128,blocks=40) | 14.33 | None | 24.95 | True | 18 | 2.4957 | [2.4909, 2.5005] | False | TIED_NOT_SEPARATED |
| rl2bit30_s0 (channels=128,blocks=40) | 14.33 | None | 25.13 | True | 18 | 2.5019 | [2.4982, 2.5056] | False | TIED_NOT_SEPARATED |
| rl2cit30_s0 (channels=128,blocks=40) | 14.33 | None | 25.21 | True | 18 | 2.5039 | [2.4974, 2.5104] | False | TIED_NOT_SEPARATED |
| rl2dit30_s0 (channels=128,blocks=40) | 14.33 | None | 24.91 | True | 18 | 2.5066 | [2.5014, 2.5119] | True | BEATS |
| rl2dit30verify_s0 (channels=128,blocks=40) | 14.33 | None | 25.46 | True | 18 | 2.5062 | [2.5018, 2.5107] | True | BEATS |
| rl2eit30_s0 (channels=128,blocks=40) | 14.33 | None | 24.71 | True | 18 | 2.5041 | [2.4986, 2.5097] | False | TIED_NOT_SEPARATED |
| rl2fit30_s0 (channels=128,blocks=40) | 14.33 | None | 24.88 | True | 18 | 2.506 | [2.5003, 2.5118] | True | BEATS |

## Deployability
- attn / cnn_attn / gnn: research-only for the numpy-fused Botzone bot (no BN-fold path; transformer/GNN ops not in the fused kernel). TLE-safe on 1 core if deployed via torch.
- temporal (CNN+GRU): research-only unless the deploy bot is extended to emit the ordered discard sequence; the CNN branch alone is BN-fuseable. TLE-safe (~CNN+small GRU).

## Context (already-run axes from the campaign, verified)
- Enhanced FEATURES (#3): 44-plane enh_192x40 / enh_384x40 + featA/B/C ablations -> all TIED (BESTNET_RESULTS.json). Richer features did not CI-beat aug_s0.
- CAPACITY (#5): raw192 / raw384 / big256 / big320 -> all TIED. Bigger CNN did not CI-beat aug_s0.

## Verdict
(auto-filled at completion — see JSON verdict fields per arch)