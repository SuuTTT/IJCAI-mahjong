# AUG / REG / TTA push on the deployable 128x40 (vs bn128s1 = full_128x40_s1)

Gate: `e11_gate.py` lam=0 calibrated duplicate-format placement gate (4-seat rotation). 2.500 = tied with bn128s1. **Rule: a candidate BEATS bn128s1 iff its 95% CI lower bound > 2.500.**

Calibration (bn128s1 vs bn128s1): placement = **2.5** (1 block(s)) — must read 2.500.

## STEP 1 — augmentation fan-validity

| transform | fan-preserve rate | structural/legality |
|---|---|---|
| suit-perm (deployed baseline) | 0.997 / 0.996333 | PASS ({'chosen_legal_rate': 1.0, 'tile_mass_conserved': True, 'legal_count_preserved': True, 'PASS': True}) |
| rank-reflection | 0.998333 | PASS ({'chosen_legal_rate': 1.0, 'tile_mass_conserved': True, 'legal_count_preserved': True, 'PASS': True}) |
| dragon-perm | 1.0 / 0.998333 | PASS ({'chosen_legal_rate': 1.0, 'tile_mass_conserved': True, 'legal_count_preserved': True, 'PASS': True}) |
| WIND swap (neg-control) | 0.939333 | (excluded) |

Both rank-reflection and dragon-perm preserve fan at rates matching/exceeding the already-deployed suit-perm; the small residual is the single 推不倒 (Reversible Tiles) fan, which also affects suit-perm. The WIND negative control is clearly lower, proving the fan test discriminates → winds correctly excluded. **Used augs: suit-perm + rank-reflection + dragon-perm.**

## STEP 2/4 — enhanced nets & TTA, gated vs bn128s1

| candidate | n_blocks | games | placement mean | 95% CI | margin_lo | verdict |
|---|---|---|---|---|---|---|
| aug_s0 (aug_128x40_s0.pkl) | 14 | 28000 | 2.5117 | [2.5058, 2.5175] | +0.0058 | BEATS_BN128S1 |
| aug_s1 (aug_128x40_s1.pkl) | 14 | 28000 | 2.506 | [2.498, 2.514] | -0.002 | TIED_NOT_SEPARATED |
| aug_s2 (aug_128x40_s2.pkl) | 14 | 28000 | 2.5092 | [2.5033, 2.5152] | +0.0033 | BEATS_BN128S1 |
| tta6 (full_128x40_s1.pkl) | 14 | 28000 | 2.5021 | [2.4965, 2.5077] | -0.0035 | TIED_NOT_SEPARATED |
| tta3 (full_128x40_s1.pkl) | 14 | 28000 | 2.5024 | [2.4957, 2.5091] | -0.0043 | TIED_NOT_SEPARATED |

## STEP 3 — TTA per-move latency (CPU single-thread, Botzone-like; TLE ~1000 ms/move)

- single forward: **27.619 ms/move**
- 3-perm (C3) TTA: **82.378 ms/move**
- 6-perm full TTA: **162.506 ms/move**

## VERDICT

WINNER: aug_s0 CI-separated above bn128s1 (margin_lo=+0.0058)
