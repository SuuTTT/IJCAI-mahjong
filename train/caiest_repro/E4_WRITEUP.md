# E4 — Offline RL (AWR) with a VERIFIED-GOOD critic fails to beat the base imitation policy

**Pillar:** the rigorous "retraining fails / RL-null" result for the negative-results paper.
**Question:** does offline Advantage-Weighted Regression (AWR), using a value critic that is
*independently verified to be good*, beat the base imitation policy (`moyu_bn_128x40`) on
duplicate-format placement? This removes the "the critic was broken" excuse from prior null
RL attempts.

**Answer (honest): NO.** Across a full β-sweep {0, 0.5, 1.0, 2.0, 5.0} × 2 seeds, with a
KL-leash (lr 5e-5, mix 0.3, 8000 steps), **no configuration beats the base policy on the
calibrated duplicate-format placement gate**. Every config is *below* base; for several the
95% CI is entirely below zero (CI-separated worse). The degradation is monotone-ish in β.
The critic is demonstrably good, so "broken critic" is ruled out.

## Critic quality (kills the "broken critic" excuse)
Critic = `ckpt/value_256x40.pkl` (ValueMT, 256×40). Advantages cached over the full labeled
set (`data/cooked_value.npz`, N = 5,865,816): A = special_points(realized place) − V_place(s),
place 1/2/3/4 → 4/3/2/1.

- **corr(V_place, realized_pts) = +0.858** (in-data) — strong critic signal.
- Held-out critic stats (from the value-model card): **4th-place AUC 0.955, place-acc 0.75, score-r 0.67.**
- Advantage weights scale correctly with β (corr(w, adv): β0.5 = 0.94, β1 = 0.84, β2 = 0.82,
  β5 = 0.77; frac clipped at w_hi rises 0.0 → 0.13 as β grows) — the good critic is genuinely
  driving the reweighting.

## Gate (duplicate-format placement)
`frontier_gate.py`: each wall seed played 4× rotating the candidate through all 4 seats vs 3
base-`moyu` refs; candidate earns 4/3/2/1 placement points; seat-bias cancelled. Calibration:
candidate == ref ⇒ 2.500. **3 independent seed-blocks** (seed0 = 70000/80000/90000), 500 seeds
× 4 rotations = 2000 games each, 60 workers/block. **Calib read exactly 2.500 in all 3 blocks**
— the gate is rock-solid, so per-block paired deltas (config − calib) have tiny variance.

## β-sweep results — placement vs base (mean over 3 blocks ± std; paired Δ vs calib, 95% CI)
Base (moyu vs moyu) calibration = **2.5000** (per-block 2.500/2.500/2.500).

| config        | β   | placement (mean±std) | Δ vs base (mean) | Δ 95% CI            | beats base? | claim-rate (Δ vs base 0.293) |
|---------------|-----|----------------------|------------------|---------------------|-------------|------------------------------|
| awr_b0_s1     | 0   | 2.4750 ± 0.0220      | −0.0250          | [−0.0797, +0.0298]  | no          | 0.611 (+0.318)               |
| awr_b0_s2     | 0   | 2.4842 ± 0.0098      | −0.0158          | [−0.0402, +0.0085]  | no          | 0.567 (+0.274)               |
| awr_b0.5_s1   | 0.5 | 2.4770 ± 0.0057      | −0.0230          | [−0.0371, −0.0088]  | no (worse*) | 0.570 (+0.278)               |
| awr_b0.5_s2   | 0.5 | 2.4747 ± 0.0141      | −0.0253          | [−0.0603, +0.0096]  | no          | 0.503 (+0.210)               |
| awr_b1.0_s1   | 1.0 | 2.4557 ± 0.0083      | −0.0443          | [−0.0650, −0.0237]  | no (worse*) | 0.573 (+0.280)               |
| awr_b1.0_s2   | 1.0 | 2.4810 ± 0.0040      | −0.0190          | [−0.0290, −0.0090]  | no (worse*) | 0.642 (+0.349)               |
| awr_b2.0_s1   | 2.0 | 2.4574 ± 0.0181      | −0.0426          | [−0.0875, +0.0022]  | no          | 0.585 (+0.292)               |
| awr_b2.0_s2   | 2.0 | 2.4672 ± 0.0270      | −0.0328          | [−0.0999, +0.0344]  | no          | 0.548 (+0.255)               |
| awr_b5.0_s1   | 5.0 | 2.4587 ± 0.0193      | −0.0413          | [−0.0893, +0.0067]  | no          | 0.502 (+0.210)               |
| awr_b5.0_s2   | 5.0 | 2.4532 ± 0.0174      | −0.0468          | [−0.0902, −0.0035]  | no (worse*) | 0.546 (+0.253)               |

\* "worse*" = 95% CI on the paired delta lies entirely below 0 → CI-separated *worse* than base.

**Every one of the 10 configs is below base** (all 30 per-block deltas negative). Four configs
are CI-separated *worse* than base; the rest straddle 0 from below. **Zero configs beat base**
(no positive CI lower bound). Trend is monotone-ish in β: β≥1 means ≈ 2.456–2.467 (Δ ≈ −0.04)
vs β=0 ≈ 2.48 (Δ ≈ −0.02) — more advantage-sharpening → more degradation, exactly the prior
expectation.

## Claim-rate effect (does AWR change claiming?)
Base `moyu` claim-rate (chi/peng/gang on held-out claim states) = **0.293** (teacher leaders = 0.251).
**Every AWR output massively over-claims: 0.50–0.64 (Δ +0.21 to +0.35).** Critically, the **β=0
pure-BC-continue control also blows up to 0.61** — so this is *not* an advantage-weighting effect;
it is weighted-BC toward the `cooked_value` decision distribution, which is far more claim-happy
than `moyu`. AWR drags the policy off the imitation optimum toward a claim-heavy regime, and
placement gets *worse*, not better. This is a clean mechanism for the null.

## Verdict (honest)
**RL-null CONFIRMED with a verified-good critic.** Offline AWR — even at β=0 (pure BC-continue),
even with the verified critic (corr 0.858, 4th-AUC 0.955), even with a KL-leash — does **not**
beat the base imitation policy on duplicate-format placement at any β. The degradation is
consistent across 2 seeds and 3 seed-blocks and grows with β. The "broken critic" excuse is
removed: the critic is good and its advantages scale the weights correctly; retraining still
fails. **The imitation ceiling is real** — squeezing the realized-special-points signal through
offline AWR moves the policy toward over-claiming and *away* from the placement optimum that
the base imitation policy already sits at. This holds regardless of how E6/Sim-11 land.

## Reproducibility / artifacts
- Trainer: `awr_critic.py` (`--seed` added for multi-seed); β=0 ⇒ uniform weights = pure BC-continue control.
- Advantages: `data/adv_cache.npz` (savez_compressed) via `cache_adv.py` + `ckpt/value_256x40.pkl`.
- AWR outputs: `ckpt/e4/awr_b{0,0.5,1.0,2.0,5.0}_s{1,2}.pkl` (10 fused resbn 128×40).
- Gate: `frontier_gate.py`, 3 blocks × (calib + 10 configs) = 33 runs → `ckpt/e4/gates/*.json`.
- Claim-rate: `e4_claimrate.py` → `ckpt/e4/claimrate.json`.
- Aggregate: `e4_aggregate.py` → `E4_RESULTS.json` (all numbers in this writeup are read from there).
