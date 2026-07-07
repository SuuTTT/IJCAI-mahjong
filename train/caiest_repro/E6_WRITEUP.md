# E6 — Is the tau=2 claim-suppression correction SCORING-FORMAT-DEPENDENT?

**Question.** E1+E2 showed the tau=2 claim-suppression correction is NULL on the DUPLICATE-format
placement metric (all 24 seat permutations summed, variance largely cancels). E6 asks the rescue
question: does tau=2 nonetheless help on SINGLE-GAME metrics — win/1st-rate, 4th-place (last)
avoidance, raw-score mean, and especially raw-score VARIANCE — that an ELO/ladder (single-game)
world actually rewards but the duplicate format hides? If yes (H1), the correction's value is
scoring-format-dependent and reconciles the ladder reality ("aggression loses") with the duplicate
null. If no (H0), the correction is null in every format.

**Design.** Same gate as E2 (`e6_gate.py`, derived from `e2_gate.py`), but instrumented to emit,
for EVERY individual game (each of the 4 seat rotations, BEFORE duplicate-permutation summing), the
candidate's raw MCR score `sim.scores[cand_seat]` and its single-game rank (avg-rank on ties, 1..4).
Candidate = moyu_bn_128x40; opponents = moyu_bn_128x40. Cells: cand_tau {0,2} x ref_tau {0,2}
(ref_tau applies the same suppression to the 3 opponents). ref_tau=0 = over-claiming homogeneous
field (kept claim-rate ~0.336); ref_tau=2 = selective field (kept ~0.27 ≈ real top-10 ~0.26). Each
cell = 3 wall-seed blocks (seed0 70000/80000/90000) x 300 seeds x 4 rotations = **3600 individual
games/cell** (14,400 games total). Single-game metrics pooled over all 3600 per-game records;
duplicate placement aggregated per-block as in E2.

**Single-game extraction VERIFIED (calibration).** In the self-play cell (cand==ref==moyu, tau=0),
single-game **mean placement = 2.5000** exactly and **mean raw score = 0.0** exactly (both required
by symmetry of a zero-sum game with 4 identical policies summed over seat rotations). Per-game raw
score has real spread (std ≈ 28.6 MCR points; 1st-rate 0.237, 4th-rate 0.147). Extraction is correct.

## Results — tau=0 vs tau=2, both fields, DUPLICATE and SINGLE-GAME

### Field ref_tau=0  (over-claiming homogeneous field, opp kept-claim 0.336)

| metric | tau=0 | tau=2 | delta (ct2−ct0) | stat | significant? |
|---|---:|---:|---:|---:|:--:|
| DUPLICATE placement     | 2.5000 | 2.4971 | −0.0029 | t=−0.28 (df=2) | no |
| SG 1st-rate (win proxy) | 0.2369 | 0.2308 | −0.0061 | z=−0.61 | no (wrong sign) |
| SG 4th-rate (last avoid)| 0.1469 | 0.1444 | −0.0025 | z=−0.30 | no |
| SG raw-score mean       | 0.000  | −0.179 | −0.179  | t=−0.27 | no |
| SG raw-score std (var)  | 28.646 | 28.470 | −0.176 (F=1.012) | — | no |
| SG mean placement       | 2.5000 | 2.5029 | +0.0029 | t=+0.13 | no |

### Field ref_tau=2  (selective field ≈ real top-10, opp kept-claim 0.269)

| metric | tau=0 | tau=2 | delta (ct2−ct0) | stat | significant? |
|---|---:|---:|---:|---:|:--:|
| DUPLICATE placement     | 2.4957 | 2.5000 | +0.0043 | t=+0.58 (df=2) | no |
| SG 1st-rate (win proxy) | 0.2356 | 0.2328 | −0.0028 | z=−0.28 | no |
| SG 4th-rate (last avoid)| 0.1511 | 0.1453 | −0.0058 | z=−0.69 | no |
| SG raw-score mean       | 0.031  | 0.000  | −0.031  | t=−0.05 | no |
| SG raw-score std (var)  | 28.498 | 28.431 | −0.067 (F=1.005) | — | no |
| SG mean placement       | 2.5043 | 2.5000 | −0.0043 | t=−0.19 | no |

(Diagonal cells ct==rt are forced 2.5000 / score-mean 0 by the cand==ref calibration trap; the
off-diagonal cell in each field carries the signal. n=3600 games/cell.)

## VERDICT (honest): **H0 — FULL NULL.** The correction is null in EVERY scoring format.

The rescue hypothesis H1 is NOT supported. tau=2 moves **no** single-game metric significantly, in
either field:

- **4th-rate (last avoidance):** the most plausible rescue channel — directionally favorable (−0.0025
  over-claim field, −0.0058 selective field) but |z| ≤ 0.69, nowhere near significant (~0.4–0.6 pp
  shifts on a 0.15 base). This is the *only* metric whose sign is consistently in the "helps"
  direction, and it is statistically indistinguishable from zero.
- **Score VARIANCE:** the headline rescue candidate (the ladder story is "aggression = high variance =
  more 4ths"). Variance ratio F = var(τ0)/var(τ2) = **1.012** (over-claim) and **1.005** (selective):
  τ=2 reduces score std by ~0.2 and ~0.07 MCR points out of ~28.5 — a **0.2–0.6% reduction**,
  utterly negligible. Suppressing marginal claims does NOT meaningfully shrink the score distribution.
- **1st-rate (win proxy):** slightly *negative* in both fields (−0.006, −0.003; |z|<0.7) — τ=2 wins
  no more games and trends to fewer.
- **Raw-score mean:** −0.18 and −0.03 MCR/game; Cohen's d = −0.006 and −0.001 (t<0.3). Null, and the
  sign matches E1's micro-score story (unilateral claim suppression forfeits some chi/peng value).
- **DUPLICATE placement:** reproduces E2 exactly (−0.0029 at rt0, +0.0043 at rt2, both |t|<0.6) —
  cross-check passes; the duplicate metric is null as established.

**Why H1 fails despite the ladder intuition.** The ladder story ("aggression loses in single-game/ELO
play") would require either (a) τ=2 cutting the 4th-rate or (b) τ=2 shrinking score variance. Neither
happens here. The marginal claims τ=2 suppresses (those with logit-margin over Pass < 2) are, on net,
near-zero-EV in *both* placement and raw-score terms — so removing them neither helps nor hurts in any
format. The over-claiming bias E1 documented (mixed-data models claim ~0.30 vs expert ~0.25) is REAL,
but it is simply **not placement- or score-relevant** at this margin: the extra claims the model makes
are low-stakes, not the variance-pumping aggression the ladder narrative imagined.

**Bottom line.** This closes the rescue. The tau=2 correction's value is **not** scoring-format-
dependent — it is null on duplicate placement (E1/E2) AND null on every single-game metric (E6),
across both an over-claiming and a selective opponent field. The honest, publishable result is a clean
**double null**: over-claiming is a measurable imitation artifact (E1) with no demonstrable competitive
cost in any scoring format we tested. The earlier real-field 3.06 must be attributed to factors outside
claim-timing (matchmaking / safe_discard / noise), consistent with E2's conclusion.

## Caveats / what would change the call
- The "selective" field is moyu_bn with τ-suppression, not the actual top-10 nets (same caveat as E2).
- 3600 games/cell gives tight CIs on rates (SE ~0.6 pp) and score mean (SE ~0.67), so a 4th-rate
  improvement above ~1.5 pp or a variance reduction above ~3% would have been detectable — none was.
- Single intervention (claim suppression only, safe_discard off), single τ value (2.0); a wider τ
  sweep or a different aggression knob is out of scope here but would be the only remaining avenue.
- All numbers read from E6_RESULTS.json (per-game arrays in ckpt/e6/gates/*.npz); no hand figures.
