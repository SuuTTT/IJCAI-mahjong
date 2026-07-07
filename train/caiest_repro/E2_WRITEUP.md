# E2 — Is the tau=2 claim-suppression PLACEMENT benefit opponent-dependent?

**Question.** E1 found the tau=2 claim-suppression correction is null/negative for placement
when opponents are moyu_bn OVER-claimers. The real-field A/B had tau=2 scoring 3.06 vs selective
finalists. Hypothesis: the correction benefit appears only when OPPONENTS are SELECTIVE (claim less).
E2 tests this by sweeping opponent selectivity directly.

**Design.** Duplicate-format placement gate (e2_gate.py, derived from e1_gate.py). Candidate =
moyu_bn_128x40 (canonical policy), reference/opponents = moyu_bn_128x40. We added `--ref-tau`,
which applies the SAME claim-suppression rule (keep chi/peng only if logit[claim]-logit[Pass] >= tau,
else Pass) to the 3 OPPONENT seats. Matrix: candidate `--claim-tau` in {0,2} x opponent `--ref-tau`
in {0,1,2,3} = 8 cells. Each cell = 3 wall-seed blocks (seed0 70000/80000/90000) x 300 seeds x 4
seat-rotations = 3600 duplicate games. Placement points: 4/3/2/1 for rank 1/2/3/4 (avg-rank on ties).

**--ref-tau verification (sanity check it works).** As ref_tau rises, opponents claim strictly less
(measured kept claim-rate over claim-legal states, candidate excluded):

| ref_tau | ref claim-rate (raw argmax) | ref claim-rate (KEPT after tau) |
|--------:|----------------------------:|--------------------------------:|
| 0       | 0.336                       | 0.336 (no-op, raw==kept)        |
| 1       | 0.334                       | 0.303                           |
| 2       | 0.332                       | 0.269                           |
| 3       | 0.332                       | 0.234                           |

ref_tau monotonically lowers the opponents' claim-rate. At ref_tau=2 the opponent field claims
0.269 -- essentially the real top-10 selective claim-rate (about 0.26). ref_tau=2-3 is the SELECTIVE
field; ref_tau=0 is the over-claiming field (matches the E1 setup). --ref-tau confirmed working.

## 8-cell placement table (mean +/- std over 3 wall-seed blocks)

| ref_tau (opp selectivity)  | cand_tau=0 (raw)  | cand_tau=2 (corrected) | correction benefit (ct2 - ct0)   |
|---------------------------:|------------------:|-----------------------:|---------------------------------:|
| 0  (over-claim, kept .336) | 2.5000 +/- 0.0000 | 2.4971 +/- 0.0181      | **-0.0029** +/- 0.0181 (t=-0.28) |
| 1  (kept .303)             | 2.5002 +/- 0.0124 | 2.5021 +/- 0.0080      | **+0.0019** +/- 0.0151 (t=+0.22) |
| 2  (selective, kept .269)  | 2.4957 +/- 0.0129 | 2.5000 +/- 0.0000      | **+0.0043** +/- 0.0129 (t=+0.58) |
| 3  (very selective, .234)  | 2.5128 +/- 0.0056 | 2.5115 +/- 0.0124      | **-0.0012** +/- 0.0177 (t=-0.12) |

(cand=ref=moyu_bn; cells where cand_tau==ref_tau are exactly self-identical policies, so they tie at
2.500 by construction -- the calibration trap -- and they pass: ct0/rt0 = 2.500, ct2/rt2 = 2.500.)

## Correction-benefit curve vs opponent selectivity

```
ref_tau:   0        1        2        3
benefit: -0.0029  +0.0019  +0.0043  -0.0012
         (rt0)    (rt1)    (rt2)    (rt3)
```

The benefit does rise from negative (rt0) to its max at rt2 and is positive across rt1-rt2 -- the
predicted SIGN pattern. BUT the magnitudes are about 0.001-0.004 placement points, while the
per-block noise is +/-0.012-0.018. Paired t-tests (df=2): every cell |t| <= 0.58, far below
t-crit(.05, df=2) = 4.303. Every benefit sits within 0.34 standard deviations of zero. Pearson
r(benefit, ref_tau) = 0.299 (weak, not significant). The "rise" is consistent with pure sampling noise.

## VERDICT (honest)

**The tau=2 claim-suppression PLACEMENT benefit is NOT supported, even vs selective opponents.**

- **Opponent-dependent? Sign-suggestive, magnitude null.** The benefit's *sign* tracks the hypothesis
  (negative at over-claiming rt0, positive and largest at the selective rt2 that matches the real
  field's ~0.26 claim-rate). This is the only crumb of support for the real-field story. But the
  effect is statistically indistinguishable from zero in every cell (all |t| < 0.6) and roughly 10x
  smaller than the noise floor. We CANNOT claim an opponent-dependent benefit on this evidence.
- **Does tau=2 help vs selective opponents?** No measurable help. At the real-field-matching ref_tau=2
  the benefit is +0.0043 +/- 0.0129 placement points -- positive in expectation but a coin-flip given
  the spread. Not a win.
- **Does it match the real field (ref_tau~2 -> benefit>0)?** Directionally yes (benefit>0 at rt1, rt2),
  but the effect is far too small and noisy to "rigorously explain" the real-field 3.06.
  **The most parsimonious read: the real-field 3.06 vs selective finalists was largely noise** (or
  driven by factors outside claim-timing, e.g. safe_discard / matchmaking), not by the tau=2 claim
  correction. We do NOT have evidence to predict a Sim-11 win from claim-suppression alone.

**Bottom line:** E1's null is robust. Making opponents selective shifts the correction benefit's sign
in the hypothesized direction but never to a statistically real magnitude. Report as a NULL: the
tau=2 placement benefit is unsupported; if a selective-field benefit exists it is below ~0.005
placement points and would need far more seeds (and ideally distinct selective opponents, not
tau-suppressed moyu) to detect.

## Caveats / what would strengthen this
- The "selective" field here is moyu_bn with tau-suppression, NOT the actual top-10 nets. A genuine
  selective opponent (a different policy, not just claim-clipped moyu) could behave differently.
- cand==ref means the cand_tau==ref_tau diagonal is a forced 2.500 tie; off-diagonal cells carry the
  real signal and all land within noise of 2.500.
- 3 wall-seed blocks give a clean mean but only df=2 for the paired test; the point estimates are
  tight (3600 games/cell) yet the cross-block std dominates the ~0.004 effect. More blocks would
  shrink the SE but the effect size itself is negligible.
