# Deploy candidates (all fused, BN-free, torch-1.4-safe, hardened loader)

## ⭐ 2026-06-07: sim6_v1_s600 — FIRST CONFIRMED upgrade over distill100b
`deploy/incoming/sim6_v1_s600.pkl`, md5 `9c1863e3b59923c5215b332bd483682c`, 55MB.
Gentle re-distill (600 steps, lr 5e-5, champ-frac 0.3) of distill100b on 5,272 sim-6
chunjiandu-vs-diverse-top-bots decisions. Beat distill100b in TWO independent judge h2hs:
+385/60g (34-25, walls 40000+) and +515/100g (55-44, walls 50000+); 0 illegal in 160 games.
=> Upload as the ladder A/B bot (Storage cnn.pkl). distill100b stays the main-bot floor until
the ladder agrees. NOTE: code zip caiest_cnn_bot.zip (md5 064a49cb…) carries the WH fix —
re-upload it for BOTH bots regardless of model.
Recipe lesson (5 fine-tunes + 2 soups evaluated): the generalizable gain lives in the first
~600-800 steps; longer fine-tunes memorize and LOSE play strength (s2800: −388/100g).

### Full bake-off (2026-06-07, official judge, seat-rotated; net per 60g unless noted)
vs distill100b: **V1 +385 & +515/100g (CONFIRMED)** · V3B +151 · V3A +119 · V2 +31 (tie) ·
soup50 −5 (tie) · soup25 −190 · s2800 −388/100g · SAFE_DISCARD −183 (keep OFF).
vs V1: V4a (37.5k set, s1200) −40 (tie) · V4b (37.5k, s2800) −389.
=> V1 stands alone. More teacher data (12k→37.5k, 6 agents) did NOT beat the gentle sim6-only
recipe head-to-head. All checkpoints synced to deploy/incoming/ + box ckpt/.

### Diverse gauntlet + defense exam (2026-06-07 evening) — V1 passes EVERY lock gate
Rebuilt diverse opponents (modern recipe, 3 ep): g_cnn16 val .808, g_resbn24 val .859.
Paired nets/60g (same walls): V1 +402 / +136 (pool +538) vs distill100b +278 / **−59** (pool +219).
Deal-in exam (1,188 real lethal states): V1 40.7% vs base 39.8% (p=.24, equal defense).
Robustness: cold start 1.84s, warm 12ms, RSS 468MB. **LOCK: sim6_v1_s600 for the final**
(unless the overnight sl2-lineage shot — modern-recipe resbn40 + V1-style distill — beats V1).
Side-finding: fresh 3-epoch resbn24 BEATS distill100b (−59) → the modern SL recipe matters.

Same code zip for all: **`deploy/caiest_cnn_bot.zip`**. Only the Storage `data/cnn.pkl` differs.
The bot auto-detects arch from the checkpoint keys; a wrong upload plays legal-fallback (never RE).

| Upload as data/cnn.pkl | md5 | gauntlet avg net | notes |
|---|---|---|---|
| **distill100b (PRIMARY)** = current `deploy/caiest_cnn/data/cnn.pkl` | `7e45c41309502865b824f90b41a0a537` | +113 | champion-imitation; gauntlet top-tier |
| dense  = `deploy/cnn_dense_fused.pkl` | `d93d4b44c2947374c32008728a8a9ac2` | +142 | dense-reward RL; gauntlet co-top |
| resbn40 base = `deploy/cnn_resbn40_fused.pkl` | `418b6413a91300b75c12892b284b44e5` | −94 | old "best"; gauntlet LAST (A/B control) |

Recommended: deploy **distill100b** as the main bot; A/B vs **dense** and the **base** control on the
ladder (the gauntlet is a proxy — the ladder is the real yardstick). Verify md5 before each upload
(the non-fused resbn40.pkl, md5 07136e8…, is a near-identical 57MB file that causes RE — do NOT upload it).

## Update: sl2distill (stronger SL recipe + champion distill) — LADDER A/B candidate
deploy/cnn_sl2distill.pkl, md5 64827482fd8b374fe6bd3c63ab059545, 57350483 bytes.
Better CLEAN metrics than distill100b: val_acc 0.887 (true held-out), champion agreement 0.767 (vs
0.730). Gauntlet TIED with distill100b (flips by seed — too noisy to separate). Plausibly better but
unconfirmed in play => A/B on the LADDER vs distill100b. distill100b stays the locked floor until the
ladder decides. Same code zip; upload as a 2nd bot's Storage cnn.pkl to compare.

### Big-N fresh-wall confirmation (2026-06-08): V1 edge is REAL but SMALL (de-hyped)
300 games walls 90000+ (box A, 0 illegal, 0 draws): V1 +36/300g = +12/100g — a marginal
edge, effectively a TIE vs its near-twin parent. Earlier +385/60g & +515/100g were partly
WALL-LUCK. (Box B walls 100000+ wedged at game 31 — env flakiness, not a bot bug; box A ran
300 clean.) HONEST LOCK RATIONALE: V1 ≈ distill100b head-to-head (expected monoculture tie),
but V1 is clearly better vs the DIVERSE gauntlet (+538 vs +219) with equal defense — and the
gauntlet ≈ the real ladder. V1 remains the pick as a safe, weakly-positive swap; distill100b
is an equally-safe fallback. Either way: upload the WH-fixed zip (064a49cb…).

### 2025-FINALIST GAUNTLET (2026-06-08) — the decisive on-distribution eval
Built 6 imitation nets of REAL 2025 finalists (BC on extracted decisions, val 0.74-0.84):
test1/selfregpo/pama/moumou/laigebao/bot32. Each candidate played all 6 (36g each, walls 200000+).
TOTAL net: **distill100b +5471 > V1 +4975 > champ2025 +4905.** distill100b beat V1 on 5/6 opponents
(+496 over 216g, consistent). This is the MOST on-distribution eval we have (real field, not our CNNs).
- Contradicts the official-CNN gauntlet (V1 +538 > distill100b +219). Sign of "V1 vs distill100b"
  FLIPS by opponent pool → non-transitive → neither robustly better (matches the near-twin 300g tie).
- champ2025_test1 (distill100b distilled toward the 2025 champ, agreement 0.809) is WORST — distilling
  on a stronger/on-distribution teacher did NOT beat the floor. Clean negative.
=> **REVISED LOCK: distill100b (the proven floor) — it wins the most on-distribution eval and has the
   most production history. V1 is an equal-risk fallback (no worse; won the official-CNN gauntlet).
   Do NOT ship champ2025.** The WH-fixed zip (064a49cb) remains the one unambiguous must-ship.

### P2 inference-time ENSEMBLE (2026-06-08) — beats the average, not the best
3-model NumPy mixture (distill100b+V1+champ2025), softmax-averaged. Memory 323MB (< single torch
468MB), cold 0.85s — deployable. Finalist gauntlet: **ensemble +5368 < distill100b +5471** (≈ tie,
−103/216g), but ABOVE the naive member-average +5117 (decorrelation gave ~+250). Conclusion: mixing
helps, but distill100b already dominates this pool so averaging in weaker-on-pool members dilutes it.
No robust win. **LOCK stays distill100b.** Ensemble is a deployable ≈-equal alternative, not an upgrade.
(Stretch: a distill100b+V1 2-model or strength-weighted mix might edge it — marginal, untried.)

### P4b 2025-ONLY base (2026-06-08) — clean negative (data scale dominates)
40-block BC on all 16 finalists (1.34M decisions, no official data), val 0.826. Finalist gauntlet
TOTAL +2348 vs distill100b +5471 (~43%). Pure on-distribution at 1.34M < official 5.87M: data
QUANTITY dominates the on-distribution benefit. Filed as a diverse gauntlet member (md5 dd1993b1),
not a candidate. The real P4 test is base2025 (official 5.87M + 2025 1.34M = full scale + on-dist).

### P4 base2025 (official 5.87M + 2025 1.34M, modern recipe) — NEGATIVE; P4 exhausted
Best val profile of any candidate (official 0.868, 2025 0.733) yet gauntlet TOTAL +4642 < distill100b
+5471 (−829/216g). Higher val-acc, LOWER play — the campaign's "agreement != play" law again. Mechanism:
BC on 16 finalists averages CONFLICTING strategies (predictive but muddled); distill100b imitates ONE
coherent strong policy -> coherence > diversity for play. Confound (noted, not used to overturn): base2025
shares training data with the imitation opponents, so the gauntlet may under-rate it; only the ladder resolves.
=> distill100b stays the lock. base2025 (md5 d69c9de9) is a reasonable LADDER A/B candidate if desired.
P4 EXHAUSTED: stronger SL base on official+2025 does not beat the focused-distillation floor.

## ============ FINAL BAKE-OFF SUMMARY (2026-06-08) ============
On-distribution finalist gauntlet (6 real-2025-finalist imitations, net/216g):
  distill100b +5471  >  ensemble +5368  >  V1 +4975  >  champ2025 +4905  >  base2025 +4642  >  base2025only +2348
LOCK = **distill100b** (deploy/caiest_cnn/data/cnn.pkl, md5 7e45c41...) + the WH-fixed zip (064a49cb...).
deploy/ship/ holds the zip + 3 swappable Storage models for LADDER A/B (the only ground truth left):
  cnn_distill100b.pkl (primary), cnn_v1.pkl (won self-play h2h), cnn_base2025.pkl (best val; gauntlet-confounded).
Nothing in-house robustly beats distill100b — matches the field/PKU-thesis ceiling. Real remaining gains:
ladder data, and (untried) P3 test-time search.

### distill_raven (distill100b → raven #1 self-play) — gauntlet NEGATIVE, but UNDERPOWERED + gauntlet-biased
Gauntlet TOTAL +4809 < distill100b +5471 (md5 66858558). Closest-ish but below. TWO caveats that
matter: (1) UNDERPOWERED — only 41/100 raven games (1,569 decisions) + gentle 400-step distill
(agreement 0.684->0.706, small move). (2) The gauntlet opponents are WEAK 24-block imitations, so the
gauntlet rewards raw exploitation of weak bots — which distill100b (chunjiandu-tuned) does best — and
may UNDER-rate models tuned for the strong REAL ladder. Ladder evidence contradicts the gauntlet:
distill100b is only ~2nd on the real ladder (8 games, mostly 2nd, 1 win). => DO NOT over-trust the
gauntlet as ground truth. Ship distill100b (safe floor) but A/B V1 + distill_raven on the real LADDER.
Get the FULL 100 raven games for a properly-powered raven distill.

### oracle_student (Suphx oracle-guiding, 1.2M subset) — gauntlet +4893, NULL lever
Gauntlet +4893 < distill100b +5471 (md5 22c86273). Public student fully recovered the oracle teacher's
val_acc (0.833 vs 0.827) -> opponents' hidden hands add ~0 predictive value for DISCARDS -> oracle-guiding
is null for SL (an RL value/defense lever per Suphx). Filed; not a candidate.

## ============ ALL IN-HOUSE LEVERS EXHAUSTED (2026-06-08) ============
Gauntlet vs distill100b +5471 (on-distribution, 6 real-finalist imitations): ensemble 5368, V1 4975,
champ2025 4905, oracle_student 4893, raven 4809, base2025 4642, base2025only 2348. NONE beat the floor.
Also null: RL x5 (parity), soups, 8-fan mask, safe-discard, JAX throughput (no leap), oracle-guiding.
LOCK = distill100b + WH-fixed zip (97b88497). Only unresolved levers need USER data: full-100-raven
distill (underpowered near-miss at 41 games) + the real LADDER A/B (the gauntlet has weak-opponent bias;
8 ladder games show distill100b ~2nd). The bot is at its achievable ceiling barring those.

### STRONG gauntlet re-judge (2026-06-09, 40-block finalist imitations) — lock CORROBORATED
distill100b +4996 > V1 +4821 > raven +4779. SAME ranking as the weak gauntlet (distill100b > V1 >
raven) -> the weak-opponent-bias did NOT hide a better candidate; distill100b is genuinely best.
BUT the gap NARROWS vs stronger opponents (V1 -496->-175, raven -662->-217) = robustness signal:
V1/raven get relatively more competitive as the field strengthens. The real ladder is stronger still
-> gap could narrow/flip there -> ladder A/B remains the decider. Lock = distill100b, now well-supported.
