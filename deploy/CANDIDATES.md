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
