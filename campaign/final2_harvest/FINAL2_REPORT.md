# IJCAI-2026 Mahjong — FINAL Stage-2 (Final2 / 决赛第2阶段) Full Analysis

Contest `6a4eb8c71ba515095a1f1417`, finished 2026-07-09. 512 duplicate walls x 24
seat permutations = 12,288 games, 4 finalists. All 12,288 match logs harvested
(0 fetch errors) and analyzed. All numbers below are computed from the logs in
`/root/final2_harvest/` on the 202.122.49.242:23543 box — nothing is estimated.

## 1. Standings (total duplicate score, the contest metric)

| # | user | bot | total | mean/game | mean placement | 1st places |
|---|------|-----|-------|-----------|----------------|------------|
| 1 | kong | shiro v1 | **+3923** | +0.319 | 2.4978 | 3062 |
| 2 | **moyu** | **kdens3 v0** | **+3329** | +0.271 | 2.4906 | 3074 |
| 3 | QiuQiuR | 丘丘人 v4 | +1877 | +0.153 | 2.4800 | 3099 |
| 4 | player152 | player2 | −9129 | −0.743 | 2.5317 | 2853 |

moyu finished **2nd**, 594 points behind kong. Note the metric inversions:
QiuQiuR had the BEST mean placement and most 1st places but placed 3rd on
total score; kong won on total score with the worst mean placement of the
top 3 (kong plays for big scores, not placements).

## 2. The headline: 1st vs 2nd was a statistical coin flip

Wall-level paired analysis (n = 512 walls; each bot plays each wall 24 times,
6 times per seat — variance from the wall itself cancels):

| pair | Δ per wall | SE | t | bootstrap P(A ahead) |
|------|-----------|----|----|------|
| kong − moyu | +1.16 | 8.74 | **0.13** | 0.557 |
| QiuQiuR − moyu | −2.84 | 10.18 | −0.28 | 0.389 |
| QiuQiuR − kong | −4.00 | 11.07 | −0.36 | 0.363 |
| kong − player152 | +25.5 | 10.11 | 2.52 | 0.994 |
| moyu − player152 | +24.3 | 9.18 | 2.65 | 0.996 |
| QiuQiuR − player152 | +21.5 | 10.77 | 2.00 | 0.976 |

The top-3 are **statistically indistinguishable** (|t| ≤ 0.36). kong's 594-pt
title margin is 0.13 standard errors; rerun the same final and moyu wins it
~44% of the time. Only player152 is genuinely weaker (t ≈ 2.0–2.7,
~1.0 pt/game). Head-to-head confirms it: moyu finishes strictly above kong in
36.53% of games vs kong above moyu 36.25% (rest ties) — moyu is marginally
ahead pairwise while behind on totals.

Duplicate-paired per-cell comparison (2048 (wall,seat) cells where both
played the same seat on the same wall): kong better in 879 cells, moyu better
in 880, ties 289 — a literal dead heat.

## 3. Reliability: zero errors — the ladder lever vanished in the final

- **Every one of the 12,288 games ended properly**: 12,088 HU (98.4%) + 200
  荒庄 draws (1.63%). No ERROR / TLE / RE / invalid-move endings for ANY bot;
  all per-turn verdicts are OK for all four bots.
- The 15–21% ERROR-ending rate seen on the open ladder is a property of the
  weak ladder field, not of finalists. Reliability was necessary (we had it)
  but bought zero differentiation at the top.
- Response times: kong 11.5 ms mean (max 1425), QiuQiuR 8.7 ms (max 1122),
  moyu 991 ms (max 2338), player152 1187 ms (max 3776). moyu runs ~100×
  slower than kong/QiuQiuR but never approached the limit. Headroom exists
  for a bigger ensemble/search if it buys strength.

## 4. Per-bot diagnostic table (12,288 games each)

| metric | kong | moyu | QiuQiuR | player152 |
|--------|------|------|---------|-----------|
| win rate | .2492 | .2502 | **.2522** | .2322 |
| — zimo | .0754 | .0734 | .0762 | .0723 |
| — ron | .1738 | .1768 | .1760 | .1599 |
| deal-in rate (点炮) | .1737 | .1693 | **.1652** | .1783 |
| mean fan on win | 12.57 | 12.47 | 12.11 | 12.66 |
| mean points per win | **44.97** | 44.48 | 43.83 | 44.86 |
| zimo win size | **65.6** | 64.9 | 62.3 | 63.5 |
| ron win size | 36.0 | 36.0 | 35.8 | 36.4 |
| mean deal-in cost | −19.98 | −19.91 | −20.40 | −19.92 |
| error games | 0 | 0 | 0 | 0 |
| score variance | 794 | 774 | 762 | 741 |
| seat-0/1/2/3 mean | 1.11/0.24/1.03/−1.10 | **1.42**/0.19/0.82/−1.34 | 0.77/0.80/0.56/−1.52 | −0.22/0.18/−1.28/−1.65 |

Score components per game (win_pts − dealin_pay − zimo_pay − bystander_pay = mean):

| component | kong | moyu | QiuQiuR | player152 |
|-----------|------|------|---------|-----------|
| win points | +11.205 | +11.127 | +11.054 | +10.415 |
| deal-in payments | −3.472 | −3.370 | −3.371 | −3.552 |
| others' zimo payments | −4.702 | −4.761 | −4.768 | −4.820 |
| bystander payments | −2.712 | −2.724 | −2.762 | −2.787 |
| **net** | **+0.319** | **+0.271** | +0.153 | −0.743 |

## 5. Paired-wall decomposition: where the 0.048 pts/game vs kong lives

Over the 2048 matched (wall, seat) cells (exact "what did the champion do with
our tiles" comparison), kong − moyu per game:

| component | Δ (kong−moyu) | reading |
|-----------|---------------|---------|
| win rate | −0.001 | moyu wins the SAME tiles just as often |
| win points | **+0.078** | kong's wins are worth more — driven by more zimo (926 vs 902; zimo pays 3×: 65.6 vs 36.0 pts) |
| deal-in rate | +0.0045 | kong actually deals in MORE |
| deal-in payments | **−0.102** | **moyu is the better defender — beats the champion by 0.10 pt/game on deal-ins** |
| passive payments | **+0.072** | kong bleeds less when others win (his extra zimo ends more races in his favor) |
| errors | 0.000 | both perfect |
| **total** | **+0.048** | |

Fan-size check on the 835 cells where BOTH converted the same seat: moyu's
hands averaged 12.61 fan vs kong's 12.48 — **moyu does not build smaller
hands**. The whole gap is zimo-vs-ron composition and passive bleed, not hand
value and not defense (where moyu is ahead of the champion).

## 6. Variance / tails

Per-game score SD ≈ 27–28 for all bots (variance 740–794); a per-game edge
of 0.05–0.3 pts is 0.002–0.01 SD — this is why 12,288 duplicate games still
could not separate the top 3. The 20 worst single-game results are all
legitimate monster hands on specific walls (a 93-fan HU wall produced −101 for
whoever sat in the paying seat in several permutations; blocks of 66–69-fan
walls likewise). None involve errors. Duplicate format correctly made these
symmetric — every finalist ate the same bombs.

## 7. Training corpus (next year's most valuable data)

Artifacts in `/root/final2_harvest/` (box 202.122.49.242:23543):

- `final2_all.jsonl.gz` (44 MB) — all 12,288 raw match logs (players,
  initdata with srand + full walltiles, per-turn logs, fan breakdowns).
  Batched copies in `raw/batch_*.jsonl.gz`; `done_mids.txt` checkpoint.
- `table.html.gz` + `table_rows.jsonl.gz` — contest table: mid ↔ 4 bots ↔
  scores ↔ timestamp for every game.
- `final2_standings.json`, `FINAL2_DIAGNOSIS.json`, `final2_stats2.json`,
  `final2_games_summary.jsonl.gz` — per-game outcome summaries (winner,
  zimo/ron, deal-in seat, fan list, scores, errors).
- `final2_bc_corpus.npz` — **all decisions of all 4 finalists** via
  FeatureAgent lockstep replay: obs(240,uint8), mask(235), act, seat, bot id,
  decision kind (draw/claim/post-claim-discard), final seat score, winner fan,
  game idx, srand. ~187 decisions/game ≈ ~2.3M samples.
- `replay_harness2.py` — **corrected replay harness** (kept local, upstream
  untouched): stock `eval/replay_harness.py` dropped the discard embedded in
  PENG/CHI displays (state drift + 2.6% crashes), ignored BUGANG entirely,
  broke on AnGang, and mislabeled every ron as Pass. All fixed here — 0
  reconstruction failures, 0 unlabeled decisions. **Port these fixes before
  reusing the old harness on any Botzone logs.**

Teacher roles for distillation: QiuQiuR = defense teacher (lowest deal-in),
kong = conversion teacher (zimo/win-value), moyu = base policy for gap mining;
per-sample `bot` labels support per-teacher or AWR-style weighting, `srand`
supports duplicate-paired counterfactual mining.

## 8. What a better agent needs (ranked, factual)

Gap components vs the champion, in points/game, with addressability:

1. **Placement/variance shaping, not raw strength (structural, biggest lever).**
   The final metric is total duplicate score over 512 walls; top-3 separation
   (≤0.17 pt/game) is far below noise (SE ≈ 0.36 pt/game at this n). To win
   *reliably* needs ~+0.5 pt/game — no plausible incremental change to hand
   play delivers that against this field. What IS controllable: kong won the
   title with a high-variance big-hand style that happened to land ahead. A
   GRP-style final-score-aware value head (play for the contest metric,
   pushing win-value when behind, folding when ahead) converts our existing
   defensive edge into total-score edge. Training-time change; data is in the corpus.
2. **Zimo/win-value conversion: +0.078 pt/game to recover.** Same tiles, same
   win rate, smaller payouts — kong self-draws more (7.54% vs 7.34%) and his
   zimo wins average 65.6 pts vs our ron-heavy mix at 36. Addressable in
   training: wait-selection that values self-draw outs and fan stacking on
   the final wait (fan-value-aware head). NOT a defense problem.
3. **Passive bleed: +0.072 pt/game.** Mostly the flip side of (2) — when kong
   ends races first, he stops paying. Partially addressable via faster
   conversion; partially irreducible.
4. **Defense (deal-ins): KEEP — we already beat the champion by 0.102 pt/game**
   (16.93% deal-in rate vs kong 17.37%, QiuQiuR 16.52%). Defense-aware
   distillation worked. Marginal further gain vs QiuQiuR's 16.52% exists but
   is small; do not trade defense away chasing (2).
5. **Reliability/infrastructure: 0.000 pt/game at final level.** Solved, and
   non-differentiating among finalists — it gates entry to the final, it does
   not win it. moyu's 991 ms/move leaves ~2× time headroom for deeper search
   if that buys (2).
6. **Exploiting the weak seat: ~+1.0 pt/game exists vs player152** for
   everyone; any opponent-modeling that squeezes the weakest player harder
   than the field does is worth more than the whole top-3 spread.

Bottom line: moyu was one coin flip from the title with the best pairwise
head-to-head among all four. The measurable skill gap to the champion is not
defense (we lead), not reliability (tied at zero), but win-conversion value
(+0.15 combined) — and the decisive lever for next year is metric-aware
(total-score/placement) value shaping plus fan-value-aware attack, trained
straight from this corpus.
