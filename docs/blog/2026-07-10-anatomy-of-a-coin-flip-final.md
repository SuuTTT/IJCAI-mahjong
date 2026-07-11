---
layout: post
title: "Anatomy of a Coin-Flip Final: 12,288 Games Analyzed"
date: 2026-07-10
tags: [mahjong, game-ai, evaluation, duplicate-format, botzone, forensics]
---

> **TL;DR.** The IJCAI-2026 Chinese-Standard-Mahjong final (Stage 2) was 512
> duplicate walls × 24 seat permutations = **12,288 games** among 4 finalists.
> We harvested and analyzed every log. Our agent (moyu) finished **2nd**, 594
> points behind the champion — a margin of **0.13 standard errors**. Rerun the
> same final and we win it ~44% of the time. This post decomposes exactly where
> those 594 points live (spoiler: not defense — we out-defended the champion),
> why zero finalists made a single error, why the bot with the *best* mean
> placement finished 3rd, and what it would actually take to win in 2027.

All numbers below are computed from the complete set of 12,288 harvested match
logs (0 fetch errors). Nothing is estimated.

## 1. Standings (total duplicate score — the contest metric)

| # | user | total | mean/game | mean placement | 1st places |
|---|------|------:|----------:|---------------:|-----------:|
| 1 | kong | **+3923** | +0.319 | 2.4978 | 3062 |
| 2 | **moyu (ours)** | **+3329** | +0.271 | 2.4906 | 3074 |
| 3 | QiuQiuR | +1877 | +0.153 | **2.4800** | **3099** |
| 4 | player152 | −9129 | −0.743 | 2.5317 | 2853 |

Note the metric inversions already visible here: **QiuQiuR had the best mean
placement and the most 1st places, and finished 3rd**; kong won on total score
with the worst mean placement of the top 3. More on that in §5.

## 2. The headline: 1st vs 2nd was a statistical coin flip

Duplicate format lets you do a wall-level paired analysis (n = 512 walls; each
bot plays each wall 24 times, 6 per seat — the wall's own luck cancels):

| pair | Δ per wall | SE | t | bootstrap P(A ahead) |
|------|-----------:|---:|--:|---------------------:|
| kong − moyu | +1.16 | 8.74 | **0.13** | 0.557 |
| QiuQiuR − moyu | −2.84 | 10.18 | −0.28 | 0.389 |
| QiuQiuR − kong | −4.00 | 11.07 | −0.36 | 0.363 |
| kong − player152 | +25.5 | 10.11 | 2.52 | 0.994 |
| moyu − player152 | +24.3 | 9.18 | 2.65 | 0.996 |
| QiuQiuR − player152 | +21.5 | 10.77 | 2.00 | 0.976 |

The top three are **statistically indistinguishable** (|t| ≤ 0.36). The 594-pt
title margin is 0.13 standard errors. Only player152 is genuinely weaker
(t ≈ 2.0–2.7, about 1 pt/game).

Head-to-head sharpens the irony: **moyu finishes strictly above kong in 36.53%
of games; kong above moyu in 36.25%** (rest ties) — we are marginally ahead
pairwise while behind on totals. And on the 2,048 (wall, seat) cells where both
bots played the same seat on the same wall: kong better in 879, moyu better in
880, ties 289. A literal dead heat.

Why couldn't 12,288 games separate them? Per-game score SD is ≈ 27–28 for every
bot; a 0.05–0.3 pt/game edge is 0.002–0.01 SD. The 20 worst single-game
results in the whole final are all legitimate monster hands on specific walls
(one 93-fan wall cost −101 to whoever sat in the paying seat), and the
duplicate format correctly made those bombs symmetric — every finalist ate the
same ones.

## 3. Reliability: the ladder lever vanished in the final

**Every one of the 12,288 games ended properly**: 12,088 wins (98.4%) + 200
exhaustive draws (1.63%). No ERROR / TLE / invalid-move endings for *any* bot;
every per-turn verdict OK.

This deserves emphasis because the open qualification ladder looks completely
different: **15–21% of ladder games end in someone's crash or timeout**, and
that unreliability is a genuine ranking lever there. Among finalists it bought
zero differentiation. Reliability is an **entry gate**: necessary to be in the
room, worthless once everyone in the room has it.

One asymmetry worth noting: kong and QiuQiuR answered in ~9–12 ms/move; moyu
(a 3-model ensemble) took 991 ms mean, 2.3 s max — ~100× slower, yet never
near the ~6 s limit. There is ~2× time headroom for a bigger ensemble or search
*if* it buys strength.

## 4. Per-bot diagnostics (12,288 games each)

| metric | kong | moyu | QiuQiuR | player152 |
|--------|-----:|-----:|--------:|----------:|
| win rate | .2492 | .2502 | **.2522** | .2322 |
| — self-draw (zimo) | .0754 | .0734 | .0762 | .0723 |
| — off discard (ron) | .1738 | .1768 | .1760 | .1599 |
| deal-in rate | .1737 | .1693 | **.1652** | .1783 |
| mean fan on win | 12.57 | 12.47 | 12.11 | 12.66 |
| mean points per win | **44.97** | 44.48 | 43.83 | 44.86 |
| zimo win size | **65.6** | 64.9 | 62.3 | 63.5 |
| ron win size | 36.0 | 36.0 | 35.8 | 36.4 |
| mean deal-in cost | −19.98 | −19.91 | −20.40 | −19.92 |
| error games | 0 | 0 | 0 | 0 |

Score components per game (win points − deal-in payments − others' zimo
payments − bystander payments = net):

| component | kong | moyu | QiuQiuR | player152 |
|-----------|-----:|-----:|--------:|----------:|
| win points | +11.205 | +11.127 | +11.054 | +10.415 |
| deal-in payments | −3.472 | −3.370 | −3.371 | −3.552 |
| others' zimo payments | −4.702 | −4.761 | −4.768 | −4.820 |
| bystander payments | −2.712 | −2.724 | −2.762 | −2.787 |
| **net** | **+0.319** | **+0.271** | +0.153 | −0.743 |

## 5. Paired-wall decomposition: where the 0.048 pt/game vs the champion lives

The 2,048 matched (wall, seat) cells answer the exact question "what did the
champion do with our tiles?" — kong − moyu, per game:

| component | Δ (kong − moyu) | reading |
|-----------|----------------:|---------|
| win rate | −0.001 | moyu wins the SAME tiles just as often |
| win points | **+0.078** | kong's wins are worth more — more zimo (926 vs 902 conversions), and zimo pays ~3×: 65.6 vs 36.0 pts |
| deal-in rate | +0.0045 | kong actually deals in MORE |
| deal-in payments | **−0.102** | **moyu is the better defender — ahead of the champion by 0.10 pt/game on deal-ins** (16.93% vs 17.37%) |
| passive payments | **+0.072** | kong bleeds less when others win — his extra zimo ends more races in his favor |
| errors | 0.000 | both perfect |
| **total** | **+0.048** | |

A fan-size check on the 835 cells where both converted the same seat: moyu's
winning hands averaged 12.61 fan vs kong's 12.48 — **we do not build smaller
hands**. The entire gap is zimo-vs-ron composition and its passive-bleed
shadow. Not hand value. Not defense — on defense we beat the champion.

### The metric-alignment story

Put §1 and §5 together and the final becomes a story about **what each bot's
objective was aligned with**:

- **Stage 1 was Swiss**, where consistency and placement matter — and
  placement-style strength is what our duplicate gates and QiuQiuR's profile
  (best win rate, best deal-in rate, best mean placement) reward.
- **Stage 2 was raw total score**, which rewards *big* wins. Zimo pays all
  three opponents (65.6 pts average here) vs one payer on ron (36.0). kong's
  high-variance, conversion-heavy style is worth more per win under this
  metric — and that, not superior play on the same tiles, is the 594 points.
- QiuQiuR is the cautionary mirror image: **best mean placement, most firsts,
  third place** — optimized for a metric the final wasn't scored on.

## 6. What a 2027 winner needs (ranked, factual)

1. **Metric-aware value shaping (structural, biggest lever).** Top-3 separation
   (≤0.17 pt/game) is far below the noise at this n (SE ≈ 0.36 pt/game);
   winning *reliably* needs ~+0.5 pt/game, which no incremental hand-play tweak
   delivers against this field. The controllable part: train a
   final-score-aware value head that plays for the *contest metric* — pushing
   win-value when behind, folding when ahead — converting an existing defensive
   edge into total-score edge.
2. **Zimo/win-value conversion: +0.078 pt/game sits on the table.** Same tiles,
   same win rate, smaller payouts. Wait-selection that values self-draw outs
   and fan stacking on the final wait. This is an attack problem, not a defense
   problem.
3. **Passive bleed: +0.072 pt/game**, mostly the flip side of (2) — end races
   first and you stop paying. Partially irreducible.
4. **Defense: keep it.** We already out-defend the champion by 0.102 pt/game.
   Do not trade defense away chasing (2).
5. **Reliability: table stakes.** 0.000 pt/game of differentiation at the final
   — but 15–21% of ladder games still end in an error, so it remains the entry
   gate.
6. **Exploit the weak seat.** ~+1.0 pt/game exists against the weakest finalist
   for everyone; opponent modeling that squeezes harder than the field does is
   worth more than the entire top-3 spread.

**Bottom line:** moyu was one coin flip from the title, with the best pairwise
head-to-head in the final. The measurable gap to the champion is not defense
(we lead), not reliability (tied at zero), but win-conversion value — and the
decisive lever for next year is metric-aware value shaping plus
fan-value-aware attack, trained directly from the 12,288-game corpus this
final produced (~2.3M decisions of all four finalists, replay-verified).

*Campaign context — how the agent got to this final in the first place — is in
the companion post:
[From 11th to Runner-Up: An Honest Mahjong AI Campaign](2026-07-10-from-11th-to-runner-up.html).*
