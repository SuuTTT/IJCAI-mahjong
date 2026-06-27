---
layout: post
title: "Why our Mahjong bot is stuck at 2nd: it over-claims"
date: 2026-06-27
---

Our Chinese-Standard-Mahjong bot **moyu** sits 11th of 16 in the IJCAI tournament. We
spent weeks trying to make it *stronger* — bigger nets, new architectures, RL, search,
cloning the top bots — and **every single lever came back null.** moyu seemed pinned at
an imitation ceiling. This post is about the diagnostic that finally found something
concrete, and the methodological trap that hid it for so long.

## The trap: wholesale cloning and self-play gates

Two things we did repeatedly, both useless:

1. **Wholesale cloning.** "The #1 bot is stronger — clone its games." We did, at scale
   (tens of thousands of decisions from the ladder leader *and* the four tournament
   finalists). **Null every time.** Behavioral cloning copies *moves*, not the value
   and search behind them; you reproduce the demonstrator's average behavior and land
   right back at its level. Agreement ≠ strength.

2. **Self-play gates.** We gated candidates by duplicate self-play *vs moyu*. This is a
   great bias-corrected strength meter — and it is **structurally blind to matchup
   gaps.** Every seat is a copy of moyu, so the gate can only answer "does this beat
   moyu in a room full of moyus?" It can *never* see "this plays differently from the
   actual opponents in a way that matters." A claim-aggression knob we tested months
   ago died on exactly this gate. It was the wrong instrument.

## The diagnostic that worked: disagreement, by decision type

Instead of cloning the leaders or gating vs ourselves, we did something surgical: **run
moyu on the leaders' *actual* decisions and ask where, specifically, it disagrees.**

The headline: moyu agrees with the tournament leaders only **~70%** of the time — not
the ~95% "dead tie" we'd seen cloning the 2025 champion. There's a real, systematic
difference. Breaking it down by decision type was the key:

| vs leader | agreement | different discard | claims-where-they-pass | passes-where-they-claim |
|---|---|---|---|---|
| Rouxqdd | 0.713 | 33% | 156 | 34 |
| Legendx | 0.733 | 30% | 143 | 66 |
| cspsept | 0.728 | 31% | 164 | 48 |
| player152 | 0.692 | 34% | 113 | 69 |

The **discard** disagreements (~30%) are roughly symmetric across tile types — that's
"both reasonable, different tile," i.e. style, not edge. But the **claim** column is
directional and consistent across *all four* leaders: **moyu takes chi/peng where they
pass about 3× more often than the reverse.** Every top bot is more selective about
claiming. **moyu over-claims.**

## Why over-claiming is a plausible root cause

In Chinese Standard Mahjong, calling chi/peng **opens your hand**: fewer fan options,
a harder time clearing the 8-fan minimum, and a committed direction. Selective,
concealed play keeps hands higher-value and more flexible. Over-claiming produces
cheap, mediocre open hands — which is exactly the shape of moyu's problem:

- It's a **2nd-place specialist** (≈28% firsts, 54% seconds, ~0% thirds, 19% fourths):
  reaches tenpai, rarely wins big.
- It's **high variance**: its four tournament-part scores were 1530 / 1286 / **1908** /
  1381 — its peak (1908) *beat the #1's best part*, but its troughs crater. Open hands
  are boom-or-bust.

Over-claiming plausibly drives **both** the win-conversion deficit and the
inconsistency. And it's the first finding that is *directional, consistent across every
leader, and measured on the real opponents* — not a guess.

## The test (the honest part)

A directional finding is a hypothesis, not a result. The fix is a one-line
inference-time rule: only keep a chi/peng if the policy prefers it over passing by a
margin τ, with τ tuned so moyu's claim-rate drops to the leaders' selective level. No
retraining.

Crucially, we are **not** gating this in self-play (it's blind here, by construction).
The only valid judge is a **real-field A/B on Botzone** against the finalists, scored on
placement / 1st-rate / variance. That test is running as this goes up.

The prior is still modest — this campaign has taught us to expect null. But for the
first time we're testing a lever grounded in a *measured, directional gap versus the
actual contest leaders*, with the right instrument. Whatever the A/B says, the lesson
stands: **when you're stuck, stop cloning the winner wholesale and stop measuring
against yourself — find the specific, directional thing you do differently, on the real
opponents' data.**
