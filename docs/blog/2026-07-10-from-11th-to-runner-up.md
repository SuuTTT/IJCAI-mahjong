---
layout: post
title: "From 11th to Runner-Up: An Honest Mahjong AI Campaign"
date: 2026-07-10
tags: [mahjong, game-ai, knowledge-distillation, ensembles, imitation-learning, evaluation, botzone]
---

> **TL;DR.** Our IJCAI-2026 Chinese-Standard-Mahjong agent sat at an *imitation
> ceiling*: a strong behavioral-cloning policy that ~20 CI-gated interventions —
> bigger nets, offline & online RL, value-guided search, PIMC, defense
> heuristics, cloning stronger players — could not improve. Exactly one
> intervention produced a replicated, deployable win: **distill, then ensemble**
> (+0.0055 placement points, twice-replicated). It carried us from 11th on the
> pre-final ladder to **2nd of 25 in the full-field simulation**, then
> **3rd of 16 in Stage 1 of the final and 2nd of 16 overall**. This post is the
> method, the mechanism, the evidence chain — and the equally instructive list
> of everything that didn't work.

## The setting: what an imitation ceiling feels like

The agent is a 128×40 ResNet (~14M params) over a 38×4×9 state encoding,
behavior-cloned from ~5.9M human decisions with symmetry augmentation (suit
permutation, rank reflection, honor permutation — Mahjong is invariant under
all three). That augmentation was the previous best model's edge.

From there, everything ties. Not "looks worse" — *ties, under a gate that would
detect +0.005*:

- capacity (192/256/384-channel variants),
- temporal encoders (GRU, transformer),
- offline RL (AWR / critic-weighted regression),
- online RL (PPO, with and without paired-wall variance reduction),
- 1-ply value-guided reranking with a verified-engaged mechanism,
- perfect-information Monte-Carlo (PIMC) search,
- behavioral cloning of stronger players (including the previous champion),
- claim-suppression and defensive-discard heuristics.

Roughly twenty interventions, each run through the same confidence-interval
gate. The ceiling was measured, not assumed.

## The one lever that worked: distill, then ensemble

1. Train BC "teachers" that differ only by seed.
2. **Distill**: train students against the teachers' *mean softmax over legal
   actions* (dark knowledge), mixed with the human label (α = 0.7 soft, 0.3
   hard, label smoothing 0.05 on the hard term, same augmentation, same 90k-step
   budget):

   `L = α · CE(student, mean teacher softmax) + (1−α) · CE(student, human label)`

3. **Ensemble** three students at inference: average their softmax over the
   legal set, play the argmax.

The composite is small (+0.0055 placement points in duplicate format) but real:
two independent 24-block gates, CI lower bounds 2.5012 and 2.5018 against a
2.500 calibrated tie. It survives deployment untouched (numpy argmax-parity
0/300 vs. the gated policy; 0 timeouts on the live judge).

Students reach validation accuracy 0.8845–0.8855 vs. 0.878 for
identically-sized plain training — a gain that transfers to placement **not at
all** for a single student. Each student alone ties the teachers. The gain
appears only at the ensemble level.

### Why it works — and two honest surprises

The mechanism is **error decorrelation on near-tie decisions**. Individual
models make occasional idiosyncratic blunders where the top-2 actions are
close; averaging suppresses them while consensus passes through. Two findings
sharpen the picture:

- **Ensembling the teachers directly does *not* work.** Three- and four-model
  teacher ensembles tie (means 2.502–2.507, CI lower bounds below 2.500 at 24
  blocks). The distillation step is load-bearing.
- **Teacher count doesn't matter.** We scanned the teacher pool from N = 1 to
  N = 14: flat. Even **N = 1 self-distillation** — a single teacher, students
  trained on its softened outputs — produces students whose ensemble clears the
  gate. So the win is not "committee knowledge": it is the *distill-then-ensemble
  operator itself*. Training on soft targets yields students whose residual
  errors are decorrelated enough for a small ensemble to remove, in a way that
  raw seed-variant BC models' errors are not.

Saturation is sharp: six students instead of three adds +0.0007 (not
separable); mixed-capacity ensembles, a second distillation generation
("born-again"), recipe-diversified students, and teacher-softmax temperature
all land within noise of the plain 3-student ensemble.

## The evidence chain: how you learn to trust +0.005

Small edges die by measurement error, so the harness is built adversarially
against ourselves.

1. **Calibrated duplicate gate.** Same wall, candidate rotated through all four
   seats; candidate ≡ reference scores exactly 2.500 *by construction*. The
   calibration is re-verified inside every run; drift = harness bug, run void.
2. **Block CIs, replication, no peeking.** 24 blocks × 2,000 games per verdict;
   any positive is re-run on a disjoint seed range before it earns a label.
   Early 6-block "wins" routinely decay — one candidate read +0.012 at 6 blocks
   and finished below 2.500 at 24.
3. **Mechanism-engagement counters.** Every intervention must prove it *fired*
   (fraction of decisions actually altered). This is not paranoia. Our E8
   value-guided discard lookahead gated as a clean null — and the engagement
   counter showed **0% of decisions altered**: the observation builder returned
   `None` for the bot's own play requests, so the lookahead silently never ran.
   The "null" was a no-op. Rebuilt (E14), verified engaged, gated again — a
   *real* null this time. Without the counter we would have published a false
   negative about value guidance; two other "defenses" in this campaign were
   caught the same way.
4. **Deploy parity + live smoke.** fp32→numpy: 0/300 argmax flips; fp16
   storage: 0/200; two real-judge matches per build with per-move latency and
   verdict audit (worst observed 1.7 s vs. a ~6 s limit).

## Field validation: Sim-11, then the final

Against a 25-entrant simulation of the real contest field (512 games/bot,
all-play-all), the ensemble finished **2nd of 25 by official rating**
(predecessor: 11th). Forensics on all 3,584 games: best win rate in the field
(29.7%), zero errors/timeouts (one competitor crashed in all 512 of its games —
on the open ladder, 15–21% of games end in someone's crash or timeout;
reliability is a ranked lever), and the residual gap to the top concentrated in
one number: deal-in rate (16.6% vs. the leader's 12.9%). Rule-based fixes for
that number *lose* (genbutsu-style filters, even shanten-gated, cost more
offense than they save — 24-block verified); whatever closes it must be
learned, not bolted on.

Then the real thing. **IJCAI-2026 final: 3rd of 16 in Stage 1 (Swiss),
advancing to the four-team Stage 2 — and 2nd of 16 overall.** Stage 2 was
12,288 duplicate games and deserves its own forensics; the short version is
that 1st vs. 2nd was a statistical coin flip (t = 0.13). The full analysis is
in the companion post:
[Anatomy of a Coin-Flip Final: 12,288 Games Analyzed](2026-07-10-anatomy-of-a-coin-flip-final.html).

## Data

Foundation: ~5.9M human decisions (official corpus) with symmetry
augmentation. Diagnosis: 4,566 harvested ladder games plus 3,584 downloaded
Sim-11 games of the exact final field. Notably, none of the opponent data
trains the policy: cloning stronger bots reproduces their *average* behavior,
not their strength (measured null), so opponent data sharpens diagnosis while
the policy's edge still comes from the human corpus + distillation.

## Portable takeaways

1. **Distill-then-ensemble ≠ ensemble.** If your seed-ensemble ties, don't
   conclude ensembling is dead — distill first, then ensemble the students.
   One teacher is enough.
2. **Validation accuracy is not a strength metric.** +0.7% val acc → 0
   placement for singles; the gain appears only at the ensemble level.
3. **Interventions must prove they fired.** A silent no-op gates as a clean
   null. Mechanism-engagement counters caught three of them here.
4. **Budget for the confirmation, not the discovery.** Every real verdict cost
   ~50k games; every unconfirmed 6-block "discovery" would have been a false
   ship.
5. **Latency-check ensembles early.** k models = k× inference; 3×14M networks
   fit a 6 s/move budget with fp16 storage + fp32 compute, verified on the live
   judge before committing.
6. **Reliability is a ranked lever — until the final.** On the open ladder,
   crashes decide ranks. Among finalists everyone had solved it (see the
   companion post): it gates entry, it doesn't win the endgame.
