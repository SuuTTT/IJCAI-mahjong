---
layout: post
title: "From a knob (τ) to a value model to RL: optimizing for special points"
date: 2026-06-29
---

We found our Mahjong bot **moyu** over-claims (calls chi/peng too eagerly), and that
claiming less improves its standing under the contest's *special-points* scoring. This
post explains the three tools we used to exploit that, in increasing sophistication:
the **τ knob**, the **value/reward model**, and **RL that uses the value model as its
critic**.

## 1. τ — the claim-suppression knob

moyu is a neural net: given the game state it outputs a score (a "logit") for each of
the 235 possible actions, and normally it just takes the highest-scoring legal one.

When the highest-scoring action is a **claim** (chi/peng) and **Pass** is also legal, we
intercept it with one rule:

> keep the claim only if `logit[claim] − logit[Pass] > τ`; otherwise Pass.

**τ is the margin a claim must beat passing by to be taken** — a single dial for
aggression:
- τ = 0 → moyu's raw behaviour (claim whenever it's the top action). Claim-rate **0.29**.
- τ = 1 → claim-rate **0.25** · τ = 2 → **0.22** · τ = 3 → **0.19**.

We swept it on the real ladder, scored by special points per deck (4/3/2/1 by placement):

| τ | special-pts/deck |
|---|---|
| 0 (raw moyu) | 2.90 |
| 1 | 2.99 |
| **2** | **3.06** |
| 3 | 2.93 |

A clean rise-then-fall: **the optimum is moderate suppression (τ=2)** — claim less than
moyu does, but not *too* little (τ=3 over-folds and loses value). The strongest human
players claim ~0.25; moyu was at 0.29. τ is the crude, zero-training way to fix that.

## 2. The value (reward) model — what it predicts

τ is blunt: one threshold for every situation. A smarter version asks, per state,
*"would claiming actually improve my expected result here?"* For that we need to predict
outcomes — a **value model** (a.k.a. reward model).

- **Input:** the game *state* — the same 38-plane `(38,4,9)` tensor the policy sees
  (your hand, all four players' discards, their melds, the winds) plus the legal-action
  mask. It does **not** take an action — it scores the *state*.
- **Output:** what the current deal will be worth to *you*:
  - `V_place` — your placement this deal (toward special points 4/3/2/1),
  - `V_4th` — the probability you finish **last** this deal,
  - `V_score` — your raw MCR score this deal.
- **Training data:** the official 98k-game set, where every decision is labelled with how
  that deal actually turned out for the player who made it.

It works, and capacity matters — the 256-channel model reached **4th-place AUC 0.955**,
placement accuracy **0.75**, score correlation **0.67** (held-out). Crucially this is
*real* signal, unlike an earlier win-probability head that was near-chance (0.55). And it
**independently confirms the lever**: across 119k claim-legal states it says "claiming
helps" in only ~46% of them — i.e. claim *less* — agreeing with τ and with human play.

This unlocks the **value-guided claim**: replace the fixed τ with `V(after-claim) −
V(after-pass)` evaluated per state — claim only when the value model says it raises your
expected placement. A per-situation optimum instead of one global knob.

## 3. Using the value model to train RL

The value model is also exactly what **reinforcement learning** needs: a **critic**.

Policy-gradient RL improves a policy (the *actor* = moyu) by increasing the probability of
actions that did **better than expected**. "Better than expected" is the **advantage**:

> `A(state, action) = (what actually happened) − V(state)`

where *what actually happened* = the special points the deal yielded, and `V(state)` =
the critic's prediction. Positive advantage → that action beat the baseline → push the
policy toward it (leashed to moyu so it doesn't drift off a cliff).

**Why this matters here:** we tried RL several times and it was always null. The autopsy
showed *why* — the **critic was broken** (its predictions had negative R², so the
"advantage" was pure noise, and the policy just random-walked). The blunt instrument
isn't the actor or the reward; it was the critic. **Now we have a critic that genuinely
predicts placement (4th-AUC 0.955).** So the advantages are real for the first time, and
the policy update can actually point uphill — toward **special points**, the contest's
own metric.

Our first version is *offline* (no slow self-play simulator): compute advantages on the
fixed dataset using the value model, then do advantage-weighted updates to moyu (weight ∝
`exp(β·A)`). It's running now. The honest prior is still modest — RL has a long null
record on this game — but for the first time the thing that broke it is fixed.

## The arc

`τ` (a hand-set knob) → `value model` (learn what each state is worth) → `value-guided
claim` and `RL` (let the learned value drive the policy). Each step is less hand-crafted
and more optimal, all aimed at the same target your own play and our data both point to:
**stop over-claiming; optimize placement, not wins.** Whether RL finally clears the bar
that the simple τ=2 overlay already passes (3.06 > 2.90) is the open question — but now
it's a fair test.
