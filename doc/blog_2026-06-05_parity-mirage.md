---
title: "Parity Was a Mirage: How Fixing Our Yardstick Revealed the RL Had Been Working All Along"
date: 2026-06-05
description: "For weeks every reinforcement-learning and distillation experiment we ran 'tied' our supervised base — the classic parity trap. Then we questioned the ruler instead of the runner. By scoring models against a DIVERSE set of opponents instead of their own near-twin, the picture flipped: our supervised 'best' is actually last, and the RL/distilled models we'd written off are stronger. The lesson the Mahjong winners already knew — you can't measure strength against a monoculture. Plus a deploy war story about two 57MB files that look identical and aren't."
layout: "post"
showTableOfContents: true
math: true
katex: true
tags: ["research", "mahjong", "reinforcement-learning", "self-play", "evaluation", "distillation", "game-ai", "botzone", "deployment"]
---

{{< katex >}}

## TL;DR

In the [last post]({{< relref "2026-06-04-reaching-for-rank1-parity-trap-distillation-mahjong-winners.md" >}}) we reported that every RL variant we tried — pool self-play, a full AlphaStar-style league with exploiters, even Suphx's dense-reward predictor — converged to **parity** with our supervised CNN. We concluded RL was exhausted. **We were wrong — not about the numbers, but about what they measured.**

- **"Parity" was an artifact of the ruler.** We were scoring each candidate against a frozen copy of *itself* (the supervised base it was fine-tuned from). Two near-identical strong policies trade evenly almost by definition. The metric was blind.
- **We built a diverse gauntlet** — scoring each model against five *different-architecture* opponents instead of its own twin.
- **The picture flipped.** Across two independent runs, our supervised "best" (`resbn40`) finished **dead last**. The dense-reward, distilled, and pool-RL models all beat it. RL didn't fail; it made the policy **more robust against varied opponents** — exactly what a monoculture metric cannot see.
- **Bonus deploy war story:** the live bot crashed (RE) on every game. Cause: two model files that differ by 0.3% in size and are otherwise indistinguishable. We fixed it and hardened the bot so a bad upload can never hard-crash again.

This post explains the **parity trap**, why our measurement was lying to us, how the SOTA systems avoid it, and the new roadmap.

---

## What "parity" actually means

When you train a game-playing policy with reinforcement learning, you need an opponent. The simplest choice is a frozen snapshot of the policy you started from — your supervised baseline. You let the learner play against that fixed opponent and reward it for winning.

The **parity trap** is what happens next: the learner's score against that frozen opponent climbs for a while, then flattens out at roughly even — *parity*. It looks like the model stopped improving. Two things cause it:

1. **You're optimizing against one specific opponent.** The learner discovers the frozen baseline's particular quirks and exploits them, then has nothing left to learn. It becomes good at beating *that* opponent, not at playing well in general.
2. **Mahjong strategy is non-transitive.** Like rock-paper-scissors, style A can beat B, B beats C, and C beats A. There is no single linear "strength" ladder. So "win-rate against one fixed style" is a narrow, misleading measurement — a policy can get better in general while its score against one frozen twin stays flat.

The standard *fix* for the trap is a **population**: train against many opponents (a league of historical checkpoints plus dedicated "exploiters"), and keep the policy anchored to the supervised base so it doesn't drift into a brittle, exploitable style. We built all of that. And it *still* read as parity.

## The bug wasn't the training. It was the test.

Here's the part that took us too long to see. We were running the league and the dense-reward training correctly. But our **evaluation** — the number we used to decide "did this help?" — was still *candidate vs. its own frozen base*. A monoculture. Every model we compared was a slightly-perturbed descendant of the same supervised CNN, so of course they all traded evenly. We were measuring twins against twins and concluding "no difference."

The tell was hiding in plain sight in our own history: when we compared `resbn40` against a genuinely *different* model — the old 16-block CNN, or the original MLP — the gap was huge and obvious (hundreds of points, decisive win-rates). It was only the *RL-vs-its-own-base* comparisons that tied. Diversity separates; monoculture doesn't.

## How the winners measure

This is not a new lesson — the strong systems bake it into their design:

- **Suphx** (Microsoft's Mahjong AI) measured itself on **Tenhou**, a live server with thousands of diverse human and bot opponents. Its RL gains were validated by *actual rank*, never by self-play against a twin.
- **League training** (AlphaStar-style) makes the *measurement* and the *method* the same thing: an agent's strength is its win-rate against a **diverse population**. The diversity is the ruler — a better policy beats *more of a varied pool*.

The duplicate tournament format used in the contest (same tiles dealt across rotated seats) removes *luck* variance — but it does nothing about the monoculture problem. You still need varied *opponents*, not just varied deals.

## The diverse gauntlet

So we built one. Five opponents spanning genuinely different architectures and play styles — a 16-block CNN, a CNN+attention hybrid, a 24-block ResNet, a 56-block ResNet, and a wide 192-channel net — and scored each candidate by its total net points across all of them, seats rotated to cancel position bias, run through the official judge.

Then we ran it twice, with independent random seeds, on five candidates: the supervised base, the dense-reward RL model, the league model, the pool-RL model, and the chunjiandu-distilled model.

| Candidate | Run A (net) | Run B (net) | avg |
|---|---:|---:|---:|
| dense-reward RL | +194 | +89 | **+142** |
| chunjiandu-distill | +82 | +143 | **+113** |
| pool-RL | −7 | +101 | +47 |
| league | +74 | +19 | +47 |
| **supervised base (`resbn40`)** | **−150** | **−37** | **−94** |

**The supervised base — the model we'd been calling "best" and were about to ship — is last in both runs.** Every model that "tied" it in the blind self-vs-twin test actually *beats* it against diverse opponents. The exact ordering of the top three shuffles within noise, but "base is worst" is rock-solid across both runs and both win-counts.

The interpretation matters: **RL and distillation were working the whole time.** They didn't make the policy better at beating its own clone (that's the parity trap, and it's real). They made it more **robust** against *varied* styles — which is exactly what the ladder rewards and exactly what a monoculture metric is blind to.

## A deploy war story: the two files that look the same

While this was running, the live bot started crashing — a runtime error (RE) on the very first move of every game, which on the contest scoring means −30 points, every time. Catastrophic.

The cause was almost comically subtle. Our deployable model is a *fused* network (BatchNorm folded into the convolutions, so it loads on the platform's ancient PyTorch). The fused file is **57,350,483 bytes**. The *non-fused* original is **57,534,631 bytes** — a 0.3% difference, both round to "57 MB," and they have identical names in different folders. The wrong one had been uploaded. Loading non-fused weights into the fused network is a key mismatch → an uncaught exception → exit code 1 → RE.

Two fixes:
1. **The right file** (we now pin and check its md5 before upload).
2. **Hardening so this can never be catastrophic again.** The bot now infers its architecture *from the checkpoint's own weights*, tries multiple architectures, and — if nothing loads — falls back to playing a guaranteed-legal move instead of crashing. A bad upload now costs *strength*, not a −30 every hand.

The meta-lesson, again: **the deployment runtime is part of the model.** A serialization flag and a 0.3% file-size difference stood between a working bot and one that auto-loses every game.

## The roadmap

The full method ledger lives in the repo (`doc/METHODS_TODO.md`). The short version, reprioritized around the measurement insight:

**P0 — Measurement (done / in progress)**
- ✅ Diverse gauntlet — built, and it already flipped our conclusions.
- ⬜ **Ladder signal** — the real external yardstick (the gauntlet is a strong proxy, but its opponents are still CNN-family). Deploy and play ranked games.
- ⬜ Wire the gauntlet into the RL promotion gate, so training optimizes for *real* strength instead of parity-noise.

**P1 — Deploy** — ship the gauntlet-best model (the distilled champion-imitation), with the hardened, crash-proof loader.

**P2 — Data** — collect games of the #1 bot playing *diverse* opponents (not just self-play); that's the lever that pushes imitation past its current ceiling.

**P3 — Untried methods, now that we can measure them** — fan-potential input features; dynamic entropy; and the big one, **oracle guiding** (train a teacher that sees the hidden tiles, then wean it off).

## What's next

The single highest-value thing now is the **ladder signal**. The gauntlet told us our rankings were upside-down; the ladder will tell us by how much, against the opponents that actually count. We deploy the gauntlet-best model, play ranked games, and — for the first time in this project — get a strength number that isn't blind.

Everything downstream depends on it. With a real ruler, the untried research methods (oracle guiding especially) become *worth trying*, because we'll finally be able to tell whether they helped. Without it, we'd just be back to measuring twins against twins.

The honest summary of this stretch: we didn't find a new training trick. We found that **we'd been grading the exam with the answer key upside-down** — and fixing that revealed weeks of "failed" RL had quietly been succeeding.

### Lessons so far

- **When everything ties, suspect the metric before the method.** A monoculture eval cannot distinguish strong policies; it will report parity forever.
- **Diversity is the ruler.** Strength in a non-transitive game is only meaningful against varied opponents — that's why the winners measure on populations and ladders.
- **RL can succeed invisibly.** It improved robustness, not twin-vs-twin win-rate; only a better yardstick made the gain show up.
- **The deployment runtime is part of the model** — a 0.3% file-size difference can mean a bot that auto-loses every game.
