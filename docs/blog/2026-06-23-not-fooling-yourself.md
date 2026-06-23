---
layout: post
title: "How Not to Fool Yourself: Verification & Reproducibility in a Competition-Bot Campaign"
date: 2026-06-23
tags: [evaluation, reproducibility, honest-ml, verification, mahjong]
---

> **TL;DR.** Over a months-long campaign to improve our Chinese-Standard-Mahjong bot, we produced
> roughly **nine results that looked like wins and weren't**. The single most valuable thing we built
> was not a model — it was the **discipline that caught them**, plus the **reproducibility** that let
> us trust, rebuild, and back up what we actually had. This post is the two halves of *not fooling
> yourself*: a verification gate that survives replication, and a setup that survives a disk wipe.

---

## Part 1 — The verification gate (a single test lies)

Our recurring failure mode has a name in this project: **the proxy that reverses.** A change looks
great on a cheap metric, and then evaporates the moment you measure the thing you actually care
about. We hit it again and again:

- **Held-out accuracy ≠ play strength.** A 256-channel imitation scored **0.767 top-1 vs the
  incumbent's 0.736** — and then *lost* by ~37 points/game head-to-head. A transformer had the
  *highest* accuracy of any architecture we tried and still didn't win more games. Accuracy proved
  nothing about play.
- **"+2.3/g" → −10/g.** A best-response RL run looked positive at small N; at large N it was ten
  points/game *below* the baseline.
- **The no-op that scored exactly 0.000.** A fine-tune "tied" the baseline to the decimal — because
  a deploy-path bug silently loaded the *base* weights instead of the candidate. The A/B was fake;
  the giveaway was an edge of **exactly** zero.
- **+1.78/g that dissolved under replication.** Our most recent "first thing to beat the incumbent"
  held at +1.78/g on one seed family — then, across a sweep of sibling models, scattered from −11 to
  +1.7, centered near parity, with a persistent seat asymmetry. The two positives were the lucky
  tail.

What stops these from reaching a conclusion is a gate built to be **hard to fool**:

1. **Bias-correction, calibrated to zero.** We score candidate-vs-incumbent in *duplicate* self-play
   — the same shuffles, seats rotated — and subtract a matched incumbent-vs-incumbent run on the
   *identical* walls. The calibration is non-negotiable: **incumbent-vs-itself must read +0.000/g.**
   If it doesn't, the harness is biased and every number is suspect.
2. **A no-op-trap proof on every run.** Before trusting a result we prove the candidate is *actually
   loaded*: distinct weight hash, distinct parameter count, and a non-zero behavioral diff vs the
   baseline. An exact 0.000 edge is treated as a bug, not a tie.
3. **Two independent seed families, both must pass.** A single family throws up lucky winners; we
   require the margin to clear the bar in *both*. The classic artifact — "+3.1 in one family, −2.4 in
   the other" — is exactly what this catches.
4. **Per-seat reporting.** Averages hide a lot. A candidate that's +6 in seat A and −2 in seat B,
   averaging +2, is not a +2 candidate — it's a seat-dependent coin flip, and we say so.
5. **Replication over cherry-picking.** One model beating the baseline is a hypothesis. *Eight*
   sibling models — varied seeds, learning rates, epochs — consistently beating it is a finding.
   When they scatter around zero instead, the single winner was noise.
6. **Real games as the gold standard.** The local simulator is the *filter*, never the verdict. The
   only result we'll act on is a candidate deployed against the **actual opponents**, scored on real
   match logs — because even a careful local gate has reversed there before.

None of this is glamorous. All of it is the difference between a campaign that concludes *"we
exhausted the lever space"* and one that ships its ninth false positive.

---

## Part 2 — Reproducibility (your result is only as good as your eval — and your model only as safe
as its backup)

**The eval gap.** For weeks we judged every change against *proxy* opponents we'd built ourselves,
and everything came back "parity." The proxies were the problem: a strong bot saturates a weak pool,
so the metric goes blind — two genuinely different candidates both read "tie" because both crush the
proxies. The fix wasn't a better model; it was an evaluation that could *see* — playing the real
field and reading the real logs. **Your result is only as trustworthy as the eval that produced it.**

**The disk wipe.** Midway through, a rented box was lost and took the cooked training data with it —
and we discovered we couldn't reproduce our own flagship model, because its recipe lived in a shell
script on an ephemeral machine that no longer existed. The lessons, now permanent:

- **Code in version control, not on the box.** The boxes are cattle; the repo is the record. (Ours
  weren't git repos at all at first — a mistake.)
- **A model card per artifact.** Git tracks code; it does *not* capture the mapping
  *(code × data × command × seed) → weights*. That mapping is the thing you actually lose. We now
  keep a per-model card: weights hash, architecture, every ordered training stage with its data hash
  and full command, and the gate/real-field numbers.
- **Big artifacts in durable object storage, hash-named.** Datasets and weights go to a hub
  (sha256-verified against the source), so a box death loses nothing. We recovered the lost champion
  dataset, archived it, and re-verified every byte.
- **Automate the stamp.** Manual discipline fails; a training wrapper that auto-writes the manifest
  doesn't.

---

## The meta-lesson

The headline result of the whole campaign is a **negative one**: against a battery of levers — supervised
variants, several flavors of RL, search, distillation, ensembles, defense, capacity, auxiliary
objectives — the incumbent is at its achievable ceiling, and the gap that keeps it off the top step
is **variance among near-equal strong players, not a fixable skill deficit.**

That's only a *trustworthy* conclusion because of the two halves above. A negative result done
sloppily is indistinguishable from "we didn't try hard enough." A negative result done with a gate
that survives replication, an eval that measures the real thing, and a setup you can rebuild from
scratch — that's a finding you can stand behind. **The methodology was the asset.**
</content>
