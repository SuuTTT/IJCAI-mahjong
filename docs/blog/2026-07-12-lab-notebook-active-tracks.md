---
layout: post
title: "Lab Notebook: Every Active Track, Explained From Scratch"
date: 2026-07-12
tags: [mahjong, game-ai, evaluation, distillation, reinforcement-learning, lab-notebook]
---

> **TL;DR.** Three days after finishing 2nd of 16 in the IJCAI-2026
> Chinese-Standard-Mahjong final, the project has forked into eleven concurrent
> tracks. They serve three masters: **papers** (two AAAI-27 submissions, a ToG
> paper, a JMLR paper), the **2027 competition entry**, and the **platform**
> (an open engine + benchmark suite). This post explains each track from first
> principles — no prior context assumed — with its status and what it feeds.
> The uncomfortable headline: our own flagship in-house result is currently
> being **corrected downward**, in public, by our own re-measurement.

**Minimal shared context** (60 seconds): Chinese Standard Mahjong (MCR) is a
4-player imperfect-information game; the IJCAI-2026 competition ran on the
Botzone platform. Our agent — a behavioral-cloning (imitation) policy distilled
into a 3-model ensemble called **kdens3** — finished **2nd of 16**, losing the
final by 594 points over 12,288 games (a statistical coin flip, t = 0.13).
Because per-game luck in mahjong dwarfs skill, all our in-house evaluation uses
**duplicate format**: the same pre-dealt wall (tile sequence) is replayed with
candidate and reference bots swapped across seats, so luck cancels and only
decision differences remain. Everything below builds on, stress-tests, or
extends that setup.

---

## 1. Integrity re-gates ("fresh-wall re-gating")

**What it is.** A full re-measurement of our headline in-house result — that
kdens3 beats its predecessor `aug_s0` by +0.0055 placement points — on
evaluation data that shares *nothing* with how the model was selected.

**Why it exists.** Two flaws were found in the original measurement. To see
them, you need to know how our gates work. A *calibrated duplicate gate* plays
candidate vs reference on blocks of duplicate walls and reports the candidate's
mean placement, where placement points are (20 − Σrank)/4 per wall — an
identity that forces a perfect self-test: if candidate == reference, the score
is **exactly 2.500 with zero variance**, algebraically, not approximately. Any
drift off 2.500 means the harness itself is biased. That calibration trap is
sound and caught many bugs. The flaws were one level up. First, **overlapping
evaluation blocks**: some confirming runs re-used wall ranges that earlier runs
had already consumed, so nominally independent blocks were correlated — and
correlated samples make confidence intervals *narrower than the truth*, i.e.
overconfident. Second, **winner's curse**: kdens3 was *selected* as the best of
several KD seeds/ensembles using gates on a particular wall region, and then
its headline number was *measured* partly on that same region. When you pick
the maximum of noisy estimates and then re-quote the estimate that made it the
maximum, you inherit its upward noise. The remedy is mechanical: re-measure
every cell of every ablation table on **fresh, disjoint wall ranges** never
touched by selection.

**Status.** In flight. The pooled clean estimate so far is **~+0.0006 — 
statistically indistinguishable from zero**. The +0.0055 was real arithmetic on
real games, but as an estimate of skill it was inflated by the selection
procedure. The full re-gated table is being produced now.

**Feeds.** Corrections to all four papers — and the evaluation-wall paper
gains its strongest first-person example: a team that built calibration traps,
replications, and mechanism counters *still* fooled itself one level up.

## 2. Ladder 3-way A/B

**What it is.** Three of our bots — **kdens3**, **kdens3f2-fold** (a
finals-corpus variant), and **aug_s0** — climbing the *public* Botzone ladder
simultaneously under an identical randomized protocol, 5 games per day each.

**Why it exists.** Track 1 shows in-house numbers can be selection-poisoned.
The ladder is the antidote: nobody selects the opponents, the walls, or the
schedule, and the bots were entered *before* any outcome was known. It is the
selection-free, real-field test of whether kdens3's edge exists at all.

**Status.** Ongoing. Current ratings: **kdens3 1141, kdens3f2-fold 1062,
aug_s0 1030** — with fold out-climbing aug_s0 roughly 2:1 from equal starts.
Interesting tension: the field says kdens3 > fold > aug_s0 while the clean
in-house gate says kdens3 ≈ aug_s0. Reconciling those two readings *is* the
evaluation-wall research question.

**Feeds.** The eval-wall paper (in-house vs real-field divergence, measured
prospectively) and the 2027 entry (which base to build on).

## 3. Final2 forensics + the champion corpus

**What it is.** We harvested all **12,288** games of the final (four
finalists) and decomposed the result. The final used *duplicate walls*: each of
512 walls was replayed under all 24 seat permutations, so every bot faced the
same luck. That lets you pair outcomes wall-by-wall — compare what bot A and
bot B did *with the same tiles* — and attribute the margin to decisions rather
than deals.

**Why it exists.** To answer "what actually separated 1st from 2nd?" with
paired statistics instead of narrative, and to turn the finalists' play into
training data.

**Status.** Done. Findings: the title was a **statistical coin flip**
(t = 0.13 over 12,288 games); we **beat the champion on deal-ins** (defense —
the thing we most feared — was not the gap); the gap sits in **zimo/score
composition** (self-draw wins pay more; see track 4); and **zero errors** were
committed by any finalist in any game. Byproduct: a corpus of **723k
champion-level decisions**, re-extracted into our feature encoding with
**action-agreement 1.000** against the raw logs (i.e., the re-encode
reconstructs every recorded action exactly).

**Feeds.** The forensics feed the papers; the corpus feeds tracks 5, 6, 7,
and 8 — the entire 2027 model line.

## 4. Score-metric gate

**What it is.** A local gate that reads **two metrics at once**: mean placement
(rank per wall) and cumulative raw score.

**Why it exists.** An embarrassment with consequences: our gates optimized
**placement**, but the final ranked bots by **cumulative raw score** — and in
MCR the two diverge, because a self-drawn win (zimo) pays about **65.6 points
on average versus ~36 for a ron** (winning off a discard). Two bots with
identical placements can differ by thousands of points if one's wins skew zimo.
That is exactly the composition gap of track 3. Optimizing a proxy metric,
however carefully, is still optimizing the wrong thing.

**Status.** Shipped. The dual-metric gate is calibration-exact (self-vs-self
reads 2.500 and 0.0 score-diff), and a full **replica of the final's format
costs ~10 minutes locally**. First finding: per-game score is ~50× noisier
than placement, so placement stays the day-to-day gate — but every shipping
decision now checks both.

**Feeds.** The 2027 entry (train for the metric that ranks you) and a clean
paper example of metric mis-specification.

## 5. Corpus-KD (the 2027 base model)

**What it is.** Knowledge distillation — training a student network to match a
teacher's output distribution — re-run with the 723k-decision **finals corpus**
(track 3) mixed into the training data, gated head-to-head (paired duplicate
walls) against kdens3.

**Why it exists.** Every model we have descends from *human* game records; ~20
gated interventions failed to beat that imitation ceiling. The finalists are
the first source of demonstrably superhuman-consistency MCR play we can train
on. The question: **does champion data break the human-data ceiling?**

**Status.** In flight (multi-seed arms: mixed all-4-finalists, top-2-only,
pure BC on corpus).

**Feeds.** The 2027 competition base model.

## 6. Score-value head

**What it is.** A network trained end-to-end on the finals corpus to predict a
game's **final score from a mid-game state** — a value function for the metric
the competition actually ranks (track 4).

**Why it exists.** You cannot do metric-aware search, RL, or risk management
without an estimate of "what is this position worth in points?". Human-data
value heads were previously weak; champion games are cleaner supervision.

**Status.** Done: correlation with realized final score **r = 0.71 overall,
0.78 late-game**. That is strong enough to be a critic (track 8) and a search
evaluator, with the standing caveat from this campaign's own history: good
value *prediction* has repeatedly failed to convert into better *control*, so
it earns nothing until a gated result says so.

**Feeds.** The 2027 model line — critic initialization for the RL pilot and
the foundation for metric-aware play.

## 7. JD-v2 (trained-in defense)

**What it is.** Danger-penalized distillation: during student training, add a
loss term that penalizes probability mass on discards likely to deal into an
opponent's waiting hand ("deal-in" = discarding the tile someone wins on).

**Why it exists.** Defense heuristics bolted on at inference time all failed
their gates. The hypothesis is that defense must be *trained in*, shaping the
policy, not vetoing it.

**Status.** v1 was a **degenerate null with a diagnosable cause**: the danger
model assigned nearly the same penalty to every candidate discard in a given
state — within-state danger spread **0.0086** against a mean level of
**0.318**. A penalty that is constant across the actions you're choosing
between is a constant in the loss: **zero gradient on the decision**, so v1
trained literally nothing about defense. v2 **centers danger within each
state** (penalize *relative* danger among that state's legal discards), which
restores the gradient. In flight.

**Feeds.** The 2027 model; also a crisp paper vignette on silent no-op losses.

## 8. RL pilot

**What it is.** PPO fine-tuning of the champion-corpus policy where the reward
is **the final's own metric** (raw duplicate score / 8), with three guardrails:
a **KL leash** to the supervised policy (the policy is penalized for drifting
from the SL distribution, so it can only spend divergence where reward justifies
it), **self-play against frozen copies** (opponents are periodic snapshots, so
the learner faces a stable, competent field instead of chasing its own tail),
and **batched central GPU inference** — one server batches network forward
passes for N parallel game environments, which matters because the platform
team *measured* model inference at **~100× the cost of the environment step**;
the env was never the bottleneck, scheduling the GPU was.

**Why it exists — and why RL failed here before.** This campaign's own record:
RL fine-tunes *degraded* strong SL bases, an earlier "RL can't win" wall was
partly a **scorer bug** (the reward code was wrong, so the null was
uninterpretable), and feasible-compute full-net RL was measured at ~50
min/iteration. The three fixes attack exactly those failure modes: the KL
leash prevents catastrophic forgetting of the strong base; the metric-aligned
reward (tracks 4 and 6: score, not placement, with a critic that predicts it at
r = 0.71) means the gradient points at the actual objective; and the batched
self-play field provides sound, cheap experience. What newly enables the whole
bet is the substrate: the **Ludus engine now passes strict byte-exact
validation on all 12,288 final games** (after the `canHu` win-legality fix —
see the [engine validation feedback](../ENGINE_VALIDATION_FEEDBACK.html)), so
the environment is a verified replica of the judge, not a hopeful
approximation.

**Status.** Launching.

**Feeds.** The 2027 entry — the only track that could yield a step-change
rather than a percent.

## 9. CIFAR-N domain extension

**What it is.** Our core paper claim — **distill-then-ensemble beats
teacher-ensembling when (and only when) the imitation target is noisy** — was
established in card games. This track tests it on **real human label noise**:
CIFAR-10N/100N, image datasets re-annotated by actual crowd workers at
documented error rates (**~9% / 17% / 40%**). Design: per noise level, train 6
teachers on the noisy labels; compare **trio-averaged teacher-ensembles**
against **ensembles of distilled students** (and singles), evaluated on the
*clean* test set.

**Why it exists.** A threshold-in-noise claim from one domain is an anecdote;
the same curve on real human noise in a second modality makes it a finding.
CIFAR-N's noise is *natural* (human disagreement), not synthetic label
flipping — the exact analogue of imitating imperfect human game records.

**Status.** In flight (CIFAR-10N running, CIFAR-100N added).

**Feeds.** The distill-then-ensemble paper (domain generality) and its journal
extension.

## 10. MCR test set + platform guide (shipped)

**What it is.** The correctness infrastructure, public: the
[**mcr-final2026-testset**](https://huggingface.co/datasets/Dannibal/mcr-final2026-testset)
(all 12,288 final games as replayable engine test cases — walls, verbatim
protocol streams, expected terminals), a stdlib **validator**, a **221-game
golden edge-case subset**, and the
[Platform Developer Guide](../PLATFORM_DEVELOPER_GUIDE.html) for implementing
the judge and deploying kdens3.

**Why it exists.** Anyone building an MCR engine — including us and the Ludus
platform — needs an oracle that says "your engine is the judge" with a number
attached, not a vibe.

**Status.** Shipped — and it has its **first real save**: the Ludus engine
claimed byte-exact replay, strict validation read **0/12,288** (a missing
`canHu` win-eligibility field that is safety-critical for RL), and the fix
verified **12,288/12,288** the same day. Full story:
[Engine Validation Feedback](../ENGINE_VALIDATION_FEEDBACK.html).

**Feeds.** The platform, every future engine, and track 8's soundness.

## 11. The papers

Four in progress. Two **AAAI-27 submissions** (deadline late July): one on
**distill-then-ensemble** — the operator, its noise-threshold mechanism, and
now the CIFAR-N extension (track 9); one on **the evaluation wall** — why
in-house evaluation systematically diverges from field performance, now
carrying the re-gate correction (track 1) and the ladder A/B (track 2) as
first-person, prospectively-registered evidence. (Both are under anonymous
review preparation, so no repository links here.) Plus the **ToG** paper (the
full campaign failure-taxonomy: how each in-house eval layer inverted ground
truth) and the **JMLR** paper (the measurement toolkit and its lessons —
calibrated gates, replication discipline, the no-op-loss and OOD-value-head
pathologies). Tracks 1–4 are actively rewriting numbers in all four; the honest
version of this project is the one where the flagship +0.0055 appears with its
own correction attached.

---

## The dependency map

Three pipelines, one diagram in prose. **Integrity → papers:** the fresh-wall
re-gates (1) produce the corrected numbers, the ladder A/B (2) and final
forensics (3) supply the field-truth and paired-wall evidence, and those flow
into the two AAAI-27 submissions, ToG, and JMLR (11) — nothing submits until
the re-gated table lands. **Corpus → 2027:** the finals corpus (3) feeds
corpus-KD (5) for the base policy, the score-value head (6) for the critic,
JD-v2 (7) for trained-in defense, and — together with the score-metric gate (4)
defining the objective — the RL pilot (8), which is the 2027 entry's
step-change bet. **Infrastructure → platform:** the test set + validator +
guide (10) certified the Ludus engine (the `canHu` catch), and that verified
engine is in turn what makes (8) trustworthy — closing the loop between the
platform track and the competition track. The through-line of all eleven: every
claim gets a calibrated gate, a fresh-data replication, and a public correction
when it shrinks.
