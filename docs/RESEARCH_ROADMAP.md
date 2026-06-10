# Research roadmap — methods we have no time to implement before 06-14

*For when the deadline levers are exhausted: the ambitious directions worth pursuing for a strong
final bot and a journal-quality writeup (à la a "Tree-of-Games" / search-augmented-LLM-agent style
paper). Each entry: the idea, why it could beat a strong imitation policy, the build cost, the
compute, and the open risk. Ordered by expected payoff. Companion to `HANDOFF_TO_WIN.md` (the
deadline plan) and `ijcai-mahjong-state.md` (history). Read those first.*

## The core thesis this roadmap addresses
We have hit the **imitation ceiling**: cloning the rank-3 bot (`lad_chunjiandu`, +4119 gauntlet)
cannot beat #1/#2, and every offline lever tried (RL ×4, advantage-weighted BC, Q-rerank, 1-ply
V-search) is at or below that ceiling. To *exceed* a strong policy you need either (a) **search**
that plans beyond one forward pass, or (b) **RL at a scale we can't reach on CPU self-play**, or
(c) a **fundamentally richer learning signal**. This roadmap is those three, made concrete.

---

## R1 — JAX/GPU-vectorized CSM env + small-net self-play RL  ★ the "real RL" path
**Idea.** Reimplement the full Chinese-Standard-Mahjong env in JAX (state as fixed arrays, branchless
transitions, `jit`+`vmap` over thousands of parallel games on GPU), train a **small** policy/value
net by self-play RL (PPO/MuZero-style) at 10⁴–10⁵ steps/s, then **distill the RL-improved small net
into the big deploy net** (or use it as the search value/rollout net).
**Why it can win.** This is how the 2020 IJCAI champion (pure RL) and AlphaZero-class agents exceed
imitation — enough self-play to discover strategy the data never showed. Our RL failed only at
~24k-game scale; PGX-scale is ~10⁶–10⁹ games.
**Build cost.** HIGH — multi-week. The killer is the **81-fan scoring calculator in branchless JAX**
(win-detection/shanten is tractable; full fan scoring is not, quickly). Framework template: **PGX**
(sotetsuk/pgx, has Sparrow-Mahjong but not CSM); **Mjx** is fast C++ but riichi rules.
**Compute.** With a fast env + a 4-block net: ~5k games/s/GPU → **8× rented A4000/3090 ≈ ~B games in
2–3 days** (genuinely Suphx-scale). The 40-block net stays forward-bound (probed: ~246 g/s) — small
net is mandatory for the throughput.
**Risk.** Env engineering time; sim-to-real gap (in-house RL gains didn't convert before — the
vectorized env + true reward must match the official judge); whether even B games beats strong SL.
**Verdict.** Right long-term; impossible in 4 days (engineering-bound, not compute-bound).

## R2 — PIMC / IS-MCTS: full determinized search at decision time  ★ best non-RL exceed-the-ceiling
**Idea.** Perfect-Information Monte Carlo: sample N plausible full hidden states (opponents' hands +
wall) consistent with all observations, solve/rollout each with the policy, aggregate (majority or
expected-value vote). Information-Set MCTS is the principled version (tree over information sets,
determinization at the leaves). Use the **value head (r=0.73)** at leaves and the policy as the
rollout/prior.
**Why it can win.** Search plans several plies ahead and explicitly weighs deal-in risk and fan
value across sampled futures — strictly more than the 1-ply V-search we tried (which was budget-
limited to ~k forwards and ignored opponent responses). This is the canonical way card-game AIs
(bridge, Skat, Hanabi) beat their policy nets.
**Build cost.** MEDIUM-HIGH — needs a deploy-embeddable game simulator + a belief model (how to
sample opponents' hands given discards/melds). The belief model quality dominates results.
**Compute.** Inference-time, CPU. The blocker is Botzone's **~6 s/turn on weak CPU** — a 40-block
rollout is ~0.3 s, so only ~15–20 forwards/turn. Needs a **tiny fast rollout/value net** (distilled
from the big one) to afford real tree width/depth. Train that net on rented GPUs.
**Risk.** Fitting a meaningful search in 6 s on Botzone CPU; belief-model accuracy; non-Hu claim
decisions also need search, not just discards.
**Verdict.** Highest-payoff non-RL lever; feasible as a serious 1–2-week effort with a small rollout
net. Our 1-ply V-search is the down-payment on this.

## R3 — Oracle-guided / privileged-information distillation done at scale (Suphx-style)
**Idea.** Train an **oracle** policy/value that sees all four hands (privileged), then distill to the
public-info student with a perfect-information→imperfect-information schedule (anneal the oracle's
extra inputs to zero). We have `oracle_extract.py` (the (50,4,9) oracle obs) already.
**Why it can win.** The oracle learns the "right" answer cheaply; the student gets a much stronger
teacher than any human bot. Suphx reports large gains from this.
**Build cost.** MEDIUM — extraction exists; needs the oracle-train + annealed-distill loop + the
official data (LOST — must re-acquire data.txt from the competition org or regenerate via self-play).
**Compute.** A few GPU-days (rented A4000s parallelize the oracle train + distill sweep).
**Risk.** Prior probe found *discard* decisions gain ~0 from hidden hands (public student recovered
the oracle); the gain may live in **claim/meld decisions**, which we never isolated. Re-test there.

## R4 — Outcome-conditioned / decision-transformer imitation
**Idea.** Condition the policy on a target return (final duplicate score) à la Decision Transformer /
upside-down RL: train on (state, target-return) → action over real games, then at inference condition
on a high target. Turns offline data into goal-directed play without RL instability.
**Why it can win.** Extracts "what do high-scoring games do here" — a stronger signal than plain BC
or our scalar advantage-weighting (which was null). Sequence-model context could capture hand
development the per-decision CNN misses.
**Build cost.** MEDIUM — new model (transformer over the turn sequence) + training; deploy must fit
torch-1.4/512MB/6s (a small transformer does).
**Compute.** GPU-days for the sequence model. **Risk.** Return-conditioning is finicky; sequence
context adds deploy cost; our scalar AWBC already came back null (this is the richer version — may
or may not rescue it).

## R5 — Opponent modeling + exploitation (online, in-match)
**Idea.** Infer each opponent's type/policy from their observed actions during the match (aggression,
defense, tenpai speed) and best-respond — push/fold and target-hand value adapt per table.
**Why it can win.** Duplicate format vs a diverse field: exploiting weak/erratic seats is points the
field-average imitation policy leaves on the table. Pairs with R2's belief model.
**Build cost.** MEDIUM — an online opponent-embedding (Bayesian or learned) + a best-response head.
**Compute.** Low (inference). **Risk.** Few in-match observations to infer from; co-adaptation noise.

## R6 — Fan-value-aware objective / risk-sensitive play
**Idea.** Optimize **expected duplicate score** (fan-weighted), not win-rate or action match —
explicitly trade hand speed vs hand value, and shape a deal-in penalty. The value head already
predicts score; make the *policy* score-aware (e.g., policy-gradient against V, or a score-weighted
training target with per-decision — not per-game — credit).
**Why it can win.** CSM rewards high-fan hands; a speed-maximizing clone under-scores. **Build cost.**
LOW–MEDIUM. **Compute.** Low. **Risk.** Close to AWBC (null) unless credit assignment is per-decision
(needs a value baseline — i.e., this collapses into R2/actor-critic).

---

## Cross-cutting infrastructure these need
- **Re-acquire the official `data.txt`** (lost in the 2026-06-07 data-loss) — R3/R4 want it; ask the
  competition org or regenerate via large self-play.
- **A deploy-embeddable fast simulator + belief model** — the shared dependency of R2/R5.
- **A small distilled rollout/value net** — the shared dependency of R1/R2 (the big net is too slow
  for self-play and search at Botzone's CPU budget).
- **A trustworthy fast eval** (done: persistent-bot gauntlet, 4.6% stuck) and current-meta opponents.

## GPU plan (what to rent and why it accelerates)
| Goal | Rent | Why |
|---|---|---|
| R1 self-play RL at scale | 4–8× RTX 3090/4090 (24GB) for 2–3 d | vectorized env + small-net RL is GPU-throughput-bound; more GPUs ≈ linearly more games |
| R3 oracle train + distill sweep | 2–4× A4000 (16GB) for 1–2 d | parallel arch/recipe sweep; 40-block fits 16GB |
| R2/R4 small-net training | 1–2× A4000 | small nets train fast; one box is enough |
| Eval at scale | the existing P4000 (ssh1) | gauntlet is CPU-bound; keep it dedicated, do NOT co-tenant |
A4000s are the value pick (cheap, 16GB fits our 40-block). 3090/4090 only matter for R1's massive
self-play. Eval stays on a dedicated CPU-heavy box; never co-tenant a gauntlet with training.

## If we miss the deadline
Ship `lad_chunjiandu` (or `distill100b`) — the proven floor. Then pursue **R2 (search) + R1 (JAX-env
RL)** as the journal work: the contribution is "search + scaled self-play closes the imitation gap in
Chinese Standard Mahjong," with the clean persistent-bot gauntlet + ladder as the evaluation, and the
documented null results (RL-without-scale, advantage-weighting, 1-ply value rerank) as honest
baselines. That negative-result map is itself publishable signal.
