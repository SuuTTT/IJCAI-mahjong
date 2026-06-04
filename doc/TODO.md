# IJCAI Mahjong — Roadmap & TODO

Living roadmap, prioritized by ROI. Deadline 2026-06-09. Current best base = **resbn40**
(40-block BatchNorm ResNet, +973 vs the 16-block CNN champ, +1070 vs the old r18 MLP).
Architecture search is **concluded** (resbn40 is the sweet spot). Full ledger: `ARCHITECTURES.md`.
Methods grounded in `deepresearch.md` + `deepresearch-gemini.md` (Suphx + PKU/Botzone winners).

## Status legend
✅ done · 🔄 running · ⏳ queued · ❓ needs data/user · ✗ tried, didn't help

---

## P0 — Deploy the strongest model (competition-critical)
- [✅] **BN-fusion deploy fix.** resbn (BatchNorm) crashed Botzone torch-1.4 (exit 120). Research
  confirms: fold Conv+BN → single Conv + legacy serialization. Done: `resbn40_fused` (numerically
  identical, +1070 vs r18, 0 illegal, 492MB, no BatchNorm). Package `deploy/caiest_cnn_bot.zip`.
- [❓] **Verify fused deploy on Botzone (#17).** USER: re-upload zip + `cnn.pkl`; confirm no RE,
  legal play. High confidence (research names our exact exit-120 + the exact fix).

## P1 — Conversion: the core ceiling (deploy-time, no retrain)
- [✗] **8-fan look-ahead masking (#16) — NULL.** `fan_mask.py` built + verified (discriminates a
  6-fan dead-end from a 45-fan flush). But resbn40+mask vs plain resbn40 = **+74 (29–29 wins, 3%
  draws) = tie.** The expert CNN *already* converts (3% draw rate, avoids dead-end tenpai on its
  own), so the mask rarely changes a decision — and it slows inference (shanten+fan per discard).
  **Not shipped by default**; available behind `FAN_MASK=1`. Lesson: the conversion ceiling is
  already mostly handled by the strong SL policy, not a separate filter.

## P2 — RL fine-tune (push beyond supervised)
- [🔄] **Pool + KL-to-SL (#15).** Parallel actor-learner (`rl_actors.py`, 22 local actors, ~25s/iter,
  ~60× the naive loop). Model pool (SL + learner snapshots) + KL-to-SL leash (MPPO-style). Running
  200 iters. So far ≈ parity with the base (reward ~0) — watching late iters / the eval gate
  `/tmp/rl3_eval.txt`. Plain single-frozen-base self-play already shown to plateau (✗).
- [⏳] **League: main-exploiter + PFSP (#18).** Add an exploiter trained only to beat the main;
  sample opponents ∝ win-rate vs main; mixture 20% active / 30% SL / 50% historical RL. The
  research's fix for the non-transitivity "parity trap" if plain pool+KL still plateaus.
- [⏳] **Global reward prediction (#19).** Train Φ(state)→final standing; shaped reward
  r_t = Φ(Sₜ)−Φ(Sₜ₋₁) for dense, low-variance signal (Suphx). Layer on if reward stays noisy.

## P3 — Data & test-time
- [❓] **Distillation on full chunjiandu set (#20).** `distill.py` ready (blends champion samples
  into official data + fine-tunes). 72 games → +121 (noise). USER: collect ~100+ 4×chunjiandu
  self-play games → direct imitation of the #1 SL+RL policy. (chunjiandu = SL+RL, so this is a
  shortcut to its RL-improved policy.)
- [⏳] **pMCPA test-time adaptation (#21).** At the deal, sample hidden states, rollouts, a few
  gradient steps to adapt to the current hand, then play. We have ~6 s/turn. Stretch; measure
  gain vs latency.

## P4 — Defense (research-listed, lower priority — our bot already deals in <2%)
- [ ] **Opponent hand estimation + safe-discard.** Secondary net predicts opponents' tiles from
  public info; enforce safe discards when an opponent is tenpai. Our deal-in is already low, so
  this is a refinement, not a ceiling-breaker.

## Tried, didn't beat the base (honest negatives)
- ✗ Plain self-play RL (learner vs single frozen base) → parity (the "parity trap").
- ✗ High-fan human-data BC fine-tune → broke the policy (overfit a narrow slice).
- ✗ Wider/deeper-conv/attention/GNN architectures → all tie-or-below resbn40.
- ✗ Distillation on 16–72 games → marginal (+121, noise); needs more data.
- ✗ 8-fan look-ahead masking → null (+74/60g, 29–29 wins); the SL CNN already converts (3% draw).
- ✗ Test-time fan-rollout planner (earlier MLP era) → hurt (solitaire over-values fan-chasing).

## Binding facts
- **resbn40 is the best base; architecture is settled.** Remaining gains = conversion (8-fan
  masking), RL beyond parity (league/exploiters/dense reward), or distilling the #1 bot (more data).
- The #1 bot (chunjiandu) is **SL + RL** — confirms the SL→RL direction; ~68% of our discards
  already match it. Closing that gap is the goal.
- Deploy constraint: Botzone = Python 3.6 / torch 1.4 / ~512MB / ~6 s per turn (BN-free + legacy
  save required; MahjongGB available).
