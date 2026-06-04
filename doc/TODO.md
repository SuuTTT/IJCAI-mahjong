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
- [✗] **Pool + KL-to-SL (#15) — PARITY.** Parallel actor-learner (`rl_actors.py`), 20-model pool +
  KL-to-SL leash, 200 iters. Final judge eval: **39–39 wins, net −85/80g = tie.** Pool+KL was NOT
  enough to break parity (non-transitivity holds). Late-iter reward ~0 even with the leash decayed.
  Conclusion: need ACTIVE exploiters + PFSP, not just a passive snapshot pool. → #18.
- [✗] **League: main-exploiter + PFSP (#18) — PARITY.** `rl_league.py` ran 400 iters, 13 exploiters
  promoted (mechanism verified: exploiters reliably reached wr 0.6 vs main, got snapshotted, reset).
  Judge eval of the league main vs base: 33-36 wins (parity, leaning base). The AlphaStar machinery
  works but did not lift the main past the expert SL base.
- [✗] **Global reward prediction / dense reward (#19) — PARITY.** Built `phi_reward.py` (Φ:(38,4,9)→
  score, trained on 192k self-play states, **val MSE 0.206 vs 0.361 baseline = R²≈0.43**, a genuinely
  informative predictor) + `rl_league_dense.py` (potential-based shaped return G_t=Σγ^(k-t)(r_k+
  γΦ(s_{k+1})−Φ(s_k))). Ran 600 dense-reward league iters, 22 exploiters promoted. Clean judge eval
  vs base: **44-52 wins, net −138/100g** — base still slightly ahead. Dense reward did NOT break parity.

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
- ✗ League (main+exploiter+PFSP, rl_league.py, 400it/13 promotes) → parity (33-36 vs base).
- ✗ Dense reward (Suphx Φ, R²≈0.43, rl_league_dense.py, 600it/22 promotes) → parity (44-52 vs base).
- ✗ Test-time fan-rollout planner (earlier MLP era) → hurt (solitaire over-values fan-chasing).

## Binding facts
- **resbn40 is the best base; architecture is settled. RL IS EXHAUSTED at parity.** FIVE RL variants
  (single-frozen, pool+KL, league+exploiters+PFSP, dense reward) ALL land at parity-or-slightly-below
  the expert SL base — the mechanisms all work (exploiters promote, Φ has R²≈0.43, KL holds); the SL
  CNN is simply the self-play ceiling at our scale. Matches the literature: beating strong SL needs
  Suphx-scale compute or genuinely-stronger DATA. Remaining real gains = (1) SHIP fused resbn40 (+973,
  in hand), (2) DISTILL the #1 bot with ≥100 games (imitates a real stronger policy — the best bet),
  (3) pMCPA test-time adaptation.
- The #1 bot (chunjiandu) is **SL + RL** — confirms the SL→RL direction; ~68% of our discards
  already match it. Closing that gap is the goal.
- Deploy constraint: Botzone = Python 3.6 / torch 1.4 / ~512MB / ~6 s per turn (BN-free + legacy
  save required; MahjongGB available).
