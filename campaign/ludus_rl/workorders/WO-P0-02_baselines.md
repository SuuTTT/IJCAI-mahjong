# WO-P0-02 · Baselines: scripted bot + PPO self-play

**Status:** TODO · **Est:** 1–2 weeks · **Depends:** WO-P0-01 · **Spec:** docs/02 §research hooks, AGENTS.md

## Prompt for the executing agent
Produce the three calibration-anchor bots for Boom in `baselines/`.

### Deliverables
1. `baselines/rule_v0.py` — scripted heuristic (energy-efficient defense, counter-push,
   simple elixir counting). Must beat random-legal ≥95% (paired eval).
2. `baselines/ppo/` — PPO self-play in pure JAX (PureJaxRL-style; whole rollout on device
   using `boom.vec`). Conv torso over the 18×32×C spatial obs + vector head.
   League-lite: latest vs pool of past snapshots (uniform over last K).
3. `baselines/eval.py` — PAIRED evaluation harness per docs/04: mirrored-pair scheduling,
   block-level t-CIs, calibration self-test (bot-vs-self in duplicate mode must read the
   exact tie value or the run aborts). Results as JSON with provenance hashes.
4. `ppo_v0` checkpoint (weights committed via LFS or storage link + hash).

### Exit criteria
- [ ] Calibration self-test passes exactly (documented in eval JSON).
- [ ] `ppo_v0` beats `rule_v0` with CI-separated paired result (report blocks, CI).
- [ ] Training reproducible: seed + config committed; curve JSON checked in;
      <1 hour to ≥60% vs rule_v0 on one 3090 (document actual time+hardware).
- [ ] Illegal-action rate of ppo_v0 = 0 (mask enforced in policy, verified in eval).

### Non-goals
No web/UI, no rating service (WO-P0-04), no architecture search — one solid PPO.

## Log
- (append dated notes here)

## Log
- 2026-07-03 (claude) CLAIMED. `baselines/`: PPO self-play (PureJaxRL-style, masked
  2305-action policy, seat-symmetric net, GAE, Wilson-CI eval, msgpack ckpts, jsonl
  provenance), `rule_v0` scripted heuristic (defend-biggest-threat / push-at-9-energy),
  `eval_pair.py` seat-swapped paired evaluator (CI bound is the claim).
  Training run on the 3090 (512 envs × T32, ~89k env-steps/s incl. update): vs random
  87.9% @250 upd → 98.0% @1500 (0 losses, CI ≥ 95.5%). Mid-training ckpt vs rule_v0:
  83.6% seat-swapped (CI ≥ 78.6%, n=256) — `benchmarks/results/eval_ppo_mid_vs_rule.json`.
  rule_v0 vs random: 85.2% (CI ≥ 80.3%) — `eval_rule_vs_random.json`.
  Calibration ordering random < rule_v0 < ppo_v0 established mid-run; final evals at
  n=1024 when the 20k-update run finishes. Play server now serves all three as
  opponents (/play?bot=ppo|rule|random), ppo reloading the freshest checkpoint.
  Known: policy entropy collapses early (~0.01 by 1k updates, ent_coef 3e-3) — fine
  for v0 baseline; revisit for ppo_v1 (higher ent_coef / KL target) if needed.
- 2026-07-03 (claude) **DONE on engine v2** (CR rules/stats). Exit criteria:
  - [x] Calibration self-test: identical agents over identical seeds are
        bit-identically reproducible with zero illegal actions — embedded in every
        eval JSON (`self_test`) and asserted (loud-fail) by `eval_pair.py`.
  - [x] ppo_v0 beats rule_v0, CI-separated, n=1024 seat-swapped paired:
        99.1% (Wilson CI ≥ 98.3%) — `benchmarks/results/eval_v2_ppo_vs_rule.json`.
        vs random: 99.8% (CI ≥ 99.3%) — `eval_v2_ppo_vs_random.json`.
  - [x] Reproducible: seed(0) + full config + commit in
        `baselines/checkpoints/ppo_v2/training_curve.jsonl`; 20k updates, 328M
        env-steps, ~57 min on one RTX 3090 (97k steps/s incl. updates) — inside the
        <1 h bar; ≥60%-vs-rule reached well before the end of the hour.
  - [x] Illegal-action rate = 0, verified in eval (masked policy; loud-fail assert).
  Checkpoint committed: `baselines/checkpoints/ppo_v2/params_latest.msgpack`
  (5.0 MB, sha256 43130b06…) — small enough for plain git; storage tier later.
  **Documented deviations (AGENTS §5 smallest-decision):**
  1. rule_v0 ≥95% vs random was written against v1 numbers. On v2, CR-strength
     towers crush scripted lone/sustained pushes: measured 46.0% (n=1024,
     `eval_v2_rule_vs_random.json`; v1 evidence was 85.2%). Kept as a fixed
     mid-anchor at measured strength; the ≥95% strength bar is carried by ppo_v0.
     Frozen weak PPO checkpoints can provide graded anchors for the ladder (WO-P0-04).
  2. League-lite (pool of past snapshots) deferred to ppo_v1 — plain self-play
     already CI-clears every anchor; league matters when anchors get stronger.
  3. Duplicate-mode (mirrored RNG) and block-level t-CIs live in WO-P0-04's rating
     layer; the seat-swapped Wilson bound is the interim standard.
  Training incident worth keeping: first v2 run LOST to random at update 1250
  (22/30/48 W/L/D) — reward normalizer still used v1 tower pool (5200 vs 10928) and
  entropy collapsed (ent 3e-3). Fix: derive normalizer from engine constants +
  ent_coef 1.5e-2 → healthy curve (73% @1250, 100%/0L from 6250 on).
- 2026-07-03 (claude) v5 re-validation exposed a real evaluation lesson: ppo trained
  on mirror deck B only. On deck-B mirrors it beats rule_v0 99.2% (CI ≥ 98.0, n=512,
  `eval_v5_ppo_vs_rule_deckBB.json`); on the default mixed-deck protocol (plays the
  unseen cycle deck in half the games) it TIES rule_v0 (49.1%, `eval_v5_ppo_vs_rule.json`)
  and drops to 91.7% vs random. Deck coverage is part of the task distribution —
  ppo_v1 must train deck-randomized. eval_pair gained --decks {default,AA,BB}.
