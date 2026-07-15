# WO-P1-05 · Catalog adoption: SMAX, PGX, Craftax + JAX Hold'em + Mahjong port

**Status:** TODO · **Est:** 3–4 weeks · **Depends:** WO-P0-04 · **Spec:** docs/03 (adoption checklist)

## Prompt for the executing agent
Grow the catalog from 1 game to 6+ using the env-as-game strategy.

### Deliverables (one PR per game, checklist from docs/03 each time)
1. **SMAX (JaxMARL)** — wrap as ladder game; duplicate scheduling = mirrored starts;
   scripted + MAPPO calibration bots.
2. **PGX board family** (start Go 9×9 + chess) — classic ladder; MCTS-lite calibration bot.
3. **Craftax** — open-world entry: seeded duplicate score-race brackets; achievements
   surface as quest metadata (P2 gamification hooks).
4. **Texas Hold'em (build, JAX)** — heads-up NLHE first; duplicate = same cards both
   orientations (standard duplicate poker); CFR-lite + rule bot anchors.
5. **Chinese Standard Mahjong (port)** — reuse the existing engine + official fan library
   integration + duplicate-format gate; 4-seat rotation scheduling already designed.
6. Upstream PR: paired-duplicate eval wrapper offered to JaxMARL (and Craftax if applicable).

### Exit criteria
- [ ] Each game: determinism suite green, ≥2 calibration bots CI-separated on its ladder,
      replays verify, obs schema versioned.
- [ ] One command (`arena games list`) shows the catalog with ladder status.
- [ ] Licenses + attribution recorded in docs/03.

## Log
- (append dated notes here)

## Log
- 2026-07-03 (claude) SMAX adoption seeded ahead of schedule (user asked to keep the
  GPU busy): JaxMARL installed on the dev box (--no-deps so it cannot touch our CUDA
  jax; unused mabrax/brax imports commented out of the editable install), MAPPO-RNN
  baseline trained on 2s3z (HeuristicEnemySMAX): 0.0 -> 1.0 win rate over 20M steps
  in 21 min on a 0.35 GPU slice shared with Boom training. Artifacts:
  `baselines/checkpoints/smax_mappo/{winrate_curve.json, actor_params.msgpack}`
  (sha256 pinned). Lesson logged: JaxMARL baselines log ONLY via wandb and discard
  returned metrics — our wrapper patches train() to return a compact win curve;
  three silent-failure iterations before evidence existed (AGENTS §2/§3 vindicated).
  Platform integration (SMAX as a site game: judge, obs schema, ladder anchors)
  remains the actual WO-P1-05 work.
