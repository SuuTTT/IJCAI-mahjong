# Ludus

**Ludus** (Latin: *game*; also the gladiator school — the place where fighters TRAIN and then enter the arena) — **a modern game-AI platform: agent ladders + human play + on-platform training, where every game is GPU-vectorized (JAX) and every match becomes well-structured training data.**

Flagship game #1: **Boom** (an original real-time card-battler). Think "Botzone rebuilt for 2026": developers upload agents (or just weights), humans play the same games in the browser, training happens on-platform and is itself gamified, and the evaluation is statistically honest (calibrated, paired, CI-rated). Naming: **Ludus** = the platform; **Boom** = the first game; future first-party games (Warbound hero-commander, Boom-RTS) are siblings under Ludus.

## Start here (for the executing agent)
1. `AGENTS.md` — engineering discipline & conventions (read FIRST, non-negotiable).
2. `PRIORITIES.md` — the ordered backlog. Take the top unclaimed work order.
3. `workorders/WO-P0-01_boom_engine.md` — the first build task, self-contained.
4. `docs/` — vision, architecture, game design, eval spec, deployment.

## The stack in one paragraph
Game cores are **pure-functional JAX** (`jit`/`vmap`, int32/fixed-point state, explicit PRNG) so one consumer GPU runs thousands of parallel matches for RL training (SMAX / MuJoCo-Playground class throughput). Humans play via **server-authoritative** matches — the same JAX core steps on the server; the browser is a thin PixiJS renderer over WebSocket. Replays are `(seed, action_log)` and bit-reproducible. Ladders use TrueSkill-family ratings **with confidence intervals**, calibration bots as fixed anchors, and paired/duplicate match scheduling. Control plane: FastAPI + Postgres + Redis + R2/S3. Training workers: rented GPUs (Vast.ai) pulling from the queue.

## Repo map
```
docs/00_VISION.md            why this platform, flywheel, monetization
docs/01_ARCHITECTURE.md      JAX-first engine, server-authoritative play, stack
docs/02_GAME_BOOM.md         flagship design: rules, tick spec, card archetypes
docs/03_CATALOG_AND_GENRES.md env-as-game adoptions (JaxMARL/PGX/Craftax) + genre roadmap
docs/04_EVAL_LADDER.md       calibration traps, paired eval, CI ratings, provenance (the moat)
docs/05_DEPLOY_COSTS.md      Vast.ai prototype → Hetzner → AWS best practice + budgets
docs/06_HERO_COMMANDER.md    the RTS+MOBA+open-world hybrid (M&B-inspired), phase P3+
workorders/                  self-contained dev prompts with contractual exit criteria
PRIORITIES.md                ordered backlog P0 → P4
AGENTS.md                    conventions for AI/human developers
```

## Status
- 2026-07-03: repo created; design + plans complete; P0 ready for development.

## License / IP posture
Original assets and names only. Mechanics inspired by classic games (mechanics are not copyrightable); no third-party names, art, or characters anywhere in code, assets, or marketing. Adopted envs (JaxMARL, PGX, Craftax) are Apache-2.0/MIT — attribution kept in `docs/03`.
