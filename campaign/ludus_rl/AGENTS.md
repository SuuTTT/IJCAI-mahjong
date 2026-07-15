# AGENTS.md — engineering discipline for Ludus developers (AI or human)

These rules are distilled from a months-long game-AI campaign that caught ~10 fabricated
"wins" and one silent no-op experiment. They are non-negotiable.

## 1. Determinism is a contract, not a hope
- Game STATE lives in int32/fixed-point. Floats appear only in observations/renders.
- Every stochastic branch consumes an explicit JAX PRNG key. No wall clock, no global RNG.
- Replay = `(env_version, seed, action_log)`. CI runs the determinism suite on every commit:
  - same seed+actions → bit-identical state trajectory, jit vs no-jit, CPU vs GPU;
  - replay of a recorded match reproduces the recorded outcome hash.
- A determinism break is a release blocker, never a "flaky test".

## 2. No silent fallbacks
- Every `try/except`-continue, "if illegal choose first legal", or default-on-missing path
  either counts its firings into an exposed metric or raises. A fallback that fires often
  means the system is not doing what it claims.
- Interventions/experiments must report a mechanism-engagement rate (how often the
  interesting code path actually fired), not just a result.

## 3. Loud-fail aggregation & provenance
- Any script that aggregates results asserts expected counts and writes an `integrity`
  field; partial results must be impossible to mistake for complete ones.
- Result artifacts embed content hashes of the model/env/script that produced them.

## 4. Claims discipline
- Numbers quoted in docs/READMEs must be read from a checked-in JSON/CSV artifact, never
  from memory or console scrollback.
- Performance claims (env-steps/s, win-rates) state hardware, batch size, and the commit.
- A/B claims need paired evaluation + block-level confidence intervals (see
  `docs/04_EVAL_LADDER.md`). The bar is the CI bound, not the mean.

## 5. Work-order protocol
- Take the top unclaimed item in `PRIORITIES.md`; its work order in `workorders/` is the
  contract. EXIT CRITERIA are contractual: do not mark done without demonstrating each one
  with a command that CI (or a reviewer) can re-run.
- Update `PRIORITIES.md` status + append a dated line to the work order's log section.
- Scope creep: if you discover the work order is under-specified, write the smallest
  decision note into the work order and proceed; do not expand scope.

## 6. Code conventions
- Python 3.11+, `jax`/`flax`/`chex`; `ruff` + `pyright` clean; tests with `pytest`.
- Engine code: pure functions only — `step(state, actions, key) -> state`; no classes with
  mutable state in the hot path; shapes static under jit.
- Repo layout (once code lands): `boom/` (engine), `arena/` (platform services),
  `baselines/` (bots/training), `web/` (client), `tests/`, `benchmarks/`.
- Commits: imperative subject + what/why body. Never commit secrets, tokens, or
  credentials; deployment secrets live in the ops vault, not the repo.

## 7. Ops hygiene (from hard experience)
- Long jobs: `setsid nohup`, logs to files, PID recorded. Assume SSH will drop.
- GPU boxes are ephemeral: state lives on the control-plane VPS/object storage; workers
  pull from queues and are safe to kill at any time.
- Billing alarms before scaling anything. One job per GPU. Watch disk (>85% = act).
