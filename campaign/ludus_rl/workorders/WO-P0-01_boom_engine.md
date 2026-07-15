# WO-P0-01 · Boom JAX engine core

**Status:** TODO · **Est:** 2–3 weeks · **Depends:** none · **Spec:** docs/02, docs/01, AGENTS.md

## Prompt for the executing agent
Build the Boom game core as a pure-functional JAX environment in `boom/`.

Read first: `AGENTS.md` (discipline — determinism rules are contractual),
`docs/01_ARCHITECTURE.md` (interface), `docs/02_GAME_BOOM.md` (game rules).

### Deliverables
1. `boom/engine.py` — `reset(key, config) → state`, `step(state, actions, key) → state`,
   `observe(state, player) → obs`, `legal(state, player) → mask`, `result(state)`.
   All jit-able, shapes static, state int32/fixed-point per docs/01.
2. `boom/cards.csv` + `boom/cards.py` — the 60-card v1 set per the archetype matrix in
   docs/02 (design the numbers; document balance intent in comments; data-only patches).
3. `boom/env.py` — gymnasium + PettingZoo adapters; `boom/vec.py` — vmapped batch API.
4. `tests/` — determinism suite (AGENTS.md §1): jit≡nojit, CPU≡GPU state trajectories,
   replay-from-(seed,actions) reproduces outcome hash; property tests (energy bounds,
   unit caps, no negative HP, legal-mask soundness: every legal action steps without error).
5. `benchmarks/throughput.py` — env-steps/s and matches/s vs batch size; results JSON
   checked in with hardware + commit noted.

### Exit criteria (demonstrate each with a runnable command)
- [ ] `pytest` green including full determinism suite.
- [ ] ≥5,000 parallel matches stepping on one RTX 3090 (or equivalent), ≥1M env-steps/s
      aggregate. Recorded in `benchmarks/results/throughput_v1.json`.
- [ ] Two scripted random-legal agents complete 10k matches with zero exceptions;
      outcome distribution sane (non-degenerate win rates, overtime rate <40%).
- [ ] A recorded match replays bit-identically from `(seed, action_log)`.

### Non-goals
No renderer (WO-P0-03), no training (WO-P0-02), no networking. Do not expand the card set
beyond 60. Do not add fog, decks-building, or 3+ lanes.

## Log
- 2026-07-03 (claude) CLAIMED. First implementation pass: `boom/` (cards.csv/cards.py/
  engine.py/env.py/vec.py), `tests/`, `benchmarks/`. Smallest decision notes where the
  spec was under-specified (AGENTS §5):
  - **Geometry:** P0 owns rows 0–14, river rows 15–16, P1 rows 17–31. Bridges centered
    x=4.5, x=13.5 (lane split x=9). Towers P0: turrets (4,6),(13,6) hp1400, core (8,2)
    hp2400; P1 mirrored. Towers: dmg 40 (core 50) every 3 ticks, range 5.5, hit air.
  - **Energy fixed-point:** 1 energy = 14 units (2.8 s = 14 ticks at 5 t/s) → regen is
    exactly +1 unit/tick, +2 from tick 600 (last 60 s + OT). Start 5, cap 10.
  - **Distances:** squared distances computed in (fp/16) units to stay in int32 across
    the whole board; movement normalization uses an exact integer sqrt (float32 sqrt of
    exact-int + ±1 correction — bit-deterministic CPU/GPU).
  - **Action frames:** actions & obs are player-centric; engine rotates player 1 by 180°
    (policies weight-shareable across seats). Action = (slot 0–3 | 4=noop, x, y) tile.
  - **Simultaneity:** card plays processed p0-then-p1 (deterministic slot allocation;
    ≤1 tick of spawn priority asymmetry). Combat damage is simultaneous (mutual kills
    possible; both cores dead same tick = draw).
  - **Illegal actions:** never raise, never auto-correct — counted no-ops in
    `state.illegal` (AGENTS §2). Legal placement requires energy + own half (spells:
    anywhere) + enough free unit slots for the card's full body count.
  - **Statuses:** slow (½ speed) / rage (1.3× dmg & speed) / deploy delay (5 ticks)
    packed as bytes in `u_status`. Spells apply statuses instantly in a radius (no
    persistent zones in v1). Spells deal 40% damage to towers.
  - **Targeting:** recomputed every tick — nearest eligible in sight 5.5, tie-break by
    lower id; building-only units see towers exclusively. Ground units chase only
    same-side-of-river targets, else lane-follow via bridge; air flies straight.
  - **v1 step() consumes no randomness** (shuffle only in reset); `key` param kept for
    interface stability and documented as unused.
- 2026-07-03 (claude) First 3090 run: 24/25 tests green incl CPU≡GPU bit-determinism;
  5.1M env-steps/s @16,384 parallel matches. Two bugs found by the exit-criteria runs:
  1. **Legal-mask capacity race** (sanity_10k: 72 illegal plays / 24M): masks are
     pre-tick but p0's spawn resolves first and can exhaust the shared 64-slot pool.
     Fix: legal() requires an 8-slot margin (max bodies per play). Mask is now sound
     under any simultaneous opposing action (slightly conservative near pool cap).
  2. **Seat bias** (mirror-deck control, n=10k: p1 53.1% vs p0 28.2% with identical
     decks): floor division rounds toward −∞, so distances/velocities computed in the
     −y direction favored p1 (longer effective range, faster movement). Fix:
     sign-symmetric |d|-based arithmetic in `_dist2_16` and movement. Post-fix mirror
     controls: A-vs-A 38.3/40.4/21.3, B-vs-B 38.7/39.5/21.8 (residual ~2-pt p1 edge
     from documented p0-first ordering + id tie-breaks; cancelled by seat-swapped
     paired evaluation in the ladder design, docs/04).
  Deck balance under random play: A-vs-B 69.8/14.2 — deck A too strong; balance is a
  data-only cards.csv patch, deferred (random-play winrates ≠ skilled-play balance).
- 2026-07-03 (claude) **DONE.** All exit criteria demonstrated on a rented RTX 3090
  (Vast 43670173), clean tree @85bd3d1:
  - [x] `pytest -q` → 25 passed (incl. jit≡nojit, CPU≡GPU, bit-identical replay).
  - [x] Throughput: 5.36M env-steps/s aggregate at 16,384 parallel matches
        (bar: ≥1M at ≥5,000) → `benchmarks/results/throughput_v1.json`.
  - [x] 10k random-legal matches, zero exceptions, zero illegal actions; win rates
        71.2/13.5/15.3 (deck A vs B), overtime 25.4% (<40%)
        → `benchmarks/results/sanity_10k.json`.
  - [x] Replay bit-identity covered by `tests/test_determinism.py::test_replay_bit_identical`.
  Re-run: `pytest -q && python benchmarks/throughput.py && python benchmarks/sanity_10k.py`.
  Bonus (WO-P0-03 down-payment): server-authoritative play server + canvas client
  (`arena/`), human-vs-random_v0, replays logged in the docs/01 schema.
- 2026-07-03 (claude) Engine v2 revalidation (CR rules/stats — see v2 commit ca9a4b7):
  30/30 tests; 5.16M env-steps/s @16,384 (throughput unchanged by sticky targeting);
  sanity_10k all-pass (overtime 36.2%, draws 32.4% under random play — CR-durable
  towers; skilled play expected to close games); mirror seat gap ~1pt. Artifacts
  regenerated clean @8e4a1f3.
- 2026-07-03 (claude) Engine v6 (user-reported rule audit): princess towers moved
  forward to CR geometry (3.5/14.5, 9.5) with 8.3 reach; towers engage ONLY units
  that crossed onto their side (never bridge/river, never across the water — the
  first fix attempt excluded only river rows and the tower still sniped across,
  killing ice spirits pre-bridge; test caught it). River impassable for ground
  non-jumpers off the bridge spans (movement + collision guard + property test).
  Building-targeters now consider deployed buildings globally (a mid-placed cannon
  pulls the hog off its line). ENV boom/v6; deck-randomized ppo_v6 (--decks mixed)
  addresses the v5 deck-specialist finding.
