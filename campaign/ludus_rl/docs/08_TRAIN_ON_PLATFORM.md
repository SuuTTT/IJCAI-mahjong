# 08 · Train on the platform: AI clones + three submission tiers

Two user-facing features that turn Ludus from "play vs our bots" into
"the platform trains YOUR bots". Design informed by the Botzone upload model
and the Tencent Kaiwu competition architecture (capped equal compute, decoupled
SDK, automated side-switching round-robin).

## Feature 1 — My Clone: an AI that plays like you, on the ladder

Every human match is already a bit-replayable record `(env_version, seed,
decks, action_log, outcome)`. The clone pipeline:

1. **Dataset**: re-simulate each of the user's replays with the deterministic
   engine; collect `(observation_t, action_t)` pairs for the human seat.
   Card-plays are rare vs no-ops (~1:60), so no-ops are subsampled to a fixed
   ratio and plays are weighted.
2. **BC v0**: train the standard ActorCritic policy head with cross-entropy on
   the user's actions. Cold-start trick: initialize from the league champion's
   trunk so 300 human actions fine-tune a competent prior instead of teaching
   a blank net to play from scratch (pure-BC-from-scratch is also supported
   for purists: `--from-scratch`).
3. **Registration**: the clone lands in the user's bot list as
   `user:clone_<name>`, playable, table-able, and rated on the ladder like any
   upload.
4. **RL polish (later)**: optional PPO fine-tune with a KL leash to the BC
   policy — "your style, but sharper". The leash strength is a user slider.
5. **Consent & data**: clones train only on the owner's replays; replays are
   consent-first per docs/04 §7.

Engagement loop: the clone page shows "trained on N of your matches" — play
more, retrain, watch your clone climb. Clone-vs-owner is one button (a table
with you on blue, your clone on red).

## Feature 2 — Three submission tiers

### Tier 1 (live today): weights upload
Flax msgpack for the published architecture, validated by loading, instantly
playable + rated. No sandbox needed, judged at engine speed. This stays the
fast path.

### Tier 2: the Agent Challenge (Kaiwu-style, agent-first)
The insight to steal from Kaiwu — and push further: the interface is a
**problem statement + SDK**, precise enough that either a human or a coding
agent (Claude, GPT, ...) can produce a submission from the document alone.

- **CHALLENGE.md** (served at `/challenge`): a Codeforces-style spec —
  observation tensor layout, action encoding, legality mask semantics, reward,
  episode structure, evaluation protocol, submission format. Self-contained:
  no need to read the engine source.
- **SDK**: `pip install ludus-boom` (the repo already installs as a package).
  `boom.sdk` exposes exactly four calls: `reset/step/observe/legal` plus a
  `submit_check()` that validates a policy file locally before upload.
  The SDK normalizes engine state into tensors and maps network outputs back
  to actions — the user never touches fixed-point internals.
- **Capped equal compute (Kaiwu's fairness core)**: competition entrants get
  identical capped training containers (one GPU-fraction, fixed wall-clock).
  Implementation: per-user supervised job slots on our GPU boxes running
  `baselines/ppo_selfplay.py --resume <their checkpoint>` with hard
  `timeout`; artifacts auto-registered. Algorithm skill decides, not wallets.
- **Self-play battery included**: the league machinery (pool training,
  CI-gated eval) is exposed as templates, so entrants experiment with reward
  shaping and self-play schedules rather than rebuilding infrastructure.
- **Evaluation**: the existing ladder IS the Kaiwu round-robin — continuous
  seat-swapped mirrored pairs, ratings with confidence intervals.
- **Anti-scripting, honestly**: scripts are welcome on the open ladder (they
  are useful calibration anchors); *competition divisions* enforce learned
  behavior structurally — evaluation under domain randomization (random decks,
  mirrored maps, stat-jittered cards within balance bounds). Brittle scripts
  crumble under randomization; learned policies generalize. Detection by
  degradation measurement, not code inspection.

### Tier 3: code upload (Botzone parity) and the visual composer
- **Code upload**: containerized judging of arbitrary code (CPU-capped,
  network-off, wall-clock-limited per move). Needed for Botzone parity;
  deferred until sandbox infrastructure justifies it — weights + challenge
  tiers cover 90% of the value at 10% of the attack surface.
- **Visual composer (Turing-Complete-style)**: a node-graph editor in the
  browser — sensor nodes (nearest enemy, elixir, tower hp) → logic nodes
  (compare, sequence, priority) → action nodes (play slot at tile). Graphs
  compile to a decision policy that runs server-side; shareable as artifacts.
  This is the on-ramp for players who will never write Python, and graphs
  make great BC/RL priors. P2.

## Build order

1. `baselines/bc_clone.py` + clone registration (Feature 1 v0) — now.
2. `/challenge` page (CHALLENGE.md rendered) — now.
3. `boom.sdk` module + `submit_check` — next session.
4. Capped training slots (supervised job queue) — after accounts.
5. Ladder scheduler daemon (continuous rated pairs) — with accounts.
6. Visual composer — P2.
