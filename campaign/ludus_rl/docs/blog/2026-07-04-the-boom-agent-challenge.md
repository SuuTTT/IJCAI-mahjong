# The Boom Agent Challenge

Write a policy for Boom, a real-time lane battler. This document is
self-contained: everything needed to produce a valid submission is below. It is
written for both humans and coding agents — paste it into your AI assistant and
ask for a submission.

## Game summary

Two players on an 18×32 tile board (x: 0–17, y: 0–31). You are always player 0
at the bottom; the engine mirrors everything for your opponent, so one policy
plays both seats. A river with two bridges (x≈3–5 and x≈14–16) splits the
board at y≈15–16. Each side has two princess towers and a king tower. Kill
more towers than you lose in 180 seconds (900 ticks at 5 ticks/s); ties go to
overtime (3× elixir) then a tower-HP tiebreak.

You hold 4 cards of an 8-card deck (4 more cycle behind). Cards cost 1–9
elixir; you regenerate elixir continuously (1 per 2.8s, 2× after 120s, 3× in
overtime, capped at 10). Units walk, target, and fight autonomously once
deployed — your ONLY decisions are which card to play, where, and when.

## Interface

Install: `git clone https://github.com/SuuTTT/ludus && pip install -e ludus`
plus `jax` and `flax`.

```python
from boom import engine, vec
import jax

state = engine.reset(jax.random.PRNGKey(seed), None)   # None = default decks
obs   = engine.observe(state, player)   # player-centric observation
mask  = vec.flat_legal(state, player)   # (2305,) bool — legal actions now
state = engine.step(state, actions, None)  # actions: int32 (2, 3) — see below
res   = engine.result(state)            # -1 ongoing / 0 p0 wins / 1 p1 wins / 2 draw
```

Everything is pure-functional JAX: `jit`, `vmap`, and `lax.scan` all work.
Deterministic: same seed + same actions = the same game, bit for bit.

## Observation (player-centric; you always see yourself at the bottom)

`obs.spatial` — float32 `(32, 18, 8)`, board-shaped channels:

| ch | meaning |
|---|---|
| 0 | your units' hp (fraction of max) at their tile |
| 1 | enemy units' hp |
| 2 | your towers' hp |
| 3 | enemy towers' hp |
| 4 | your air units flag |
| 5 | enemy air units flag |
| 6 | unit deploy-delay flag (both sides) |
| 7 | terrain: river=1, bridge=0.5 |

`obs.vector` — float32 `(12,)`: `[your elixir/10, enemy elixir/10, tick/900,
double-elixir flag, overtime flag, your 4 hand card-ids /60, their costs /10]`.

## Actions

One action per tick: `[slot, x, y]` int32.
- `slot` 0–3 plays hand card `slot` at player-frame tile `(x, y)`; `slot = 4`
  is no-op (x, y ignored).
- Flat encoding for masks/nets: `a_flat = 1 + slot*576 + y*18 + x`, no-op = 0,
  total 2305. Convert with `vec.flat_to_triple(a_flat)`.
- Legality: your own half (y ≤ 14), any tile for spells, pocket rows 17–23
  after you destroy the corresponding princess tower. Illegal plays are
  counted no-ops — always apply the mask.

## Evaluation

Your policy plays seat-swapped mirrored pairs (same seeds both ways) against
the ladder pool. Ratings come with 95% confidence intervals; the leaderboard
shows the interval, not just the point estimate. Decks are randomized in the
mixed division — policies that only pilot one deck collapse there.

## Submission (weights tier)

Train any policy with the `ActorCritic` architecture in
`baselines/ppo_selfplay.py` (or initialize from the published league
checkpoints at `huggingface.co/Dannibal/ludus-boom-league`), then upload the
Flax msgpack at `/submit` on the platform. It is validated by loading and
becomes a rated, playable opponent immediately.

A complete training run that reaches ~100% vs the random anchor:

```bash
python baselines/ppo_selfplay.py --decks mixed --updates 20000 \
    --pool-dir <dir of frozen opponents>  --out my_bot
# upload my_bot/params_latest.msgpack
```

## Tips

- The engine runs ~5M steps/s on one consumer GPU — brute exploration is
  affordable; sample efficiency is not the bottleneck, credit assignment is.
- Elixir sitting at 10 is wasted regen; the strongest baselines are penalized
  for it in training.
- Watch your bot lose: every match is a replay; `/play?a=user:you&b=ppo`
  spectates your bot against the champion live.
