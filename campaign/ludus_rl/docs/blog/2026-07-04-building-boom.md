# Building Boom: a CR-faithful RL arena in two days, and every bug that taught us something

*Ludus devlog #1 — 2026-07-04*

## What we built

Boom is the flagship game of **Ludus**, a platform for training, playing, and
ranking game AIs honestly. In two days it went from an empty repo to:

- a **pure-functional JAX engine** (int32 fixed-point state, bit-identical
  replays, CPU≡GPU determinism) running **2.7M+ env-steps/s** on one RTX 3090;
- **tournament-standard numeric parity** with the classic lane-battler it's
  modeled on: 60 cards with exact hp/damage/hit-speed/range/cost, real tower
  geometry, elixir schedule (1×/2×/3×), knockback, spell flight times, tower
  freeze, unit collision with mass, river terrain, bridge chokepoints,
  building aggro, pocket unlocks, sudden death, tower-HP tiebreaks;
- a **browser client** (PixiJS) and a **cinematic replay renderer** (near
  top-down camera, verified frame-by-eye) — humans and agents share one
  server-authoritative engine and one replay format;
- a **CI-honest ladder** (mirrored pairs, ratings with confidence intervals,
  freeze-on-drift calibration) and a **generational self-play league** where a
  new agent takes the crown only by beating the champion through a 95%-CI gate.

Everything ships with provenance: every artifact embeds the commit, judge-code
hash, and integrity counts. Numbers quoted below are read from those artifacts.

## The bugs that made the engine honest

The most valuable part of this build wasn't the features — it was what the
verification machinery (and one sharp playtester) caught:

1. **A 25-point seat bias from floor division.** Mirror-deck controls showed
   player 1 winning 53% vs 28% with identical decks and policies. Cause: `//`
   rounds toward −∞, so distances and velocities computed in the −y direction
   were systematically larger — player 1 literally out-ranged and out-ran
   player 0. Fix: |d|-based, mirror-symmetric integer arithmetic. Lesson:
   *fairness is a measurable property; measure it with mirror controls.*

2. **The deck-specialist illusion.** A PPO agent hit 99% vs the scripted bot —
   on the deck it trained on. Dealt an unseen deck, it fell to a coin flip. One
   eval protocol said "superhuman", the other said "tie". Lesson: *report the
   distribution you trained on, and evaluate off it; our league now trains
   deck-randomized.*

3. **Towers that sniped the bridge.** Playtesting caught towers firing on units
   mid-bridge, then (after the first fix) firing *across the river*. The real
   rule is categorical, not metric: **a tower engages only units that have
   crossed onto its side.** Two range-tweak attempts failed before the rule
   fix; the ice-spirit-connects test now pins it. Lesson: *when a fix is a
   magic number, look for the missing rule.*

4. **Silent no-op experiments.** A MARL training run "completed" in 35 minutes
   with zero artifacts — its framework logs only to a disabled service and
   discards returned metrics. A separate run trained on CPU for hours because
   an env var leaked into the launch shell (the GPU sat idle at 365 steps/s vs
   4M). Lesson: *a run without artifacts didn't happen; assert on evidence,
   not exit codes.*

5. **Shipping sight-unseen.** The first "cinematic" video had its camera depth
   axis inverted — obvious in a single glance no one took before encoding
   5,000 frames. The renderer now has a `--preview-at` flag and no video ships
   without an eyeballed frame. Corollary from the same day: *syncing files
   isn't deploying* — a server process keeps its old module until restarted,
   and two "fixed" bugs weren't, live, until the restart discipline landed.

## The league, live

The self-play league runs generational hill-climbing: generation *k* trains
from the champion's weights, then must beat the champion in seat-swapped
mirrored pairs with a Wilson 95% lower bound above 0.5. Early lineage:

| gen | vs champion | vs random | vs rule_v0 | verdict |
|-----|------------|-----------|------------|---------|
| 0   | —          | 89.8%     | 43.0%      | founder |
| 1   | 85.2% (CI ≥ 80.3%) | 97.7% | 47.7% | **promoted** |

The reigning champion is always the live opponent on the play page, and any
ancestor can be challenged from the league page — you can feel the lineage
get stronger.

## What's next

- **P0 exit**: pip-install quickstart so a stranger can train, play, and rate
  an agent in under an hour on one GPU.
- **Platform parity** with the classic university competition sites: accounts,
  bot upload with versioning, per-game docs, match tables, groups/contests —
  then our differentiators: weights-only submissions judged at millions of
  steps per second, consent-first human replay datasets, and ratings that
  never ship without confidence intervals.
- **More games**: SMAX micro-battles (a MAPPO baseline already trains to 1.0
  win rate on 2s3z in 21 minutes on a fraction of one GPU), Hold'em, Mahjong.

*Stats are unprotectable facts; names and art here are original. The engine,
tests, and every artifact quoted are at github.com/SuuTTT/ludus.*
