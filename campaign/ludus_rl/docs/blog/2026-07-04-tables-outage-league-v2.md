# Game tables, a day-long outage, and teaching agents to stop hoarding elixir

*Ludus devlog #2 — 2026-07-04*

## The milestone

Ludus now covers the core loop of the classic university competition platforms,
plus the pieces that make it ours:

- **Game tables (创建游戏桌)**: create a table with any mix of seats — human vs
  human (share a join link; both players see themselves at the bottom via
  per-seat mirrored views), human vs any bot, or bot vs bot to spectate.
- **Bot submission**: upload a weights-only checkpoint for the published
  architecture; it's validated by actually loading it, then becomes a playable
  opponent and a table participant immediately. No sandbox, judged at engine
  speed.
- **Self-play league with public lineage**: every generation's gate result —
  promotion or failure — is on the league page, with confidence intervals.
- **A devlog** (you're reading it) rendered from markdown in the repo.

## The outage: five wrong theories and one measurement

Most of the day the game page said "connecting…". The debugging trail is worth
recording because every wrong theory *looked* confirmed until a better
instrument existed:

1. *"The event loop is blocked by game ticks"* — moved ticks to a worker
   thread. Still broken.
2. *"JAX deadlocks across threads"* — a minimal repro seemed to prove it:
   the same warm function returned in milliseconds on the compiling thread and
   hung from a pool thread. Moved all JAX to one dedicated thread. Still broken.
3. *"The persistent compilation cache is poisoned"* — removed it. Still broken.
4. *"Zombie processes are starving the box"* — two forgotten test runs really
   were burning ten cores for hours (reaped), and compiles really did take five
   minutes under that load. Calmer box. Still broken.
5. The instrument that ended it: a signal-handler stack dumper (works even
   when the event loop is wedged) plus a timed benchmark in a bare process.
   Verdict: the engine compiled exactly once and then **executed at ~275
   seconds per game tick on CPU**. Not a deadlock anywhere — an XLA-CPU
   pathology in the v8 engine. The same program on the GPU: microseconds.

The fix was one line of deployment config (serve from the GPU, sharing it with
training via on-demand allocation). The lesson costs more than the fix:
*theories that fit the symptoms are cheap; measure the thing itself.* Every
"deadlock" was a five-minute compile or a 275-second tick wearing a trench
coat. And the meta-lesson: the site was "verified" green for hours because the
checks tested pages and handshakes — from now on verification plays actual
matches: frames must flow, cards must spawn units, a two-human table must
finish.

## The league plateaued — and told us why

The generational league produced a steep early curve (43% → 78% vs the
scripted anchor in four generations), then **generation 4 defended its crown
against seventeen straight challengers**. That's the promotion gate working —
no coronation without a statistically significant win — but it exposed the
training design: every challenger was the champion's own clone, trained a bit
longer against itself. Mirror self-play converged to one equilibrium and
stopped discovering counters.

A human playtest exposed the equilibrium's character: put two champions at a
table, and after the first tower fell **neither side played another card**.
Under the reward (terminal win/loss plus tower-damage shaping), passivity was
rational — the leader protects its lead by hoarding, and the loser's distant
defeat is discounted into irrelevance.

## League v2

Three changes, shipped together:

1. **Elixir-overflow penalty**: every tick spent capped at max elixir now
   costs both sides reward. Wasted regen is objectively bad play, and now the
   loser (and the leader) bleed for sitting still.
2. **Opponent-pool training**: half of all updates play against a frozen,
   randomly sampled past generation instead of the mirror — with the frozen
   opponent's turns masked out of the learning signal. This is the standard
   cure for self-play collapse.
3. **Livelier serving**: the served bot samples at low temperature instead of
   playing greedy argmax, which amplified degenerate freeze-loops.

Measured immediately after: a champion-vs-champion table ran its full 1200
ticks into overtime to a decisive result, with ten fresh deployments in the
final third — the exact window that used to be frozen. The deeper effect —
generations that *learned* aggression — compounds as v2 generations pass the
gate.

## What's next

- **Accounts + auto-rated ladder** for uploaded bots (continuous seat-swapped
  pairs, ratings with intervals on the site).
- **Train-on-your-replays**: every human match is already a bit-replayable
  record; behavior cloning from them is the signature feature we've been
  building toward.
- **Root-cause the XLA-CPU pathology** (it holds the CPU test suite hostage).
- **More games**: SMAX micro-battles already train to 1.0 win rate on one GPU
  fraction; Hold'em and Mahjong next.

*All numbers above are read from committed artifacts. Engine, tests, league
protocol: github.com/SuuTTT/ludus.*
