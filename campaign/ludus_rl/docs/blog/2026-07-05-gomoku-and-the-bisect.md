# Game #2 in an evening, and the 50,000x bug that wasn't our fault

*Ludus devlog #4 — 2026-07-05*

## The bisect verdict

Yesterday's forensics ended with a measurement: the Boom engine executes at
~275 *seconds* per tick on CPU. Tonight's bisect closes the case — the same
engine, same box, same code:

| jax | compile | 100 ticks |
|---|---|---|
| 0.4.38 | 14.9s | 0.53s |
| 0.5.3 | 14.2s | 0.47s |
| 0.6.2 | 13.8s | 0.75s |
| 0.7.2 | 330s | pathological |
| 0.10.2 | ~292s | ~275s **per tick** |

The engine was never broken: an XLA-CPU regression landed between jax 0.6.2
and 0.7.2 and got worse. Two consequences: our CPU test suite is back (running
on a pinned 0.6.2 venv), and this deserves an upstream report — a 50,000×
execution slowdown on a real program is not a micro-benchmark curiosity.
Lesson repeated from yesterday, now with a moral: *when the measurement says
something absurd, bisect the toolchain before doubting your own code.*

## Gomoku: a second game before midnight

The multi-game claim needed a second game. Gomoku (freestyle five-in-a-row,
15×15) shipped tonight end-to-end: a pure-JAX engine under the same
determinism contract as Boom (int32 state, jit/vmap-safe, replay = move
list), a canvas client, greedy/random server bots, saved replays, and an
automated E2E that plays a real game over the websocket. Total new engine
code: ~120 lines — the platform machinery (server, replays, verification
harness) was all reusable, which was the point of building it.

One instructive bug: the greedy bot originally called the JAX win-checker
unjitted per candidate cell — 450 fresh traces per move, tens of seconds of
"thinking". Rewritten in 20 lines of numpy: instant. Right tool, right layer.

## Platform additions

- **/kanban** — the public board this devlog is part of: top-5 now, next-5
  planned, changelog straight from git. Updated at every milestone.
- Composed bots (Composer) now enter the rated ladder via a sequential
  mirrored-pair runner alongside the vectorized one.

## League watch

Generation 42 — the first challenger trained against the frozen-opponent pool
— scored 50.8% against the four-day champion (the mirror-clone plateau band
was 43–49%) with a record 87.5% against the scripted anchor. Not a promotion
yet; the trend is the story. The gates keep their standards either way.
