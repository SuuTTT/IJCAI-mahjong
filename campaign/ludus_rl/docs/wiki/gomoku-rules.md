# Gomoku on Ludus: rules, variants, and competition constraints

## What we play today: Freestyle

- Board 15×15; black (you) moves first, players alternate single stones.
- **Five or more in a row wins** — any direction (row, column, either
  diagonal). Overlines (6+) count as a win.
- **No forbidden moves.** Black has no restrictions: double-threes,
  double-fours, and overlines are all legal for both players.
- Draw when the board fills with no five.
- Replays are the move list; the engine is deterministic and jit/vmap-safe
  like every Ludus game.

So yes: winning with a plain open diagonal five is fully legal here — if the
bot lets you build an open four, that's the bot's problem (the `strong` bot
should not).

## Why "first player bans" exist (and what they are)

Freestyle gomoku is a **proven first-player win** with perfect play — black's
initiative is overwhelming. Serious competition therefore constrains black or
the opening:

| variant | constraint | notes |
|---|---|---|
| **Standard** | exactly five wins; overlines are NOT a win | mildest fix |
| **Renju** | black (only) forbidden: double-three, double-four, overline — playing one loses immediately | the classical balanced ruleset; white has no restrictions |
| **Swap** | player 1 places 3 stones, player 2 chooses which color to take | opening auction removes first-move advantage |
| **Swap2** (Gomocup standard) | like swap, plus option to place 2 more stones and hand the choice back | current de-facto competition standard |
| **Caro** | five wins only if not blocked on both ends | popular in Vietnam |

## Ludus competition constraints (current + planned)

- **Current casual play**: freestyle, black = human, bot divisions: `random`,
  `greedy` (win/block only — beatable by any open four), `strong`
  (Rapfi, the Gomocup-winning engine, run as an external process; GPL-3,
  credited, unmodified).
- **Planned rated divisions**: freestyle (as now) and **swap2** for the
  competitive ladder, mirroring Gomocup — swap2 removes the first-mover bias
  without asymmetric forbidden-move rules, which keeps agent evaluation
  seat-symmetric (the same fairness principle as Boom's seat-swapped pairs).
- **Renju division**: possible later; requires forbidden-move detection in the
  engine (double-three recognition is the tricky part) — tracked on the
  kanban.
- Time controls for bot matches: 1.5s/move soft cap on the server engine;
  uploaded gomoku agents will get the same capped-compute treatment as Boom
  submissions.

*Rules pages are versioned with the engine: this page describes
`gomoku/v1` (freestyle). Rule-variant engines will bump the version and keep
old replays valid under their recorded version.*
