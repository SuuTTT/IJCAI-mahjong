---
license: other
license_name: competition-game-logs
license_link: LICENSE
language:
- zh
- en
tags:
- mahjong
- chinese-standard-mahjong
- mcr
- game-logs
- engine-testing
- botzone
- ijcai-2026
pretty_name: MCR Final-2026 Engine Correctness Test Set
size_categories:
- 10K<n<100K
---

# MCR Final-2026 Engine Correctness Test Set

**A correctness test dataset for Chinese Standard Mahjong (MCR, 国标麻将) engine and
judge implementers**, built from the **12,288 official IJCAI-2026 Mahjong AI Competition
Final Stage-2 games** (Botzone, contest finished 2026-07-09; 512 duplicate walls x 24
seat/wind permutations, 4 finalist bots, zero protocol errors).

The product is simple: **any new MCR judge/engine (e.g. a JAX implementation) must
reproduce these games exactly** — every deal, every draw, every claim resolution, every
win/fan decision, and the final four-seat score of all 12,288 games. If it does, it is
protocol- and rules-equivalent to the official judge on ~590k decision points. The games
were played by four independent, strong, mutually adversarial policies, so the stream
exercises claim priority, kong edge cases, robbed kongs, 8-fan boundary wins and monster
hands far better than random play does.

## Files

| file | contents |
|---|---|
| `mcr_final2026_full.jsonl.gz` | all 12,288 games, one JSON record per line (~14 MB) |
| `mcr_final2026_golden.jsonl` | 221 curated edge-case games (uncompressed, browsable) — every record also appears in the full file |
| `validate_engine.py` | reference validator: dataset self-test + engine harness (pure stdlib) |
| `TAGS_SUMMARY.json` | per-tag counts, request-grammar census, judge-vs-MahjongGB scan details, golden game ids |
| `LICENSE` | usage note (competition game logs, research use) |

Status at publication: `validate_engine.py --self-test` passes **12,288/12,288** games;
the built-in demo engine reproduces **221/221** golden games through the engine harness.

## Record schema (one game per line)

```jsonc
{
  "game_id": "6a4eba831ba515095a1f155b",   // Botzone match id
  "srand":   2143175072,                   // judge RNG seed that generated the wall
  "quan":    0,                            // prevalent wind 0-3 (0=E); 3072 games each
  "players": ["player152/player2", ...],   // seat 0..3 -> "user/bot" (metadata only)
  "wall":    ["W4","F3", ... 136 tiles],   // full ordered wall — see deal order below
  "turns": [                               // the complete lockstep protocol stream
    {
      "request":  {"0": "2 T8", "1": "3 0 DRAW", ...},  // judge -> each seat (verbatim)
      "display":  {"action": "DRAW", "player": 0, "tile": "T8",
                   "canHu": [-3,-4,-4,-4], "tileCnt": [20,21,21,21]},
      "responses": {"0": "PLAY T9", "1": "PASS", "2": "PASS", "3": "PASS"}
    },
    // ... terminal turn: display.action == "HU" (fan/fanCnt/score) or "HUANG",
    //     no request/responses on the terminal turn
  ],
  "expected": {
    "ending":    "ron",          // "zimo" | "ron" | "draw"  (draw = 荒庄)
    "winner":    2,              // seat, or null for draw
    "discarder": 3,              // seat that dealt in (ron only; for qianggang = the BUGANG player)
    "qianggang": false,          // true = robbed kong (HU immediately after BUGANG)
    "fan":       [["清龙",16,1],["平和",2,1], ...],  // [name, value, count]
    "fan_total": 20,
    "scores":    [-8,-8,44,-28]  // final duplicate-format game scores, sum = 0
  },
  "tags": ["ron","multi_claim", ...],
  "known_discrepancy": { ... }   // only on gb_discrepancy / gb_fan_diff games (see below)
}
```

Tiles: `W1-W9` characters (万), `B1-B9` dots (饼), `T1-T9` bamboo (条), `F1-F4` winds
(东南西北), `J1-J3` dragons (中发白). No flowers. Seat `s` has seat wind `s`; **seat 0 is
always the dealer** (first to draw) regardless of `quan`, which is the prevalent wind
(uniform 0-3 across the 24 permutations of each wall).

### Wall / deal / draw convention (reproduce this or nothing matches)

Each seat owns a private 34-tile wall segment: **seat `s` draws from
`wall[34*s .. 34*s+33]`, consumed from the back** (`wall[34*s+33]` first).
The deal gives each seat the last 13 tiles of its segment in back-to-front order:
`hand[s][i] == wall[34*s + 33 - i]`. The dealer's first draw is `wall[20]`, i.e.
`wall[34*s + 20]` for `s = 0`. Kong replacement draws come from the drawer's own
segment, same order — there is no separate dead wall. `tileCnt[s]` in every display is
the number of tiles remaining in seat `s`'s segment. The game is a draw (HUANG, 荒庄)
when the seat that would draw next has an empty segment. All 12,288 games were verified
to obey this convention exactly. `srand` is informative (the judge's C `srand` seed) —
you never need to re-derive the wall from it; the wall is given explicitly.

### Protocol stream: requests, displays, responses

`turns[k]` is one judge step: `request` is the exact per-seat request string the judge
sent (Botzone Chinese-Standard-Mahjong "simple interaction" format), `display` is the
judge's public event record, `responses` is what the four bots answered. Complete
request-grammar census over all 12,288 games (counts in `TAGS_SUMMARY.json`):

```
0 <seat> <quan>                    game start (own seat wind, prevalent wind)
1 0 0 0 0 <t1> ... <t13>           deal (the four 0s are flower counts, always 0)
2 <tile>                           your own draw
3 <p> DRAW                         player p drew (tile hidden)
3 <p> PLAY <tile>                  player p discarded <tile>
3 <p> PENG <tile>                  player p pengs the last discard AND discards <tile>
3 <p> CHI <mid> <tile>             player p chis the last discard (run's middle tile =
                                   <mid>) AND discards <tile>
3 <p> GANG                         AMBIGUOUS: melded kong of the last discard, OR a
                                   concealed kong (AnGang) if p just drew — the AnGang
                                   tile is never in the request (only in the display)
3 <p> BUGANG <tile>                player p promotes an exposed peng of <tile> to a kong
```

Legal responses: `PASS`, `PLAY <t>`, `PENG <t-to-discard>`, `CHI <mid> <t-to-discard>`,
`GANG` (meld last discard), `GANG <t>` (concealed kong, after own draw), `BUGANG <t>`,
`HU`.

### The four classic replay traps (all fixed in this dataset's semantics)

These are real bugs found in a published replay harness for this log format. An engine
or log consumer that gets any of them wrong diverges within a few games:

1. **PENG/CHI events embed the claimer's follow-up discard.** `{"action":"PENG",
   "player":2,"tile":"W4"}` means *seat 2 pengs the live discard and then discards W4*.
   The claimed tile is NOT in the event — it is the last live discard. For CHI,
   `tileCHI` is the run's middle tile and `tile` is again the follow-up discard.
   Dropping these embedded discards causes silent state drift and later crashes.
2. **`GANG` is ambiguous and must be discriminated by live-discard tracking.** A `GANG`
   while an unclaimed discard is on the table is a melded kong of that discard; a `GANG`
   with **no** live discard (i.e. right after the player's own draw) is a concealed kong
   (AnGang) whose tile appears only in the display, never in the request.
3. **`BUGANG` is a first-class event** (promoted kong). It kills the live discard,
   offers every other seat a robbed-kong (`qianggang`) HU claim, and on PASS the kong
   player draws a replacement tile.
4. **Terminal labeling:** a ron is the winner's *claim* on the live discard (or on the
   BUGANG tile — qianggang); a zimo is a win on the winner's own pending draw. The
   `expected` block encodes the corrected labels: `ending`/`winner`/`discarder`/
   `qianggang`.

### Rules the games exercise (and your engine must implement)

- **Minimum 8 fan to win** (MCR): the judge only granted HU with `fan_total >= 8`
  (flowerless Botzone MCR scoring; `fan` lists each 番种 as name/value/count exactly as
  the judge scored it; `fan_total == sum(value*cnt)` holds on every win).
- **Claim priority: HU > PENG/GANG > CHI.** On multiple HU claims for the same tile the
  seat nearest downstream (counter-clockwise) of the discarder wins. CHI is only legal
  from the left neighbour. Verified on every multi-claim turn in the corpus
  (1,622 multi-claim games, 140 with competing HU/claim priority on the same tile).
- **Scoring** (duplicate format, per game, sums to 0):
  - zimo: winner `+3*(8+fan)`, each other seat `-(8+fan)`
  - ron: winner `+(3*8+fan)`, discarder `-(8+fan)`, bystanders `-8` (qianggang: the
    BUGANG player is the discarder)
  - draw (荒庄): all 0.
- **`canHu` per-step oracle:** every display carries `canHu[4]`; a value `>= 0` is the
  exact fan count that seat could win with on the current tile event; negative values
  are judge-internal codes (-4: not applicable / own event, -3: cannot win). Verified:
  whenever a game ends, the winning event's `canHu[winner]` equals the final `fanCnt`.
  Your engine can check its fan calculator at every single step against this field.

## Golden subset & edge-case tags

`mcr_final2026_golden.jsonl` — 221 games chosen deterministically (lowest game_id per
tag; all games of the rare tags). A game carries all tags that apply to it.

| tag | full set | golden | meaning |
|---|---|---|---|
| `ron` | 8,436 | 109 | win on another seat's discard |
| `zimo` | 3,652 | 92 | self-drawn win |
| `draw` | 200 | 20 | 荒庄, wall exhausted, all scores 0 |
| `fan8_boundary` | 1,778 | 36 | win at exactly the 8-fan legal minimum |
| `multi_claim` | 1,622 | 57 | ≥2 seats claimed the same tile (priority resolution) |
| `multi_hu` | 140 | 20 | ≥2 competing HU claims on one tile |
| `bugang` | 1,204 | 54 | promoted kong occurred |
| `minggang` | 939 | 25 | melded kong of a discard occurred |
| `angang` | 519 | 23 | concealed kong occurred |
| `qianggang` | 28 | 28 (all) | robbed-kong win (HU right after BUGANG) |
| `fan_qiangganghu` | 28 | 28 (all) | 抢杠和 scored in the fan list |
| `fan_gangshangkaihua` | 46 | 8 | 杠上开花 (win on kong replacement tile) |
| `fan_last_tile_zimo` | 9 | 5 | 妙手回春 (last wall tile, self-drawn) |
| `fan_last_tile_ron` | 4 | 4 (all) | 海底捞月 (win on the final discard) |
| `big_fan` | 20 | 10 | fan_total ≥ 48 (max in corpus: 93) |
| `gb_discrepancy` | 20 | 20 (all) | KNOWN-DISCREPANCY, see below |
| `gb_fan_diff` | 98 | 20 | KNOWN-DISCREPANCY (milder), see below |

## KNOWN-DISCREPANCY cases (judge vs cai-style MahjongGB replay) — do not "fix" them

Every HU ending was re-scored with the community python `MahjongGB` (PyMahjongGB) fan
calculator by replaying hands/melds through the widely-copied "cai" FeatureAgent (the
encoder used by most Botzone MCR RL/imitation codebases). Result over 12,088 wins:

| status | games | meaning |
|---|---|---|
| `agree` | 11,970 | replayed MahjongGB fan == judge fan, fan by fan |
| `cai_replay_fan_diff` | 98 | cai-style replay scores lower but still ≥8 fan |
| `cai_replay_below_8fan_gate` | 20 | cai-style replay scores the judge-accepted zimo **below 8 fan** — a cai-gated engine would have forbidden a legal win |

All 118 disagreements have the **same root cause, in the replay layer, not in
MahjongGB**: the cai FeatureAgent computes `is4thTile = shownTiles[winTile] == 4`, and
the just-drawn tile is never counted in `shownTiles`, so 和绝张 (Last Of Its Kind,
4 fan) can never fire on a self-drawn win. With `is4thTile` corrected to
`shownTiles + (1 if self-drawn) == 4`, **python MahjongGB matches the judge on all
12,088 wins**. The 118 games are tagged (`gb_discrepancy` = the 20 gate-relevant ones,
`gb_fan_diff` = the 98 milder ones) and carry a `known_discrepancy` field with both fan
lists side by side. **The official Botzone judge is the ground truth for engine
acceptance.** If your engine embeds a cai-style feature/legality layer, these are
exactly the games where you will diverge — handle them consciously rather than
discovering them as flaky test failures.

## Validating your engine

```bash
# 1. dataset self-test (no engine needed; re-checks every game against the
#    reference replay semantics: wall order, per-step hand/meld legality, claim
#    priority, terminal classification, exact score arithmetic):
python3 validate_engine.py --self-test mcr_final2026_full.jsonl.gz --jobs 16
#    -> self-test: 12288/12288 games PASS

# 2. interface smoke test with the built-in replay engine:
python3 validate_engine.py --demo mcr_final2026_golden.jsonl

# 3. your engine (start with the golden subset, then run the full set):
python3 validate_engine.py --engine mymodule:MyEngine mcr_final2026_golden.jsonl
python3 validate_engine.py --engine mymodule:MyEngine mcr_final2026_full.jsonl.gz
```

Your engine implements two methods (see the docstring in `validate_engine.py`):
`reset(wall, quan, srand) -> turn` and `step(responses) -> turn`, where a turn is
`{"requests": {seat: str}, "display": {...}}`. The validator feeds the logged responses
and requires byte-exact request strings and display events (`--loose` compares only
action/player/tile/tileCHI/hand/fan/fanCnt/score). Acceptance = 12,288/12,288 games
reproduced.

## Provenance & stats

- IJCAI-2026 Mahjong AI Competition, Final Stage 2 (Botzone contest
  `6a4eb8c71ba515095a1f1417`, finished 2026-07-09). 512 duplicate walls x 24 seat/wind
  permutations; finalists: kong/shiro, moyu/kdens3, QiuQiuR/丘丘人, player152/player2.
- 12,088 HU endings + 200 荒庄 draws; **zero** ERROR/TLE/invalid-move events anywhere in
  the corpus (all per-turn verdicts `OK`) — every game is a clean, complete protocol
  trace.
- Harvested from the public Botzone match archive; bot response `time`/`memory`/`debug`
  fields were stripped, everything else is verbatim.

## License / usage

Game logs of a public AI competition (Botzone platform), redistributed for research and
engineering use: engine testing, rules verification, imitation-learning research. The
bot policies that produced the moves remain the property of their authors; no bot code
or model weights are included. If you use this dataset in a publication, please cite the
IJCAI-2026 Mahjong AI Competition and link this dataset card.
