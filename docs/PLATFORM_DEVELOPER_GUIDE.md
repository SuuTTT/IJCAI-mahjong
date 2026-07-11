---
layout: default
title: "Platform Developer Guide — Chinese Standard Mahjong + the kdens3 Champion Bot"
---

# Platform Developer Guide: Chinese Standard Mahjong + the kdens3 Champion Bot

**Audience:** developers of a self-hosted, Botzone-like agent-arena platform (the kind that hosts
Minecraft / Clash Royale / RedAlert–style games) who want to
**(A)** implement Chinese Standard Mahjong as a platform game (judge/engine), and
**(B)** ship our IJCAI-2026 runner-up agent (**kdens3**) as the built-in "ultimate" opponent,
plus easier tiers below it.

Everything in this guide is grounded in this repository. Where a fact could not be verified
in-repo it is explicitly flagged. Key reference files:

| What | Where in this repo |
|---|---|
| Reference game engine (complete MCR flow) | `train/caiest_repro/sim_cnn.py` (class `Sim`); leaner twin: `train/sim.py` |
| Champion bot source (Botzone zip payload) | `deploy/caiest_cnn/` (`__main__.py`, `feature.py`, `ensemble_infer.py`, `numpy_resfused.py`, `model_resfused.py`) |
| Observation / action encoding | `deploy/caiest_cnn/feature.py` (byte-identical to `train/caiest_repro/feature.py`) |
| Duplicate-format evaluation | `eval/duplicate_eval.py`, `eval/run_match.py`, `eval/run_match_kr.py` |
| Baseline bots | `eval/sample.cpp` (random), `bot/mahjong_bot.py` (shanten heuristic) |
| JAX env + native-JAX net forward (§5) | `train/jax_env/` (`resnet_jax.py`, `obs38.py`, `agari_jax.py`, `fan_reward.py`, `train_ppo_ws.py`), `docs/JAX_RL_PROGRESS.md`, `train/jax_throughput_probe.py` |
| Model provenance | `doc/moyu_MODEL_CARD.md`, `doc/resbn40_MODEL_CARD.md`, `train/caiest_repro/UPLOAD_LOG.md` |
| Contest result forensics | `docs/blog/2026-07-10-anatomy-of-a-coin-flip-final.md` |

---

## 1. The game in five minutes (what your judge must enforce)

Chinese Standard Mahjong (国标麻将, "MCR" — Mahjong Competition Rules), 4 players, **136 tiles**,
**8-fan minimum to win**. No flower tiles in the Botzone/competition variant
(`flowerCount=0` everywhere; see `train/caiest_repro/sim_cnn.py:69`).

### 1.1 Tiles (34 kinds × 4 copies = 136)

| Code | Kind | Count | Notes |
|---|---|---|---|
| `W1`–`W9` | Characters (万, Wan) | 9 × 4 | suited |
| `T1`–`T9` | Bamboo (条, Tiao) | 9 × 4 | suited |
| `B1`–`B9` | Dots (筒, Bing) | 9 × 4 | suited |
| `F1`–`F4` | Winds (东南西北: E S W N) | 4 × 4 | honors |
| `J1`–`J3` | Dragons (中发白) | 3 × 4 | honors |

Wall construction reference: `_full_wall` in `train/caiest_repro/sim_cnn.py:52-57`.
Only `W/T/B` tiles can be CHI'd (sequences); `F/J` can only form triplets/kongs.

### 1.2 Setup and turn flow

- The shuffled 136-tile wall is split into **four private 34-tile segments, one per seat**
  (Botzone convention; `sim_cnn.py:116-117`). Each player draws only from their own segment.
- Each player is dealt **13 tiles**; the dealer (seat 0 / East in game-wind terms) takes the
  first draw, giving them the 14th tile (`sim_cnn.py:118-121, 175-176`).
- On your turn: draw 1 tile, then either **HU** (self-draw win), declare a **concealed kong
  (AnGang)** or **added kong (BuGang)**, or **discard** one tile.
- After any kong, the same player **draws again** before discarding.
- Every discard is offered to the other three players for claims (see priority below).
- If the current player's wall segment is empty when they must draw, the game ends in an
  **exhaustive draw** (荒庄 / "HUANG", `sim_cnn.py:173-174`) — all four scores are 0.

### 1.3 Claims on a discard — priority order

Reference: `_resolve_claims`, `sim_cnn.py:237-301`. Priority is strict:

1. **HU** (win on the discard, 点炮) — checked for all three other players in seat order from
   the discarder; requires calculated fan ≥ 8.
2. **PENG** (pung: 2 matching tiles in hand) or **GANG** (exposed kong: 3 matching tiles in
   hand). Kong only if the wall is non-empty (the claimant must draw a replacement).
3. **CHI** (sequence) — **only the next player** (left neighbor), only for suited tiles.

After a PENG or CHI, the claimant must immediately discard — and **that discard is itself
claimable**, recursively (`sim_cnn.py:273, 297`). Your judge must loop claim resolution, not
handle it once.

### 1.4 Kongs and robbing the kong

| Kong type | How | Visible to others? | Special rule |
|---|---|---|---|
| **AnGang** (concealed, 暗杠) | 4 identical tiles in hand, own turn | Event visible, tile hidden | — |
| **GANG** (exposed, 明杠) | claim a discard with 3 in hand | fully visible | — |
| **BuGang** (added, 补杠) | own turn, add drawn/held tile to an existing PENG | fully visible | **Rob-the-kong (抢杠和):** any other player may HU on the BuGang tile (`sim_cnn.py:200-211`, `_check_claims_hu_only:225-235`) |

### 1.5 Winning: the 8-fan minimum

A hand only wins if the MCR fan calculator scores **≥ 8 fan** (excluding the flat 8-point base).
Both the reference engine and the bot gate HU behind this check — an "illegal HU" attempt is
downgraded to a discard in the engine (`sim_cnn.py:188-191`) and never offered by the bot
(`feature.py:_check_mahjong`, lines 371-392). Use the `MahjongGB` library (§2.2) rather than
reimplementing MCR's 81 fan patterns.

---

## 2. Implementing the judge/engine on your platform

### 2.1 Reference implementation

`train/caiest_repro/sim_cnn.py` (class `Sim`, line 77; main loop `_loop`, lines 168-223) is a
complete, battle-tested MCR game engine in ~330 lines of Python: deal, draw, discard, claim
resolution with correct priority and recursion, all three kong types, rob-the-kong, self-draw
vs discard wins, wall exhaustion, and official scoring. `train/sim.py` is the same engine
without the CNN-feature and curriculum extras. Port either directly, or use them as the
executable spec for your own judge.

State machine (as implemented):

```text
DEAL: shuffle 136 -> 4x34 private walls; 13 tiles each; cur = dealer
LOOP (cur):
  if wall[cur] empty -> GAME OVER (HUANG, all scores 0)
  cur draws tile t
  ask cur:
    HU?      fan(hand, t, self_drawn=True) >= 8 -> SCORE_SELFDRAW, GAME OVER
    ANGANG?  4 copies in hand and wall non-empty -> meld, broadcast, goto LOOP (cur draws again)
    BUGANG?  t (or held tile) extends own PENG   -> broadcast
             offer HU-on-BuGang to other 3 (rob-the-kong) -> if HU: SCORE_RONG, GAME OVER
             else goto LOOP (cur draws again)
    else PLAY d (discard)
  RESOLVE_CLAIMS(d, src=cur):                    # priority HU > PENG/GANG > CHI
    for s in seat order after src: if fan(s, d) >= 8 and s claims HU -> SCORE_RONG, GAME OVER
    for s: PENG/GANG with 2/3 copies (GANG needs non-empty wall)
       GANG  -> meld; cur = s; goto LOOP (replacement draw)
       PENG  -> meld; s discards d2; RESOLVE_CLAIMS(d2, s)      # recursive!
    next player only: CHI (suited) -> meld; s discards d2; RESOLVE_CLAIMS(d2, s)
    no claim -> cur = next player; goto LOOP
```

**Simplifications in `Sim` that a production judge should fix:**

- `isWallLast` and `is4thTile` are hard-coded `False` in the fan call (`sim_cnn.py:69-71`), so
  the "last tile of wall / last of its kind" fan are never credited. Compute them properly.
- Robbing a **concealed** kong with Thirteen Orphans (the one MCR exception that allows it) is
  not implemented — only BuGang can be robbed.
- A `max_turns=300` safety cap ends the game as a draw (`sim_cnn.py:163, 223`); keep some cap.
- `Sim` deals from four pre-split stacks for speed; that matches the Botzone per-seat-wall
  convention, but verify it matches whatever wall rule you announce to users.

### 2.2 Fan calculation: use MahjongGB

The whole repo — engine and bot — scores hands with the **`MahjongGB`** library
(pip package **`PyMahjongGB`**, import name `MahjongGB`). Imports:
`train/caiest_repro/sim_cnn.py:31-37`, `deploy/caiest_cnn/feature.py` (and
`deploy/build/mahjong_bot.py:16`, which also uses its shanten helpers
`RegularShanten` / `SevenPairsShanten` / `MahjongShanten`).

The exact call, from `Sim._fan` (`sim_cnn.py:60-74`):

```python
from MahjongGB import MahjongFanCalculator

packs = tuple((meld_type, tile, 1) for meld_type, tile in melds)  # ("CHI"/"PENG"/"GANG", tile, offer)
r = MahjongFanCalculator(
    pack=packs,                # exposed melds
    hand=tuple(concealed),     # concealed tiles, winning tile EXCLUDED
    winTile=win,
    flowerCount=0,             # no flowers in this variant
    isSelfDrawn=is_self,       # zimo?
    is4thTile=False,           # production judge: compute for real
    isAboutKong=is_kong,       # kong-replacement draw / robbed kong
    isWallLast=False,          # production judge: compute for real
    seatWind=seat,             # 0..3
    prevalentWind=quan,        # 0..3 (round wind)
)
fan = sum(v for v, name in r)  # list of (fan_value, fan_name) pairs
win_ok = fan >= 8
```

The calculator raises on non-winning shapes — wrap it in try/except (the engine returns `-1`
on any exception, `sim_cnn.py:61,74`).

### 2.3 Single-game scoring (raw score)

Zero-sum, base 8 per opponent. Verbatim from `sim_cnn.py:303-311`:

```python
def _score_selfdraw(self, w, f):            # zimo: all three pay
    for s in range(4):
        self.scores[s] = 3*(8+f) if s == w else -(8+f)

def _score_rong(self, w, src, f):           # win by discard
    for s in range(4):
        if s == w:   self.scores[s] = 24 + f      # (8+f) from discarder + 8 from each other
        elif s == src: self.scores[s] = -(8+f)    # discarder pays base + fan
        else:        self.scores[s] = -8          # the other two pay base only
```

Exhaustive draw: all four scores are 0. On Botzone-style platforms an **invalid move scores a
heavy penalty** (treat it as an immediate loss for that seat; the IJCAI ladder showed 15–21%
of public-ladder games ending in bot ERRORs, so make the failure mode explicit and visible).

### 2.4 Match formats to support

1. **Single game, raw score** — §2.3 as-is. High variance; fine for casual ladder play.
2. **Duplicate-wall format** (the competition format): the *same* pre-dealt wall is replayed
   under **all 24 seat permutations** of the 4 contestants, cancelling luck. Reference:
   `eval/duplicate_eval.py` (`itertools.permutations(range(4))`, line 79). Two aggregation
   conventions — both were used at IJCAI-2026, so support both:
   - **Rank points:** per wall, sum each bot's 24-game raw scores, rank the four totals,
     award **4/3/2/1** points (ties split), sum rank points across walls; raw score as
     tiebreak (`eval/duplicate_eval.py:55-69,131-149`). Used by Swiss-style stages.
   - **Cumulative raw score:** just sum raw scores over all games. This was the Stage-2
     final metric (512 walls × 24 perms = 12,288 games;
     `docs/blog/2026-07-10-anatomy-of-a-coin-flip-final.md`). Note the two metrics can
     disagree on the winner — at IJCAI-2026 the bot with the best mean placement finished 3rd
     on cumulative score.
3. **Swiss stage** for large fields (16+ bots), then a small duplicate final — the IJCAI-2026
   structure (Stage-1 Swiss → 4-team duplicate Stage 2).

---

## 3. The bot I/O protocol (Botzone-compatible)

Implement this exactly and our bot (and any Botzone Mahjong bot) runs unmodified. The
authoritative parser is `process()` in `deploy/caiest_cnn/__main__.py:295-344`.

### 3.1 Envelope

Each turn the platform writes **one line of JSON** to the bot's stdin and reads **one line of
JSON** from its stdout:

```json
{"requests": ["0 0 0", "1 0 0 0 0 W1 W2 ...", "2 T7"], "responses": ["PASS", "PASS"]}
```

- **Stateless mode** (`long_running=false`): send the FULL history (`requests` = everything so
  far, `responses` = the bot's past answers) and restart or keep the process; the bot decides
  `requests[-1]`.
- **Long-running mode**: keep the process alive and send only the newest request each turn,
  e.g. `{"requests": ["3 1 PLAY T4"], "responses": []}`. On classic Botzone the bot signals it
  wants to stay alive by printing the marker line `>>>BOTZONE_REQUEST_KEEP_RUNNING<<<` after a
  raw (non-JSON) response.

Bot output: `{"response": "PLAY W9"}` (optionally with a `"debug"` field — surface it in your
match log; ours prints its model identity and RSS on turn 1, `__main__.py:346-364`).

**Our bot supports both modes in one binary** (`run()`, `__main__.py:433-458`): it keeps state
in process globals, and if it wakes up cold with a multi-entry history it reconstructs state by
replaying `requests`/`responses` (`run_json`, lines 420-431). So your platform may use either
mode; long-running saves ~0.65 s cold-start per turn. `BOTZONE_JSON=0` switches it to raw-line
(non-JSON) output with the keep-running marker, which is what the local harness
`eval/run_match_kr.py` uses.

### 3.2 Request grammar

Seats are `0..3`; winds `0..3` = E/S/W/N. `<tile>` uses the codes from §1.1.

| Request | Meaning |
|---|---|
| `0 <seat> <prevalentWind>` | Init: your seat and the round wind. Answer `PASS`. |
| `1 <f0> <f1> <f2> <f3> <13 tiles>` | Deal: four flower counts (always 0 here) then your 13 tiles. Answer `PASS`. |
| `2 <tile>` | You drew `<tile>`. Answer one of `PLAY`/`GANG <t>` (concealed)/`BUGANG <t>`/`HU`. |
| `3 <p> DRAW` | Player `p` drew (tile hidden). Answer `PASS`. |
| `3 <p> PLAY <tile>` | Player `p` discarded `<tile>`. Answer `PASS`/`PENG`/`CHI`/`GANG`/`HU`. |
| `3 <p> PENG <tile>` | `p` pung'd the last discard and then discarded `<tile>`. Claimable like a PLAY. |
| `3 <p> CHI <mid> <tile>` | `p` chi'd the last discard (sequence's **middle** tile = `<mid>`) and then discarded `<tile>`. Claimable like a PLAY. |
| `3 <p> GANG` | `p` declared a kong. **Context-dependent:** if `p`'s previous event was their own draw it is a concealed kong (tile hidden); otherwise it is an exposed kong of the last discard. Answer `PASS`. |
| `3 <p> BUGANG <tile>` | `p` added `<tile>` to an existing PENG. Answer `HU` (rob the kong) or `PASS`. |

Two subtleties your judge must reproduce (both visible in `__main__.py:313-343` and
`_replay_event:369-418`):

- **Echoes:** the judge sends every player's action to *everyone, including the actor*
  (`3 <yourSeat> PLAY W9` arrives after your own `PLAY W9`). Bots use the echo as confirmation.
- **Claim preemption:** if you answer `CHI ...` but another player answered `PENG`/`GANG`/`HU`
  (higher priority), your claim silently loses — the next request shows *their* action. The
  judge's echo is the only confirmation a claim succeeded. (Getting this wrong caused
  wrong-HU declarations in an early version of our tooling; the bot now applies a recorded
  claim only when the judge's echo confirms it.)
- **Concealed-kong echo carries no tile:** after you respond `GANG W5` on your draw turn, the
  echo is just `3 <you> GANG` — the bot remembers which tile it kong'd (`angang` global,
  `__main__.py:310,318`).

### 3.3 Response grammar

| Response | When | Meaning |
|---|---|---|
| `PASS` | any | no action |
| `PLAY <tile>` | own draw turn | discard `<tile>` |
| `HU` | own draw (zimo), someone's discard, someone's BUGANG | declare win |
| `PENG <tile>` | someone's discard | pung the discard, then discard `<tile>` |
| `CHI <mid> <tile>` | someone's discard | chi the discard into the sequence whose middle tile is `<mid>`, then discard `<tile>` |
| `GANG` | someone's discard | exposed kong of the discard |
| `GANG <tile>` | own draw turn | concealed kong of `<tile>` |
| `BUGANG <tile>` | own draw turn | add `<tile>` to your existing PENG |

### 3.4 Worked example (seat 0's transcript)

```text
platform -> {"requests":["0 0 0"],"responses":[]}
bot      <- {"response":"PASS"}
platform -> {"requests":["1 0 0 0 0 W1 W2 W3 T5 T6 B2 B3 B4 F1 F1 J3 W9 T9"],"responses":[]}
bot      <- {"response":"PASS"}
platform -> {"requests":["2 T7"],"responses":[]}          # we draw T7
bot      <- {"response":"PLAY W9"}
platform -> {"requests":["3 0 PLAY W9"],"responses":[]}   # our own echo
bot      <- {"response":"PASS"}
platform -> {"requests":["3 1 DRAW"],"responses":[]}
bot      <- {"response":"PASS"}
platform -> {"requests":["3 1 PLAY T4"],"responses":[]}   # left neighbor discards T4
bot      <- {"response":"CHI T5 B2"}                      # chi T4-T5-T6, discard B2
platform -> {"requests":["3 0 CHI T5 B2"],"responses":[]} # echo: our chi went through
bot      <- {"response":"PASS"}
...
platform -> {"requests":["3 2 PLAY W4"],"responses":[]}
bot      <- {"response":"HU"}                              # win by discard (fan >= 8)
```

---

## 4. Deploying kdens3 — the "ultimate" tier

### 4.1 What it is

**kdens3** is the agent that finished **2nd of 16 at the IJCAI-2026 Chinese Standard Mahjong
competition** — 3rd of 16 in the Stage-1 Swiss, then 594 points behind the champion
(t = 0.13, a statistical coin flip) over the 12,288-game duplicate final, with **zero
errors / timeouts / invalid moves in every game** (`README.md`,
`docs/blog/2026-07-10-anatomy-of-a-coin-flip-final.md`).

Technically (verified in `train/caiest_repro/UPLOAD_LOG.md:208-239`, `deploy/caiest_cnn/`):

- An **ensemble of 3 knowledge-distilled ResNet students**, each **128 channels × 40 residual
  blocks** (~14.3M params; `model_resfused.py:16`), trained with BatchNorm and deployed
  **BN-folded** (`ResFused`, numerically identical in eval mode).
- **Ensembling rule:** each model computes a softmax over the *legal* actions only; the three
  distributions are arithmetically averaged; the bot plays the argmax
  (`ensemble_infer.py:20-33`). Fully **deterministic** — no sampling, no RNG.
- **NumPy-only inference.** No torch, no GPU (`numpy_resfused.py`; the `numpy_only` marker file
  or `NUMPY_ONLY=1` skips torch entirely, `__main__.py:19-51`).
- Weights stored as **fp16 `.npz` (~26.6 MB each), loaded and computed in fp32**
  (`numpy_resfused.py:28`); fp16-vs-fp32 argmax parity 0/200, deploy-vs-trained parity 0/1000
  on real competition states (`UPLOAD_LOG.md`).

### 4.2 Get the pieces

| Piece | Where |
|---|---|
| Bot source (zip payload) | this repo, `deploy/caiest_cnn/` — you need `__main__.py`, `feature.py`, `ensemble_infer.py`, `numpy_resfused.py` (+ `model.py`/`model_resfused.py` only if you also want the torch path) |
| Student weights | HF **[`Dannibal/ijcai-mahjong-ckpts-2026`](https://huggingface.co/Dannibal/ijcai-mahjong-ckpts-2026)** → `deploy/kdens_s0_fp16.npz`, `deploy/kdens_s1_fp16.npz`, `deploy/kdens_s2_fp16.npz` (26.6 MB each; fp32 twins `deploy/kdens_s{0,1,2}.npz`, 53 MB each) |
| Ready-made shipped zips | same HF repo, `deploy/bot_KDENS_plain.zip` (and variants) — the exact competition builds |
| Single-teacher tier (aug_s0) | same HF repo, `ckpt/aug/aug_128x40_s0.pkl` (torch ckpt; convert/fold to `.npz` for numpy deploy) |
| moyu-era artifacts (predecessor) | HF `datasets/Dannibal/ijcai-mahjong-moyu-binaries-public` (see `doc/moyu_MODEL_CARD.md:93-109` for sha256s) |

Runtime dependencies: **Python ≥3.6, `numpy`, `PyMahjongGB`**. That's the whole list.

### 4.3 Assemble and run

```bash
pip install numpy PyMahjongGB

mkdir -p kdens3_bot/data && cd kdens3_bot
cp <repo>/deploy/caiest_cnn/{__main__.py,feature.py,ensemble_infer.py,numpy_resfused.py} .
touch numpy_only                       # skip torch entirely (~91MB base RSS instead of ~471MB)
# put kdens_s0_fp16.npz kdens_s1_fp16.npz kdens_s2_fp16.npz into data/

ENSEMBLE_NPZS="data/kdens_s0_fp16.npz,data/kdens_s1_fp16.npz,data/kdens_s2_fp16.npz" \
  python3 __main__.py
```

Model selection knobs (`__main__.py:58-158`): `ENSEMBLE_NPZS` (comma-separated list → the
3-model ensemble; highest priority), or a one-line `model.cfg` naming a single `.npz` in
`data/` (single-model tiers), or the canonical fallback `data/cnn.npz`. The competition zips
baked the ensemble spec into the zip instead of using the env var — the archived zips on HF
show that wrapper; the repo source uses `ENSEMBLE_NPZS`.

Quick interactive smoke:

```bash
$ ENSEMBLE_NPZS=... python3 __main__.py
{"requests":["0 0 0"],"responses":[]}
{"response": "PASS", "debug": "v=0612c PIMC=off rss=...MB ensemble|ensemble:3"}
{"requests":["1 0 0 0 0 W1 W2 W3 T5 T6 B2 B3 B4 F1 F1 J3 W9 T9"],"responses":[]}
{"response": "PASS"}
{"requests":["2 T7"],"responses":[]}
{"response": "PLAY ..."}
```

Check the turn-1 `debug` string: it must contain `ensemble:3` (ensemble engaged) and shows the
live RSS.

### 4.4 Resource budget

| Resource | kdens3 measured | Recommended platform limit |
|---|---|---|
| CPU | 1 core, single-threaded NumPy | 1 core |
| Time/move | median ~0.9–1.5 s, max ~1.7 s (Botzone-class CPU; `UPLOAD_LOG.md:217-218,231`) | 6 s (the Botzone limit; big margin) |
| RAM | ~91 MB NumPy base + 3 × ~53 MB fp32 weights in memory ⇒ ~250–350 MB class; ran within Botzone's per-turn cap | 512 MB |
| Disk | 3 × 26.6 MB weights + ~50 KB code | 128 MB |
| Cold start | ~0.65 s (lean file set; a fat 25-file dir once caused a 15.6 s turn-1 timeout — keep the payload lean, `doc/moyu_MODEL_CARD.md:23`) | — |

*(An exact "330 MB" RSS figure could not be verified in-repo; measure via the turn-1 debug
line on your hardware.)*

Reliability: crash-safe by construction — any exception in a turn emits `PASS`
(`__main__.py:457-458`), and `PASS` is always legal at claim decisions. This is a large part
of why it had 0 errors in 12,288 final games while public-ladder bots error out in 15–21% of
games; give your built-in bots the same property.

### 4.5 Wrapping options

**Option A — process per seat (recommended, language-agnostic):** run
`python3 __main__.py` per seat, speak §3 over stdin/stdout, kill after the game. Use
long-running mode within a game. This is exactly how Botzone runs it. `deploy/local_ai.py`
is a working subprocess driver you can crib (sentinel handshake, timeouts).

**Option B — in-process Python class:** skip the pipe and reuse the same modules:

```python
import __main__ as kdens          # the bot module, with ENSEMBLE_NPZS set before import

class KdensSeat:
    def __init__(self, seat, prevalent_wind):
        kdens.agent = None
        kdens.process(f"0 {seat} {prevalent_wind}")
    def on_request(self, line: str) -> str:      # line = one §3.2 request
        return kdens.process(line)
```

(One instance per seat per game; the module keeps per-game state in globals, so use separate
processes — or copies of the module — for concurrent games.)

### 4.6 Observation and action encoding (for your own agents / debugging)

From `deploy/caiest_cnn/feature.py` (`OBS_SIZE=38`, `ACT_SIZE=235`, lines 20-42).
Observation is a **38 × 4 × 9** binary tensor (34 tile kinds laid out 4×9):

| Planes | Content |
|---|---|
| 0 | seat wind |
| 1 | prevalent wind |
| 2–5 | own hand (count-encoded: n copies → first n planes) |
| 6–21 | discard history, 4 planes × 4 players |
| 22–37 | exposed melds (chi/peng/gang expanded to tiles), 4 planes × 4 players |

Action space, 235 discrete actions with a legality mask:

| Action | Indices | Count |
|---|---|---|
| Pass | 0 | 1 |
| Hu | 1 | 1 |
| Play (discard) | 2–35 | 34 |
| Chi | 36–98 | 63 (3 suits × 7 middles × 3 positions) |
| Peng | 99–132 | 34 |
| Gang (exposed) | 133–166 | 34 |
| AnGang | 167–200 | 34 |
| BuGang | 201–234 | 34 |

---

## 5. JAX-First Integration

If your platform is JAX-first (pgx / JaxMARL-style vectorized engines), you do **not** need the
NumPy sidecar process of §4 to host kdens3 — the net runs natively in JAX — and this repo
already contains a partial JAX Mahjong environment you can build on. This section describes
what exists, how to run kdens3 in-graph, the honest limitations, and a roadmap to a fully
native JAX MCR engine.

### 5.1 What already exists in this repo (state as committed)

All under `train/jax_env/` unless noted; status doc: `docs/JAX_RL_PROGRESS.md`; the original
go/no-go throughput probe is `train/jax_throughput_probe.py`.

| Piece | File(s) | State |
|---|---|---|
| Vectorized env core (Phase 1) | `train/jax_env/csm_env.py`, `bench.py` | **Done.** State arrays, reset/deal, draw→discard round-robin, obs encoding, fixed action space, `jit`+`vmap` rollout. **Discard-only: no CHI/PENG/GANG claims yet** (`train/jax_env/README.md`, Phase 3 unchecked). |
| Win detection (Phase 2) | `agari.py` (numpy ref), `agari_jax.py` (GPU), `build_agari_tables.py` | **Done & validated:** numpy ref 100% vs `MahjongGB` on 10k hands; JAX 0 mismatches vs ref on 20k hands; 0 false wins in 131,072 self-play games (`docs/JAX_RL_PROGRESS.md:8-12,41-42`). Batched via precomputed per-group feasibility tables + a small DP. |
| Terminal fan reward | `fan_reward.py` | **Hybrid:** JAX detects the win per step; Python `MahjongGB` scores exact fan at the (rare) terminals; 8-fan floor + MCR duplicate scoring. Reproduces 50/52 real game finishes (`docs/JAX_RL_PROGRESS.md:13-16`). |
| Deploy-parity observation | `obs38.py` | JAX-batched 38-plane CAIEST obs, **byte-exact vs `feature.py`** for the discard-only env (meld planes zero there; see §5.3). |
| Native-JAX net forward | `resnet_jax.py` | **Validated** — see §5.2. |
| Warm-started self-play PPO | `train_ppo_ws.py` (from-scratch twin: `train_ppo.py`) | Built and runs; full-40-block-net RL measured **compute-infeasible** (~50 min/iter on an A4000, forward-bound — `CHANGELOG.md:33-41`). Useful as integration reference, not as a training recipe. |
| Throughput | `train/jax_env/README.md`, `bench.py` | Phase-1 env + small conv policy: 424k–1.53M env-steps/s (batch 256–16384, one RTX 3060) ≈ 5.9k–21k full games/s; win-aware random-policy self-play measured 589k games/s at B=65536 on an A4000. |

### 5.2 Running kdens3 natively in JAX

`train/jax_env/resnet_jax.py` is a JAX forward of the exact deploy architecture
(BN-folded ResNet, stem Conv3×3 → N residual blocks → 512-d foot → 235 logits). It loads the
**same public `.npz` weight files** the NumPy bot uses and was validated against the NumPy
deploy net: **argmax agreement 16/16, max logit error ≈0.005** (`CHANGELOG.md:34-35`).
So the whole champion tier is: load the 3 student `.npz` files (§4.2), run this forward per
student, mean the masked softmaxes, argmax — batched and jittable, no host round-trip.

```python
# adapted from train/jax_env/resnet_jax.py + deploy/caiest_cnn/ensemble_infer.py
import jax, jax.numpy as jnp
from resnet_jax import load_params, forward_feats     # train/jax_env/resnet_jax.py

def load_student(npz_path):
    p = load_params(npz_path)          # fp16 npz -> fp32 JAX pytree (same files as §4.2)
    nb = p.pop('_blocks')              # 40
    return p, nb

STUDENTS = [load_student(f"kdens_s{i}_fp16.npz") for i in range(3)]

@jax.jit
def kdens3_act(obs, mask):
    """obs (B,38,4,9) float32, mask (B,235) bool -> greedy action (B,) int32.
    Same semantics as ensemble_infer.Ensemble: mean softmax over LEGAL actions, argmax."""
    m = mask.astype(jnp.float32)
    acc = jnp.zeros_like(m)
    for p, nb in STUDENTS:             # 3 students, unrolled at trace time
        logits, _ = forward_feats(p, obs, nb)
        logits = jnp.where(m > 0, logits, -1e30)
        acc = acc + jax.nn.softmax(logits, axis=-1)
    return jnp.argmax(acc * m, axis=-1)
```

One `kdens3_act` call scores every table in a vectorized arena at once — this is the natural
deployment on a pgx-style platform (the per-seat process of §4.5 remains available for
non-JAX callers). Notes:

- `forward_feats` also returns the 512-d penultimate features (`resnet_jax.py:30-44`) —
  handy if you later attach a value head, exactly as `train_ppo_ws.py:36-50` does.
- **Re-verify argmax parity on your hardware** before shipping: GPU TF32 matmuls can perturb
  logits; the repo's own discipline was 0/1000 argmax flips deploy-vs-trained on real
  competition states. Run a few hundred states through both the NumPy path
  (`numpy_resfused.py`) and the JAX path and require zero argmax flips (float32-precision
  convs, e.g. `jax.default_matmul_precision('float32')`, if you see any).
- The 16/16 argmax / ~0.005-logit validation on record was performed with the same-family
  deploy net `cnn_lad_chunjiandu.npz` (identical `.npz` layout and architecture); per-student
  parity checks for `kdens_s{0,1,2}` are your smoke test, per the previous bullet.

### 5.3 Honest limitations (read before you commit to a design)

**(1) The env is not end-to-end jittable — fan scoring is Python-on-CPU.**
Win *detection* runs on GPU every step (`agari_jax.py`), but exact fan *scoring* calls the
Python `MahjongGB` library on the host at terminal states (`fan_reward.py`,
`train_ppo_ws.py:107-139` `score_terminals` — a per-game Python loop after each rollout batch).
This is deliberate: fan depends on the max-fan *decomposition* of the hand, which is exactly
what makes `MahjongFanCalculator` complex; vectorizing all 81 fans in JAX was judged
impractical and error-prone (`docs/JAX_RL_PROGRESS.md:18-20`). Consequences: `lax.scan`
rollouts must return terminal hands to the host for scoring (or use a batched
`jax.pure_callback`), and once your policy net is small the host-side Python scorer, not the
GPU, is the throughput ceiling. (In-repo measurements cover the full-net case, which was
GPU-forward-bound; the small-net/CPU-bound regime is expected from the design, not a logged
measurement — budget for it.)

> **(2) Warning — the `verbose=False` scorer bug. Do not reintroduce it.**
> `MahjongFanCalculator(..., verbose=False)` returns `(fanCount, fanName)` **2-tuples**;
> `verbose=True` returns `(fanValue, count, cn, en)` **4-tuples**. Training code here once
> unpacked 2-tuples as if they were 4-tuples: `sum(fp * c for fp, c, *_ in fans)` bound `c` to
> the *name string*, `int * str` raised inside a broad `except`, and **every win silently
> scored fan = 0** — the "RL can't reach 8 fan" null was partly this instrumentation artifact
> (measured: win8 0.00% buggy → 53.25% fixed, same policy;
> `docs/FINDINGS_2026-06-14.md:5-45`, `CHANGELOG.md:37-39`). Rules: either call with
> `verbose=True` and `sum(fanValue * count …)` (`train_ppo_ws.py:126-127`), or with
> `verbose=False` and `sum(cnt for cnt, _ in result)` — and never let a bare `except` around
> the scorer default to 0 without a counter/log you monitor.

**(3) Encodings must match the deploy net exactly.**
The JAX path must feed the same **38×4×9 obs** and **235-action** space as §4.6.
`obs38.py` is the byte-exact JAX obs encoder, but note it currently covers the
**discard-only** env: meld ("half-flush") planes 22–37 are always zero there
(`obs38.py:6-8,17`). Once your engine has claims, encode melds exactly as
`deploy/caiest_cnn/feature.py` does or the net's inputs are silently off-distribution.
The trickiest index arithmetic is the **Chi block** (63 actions):

```text
chi_action = 36 + suit_index('WTB') * 21 + (mid - 2) * 3 + (claimed - mid + 1)
# suit_index: W=0 T=1 B=2;  mid = middle tile number of the sequence (2..8);
# claimed = the discarded tile's number; last term is 0/1/2 = claimed is low/mid/high
```

Canonical implementations: `deploy/caiest_cnn/feature.py:167-171` (mask construction),
`:289-293` (`action2response`), `:318` (`response2action`). This formula is the classic place
to introduce an off-by-one (mid vs claimed confusion) — unit-test all 63 Chi actions
round-trip through `action2response`/`response2action` against those lines before trusting
any JAX reimplementation. *(A historical Chi-formula bug in a `response2action`
reimplementation is remembered from the campaign but is not written up in this repo's docs —
treat the cited `feature.py` lines as the single source of truth.)*

### 5.4 Roadmap: a fully-native JAX MCR engine (pgx-style)

What exists here is a validated foundation (win detection, obs, net forward, hybrid reward),
not a finished pgx-style game. The staged path we recommend (and partially executed):

- **v0 — what's in `train/jax_env/` today:** jitted discard-only flow + GPU win detection +
  host-side terminal fan scoring. Good enough for RL experiments; not a full game.
- **v1 — full game, hybrid scoring:** implement claims (Phase 3, `train/jax_env/README.md:25-26`)
  branchlessly inside the jitted step:
  - **Fixed-size state tensors:** wall as a `(136,)` int32 dealt-order permutation with 4
    per-seat pointers (§1.2 wall convention), hands as `(4,34)` int8 counts (as `csm_env.py`
    already does), melds as a `(4,4,3)` int array (type, tile, offer) with a count — never
    Python lists.
  - **Branchless claim resolution:** compute HU/PENG-GANG/CHI eligibility masks for all four
    seats simultaneously, apply the §1.3 priority as a vectorized `argmax` over
    (priority, seat-order) keys, and resolve the post-claim discard as a normal step —
    the recursive chain of `sim_cnn.py:_resolve_claims` becomes a loop of masked steps.
  - **Scoring stays hybrid:** keep `MahjongGB` on the host, but batch it — collect terminal
    states across the whole vmapped batch and score them in one host visit per rollout
    (what `train_ppo_ws.py` does), or wrap it in a batched `jax.pure_callback` at terminals
    only. Terminal states are rare (~1 per ~72 steps × only won games), so this costs little
    until nets get tiny.
  - Validate exactly the way this repo did: every component gated against `MahjongGB` /
    `sim_cnn.py` ground truth on 10k+ cases before use (`docs/JAX_RL_PROGRESS.md:7-16`).
- **v2 — tensorized scoring for the hot path:** implement the ~10 most frequent fans as JAX
  table lookups/DPs (the `agari_jax.py` precomputed-feasibility-table + small-DP pattern
  generalizes), use them for in-graph reward, and keep the exact `MahjongGB` callback as a
  fallback for rare/composite hands **and as the referee for scoring that decides matches** —
  approximate fans are acceptable for RL shaping, never for official match results.

The throughput prize is real — the Phase-1 measurements above are ~85× the CPU self-play rate
on one mid-tier GPU (`train/jax_env/README.md:7-17`) — but the fan calculator is the hard
part, and every shortcut around it in this campaign that went unvalidated produced a false
conclusion. Gate each stage against the Python ground truth.

### 5.5 Build ownership & acceptance tests

**Ownership.** The platform team implements the native JAX MCR engine inside its own codebase,
under its own jit/test conventions. The mahjong-campaign side supplies the spec (this guide +
the reference `Sim`, §2.1), the validated kdens3 JAX forward (§5.2), and the two acceptance
suites below. **The engine is DONE when both suites pass** — neither before, nor is anything
else required.

**Acceptance Suite A — replay equivalence vs the official judge.**
Replay all **12,288 IJCAI-2026 Final Stage-2 games** through the new engine and require an
exact match with the official judge's outcome on every game: each action's legality, every
claim resolution (priority + preemption), every win/fan decision, and the final four-seat
score. **The oracle corpus is published as a public HuggingFace dataset:**
**<https://huggingface.co/datasets/Dannibal/mcr-final2026-testset>** — one documented JSON
record per game (full ordered wall, judge `srand`, the verbatim per-seat request/response
protocol stream, and an `expected` terminal block), plus a 221-game golden edge-case subset
(qianggang, AnGang/BuGang, multi-claim priority, 8-fan boundary, known judge-vs-MahjongGB
replay discrepancies) and `validate_engine.py`, a stdlib reference validator with a
documented engine-interface stub (`reset(wall, quan, srand)` / `step(responses)`); its
`--self-test` passes 12,288/12,288 games. The dataset card documents the wall/deal order,
the PENG/CHI-embedded-discard convention, GANG discrimination, claim priority and score
arithmetic — read its "four classic replay traps" section before writing the engine. It is
the same corpus behind `docs/blog/2026-07-10-anatomy-of-a-coin-flip-final.md`, whose
headline totals — 12,088 wins + 200 exhaustive draws, zero ERROR endings — double as quick
sanity checks for your replay run.

**Acceptance Suite B — gate equivalence.**
Run the campaign's calibrated duplicate placement gate **on the JAX engine**: kdens3
(3-ensemble) vs `aug_s0`, **≥24,000 games** (the campaign's confirming gates used ~24 blocks
× 2,000 games; `train/caiest_repro/UPLOAD_LOG.md:208-211`). Two requirements:

1. The placement estimate must agree with the CPU reference gate within its 95% CI
   (reference: kdens3 mean placement 2.5054–2.5057, CI lower bound > 2.500).
2. **The calibration trap must still hold:** candidate == reference (aug_s0 vs aug_s0) must
   read **exactly 2.500** (`train/caiest_repro/AUG_WRITEUP.md:3-5`,
   `train/caiest_repro/ARCH_WRITEUP.md:3`). If the self-vs-self gate drifts off 2.500, the
   engine or the harness's seating/permutation logic is biased — fix that before reading any
   other number.

Together, the engine + the 12,288-game oracle corpus + the calibrated gate harness + the
baseline ladder (§6) are intended for release as an open benchmark suite.

---

## 6. The difficulty ladder

| Tier | Bot | Source | Behavior / strength |
|---|---|---|---|
| Easy | random-legal | `eval/sample.cpp` (the official-style sample bot) | shuffles its hand and discards; never claims, never wins. Pure tutorial fodder. |
| Medium | shanten-greedy | `bot/mahjong_bot.py` (same heuristic is the no-model fallback in `bot/ml_bot.py`) | discards to minimize shanten, PENG/CHI when it reduces shanten, HU only when `MahjongGB` confirms ≥8 fan. Plays "sensibly" but fan-blind. |
| Hard | **aug_s0** — single 128×40 CNN | `deploy/caiest_cnn/` + HF `ckpt/aug/aug_128x40_s0.pkl` (fold to `.npz`, point `model.cfg` at it) | the strongest *single* net of the campaign (~28 ms/move CPU); kdens3's gate reference. |
| Ultimate | **kdens3** — 3× KD ensemble | §4 | IJCAI-2026 runner-up; CI-separated **above** aug_s0 (+0.0055 mean placement, twice replicated — `train/caiest_repro/UPLOAD_LOG.md:208-211`). |

The Hard↔Ultimate gap is real but small (that +0.0055 was the only lever that beat the
imitation ceiling all campaign); the Medium↔Hard gap is enormous. If you want a smoother ramp,
add a "single kdens student" tier between them (any one `kdens_s*.npz` via `model.cfg` —
singles gate at aug_s0 parity).

---

## 7. Smoke tests and replay verification

Before going live, verify your judge and the bot against each other:

1. **Protocol echo test:** pipe the §3.4 transcript into the bot; every response must parse
   under §3.3 and be legal.
2. **Self-play soak:** four kdens3 seats, ≥1,000 games. Assert: zero exceptions, zero illegal
   responses, draw rate low single-digit % (2.8% on the real ladder; the final measured 1.63%
   exhaustive draws), per-game scores zero-sum, HU hands re-validate at ≥8 fan.
3. **Replay determinism:** kdens3 is deterministic, so re-feeding a logged request stream must
   reproduce the logged responses bit-for-bit. This is the strongest single check of your
   judge's request grammar.
4. **Cross-check scoring** against §2.3 on logged wins, and legality with
   `eval/catch_illegal.py` / `eval/oneshot_legality.py`.
5. **Duplicate harness:** run `eval/duplicate_eval.py`-style wall replays (4 walls × 24 perms)
   between tiers; the ladder order of §6 must reproduce.

> **Warning — do not copy `eval/replay_harness.py` semantics into your judge.** It is a
> failure-*mining* tool, self-documented as only ~59% faithful, with known holes:
> it snapshots decisions only on draws (the mandatory discard embedded in a PENG/CHI claim is
> not reconstructed as its own decision), it has **no BUGANG branch at all** (added kongs are
> silently skipped, so rob-the-kong never happens), and it does not distinguish AnGang from
> exposed GANG (both become `Player N Gang`, which desyncs the feature encoder on concealed
> kongs). There is no fixed "harness2" in this repo; the trustworthy references are the live
> bot path (`deploy/caiest_cnn/__main__.py` — including the echo-confirmed claim logic of
> `_replay_event`) and the audit oracle `eval/replay_audit.py`. Also see
> `docs/STATUS_2026-06-07.md`: a preempted-CHI desync once caused wrong-HU declarations worth
> −30 each; echo-confirmation is the fix.

---

## 8. Integration checklist

- [ ] Judge implements §1 rules: 136 tiles, per-seat walls, claim priority HU > PENG/GANG > CHI, recursive claim resolution, all 3 kong types + rob-the-kong, 8-fan minimum via `MahjongGB`, exhaustion draw.
- [ ] Judge computes `isWallLast` / `is4thTile` honestly (unlike the reference sim).
- [ ] Scoring: §2.3 raw formulas; zero-sum asserted per game; explicit invalid-move penalty.
- [ ] Protocol: §3 grammar byte-exact, including self-echoes, claim preemption, the tile-less concealed-kong echo, and the flower-count fields in request `1`.
- [ ] Both stateless (full-history) and long-running modes supported (our bot handles either).
- [ ] Duplicate-wall match runner: same wall × 24 seat permutations; both rank-point and cumulative-score aggregation.
- [ ] kdens3 deployed per §4.3 with `numpy_only`; turn-1 debug shows `ensemble:3`.
- [ ] Resource limits ≥ §4.4 (1 core / 512 MB / 6 s per move).
- [ ] Easier tiers wired (§6) and the ladder order verified by a duplicate gauntlet.
- [ ] JAX-first deployments: per-student argmax parity vs the NumPy path verified (§5.2), scorer called per the §5.3 warning, encodings round-trip-tested against `feature.py`.
- [ ] §7 smoke tests green, incl. bit-exact replay determinism.
- [ ] License cleared (below) and weight files' checksums verified against `doc/moyu_MODEL_CARD.md` / HF LFS sha256s.

---

## 9. License and links

**License:** this repository currently ships **no LICENSE file**. Before embedding the code or
weights in a third-party platform, ask the maintainer to add one (MIT or Apache-2.0
recommended) or obtain written permission. Note also that `MahjongGB`/PyMahjongGB and the
Botzone protocol conventions have their own upstream licenses/terms — check them independently.

**Links**

- Project site & blog: <https://suuttt.github.io/IJCAI-mahjong/> — start with
  [From 11th to Runner-Up](https://suuttt.github.io/IJCAI-mahjong/blog/2026-07-10-from-11th-to-runner-up.html)
  and [Anatomy of a Coin-Flip Final](https://suuttt.github.io/IJCAI-mahjong/blog/2026-07-10-anatomy-of-a-coin-flip-final.html)
- Code: <https://github.com/SuuTTT/IJCAI-mahjong>
- Champion weights + shipped bot zips: <https://huggingface.co/Dannibal/ijcai-mahjong-ckpts-2026>
- Predecessor (moyu) artifacts + official 98k dataset: <https://huggingface.co/datasets/Dannibal/ijcai-mahjong-moyu-binaries-public>
- Fan calculator: `pip install PyMahjongGB` (import `MahjongGB`)

*Facts in this guide were verified against the repository at the commit that introduced this
file; anything that could not be verified in-repo is flagged inline.*
