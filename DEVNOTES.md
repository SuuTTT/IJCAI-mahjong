# IJCAI Mahjong AI — Development Notes

## 1. What the Bot Does

### Protocol
Botzone drives all bots through **JSON stdin/stdout**. Each turn, the bot receives a JSON object containing the full history of requests and responses for the current game, plus one new request it must answer. It prints a JSON object with a single `"response"` key and exits.

```
stdin  → {"requests": [..., "2 W5"],  "responses": [..., "PASS"]}
stdout ← {"response": "PLAY B3"}
```

The game runs on Ubuntu 16.04, single core, **1 second** per response (6 seconds for Python due to a 6× multiplier). Memory is capped at 256 MB.

### Request types the bot handles

| Request prefix | Meaning | Bot's job |
|---|---|---|
| `0 pid wind` | Game init: my seat & prevalent wind | Store, output `PASS` |
| `1 f0 f1 f2 f3 t1…t13` | Initial deal (13 tiles) | Store hand, output `PASS` |
| `2 tile` | I drew `tile` | Decide: `HU` / `PLAY t` / `GANG t` / `BUGANG t` |
| `3 pid PLAY tile` | Player `pid` discarded `tile` | Decide: `HU` / `PENG t` / `GANG` / `CHI mid t` / `PASS` |
| `3 pid DRAW` | Player `pid` drew (no tile shown) | `PASS` |
| `3 pid GANG` | Player `pid` declared concealed kong | `PASS` |
| `3 pid BUGANG tile` | Player `pid` upgraded PENG → GANG | Decide: `HU` (抢杠和) or `PASS` |
| `3 pid PENG/CHI …` | Other players' melds | `PASS` |

### Decision logic (v0.1)

**After my draw (`2 tile`):**
1. Check if I can win: pass hand (without the drawn tile) + drawn tile to the fan calculator. If total fan ≥ 8 → `HU`.
2. Check BUGANG: if I penged a tile earlier and drew another copy, upgrade to kong — but only if it does not increase my shanten number.
3. Check ANGANG (concealed kong): if I have 4 copies of any tile, kong it — only if shanten is not harmed.
4. Otherwise, try removing each tile one by one, compute shanten of the remaining 13, and discard the tile that minimizes shanten.

**After another player discards (`3 pid PLAY tile`):**
1. Check HU (winning by claim): same fan-calculator guard.
2. Check GANG (have 3 of the discarded tile): accept if it does not raise shanten.
3. Check PENG (have 2): simulate peng → best discard; accept if that gives lower shanten than current hand.
4. Check CHI (only if I am the next player): same simulation.
5. Otherwise `PASS`.

**Safety invariant:** The bot never outputs an action unless it has already verified legality from its own state. No HU without fan ≥ 8. No PLAY of a tile not in hand. This is the most important property — a single `WA` or `WH` penalty is −30 points, dwarfing a typical winning score.

---

## 2. How We Test

### Unit tests (`tests/test_bot.py`, 22 cases)

Run with:
```bash
cd /home/coder/IJCAI-mahjong
python3 -m pytest tests/test_bot.py -v
```

Covers:
- **Tile encoding**: `tile_id` / `tile_from_id` round-trips for all 34 tile types.
- **Shanten calculator**: tenpai hands return 0, known-far hands return the expected number, 7-pairs detection.
- **Fan calculator**: confirmed high-fan hand returns ≥ 8, invalid hand returns −1.
- **State reconstruction**: `apply_deal`, `apply_draw`, `apply_my_play`, `apply_my_peng`, `can_chi`, `can_peng` all work correctly.
- **Decision logic**: `decide_after_draw` always returns a tile that is in the hand; `decide_after_discard` returns `PASS` when the bot is not the next player and cannot benefit.
- **Integration**: three subprocess tests spin up the Python bot process with real JSON and verify the response format.

### Stress test (`tests/stress_test.py`, 200 games)

Run with:
```bash
python3 tests/stress_test.py
```

Simulates 200 independent games (seeded for reproducibility). Each game:
1. Shuffles a fresh deck and deals 13 tiles to player 0.
2. Drives the C++ bot through up to 40 draw-play cycles per game.
3. After each draw, verifies:
   - Response is one of `HU` / `PLAY tile` / `GANG tile` / `BUGANG tile`.
   - The discarded tile exists in the tracked hand.
   - When notified of its own play, the bot responds `PASS`.
4. Also drives one "other player draws + plays" event per turn, verifying the response is one of the legal responses.

**Current result:** 200 games, 5 432 draw turns, **0 illegal moves detected**.

### Build pipeline

```bash
cd /home/coder/IJCAI-mahjong/bot
make all     # builds bot_local, bot_submit.cpp, mahjong_bot.zip
make test    # runs unit tests + stress test
```

---

## 3. Can We Test Locally as if It Were the Official Competition?

### What the official competition actually does

The official format is **duplicate mahjong (复式赛)**:

- 4 fixed tile walls are used.
- For each wall, all **24 seat-order permutations** are played (4! = 24 games).
- A game's scores are accumulated into **小分 (micro-scores)**.
- After 24 games on one wall, the 4 players' micro-score totals are ranked 1–4 and converted to **4/3/2/1 ranking points**.
- The 4 walls × ranking points are summed; ties broken by total micro-score.

This means random luck is largely cancelled: the same tile walls are seen by all bots, so the ranking measures **strategy** rather than who drew a lucky hand.

### What we can replicate locally

| Component | Replicable? | How |
|---|---|---|
| Single-game I/O protocol | ✅ Yes | Feed bot JSON directly; use the official `judge/main.cpp` |
| Fan calculation & win validation | ✅ Yes | MahjongGB embedded in bot; PyMahjongGB for tests |
| Shanten & legality | ✅ Yes | Same algorithm used in bot and tests |
| Judge behavior (WA/WH/scoring) | ✅ Yes (with effort) | Compile `judge/main.cpp` with jsoncpp + boost; then write a 4-bot runner |
| Duplicate packet evaluation | ✅ Yes | Script: fix wall seed, run 24 permutations, accumulate micro-scores |
| Opponent diversity | ⚠️ Partial | Only one bot right now; need opponent pool (random, heuristic, past checkpoints) |
| Exact Botzone ranking | ❌ No | We cannot replay Botzone's matchmaking schedule or see other teams' bots |

### Practical local duplicate evaluator (what to build next)

```
eval/
  duplicate_eval.py     # fix 4 walls × 24 seat orders, call judge for each
  opponents.yaml        # list of opponent bots (paths to binaries)
  run_eval.sh           # one-command duplicate evaluation
  seedsets/             # pre-generated wall seeds for stable comparison
```

A minimal evaluation loop:

```python
# pseudo-code
for wall_seed in range(4):
    micro_scores = {pid: 0 for pid in range(4)}
    for perm in all_permutations([0,1,2,3]):   # 24 permutations
        scores = run_single_game(wall_seed, seat_order=perm, bots=[...])
        for pid, score in zip(perm, scores):
            micro_scores[pid] += score
    ranking_pts = scores_to_ranking(micro_scores)
    # accumulate ranking_pts
```

This is the only evaluation metric that directly predicts contest placement. Ordinary win-rate or Elo on random games will mislead you.

### Setting up the local judge

```bash
# Install dependencies
sudo apt-get install libboost-dev libjsoncpp-dev

# Compile the judge
cd /workspace/Chinese-Standard-Mahjong/judge
g++ -O2 -std=c++14 \
    -I/workspace/Chinese-Standard-Mahjong/fan-calculator-usage \
    main.cpp -lboost_system -ljsoncpp -o judge

# The judge reads a JSON "match request" on stdin and writes round results.
# A wrapper script needs to run 4 bots as child processes and pipe I/O.
```

---

## 4. Will a Learning Model Perform Better?

**Yes, substantially.** Here is what the evidence says and what the gap looks like.

### Gap between rule-based and learning-based bots

Historical IJCAI mahjong results (from official papers and public reports):

| Approach | Typical tier |
|---|---|
| Random play (official sample) | Bottom |
| Hand-crafted heuristic (shanten + danger) | Mid-table |
| Supervised learning (imitation of strong players) | Top-30 |
| SL warm-start + PPO self-play | Top-10 |
| SL + PPO + league training + risk model | Champion contender |

Our v0.1 is in the "hand-crafted heuristic" category. It never misplays a legal move, which is a necessary precondition for everything above it — but it has no model of **risk**, **opponent behavior**, or **long-horizon planning**.

### Why heuristics hit a ceiling

1. **No danger estimation.** Our bot discards whatever minimizes shanten, even if that tile is the winning tile of another player. A model trained on game data learns discard danger implicitly.
2. **No opponent modelling.** We cannot infer what other players are building from their discards and melds.
3. **No fan shaping.** Two hands at tenpai-0 may have vastly different expected fan counts (e.g., 8 fan vs. 48 fan). Shanten alone cannot distinguish them.
4. **Static PENG/CHI evaluation.** We PENG only if it immediately reduces shanten — but sometimes the right play is to wait for a higher-value tenpai later.

---

## 5. Future Version Roadmap

### v0.2 — Danger-aware discard (1–2 weeks)

**Goal:** Stop losing points to obvious放铳 (dealing into wins).

Changes needed:
- Track all discarded and claimed tiles globally (already partially done via `shownTile`).
- Implement a tile danger score: if tile X has had 3 copies discarded/claimed, the 4th copy is 绝张 (last tile) — very dangerous to discard.
- Apply a danger penalty to the shanten-minimization score: `score = -shanten * 100 + useful_tiles * 2 - danger * 50`.
- Add "defense mode": if we detect another player is likely tenpai (few discards, melds visible), switch to safe discards even at cost of shanten.

Files to add: `bot/danger.cpp` (or inline in `bot.cpp`), updated `computeDiscard()`.

### v0.3 — Supervised learning baseline (2–4 weeks)

**Goal:** Replace the heuristic discard decision with a neural network trained on game logs from strong AIs.

Architecture:
```
Input features (per position):
  - My hand: one-hot over 34 tile types × 4 counts = 136 bits
  - My packs: 4 pack slots × (type, tile) = sparse encoding
  - Discards by each player: 34 × 4 = 136 bits per player (3 others)
  - Shown tile counts: 34 × 4 int = 136 values
  - Seat wind, prevalent wind, turn number
  Total: ~600 features

Output (policy head):
  - Logit for each of the 34 tile types (softmax over legal discards)

Output (value head):
  - Expected duplicate ranking score [-1, +1]
```

Training:
```
Data source: "game datasets from strong AIs" linked from 2026 contest page.
             Also 2020 human player dataset (~530k games).
Labels: winner's discard choices (behavior cloning / imitation learning).
Loss: cross-entropy on discards + optional value regression.
Augmentation: suit permutation (W↔B↔T, 6 ways) × tile reflection (1↔9, 2 ways) = 12×.
```

Estimated effort: 1 GPU × 1–3 days training; 1 week data pipeline.

Network size: ResNet-18 or a lightweight ~10-block variant fits easily in Botzone's 256 MB RAM and runs in < 200 ms per move with no GPU.

Files to add:
```
train/
  preprocess_logs.py   # Botzone log → (state, action) pairs
  dataset.py
  model.py             # ResNet policy+value
  train_bc.py          # behavior cloning
policy/
  infer.py             # numpy-only inference (no PyTorch at serve time)
  checkpoints/
```

The trained model weights are exported as a numpy `.npz` file (< 50 MB) and loaded at runtime. They go into Botzone Storage, not the zip.

### v0.4 — PPO self-play (4–8 weeks)

**Goal:** Push well past imitation into superhuman territory.

The key insight from the IJCAI 2020 champion and runner-up: SL is not enough because human players are not optimal. Self-play breaks the ceiling.

Setup:
```
Distributed rollout:
  N CPU workers each run a local game environment (4 bots, same policy checkpoint)
  Workers send (state, action, reward) batches to a learner GPU

Learner (PPO):
  γ = 0.997, λ (GAE) = 0.95, clip ε = 0.2
  entropy coefficient 0.01–0.02 (encourages exploration)
  8–16 mini-batches per update

Reward shaping:
  Primary: duplicate ranking score from 96-game packet
  Auxiliary: +small reward for reaching tenpai, −penalty for WA/WH
  Normalize rewards to zero mean, unit variance per batch

Curriculum (from ALONG's public report):
  Start with 100 fixed walls, promote to 200, 400, … when win rate vs.
  previous checkpoint exceeds 80%.
```

Files to add:
```
train/
  env.py               # Python game environment wrapping judge
  train_ppo.py         # PPO learner
  rollout_worker.py    # multiprocessing rollout
eval/
  duplicate_eval.py    # as described in §3
  league.py            # maintain opponent pool
```

### v0.5 — League training and risk model (8–16 weeks)

**Goal:** Robustness against diverse opponents; controlled放铳 risk.

- **Opponent pool**: mix of current checkpoint, past checkpoints, the SL model, and rule-based bots. This prevents "mode collapse" where the bot only knows how to beat itself.
- **Risk estimator**: a small MLP trained on (discard, known game state) → probability of dealing into a win. Used as a veto on dangerous discards even when they'd be good for shanten.
- **Selective MCTS**: for the last 3–5 tiles before tenpai, run a shallow search (50–200 rollouts) to evaluate competing tenpai shapes.

---

## 6. Summary Table

| Version | Key addition | Effort | Expected tier |
|---|---|---|---|
| v0.1 (done) | Shanten discard + safe HU | 1 day | Sample bot + |
| v0.2 | Danger-aware discard | 1–2 weeks | Mid-table |
| v0.3 | Supervised learning | 2–4 weeks | Top-30 |
| v0.4 | PPO self-play | 4–8 weeks | Top-10 |
| v0.5 | League + risk model | 8–16 weeks | Champion contender |

The biggest single jump is **v0.3 → v0.4**: supervised learning reaches diminishing returns fairly quickly because human gameplay has systematic biases. Self-play with a good reward signal (duplicate ranking score, not win/loss) is where the real ceiling lies.

The biggest engineering trap is **measuring the wrong thing**: if you optimize for win-rate in random games, you will overfit to aggressive play that scores high variance but loses badly in the duplicate format. Always evaluate with the `duplicate_eval.py` packet.
