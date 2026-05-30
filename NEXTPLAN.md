# Next-Step Plan — Grounded in Official Resources

Sources examined:
- https://botzone.org.cn/static/gamecontest2026a_cn.html  (contest page)
- https://github.com/ailab-pku/Mahjong-LLM              (LocalAI adapter + feature agent)
- https://github.com/ailab-pku/PyMahjongGB              (fan/shanten library)
- https://disk.pku.edu.cn/link/AA8CB7A57AFDCD48CAA7C749E04B5B6FAA (dataset)
- Official judge `judge/main.cpp` already at `/workspace/Chinese-Standard-Mahjong/judge/`

---

## What the Official Resources Actually Give Us

### 1. Strong AI Game Dataset  (`disk.pku.edu.cn` link)
This is the single most important resource for v0.3 supervised learning.
We do not yet know the exact format until we download it, but from historical
contest structure it will be Botzone-format JSON game logs.

**Action needed:** Download the dataset and inspect one file to confirm format.

```bash
# Once downloaded, inspect a sample log:
python3 -c "
import json
with open('sample_game.json') as f:
    g = json.load(f)
print(list(g.keys()))
print(g['initdata'])
print(g['log'][0])
"
```

Expected format (from Botzone log structure): a list of `(output, input)` pairs
where `output` is the judge's broadcast and `input` is each player's response.

---

### 2. Mahjong-LLM Repo — Three Directly Usable Components

#### 2a. `local_ai/local_ai.py` — Live Botzone Testing Without Uploading
This is the **most important tool we were missing**.  It is an HTTP polling adapter
that lets your bot run on your machine while playing live Botzone matches.

```
Botzone server  <--HTTP poll-->  local_ai.py  <--stdin/stdout-->  your bot process
```

You get a private URL from Botzone: `https://www.botzone.org.cn/api/<uid>/<secret>/localai`
Then run:

```bash
python local_ai/local_ai.py \
  --localai-url "https://www.botzone.org.cn/api/UID/SECRET/localai" \
  --bot-cmd python /home/coder/IJCAI-mahjong/bot/keeprunning_bot.py \
  --bot-cwd /home/coder/IJCAI-mahjong/bot
```

Your bot plays real matches and you see all I/O locally.  This is far better than
uploading blind — you can print debug info to stderr and watch every decision.

**Critical requirement:** The bot must use the **Keep Running** protocol, not the
one-shot JSON protocol our current v0.1 uses.  See §Step 1 below.

#### 2b. `local_bots/mahjong/botzone_engine.py` — Clean Protocol Loop
This file (240 lines) wraps the raw `0/1/2/3` request stream into a cleaner
observation API.  Its `run_botzone_loop(agent_cls, decide_fn)` function handles:
- Startup handshake (`"1"`)
- All request types (DRAW, PLAY, GANG, BUGANG, CHI, PENG)
- `>>>BOTZONE_REQUEST_KEEP_RUNNING<<<` sentinel after every response
- Translating Botzone strings into structured `"Draw W5"` / `"Player 2 Play B3"` observations

We should adopt this engine as-is rather than rewriting.  It is well-tested and
already handles edge cases (AnGang vs normal GANG, BUGANG response, etc.).

#### 2c. `local_bots/mahjong/sample.py` — Reference Feature Space
This is the **reference feature engineering implementation** from the contest
organizers.  It contains `FeatureAgent2Adapted` which defines:

```
Normal obs: 240 dims
  quan(1) + men(1) + unseen34 + hand14 + wall10
  + for each of 4 players: (history29 + meld4×4)

Oracle obs: 312 dims  (includes other players' hands — training only)
  quan(1) + men(1) + unseen34
  + for each of 4 players: (history29 + meld4×4 + hand14 + wall10)

Action space:
  pass(1) + hu(1) + play(34) + chi(63) + peng(34) + gang(34) + angang(34) + bugang(34)
  = 235 possible actions  (masked to legal subset each turn)
```

This is what the strong bots use as input.  We should use the same feature space
so our supervised model is comparable to the field.

---

### 3. PyMahjongGB — `MahjongShanten` with Pack Support
The full signature (undocumented until now):

```python
MahjongShanten(pack=((type, tile, offer), ...), hand=(tile, ...)) -> int
```

This is better than `RegularShanten` for hands with declared packs because it
correctly accounts for the reduced mentsu count.  We should switch to it in
`mahjong_bot.py`.

---

## Prioritised Step-by-Step Plan

### STEP 1 — Switch to Keep Running Protocol (1 day)
**Why first:** Required for local_ai.py testing, and reduces cold-start overhead
on Botzone by 5–10× (bot stays alive across turns instead of restarting).

**What to build:** `bot/keeprunning_bot.py` — wraps our existing decision logic
inside `botzone_engine.run_botzone_loop`.

```
bot/
  keeprunning_bot.py     # new entry point using botzone_engine
  botzone_engine.py      # copy from Mahjong-LLM (unchanged)
  mahjong_bot.py         # existing logic, expose decide_fn
```

The `decide_fn` signature matches what `botzone_engine` expects:
```python
def decide_fn(_, obs: str) -> str:
    # obs is like "Draw W5" or "Player 2 Play B3"
    # return like "Play B4" or "Hu" or "Pass"
    ...
```

**Test:** Run `local_ai.py` against Botzone; watch debug output in terminal.

---

### STEP 2 — Compile Local Judge + 4-Bot Runner (2 days)
**Why:** This is the only way to measure duplicate-score performance, which is
what the contest actually ranks on.

The official judge is at `/workspace/Chinese-Standard-Mahjong/judge/main.cpp`.
It needs `jsoncpp` and `boost`:

```bash
sudo apt-get install libboost-all-dev libjsoncpp-dev
cd /workspace/Chinese-Standard-Mahjong/judge
g++ -O2 -std=c++14 \
    -I../fan-calculator-usage \
    main.cpp -lboost_system -ljsoncpp -o judge
```

**4-bot runner** (`eval/run_match.py`):

```python
# Takes 4 bot commands + a wall seed, runs one game, returns 4 scores.
# Judge reads from stdin; it receives a JSON "log" of request→response pairs.
# We wrap this: for each round, broadcast judge output to all 4 bots,
# collect their responses, feed back to judge.
```

**Duplicate evaluator** (`eval/duplicate_eval.py`):

```python
# For N wall seeds × 24 seat permutations:
#   - run 4-bot game
#   - accumulate micro-scores per seat
# Convert to 4/3/2/1 ranking points
# Report: mean ranking pt, micro-score, WA rate, TLE rate
```

**Target:** Be able to run `make eval` and get a duplicate score for any
pair of bots within 30 minutes on CPU.

---

### STEP 3 — Download and Parse Strong AI Dataset (3 days)
**Source:** `https://disk.pku.edu.cn/link/AA8CB7A57AFDCD48CAA7C749E04B5B6FAA`

**Pipeline:**

```
data/
  raw/          # downloaded files (Botzone log JSON)
  parsed/       # per-turn (features, action_label) pairs as .npz
  augmented/    # suit-permuted + tile-flipped versions (×12)
  splits/       # train.npz / val.npz / test.npz
```

**Parsing script** (`data/parse_logs.py`):

```python
# For each game log:
#   For each turn where winner made a discard decision:
#     1. Reconstruct game state at that point
#     2. Extract 240-dim feature vector (using FeatureAgent2Adapted from sample.py)
#     3. Record the action taken as label (index in 0–234 action space)
#     4. Record legal action mask (binary 235-dim vector)
#   Write (features, labels, masks) to .npz
```

**Augmentation** (×12 multiplier, free data):
- 6 suit permutations (W/B/T — all 6 orderings are equivalent by symmetry)
- 2 tile reflections (1↔9, 2↔8, 3↔7, 4↔6, 5 stays — sequence structure is symmetric)

**Expected scale:** Historical datasets have ~500k games × ~25 decisions each
= ~12.5M labeled turns before augmentation.  After ×12: ~150M samples.
Training on a fraction of this (2–5M) is enough for a strong SL baseline.

---

### STEP 4 — Supervised Learning Baseline (1 week GPU)
**Architecture** (`train/model.py`):

```python
# Input: 240-dim observation vector (float32)
# Backbone: MLP with residual connections
#   Linear(240 → 512) → LayerNorm → ReLU
#   × 6 residual blocks: Linear(512→512) → LN → ReLU → Linear(512→512) → LN + skip
# Policy head: Linear(512 → 235) → masked softmax (zero illegal actions)
# Value head: Linear(512 → 1) → tanh (normalized duplicate ranking score)
```

Why MLP not CNN: the 240-dim feature already encodes tile counts and positions
in a structured way.  CNNs are better if input is a 2D tile grid; here MLP+residual
is simpler and faster to train.  ResNet-style is fine for v0.3, switch to
transformer for v0.4+ if helpful.

**Training** (`train/train_bc.py`):

```python
optimizer = AdamW(lr=3e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(T_max=50_epochs)
loss = CrossEntropy(logits_masked, label) + 0.1 * MSE(value_head, 0)
# The 0.1*value term is a regularizer until we have actual value targets
batch_size = 2048
epochs = 50  # ~6h on RTX 4090 with 5M samples
```

**Export:** Save policy weights as `numpy .npz` (no PyTorch at serve time).
Load at startup, run forward pass with `numpy` matrix ops (~2ms on CPU for MLP).

**Expected improvement:** SL baseline typically jumps from ~40% win-rate
(vs. random) to ~65–70% win-rate (from 2020 USTC report), and more importantly
reduces 放铳 rate significantly.

---

### STEP 5 — PPO Self-Play (3–4 weeks)
Only begin after Step 4 is verified to beat pure heuristic in duplicate eval.

**Environment** (`train/env.py`):
- Wraps the compiled judge binary
- Provides `reset(wall_seed)` and `step(action)` compatible with gym API
- Returns duplicate-packet reward (not per-game reward)

**Key design choices (from public champion reports):**
- Reward: **duplicate ranking score** (1–4 points per packet), NOT win/loss per game
- GAE λ = 0.95, γ = 0.997 (long-horizon credit assignment)
- Entropy coefficient 0.01–0.02 (prevent mode collapse)
- Curriculum: start with 100 fixed walls, double when win-rate > 80% vs. prev checkpoint
- Opponent pool: 70% current policy, 20% historical checkpoints, 10% SL baseline

**Infrastructure needed:**
- 1 learner GPU process (policy + value update)
- 16–64 CPU rollout workers (each runs 4-bot game with judge subprocess)
- Shared replay buffer (or synchronous actor-learner)

**Estimated time to beat SL baseline:** 1–2 weeks of continuous training.

---

## Summary Timeline

```
Week 1 (NOW):
  Day 1-2:  Step 1 — Keep Running bot + local_ai.py live testing
  Day 3-4:  Step 2 — Compile judge + 4-bot runner + duplicate eval script
  Day 5-7:  Step 3 begins — download dataset, write parser, verify format

Week 2:
  Day 1-3:  Step 3 continues — full pipeline, augmentation, train/val split
  Day 4-7:  Step 4 — SL training run, eval against heuristic bot

Week 3-4:
  Step 5 — PPO warm-start from SL checkpoint
  Ongoing: duplicate eval after each checkpoint, submit best to Botzone

Week 5+:
  Opponent pool, league training, risk model
  Submit to Simulation rounds as they open
```

**Contest deadline: 2026-06-09.** That gives roughly 10 days from now.
Realistic target for contest day: have Step 1+2 solid (no WA/WH, local eval
working), and optionally Step 4 partially trained.

---

## Immediate Actions (Today)

1. **Get LocalAI URL from Botzone** — log in, go to your bot page, find the `/localai`
   endpoint URL.  This unlocks live local testing.

2. **Port bot to Keep Running** — copy `botzone_engine.py` from Mahjong-LLM, wire
   existing `decide_after_draw` / `decide_after_discard` into it.

3. **Download the dataset** — go to `https://disk.pku.edu.cn/link/AA8CB7A57AFDCD48CAA7C749E04B5B6FAA`
   and start the download.  Even just inspecting 10 games tells us the format.

4. **Compile the judge** — one-time setup unlocks all local eval.
