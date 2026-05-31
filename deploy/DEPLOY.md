# Deployment Guide

Two ways to get a bot into Botzone Simulation-7. **Path A (LocalAI) is recommended
for the ML bot**; Path B is the dependency-free C++ fallback.

---

## Path A — ML bot via LocalAI (recommended)

The trained ML bot (`bot/ml_bot.py`) runs **on this machine** (Ubuntu 24.04, GPU,
working MahjongGB + numpy). The LocalAI adapter polls your Botzone endpoint over
HTTP and relays actions. This sidesteps Botzone's Ubuntu-16.04 environment entirely
— no `.so`/glibc/numpy-version risk. It is the official LLM-track deployment path.

### One-time setup
1. Log in to Botzone, create a bot for **Chinese-Standard-Mahjong** (this bot is just
   a router; the real logic runs here). Enable **LocalAI** on it.
2. Copy its LocalAI URL: `https://www.botzone.org.cn/api/<UID>/<SECRET>/localai`

### Run
```bash
cd /home/coder/IJCAI-mahjong
LOCALAI_URL="https://www.botzone.org.cn/api/<UID>/<SECRET>/localai" \
    bash deploy/run_localai.sh
```
Leave it running, then join **Simulation-7** (or a manual match) on Botzone with that
bot. Each game request is routed here, ml_bot decides, the action is sent back.

- Model is selectable: `MODEL=train/checkpoints/bc_v2_weights.npz bash deploy/run_localai.sh`
  - `bc_v3_ft_weights.npz` (default) — all-players data, balanced offense+defense
  - `bc_v2_weights.npz` — winner-only, more aggressive
- Concurrency: the adapter starts one persistent bot process per live match.
- Stop with Ctrl-C.

### Why this is safe
- Verified **0 illegal moves in 200 judge-run games** (`eval/ml_eval.py`).
- All legality decisions (HU fan≥8, PENG/CHI/GANG, wall-end) are validated against the
  feature-agent state plus emit-time physical re-checks (`verify_draw`/`verify_claim`).
- `OPENBLAS_NUM_THREADS=1` matches Botzone's single core and keeps inference ~1.3 ms.

---

## Path B — C++ heuristic bot (dependency-free direct upload)

`bot/bot_submit.cpp` is a single self-contained file (fan calculator + shanten +
danger-aware discard embedded). No external dependencies — compiles and runs on
Botzone's environment directly.

### Upload
1. Botzone → My Bots → new **Chinese-Standard-Mahjong** C/C++ bot.
2. Paste the contents of `bot/bot_submit.cpp`. Submit.

Use this as a stable baseline / fallback. It is the heuristic line (no ML model).
Note: the ML legality fixes (CHI offer, wall-count) discovered this session are NOT
yet ported to the C++ bot — prefer Path A for competition.

---

## Validation before deploying
```bash
cd /home/coder/IJCAI-mahjong
OPENBLAS_NUM_THREADS=1 python3 tests/test_legality_judge.py 60   # expect: 0 illegal
OPENBLAS_NUM_THREADS=1 python3 eval/ml_eval.py train/checkpoints/bc_v3_ft_weights.npz 100
```

## Post-deploy (the only validation that matters)
After Simulation-7, pull the match list for the bot and re-run the same analysis as
`doc/analysis`: filter `[-30,10,10,10]` failure-compensation and all-zero games, count
true wins (score > +10). Compare the corrected true-win rate against the deployed
heuristic's 5.49% baseline. Feed losses back into training (especially deal-ins → defense).
