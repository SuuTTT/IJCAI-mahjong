# How to Submit the Current (ML) Version to Botzone

Botzone gives you two upload slots: **source code** and a **Storage `data/` folder**
(≤ 268 MB, mounted at `./data/` when your bot runs). The ML bot uses both.

## Files
- `deploy/mahjong_ml_bot.zip` — the bot source (Python, `__main__.py` at root). Rebuild with `bash deploy/build_zip.sh`.
- `train/checkpoints/bc_v3_ft_weights.npz` — the 14 MB model → goes into Storage.

## Steps
1. **Create the bot**: Botzone → My Bots → new bot, game **Chinese-Standard-Mahjong**, language **Python 3**.
2. **Upload code**: upload `deploy/mahjong_ml_bot.zip` as the source (it has `__main__.py` at the root, as Botzone requires for multi-file Python).
3. **Upload the model to Storage**: in the 用户存储空间 page, upload `bc_v3_ft_weights.npz` so its path is `data/bc_v3_ft_weights.npz`. (14 MB, well under the 256 MB limit.)
4. **Smoke test**: run a manual match / debug on Botzone. Watch the log:
   - Bot plays `PLAY`/`PENG`/`HU` and occasionally wins → **MahjongGB is available, full strength.**
   - Bot only ever `PLAY`/`PASS`, never `HU` → MahjongGB is **missing** on Botzone (see below).
5. **Enter Simulation-7** with the bot.

## The one dependency to confirm: `MahjongGB`
The bot needs `numpy` (almost certainly present) and `MahjongGB` (the official fan
library) to decide HU. The zip **degrades gracefully**: if `MahjongGB` is missing it
still runs **legally** (no crash, no illegal move) but can never declare a win.

To check definitively, upload `deploy/probe_bot.py` as a throwaway Python bot and run
one match — its first response encodes whether the imports succeeded (see that file).

### If MahjongGB is NOT on Botzone
Then a pure-Python ML bot can't win there. Fallback options, in order:
1. **C++ submission** (`bot/bot_submit.cpp`) — self-contained heuristic, fan calculator
   embedded, zero dependencies. Works today. (Weaker: this is the line that scored 5.49%.)
2. **Port the MLP to C++** (next dev task): embed the fan calc (already in bot_submit.cpp)
   + read `data/bc_v3_ft_weights.npz` + do the 240-dim features and matmuls in C++. This
   gives a dependency-free ML bot — the proper competition deployment.

## Validation already done locally
- 0 illegal moves in 200 + 60 judge-run games (`tests/test_legality_judge.py`).
- Runs correctly from a simulated Botzone layout (`python __main__.py`, weights in `./data/`).
