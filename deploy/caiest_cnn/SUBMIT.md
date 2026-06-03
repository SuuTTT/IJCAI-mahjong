# Deploy: CNN agent (Chinese Standard Mahjong) — replaces the MLP r18

This is the reproduced PKU-lineage **CNN** (16-block ResNet, ~10M params), trained on the
official 98k-game dataset. It **crushes our old r18 MLP** through the official judge
(e.g. epoch-3: r18 0–2 vs CNN, +1623/60 games, draws 10%). It is **legal by construction**
(feature.py fan-gates HU via MahjongFanCalculator: HU only offered when fan >= 8).

## FIX LOG (2026-06-02)
First deploy got RE: `EOFError` from caiest's `input()` loop when Botzone closed stdin.
Rewrote `__main__.py` to mirror our proven `ml_bot.run()`: read via `sys.stdin.readline()`,
BREAK on EOF, skip a leading "1" only if present, wrap each request (errors -> PASS).
Validated: reproduces no crash on INIT-then-close; beats r18 18/20, 0 illegal.
WATCH: first crash log showed memory ~278 MB at load; default Botzone limit is 256 MB/turn.
If you get MLE, switch to the smaller `small_8x128` model (arch search is training it).

## Botzone environment (verified from wiki + benchmarks)
- Language: **Python 3.6** (torch 1.4.0 + numpy available; PyMahjongGB present).
- The model is saved with `_use_new_zipfile_serialization=False` so torch 1.4 can load it.
- Memory limit 256 MB/turn — fits (caiest runs this ranked).

## Upload steps
1. **Model → Storage.** Upload `deploy/caiest_cnn/data/cnn.pkl` (~40 MB) via Botzone
   "Manage Storage". It will be accessible at runtime under `data/`.
   (The bot auto-discovers the largest `*.pkl` under `data/`.)
2. **Code → Bot.** Upload `deploy/caiest_cnn_bot.zip` (contains `__main__.py`, `feature.py`,
   `model.py`, `agent.py`) as a **Python 3.6** bot.
3. Run a Debug match. Expect legal play (no -30) and actual wins / claims. If it ever can't
   load the model it would fall back to erroring — check the match log.

## Files
- `__main__.py` — Botzone keep-running entry; loads model from `data/`.
- `feature.py` — (38,4,9) feature encoder + legal-action mask + HU fan-gate.
- `model.py` — CNNModel (16 ResNet blocks).
- `agent.py` — action<->response mapping base.
- `data/cnn.pkl` — trained weights (legacy torch-1.4 format).

## Reproduce / retrain (ours, from public data)
- `train/caiest_repro/preprocess_single.py` → `data/cooked_single.npz` (5.87M samples)
- `train/caiest_repro/train_repro.py --epochs N` → `log/checkpoint_without0/best.pkl`
- Re-save legacy + copy to `deploy/caiest_cnn/data/cnn.pkl`:
  `python3 -c "import torch; torch.save(torch.load('BEST.pkl',map_location='cpu'),'data/cnn.pkl',_use_new_zipfile_serialization=False)"`
