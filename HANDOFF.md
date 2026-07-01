# HANDOFF — IJCAI-mahjong (code + competition)
**Timestamp: 2026-07-01.** Credentials/box-access are in a **separate secure handoff** (not in this public repo — ask the owner for `HANDOFF_MAHJONG_2026-07-01.md`).

## What this repo is
All code for the IJCAI-2026 Chinese Standard Mahjong (MCR) bot "moyu" + experiments + blog. The live experiment code + models are on a **4×3090 GPU box** (see secure handoff for SSH), primarily under `train/caiest_repro/` (mirrored on the box at `/root/IJCAI-mahjong/train/caiest_repro/`). Botzone deploy machinery is in `/root/realfield_build/` on the box.

## Competition status (as of 2026-07-01)
- Bot was **11th/19 in sim10** with the entry `distill` (net `cnn_lad_chunjiandu`, 128×40) — which turned out to be **under-trained**.
- **THE WIN:** a from-scratch, fully-converged same-size net **`bn128s1`** (`ckpt/e1b/full_128x40_s1.pkl`) **CI-beats `distill` by +0.0185** (22-block calibrated gate, CI [2.510, 2.527]) and is **TLE-safe**. It is now **deployed to Sim-11**.
- **Schedule: Sim-11 = 2026-07-04, Final = 2026-07-07** (top-16, Swiss + duplicate format).
- Final-0 still runs `distill` (safe fallback). **Action at sim11: if bn128s1 beats 11th, dispatch it to Final-0 too.**

## The campaign verdict (what to believe / not repeat)
- moyu **over-claims** chi/peng vs experts — real but **performance-irrelevant** (all corrections null).
- **NULL levers (do not re-run expecting a win):** claim-suppression (τ), RL/AWR (even with a verified-good 0.955 value critic), value-guided play (both risk directions), PIMC/MCTS, cloning top players, ensembling, capacity beyond ~128ch, safe-discard.
- **What worked = training convergence.** distill was under-trained; convergence is the fix; capacity saturates ~128 channels (bigger nets beat it only marginally and risk TLE).
- **Running now (win lever):** enhanced-recipe 128×40 (`e11_train.py`: verified game-symmetry augmentation suit×rank-reflect×dragon ≈72×, + label-smoothing + EMA, 130k steps) and TTA (`e11_gate.py`). Deliverables `AUG_RESULTS.json`/`AUG_WRITEUP.md`. If any CI-beats bn128s1 + TLE-clean → deploy it before the final.

## Evaluation discipline (CRITICAL — this project has over-claimed ~10×)
- Use the **calibrated duplicate-format gate** `e8_gate.py --lam 0`: X-vs-X MUST return **2.500**.
- **Deploy/claim ONLY on multi-block, CI-separated results** (95% CI lower bound > 2.500). Point estimates lie (the retracted "3.06" and "2.5314" were small-sample flukes).
- Read every number from a saved JSON, not from logs/impressions.

## Key result files (box `train/caiest_repro/`)
`E1_RESULTS.json` (over-claim generality sweep), `E1B_*` (convergence), `E3_RESULTS.json` (value-model scaling, 0.955 AUC), `E4_RESULTS.json` (RL-null), `E6_RESULTS.json` (correction double-null), `BN128S1_CONFIRM.json` (the win), `BN384_CONFIRM.json` (bigger-net confirm), `AUG_*` (running). Real-field: `/root/realfield_build/N200_RESULTS.json`, `UPLOAD_LOG.md`.

## Gotchas
- Botzone: stored cookies = wrong accounts; **email-login as moyu** (secure handoff).
- Long jobs need **`setsid`** or they die on SSH close.
- Box disk ~7 GB free — use `savez_compressed`, guard `df -h`.
- Bigger nets risk TLE (Botzone ~1 s/move; 128×40 = 28 ms, safe).

## Full history
The dense chronological log of every experiment/decision is in the owner's Claude Code memory (`ijcai-mahjong-state.md`) — ask the owner. The two papers have their own handoffs (ToG in `tog-mahjong-paper`; JMLR results on box `/root/jmlr*`).
