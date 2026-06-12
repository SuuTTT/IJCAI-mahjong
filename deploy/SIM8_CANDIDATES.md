# Sim-8 candidate bots (A/B on your own practice tables)

Three bots to compare. Botzone pattern: upload the small CODE zip as bot source; upload model
.pkl files to Botzone Storage under `data/<name>`. The bot auto-loads from `data/`.

## 1. PLAIN (the verified ship, baseline)
- Code: `deploy/ship/bot_lad_chunjiandu.zip` (md5 ad016476)
- Storage: `data/cnn_lad_chunjiandu.pkl` (md5 d517e6a9)
- model.cfg -> cnn_lad_chunjiandu.pkl

## 2. DISTILL100B (SL floor, alt baseline)
- Code: `deploy/ship/bot_distill100b.zip` (md5 db86fd5b)
- Storage: `data/cnn_distill100b.pkl` (md5 7e45c413)

## 3. NET-PIMC (the new search bot — TEST THIS)
- Code: `deploy/ship/bot_netpimc.zip` (33KB)
- Storage (THREE files):
  - `data/cnn.pkl`   <- copy of cnn_lad_chunjiandu.pkl (main policy)
  - `data/fast8.pkl` <- deploy/ship/fast8.pkl (rollout policy, 8-block)
  - `data/vbig.pkl`  <- deploy/ship/vbig.pkl  (value leaf, r=0.892)
- Auto-enables opponent-aware net-rollout search when fast8.pkl + vbig.pkl are present.
  Anytime, ~4s/turn, hard-capped under Botzone's 6s (never TLEs). Falls back to plain policy
  on any error (search can't break the bot).

## How to read sim-8
Seat one of each against the same opponents; net-PIMC wins iff it beats PLAIN by a margin that
clears noise (don't trust <a few hundred net over a handful of games). My held-out pre-screen
(720 games/arm) lands overnight — check before trusting sim-8's small sample.
