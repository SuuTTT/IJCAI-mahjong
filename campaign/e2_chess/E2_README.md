# E2 — Chess BC with rating-stratified noise: PREP artifacts

Experiment E2 of `PLAN_domain_extension.md`: distill-then-ensemble noise-threshold
claim with naturally-occurring noise = skill of the imitated player. BC per rating
band; prediction: student-ens minus teacher-ens gap grows as source rating drops.
Eval = fixed ladder of nodes-limited Stockfish. This directory contains the PREP
outputs only (data + encoding + eval harness); the training grid is a later decision.

## Layout

```
/root/e2_chess/
  src/            all scripts (filter_stream.py, filter_elite.py, chess_enc.py,
                  encode_games.py, enc_selftest.py, elo_ladder.py, throughput_probe.py,
                  elite_backfill.sh)
  data/           filtered PGNs per band (see below)
  enc/            sharded npz samples (10k games per band encoded in prep)
  logs/           status JSONs + job logs
  venv/           python 3.12 venv: python-chess 1.11.2, numpy 2.5.1, torch 2.13.0+cu130
  E2_PREP.json    machine-readable prep summary
```

## Data sources & filters

Bands: **0800–1200**, **1600–2000**, **2400+** (both players' Elo inside the band).

1. `data/band_0800-1200.pgn`, `data/band_1600-2000.pgn` — streamed from
   `https://database.lichess.org/standard/lichess_db_standard_rated_2026-06.pgn.zst`
   (curl | zstd -dc | `src/filter_stream.py`, never hits disk uncompressed).
   Kept iff: Event contains "Rated Classical" or "Rated Rapid" (game or tournament),
   Termination == "Normal" (drops time-forfeit/abandoned/cheat-flag), both Elos in
   band, movetext has `%clk` clock comments, game reached move 6 (>= ~11 plies).
   Quota 210k games/band; both low bands filled from the first ~15% of the month.
2. `data/band_2400plus.pgn` — same stream/filters; a full month only yields ~20k
   games with BOTH players 2400+ in rapid/classical, so this file is the
   with-clocks complement only.
3. `data/band_2400plus_elite.pgn` — main 2400+ source: **Lichess Elite Database**
   (`https://database.nikonoel.fr/lichess_elite_YYYY-MM.zip`, months backfilled
   from 2025-11 downward until quota; see `logs/elite_backfill_status.json`).
   Same filters, EXCEPT no `%clk` requirement (elite db strips clock comments);
   blitz games in the elite db are dropped (rapid/classical only, keeping time
   control consistent across bands). Elite raw PGNs are deleted after filtering
   (re-downloadable). Header quirk: elite uses `[LichessURL]` instead of `[Site]`.

**Caveats to document in the paper:** 2400+ band spans 2024–2025 months (elite)
plus 2026-06 (with-clocks month slice) while low bands are 2026-06 only; elite
band games have no clock comments. Neither matters for BC on (board, move) pairs,
but note it.

## Encoding (`src/chess_enc.py`, `src/encode_games.py`)

OBS: `(18, 8, 8)` uint8, **absolute orientation** (no board flipping;
plane 12 = side-to-move):
- 0–5 white P,N,B,R,Q,K; 6–11 black P,N,B,R,Q,K (`x[c, rank, file]`, rank 0 = white's back rank)
- 12 side-to-move (all-ones = white); 13–16 castling rights WK,WQ,BK,BQ (all-ones planes)
- 17 en-passant target square one-hot

ACTION: **AlphaZero-style 4672** = `from_square(64) * 73 + move_plane(73)`,
absolute coords: planes 0–55 queen-slides (8 dirs x 7 distances, N,NE,…,NW),
56–63 knight moves, 64–72 underpromotions ((df+1)*3 + {N,B,R}); queen promotions
are encoded as plain pawn slides. `index_to_move()` inverts (adds queen promotion
on last-rank pawn slides); `legal_action_mask()` provided for masked decoding.
`src/enc_selftest.py` round-trips every legal move of 200 random games
(collision-free, exact inverse) — run it after any change.

Shard format (npz, compressed, `--shard-size 2500` games/shard):
- per position: `obs (M,18,8,8) u8`, `action (M,) i32`, `game_idx (M,) i32`,
  `mover_elo (M,) i16` (Elo of the player who made the move)
- per game: `game_id (G,) str`, `white_elo/black_elo (G,) i16`,
  `result (G,) i8` (1/0/-1 = white/draw/black)

Every replayed move is explicitly asserted legal via python-chess; games with
parse errors/illegal SAN are counted (`encode_stats.json`) and skipped whole.

## Eval ladder (`src/elo_ladder.py`)

Fixed ladder = Stockfish 16 (`/usr/games/stockfish`, apt), `Threads=1 Hash=16`,
levels = node limits `{1, 16, 64, 256, 1024, 4096}`. Policy = callable
`(chess.Board) -> legal chess.Move`. N games/level, alternating colors, draws by
rule + 300-ply adjudication; reports score and naive Elo diff
`-400*log10(1/s-1)`. The ladder is internal/fixed — band-vs-band deltas need no
absolute anchor; absolute Elo anchoring (e.g. vs maia/lichess bots) is optional
later. Smoke-verified: random policy scores 0.0 at all levels (~0.05 s/game).

## Throughput (`src/throughput_probe.py`)

Loads npz shards and times a dummy policy ResNet (18->ch, N residual blocks,
4672-way head) fwd+bwd and inference on a free GPU — numbers in `E2_PREP.json`
(`throughput`), used to size the training grid (K teachers x 3 bands x seeds).

## Not done yet (deliberately)

- Full-band encoding (only 10k-game samples per band encoded; run
  `encode_games.py` without `--max-games` per band when training is green-lit;
  keep `--workers <= 20` while the box is shared).
- Training grid, teacher/student/ensemble code, Elo anchoring — later decision.
