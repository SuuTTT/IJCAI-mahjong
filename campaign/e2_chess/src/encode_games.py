#!/usr/bin/env python3
"""E2 chess BC: encode filtered band PGNs into sharded npz.

Per shard (npz, compressed):
  obs      (M,18,8,8) uint8   board planes (see chess_enc.py)
  action   (M,)       int32   AlphaZero 4672 index of the move actually played
  game_idx (M,)       int32   index into the per-shard game tables below
  mover_elo(M,)       int16   Elo of the player who made the move
  game_id  (G,)       <U16    lichess game id (from Site header)
  white_elo(G,) black_elo(G,) int16
  result   (G,)       int8    1 white win / 0 draw / -1 black win

Every replayed move is asserted legal (python-chess); illegal/broken games are
counted and skipped. Parallel over worker processes (chunks of raw game text).

Usage: encode_games.py --pgn F --out-dir D [--max-games N] [--shard-size 2500]
                       [--workers 12] [--min-elo E] (extra band gate, for elite pgn)
"""
import argparse, io, json, os, sys, time
import multiprocessing as mp
import numpy as np
import chess, chess.pgn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chess_enc import encode_board, move_to_index

RESULT_MAP = {"1-0": 1, "1/2-1/2": 0, "0-1": -1}


def iter_raw_games(path, max_games=None):
    buf, n = [], 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("[Event ") and buf:
                yield "".join(buf); buf = []
                n += 1
                if max_games and n >= max_games:
                    return
            buf.append(line)
    if buf:
        yield "".join(buf)


def encode_chunk(args):
    texts, min_elo = args
    obs, act, gidx, melo = [], [], [], []
    gid, welo, belo, res = [], [], [], []
    stats = {"games_ok": 0, "games_bad": 0, "moves": 0, "illegal": 0}
    for text in texts:
        try:
            game = chess.pgn.read_game(io.StringIO(text))
            if game is None or game.errors:
                stats["games_bad"] += 1; stats["illegal"] += len(game.errors) if game else 0
                continue
            h = game.headers
            we, be = int(h["WhiteElo"]), int(h["BlackElo"])
            if min_elo and min(we, be) < min_elo:
                continue
            r = RESULT_MAP.get(h.get("Result", "*"))
            if r is None:
                stats["games_bad"] += 1; continue
            g = len(gid)
            board = game.board()
            n0 = len(act)
            ok = True
            for mv in game.mainline_moves():
                if mv not in board.legal_moves:   # explicit legality check
                    stats["illegal"] += 1; ok = False; break
                obs.append(encode_board(board))
                act.append(move_to_index(mv))
                gidx.append(g)
                melo.append(we if board.turn == chess.WHITE else be)
                board.push(mv)
            if not ok:
                del obs[n0:], act[n0:], gidx[n0:], melo[n0:]
                stats["games_bad"] += 1; continue
            url = h.get("Site", "")
            if "lichess.org" not in url:
                url = h.get("LichessURL", url)  # elite-db uses LichessURL
            gid.append(url.rsplit("/", 1)[-1][:16])
            welo.append(we); belo.append(be); res.append(r)
            stats["games_ok"] += 1
            stats["moves"] += len(act) - n0
        except Exception:
            stats["games_bad"] += 1
    if not act:
        return None, stats
    pack = dict(
        obs=np.asarray(obs, dtype=np.uint8),
        action=np.asarray(act, dtype=np.int32),
        game_idx=np.asarray(gidx, dtype=np.int32),
        mover_elo=np.asarray(melo, dtype=np.int16),
        game_id=np.asarray(gid),
        white_elo=np.asarray(welo, dtype=np.int16),
        black_elo=np.asarray(belo, dtype=np.int16),
        result=np.asarray(res, dtype=np.int8),
    )
    return pack, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--shard-size", type=int, default=2500, help="games per shard")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min-elo", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    total = {"games_ok": 0, "games_bad": 0, "moves": 0, "illegal": 0}
    shard_i = 0

    def chunks():
        buf = []
        for g in iter_raw_games(a.pgn, a.max_games):
            buf.append(g)
            if len(buf) >= a.shard_size:
                yield (buf, a.min_elo); buf = []
        if buf:
            yield (buf, a.min_elo)

    with mp.Pool(a.workers) as pool:
        for pack, stats in pool.imap(encode_chunk, chunks()):
            for k in total:
                total[k] += stats[k]
            if pack is not None:
                out = os.path.join(a.out_dir, f"shard_{shard_i:04d}.npz")
                np.savez_compressed(out, **pack)
                shard_i += 1
            el = time.time() - t0
            print(json.dumps({**total, "shards": shard_i, "elapsed_s": round(el, 1),
                              "moves_per_s": round(total["moves"] / max(el, 1e-9), 1)}),
                  flush=True)
    with open(os.path.join(a.out_dir, "encode_stats.json"), "w") as f:
        json.dump({**total, "shards": shard_i, "elapsed_s": round(time.time() - t0, 1),
                   "pgn": a.pgn, "shard_size": a.shard_size}, f, indent=2)


if __name__ == "__main__":
    main()
