#!/usr/bin/env python3
"""E2 chess BC: fixed eval ladder = nodes-limited Stockfish levels.

A policy is a callable (chess.Board) -> chess.Move (must return a legal move).
For each ladder level (Stockfish, Threads=1 Hash=16, Limit(nodes=N)) we play
n_games alternating colors; draws by chess rules, plus a max-ply adjudication
draw. Reports per-level score in [0,1] and the naive Elo difference
  elo_diff = -400*log10(1/s - 1)  (clamped to +-1000 at s in {0,1}).

The ladder is INTERNAL and fixed: absolute Elo anchoring is not attempted here;
band-to-band comparisons only need the fixed ladder. Smoke test:
  python elo_ladder.py --policy random --games 6 --levels 1,64,1024
"""
import argparse, json, math, random, sys, time
import chess, chess.engine

STOCKFISH = "/usr/games/stockfish"
DEFAULT_LEVELS = [1, 16, 64, 256, 1024, 4096]
MAX_PLIES = 300


class RandomPolicy:
    name = "random"
    def __call__(self, board):
        return random.choice(list(board.legal_moves))


def play_game(policy, engine, nodes, policy_is_white, seed=None):
    board = chess.Board()
    limit = chess.engine.Limit(nodes=nodes)
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        if board.turn == (chess.WHITE if policy_is_white else chess.BLACK):
            mv = policy(board)
            assert mv in board.legal_moves, f"policy returned illegal move {mv} in {board.fen()}"
        else:
            mv = engine.play(board, limit).move
        board.push(mv)
    if board.ply() >= MAX_PLIES and not board.is_game_over(claim_draw=True):
        return 0.5  # adjudicated draw
    r = board.result(claim_draw=True)
    if r == "1-0":
        return 1.0 if policy_is_white else 0.0
    if r == "0-1":
        return 0.0 if policy_is_white else 1.0
    return 0.5


def elo_diff(score):
    s = min(max(score, 1e-3), 1 - 1e-3)
    return round(-400.0 * math.log10(1.0 / s - 1.0), 1)


def run_ladder(policy, levels=DEFAULT_LEVELS, n_games=20, verbose=True):
    out = {"policy": getattr(policy, "name", "unnamed"), "n_games_per_level": n_games,
           "max_plies": MAX_PLIES, "levels": {}}
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    engine.configure({"Threads": 1, "Hash": 16})
    try:
        for nodes in levels:
            t0 = time.time()
            scores = [play_game(policy, engine, nodes, policy_is_white=(i % 2 == 0))
                      for i in range(n_games)]
            s = sum(scores) / len(scores)
            out["levels"][str(nodes)] = {
                "score": round(s, 4), "elo_diff_vs_level": elo_diff(s),
                "w_d_l": [sum(x == 1 for x in scores), sum(x == 0.5 for x in scores),
                          sum(x == 0 for x in scores)],
                "sec": round(time.time() - t0, 1)}
            if verbose:
                print(f"nodes={nodes}: {out['levels'][str(nodes)]}", flush=True)
    finally:
        engine.quit()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="random", choices=["random"])
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--levels", default=",".join(map(str, DEFAULT_LEVELS)))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    levels = [int(x) for x in a.levels.split(",")]
    res = run_ladder(RandomPolicy(), levels, a.games)
    js = json.dumps(res, indent=2)
    print(js)
    if a.out:
        with open(a.out, "w") as f:
            f.write(js)


if __name__ == "__main__":
    main()
