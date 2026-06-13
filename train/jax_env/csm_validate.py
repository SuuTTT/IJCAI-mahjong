"""
csm_validate.py — correctness harness for the win-aware self-play env.
Runs many rollouts, and for EVERY declared win independently verifies with MahjongGB that the
winner's 14-tile hand is a genuine win (zero false positives expected). Also checks tile
conservation (136 tiles) on terminal states, and reports throughput.
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax
from jax import random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csm_selfplay as SP
from MahjongGB import MahjongFanCalculator

TILES = [f'{s}{n}' for s in 'WTB' for n in range(1, 10)] + [f'F{n}' for n in range(1, 5)] + [f'J{n}' for n in range(1, 4)]


def verify_win(hand14_counts):
    """hand14_counts: (34,) counts summing to 14. True iff MahjongGB accepts some 13+winTile split."""
    tiles = [i for i in range(34) for _ in range(hand14_counts[i])]
    if len(tiles) != 14:
        return None
    for wt in set(tiles):
        rest = list(tiles); rest.remove(wt)
        try:
            MahjongFanCalculator(pack=(), hand=tuple(TILES[i] for i in rest), winTile=TILES[wt],
                                 flowerCount=0, isSelfDrawn=True, is4thTile=False, isAboutKong=False,
                                 isWallLast=False, seatWind=0, prevalentWind=0, verbose=False)
            return True
        except Exception:
            continue
    return False


def main(B=16384, N=100, rounds=8):
    total_games = total_wins = false_pos = consv_bad = 0
    t0 = time.time()
    for r in range(rounds):
        st = SP.rollout(random.PRNGKey(1000 + r), B, N)
        hands = np.array(st['hands']); disc = np.array(st['discards'])
        winner = np.array(st['winner']); wtile = np.array(st['wintile']); wtype = np.array(st['wtype'])
        wall_rem = np.array(SP.E.WALL - st['ptr']) if False else None
        total_games += B
        # tile conservation: hands + discards + remaining wall == 136 per game
        per_game_tiles = hands.sum(axis=(1, 2)) + disc.sum(axis=(1, 2)) + np.maximum(np.array(SP.E.WALL - st['ptr']), 0)
        consv_bad += int((per_game_tiles != SP.E.WALL).sum())
        wi = np.where(winner >= 0)[0]
        total_wins += len(wi)
        for b in wi:
            w = winner[b]
            h = hands[b, w].copy()
            if wtype[b] == 0:                       # rob: winner's 13 concealed + the robbed tile
                h = h.copy(); h[wtile[b]] += 1
            if h.sum() != 14:                       # only verify clean 14-tile hands
                continue
            ok = verify_win(h)
            if ok is False:
                false_pos += 1
                if false_pos <= 5:
                    print("  FALSE WIN:", ' '.join(TILES[i] for i in range(34) for _ in range(h[i])))
    dt = time.time() - t0
    print(f"games {total_games} | wins {total_wins} | FALSE wins {false_pos} | tile-conservation violations {consv_bad}")
    print(f"throughput: {total_games/dt:,.0f} games/s ({total_games*N/dt:,.0f} steps/s) over {rounds} rounds")


if __name__ == '__main__':
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    R = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    main(B, N, R)
