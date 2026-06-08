"""
oracle_extract.py (P3-RL / Suphx oracle-guiding, foundation) — extract ORACLE features from the
official data.txt: the normal 38-plane acting-seat obs PLUS the 3 opponents' TRUE concealed hands as
count planes (the full log is omniscient). Oracle obs = (38+12, 4, 9): 38 normal + 3 opp x 4 count.
The oracle teacher (sees everything) is trained on these, then distilled to the 38-plane student
(public info) via KL — the Suphx "oracle guiding" scaffold (hidden info is a TRAIN-time crutch only).

  python3 oracle_extract.py --data data/data.txt --out data/oracle_cooked.npz --limit-matches 20000
Writes obs int8 (N,50,4,9), mask bool (N,235), act int16 (N,).  (--limit-matches caps for a subset.)
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from feature import FeatureAgent

TILES = FeatureAgent.TILE_LIST          # 34 tiles
TIDX = {t: i for i, t in enumerate(TILES)}

def opp_planes(true_hands, seat):
    """3 opponents' concealed hands -> (12,4,9): for each opp, 4 count-planes (>=1,>=2,>=3,>=4)."""
    pl = np.zeros((12, 34), np.int8)
    k = 0
    for d in (1, 2, 3):
        o = (seat + d) % 4
        cnt = np.zeros(34, np.int8)
        for t in true_hands[o]:
            if t in TIDX: cnt[TIDX[t]] += 1
        for lvl in range(4):
            pl[k] = (cnt > lvl).astype(np.int8); k += 1
    return pl.reshape(12, 4, 9) if False else pl  # keep (12,34) -> reshape later with obs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--limit-matches', type=int, default=0)
    a = ap.parse_args()
    O, M, A = [], [], []
    agents = None; hands = None; quan = 0; cur_tile = None; mid = -1
    def flush_decision(seat, obs):
        # obs: dict from FeatureAgent (38,4,9 + mask). Append opp planes -> (50,4,9).
        base = obs['observation'].astype(np.int8)                 # (38,4,9)
        opp = opp_planes(hands, seat).reshape(12, 4, 9) if False else None
        # build (12,4,9) from (12,34): pad 34->36 then reshape
        op34 = opp_planes(hands, seat)                            # (12,34)
        op = np.zeros((12, 36), np.int8); op[:, :34] = op34
        full = np.concatenate([base, op.reshape(12, 4, 9)], axis=0)  # (50,4,9)
        O.append(full); M.append(obs['action_mask'].astype(np.bool_))
    with open(a.data, encoding='UTF-8') as f:
        obs_buf = [[] for _ in range(4)]; act_buf = [[] for _ in range(4)]
        for line in f:
            t = line.split()
            if not t: continue
            if t[0] == 'Match':
                agents = [FeatureAgent(i) for i in range(4)]; hands = [[] for _ in range(4)]
                mid += 1
                if a.limit_matches and mid >= a.limit_matches: break
                if mid % 4000 == 0: print(f"match {mid} samples {len(O)}", flush=True)
            elif t[0] == 'Wind':
                quan = int(t[1]) if len(t) > 1 else 0
                for ag in agents: ag.request2obs(line.strip())
            elif t[0] == 'Player':
                p = int(t[1])
                if t[2] == 'Deal':
                    hands[p] = list(t[3:]); agents[p].request2obs(' '.join(t[2:]))
                elif t[2] == 'Draw':
                    tile = t[3]; hands[p].append(tile)
                    o = agents[p].request2obs('Draw %s' % tile)
                    if int(o['action_mask'].sum()) > 1:
                        obs_buf[p].append((p, o)); act_buf[p].append(None)   # decided on Play below
                    for i in range(4):
                        if i != p: agents[i].request2obs('Player %d Draw' % p)
                elif t[2] == 'Play':
                    tile = t[3]
                    if tile in hands[p]: hands[p].remove(tile)
                    # record the discard decision with oracle obs (taken BEFORE removing from view)
                    if obs_buf[p] and act_buf[p] and act_buf[p][-1] is None:
                        _, o = obs_buf[p][-1]
                        if int(o['action_mask'].sum()) > 1:
                            flush_decision(p, o)
                            A.append(agents[p].OFFSET_ACT['Play'] + agents[p].OFFSET_TILE[tile])
                        act_buf[p][-1] = tile
                    for i in range(4): agents[i].request2obs('Player %d Play %s' % (p, tile))
                    cur_tile = tile
                elif t[2] == 'Chi':
                    midt = t[3]
                    # consume from hand: the two non-mid tiles of the chi set (approx — skip exact)
                    for i in range(4): agents[i].request2obs('Player %d Chi %s' % (p, midt))
                elif t[2] == 'Peng':
                    for _ in range(2):
                        if cur_tile in hands[p]: hands[p].remove(cur_tile)
                    for i in range(4): agents[i].request2obs('Player %d Peng' % p)
                elif t[2] in ('Gang', 'AnGang', 'BuGang'):
                    for i in range(4): agents[i].request2obs('Player %d Gang %s' % (p, t[3] if len(t) > 3 else ''))
            elif t[0] == 'Score':
                for x in obs_buf: x.clear()
                for x in act_buf: x.clear()
    if not O:
        print("no samples"); return
    obs = np.stack(O).astype(np.int8); mask = np.stack(M).astype(np.bool_); act = np.array(A, np.int16)
    n = min(len(obs), len(act))
    np.savez_compressed(a.out, obs=obs[:n], mask=mask[:n], act=act[:n])
    print(f"oracle: {n} samples, obs {obs[:n].shape} -> {a.out}", flush=True)

if __name__ == '__main__':
    main()
