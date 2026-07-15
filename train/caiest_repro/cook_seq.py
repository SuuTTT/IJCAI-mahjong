"""
cook_seq.py — re-cook data.txt to emit, PER DECISION (same order/filter as cook_parallel), the
ordered global DISCARD-EVENT sequence up to that decision, from the deciding player's viewpoint.
Each event token = tile_type(0-33)*4 + rel_seat(0-3) in [0,135]; pad=136 (left-pad to L).
Verifies alignment by re-deriving the act array and (optionally) comparing to cooked_act.npy.

  python3 cook_seq.py --maxmatch 5000 --out data/_seqcheck.npz   # subset align check
  python3 cook_seq.py --out data/cooked_seq.npz                   # full
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from feature import FeatureAgent

HERE = os.path.dirname(os.path.abspath(__file__)); DDIR = os.path.join(HERE, "data")
DATA = os.path.join(DDIR, "data.txt")
L = 48; PAD = 136
TILE_LIST = FeatureAgent.TILE_LIST
OFFSET_TILE = FeatureAgent.OFFSET_TILE   # tile-name -> 0..33


def tok(tile_name, rel):
    return OFFSET_TILE[tile_name] * 4 + rel


def process(lines, maxmatch=0):
    all_seq, all_act = [], []
    obs = [[] for _ in range(4)]; actions = [[] for _ in range(4)]; seqs = [[] for _ in range(4)]
    agents = None; curTile = None
    disc = []  # global ordered list of (tile_name, abs_player)
    nmatch = 0

    def snap(j):
        # last-L events from player j's viewpoint (rel seat = (abs - j) % 4), left-padded
        s = [PAD] * L
        ev = disc[-L:]
        for k, (tn, ap) in enumerate(ev):
            s[L - len(ev) + k] = tok(tn, (ap - j) % 4)
        return np.array(s, np.int16)

    def rec(j, o):
        obs[j].append(o); actions[j].append(0); seqs[j].append(snap(j))

    def flush():
        for j in range(4):
            for i, a in enumerate(actions[j]):
                o = obs[j][i]
                if np.sum(o["action_mask"]) != 1:
                    all_seq.append(seqs[j][i]); all_act.append(a)

    for line in lines:
        t = line.split()
        if not t: continue
        if t[0] == "Match":
            agents = [FeatureAgent(i) for i in range(4)]; disc = []
        elif t[0] == "Wind":
            for ag in agents: ag.request2obs(line.strip())
        elif t[0] == "Player":
            p = int(t[1])
            if t[2] == "Deal":
                agents[p].request2obs(" ".join(t[2:]))
            elif t[2] == "Draw":
                for i in range(4):
                    if i == p:
                        o = agents[p].request2obs(" ".join(t[2:])); rec(p, o)
                    else:
                        agents[i].request2obs(" ".join(t[:3]))
            elif t[2] == "Play":
                actions[p].pop(); actions[p].append(agents[p].response2action(" ".join(t[2:]))); seqs[p].pop(); seqs[p].append(snap(p))
                curTile = t[3]
                disc.append((curTile, p))                     # global discard event (viewpoint-invariant order)
                for i in range(4):
                    if i == p: agents[p].request2obs(line.strip())
                    else:
                        o = agents[i].request2obs(line.strip()); rec(i, o)
            elif t[2] == "Chi":
                actions[p].pop(); actions[p].append(agents[p].response2action("Chi %s %s" % (curTile, t[3]))); seqs[p].pop(); seqs[p].append(snap(p))
                for i in range(4):
                    if i == p:
                        o = agents[p].request2obs("Player %d Chi %s" % (p, t[3])); rec(p, o)
                    else:
                        agents[i].request2obs("Player %d Chi %s" % (p, t[3]))
            elif t[2] == "Peng":
                actions[p].pop(); actions[p].append(agents[p].response2action("Peng %s" % t[3])); seqs[p].pop(); seqs[p].append(snap(p))
                for i in range(4):
                    if i == p:
                        o = agents[p].request2obs("Player %d Peng %s" % (p, t[3])); rec(p, o)
                    else:
                        agents[i].request2obs("Player %d Peng %s" % (p, t[3]))
            elif t[2] == "Gang":
                actions[p].pop(); actions[p].append(agents[p].response2action("Gang %s" % t[3])); seqs[p].pop(); seqs[p].append(snap(p))
                for i in range(4):
                    agents[i].request2obs("Player %d Gang %s" % (p, t[3]))
            elif t[2] == "AnGang":
                actions[p].pop(); actions[p].append(agents[p].response2action("AnGang %s" % t[3])); seqs[p].pop(); seqs[p].append(snap(p))
                for i in range(4):
                    if i == p: agents[p].request2obs("Player %d AnGang %s" % (p, t[3]))
                    else: agents[i].request2obs("Player %d AnGang" % p)
            elif t[2] == "BuGang":
                actions[p].pop(); actions[p].append(agents[p].response2action("BuGang %s" % t[3])); seqs[p].pop(); seqs[p].append(snap(p))
                for i in range(4):
                    if i == p: agents[p].request2obs("Player %d BuGang %s" % (p, t[3]))
                    else:
                        o = agents[i].request2obs("Player %d BuGang %s" % (p, t[3])); rec(i, o)
            elif t[2] == "Hu":
                actions[p].pop(); actions[p].append(agents[p].response2action("Hu")); seqs[p].pop(); seqs[p].append(snap(p))
            if t[2] in ["Peng", "Gang", "Hu"]:
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p = int(t[k + 1])
                        if t[k + 2] == "Chi":
                            actions[p].pop(); actions[p].append(agents[p].response2action("Chi %s %s" % (curTile, t[k + 3]))); seqs[p].pop(); seqs[p].append(snap(p))
                        elif t[k + 2] == "Peng":
                            actions[p].pop(); actions[p].append(agents[p].response2action("Peng %s" % t[k + 3])); seqs[p].pop(); seqs[p].append(snap(p))
                        elif t[k + 2] == "Gang":
                            actions[p].pop(); actions[p].append(agents[p].response2action("Gang %s" % t[k + 3])); seqs[p].pop(); seqs[p].append(snap(p))
                        elif t[k + 2] == "Hu":
                            actions[p].pop(); actions[p].append(agents[p].response2action("Hu")); seqs[p].pop(); seqs[p].append(snap(p))
                    else: break
        elif t[0] == "Score":
            flush()
            for x in obs: x.clear()
            for x in actions: x.clear()
            for x in seqs: x.clear()
            nmatch += 1
            if maxmatch and nmatch >= maxmatch: break
    if not all_act:
        return np.zeros((0, L), np.int16), np.zeros((0,), np.int16)
    return np.stack(all_seq).astype(np.int16), np.array(all_act, np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxmatch", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    with open(DATA, encoding="UTF-8") as f:
        lines = f.readlines()
    print(f"read {len(lines):,} lines ({time.time()-t0:.0f}s)", flush=True)
    seq, act = process(lines, a.maxmatch)
    print(f"samples {len(act):,} seq{seq.shape} ({time.time()-t0:.0f}s)", flush=True)
    # alignment check vs cooked_act.npy
    ck = os.path.join(DDIR, "cooked_act.npy")
    if os.path.exists(ck):
        ca = np.load(ck)
        n = len(act)
        ok = (len(ca) >= n) and np.array_equal(act, ca[:n].astype(np.int16))
        print(f"ALIGN vs cooked_act[:{n}] = {ok}  (cooked N={len(ca):,})", flush=True)
        if not ok and len(ca) >= n:
            mism = int((act != ca[:n].astype(np.int16)).sum())
            print(f"  mismatches={mism}/{n}  first@{int(np.argmax(act != ca[:n].astype(np.int16)))}", flush=True)
    np.savez_compressed(a.out, seq=seq, act=act)
    print(f"saved {a.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
