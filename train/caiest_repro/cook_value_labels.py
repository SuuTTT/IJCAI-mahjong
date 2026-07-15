"""
cook_value_labels.py — VALUE-HEAD V2 support: re-parse the official data.txt with the
EXACT per-decision extraction/filter of cook_parallel.py (which produced cooked_obs.npy /
cooked_act.npy), but emit per kept decision only the LABELS:

  act      int16  (verification: must equal cooked_act.npy exactly, else abort)
  score    int32  acting seat's final duplicate score from the game's 'Score a b c d' line
  game     int32  global match index (0-based, order of appearance in data.txt)
  seat     int8   acting seat 0-3
  step     int16  index of this decision among the SEAT's kept decisions in the game
  gamelen  int16  number of kept decisions for that seat in the game
  verified int8   scalar 1 iff act matched cooked_act.npy

step/gamelen semantics match final2_harvest/build_corpus_cai.py (per game+seat sequence),
so stage thirds and the GRP metric are computed identically to v1.

Output: data/official_value_labels.npz  (aligned 1:1 with cooked_obs.npy rows)
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
from feature import FeatureAgent

HERE = os.path.dirname(os.path.abspath(__file__))
DDIR = os.path.join(HERE, "data")
DATA = "/root/IJCAI-mahjong-full/IJCAI-mahjong/train/caiest_repro/data/data.txt"
CACT = "/root/IJCAI-mahjong-full/IJCAI-mahjong/train/caiest_repro/data/cooked_act.npy"
OUT = os.path.join(DDIR, "official_value_labels.npz")


def process_lines(args):
    """EXACT decision loop of cook_parallel.process_lines, but store keep-flags and emit
    labels instead of obs/mask. match ids are global via match_offset."""
    lines, match_offset = args
    A, S, G, SE, ST, GL = [], [], [], [], [], []
    keep = [[] for _ in range(4)]
    actions = [[] for _ in range(4)]
    agents = None; curTile = None
    matchid = match_offset - 1

    def flush(scores):
        for j in range(4):
            kept = [actions[j][i] for i in range(len(actions[j])) if keep[j][i]]
            n = len(kept)
            for i, a in enumerate(kept):
                A.append(a); S.append(scores[j]); G.append(matchid)
                SE.append(j); ST.append(i); GL.append(n)

    def rec(p, o):
        keep[p].append(int(np.sum(o["action_mask"])) != 1)

    for line in lines:
        t = line.split()
        if not t: continue
        if t[0] == "Match":
            agents = [FeatureAgent(i) for i in range(4)]
            matchid += 1
        elif t[0] == "Wind":
            for ag in agents: ag.request2obs(line.strip())
        elif t[0] == "Player":
            p = int(t[1])
            if t[2] == "Deal":
                agents[p].request2obs(" ".join(t[2:]))
            elif t[2] == "Draw":
                for i in range(4):
                    if i == p:
                        rec(p, agents[p].request2obs(" ".join(t[2:]))); actions[p].append(0)
                    else:
                        agents[i].request2obs(" ".join(t[:3]))
            elif t[2] == "Play":
                actions[p].pop(); actions[p].append(agents[p].response2action(" ".join(t[2:])))
                for i in range(4):
                    if i == p: agents[p].request2obs(line.strip())
                    else:
                        rec(i, agents[i].request2obs(line.strip())); actions[i].append(0)
                curTile = t[3]
            elif t[2] == "Chi":
                actions[p].pop(); actions[p].append(agents[p].response2action("Chi %s %s" % (curTile, t[3])))
                for i in range(4):
                    if i == p:
                        rec(p, agents[p].request2obs("Player %d Chi %s" % (p, t[3]))); actions[p].append(0)
                    else:
                        agents[i].request2obs("Player %d Chi %s" % (p, t[3]))
            elif t[2] == "Peng":
                actions[p].pop(); actions[p].append(agents[p].response2action("Peng %s" % t[3]))
                for i in range(4):
                    if i == p:
                        rec(p, agents[p].request2obs("Player %d Peng %s" % (p, t[3]))); actions[p].append(0)
                    else:
                        agents[i].request2obs("Player %d Peng %s" % (p, t[3]))
            elif t[2] == "Gang":
                actions[p].pop(); actions[p].append(agents[p].response2action("Gang %s" % t[3]))
                for i in range(4):
                    agents[i].request2obs("Player %d Gang %s" % (p, t[3]))
            elif t[2] == "AnGang":
                actions[p].pop(); actions[p].append(agents[p].response2action("AnGang %s" % t[3]))
                for i in range(4):
                    if i == p: agents[p].request2obs("Player %d AnGang %s" % (p, t[3]))
                    else: agents[i].request2obs("Player %d AnGang" % p)
            elif t[2] == "BuGang":
                actions[p].pop(); actions[p].append(agents[p].response2action("BuGang %s" % t[3]))
                for i in range(4):
                    if i == p: agents[p].request2obs("Player %d BuGang %s" % (p, t[3]))
                    else:
                        rec(i, agents[i].request2obs("Player %d BuGang %s" % (p, t[3]))); actions[i].append(0)
            elif t[2] == "Hu":
                actions[p].pop(); actions[p].append(agents[p].response2action("Hu"))
            if t[2] in ["Peng", "Gang", "Hu"]:
                for k in range(5, 15, 5):
                    if len(t) > k:
                        p = int(t[k + 1])
                        if t[k + 2] == "Chi":
                            actions[p].pop(); actions[p].append(agents[p].response2action("Chi %s %s" % (curTile, t[k + 3])))
                        elif t[k + 2] == "Peng":
                            actions[p].pop(); actions[p].append(agents[p].response2action("Peng %s" % t[k + 3]))
                        elif t[k + 2] == "Gang":
                            actions[p].pop(); actions[p].append(agents[p].response2action("Gang %s" % t[k + 3]))
                        elif t[k + 2] == "Hu":
                            actions[p].pop(); actions[p].append(agents[p].response2action("Hu"))
                    else: break
        elif t[0] == "Score":
            scores = [int(x) for x in t[1:5]]
            flush(scores)
            for x in keep: x.clear()
            for x in actions: x.clear()
    return (np.array(A, np.int16), np.array(S, np.int32), np.array(G, np.int32),
            np.array(SE, np.int8), np.array(ST, np.int16), np.array(GL, np.int16))


def _worker(args):
    try:
        return process_lines(args)
    except Exception as e:
        sys.stderr.write("worker err (offset %s): %s\n" % (args[1], e))
        raise


def split_chunks(path, nchunks):
    with open(path, encoding="UTF-8") as f:
        lines = f.readlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("Match")]
    nm = len(starts)
    per = max(1, (nm + nchunks - 1) // nchunks)
    chunks = []
    for c in range(0, nm, per):
        a = starts[c]
        b = starts[c + per] if c + per < nm else len(lines)
        chunks.append((lines[a:b], c))       # (lines, global match offset)
    return chunks, nm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=96)
    a = ap.parse_args()
    t0 = time.time()
    chunks, nm = split_chunks(DATA, a.workers * 2)
    print(f"matches {nm} split into {len(chunks)} chunks; workers {a.workers}", flush=True)
    with mp.Pool(a.workers) as p:
        res = p.map(_worker, chunks)
    act = np.concatenate([r[0] for r in res])
    score = np.concatenate([r[1] for r in res])
    game = np.concatenate([r[2] for r in res])
    seat = np.concatenate([r[3] for r in res])
    step = np.concatenate([r[4] for r in res])
    gamelen = np.concatenate([r[5] for r in res])
    print(f"total samples {len(act):,}  ({time.time()-t0:.0f}s)", flush=True)

    ref = np.load(CACT, mmap_mode="r")
    ok = (len(act) == len(ref)) and bool(np.array_equal(act, np.asarray(ref)))
    print(f"ALIGN vs cooked_act.npy: n_new={len(act):,} n_ref={len(ref):,} equal={ok}", flush=True)
    if not ok:
        print("ALIGN_FAIL — refusing to write labels", flush=True)
        sys.exit(2)
    print(f"score stats: mean={score.mean():.2f} std={score.std():.2f} "
          f"min={score.min()} max={score.max()} zero_frac={(score==0).mean():.3f}", flush=True)
    print(f"games={game.max()+1} gamelen mean={gamelen.mean():.1f} max={gamelen.max()}", flush=True)
    np.savez_compressed(OUT, act=act, score=score, game=game, seat=seat,
                        step=step, gamelen=gamelen, verified=np.int8(1))
    print(f"ALIGN_OK wrote {OUT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
