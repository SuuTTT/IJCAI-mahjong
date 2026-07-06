"""extract_danger_sim11.py — (obs, discard_tile, dealt_in) triples from real field games.

For each game: reconstruct all decisions; every action in the Play range is a discard
sample. If the game ended by rong, the LAST discard of the game is the deal-in (label 1);
all other discards label 0. Run over a dir of display-stream .log files.
"""
import os, sys, json, glob, argparse
import numpy as np
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.replay_harness import reconstruct
from data.parse_botzone_logs import win_info
from data.feature_agent import ACT as SACT

PLAY0 = SACT["Play"]


def _one(p):
    try:
        kind, winner, fan, scores = win_info(p)
        if kind != "hu" or not scores:
            return None
        negs = sorted(s for s in scores if s < 0)
        rong = len(set(negs)) > 1
        quan, decisions, _ = reconstruct(p)
    except Exception:
        return None
    obs_l, til_l = [], []
    bobs, bmask, bact = [], [], []
    for d in decisions:
        t = d.get("taken")
        if t is None or t < 0:
            continue
        bobs.append(np.asarray(d["obs"], np.float16))
        bmask.append(np.asarray(d["mask"], np.bool_))
        bact.append(t)
        if PLAY0 <= t < PLAY0 + 34:
            obs_l.append(np.asarray(d["obs"], np.float16))
            til_l.append(t - PLAY0)
    if not obs_l:
        return None
    y = np.zeros(len(obs_l), np.int8)
    if rong:
        y[-1] = 1                      # last discard of a rong game = the deal-in
    return (np.stack(obs_l), np.array(til_l, np.int8), y,
            np.stack(bobs), np.stack(bmask), np.array(bact, np.int16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=40)
    a = ap.parse_args()
    logs = sorted(glob.glob(os.path.join(a.root, "**", "*.log"), recursive=True))
    print("games:", len(logs), flush=True)
    O, T, Y, BO, BM, BA = [], [], [], [], [], []
    with mp.Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(_one, logs, chunksize=8)):
            if r is None:
                continue
            O.append(r[0]); T.append(r[1]); Y.append(r[2])
            BO.append(r[3]); BM.append(r[4]); BA.append(r[5])
            if i % 500 == 0:
                print(i, "games,", sum(len(y) for y in Y), "samples,", sum(int(y.sum()) for y in Y), "positives", flush=True)
    obs = np.concatenate(O); til = np.concatenate(T); y = np.concatenate(Y)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, obs=obs, tile=til, y=y)
    bo = np.concatenate(BO); bm = np.concatenate(BM); ba = np.concatenate(BA)
    np.savez_compressed(a.out.replace(".npz", "_bc.npz"), obs=bo, mask=bm, act=ba)
    print("DONE danger=%d positives=%d (%.2f%%); bc=%d -> %s" % (len(y), int(y.sum()), 100 * y.sum() / len(y), len(ba), a.out), flush=True)


if __name__ == "__main__":
    main()
