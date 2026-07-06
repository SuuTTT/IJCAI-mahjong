"""extract_danger_cai.py — like extract_danger_sim11 but emits the CAIEST (38,4,9)
encoding by replaying cai FeatureAgents in lockstep with the display stream.

Outputs:
  <out>          obs(f16 N,38,4,9) tile(i8) y(i8)          — danger view (discards)
  <out>_bc.npz   obs(f16) mask(bool N,235) act(i16)         — all decisions (BC view)
"""
import os, sys, json, glob, argparse
import numpy as np
import multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train", "caiest_repro"))
from eval.replay_harness import _disp_stream
from data.parse_botzone_logs import win_info
from feature import FeatureAgent as CaiAgent

PLAY0 = CaiAgent.OFFSET_ACT["Play"]
OT = CaiAgent.OFFSET_TILE


def _snap(ag):
    d = ag._obs()
    return d["observation"].astype(np.float16), d["action_mask"].astype(np.bool_)


def _one(p):
    try:
        kind, winner, fan, scores = win_info(p)
        if kind not in ("hu", "draw"):
            return None
        rong = False
        if kind == "hu" and scores:
            negs = sorted(s for s in scores if s < 0)
            rong = len(set(negs)) > 1
        records = json.load(open(p))
        disps = list(_disp_stream(records))
    except Exception:
        return None
    quan = 0
    cais = None
    dec = []          # (obs, mask, act)  act in 235-space
    try:
        for disp in disps:
            a = disp["action"]
            if a == "INIT":
                quan = disp.get("quan", 0)
            elif a == "DEAL":
                cais = [CaiAgent(s) for s in range(4)]
                for s in range(4):
                    cais[s].request2obs(f"Wind {quan}")
                hands = disp["hand"]
                for s in range(4):
                    cais[s].request2obs("Deal " + " ".join(hands[s]))
            elif a == "DRAW":
                pid = disp["player"]; tile = disp["tile"]
                for s in range(4):
                    cais[s].request2obs(f"Draw {tile}" if s == pid else f"Player {pid} Draw")
                o, m = _snap(cais[pid])
                dec.append([o, m, None, pid, "draw"])
            elif a == "PLAY":
                pid = disp["player"]; tile = disp["tile"]
                for d in reversed(dec):
                    if d[3] == pid and d[4] == "draw" and d[2] is None:
                        d[2] = PLAY0 + OT[tile]; break
                else:
                    # discard after a claim (no draw decision): snapshot now
                    o, m = _snap(cais[pid])
                    dec.append([o, m, PLAY0 + OT[tile], pid, "postclaim"])
                # bystander views of this discard (for multi-perspective danger labels)
                last_bystanders = []
                for s in range(4):
                    if s != pid:
                        try:
                            ox = cais[s]._obs()["observation"].astype(np.float16)
                            last_bystanders.append((ox, PLAY0 + OT[tile]))
                        except Exception:
                            pass
                for s in range(4):
                    cais[s].request2obs(f"Player {pid} Play {tile}")
            elif a == "CHI":
                pid = disp["player"]; mid = disp["tileCHI"]
                for s in range(4):
                    cais[s].request2obs(f"Player {pid} Chi {mid}")
            elif a == "PENG":
                pid = disp["player"]
                for s in range(4):
                    cais[s].request2obs(f"Player {pid} Peng")
            elif a == "GANG":
                pid = disp["player"]
                for s in range(4):
                    cais[s].request2obs(f"Player {pid} Gang")
    except Exception:
        return None
    obs_l, til_l = [], []
    bobs, bmask, bact = [], [], []
    for o, m, act, pid, k in dec:
        if act is None or act < 0:
            continue
        bobs.append(o); bmask.append(m); bact.append(act)
        if PLAY0 <= act < PLAY0 + 34:
            obs_l.append(o); til_l.append(act - PLAY0)
    if not obs_l:
        return None
    y = np.zeros(len(obs_l), np.int8)
    if rong:
        y[-1] = 1
        try:
            for ox, actx in last_bystanders:
                obs_l.append(ox); til_l.append(actx - PLAY0)
                y = np.append(y, np.int8(1))
        except Exception:
            pass
    return (np.stack(obs_l), np.array(til_l, np.int8), y,
            np.stack(bobs), np.stack(bmask), np.array(bact, np.int16))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    logs = sorted(glob.glob(os.path.join(a.root, "**", "*.log"), recursive=True))
    print("games:", len(logs), flush=True)
    O, T, Y, BO, BM, BA = [], [], [], [], [], []
    ok = 0
    with mp.Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(_one, logs, chunksize=8)):
            if r is None:
                continue
            ok += 1
            O.append(r[0]); T.append(r[1]); Y.append(r[2])
            BO.append(r[3]); BM.append(r[4]); BA.append(r[5])
            if ok % 500 == 0:
                print(ok, "games,", sum(len(y) for y in Y), "discards,", sum(int(y.sum()) for y in Y), "positives", flush=True)
    obs = np.concatenate(O); til = np.concatenate(T); y = np.concatenate(Y)
    np.savez_compressed(a.out, obs=obs, tile=til, y=y)
    bo = np.concatenate(BO); bm = np.concatenate(BM); ba = np.concatenate(BA)
    np.savez_compressed(a.out.replace(".npz", "_bc.npz"), obs=bo, mask=bm, act=ba)
    print("DONE games_ok=%d danger=%d positives=%d (%.2f%%) bc=%d" % (
        ok, len(y), int(y.sum()), 100 * y.sum() / len(y), len(ba)), flush=True)


if __name__ == "__main__":
    main()
