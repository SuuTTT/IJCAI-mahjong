"""
parse_botzone_logs.py — turn real Botzone match logs (display-event JSON, like the
competitor replays in others/) into (a) WINNER-trajectory training samples and (b)
per-game / per-seat stats. Unlike data.txt (2022 human games), these are the ACTUAL
CONTEST distribution — strong bots converting against the real field — so a BC/fine-tune
on them is in-distribution for the ladder (the 2022 high-fan BC failed precisely because
it was out-of-distribution).

Dedups repeated games (same srand seen from multiple seats). Reuses replay_harness's
reconstruction to get per-decision (obs, mask, taken); keeps the WINNER's decisions where
the taken action is concrete (>=0).

  python3 data/parse_botzone_logs.py others/ --out data/processed/contest_winner.npz
  python3 data/parse_botzone_logs.py others/ --stats   # just print stats, no npz
"""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from eval.replay_harness import reconstruct, _disp_stream

def game_srand(path):
    d = json.load(open(path))
    for rec in d:
        disp = (rec.get("output") or {}).get("display") or {}
        if disp.get("action") == "INIT":
            return disp.get("srand")
    return None

def win_info(path):
    """Return (kind, winner, fan, scores) from the terminal HU display, else draw."""
    d = json.load(open(path))
    for rec in reversed(d):
        disp = (rec.get("output") or {}).get("display") or {}
        if disp.get("action") == "HU":
            return ("hu", disp.get("player"), disp.get("fanCnt"), disp.get("score"))
        if disp.get("action") == "HUANG":
            return ("draw", -1, 0, [0, 0, 0, 0])
    return ("unknown", -1, 0, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="data/processed/contest_winner.npz")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()

    logs = [p for p in glob.glob(os.path.join(a.root, "**", "*.log"), recursive=True)]
    seen_srand = {}
    for p in logs:
        sr = game_srand(p)
        # keep the first log per unique game (full hands present in all seats' logs)
        if sr not in seen_srand:
            seen_srand[sr] = p
    games = list(seen_srand.values())

    print(f"{len(logs)} log files -> {len(games)} unique games")
    n_hu = n_draw = 0; fans = []; win_kind = {"selfdraw": 0, "rong": 0}
    obs_l, mask_l, act_l = [], [], []
    for p in games:
        kind, winner, fan, scores = win_info(p)
        if kind == "hu":
            n_hu += 1; fans.append(fan)
            if scores:
                # rong: exactly one loser more-negative than the others
                negs = sorted(s for s in scores if s < 0)
                win_kind["rong" if len(set(negs)) > 1 else "selfdraw"] += 1
            quan, decisions, _ = reconstruct(p)
            for d in decisions:
                if d["seat"] == winner and d["kind"] == "draw" and d["taken"] is not None and d["taken"] >= 0:
                    obs_l.append(d["obs"]); mask_l.append(d["mask"]); act_l.append(d["taken"])
        elif kind == "draw":
            n_draw += 1

    print(f"  outcomes: HU={n_hu}  draw={n_draw}  draw-rate={100*n_draw/max(1,len(games)):.0f}%")
    if fans:
        print(f"  winning fan: mean={np.mean(fans):.1f} min={min(fans)} max={max(fans)}  "
              f"by={win_kind}")
    print(f"  winner draw-turn samples extracted: {len(obs_l)}")
    print(f"  (for reference: a usable BC fine-tune wants >=50k-100k samples => "
          f"~{max(1,100000//max(1,len(obs_l)//max(1,n_hu)))//1} games-scale more)")
    if not a.stats and obs_l:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        np.savez_compressed(a.out, obs=np.asarray(obs_l, np.uint8),
                            mask=np.asarray(mask_l, np.bool_), act=np.asarray(act_l, np.int16))
        print(f"  wrote {a.out}")

if __name__ == "__main__":
    main()
