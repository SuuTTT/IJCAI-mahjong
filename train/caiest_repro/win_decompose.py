"""
win_decompose.py -- decompose the raw-score edge of a candidate checkpoint into the same
components the real-final technical report used (win rate, win-VALUE / self-draw-vs-rong
composition, deal-in rate) to check whether our RL-confirmed win over aug_s0 actually comes
from the same mechanism kong used to beat kdens3 (higher-fan/self-draw win conversion, not
win rate or defense), or from something else entirely.

Reuses sim_cnn.Sim exactly as the gate does (same greedy policy loading), but reads
sim.win_info (patched into sim_cnn.py: (winner_seat, wintype, fan, discarder_or_None)) instead
of just sim.scores, to get the full per-game breakdown.

  python3 win_decompose.py --cand ckpt/CONFIRMED_WIN_20260821/rl2d_it30_fused.pkl \
      --ref ckpt/aug/aug_128x40_s0.pkl --games 2000 --workers 48 --seed0 700000
"""
import os, sys, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, multiprocessing as mp
from sim_cnn import Sim
from models_explore import build

torch.set_num_threads(1)
_CACHE = {}


def _load_policy(path, kind, cfg):
    key = (path, kind, tuple(sorted(cfg.items())))
    if key not in _CACHE:
        m = build(kind, **cfg)
        sd = torch.load(path, map_location="cpu")
        m.load_state_dict(sd)
        m.eval()
        _CACHE[key] = m
    return _CACHE[key]


def _greedy_lg(m):
    def fn(obs, mask):
        with torch.no_grad():
            lg = m({"is_training": False, "obs": {
                "observation": torch.from_numpy(np.ascontiguousarray(obs)),
                "action_mask": torch.from_numpy(np.ascontiguousarray(mask))}})
        return [int(lg.numpy().flatten().argmax())]
    return fn


def _work(arg):
    seed, cand, ck, ccfg, ref, rk, rcfg = arg
    mc = _load_policy(cand, ck, ccfg)
    mr = _load_policy(ref, rk, rcfg)
    fc = _greedy_lg(mc)
    fr = _greedy_lg(mr)
    rows = []
    for cs in range(4):
        pols = [fr, fr, fr, fr]
        pols[cs] = fc
        sim = Sim(pols, seed=seed, quan=0, learner_seats=[], cnn=True)
        sim.play()
        wi = sim.win_info
        cand_score = sim.scores[cs]
        cand_won = wi is not None and wi[0] == cs
        cand_dealt_in = wi is not None and wi[1] == "rong" and wi[3] == cs
        win_type = wi[1] if (wi is not None and cand_won) else None
        win_fan = wi[2] if (wi is not None and cand_won) else None
        rows.append(dict(score=cand_score, won=cand_won, dealt_in=cand_dealt_in,
                          win_type=win_type, win_fan=win_fan, drawn=(wi is None)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--cand-kind", default="resbn_fused")
    ap.add_argument("--cand-cfg", default="channels=128,blocks=40")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-kind", default="resbn_fused")
    ap.add_argument("--ref-cfg", default="channels=128,blocks=40")
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=700000)
    a = ap.parse_args()

    def parse_cfg(s):
        cfg = {}
        for kv in s.split(","):
            kv = kv.strip()
            if not kv:
                continue
            k, v = kv.split("=")
            cfg[k] = int(v)
        return cfg

    ccfg = parse_cfg(a.cand_cfg)
    rcfg = parse_cfg(a.ref_cfg)
    args = [(a.seed0 + i, a.cand, a.cand_kind, ccfg, a.ref, a.ref_kind, rcfg) for i in range(a.games)]
    t0 = time.time()
    with mp.Pool(a.workers) as p:
        res = p.map(_work, args, chunksize=4)
    rows = [r for game_rows in res for r in game_rows]
    n = len(rows)

    n_won = sum(1 for r in rows if r["won"])
    n_dealt_in = sum(1 for r in rows if r["dealt_in"])
    n_drawn = sum(1 for r in rows if r["drawn"])
    n_selfdraw_wins = sum(1 for r in rows if r["win_type"] == "selfdraw")
    n_rong_wins = sum(1 for r in rows if r["win_type"] == "rong")
    fans = [r["win_fan"] for r in rows if r["won"]]
    mean_score = float(np.mean([r["score"] for r in rows]))

    out = dict(
        n_games=n,
        win_rate=round(n_won / n, 4),
        deal_in_rate=round(n_dealt_in / n, 4),
        draw_rate=round(n_drawn / n, 4),
        selfdraw_share_of_wins=round(n_selfdraw_wins / n_won, 4) if n_won else None,
        rong_share_of_wins=round(n_rong_wins / n_won, 4) if n_won else None,
        mean_fan_per_win=round(float(np.mean(fans)), 3) if fans else None,
        mean_score_per_game=round(mean_score, 4),
        seconds=round(time.time() - t0, 1),
    )
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
