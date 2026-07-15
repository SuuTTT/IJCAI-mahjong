"""Evaluate a net-ensemble policy vs a fixed minimax opponent ladder.
Net inference on CPU; games parallelized across CPU workers. Deterministic
policies, so game variety comes from random paired openings."""
import math
import multiprocessing as mp
import numpy as np
import torch
import othello as oth
import teacher as T
import game as G
from model import PolicyCNN, encode_planes, NSQ

_W = {}  # per-worker state


def _load_models(paths, ch):
    models = []
    for p in paths:
        m = PolicyCNN(ch=ch)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval()
        models.append(m)
    return models


def _ensemble_move(models, me, opp):
    """Average softmax over models, mask to legal, argmax."""
    legal = oth.legal_moves(me, opp)
    if legal == 0:
        return None
    X = encode_planes(np.array([me], dtype=np.int64), np.array([opp], dtype=np.int64))
    xt = torch.from_numpy(X)
    with torch.no_grad():
        probs = None
        for m in models:
            p = torch.softmax(m(xt), dim=1)[0]
            probs = p if probs is None else probs + p
    probs = probs.numpy()
    mask = np.array([(legal >> s) & 1 for s in range(NSQ)], dtype=bool)
    probs = np.where(mask, probs, -1.0)
    return int(np.argmax(probs))


def _init(paths, ch, opp_depth):
    torch.set_num_threads(1)          # CRITICAL: avoid thread oversubscription
    torch.set_num_interop_threads(1)
    _W["models"] = _load_models(paths, ch)
    _W["opp_depth"] = opp_depth


def _play_pair(seed):
    models = _W["models"]
    d = _W["opp_depth"]
    net_pol = lambda me, opp: _ensemble_move(models, me, opp)
    opp_pol = lambda me, opp: T.best_move(me, opp, d)
    import random
    opening = G.random_opening(random.Random(seed), 4)
    pts = 0.0
    # net as black
    res, _, _ = G.play_game(net_pol, opp_pol, opening)
    pts += 1.0 if res == 1 else (0.5 if res == 0 else 0.0)
    # net as white
    res, _, _ = G.play_game(opp_pol, net_pol, opening)
    pts += 1.0 if res == -1 else (0.5 if res == 0 else 0.0)
    return pts


def eval_arm(model_paths, ch, opp_depth, n_pairs, seed, workers):
    seeds = [((seed << 20) ^ i) & 0x7FFFFFFF for i in range(n_pairs)]
    ctx = mp.get_context("spawn")   # spawn: avoid inheriting CUDA state from parent
    with ctx.Pool(workers, initializer=_init,
                  initargs=(model_paths, ch, opp_depth)) as pool:
        pts = pool.map(_play_pair, seeds)
    total_pts = float(np.sum(pts))
    n_games = 2 * n_pairs
    wr = total_pts / n_games
    ci = wilson_ci(total_pts, n_games)
    return {"opp_depth": opp_depth, "winrate": wr, "n_games": n_games,
            "points": total_pts, "ci95": ci}


def wilson_ci(points, n, z=1.96):
    """Wilson score interval treating points (draws=0.5) as successes."""
    if n == 0:
        return [0.0, 1.0]
    p = points / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return [round(center - half, 4), round(center + half, 4)]
