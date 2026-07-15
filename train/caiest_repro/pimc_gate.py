"""
pimc_gate.py -- TEST-TIME SEARCH gate: determinized value-guided lookahead (PIMC) vs the base policy
aug_s0, in the calibrated DUPLICATE-format placement gate (4-seat rotation).

Calibration trap preserved: with --N 0 the search is OFF and the cand seat plays raw argmax, so
aug_s0-vs-aug_s0 MUST read placement 2.500 exactly.

THE SEARCH (fires only when the policy's argmax is a DISCARD, i.e. a Play action):
  1. Determinize: sample N hidden worlds (opp hands + wall) consistent with all observations
     (determinize.sample_shown from the cand seat's FeatureAgent view).
  2. For the policy's top-K legal discards within DELTA logit of the top:
       for each of the N worlds, play all 4 seats forward with the cheap conversion playout for H
       plies (csm_rollout). Terminal Hu -> exact our avg_rank; truncated -> V_place leaf (batched).
       Aggregate = mean placement over the N worlds (LOWER = better).
  3. Override the policy's discard with the best-search discard iff its mean placement beats the
     policy's own discard by > MARGIN. (N=0 or margin=inf -> raw argmax -> calibration 2.500.)

Only the DISCARD axis is hooked; Hu / claim / gang decisions fall through to the raw policy.

  python3 pimc_gate.py --cand ckpt/aug/aug_128x40_s0.pkl --ref ckpt/aug/aug_128x40_s0.pkl \
     --value ckpt/value_256x40.pkl --N 20 --H 20 --topk 5 --delta 3.0 --margin 0.0 \
     --seeds 400 --workers 40 --seed0 90000 --out pimc_N20H20.json
"""
import os, sys, json, argparse, time, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/root/IJCAI-mahjong/deploy/caiest_cnn")   # determinize, csm_rollout, pimc_obs
import numpy as np, torch, torch.nn as nn, multiprocessing as mp
from sim_cnn import Sim
from models_explore import build
from feature import FeatureAgent as CaiAgent

import determinize as _D
import pimc_rollout as _RO

torch.set_num_threads(1)
_CACHE = {}
PASS, HU, PLAY = 0, 1, 2
CHI0, PENG0, GANG0 = 36, 99, 133
TILE_LIST = CaiAgent.TILE_LIST


# ---------- value model (ValueMT, identical to train_value.py / e8_gate.py) ----------
IN_PLANES = 38
class _BNBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x))); y = self.b2(self.c2(y)); return torch.relu(x + y)

class ValueMT(nn.Module):
    def __init__(self, channels=128, blocks=20):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(IN_PLANES, channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.body = nn.Sequential(*(_BNBlock(channels) for _ in range(blocks)))
        self.place_head  = nn.Sequential(nn.Linear(channels, 128), nn.ReLU(), nn.Linear(128, 4))
        self.fourth_head = nn.Sequential(nn.Linear(channels, 128), nn.ReLU(), nn.Linear(128, 1))
        self.score_head  = nn.Sequential(nn.Linear(channels, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, obs):
        x = self.body(self.stem(obs)); x = x.mean(dim=(2, 3))
        return self.place_head(x), self.fourth_head(x).squeeze(1), self.score_head(x).squeeze(1)

_PLACE_VEC = np.array([1., 2., 3., 4.])


def _load_value(path):
    key = ("VALUE", path)
    if key not in _CACHE:
        ck = torch.load(path, map_location="cpu")
        net = ValueMT(ck["channels"], ck["blocks"]); net.load_state_dict(ck["state"]); net.eval()
        _CACHE[key] = net
    return _CACHE[key]


def _value_expplace(net, obs_batch):
    with torch.no_grad():
        pl, fo, sc = net(torch.from_numpy(np.ascontiguousarray(obs_batch.astype(np.float32))))
        sm = torch.softmax(pl, 1).numpy()
    return (sm * _PLACE_VEC).sum(1)


def _parse_cfg(s):
    cfg = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv: continue
        k, v = kv.split("="); cfg[k] = int(v)
    return cfg


def _load_policy(path, kind, cfg):
    key = (path, kind, tuple(sorted(cfg.items())))
    if key not in _CACHE:
        m = build(kind, **cfg)
        if path.endswith(".npz"):
            z = np.load(path)
            sd = {k: torch.from_numpy(z[k]) for k in z.keys() if k != "meta_blocks"}
        else:
            sd = torch.load(path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd and not any(
                    k.startswith(("stem", "body", "foot", "head")) for k in sd):
                sd = sd["state_dict"]
        m.load_state_dict(sd); m.eval()
        _CACHE[key] = m
    return _CACHE[key]


def _logits(m, obs, mask):
    with torch.no_grad():
        lg = m({"is_training": False, "obs": {
            "observation": torch.from_numpy(np.ascontiguousarray(obs)),
            "action_mask": torch.from_numpy(np.ascontiguousarray(mask))}})
    return lg.numpy().flatten()


def _greedy_lg(m):
    def fn(obs, mask, return_logits=False):
        lg = _logits(m, obs, mask)
        if return_logits:
            return lg
        return [int(lg.argmax())]
    return fn


# ---------- the PIMC search ----------
def pimc_search_play(lg, mask, agent, vnet, N, H, topk, delta, margin, rng):
    """Return a Play action index (better discard) or None to keep the policy's discard."""
    legal = np.flatnonzero(mask)
    plays = [int(i) for i in legal if PLAY <= i < CHI0]
    if len(plays) < 2:
        return None
    plays.sort(key=lambda i: -float(lg[i]))
    top = float(lg[plays[0]])
    cands = [i for i in plays[:topk] if top - float(lg[i]) <= delta]
    if len(cands) < 2:
        return None

    my_hand = list(agent.hand)
    packs = [list(agent.packs[p]) for p in range(4)]
    # opponents' concealed kongs hide the tile identity ('CONCEALED') -> we can't score/encode that
    # seat's meld; defer to the policy in those (rare) states rather than approximate. (marker -2)
    for p in range(1, 4):
        for tri in packs[p]:
            if tri[1] == "CONCEALED":
                return -2
    shown = dict(agent.shownTiles)
    sw = int(agent.seatWind); prevalent = int(getattr(agent, "prevalentWind", 0))
    seatwinds = [(sw + p) % 4 for p in range(4)]
    base_discards = [list(agent.history[p]) for p in range(4)]

    post = {}; dtile = {}
    for i in cands:
        t = TILE_LIST[i - PLAY]
        if t in my_hand:
            h = list(my_hand); h.remove(t); post[i] = h; dtile[i] = t
    cands = [i for i in cands if i in post]
    if len(cands) < 2:
        return None

    scores = {i: [] for i in cands}
    leaf_obs = []; leaf_owner = []
    for i in cands:
        for _ in range(N):
            world = _D.sample_shown(my_hand, packs, shown, rng)
            kind, val = _RO.rollout_leaf(post[i], dtile[i], world, packs, seatwinds, prevalent,
                                         base_discards, shown, H, rng)
            if kind == "term":
                scores[i].append(val)
            else:
                leaf_obs.append(val); leaf_owner.append(i)
    if leaf_obs:
        expp = _value_expplace(vnet, np.stack(leaf_obs))
        for k, i in enumerate(leaf_owner):
            scores[i].append(float(expp[k]))

    means = {i: (sum(scores[i]) / len(scores[i]) if scores[i] else 2.5) for i in cands}
    raw = cands[0]                                  # policy argmax discard
    best = min(cands, key=lambda i: means[i])
    if means[raw] - means[best] > margin:
        return best
    return None


class PIMCSim(Sim):
    def __init__(self, *a, cand_seat=0, vnet=None, N=0, H=20, topk=5, delta=3.0, margin=0.0,
                 seed=0, stats=None, **k):
        super().__init__(*a, seed=seed, **k)
        self.cand_seat = cand_seat
        self.vnet = vnet; self.N = N; self.H = H; self.topk = topk
        self.delta = delta; self.margin = margin
        self.stats = stats
        self._rng = random.Random(seed * 7919 + cand_seat + 1)

    def _ask(self, seat):
        from sim_cnn import ACT
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        lg = self.policies[seat](obs[None, :], mask[None, :], return_logits=True)
        raw_a = int(lg.argmax())
        if not mask[raw_a]:
            raw_a = int(np.flatnonzero(mask)[0])
        act = raw_a
        if seat == self.cand_seat and self.N > 0 and PLAY <= raw_a < CHI0:
            t0 = time.time()
            pick = pimc_search_play(lg, mask, self.cai[seat], self.vnet,
                                    self.N, self.H, self.topk, self.delta, self.margin, self._rng)
            if pick == -2:                       # concealed-kong guard: deferred, not a search move
                if self.stats is not None:
                    self.stats["guarded"] += 1
            else:
                if pick is not None and mask[pick]:
                    act = pick
                if self.stats is not None:
                    self.stats["search_moves"] += 1
                    self.stats["search_ms"] += (time.time() - t0) * 1000.0
                    if act != raw_a:
                        self.stats["overridden"] += 1
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        if seat in self.learner_seats:
            self.traj[seat].append((obs, mask, act))
        return act


def _work(arg):
    seed, cand, ck, ccfg, ref, rk, rcfg, vpath, N, H, topk, delta, margin = arg
    mc = _load_policy(cand, ck, ccfg); mr = _load_policy(ref, rk, rcfg)
    vnet = _load_value(vpath) if N > 0 else None
    fc = _greedy_lg(mc); fr = _greedy_lg(mr)
    pts = [0, 0, 0, 0]; placement_sum = 0.0
    stats = {"search_moves": 0, "search_ms": 0.0, "overridden": 0, "guarded": 0}
    pg_rank = []
    for cs in range(4):
        pols = [fr, fr, fr, fr]; pols[cs] = fc
        sim = PIMCSim(pols, seed=seed, quan=0, learner_seats=[], cnn=True,
                      cand_seat=cs, vnet=vnet, N=N, H=H, topk=topk, delta=delta,
                      margin=margin, stats=stats)
        sim.play()
        sc = sim.scores
        cand_score = sc[cs]
        greater = sum(1 for j in range(4) if sc[j] > cand_score)
        equal = sum(1 for j in range(4) if sc[j] == cand_score)
        avg_rank = greater + (equal + 1) / 2.0
        placement_sum += (5.0 - avg_rank)
        r = max(0, min(3, int(round(avg_rank)) - 1)); pts[r] += 1
        pg_rank.append(float(avg_rank))
    return placement_sum, pts, stats, pg_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True); ap.add_argument("--cand-kind", default="resbn_fused")
    ap.add_argument("--cand-cfg", default="channels=128,blocks=40")
    ap.add_argument("--ref", required=True); ap.add_argument("--ref-kind", default="resbn_fused")
    ap.add_argument("--ref-cfg", default="channels=128,blocks=40")
    ap.add_argument("--value", default="ckpt/value_256x40.pkl")
    ap.add_argument("--N", type=int, default=0)
    ap.add_argument("--H", type=int, default=20)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--delta", type=float, default=3.0)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=90000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ccfg = _parse_cfg(a.cand_cfg); rcfg = _parse_cfg(a.ref_cfg)
    args = [(a.seed0 + i, a.cand, a.cand_kind, ccfg, a.ref, a.ref_kind, rcfg, a.value,
             a.N, a.H, a.topk, a.delta, a.margin) for i in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers) as p:
        res = p.map(_work, args, chunksize=1)
    ngames = len(res) * 4
    tot_pts = sum(r[0] for r in res)
    dist = [0, 0, 0, 0]
    st = {"search_moves": 0, "search_ms": 0.0, "overridden": 0, "guarded": 0}
    pg_rank = []
    for r in res:
        for i in range(4): dist[i] += r[1][i]
        for k in st: st[k] += r[2][k]
        pg_rank.extend(r[3])
    pg_rank = np.asarray(pg_rank, dtype=np.float64)
    placement_pts = tot_pts / ngames if ngames else 0.0
    per_move_ms = (st["search_ms"] / st["search_moves"]) if st["search_moves"] else 0.0
    out = dict(cand=os.path.basename(a.cand), ref=os.path.basename(a.ref),
               value=os.path.basename(a.value), N=a.N, H=a.H, topk=a.topk, delta=a.delta,
               margin=a.margin, games=ngames, seeds=len(res),
               placement_pts=round(placement_pts, 4),
               dist_1234=dist,
               first_pct=round(100 * dist[0] / ngames, 2) if ngames else 0.0,
               fourth_pct=round(100 * dist[3] / ngames, 2) if ngames else 0.0,
               sg_win_rate=round(float((pg_rank <= 1.5).mean()), 4),
               sg_fourth_rate=round(float((pg_rank >= 3.5).mean()), 4),
               search_moves=st["search_moves"], guarded=st["guarded"],
               override_rate=round(st["overridden"] / st["search_moves"], 4) if st["search_moves"] else 0.0,
               per_move_ms=round(per_move_ms, 2),
               seconds=round(time.time() - t0, 1), seed0=a.seed0)
    with open(a.out, "w") as f: json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)
    if ngames == 0:
        print("FAIL: n=0 games", flush=True); sys.exit(2)


if __name__ == "__main__":
    main()
