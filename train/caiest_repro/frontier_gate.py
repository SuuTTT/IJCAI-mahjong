"""
frontier_gate.py — PLACEMENT-points vs aggression-knob frontier for IJCAI-2026 Mahjong (duplicate format).

Maps the contest metric (rank 4/3/2/1 placement points) as a function of the push/fold aggression knob
(safe_discard.py: SAFE_DISCARD env + SAFE_* thresholds), to decide BALANCE vs SWEET-SPOT.

Design (true duplicate-PLACEMENT, seat-bias cancelled):
  For each wall seed, play the SAME wall 4 times, rotating the CANDIDATE through every seat 0..3,
  with 3 REFERENCE (moyu) bots in the other seats. In each game we rank the 4 micro-scores -> the
  candidate earns 4/3/2/1. Averaging the candidate's placement points over (seeds x 4 rotations)
  gives an unbiased placement-points estimate, AND the 1st/2nd/3rd/4th distribution.

Aggression knob is applied ONLY to the candidate seat, via safe_discard.choose_discard, which reads
the candidate's own CaiAgent (hand/history/packs/shownTiles) — exactly the deploy-time integration.

Calibration / no-op trap: knob OFF (SAFE_DISCARD unset) must reproduce the raw model; and
candidate==reference at knob OFF must give placement-pts ~= 2.5 (4 identical policies tie -> 2.5).

n=0 games => FAIL.

  python3 frontier_gate.py --cand CAND.pkl --ref moyu.pkl --seeds 1200 --workers 120 \
     --knob "off" --seed0 70000 --out gate.json
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, multiprocessing as mp
from sim_cnn import Sim
from models_explore import build
import safe_discard as SD
from feature import FeatureAgent as CaiAgent

# safe_discard.py reads its thresholds from env/os at import; we override its module globals
# per-worker so a single process can sweep multiple knob settings deterministically.
def _sd_configure(sd_mode, min_turn, min_melds, top_k, fold_turn, fold_shanten):
    SD._MODE = sd_mode
    SD.MIN_TURN = min_turn
    SD.MIN_MELDS = min_melds
    SD.TOP_K = top_k
    SD.FOLD_TURN = fold_turn
    SD.FOLD_SHANTEN = fold_shanten
    # safe_discard.choose_discard checks os.environ.get("SAFE_DISCARD")=="2" for fold; set it.
    os.environ["SAFE_DISCARD"] = str(sd_mode)
SD._configure = _sd_configure
SD._MODE = 0

torch.set_num_threads(1)
_CACHE = {}
PLAY = CaiAgent.OFFSET_ACT['Play']           # 2
TILE_LIST = CaiAgent.TILE_LIST
OFFSET_TILE = CaiAgent.OFFSET_TILE

def _parse_cfg(s):
    cfg = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv: continue
        k, v = kv.split("="); cfg[k] = int(v)
    return cfg

def _load(path, kind, cfg):
    key = (path, kind, tuple(sorted(cfg.items())))
    if key not in _CACHE:
        m = build(kind, **cfg)
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

# Aggression knob presets. Each maps to env vars consumed by safe_discard.choose_discard.
# AGGRESSIVE (off) -> raw argmax, never defends. More DEFENSIVE -> earlier/wider genbutsu + fold.
KNOBS = {
    # name        SAFE_DISCARD  MIN_TURN MIN_MELDS TOP_K  FOLD_TURN FOLD_SHANTEN
    "off":        (0,           99,      9,        0,     99,       99),   # raw model (= bot C / A raw)
    "k1_mild":    (1,           14,      2,        2,     99,       99),   # deploy default genbutsu swap
    "k2_earlier": (1,           10,      2,        3,     99,       99),   # defend earlier, wider top-K
    "k3_loose":   (1,            8,      1,        4,     99,       99),   # defend vs 1-meld too
    "k4_fold":    (2,            9,      2,        3,      9,        3),    # + dead-shape fold (deploy v2)
    "k5_foldagg": (2,            8,      1,        4,      8,        2),    # aggressive folding (most defensive)
}

def apply_knob(agent, lg, mask):
    """Given candidate CaiAgent + raw logits + mask, return the chosen action index, knob-modified.
    Only discard (Play) decisions are altered; Hu/Chi/Peng/Gang/Pass untouched (== raw argmax)."""
    a = int(lg.argmax())
    sd_mode = SD._MODE
    if sd_mode == 0:
        return a
    # only alter pure discard turns
    if not (PLAY <= a < PLAY + 34):
        return a
    # legal discard tiles, ranked by logit (best first)
    legal = [i for i in range(PLAY, PLAY + 34) if mask[i]]
    if not legal:
        return a
    legal.sort(key=lambda i: lg[i], reverse=True)
    ranked_tiles = [TILE_LIST[i - PLAY] for i in legal]
    chosen_tile = SD.choose_discard(agent, ranked_tiles, top_k=SD.TOP_K)
    return PLAY + OFFSET_TILE[chosen_tile]


class PlacementSim(Sim):
    """Sim where ONE seat (cand_seat) applies the aggression knob; others are raw policies."""
    def __init__(self, *a, cand_seat=0, **k):
        super().__init__(*a, **k)
        self.cand_seat = cand_seat

    def _ask(self, seat):
        from sim_cnn import ACT  # imported in module
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        lg = self.policies[seat](obs[None, :], mask[None, :], return_logits=True)
        if seat == self.cand_seat and self.cai is not None:
            act = apply_knob(self.cai[seat], lg, mask)
        else:
            act = int(lg.argmax())
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        if seat in self.learner_seats:
            self.traj[seat].append((obs, mask, act))
        return act


def _greedy_lg(m):
    def fn(obs, mask, return_logits=False):
        lg = _logits(m, obs, mask)
        if return_logits:
            return lg
        return [int(lg.argmax())]
    return fn


_G = {}
def _work(arg):
    seed, cand, ck, ccfg, ref, rk, rcfg, knob = arg
    SD._configure(*KNOBS[knob])
    mc = _load(cand, ck, ccfg); mr = _load(ref, rk, rcfg)
    fc = _greedy_lg(mc); fr = _greedy_lg(mr)
    pts = [0, 0, 0, 0]   # placement counts: index 0=1st .. 3=4th, for candidate
    placement_sum = 0.0
    micro_cand = 0
    seat_pts = [[0, 0, 0, 0] for _ in range(4)]  # per cand-seat 1st/2nd/3rd/4th
    for cs in range(4):
        pols = [fr, fr, fr, fr]; pols[cs] = fc
        sim = PlacementSim(pols, seed=seed, quan=0, learner_seats=[], cnn=True, cand_seat=cs)
        sim.play()
        sc = sim.scores
        # rank of candidate seat (1=best). ties: average rank.
        cand_score = sc[cs]
        greater = sum(1 for j in range(4) if sc[j] > cand_score)
        equal = sum(1 for j in range(4) if sc[j] == cand_score)  # includes self
        # average competition rank for ties
        avg_rank = greater + (equal + 1) / 2.0   # 1-based
        ppt = 5.0 - avg_rank                      # 4/3/2/1 for rank 1/2/3/4
        placement_sum += ppt
        micro_cand += cand_score
        # discrete bucket for distribution (round avg_rank to nearest place for reporting)
        r = int(round(avg_rank)) - 1
        r = max(0, min(3, r))
        pts[r] += 1
        seat_pts[cs][r] += 1
    return placement_sum, pts, micro_cand, seat_pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True); ap.add_argument("--cand-kind", default="resbn")
    ap.add_argument("--cand-cfg", default="channels=128,blocks=40")
    ap.add_argument("--ref", required=True); ap.add_argument("--ref-kind", default="resbn")
    ap.add_argument("--ref-cfg", default="channels=128,blocks=40")
    ap.add_argument("--knob", default="off", choices=list(KNOBS.keys()))
    ap.add_argument("--seeds", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=120)
    ap.add_argument("--seed0", type=int, default=70000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ccfg = _parse_cfg(a.cand_cfg); rcfg = _parse_cfg(a.ref_cfg)
    args = [(a.seed0 + i, a.cand, a.cand_kind, ccfg, a.ref, a.ref_kind, rcfg, a.knob)
            for i in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers) as p:
        res = p.map(_work, args, chunksize=2)
    ngames = len(res) * 4   # 4 rotations per seed
    tot_pts = sum(r[0] for r in res)
    dist = [0, 0, 0, 0]
    micro = 0
    seat_dist = [[0, 0, 0, 0] for _ in range(4)]
    for r in res:
        for i in range(4): dist[i] += r[1][i]
        micro += r[2]
        for cs in range(4):
            for i in range(4): seat_dist[cs][i] += r[3][cs][i]
    placement_pts = tot_pts / ngames if ngames else 0.0
    top_half = (dist[0] + dist[1]) / ngames if ngames else 0.0
    out = dict(
        cand=os.path.basename(a.cand), ref=os.path.basename(a.ref), knob=a.knob,
        knob_cfg=dict(zip(["SAFE_DISCARD", "MIN_TURN", "MIN_MELDS", "TOP_K", "FOLD_TURN", "FOLD_SHANTEN"], KNOBS[a.knob])),
        games=ngames, seeds=len(res),
        placement_pts=round(placement_pts, 4),
        dist_1234=dist,
        dist_pct=[round(100 * d / ngames, 1) for d in dist] if ngames else [0, 0, 0, 0],
        top_half_pct=round(100 * top_half, 1),
        first_pct=round(100 * dist[0] / ngames, 1) if ngames else 0.0,
        fourth_pct=round(100 * dist[3] / ngames, 1) if ngames else 0.0,
        micro_cand_per_game=round(micro / ngames, 3) if ngames else 0.0,
        seat_dist_1234=seat_dist,
        seconds=round(time.time() - t0, 1),
        seed0=a.seed0,
    )
    with open(a.out, "w") as f: json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    if ngames == 0:
        print("FAIL: n=0 games", flush=True); sys.exit(2)


if __name__ == "__main__":
    main()
