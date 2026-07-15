"""
winprob_gate.py — PLACEMENT-points gate for the WITHIN-GAME WIN-RATE controller (IJCAI-2026 MCR).

Identical proven harness to frontier_gate.py / targeted_gate.py (true duplicate placement: same wall
played 4x rotating the CANDIDATE through all seats vs 3 moyu references; candidate earns 4/3/2/1;
seat-bias cancelled; no-op trap = cand==ref@off -> 2.5000). The ONLY difference from targeted_gate is
the override path: knob="winprob" uses winprob_controller.choose_discard, which folds (genbutsu) ONLY
when P(I win this hand) < TAU. Compare vs raw (knob=off, ~2.51). n=0 => FAIL.

  python3 winprob_gate.py --cand base_resbn40_v3.pkl --ref moyu_bn_128x40.pkl --seeds 600 \
     --workers 48 --knob winprob --tau 0.15 --seed0 70000 --out wp_tau0.15.json
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, multiprocessing as mp
from sim_cnn import Sim
from models_explore import build
import winprob_controller as WC
from feature import FeatureAgent as CaiAgent

torch.set_num_threads(1)
_CACHE = {}
PLAY = CaiAgent.OFFSET_ACT['Play']
TILE_LIST = CaiAgent.TILE_LIST
OFFSET_TILE = CaiAgent.OFFSET_TILE
_WP_ON = False


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


def apply_winprob(agent, lg, mask):
    a = int(lg.argmax())
    if not _WP_ON:
        return a
    if not (PLAY <= a < PLAY + 34):
        return a
    legal = [i for i in range(PLAY, PLAY + 34) if mask[i]]
    if not legal:
        return a
    legal.sort(key=lambda i: lg[i], reverse=True)
    ranked_tiles = [TILE_LIST[i - PLAY] for i in legal]
    chosen_tile = WC.choose_discard(agent, ranked_tiles, top_k=WC.TOP_K)
    return PLAY + OFFSET_TILE[chosen_tile]


class PlacementSim(Sim):
    def __init__(self, *a, cand_seat=0, **k):
        super().__init__(*a, **k)
        self.cand_seat = cand_seat

    def _ask(self, seat):
        from sim_cnn import ACT
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        lg = self.policies[seat](obs[None, :], mask[None, :], return_logits=True)
        if seat == self.cand_seat and self.cai is not None:
            act = apply_winprob(self.cai[seat], lg, mask)
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
        return lg if return_logits else [int(lg.argmax())]
    return fn


def _work(arg):
    global _WP_ON
    seed, cand, ck, ccfg, ref, rk, rcfg, knob, tau, topk, minturn, minmelds = arg
    _WP_ON = (knob == "winprob")
    if _WP_ON:
        WC.TAU = tau; WC.TOP_K = topk; WC.MIN_TURN = minturn; WC.MIN_MELDS = minmelds
        for kk in WC.STATS: WC.STATS[kk] = 0 if isinstance(WC.STATS[kk], int) else 0.0
    mc = _load(cand, ck, ccfg); mr = _load(ref, rk, rcfg)
    fc = _greedy_lg(mc); fr = _greedy_lg(mr)
    pts = [0, 0, 0, 0]
    placement_sum = 0.0
    micro_cand = 0
    seat_pts = [[0, 0, 0, 0] for _ in range(4)]
    for cs in range(4):
        pols = [fr, fr, fr, fr]; pols[cs] = fc
        sim = PlacementSim(pols, seed=seed, quan=0, learner_seats=[], cnn=True, cand_seat=cs)
        sim.play()
        sc = sim.scores
        cand_score = sc[cs]
        greater = sum(1 for j in range(4) if sc[j] > cand_score)
        equal = sum(1 for j in range(4) if sc[j] == cand_score)
        avg_rank = greater + (equal + 1) / 2.0
        ppt = 5.0 - avg_rank
        placement_sum += ppt
        micro_cand += cand_score
        r = max(0, min(3, int(round(avg_rank)) - 1))
        pts[r] += 1
        seat_pts[cs][r] += 1
    wp_stats = dict(WC.STATS) if _WP_ON else None
    return placement_sum, pts, micro_cand, seat_pts, wp_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True); ap.add_argument("--cand-kind", default="resbn")
    ap.add_argument("--cand-cfg", default="channels=128,blocks=40")
    ap.add_argument("--ref", required=True); ap.add_argument("--ref-kind", default="resbn")
    ap.add_argument("--ref-cfg", default="channels=128,blocks=40")
    ap.add_argument("--knob", default="off", choices=["off", "winprob"])
    ap.add_argument("--seeds", type=int, default=600)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=70000)
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--minturn", type=int, default=6)
    ap.add_argument("--minmelds", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ccfg = _parse_cfg(a.cand_cfg); rcfg = _parse_cfg(a.ref_cfg)
    args = [(a.seed0 + i, a.cand, a.cand_kind, ccfg, a.ref, a.ref_kind, rcfg, a.knob,
             a.tau, a.topk, a.minturn, a.minmelds) for i in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers) as p:
        res = p.map(_work, args, chunksize=2)
    ngames = len(res) * 4
    tot_pts = sum(r[0] for r in res)
    dist = [0, 0, 0, 0]; micro = 0
    seat_dist = [[0, 0, 0, 0] for _ in range(4)]
    for r in res:
        for i in range(4): dist[i] += r[1][i]
        micro += r[2]
        for cs in range(4):
            for i in range(4): seat_dist[cs][i] += r[3][cs][i]
    wp = {"discards": 0, "fired": 0, "noop_same": 0, "wp_sum": 0.0, "wp_fired_sum": 0.0, "low_wp": 0}
    have = False
    for r in res:
        if len(r) > 4 and r[4] is not None:
            have = True
            for kk in wp: wp[kk] += r[4][kk]
    placement_pts = tot_pts / ngames if ngames else 0.0
    out = dict(
        cand=os.path.basename(a.cand), ref=os.path.basename(a.ref), knob=a.knob,
        games=ngames, seeds=len(res),
        placement_pts=round(placement_pts, 4),
        dist_1234=dist,
        dist_pct=[round(100 * d / ngames, 1) for d in dist] if ngames else [0, 0, 0, 0],
        first_pct=round(100 * dist[0] / ngames, 1) if ngames else 0.0,
        fourth_pct=round(100 * dist[3] / ngames, 1) if ngames else 0.0,
        micro_cand_per_game=round(micro / ngames, 3) if ngames else 0.0,
        seat_dist_1234=seat_dist,
        tau=a.tau if a.knob == "winprob" else None,
        topk=a.topk, minturn=a.minturn, minmelds=a.minmelds,
        wp_discards=wp["discards"] if have else None,
        wp_low=wp["low_wp"] if have else None,          # # discards with P(win)<tau (trigger)
        wp_fired=wp["fired"] if have else None,         # # discards actually swapped (trigger + safe tile)
        wp_low_pct=round(100.0 * wp["low_wp"] / wp["discards"], 3) if (have and wp["discards"]) else None,
        wp_fire_pct=round(100.0 * wp["fired"] / wp["discards"], 3) if (have and wp["discards"]) else None,
        wp_mean=round(wp["wp_sum"] / wp["discards"], 4) if (have and wp["discards"]) else None,
        wp_mean_fired=round(wp["wp_fired_sum"] / wp["fired"], 4) if (have and wp["fired"]) else None,
        seconds=round(time.time() - t0, 1), seed0=a.seed0,
    )
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)
    if ngames == 0:
        print("FAIL: n=0 games", flush=True); sys.exit(2)


if __name__ == "__main__":
    main()
