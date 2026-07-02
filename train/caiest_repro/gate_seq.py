"""
gate_seq.py — calibrated duplicate placement gate for the TEMPORAL CNN+GRU net. Identical scoring
to e11_gate (4-seat rotation, base-vs-base = 2.500), but the candidate seat's policy is the
temporal net, which also consumes the per-seat ordered discard sequence produced in-sim by a
seq-aware FeatureAgent subclass (CaiSeq). Shared files (sim_cnn/feature) are NOT modified: we
monkeypatch sim_cnn.CaiAgent -> CaiSeq and subclass Sim._ask. Ref policy = aug_s0 (resbn_fused).

  python3 gate_seq.py --cand ckpt/archx/temporal_s0.pkl --cand-cfg channels=128,blocks=40,emb=64,gru=256 \
     --ref ckpt/aug/aug_128x40_s0.pkl --seeds 500 --workers 48 --seed0 900000 --out cell.json
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, multiprocessing as mp
import sim_cnn
from sim_cnn import Sim
from feature import FeatureAgent
from models_seq import build_seq, PAD, VOCAB
from models_explore import build
torch.set_num_threads(1)
L = 48


class CaiSeq(FeatureAgent):
    """FeatureAgent that also tracks the GLOBAL ordered discard events, exposing disc_seq()
    from this agent's seat viewpoint (tile*4 + rel_seat, left-pad PAD=136), matching cook_seq."""
    def __init__(self, seatWind):
        super().__init__(seatWind); self._gdisc = []
    def request2obs(self, msg):
        t = msg.split()
        if len(t) >= 4 and t[0] == "Player" and t[2] == "Play":
            self._gdisc.append((t[3], int(t[1])))
        return super().request2obs(msg)
    def disc_seq(self):
        s = np.full(L, PAD, np.int64); ev = self._gdisc[-L:]
        base = L - len(ev)
        for k, (tn, ap) in enumerate(ev):
            s[base + k] = self.OFFSET_TILE[tn] * 4 + ((ap - self.seatWind) % 4)
        return s


class SeqSim(Sim):
    seq_seat = None; seq_fn = None
    def _ask(self, seat):
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return sim_cnn.ACT["Pass"]
        if seat == self.seq_seat and self.seq_fn is not None:
            sq = self.cai[seat].disc_seq()
            act = int(self.seq_fn(obs[None, :], mask[None, :], sq[None, :]))
        else:
            act = int(self.policies[seat](obs[None, :], mask[None, :])[0])
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        if seat in self.learner_seats:
            self.traj[seat].append((obs, mask, act))
        return act


_CACHE = {}

def _parse_cfg(s):
    cfg = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv: continue
        k, v = kv.split("="); cfg[k] = int(v)
    return cfg

def _load_ref(path, kind, cfg):
    key = ("ref", path, kind, tuple(sorted(cfg.items())))
    if key not in _CACHE:
        m = build(kind, **cfg)
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd and not any(k.startswith(("stem","body","foot","head")) for k in sd):
            sd = sd["state_dict"]
        m.load_state_dict(sd); m.eval(); _CACHE[key] = m
    return _CACHE[key]

def _load_temporal(path, cfg, kind="temporal"):
    key = ("tmp", path, kind, tuple(sorted(cfg.items())))
    if key not in _CACHE:
        m = build_seq(kind, **cfg)
        m.load_state_dict(torch.load(path, map_location="cpu")); m.eval(); _CACHE[key] = m
    return _CACHE[key]

def _ref_fn(m):
    def fn(obs, mask):
        with torch.no_grad():
            lg = m({"is_training": False, "obs": {
                "observation": torch.from_numpy(np.ascontiguousarray(obs)),
                "action_mask": torch.from_numpy(np.ascontiguousarray(mask))}})
        return [int(lg.numpy().flatten().argmax())]
    return fn

def _seq_fn(m):
    def fn(obs, mask, sq):
        with torch.no_grad():
            lg = m({"is_training": False, "seq": torch.from_numpy(np.ascontiguousarray(sq)),
                    "obs": {"observation": torch.from_numpy(np.ascontiguousarray(obs)),
                            "action_mask": torch.from_numpy(np.ascontiguousarray(mask))}})
        return int(lg.numpy().flatten().argmax())
    return fn


def _work(arg):
    seed, cand, ck2, ccfg, ref, rk, rcfg = arg
    mc = _load_temporal(cand, ccfg, ck2); mr = _load_ref(ref, rk, rcfg)
    fc = _seq_fn(mc); fr = _ref_fn(mr)
    placement_sum = 0.0; pts = [0, 0, 0, 0]; micro = 0; pg_rank = []
    for cs in range(4):
        pols = [fr, fr, fr, fr]
        sim = SeqSim(pols, seed=seed, quan=0, learner_seats=[], cnn=True)
        sim.seq_seat = cs; sim.seq_fn = fc
        sim.play()
        sc = sim.scores; cand_score = sc[cs]
        greater = sum(1 for j in range(4) if sc[j] > cand_score)
        equal = sum(1 for j in range(4) if sc[j] == cand_score)
        avg_rank = greater + (equal + 1) / 2.0
        placement_sum += 5.0 - avg_rank; micro += cand_score
        r = max(0, min(3, int(round(avg_rank)) - 1)); pts[r] += 1
        pg_rank.append(float(avg_rank))
    return placement_sum, pts, micro, pg_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True); ap.add_argument("--cand-kind", default="temporal"); ap.add_argument("--cand-cfg", default="channels=128,blocks=40,emb=64,gru=256")
    ap.add_argument("--ref", required=True); ap.add_argument("--ref-kind", default="resbn_fused")
    ap.add_argument("--ref-cfg", default="channels=128,blocks=40")
    ap.add_argument("--seeds", type=int, default=500); ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=900000); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ccfg = _parse_cfg(a.cand_cfg); rcfg = _parse_cfg(a.ref_cfg)
    # monkeypatch: sim uses seq-aware agents (drop-in; adds global-discard tracking)
    sim_cnn.CaiAgent = CaiSeq
    args = [(a.seed0 + i, a.cand, a.cand_kind, ccfg, a.ref, a.ref_kind, rcfg) for i in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init) as p:
        res = p.map(_work, args, chunksize=1)
    ngames = len(res) * 4
    tot = sum(r[0] for r in res); dist = [0, 0, 0, 0]; micro = 0; pg = []
    for r in res:
        for i in range(4): dist[i] += r[1][i]
        micro += r[2]; pg.extend(r[3])
    pg = np.asarray(pg, np.float64)
    out = dict(cand=os.path.basename(a.cand), ref=os.path.basename(a.ref), games=ngames, seeds=len(res),
               placement_pts=round(tot / ngames, 4) if ngames else 0.0, dist_1234=dist,
               first_pct=round(100 * dist[0] / ngames, 2), fourth_pct=round(100 * dist[3] / ngames, 2),
               sg_win_rate=round(float((pg <= 1.5).mean()), 4), sg_fourth_rate=round(float((pg >= 3.5).mean()), 4),
               micro_cand_per_game=round(micro / ngames, 3), seconds=round(time.time() - t0, 1), seed0=a.seed0)
    with open(a.out, "w") as f: json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)

def _init():
    import sim_cnn as _s; _s.CaiAgent = CaiSeq  # ensure workers use seq-aware agent

if __name__ == "__main__":
    main()
