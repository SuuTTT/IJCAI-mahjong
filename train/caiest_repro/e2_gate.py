"""
e1_gate.py — duplicate-format PLACEMENT gate (copy of frontier_gate.py) with an added
claim-suppression overlay (--claim-tau). Used by E1 to measure each model's placement
(a) raw and (b) with a tau-margin claim suppression, vs the moyu reference.

Calibration trap preserved: moyu-vs-moyu raw must read ~2.500.

Claim-suppression overlay (tau): at a CLAIM-LEGAL state (Pass available AND >=1 Chi/Peng legal)
where the raw argmax is a claim (action in [36,133)), the claim is only kept if
    logit[best_legal_claim] - logit[Pass] >= tau
otherwise the action is forced to Pass. tau=0 => no-op (== raw). This directly tests whether
suppressing marginal over-claims improves placement. (Independent of the safe_discard knob; here
knob is left 'off' so the ONLY intervention is claim suppression.)

  python3 e1_gate.py --cand C.pkl --cand-kind resbn_fused --cand-cfg channels=128,blocks=40 \
     --ref /root/assets/moyu_bn_128x40.pkl --ref-kind resbn --ref-cfg channels=128,blocks=40 \
     --claim-tau 2.0 --seeds 400 --workers 100 --seed0 70000 --out gate.json
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, multiprocessing as mp
from sim_cnn import Sim
from models_explore import build
import safe_discard as SD
from feature import FeatureAgent as CaiAgent

torch.set_num_threads(1)
_CACHE = {}
PLAY = CaiAgent.OFFSET_ACT['Play']
CHI0, PENG0, GANG0 = 36, 99, 133


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


def apply_claim_tau(lg, mask, tau):
    """If argmax is a claim at a claim-legal state and its margin over Pass < tau, force Pass."""
    a = int(lg.argmax())
    if tau <= 0:
        return a
    if not (CHI0 <= a < GANG0):
        return a
    if not mask[0]:           # Pass not legal -> cannot suppress
        return a
    legal_claims = [i for i in range(CHI0, GANG0) if mask[i]]
    if not legal_claims:
        return a
    best_claim = max(lg[i] for i in legal_claims)
    if best_claim - lg[0] < tau:
        return 0              # force Pass
    return a


class PlacementSim(Sim):
    def __init__(self, *a, cand_seat=0, claim_tau=0.0, ref_tau=0.0, count=None, **k):
        super().__init__(*a, **k)
        self.cand_seat = cand_seat
        self.claim_tau = claim_tau
        self.ref_tau = ref_tau
        # count: optional dict to tally claim-legal states + claims kept, per role,
        # so we can sanity-check that ref_tau actually lowers the ref bots claim-rate.
        self.count = count

    def _ask(self, seat):
        from sim_cnn import ACT
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        lg = self.policies[seat](obs[None, :], mask[None, :], return_logits=True)
        is_cand = (seat == self.cand_seat)
        tau = self.claim_tau if is_cand else self.ref_tau
        # tally claim-legal opportunities for the ref role (sanity-check of ref_tau)
        if self.count is not None and (not is_cand):
            raw_a = int(lg.argmax())
            if mask[0] and any(mask[i] for i in range(CHI0, GANG0)):
                self.count["ref_claim_legal"] += 1
                if CHI0 <= raw_a < GANG0:
                    self.count["ref_claim_raw"] += 1
        if tau > 0:
            act = apply_claim_tau(lg, mask, tau)
        else:
            act = int(lg.argmax())
        if self.count is not None and (not is_cand):
            if mask[0] and any(mask[i] for i in range(CHI0, GANG0)) and CHI0 <= act < GANG0:
                self.count["ref_claim_kept"] += 1
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


def _work(arg):
    seed, cand, ck, ccfg, ref, rk, rcfg, claim_tau, ref_tau = arg
    mc = _load(cand, ck, ccfg); mr = _load(ref, rk, rcfg)
    fc = _greedy_lg(mc); fr = _greedy_lg(mr)
    pts = [0, 0, 0, 0]; placement_sum = 0.0; micro_cand = 0
    cnt = {"ref_claim_legal": 0, "ref_claim_raw": 0, "ref_claim_kept": 0}
    for cs in range(4):
        pols = [fr, fr, fr, fr]; pols[cs] = fc
        sim = PlacementSim(pols, seed=seed, quan=0, learner_seats=[], cnn=True,
                           cand_seat=cs, claim_tau=claim_tau, ref_tau=ref_tau, count=cnt)
        sim.play()
        sc = sim.scores
        cand_score = sc[cs]
        greater = sum(1 for j in range(4) if sc[j] > cand_score)
        equal = sum(1 for j in range(4) if sc[j] == cand_score)
        avg_rank = greater + (equal + 1) / 2.0
        ppt = 5.0 - avg_rank
        placement_sum += ppt; micro_cand += cand_score
        r = max(0, min(3, int(round(avg_rank)) - 1)); pts[r] += 1
    return placement_sum, pts, micro_cand, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True); ap.add_argument("--cand-kind", default="resbn_fused")
    ap.add_argument("--cand-cfg", default="channels=128,blocks=40")
    ap.add_argument("--ref", required=True); ap.add_argument("--ref-kind", default="resbn")
    ap.add_argument("--ref-cfg", default="channels=128,blocks=40")
    ap.add_argument("--claim-tau", type=float, default=0.0)
    ap.add_argument("--ref-tau", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=70000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ccfg = _parse_cfg(a.cand_cfg); rcfg = _parse_cfg(a.ref_cfg)
    args = [(a.seed0 + i, a.cand, a.cand_kind, ccfg, a.ref, a.ref_kind, rcfg, a.claim_tau, a.ref_tau)
            for i in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers) as p:
        res = p.map(_work, args, chunksize=2)
    ngames = len(res) * 4
    tot_pts = sum(r[0] for r in res)
    dist = [0, 0, 0, 0]; micro = 0
    cl_legal = cl_raw = cl_kept = 0
    for r in res:
        for i in range(4): dist[i] += r[1][i]
        micro += r[2]
        c = r[3]
        cl_legal += c["ref_claim_legal"]; cl_raw += c["ref_claim_raw"]; cl_kept += c["ref_claim_kept"]
    placement_pts = tot_pts / ngames if ngames else 0.0
    ref_claim_rate_raw = round(cl_raw / cl_legal, 4) if cl_legal else 0.0
    ref_claim_rate_kept = round(cl_kept / cl_legal, 4) if cl_legal else 0.0
    out = dict(cand=os.path.basename(a.cand), ref=os.path.basename(a.ref),
               claim_tau=a.claim_tau, ref_tau=a.ref_tau, games=ngames, seeds=len(res),
               ref_claim_legal_states=cl_legal,
               ref_claim_rate_raw=ref_claim_rate_raw,
               ref_claim_rate_kept=ref_claim_rate_kept,
               placement_pts=round(placement_pts, 4),
               dist_1234=dist,
               dist_pct=[round(100 * d / ngames, 1) for d in dist] if ngames else [0, 0, 0, 0],
               first_pct=round(100 * dist[0] / ngames, 1) if ngames else 0.0,
               fourth_pct=round(100 * dist[3] / ngames, 1) if ngames else 0.0,
               micro_cand_per_game=round(micro / ngames, 3) if ngames else 0.0,
               seconds=round(time.time() - t0, 1), seed0=a.seed0)
    with open(a.out, "w") as f: json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)
    if ngames == 0:
        print("FAIL: n=0 games", flush=True); sys.exit(2)


if __name__ == "__main__":
    main()
