"""e17_danger_gate.py — paired A/B of margin-gated danger tie-breaking in a
punishing (field-clone) environment.

Per seed & seat rotation, the SAME wall is played twice:
  arm A: cand ensemble plain          vs 3x opponent model
  arm B: cand ensemble + danger tie-break vs 3x opponent model
Overlay rule (B): at a discard decision, among legal Play actions whose policy
logit >= max_logit - margin, choose the one with the LOWEST danger logit.
Calibration: --danger-off makes arm B identical to arm A => diff must be 0.0.
Reports per-arm placement, paired diff, tie/changed rates, cand deal-in counts.
"""
import argparse, json, os, sys, time, collections
import numpy as np
import multiprocessing as mp
from e12_ens_gate import _load, _parse_spec, _ens_fn, PSim
from data.feature_agent import ACT as SACT

PLAY0 = SACT["Play"]
CFG = dict(margin=1.0, enabled=1)
STATS = None


class ABSim(PSim):
    """cand seat with optional danger tie-break; tracks cand deal-ins."""
    def __init__(self, *a, cand_seat=None, danger_fn=None, use_danger=False, **k):
        self.cand_seat = cand_seat
        self.danger_fn = danger_fn
        self.use_danger = use_danger
        self.cand_pending = False      # cand discarded, awaiting immediate claim
        self.dealt_in = 0
        super().__init__(*a, **k)

    def _ask(self, seat):
        from sim_cnn import ACT
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        lg = np.asarray(self.policies[seat](obs[None, :], mask[None, :], return_logits=True)).ravel()
        act = int(lg.argmax())
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        legal = np.flatnonzero(mask)
        has_play = any(PLAY0 <= i < PLAY0 + 34 for i in legal)
        # deal-in bookkeeping: pure-claim Hu right after cand's discard
        if act == SACT["Hu"] and not has_play and self.cand_pending and seat != self.cand_seat:
            self.dealt_in += 1
            self.cand_pending = False
        if PLAY0 <= act < PLAY0 + 34:
            if seat == self.cand_seat:
                if self.use_danger and CFG["enabled"]:
                    STATS["discards"] += 1
                    plays = [i for i in legal if PLAY0 <= i < PLAY0 + 34]
                    mx = max(lg[i] for i in plays)
                    tied = [i for i in plays if lg[i] >= mx - CFG["margin"]]
                    if len(tied) > 1:
                        STATS["ties"] += 1
                        dz = np.asarray(self.danger_fn(obs[None, :], np.ones((1, 235), np.float32), return_logits=True)).ravel()
                        pick = min(tied, key=lambda i: dz[i])
                        if pick != act:
                            STATS["changed"] += 1
                        act = pick
                self.cand_pending = True
            else:
                self.cand_pending = False
        return act


def _work(arg):
    seed, cands, opps, dpath = arg
    global STATS
    STATS = collections.Counter()
    fc = _ens_fn([_load(_parse_spec(p)[0], "resbn_fused", _parse_spec(p)[1]) for p in cands])
    fo = _ens_fn([_load(_parse_spec(p)[0], "resbn_fused", _parse_spec(p)[1]) for p in opps])
    fd = _ens_fn([_load(_parse_spec(p)[0], "resbn_fused", _parse_spec(p)[1]) for p in dpath.split(",")])
    pA = pB = 0.0
    dinA = dinB = 0
    for cs in range(4):
        for use_d, tag in ((False, "A"), (True, "B")):
            pols = [fo] * 4
            pols[cs] = fc
            sim = ABSim(pols, cand_seat=cs, danger_fn=fd, use_danger=use_d,
                        seed=seed, quan=0, learner_seats=[], cnn=True)
            sim.play()
            sc = sim.scores
            c = sc[cs]
            greater = sum(1 for j in range(4) if sc[j] > c)
            equal = sum(1 for j in range(4) if sc[j] == c)
            plc = 5.0 - (greater + (equal + 1) / 2.0)
            if tag == "A":
                pA += plc; dinA += sim.dealt_in
            else:
                pB += plc; dinB += sim.dealt_in
    return pA, pB, dinA, dinB, dict(STATS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--opp", required=True)
    ap.add_argument("--danger", required=True)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=700000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--danger-off", action="store_true")
    a = ap.parse_args()
    CFG["margin"] = a.margin
    CFG["enabled"] = 0 if a.danger_off else 1
    cands = a.cand.split(",")
    opps = a.opp.split(",")
    args = [(a.seed0 + i, cands, opps, a.danger) for i in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers) as p:
        res = p.map(_work, args, chunksize=1)
    n = len(res) * 4
    pA = sum(r[0] for r in res) / n
    pB = sum(r[1] for r in res) / n
    diffs = np.array([(r[1] - r[0]) / 4.0 for r in res])
    se = diffs.std(ddof=1) / np.sqrt(len(diffs))
    agg = collections.Counter()
    for r in res:
        agg.update(r[4])
    disc = max(1, agg.get("discards", 0))
    out = dict(cand=[os.path.basename(c) for c in cands], opp=[os.path.basename(o) for o in opps],
               danger=os.path.basename(a.danger), margin=a.margin, enabled=CFG["enabled"],
               games_per_arm=n, plc_plain=round(pA, 4), plc_danger=round(pB, 4),
               diff=round(pB - pA, 4), diff_se=round(float(se), 4),
               dealins_plain=sum(r[2] for r in res), dealins_danger=sum(r[3] for r in res),
               tie_rate=round(agg.get("ties", 0) / disc, 4),
               changed_rate=round(agg.get("changed", 0) / disc, 4),
               seconds=round(time.time() - t0, 1), seed0=a.seed0)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
