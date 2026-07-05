"""e16_safe_gate.py — duplicate gate for the deploy-time genbutsu defense overlay.

cand seat plays the ensemble WITH safe_discard.choose_discard applied to its discards
(the exact deploy code path); ref seats play the same/other ensemble plain.
Calibration trap: --safe-off with cand==ref must give exactly 2.500.
Reports mechanism engagement: discard decisions seen / threat-active / action-changed.
"""
import argparse, json, os, sys, time, collections
import numpy as np
import multiprocessing as mp
from e12_ens_gate import _load, _parse_spec, _ens_fn, PSim
from data.feature_agent import ACT as SACT, TILE_LIST as STILES
import safe_discard

PLAY0 = SACT["Play"]
TILES = STILES
CFG = dict(top_k=2, enabled=1)
STATS = None


class SafeSim(PSim):
    def __init__(self, *a, safe_seat=None, **k):
        self.safe_seat = safe_seat
        super().__init__(*a, **k)

    def _ask(self, seat):
        from sim_cnn import ACT
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        lg = self.policies[seat](obs[None, :], mask[None, :], return_logits=True)
        lg = np.asarray(lg).ravel()
        act = int(lg.argmax())
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        if seat != self.safe_seat or not CFG["enabled"]:
            return act
        if not (PLAY0 <= act < PLAY0 + 34):
            return act                                   # not a discard choice
        STATS["discards"] += 1
        legal_plays = [i for i in np.flatnonzero(mask) if PLAY0 <= i < PLAY0 + 34]
        order = sorted(legal_plays, key=lambda i: -lg[i])
        ranked_tiles = [TILES[i - PLAY0] for i in order]
        ag = self.cai[seat]
        # engagement bookkeeping mirroring choose_discard's internal conditions
        if len(ag.history[0]) >= safe_discard.MIN_TURN and \
           any(len(ag.packs[p]) >= safe_discard.MIN_MELDS for p in (1, 2, 3)):
            STATS["threat"] += 1
        t = safe_discard.choose_discard(ag, ranked_tiles, top_k=CFG["top_k"])
        chosen = PLAY0 + TILES.index(t)
        if chosen != act:
            STATS["changed"] += 1
        return chosen


def _work(arg):
    seed, cands, refs = arg
    global STATS
    STATS = collections.Counter()
    fc = _ens_fn([_load(_parse_spec(p)[0], "resbn_fused", _parse_spec(p)[1]) for p in cands])
    fr = _ens_fn([_load(_parse_spec(p)[0], "resbn_fused", _parse_spec(p)[1]) for p in refs])
    placement_sum = 0.0
    for cs in range(4):
        pols = [fr] * 4
        pols[cs] = fc
        sim = SafeSim(pols, safe_seat=cs, seed=seed, quan=0, learner_seats=[], cnn=True)
        sim.play()
        sc = sim.scores
        c = sc[cs]
        greater = sum(1 for j in range(4) if sc[j] > c)
        equal = sum(1 for j in range(4) if sc[j] == c)
        placement_sum += 5.0 - (greater + (equal + 1) / 2.0)
    return placement_sum, dict(STATS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=600000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-turn", type=int, default=12)
    ap.add_argument("--min-melds", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--safe-off", action="store_true")
    a = ap.parse_args()
    safe_discard.MIN_TURN = a.min_turn
    safe_discard.MIN_MELDS = a.min_melds
    CFG["top_k"] = a.top_k
    CFG["enabled"] = 0 if a.safe_off else 1
    cands = a.cand.split(",")
    refs = a.ref.split(",")
    args = [(a.seed0 + i, cands, refs) for i in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers) as p:
        res = p.map(_work, args, chunksize=1)
    ngames = len(res) * 4
    pts = sum(r[0] for r in res) / ngames if ngames else 0.0
    agg = collections.Counter()
    for _, st in res:
        agg.update(st)
    disc = max(1, agg.get("discards", 0))
    out = dict(cand=[os.path.basename(c) for c in cands], ref=[os.path.basename(r) for r in refs],
               overlay=dict(enabled=CFG["enabled"], min_turn=a.min_turn, min_melds=a.min_melds, top_k=a.top_k),
               games=ngames, placement_pts=round(pts, 4),
               discard_decisions=agg.get("discards", 0),
               threat_rate=round(agg.get("threat", 0) / disc, 4),
               changed_rate=round(agg.get("changed", 0) / disc, 4),
               seconds=round(time.time() - t0, 1), seed0=a.seed0)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)
    if ngames == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
