"""e20_fold_gate.py — paired A/B of a COHERENT fold policy (sticky, claim-suppressing).

Fold latch (per hand, cand seat): once [best-shanten >= SH_FOLD] AND [turn >= MIN_TURN]
AND [some opponent has >= MIN_MELDS melds  OR  best-shanten >= SH_DEAD], stay folded.
While folded: discard the SAFEST legal tile (genbutsu vs threats > all-4-seen >
honors-3-seen > terminals > policy order), and PASS all Chi/Peng/Gang claims. Hu always taken.
Arms play the SAME wall: A = plain ensemble, B = ensemble + fold policy.
Calibration: --fold-off => diff must be exactly 0. Deal-ins tracked per arm.
"""
import argparse, json, os, sys, time, collections
import numpy as np
import multiprocessing as mp
from e12_ens_gate import _load, _parse_spec, _ens_fn, PSim
from data.feature_agent import ACT as SACT, TILE_LIST as STILES
from MahjongGB import MahjongShanten

PLAY0 = SACT["Play"]
CFG = dict(sh_fold=2, sh_dead=3, min_turn=8, min_melds=2, enabled=1)
STATS = None


def best_shanten(ag):
    sh = 99
    pk = tuple(ag.packs[0])
    for t in set(ag.hand):
        h = list(ag.hand)
        h.remove(t)
        try:
            s = MahjongShanten(pack=pk, hand=tuple(h))
        except Exception:
            continue
        if s < sh:
            sh = s
    return sh


def safety_key(ag, tile, threats):
    """Lower = safer."""
    gen_all = all(tile in ag.history[p] for p in threats) if threats else False
    gen_any = any(tile in ag.history[p] for p in threats)
    try:
        seen = ag.shownTiles[tile]
    except Exception:
        seen = 0
    honor = tile[0] in ("F", "J")
    terminal = len(tile) == 2 and tile[1] in ("1", "9") and not honor
    return (0 if gen_all else 1 if gen_any else 2,
            0 if (honor and seen >= 3) else 1,
            -seen,
            0 if honor else 1 if terminal else 2)


class FoldSim(PSim):
    def __init__(self, *a, cand_seat=None, use_fold=False, **k):
        self.cand_seat = cand_seat
        self.use_fold = use_fold
        self.folded = False
        self.cand_pending = False
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
        if act == SACT["Hu"] and not has_play and self.cand_pending and seat != self.cand_seat:
            self.dealt_in += 1
            self.cand_pending = False
        if seat != self.cand_seat:
            if PLAY0 <= act < PLAY0 + 34:
                self.cand_pending = False
            return act
        if not (self.use_fold and CFG["enabled"]):
            if PLAY0 <= act < PLAY0 + 34:
                self.cand_pending = True
            return act
        ag = self.cai[seat]
        threats = [p for p in (1, 2, 3) if len(ag.packs[p]) >= CFG["min_melds"]]
        # latch check on our discard turns
        if PLAY0 <= act < PLAY0 + 34:
            STATS["discards"] += 1
            if not self.folded:
                sh = best_shanten(ag)
                if len(ag.history[0]) >= CFG["min_turn"] and (
                        (threats and sh >= CFG["sh_fold"]) or sh >= CFG["sh_dead"]):
                    self.folded = True
                    STATS["folds"] += 1
            if self.folded:
                plays = [i for i in legal if PLAY0 <= i < PLAY0 + 34]
                pick = min(plays, key=lambda i: safety_key(ag, STILES[i - PLAY0], threats))
                if pick != act:
                    STATS["changed"] += 1
                act = pick
                STATS["folded_discards"] += 1
            self.cand_pending = True
            return act
        # claim decision while folded: pass everything except Hu
        if self.folded and act != SACT["Hu"] and act != SACT["Pass"]:
            STATS["claims_suppressed"] += 1
            return SACT["Pass"]
        return act


def _work(arg):
    seed, cands, refs = arg
    global STATS
    STATS = collections.Counter()
    fc = _ens_fn([_load(_parse_spec(p)[0], "resbn_fused", _parse_spec(p)[1]) for p in cands])
    fr = _ens_fn([_load(_parse_spec(p)[0], "resbn_fused", _parse_spec(p)[1]) for p in refs])
    pA = pB = 0.0
    dinA = dinB = 0
    for cs in range(4):
        for use_fold in (False, True):
            pols = [fr] * 4
            pols[cs] = fc
            sim = FoldSim(pols, cand_seat=cs, use_fold=use_fold, seed=seed, quan=0,
                          learner_seats=[], cnn=True)
            sim.play()
            sc = sim.scores
            c = sc[cs]
            greater = sum(1 for j in range(4) if sc[j] > c)
            equal = sum(1 for j in range(4) if sc[j] == c)
            plc = 5.0 - (greater + (equal + 1) / 2.0)
            if use_fold:
                pB += plc; dinB += sim.dealt_in
            else:
                pA += plc; dinA += sim.dealt_in
    return pA, pB, dinA, dinB, dict(STATS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=800000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sh-fold", type=int, default=2)
    ap.add_argument("--sh-dead", type=int, default=3)
    ap.add_argument("--min-turn", type=int, default=8)
    ap.add_argument("--min-melds", type=int, default=2)
    ap.add_argument("--fold-off", action="store_true")
    a = ap.parse_args()
    CFG.update(sh_fold=a.sh_fold, sh_dead=a.sh_dead, min_turn=a.min_turn,
               min_melds=a.min_melds, enabled=0 if a.fold_off else 1)
    cands = a.cand.split(",")
    refs = a.ref.split(",")
    args = [(a.seed0 + i, cands, refs) for i in range(a.seeds)]
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
    out = dict(cand=[os.path.basename(c) for c in cands], overlay=dict(**CFG),
               games_per_arm=n, plc_plain=round(pA, 4), plc_fold=round(pB, 4),
               diff=round(pB - pA, 4), diff_se=round(float(se), 4),
               dealins_plain=sum(r[2] for r in res), dealins_fold=sum(r[3] for r in res),
               folds=agg.get("folds", 0), folded_discards=agg.get("folded_discards", 0),
               changed=agg.get("changed", 0), claims_suppressed=agg.get("claims_suppressed", 0),
               discards=agg.get("discards", 0),
               seconds=round(time.time() - t0, 1), seed0=a.seed0)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
