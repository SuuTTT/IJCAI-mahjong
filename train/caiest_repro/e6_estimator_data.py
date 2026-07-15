"""e6_estimator_data.py — E6-P2(a) training data for the opponent-strength estimator.

Replays the EXACT Phase-1 field setups (field_response.py FIELDS): kdens3 at
seat 0 vs each 3-opponent field, --ngames games per field, logging seat-0's
(38,4,9) obs snapshot at its 4th/8th/12th discard decision. Seeds are a FRESH
block (default 10000000+) disjoint from Phase-1 (7.0M-8.2M) and from the
switcher eval/gate, so the estimator never trains on a deal it is evaluated on.

  python3 e6_estimator_data.py --ngames 2500 --workers 110 \
      --seed0 10000000 --out data/e6_est_snaps.npz
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, multiprocessing as mp
from sim_cnn import Sim, ACT
from e6_switch_common import (FIELDS, FIELD_ORDER, CLS_OF_FIELD, KD_PATHS,
                              PLAY_LO, PLAY_HI, SNAP_TURNS, policy_fn, preload_all)


class SnapSim(Sim):
    def __init__(self, policy_fn, **k):
        super().__init__(policy_fn, **k)
        self.turn = 0; self.snaps = {}

    def _ask(self, seat):
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        if seat == 0 and mask[PLAY_LO:PLAY_HI].any():
            self.turn += 1
            if self.turn in SNAP_TURNS:
                self.snaps[self.turn] = (obs > 0).astype(np.uint8)
        lg = self.policies[seat](obs[None, :], mask[None, :], return_logits=True)
        act = int(lg.argmax())
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        return act


def _work(arg):
    seed, field = arg
    pols = [policy_fn(KD_PATHS)] + [policy_fn([p]) for p in FIELDS[field]]
    sim = SnapSim(pols, seed=seed, quan=0, learner_seats=[], cnn=True)
    sim.play()
    return field, seed, {t: o for t, o in sim.snaps.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ngames", type=int, default=2500)
    ap.add_argument("--workers", type=int, default=110)
    ap.add_argument("--seed0", type=int, default=10000000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    preload_all()
    data = {t: {"X": [], "y3": [], "field": [], "seed": []} for t in SNAP_TURNS}
    for fi, field in enumerate(FIELD_ORDER):
        t0 = time.time()
        s0 = a.seed0 + fi * 100000
        args = [(s0 + i, field) for i in range(a.ngames)]
        with mp.Pool(a.workers) as p:
            res = p.map(_work, args, chunksize=4)
        got = {t: 0 for t in SNAP_TURNS}
        for f, seed, snaps in res:
            for t, o in snaps.items():
                d = data[t]
                d["X"].append(o); d["y3"].append(CLS_OF_FIELD[f])
                d["field"].append(fi); d["seed"].append(seed)
                got[t] += 1
        print(f"FIELD {field}: {len(res)} games seed0={s0} snaps/turn={got} "
              f"({time.time()-t0:.1f}s)", flush=True)
    out = {}
    for t in SNAP_TURNS:
        d = data[t]
        out[f"X{t}"] = np.stack(d["X"]).astype(np.uint8)
        out[f"y{t}"] = np.asarray(d["y3"], dtype=np.int64)
        out[f"field{t}"] = np.asarray(d["field"], dtype=np.int64)
        out[f"seed{t}"] = np.asarray(d["seed"], dtype=np.int64)
    np.savez_compressed(a.out, **out)
    meta = {t: int(len(data[t]["y3"])) for t in SNAP_TURNS}
    print("SAVED", a.out, "samples per turn:", json.dumps({str(k): v for k, v in meta.items()}),
          flush=True)


if __name__ == "__main__":
    main()
