"""gen_danger_data.py — self-play data for a discard-danger head.

Runs kdens3-vs-kdens3 sims; for EVERY discard decision records
(obs uint8, discard tile idx, dealt_in flag) where dealt_in=1 iff the very next
event is an opponent winning on that tile (rong). Shards npz per worker chunk.
"""
import argparse, os, sys, time
import numpy as np
import multiprocessing as mp
from e12_ens_gate import _load, _parse_spec, _ens_fn, PSim
from data.feature_agent import ACT as SACT

PLAY0 = SACT["Play"]


class RecSim(PSim):
    def __init__(self, *a, rec=None, **k):
        self.rec = rec
        self._pending = None          # (seat, obs, tile_idx) awaiting outcome
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
        has_play = any(PLAY0 <= i < PLAY0 + 34 for i in np.flatnonzero(mask))
        if PLAY0 <= act < PLAY0 + 34:
            # flush previous pending as safe (it wasn't immediately claimed for win)
            if self._pending is not None:
                s, o, t = self._pending
                self.rec.append((o, t, 0))
            self._pending = (seat, obs.astype(np.float16), act - PLAY0)
        elif act == SACT.get("Hu", 1) and not has_play and self._pending is not None:
            # pure claim-Hu right after a discard (rong) -> that discard dealt in
            s, o, t = self._pending
            self.rec.append((o, t, 1))
            self._pending = None
        return act

    def play(self):
        r = super().play()
        if self._pending is not None:
            s, o, t = self._pending
            self.rec.append((o, t, 0))
            self._pending = None
        return r


def _work(arg):
    seed, cands, ngames, outdir = arg
    fc = _ens_fn([_load(_parse_spec(p)[0], "resbn_fused", _parse_spec(p)[1]) for p in cands])
    rec = []
    for g in range(ngames):
        sim = RecSim([fc] * 4, rec=rec, seed=seed * 100 + g, quan=0, learner_seats=[], cnn=True)
        try:
            sim.play()
        except Exception:
            continue
    if not rec:
        return 0, 0
    obs = np.stack([r[0] for r in rec]).astype(np.float16)
    til = np.array([r[1] for r in rec], np.int8)
    lab = np.array([r[2] for r in rec], np.int8)
    np.savez_compressed(os.path.join(outdir, "danger_%07d.npz" % seed), obs=obs, tile=til, y=lab)
    return len(lab), int(lab.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--chunks", type=int, default=1000)      # worker chunks
    ap.add_argument("--games-per-chunk", type=int, default=50)
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=900000)
    ap.add_argument("--outdir", default="danger_data")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    cands = a.cand.split(",")
    args = [(a.seed0 + i, cands, a.games_per_chunk, a.outdir) for i in range(a.chunks)]
    t0 = time.time()
    tot = pos = 0
    with mp.Pool(a.workers) as p:
        for n, k in p.imap_unordered(_work, args, chunksize=1):
            tot += n; pos += k
            if tot and tot % 50000 < n:
                print("decisions=%d dealins=%d (%.2f%%) elapsed=%.0fs" % (tot, pos, 100 * pos / max(1, tot), time.time() - t0), flush=True)
    print("DONE decisions=%d dealins=%d games=%d" % (tot, pos, a.chunks * a.games_per_chunk), flush=True)


if __name__ == "__main__":
    main()
