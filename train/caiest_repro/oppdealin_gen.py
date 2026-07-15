"""
oppdealin_gen.py -- OFF-POLICY per-candidate deal-in data generation (v2).

kdens3 (3-net ensemble) plays all 4 seats (same self-play as oppbelief_gen.py). At every
DISCARD decision by seat s, we record the seat's 38-plane public obs AND -- the key idea --
the GROUND-TRUTH immediate deal-in danger of EVERY legal candidate discard tile T, not just
the one kdens3 chose. Because in the simulator we KNOW the true opponent hands, a discard T
"would deal in" iff some opponent o could Ron on T:

    _fan(self.hand[o], self.melds[o], T, o, quan, is_self=False, is_kong=False) >= 8

This reuses the simulator's OWN fan/Hu check -- the exact same _fan the engine calls in
_resolve_claims to decide a real Ron -- so labels are the engine's ground truth, not hand-rolled.

Per DISCARD decision we store one compact row:
  obs    (38,4,9) int8 : seat s's public observation (identical encoding to dealin_pc)
  legal  (34,)    int8 : legal discard tile types (mask[Play:Chi])
  dealin (34,)    int8 : 1 iff discarding that tile type deals into some opponent (only
                          meaningful where legal==1)
  game    int32        : game id (= seed) for game-disjoint splitting in the trainer

The trainer expands each row into per-candidate (state, tile T) training examples over the
legal tiles, adding the 39th one-hot candidate plane at ((T)//9,(T)%9) -- same convention as
dealin_pc_train.py.

  python3 oppdealin_gen.py --games 25000 --workers 48 --seed0 7000000 --tag full
"""
import os, sys, argparse, time, json
sys.path.insert(0, "/root/caiest_repro")
import numpy as np, torch, multiprocessing as mp
import models_explore
from sim_cnn import Sim, ACT, TILE_INDEX, TILE_LIST, _fan, HAS_FAN

torch.set_num_threads(1)
KD = ["ckpt/kd/kd_128x40_s0.pkl", "ckpt/kd/kd_128x40_s1.pkl", "ckpt/kd/kd_128x40_s2.pkl"]
OUTDIR = "/root/caiest_repro/data/oppdealin"
PLAY, CHI = ACT["Play"], ACT["Chi"]          # 2, 36  -> 34 discard tile types
_G = {}


def _init():
    assert HAS_FAN, "MahjongGB fan calculator unavailable -- labels would be wrong; abort."
    models_explore.IN_PLANES = 38
    from models_explore import build
    kd = []
    for p in KD:
        m = build("resbn_fused", channels=128, blocks=40)
        m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); kd.append(m)
    _G["kd"] = kd


def _ens_logits(obs, mask):
    mk = mask.flatten().astype(np.float32); acc = None
    for m in _G["kd"]:
        ob = np.ascontiguousarray(obs).astype(np.float32)
        with torch.no_grad():
            lg = m({"is_training": False, "obs": {
                "observation": torch.from_numpy(ob),
                "action_mask": torch.from_numpy(np.ascontiguousarray(mask))}}).numpy().flatten()
        lg = np.where(mk > 0, lg, -1e30); lg = lg - lg.max()
        p = np.exp(lg) * (mk > 0); s = p.sum()
        p = p / s if s > 0 else (mk / max(1.0, mk.sum()))
        acc = p if acc is None else acc + p
    return np.log(np.where(acc / len(_G["kd"]) > 0, acc / len(_G["kd"]), 1e-30))


class DealInSim(Sim):
    def _ask(self, seat):
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        lg = _ens_logits(obs[None, :], mask[None, :])
        act = int(lg.argmax())
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        if PLAY <= act < CHI:                                   # a DISCARD decision
            legal = mask[PLAY:CHI].astype(np.int8)              # (34,)
            dealin = np.zeros(34, np.int8)
            opps = [(seat + r) % 4 for r in (1, 2, 3)]
            for ti in np.flatnonzero(legal):
                t = TILE_LIST[ti]
                for o in opps:
                    # EXACT engine Ron gate: opp hand + T is a >=8-fan winning hand.
                    if _fan(self.hand[o], self.melds[o], t, o, self.quan, False, False) >= 8:
                        dealin[ti] = 1
                        break
            self._S["obs"].append(obs.astype(np.int8))
            self._S["legal"].append(legal)
            self._S["dealin"].append(dealin)
            self._S["game"].append(self._gid)
        return act


def _work(arg):
    chunk, s0, s1, tag = arg
    S = {"obs": [], "legal": [], "dealin": [], "game": []}
    for seed in range(s0, s1):
        sim = DealInSim([None] * 4, seed=seed, quan=0, learner_seats=[], cnn=True)
        sim._S = S; sim._gid = seed
        sim.play()
    d = os.path.join(OUTDIR, tag); os.makedirs(d, exist_ok=True)
    outp = os.path.join(d, f"shard_{chunk:04d}.npz")
    np.savez(outp,
             obs=np.asarray(S["obs"], np.int8),
             legal=np.asarray(S["legal"], np.int8),
             dealin=np.asarray(S["dealin"], np.int8),
             game=np.asarray(S["game"], np.int32))
    ndec = len(S["obs"])
    lg = np.asarray(S["legal"], np.int8); di = np.asarray(S["dealin"], np.int8)
    ncand = int(lg.sum()); npos = int((di * lg).sum())
    return chunk, ndec, ncand, npos, s1 - s0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=25000)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=7_000_000)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--shard_games", type=int, default=250)
    a = ap.parse_args()
    d = os.path.join(OUTDIR, a.tag); os.makedirs(d, exist_ok=True)
    per = max(1, a.shard_games)
    tasks = []; c = 0; s = a.seed0
    while s < a.seed0 + a.games:
        e = min(s + per, a.seed0 + a.games)
        tasks.append((c, s, e, a.tag)); c += 1; s = e
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init) as p:
        res = p.map(_work, tasks, chunksize=1)
    ndec = sum(r[1] for r in res); ncand = sum(r[2] for r in res)
    npos = sum(r[3] for r in res); ngame = sum(r[4] for r in res)
    base_rate = npos / max(1, ncand)
    manifest = dict(tag=a.tag, games=ngame, decisions=ndec, candidate_rows=ncand,
                    candidate_pos=npos, per_candidate_base_rate=round(base_rate, 6),
                    seed0=a.seed0, seed_range=[a.seed0, a.seed0 + a.games],
                    shards=len(tasks), kd=KD, seconds=round(time.time() - t0, 1),
                    decisions_per_game=round(ndec / max(1, ngame), 2),
                    cand_per_decision=round(ncand / max(1, ndec), 2))
    json.dump(manifest, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    print(f"GEN DONE tag={a.tag} games={ngame} decisions={ndec} candidate_rows={ncand} "
          f"pos={npos} PER_CANDIDATE_BASE_RATE={base_rate:.4f} "
          f"({ndec/max(1,ngame):.1f} dec/game) {time.time()-t0:.0f}s -> {d}", flush=True)


if __name__ == "__main__":
    main()
