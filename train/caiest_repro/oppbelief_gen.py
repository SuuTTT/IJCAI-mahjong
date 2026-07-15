"""
oppbelief_gen.py -- self-play data generation for the OPPONENT HAND-BELIEF model.

kdens3 (3-net ensemble deploy rule) plays ALL 4 seats. At every DISCARD decision by a
seat s, record:
  INPUT  obs   : seat s's 38-plane public-view feature (own hand + all discards + melds +
                 wall counts; opponents' concealed tiles NOT in it -- that's the point).
  TARGET tgt   : (3,34) int8 count of each of the 3 opponents' CONCEALED tiles
                 (self.hand[opp]), indexed by RELATIVE position (rel 1=next,2=across,3=prev)
                 -> seat-invariant.
  unseen (34)  : count of each tile type UNSEEN by s (= 3 opp hands + all 4 wall remainders)
                 -> exactly the pool for the uniform hypergeometric baseline.
  hsz (3)      : the 3 relative opponents' concealed hand sizes (for the baseline).

Writes shards to data/oppbelief/<tag>/shard_XXX.npz. Split-by-game done in the trainer.

  python3 oppbelief_gen.py --games 30000 --workers 60 --seed0 5000000 --tag full
"""
import os, sys, argparse, time, glob
sys.path.insert(0, "/root/caiest_repro")
import numpy as np, torch, multiprocessing as mp
import models_explore
from sim_cnn import Sim, ACT, TILE_INDEX

torch.set_num_threads(1)
KD = ["ckpt/kd/kd_128x40_s0.pkl", "ckpt/kd/kd_128x40_s1.pkl", "ckpt/kd/kd_128x40_s2.pkl"]
OUTDIR = "/root/caiest_repro/data/oppbelief"
_G = {}


def _init():
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


class BeliefSim(Sim):
    def _ask(self, seat):
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        lg = _ens_logits(obs[None, :], mask[None, :])
        act = int(lg.argmax())
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        if ACT["Play"] <= act < ACT["Chi"]:                      # a discard decision
            tgt = np.zeros((3, 34), np.int8); hsz = np.zeros(3, np.int8)
            for rel in (1, 2, 3):
                o = (seat + rel) % 4
                for t in self.hand[o]:
                    tgt[rel - 1, TILE_INDEX[t]] += 1
                hsz[rel - 1] = len(self.hand[o])
            unseen = np.zeros(34, np.int8)
            for o in range(4):
                if o != seat:
                    for t in self.hand[o]:
                        unseen[TILE_INDEX[t]] += 1
                for t in self.walls[o]:
                    unseen[TILE_INDEX[t]] += 1
            self._S["obs"].append(obs.astype(np.int8))
            self._S["tgt"].append(tgt); self._S["uns"].append(unseen)
            self._S["hsz"].append(hsz); self._S["game"].append(self._gid)
        return act


def _work(arg):
    chunk, s0, s1, tag = arg
    S = {"obs": [], "tgt": [], "uns": [], "hsz": [], "game": []}
    for seed in range(s0, s1):
        sim = BeliefSim([None] * 4, seed=seed, quan=0, learner_seats=[], cnn=True)
        sim._S = S; sim._gid = seed
        sim.play()
    d = os.path.join(OUTDIR, tag); os.makedirs(d, exist_ok=True)
    outp = os.path.join(d, f"shard_{chunk:04d}.npz")
    np.savez(outp,
             obs=np.asarray(S["obs"], np.int8), tgt=np.asarray(S["tgt"], np.int8),
             uns=np.asarray(S["uns"], np.int8), hsz=np.asarray(S["hsz"], np.int8),
             game=np.asarray(S["game"], np.int32))
    return chunk, len(S["obs"]), s1 - s0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30000)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=5_000_000)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--shard_games", type=int, default=250)   # small shards -> incremental writes
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
    nsamp = sum(r[1] for r in res); ngame = sum(r[2] for r in res)
    import json
    manifest = dict(tag=a.tag, games=ngame, samples=nsamp, seed0=a.seed0,
                    seed_range=[a.seed0, a.seed0 + a.games], shards=len(tasks),
                    kd=KD, seconds=round(time.time() - t0, 1),
                    per_game_samples=round(nsamp / max(1, ngame), 2))
    json.dump(manifest, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    print(f"GEN DONE tag={a.tag} games={ngame} samples={nsamp} "
          f"({nsamp/max(1,ngame):.1f}/game) {time.time()-t0:.0f}s -> {d}", flush=True)


if __name__ == "__main__":
    main()
