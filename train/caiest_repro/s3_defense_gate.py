"""
s3_defense_gate.py -- DEPLOYABLE DEFENSIVE DISCARD POLICY gate vs kdens3 (CPU-only).

Base = kdens3 (3-net ensemble deploy rule, mean-softmax-over-legal), UNCHANGED for
chi/peng/gang/hu/pass and single-legal-discard decisions. On a discard decision with
>1 legal discard:
  - r = per-candidate deal-in prob of kdens3's argmax tile (pc ensemble, mean of 3 sigmoids)
  - if r <= tau: keep kdens3 argmax (danger below threshold -> offense intact)
  - else: among kdens3's TOP-K discards by policy prob, pick the LOWEST deal-in tile;
          override to it ONLY if strictly safer than the argmax (a safer alt exists).

This only ever diverts among kdens3's already-good tiles when danger is high.

NULL-CAL: --null runs the FULL override code with tau=1.0 -> r<=1.0 always -> never
override -> reproduces kdens3 exactly -> paired placement 2.5000 (and 0 overrides).
The gate is the arbiter; AUROC does not imply better placement.

Deal-in model = ckpt/dealin_pc/dealin_pc_s{0,1,2}.pt (DealInFused, 39-plane: 38 obs +
1 one-hot candidate-tile plane at ((a-2)//9,(a-2)%9)).

Paired duplicate placement, defensive seat vs 3x kdens3, 4 seat-rotations, scoring
identical to e12_cond_gate / s2_oracle_gate. Fresh disjoint walls: block b -> seed0 =
9_000_000 + b*3000.

  python3 s3_defense_gate.py --blocks 0 --seeds 200 --null --out /tmp/def_nullcal.json
  python3 s3_defense_gate.py --blocks 0,1,2,3 --seeds 500 --tau 0.5 --K 3 --workers 60 --out results/x.json
"""
import os, sys, json, argparse, time, math
sys.path.insert(0, "/root/caiest_repro")
import numpy as np, multiprocessing as mp
import torch
torch.set_num_threads(1)
import models_explore
from sim_cnn import Sim, ACT, TILE_LIST
from dealin_pc_train import DealInFused

KD = ["ckpt/kd/kd_128x40_s0.pkl", "ckpt/kd/kd_128x40_s1.pkl", "ckpt/kd/kd_128x40_s2.pkl"]
PC = ["ckpt/dealin_pc/dealin_pc_s0.pt", "ckpt/dealin_pc/dealin_pc_s1.pt",
      "ckpt/dealin_pc/dealin_pc_s2.pt"]
_G = {}


def _init_worker():
    models_explore.IN_PLANES = 38
    from models_explore import build
    kd = []
    for p in KD:
        m = build("resbn_fused", channels=128, blocks=40)
        m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); kd.append(m)
    _G["kd"] = kd
    pc = []
    for p in PC:
        m = DealInFused(128, 40)
        m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); pc.append(m)
    _G["pc"] = pc


def _ens_logits(obs, mask):
    """kdens3 deploy rule: mean softmax over legal, log(avg) (matches e12/s1/s2)."""
    mk = mask.flatten().astype(np.float32)
    acc = None
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
    avg = acc / len(_G["kd"])
    return np.log(np.where(avg > 0, avg, 1e-30))


def _dealin(obs38, acts):
    """Per-candidate deal-in prob (mean of 3 pc sigmoids) for each action in `acts`."""
    K = len(acts)
    x = np.zeros((K, 39, 4, 9), np.float32)
    x[:, :38] = np.ascontiguousarray(obs38).astype(np.float32)[None]
    for i, a in enumerate(acts):
        idx = a - ACT["Play"]
        x[i, 38, idx // 9, idx % 9] = 1.0
    xt = torch.from_numpy(x)
    acc = None
    for m in _G["pc"]:
        with torch.no_grad():
            p = torch.sigmoid(m(xt)).numpy().reshape(-1)
        acc = p if acc is None else acc + p
    return acc / len(_G["pc"])


def _placement(scores, seat):
    c = scores[seat]
    greater = sum(1 for j in range(4) if scores[j] > c)
    equal = sum(1 for j in range(4) if scores[j] == c)
    return 5.0 - (greater + (equal + 1) / 2.0)


class DefSim(Sim):
    search_seat = 0
    null = False
    tau = 0.5
    K = 3

    def _kd_ask(self, seat):
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"], None, None, None
        lg = _ens_logits(obs[None, :], mask[None, :])
        act = int(lg.argmax())
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        return act, lg, mask, obs

    def _ask(self, seat):
        kd_act, kd_lg, mask, obs = self._kd_ask(seat)
        if seat != self.search_seat or self.null or kd_lg is None:
            return kd_act
        if not (ACT["Play"] <= kd_act < ACT["Chi"]):
            return kd_act
        legal = [a for a in range(ACT["Play"], ACT["Chi"]) if mask[a]]
        if len(legal) <= 1:
            return kd_act
        self._decisions = getattr(self, "_decisions", 0) + 1
        # kdens3 top-K discards by policy prob (kd_act is top-1)
        legal_sorted = sorted(legal, key=lambda a: kd_lg[a], reverse=True)
        k = len(legal_sorted) if self.K <= 0 else min(self.K, len(legal_sorted))
        cand = legal_sorted[:k]
        if kd_act not in cand:                 # safety: always include the argmax
            cand = [kd_act] + cand
        di = _dealin(obs, cand)
        risk = float(di[cand.index(kd_act)])
        if risk <= self.tau:                   # danger below threshold -> keep offense
            return kd_act
        j = int(np.argmin(di))
        if di[j] < risk - 1e-12:               # strictly safer alternative exists
            self._override = getattr(self, "_override", 0) + 1
            self._risk_saved = getattr(self, "_risk_saved", 0.0) + (risk - float(di[j]))
            return cand[j]
        return kd_act


def _work(arg):
    block, seed, tau, K, null = arg
    psum = 0.0; ov = dec = 0
    for cs in range(4):
        sim = DefSim([None] * 4, seed=seed, quan=0, learner_seats=[], cnn=True)
        sim.search_seat = cs; sim.null = null; sim.tau = tau; sim.K = K
        sim.play()
        psum += _placement(sim.scores, cs)
        ov += getattr(sim, "_override", 0); dec += getattr(sim, "_decisions", 0)
    return block, seed, psum, ov, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="0,1,2,3")
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=3)          # K<=0 -> all legal discards
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    blocks = [int(x) for x in a.blocks.split(",")]
    tasks = []; seedmap = {}
    for b in blocks:
        s0 = 9_000_000 + b * 3000
        seedmap[str(b)] = [s0, s0 + a.seeds]
        tasks += [(b, s, a.tau, a.K, a.null) for s in range(s0, s0 + a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init_worker) as p:
        res = p.map(_work, tasks, chunksize=1)
    per_block = {b: [] for b in blocks}; tot_ov = tot_dec = 0
    for b, s, psum, ov, dec in res:
        per_block[b].append(psum / 4.0); tot_ov += ov; tot_dec += dec
    block_means = {b: float(np.mean(per_block[b])) for b in blocks}
    bm = np.array([block_means[b] for b in blocks], dtype=np.float64)
    n = len(bm)
    block_mean = float(bm.mean())
    block_sd = float(bm.std(ddof=1)) if n > 1 else 0.0
    se = block_sd / math.sqrt(n) if n > 1 else 0.0
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 8: 2.365}.get(n, 1.96)
    ci = tcrit * se
    lo, hi = block_mean - ci, block_mean + ci
    if a.null:
        verdict = ("NULL-CAL OK (== kdens3, 2.5000, 0 overrides)"
                   if abs(block_mean - 2.5) < 1e-9 and tot_ov == 0
                   else f"NULL-CAL BROKEN (dev {block_mean-2.5:+.8f}, ov={tot_ov}) -- DEBUG")
    elif lo > 2.5:
        verdict = "DEFENSE BEATS kdens3 (CI>2.5) -- first deployable champion-beater"
    elif hi < 2.5:
        verdict = "DEFENSE LOSES to kdens3 (CI<2.5) -- danger prediction != better placement"
    else:
        verdict = "DEFENSE TIES kdens3 (CI spans 2.5) -- no clear placement gain"
    out = dict(
        experiment="deployable defensive discard policy (pc deal-in, top-K safe divert) vs kdens3",
        null_run=bool(a.null), tau=a.tau, K=a.K, kd=KD, pc=PC,
        blocks=blocks, seeds_per_block=a.seeds, seed_ranges=seedmap,
        n_blocks=n, n_seeds=len(tasks), n_games=len(tasks) * 4,
        block_means={str(b): round(block_means[b], 4) for b in blocks},
        block_mean_placement=round(block_mean, 6),
        block_sd=round(block_sd, 5), ci95_halfwidth=round(ci, 5),
        ci95=[round(lo, 5), round(hi, 5)],
        override_fraction=round(tot_ov / max(1, tot_dec), 4),
        n_discard_decisions=tot_dec, n_overrides=tot_ov,
        verdict=verdict, seconds=round(time.time() - t0, 1), workers=a.workers)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
