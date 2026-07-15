"""
s4_defense_value_gate.py -- VALUE-AWARE ACTION-VALUE defensive discard policy vs kdens3.

Synthesis of the two failures:
  * Stage-1 offense search (value-head hand-max, blind to defense) -> 2.026 (lost)
  * naive defense override (deal-in min, blind to offense-EV)       -> ~2.4995 (tied/lost)
Here we combine BOTH signals into one 1-ply action-value and pick argmax:

  A(T) = V_ens(post-discard state, src)  -  lambda * P_dealin(state, T) * L

  V_ens  = mean of 5 good source-conditioned value heads (score/SC units), src=DEPLOY_SRC
  P_dealin = mean of 3 per-candidate deal-in heads (validated per-tile P(Ron))
  L      = deal-in loss in value units = (8 + avg_fan)/SC = (8+12.5)/30 = 0.6834
           (avg winner fan = 12.5 from the Final2 corpus; SC=30 from f2_value_v2)

Base = kdens3, UNCHANGED for non-discard / single-legal-discard. On a >1-legal discard,
candidates = kdens3 TOP-K discards by policy prob; pick argmax A(T), tie-break -> kdens3
log-prob (so a null evaluator reproduces kdens3).

NULL-CAL (--null): V_ens forced to 0 AND lambda=0 -> A==0 for all candidates -> tie-break
-> kdens3 argmax -> paired placement 2.5000, 0 overrides (exercises the full selection path).
lambda=0 (non-null) = pure value-max among top-K = Stage-1 cross-check (Stage-1 full-legal=2.026).

kdens3 on CPU; value + deal-in ensembles on GPU (round-robin). Paired duplicate placement,
4 blocks x 500 seeds x 4 rotations, disjoint walls seed0 = 9_500_000 + block*3000.

  python3 s4_defense_value_gate.py --blocks 0 --seeds 200 --null --workers 32 --out /tmp/dv_null.json
  python3 s4_defense_value_gate.py --blocks 0,1,2,3 --seeds 500 --lam 1 --K 3 --workers 40 --out results/x.json
"""
import os, sys, json, argparse, time, copy, math
sys.path.insert(0, "/root/caiest_repro")
import numpy as np, torch, multiprocessing as mp
import models_explore
from sim_cnn import Sim, ACT, TILE_LIST
from dealin_pc_train import DealInFused

torch.set_num_threads(1)
KD = ["ckpt/kd/kd_128x40_s0.pkl", "ckpt/kd/kd_128x40_s1.pkl", "ckpt/kd/kd_128x40_s2.pkl"]
VALUE = ["results/VALUE_C_60K.pt", "results/VALUE_C_60K_s1.pt", "results/VALUE_C_60K_s3.pt",
         "results/VALUE_C_60K_s4.pt", "results/VALUE_C_60K_s6.pt"]
PC = ["ckpt/dealin_pc/dealin_pc_s0.pt", "ckpt/dealin_pc/dealin_pc_s1.pt",
      "ckpt/dealin_pc/dealin_pc_s2.pt"]
DEPLOY_SRC = 0
L_DEALIN = (8 + 12.5) / 30.0          # 0.6834 value units
_G = {}


def _init_worker(null, ngpu):
    ident = mp.current_process()._identity
    idx = (ident[0] - 1) if ident else 0
    dev = f"cuda:{idx % ngpu}" if ngpu > 0 else "cpu"
    torch.cuda.set_device(idx % ngpu) if ngpu > 0 else None
    _G["dev"] = dev; _G["null"] = null
    models_explore.IN_PLANES = 38
    from models_explore import build
    kd = []
    for p in KD:
        m = build("resbn_fused", channels=128, blocks=40)
        m.load_state_dict(torch.load(p, map_location="cpu")); m.eval(); kd.append(m)
    _G["kd"] = kd
    from f2_value_v2 import VNet
    vs = []
    for p in VALUE:
        n = VNet(cond=True); n.load_state_dict(torch.load(p, map_location="cpu"))
        n.eval().to(dev); vs.append(n)
    _G["vnet"] = vs
    pc = []
    for p in PC:
        m = DealInFused(128, 40); m.load_state_dict(torch.load(p, map_location="cpu"))
        m.eval().to(dev); pc.append(m)
    _G["pc"] = pc


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
    avg = acc / len(_G["kd"])
    return np.log(np.where(avg > 0, avg, 1e-30))


def _value_ens(obs_arr):
    if _G["null"]:
        return np.zeros(len(obs_arr), np.float32)
    dev = _G["dev"]
    ob = torch.from_numpy(np.ascontiguousarray(obs_arr)).float().to(dev)
    sb = torch.full((len(obs_arr),), DEPLOY_SRC, dtype=torch.long, device=dev)
    acc = None
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(dev != "cpu")):
        for n in _G["vnet"]:
            v = n(ob, sb).float().cpu().numpy()
            acc = v if acc is None else acc + v
    return acc / len(_G["vnet"])


def _dealin_ens(obs38, acts):
    K = len(acts); x = np.zeros((K, 39, 4, 9), np.float32)
    x[:, :38] = np.ascontiguousarray(obs38).astype(np.float32)[None]
    for i, a in enumerate(acts):
        idx = a - ACT["Play"]; x[i, 38, idx // 9, idx % 9] = 1.0
    xt = torch.from_numpy(x).to(_G["dev"]); acc = None
    with torch.no_grad():
        for m in _G["pc"]:
            p = torch.sigmoid(m(xt)).float().cpu().numpy().reshape(-1)
            acc = p if acc is None else acc + p
    return acc / len(_G["pc"])


def _placement(scores, seat):
    c = scores[seat]
    greater = sum(1 for j in range(4) if scores[j] > c)
    equal = sum(1 for j in range(4) if scores[j] == c)
    return 5.0 - (greater + (equal + 1) / 2.0)


class DVSim(Sim):
    search_seat = 0
    lam = 1.0
    K = 3

    def _ask(self, seat):
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        kd_lg = _ens_logits(obs[None, :], mask[None, :])
        kd_act = int(kd_lg.argmax())
        if not mask[kd_act]:
            kd_act = int(np.flatnonzero(mask)[0])
        if seat != self.search_seat:
            return kd_act
        if not (ACT["Play"] <= kd_act < ACT["Chi"]):
            return kd_act
        legal = [a for a in range(ACT["Play"], ACT["Chi"]) if mask[a]]
        if len(legal) <= 1:
            return kd_act
        self._decisions = getattr(self, "_decisions", 0) + 1
        legal_sorted = sorted(legal, key=lambda a: kd_lg[a], reverse=True)
        k = len(legal_sorted) if self.K <= 0 else min(self.K, len(legal_sorted))
        cand = legal_sorted[:k]
        if kd_act not in cand:
            cand = [kd_act] + cand
        post = np.stack([self._post_obs(seat, TILE_LIST[a - ACT["Play"]]) for a in cand])
        V = _value_ens(post)
        P = _dealin_ens(obs, cand)
        A = V - self.lam * P * L_DEALIN
        best = max(range(len(cand)), key=lambda i: (float(A[i]), float(kd_lg[cand[i]])))
        best_act = cand[best]
        if best_act != kd_act:
            self._override = getattr(self, "_override", 0) + 1
        return best_act

    def _post_obs(self, seat, tile):
        a = copy.deepcopy(self.cai[seat])
        a.request2obs(f"Player {seat} Play {tile}")
        return a.obs.reshape(38, 4, 9).astype(np.float32)


def _work(arg):
    block, seed, lam, K, null = arg
    psum = 0.0; ov = dec = 0
    for cs in range(4):
        sim = DVSim([None] * 4, seed=seed, quan=0, learner_seats=[], cnn=True)
        sim.search_seat = cs; sim.lam = (0.0 if null else lam); sim.K = K
        sim.play()
        psum += _placement(sim.scores, cs)
        ov += getattr(sim, "_override", 0); dec += getattr(sim, "_decisions", 0)
    return block, seed, psum, ov, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="0,1,2,3")
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ngpu = torch.cuda.device_count()
    blocks = [int(x) for x in a.blocks.split(",")]
    tasks = []; seedmap = {}
    for b in blocks:
        s0 = 9_500_000 + b * 3000
        seedmap[str(b)] = [s0, s0 + a.seeds]
        tasks += [(b, s, a.lam, a.K, a.null) for s in range(s0, s0 + a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init_worker, initargs=(a.null, ngpu)) as p:
        res = p.map(_work, tasks, chunksize=1)
    per_block = {b: [] for b in blocks}; tot_ov = tot_dec = 0
    for b, s, psum, ov, dec in res:
        per_block[b].append(psum / 4.0); tot_ov += ov; tot_dec += dec
    block_means = {b: float(np.mean(per_block[b])) for b in blocks}
    bm = np.array([block_means[b] for b in blocks], dtype=np.float64)
    n = len(bm); block_mean = float(bm.mean())
    block_sd = float(bm.std(ddof=1)) if n > 1 else 0.0
    se = block_sd / math.sqrt(n) if n > 1 else 0.0
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 8: 2.365}.get(n, 1.96)
    ci = tcrit * se; lo, hi = block_mean - ci, block_mean + ci
    if a.null:
        verdict = ("NULL-CAL OK (== kdens3, 2.5000, 0 overrides)"
                   if abs(block_mean - 2.5) < 1e-9 and tot_ov == 0
                   else f"NULL-CAL BROKEN (dev {block_mean-2.5:+.8f}, ov={tot_ov}) -- DEBUG")
    elif lo > 2.5:
        verdict = "BEATS kdens3 (CI>2.5) -- FIRST DEPLOYABLE CHAMPION-BEATER"
    elif hi < 2.5:
        verdict = "LOSES to kdens3 (CI<2.5)"
    else:
        verdict = "TIES kdens3 (CI spans 2.5)"
    out = dict(experiment="value-aware action-value defense A=V-lam*P_dealin*L vs kdens3",
               null_run=bool(a.null), lam=a.lam, K=a.K, L_dealin=L_DEALIN, deploy_src=DEPLOY_SRC,
               SC=30.0, avg_fan=12.5, kd=KD, value=VALUE, pc=PC, ngpu=ngpu,
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
