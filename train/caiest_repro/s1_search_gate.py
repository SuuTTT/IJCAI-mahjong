"""
s1_search_gate.py -- Stage-1 test-time-search placement gate (2027 flagship pilot).

Search policy = kdens3 deploy rule EVERYWHERE, EXCEPT: on a discard decision (the
kdens3 argmax is a Play action) with >1 legal discard, pick the discard tile whose
post-discard own-state is scored highest by the source-conditioned value head
(VALUE_C_60K.pt), batched on GPU. Ties in value are broken toward the kdens3
preferred discard -> with a constant/null value head the search reproduces kdens3
EXACTLY (null-value calibration -> placement 2.5000).

Gate: paired duplicate placement (SAME structure as e12_cond_gate.py): search seat vs
kdens3 in the other three, rotate through 4 seats, average placement (5-(greater+(equal+1)/2)).

Disjoint fresh walls: block b -> seed0 = 8_000_000 + b*3000, 500 seeds/block.

  python3 s1_search_gate.py --blocks 0,1,2,3 --workers 32 --out results/SEARCH_PILOT.json
  python3 s1_search_gate.py --blocks 0 --seeds 200 --null --out results/SEARCH_NULLCAL.json
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,2,3,7")   # physical 0,2,3,7 -> cuda:0..3
import sys, json, argparse, time, copy, math
sys.path.insert(0, "/root/caiest_repro")
import numpy as np, torch, multiprocessing as mp
import models_explore
from sim_cnn import Sim, ACT, TILE_LIST

torch.set_num_threads(1)

KD = ["ckpt/kd/kd_128x40_s0.pkl", "ckpt/kd/kd_128x40_s1.pkl", "ckpt/kd/kd_128x40_s2.pkl"]
VALUE_PT = "results/VALUE_C_60K.pt"
DEPLOY_SRC = 0            # a Final2 finalist source id (0-3); NOT official (4)
N_GPU = 4

_G = {}

def _init_worker(null_value):
    ident = mp.current_process()._identity
    idx = (ident[0] - 1) if ident else 0
    dev_ord = idx % N_GPU
    torch.cuda.set_device(dev_ord)
    _G["dev"] = f"cuda:{dev_ord}"
    _G["null"] = null_value
    models_explore.IN_PLANES = 38
    from models_explore import build
    kd = []
    for p in KD:
        m = build("resbn_fused", channels=128, blocks=40)
        m.load_state_dict(torch.load(p, map_location="cpu")); m.eval()
        kd.append(m)
    _G["kd"] = kd
    if not null_value:
        from f2_value_v2 import VNet
        net = VNet(cond=True)
        net.load_state_dict(torch.load(VALUE_PT, map_location="cpu"))
        net.eval().to(_G["dev"])
        _G["vnet"] = net

def _ens_logits(obs, mask):
    """kdens3 deploy: mean softmax over legal, returned as log(avg) (matches e12)."""
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

def _value_batch(obs_arr):
    if _G["null"]:
        return np.zeros(len(obs_arr), np.float32)
    dev = _G["dev"]; net = _G["vnet"]
    ob = torch.from_numpy(np.ascontiguousarray(obs_arr)).float().to(dev)
    sb = torch.full((len(obs_arr),), DEPLOY_SRC, dtype=torch.long, device=dev)
    with torch.no_grad(), torch.cuda.amp.autocast():
        return net(ob, sb).float().cpu().numpy()

class SearchSim(Sim):
    search_seat = 0
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
        # only override a genuine discard decision with >1 legal discard
        if not (ACT["Play"] <= kd_act < ACT["Chi"]):
            return kd_act
        legal = [a for a in range(ACT["Play"], ACT["Chi"]) if mask[a]]
        if len(legal) <= 1:
            return kd_act
        cand = np.stack([self._post_obs(seat, TILE_LIST[a - ACT["Play"]]) for a in legal])
        vals = _value_batch(cand)
        # primary: value (higher better); tie-break: kdens3 log-prob (-> null == kdens3)
        best = max(range(len(legal)), key=lambda i: (float(vals[i]), float(kd_lg[legal[i]])))
        return legal[best]
    def _post_obs(self, seat, tile):
        a = copy.deepcopy(self.cai[seat])
        a.request2obs(f"Player {seat} Play {tile}")
        return a.obs.reshape(38, 4, 9).astype(np.float32)

def _work(arg):
    block, seed = arg
    psum = 0.0
    for cs in range(4):
        sim = SearchSim([None] * 4, seed=seed, quan=0, learner_seats=[], cnn=True)
        sim.search_seat = cs
        sim.play()
        sc = sim.scores; c = sc[cs]
        greater = sum(1 for j in range(4) if sc[j] > c)
        equal = sum(1 for j in range(4) if sc[j] == c)
        psum += 5.0 - (greater + (equal + 1) / 2.0)
    return block, seed, psum

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="0,1,2,3")
    ap.add_argument("--seeds", type=int, default=500)     # seeds per block
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    blocks = [int(x) for x in a.blocks.split(",")]
    tasks = []
    seedmap = {}
    for b in blocks:
        s0 = 8_000_000 + b * 3000
        seedmap[b] = [s0, s0 + a.seeds]
        for s in range(s0, s0 + a.seeds):
            tasks.append((b, s))
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init_worker, initargs=(a.null,)) as p:
        res = p.map(_work, tasks, chunksize=1)
    # aggregate per block (placement per game = psum/4)
    per_block = {b: [] for b in blocks}
    for b, s, psum in res:
        per_block[b].append(psum / 4.0)
    block_means = {b: float(np.mean(per_block[b])) for b in blocks}
    all_games = [v for b in blocks for v in per_block[b]]
    overall_game_mean = float(np.mean(all_games))
    bm = np.array([block_means[b] for b in blocks], dtype=np.float64)
    n = len(bm)
    block_mean = float(bm.mean())
    block_sd = float(bm.std(ddof=1)) if n > 1 else 0.0
    se = block_sd / math.sqrt(n) if n > 1 else 0.0
    # 95% CI across blocks (t for small n)
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
             8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201}.get(n, 1.96)
    ci = tcrit * se
    lo, hi = block_mean - ci, block_mean + ci
    if not a.null:
        if lo > 2.5:
            verdict = "BEATS kdens3 (95% CI above 2.500)"
        elif hi < 2.5:
            verdict = "LOSES to kdens3 (95% CI below 2.500)"
        else:
            verdict = "TIES kdens3 (95% CI includes 2.500)"
    else:
        verdict = ("NULL-CAL OK (== kdens3, 2.5000)" if abs(block_mean - 2.5) < 1e-6
                   else f"NULL-CAL BROKEN (dev {block_mean-2.5:+.6f}) -- DEBUG HARNESS")
    out = dict(
        experiment="Stage1 1-ply value-guided discard vs kdens3, paired duplicate placement",
        null_value_run=bool(a.null), deploy_src=DEPLOY_SRC, value_head=VALUE_PT, kd=KD,
        blocks=blocks, seeds_per_block=a.seeds, seed_ranges=seedmap,
        n_blocks=n, n_games=len(all_games) * 1,
        block_means={str(b): round(block_means[b], 4) for b in blocks},
        block_mean_placement=round(block_mean, 4),
        overall_game_mean_placement=round(overall_game_mean, 4),
        block_sd=round(block_sd, 4), ci95_halfwidth=round(ci, 4),
        ci95=[round(lo, 4), round(hi, 4)], verdict=verdict,
        seconds=round(time.time() - t0, 1), workers=a.workers)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
