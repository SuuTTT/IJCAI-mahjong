"""
s2_oracle_gate.py -- ORACLE-ROLLOUT 1-ply discard search diagnostic (CPU-only).

Answers: is discard-search worth pursuing at all? At each search-seat discard
decision (kdens3 argmax is a Play with >1 legal discard), for each legal discard:
deep-copy the FULL sim (TRUE hidden state -- oracle upper bound, NOT deployable),
force that discard, continue to hand-end with kdens3 in ALL seats, record the
search seat's (placement, score). Pick the discard maximizing (placement, score,
prefers-kdens3-argmax). Rollout is deterministic (kdens3 argmax) -> 1 rollout/candidate.

Null control (--null): search seat returns kdens3 argmax, no rollout -> reproduces
kdens3 exactly -> paired placement 2.5000 (harness soundness proof).

Gate: paired duplicate placement, search seat vs 3x kdens3, 4 seat-rotations, same
scoring as e12_cond_gate.py / s1_search_gate.py. Fresh disjoint walls:
block b -> seed0 = 8_500_000 + b*3000.

  python3 s2_oracle_gate.py --blocks 0,1 --seeds 250 --workers 60 --out results/SEARCH_ORACLE.json
  python3 s2_oracle_gate.py --blocks 0 --seeds 120 --null --out /tmp/oracle_nullcal.json
"""
import os, sys, json, argparse, time, copy, math
sys.path.insert(0, "/root/caiest_repro")
import numpy as np, multiprocessing as mp
import torch
torch.set_num_threads(1)
import models_explore
from sim_cnn import Sim, ACT, TILE_LIST

KD = ["ckpt/kd/kd_128x40_s0.pkl", "ckpt/kd/kd_128x40_s1.pkl", "ckpt/kd/kd_128x40_s2.pkl"]
MAX_LEGAL = 14            # cap: skip search on rare >14-legal-discard decisions
_G = {}


def _init_worker():
    models_explore.IN_PLANES = 38
    from models_explore import build
    kd = []
    for p in KD:
        m = build("resbn_fused", channels=128, blocks=40)
        m.load_state_dict(torch.load(p, map_location="cpu")); m.eval()
        kd.append(m)
    _G["kd"] = kd


def _ens_logits(obs, mask):
    """kdens3 deploy rule: mean softmax over legal, log(avg) (matches e12/s1)."""
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


def _placement(scores, seat):
    c = scores[seat]
    greater = sum(1 for j in range(4) if scores[j] > c)
    equal = sum(1 for j in range(4) if scores[j] == c)
    return 5.0 - (greater + (equal + 1) / 2.0)


class OracleSim(Sim):
    search_seat = 0
    null = False

    def _kd_ask(self, seat):
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"], None, None
        lg = _ens_logits(obs[None, :], mask[None, :])
        act = int(lg.argmax())
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        return act, lg, mask

    def _ask(self, seat):
        self._asks = getattr(self, "_asks", 0) + 1
        kd_act, kd_lg, mask = self._kd_ask(seat)
        if seat != self.search_seat or self.null or kd_lg is None:
            return kd_act
        if not (ACT["Play"] <= kd_act < ACT["Chi"]):
            return kd_act
        legal = [a for a in range(ACT["Play"], ACT["Chi"]) if mask[a]]
        if len(legal) <= 1:
            return kd_act
        self._decisions = getattr(self, "_decisions", 0) + 1
        if len(legal) > MAX_LEGAL:
            self._skipped = getattr(self, "_skipped", 0) + 1
            return kd_act
        best_key = None; best_act = kd_act
        for a in legal:
            tile = TILE_LIST[a - ACT["Play"]]
            clone = copy.deepcopy(self)
            clone.search_seat = -1          # pure kdens3 continuation, no recursion
            clone._asks = 0
            place, score = clone._rollout_discard(seat, tile)
            self._depth_sum = getattr(self, "_depth_sum", 0) + clone._asks
            self._depth_n = getattr(self, "_depth_n", 0) + 1
            key = (place, score, 1 if a == kd_act else 0)   # ties -> prefer kdens3 argmax
            if best_key is None or key > best_key:
                best_key = key; best_act = a
        if best_act != kd_act:
            self._override = getattr(self, "_override", 0) + 1
        return best_act

    def _rollout_discard(self, seat, tile):
        """self is the clone (search_seat=-1). Force `seat` to play `tile`, continue
        to hand-end with kdens3, return (placement, score) for `seat`."""
        if tile not in self.hand[seat]:
            tile = self.hand[seat][0]
        self.hand[seat].remove(tile)
        self._broadcast(f"Player {seat} Play {tile}")
        nxt = self._resolve_claims(tile, seat)
        if nxt != "HU":
            self.cur = nxt
            self._loop(300)
        return _placement(self.scores, seat), self.scores[seat]


def _work(arg):
    block, seed = arg
    psum = 0.0
    ov = dec = dsum = dn = skip = 0
    for cs in range(4):
        sim = OracleSim([None] * 4, seed=seed, quan=0, learner_seats=[], cnn=True)
        sim.search_seat = cs
        sim.null = _G.get("null", False)
        sim.play()
        psum += _placement(sim.scores, cs)
        ov += getattr(sim, "_override", 0); dec += getattr(sim, "_decisions", 0)
        dsum += getattr(sim, "_depth_sum", 0); dn += getattr(sim, "_depth_n", 0)
        skip += getattr(sim, "_skipped", 0)
    return block, seed, psum, ov, dec, dsum, dn, skip


def _init_null(null):
    _init_worker()
    _G["null"] = null


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="0,1")
    ap.add_argument("--seeds", type=int, default=250)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    blocks = [int(x) for x in a.blocks.split(",")]
    tasks = []; seedmap = {}
    for b in blocks:
        s0 = 8_500_000 + b * 3000
        seedmap[str(b)] = [s0, s0 + a.seeds]
        tasks += [(b, s) for s in range(s0, s0 + a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init_null, initargs=(a.null,)) as p:
        res = p.map(_work, tasks, chunksize=1)
    per_block = {b: [] for b in blocks}
    tot_ov = tot_dec = tot_dsum = tot_dn = tot_skip = 0
    for b, s, psum, ov, dec, dsum, dn, skip in res:
        per_block[b].append(psum / 4.0)
        tot_ov += ov; tot_dec += dec; tot_dsum += dsum; tot_dn += dn; tot_skip += skip
    block_means = {b: float(np.mean(per_block[b])) for b in blocks}
    bm = np.array([block_means[b] for b in blocks], dtype=np.float64)
    n = len(bm)
    block_mean = float(bm.mean())
    block_sd = float(bm.std(ddof=1)) if n > 1 else 0.0
    se = block_sd / math.sqrt(n) if n > 1 else 0.0
    tcrit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 8: 2.365, 12: 2.201}.get(n, 1.96)
    ci = tcrit * se
    lo, hi = block_mean - ci, block_mean + ci
    if a.null:
        verdict = ("NULL-CAL OK (== kdens3, 2.5000)" if abs(block_mean - 2.5) < 1e-6
                   else f"NULL-CAL BROKEN (dev {block_mean - 2.5:+.6f}) -- DEBUG")
    elif lo > 2.5:
        verdict = ("ORACLE ROLLOUT BEATS kdens3 (CI>2.5) -> discard-search has headroom; "
                   "build deployable PIMC next")
    elif hi < 2.5:
        verdict = ("ORACLE ROLLOUT LOSES to kdens3 (CI<2.5) -> even perfect-info 1-ply "
                   "discard search under kdens3 continuation cannot beat it; PIVOT")
    else:
        verdict = ("ORACLE ROLLOUT TIES kdens3 (CI spans 2.5) -> no clear headroom from "
                   "1-ply discard search; PIVOT or deeper search")
    out = dict(
        experiment="oracle-rollout 1-ply discard search (true-state, kdens3 continuation) vs kdens3",
        oracle_note="uses TRUE hidden state in rollouts = non-deployable upper bound / ceiling",
        null_run=bool(a.null), kd=KD, blocks=blocks, seeds_per_block=a.seeds,
        seed_ranges=seedmap, n_blocks=n, n_seeds=len(tasks), n_games=len(tasks) * 4,
        block_means={str(b): round(block_means[b], 4) for b in blocks},
        block_mean_placement=round(block_mean, 4),
        block_sd=round(block_sd, 4), ci95_halfwidth=round(ci, 4),
        ci95=[round(lo, 4), round(hi, 4)],
        override_fraction=round(tot_ov / max(1, tot_dec), 4),
        n_search_decisions=tot_dec, n_overrides=tot_ov, n_skipped_gt_maxlegal=tot_skip,
        mean_rollout_depth=round(tot_dsum / max(1, tot_dn), 1),
        verdict=verdict, seconds=round(time.time() - t0, 1), workers=a.workers)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
