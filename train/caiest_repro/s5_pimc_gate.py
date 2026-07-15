"""
s5_pimc_gate.py -- PIMC (Perfect-Information Monte Carlo) discard search vs kdens3.

Deployable version of the oracle (s2). At each search-seat discard decision, sample
N determinized worlds (resample the tiles hidden from the search seat -- opponents'
concealed hands + walls -- keeping sizes; search-seat hand + all melds + all discards
fixed), evaluate EVERY candidate discard on the SAME N worlds (common random numbers
= paired, low variance), average the search seat's placement over worlds, pick best.

Calibration gates (run these first, in order):
  --null        : search seat returns kdens3 argmax, no rollout -> MUST be 2.5000.
  --true_state  : skip determinization (1 world = truth) -> MUST reproduce the oracle
                  (~3.5); proves the PIMC scaffold is the oracle when beliefs are perfect.
Then the real gate: N>1, determinization on.

  python3 s5_pimc_gate.py --blocks 0 --seeds 120 --null --out /tmp/pimc_null.json
  python3 s5_pimc_gate.py --blocks 0 --seeds 60 --true_state --n_worlds 1 --out /tmp/pimc_oracle_xcheck.json
  python3 s5_pimc_gate.py --blocks 0,1 --seeds 120 --n_worlds 100 --workers 40 --out results/PIMC_GATE.json

Determinization is UNIFORM in v1 (opponent-belief-weighted sampling = v2).
Walls: block b -> seed0 = 9_800_000 + b*3000 (fresh region, disjoint from oracle/defense).
"""
import os, sys, json, argparse, time, copy, math
sys.path.insert(0, "/root/caiest_repro")
import numpy as np, multiprocessing as mp
import torch
torch.set_num_threads(1)
import models_explore
from sim_cnn import Sim, ACT, TILE_LIST

KD = ["ckpt/kd/kd_128x40_s0.pkl", "ckpt/kd/kd_128x40_s1.pkl", "ckpt/kd/kd_128x40_s2.pkl"]
MAX_LEGAL = 14
_G = {}


def _init_worker(cfg):
    models_explore.IN_PLANES = 38
    from models_explore import build
    kd = []
    for p in KD:
        m = build("resbn_fused", channels=128, blocks=40)
        m.load_state_dict(torch.load(p, map_location="cpu")); m.eval()
        kd.append(m)
    _G["kd"] = kd
    _G.update(cfg)


def _ens_logits(obs, mask):
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


def _determinize(world, search_seat, rng):
    """Resample tiles hidden from `search_seat`: pool the opponents' concealed hands +
    ALL walls, shuffle, redeal into the same-sized slots. Search seat's hand, all melds,
    all discards, wall COUNTS are untouched -> world stays consistent with observations.
    Each opponent's FeatureAgent hand + obs HAND planes are rebuilt from the new hand."""
    pool, hand_sizes = [], {}
    for s in range(4):
        if s != search_seat:
            hand_sizes[s] = len(world.hand[s])
            pool.extend(world.hand[s])
    wall_sizes = [len(w) for w in world.walls]
    for w in world.walls:
        pool.extend(w)
    rng.shuffle(pool)
    i = 0
    for s in range(4):
        if s != search_seat:
            nh = list(pool[i:i + hand_sizes[s]]); i += hand_sizes[s]
            world.hand[s] = nh
            if world.cai is not None:
                world.cai[s].hand = list(nh)
                world.cai[s]._hand_embedding_update()   # rebuild obs HAND planes from nh
            # data.feature_agent.FeatureAgent (self.agents[s]) tracks its OWN concealed
            # hand and drives the Peng/Gang/Chi tile removals + Chi legality read in
            # _resolve_claims. It MUST be resynced to the resampled hand, or a later GANG
            # runs `hand.remove(tile)` for a tile the STALE dealt hand never held ->
            # `ValueError: x not in list`. (packs/history/shown/wall_counts are PUBLIC and
            # unchanged, so only .hand + its HAND obs planes need rebuilding.)
            if getattr(world, "agents", None) is not None:
                world.agents[s].hand = list(nh)
                world.agents[s]._update_hand()
    for s in range(4):
        world.walls[s] = list(pool[i:i + wall_sizes[s]]); i += wall_sizes[s]


class PIMCSim(Sim):
    search_seat = 0
    null = False
    true_state = False
    n_worlds = 1

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
        nW = 1 if self.true_state else self.n_worlds
        place_sum = {a: 0.0 for a in legal}
        score_sum = {a: 0.0 for a in legal}
        for _ in range(nW):
            world = copy.deepcopy(self)
            world.search_seat = -1
            if not self.true_state:
                _determinize(world, seat, self._rng)
            for a in legal:
                tile = TILE_LIST[a - ACT["Play"]]
                clone = copy.deepcopy(world)
                clone._asks = 0
                place, score = clone._rollout_discard(seat, tile)
                place_sum[a] += place; score_sum[a] += score
                self._depth_sum = getattr(self, "_depth_sum", 0) + clone._asks
                self._depth_n = getattr(self, "_depth_n", 0) + 1
        best_key = None; best_act = kd_act
        for a in legal:
            key = (place_sum[a] / nW, score_sum[a] / nW, 1 if a == kd_act else 0)
            if best_key is None or key > best_key:
                best_key = key; best_act = a
        if best_act != kd_act:
            self._override = getattr(self, "_override", 0) + 1
        return best_act

    def _rollout_discard(self, seat, tile):
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
        sim = PIMCSim([None] * 4, seed=seed, quan=0, learner_seats=[], cnn=True)
        sim.search_seat = cs
        sim.null = _G.get("null", False)
        sim.true_state = _G.get("true_state", False)
        sim.n_worlds = _G.get("n_worlds", 1)
        sim._rng = np.random.RandomState((seed * 4 + cs) % (2**31 - 1))
        sim.play()
        psum += _placement(sim.scores, cs)
        ov += getattr(sim, "_override", 0); dec += getattr(sim, "_decisions", 0)
        dsum += getattr(sim, "_depth_sum", 0); dn += getattr(sim, "_depth_n", 0)
        skip += getattr(sim, "_skipped", 0)
    return block, seed, psum, ov, dec, dsum, dn, skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="0,1")
    ap.add_argument("--seeds", type=int, default=120)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--n_worlds", type=int, default=100)
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--true_state", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    blocks = [int(x) for x in a.blocks.split(",")]
    tasks = []; seedmap = {}
    for b in blocks:
        s0 = 9_800_000 + b * 3000
        seedmap[str(b)] = [s0, s0 + a.seeds]
        tasks += [(b, s) for s in range(s0, s0 + a.seeds)]
    cfg = dict(null=a.null, true_state=a.true_state, n_worlds=a.n_worlds)
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init_worker, initargs=(cfg,)) as p:
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
    mode = "NULL" if a.null else ("TRUE_STATE(oracle xcheck)" if a.true_state else f"PIMC N={a.n_worlds}")
    if a.null:
        verdict = ("NULL-CAL OK (== kdens3, 2.5000)" if abs(block_mean - 2.5) < 1e-6
                   else f"NULL-CAL BROKEN (dev {block_mean - 2.5:+.6f}) -- DEBUG")
    elif a.true_state:
        verdict = (f"TRUE-STATE xcheck = {block_mean:.4f} (should reproduce oracle ~3.5; "
                   "proves PIMC scaffold == oracle when beliefs perfect)")
    elif lo > 2.5:
        verdict = f"PIMC N={a.n_worlds} BEATS kdens3 (CI>2.5) -> first deployable champion-beater"
    elif hi < 2.5:
        verdict = f"PIMC N={a.n_worlds} LOSES (CI<2.5) -> uniform determinization too weak; add belief model / more worlds"
    else:
        verdict = f"PIMC N={a.n_worlds} TIES (CI spans 2.5) -> partial recovery; scale worlds / add belief model"
    out = dict(
        experiment="PIMC 1-ply discard search (uniform determinization, kdens3 rollout) vs kdens3",
        mode=mode, null_run=bool(a.null), true_state=bool(a.true_state), n_worlds=a.n_worlds,
        kd=KD, blocks=blocks, seeds_per_block=a.seeds, seed_ranges=seedmap,
        n_blocks=n, n_seeds=len(tasks), n_games=len(tasks) * 4,
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
