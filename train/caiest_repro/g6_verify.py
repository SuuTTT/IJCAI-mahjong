"""
g6_verify.py -- correctness harness for g6_engine.

modes:
  smoke   : run one exact-mode game (tiny config), confirm it completes + timing.
  deceq   : at the first few REAL search-decision states, compute val_sum two ways
            - s6's ORIGINAL sequential rollout (Sim._loop / Sim._resolve_claims /
              s6.PIMCVSim._rollout_vcut), and
            - g6's LOCKSTEP generator port (_batched_search, exact evaluator),
            with an identical determinization RNG. Require BIT-FOR-BIT identical
            val_sum and best_act. This isolates & proves the generator/lockstep port.
"""
import os, sys, copy, time, argparse
sys.path.insert(0, "/root/caiest_repro")
import numpy as np
import s6_pimc_vcut as s6
from s6_pimc_vcut import _determinize, _placement, TILE_LIST
from sim_cnn import ACT
import g6_engine
from g6_engine import G6Sim, init_infer


class _Stop(Exception):
    pass


def s6_seq_search(sim, seat, legal, kd_act, nW, rng):
    """Verbatim replica of s6.PIMCVSim._ask world loop, using s6's ORIGINAL _rollout_vcut."""
    val_sum = {a: 0.0 for a in legal}
    w = 0; tries = 0
    while w < nW and tries < nW * 6 + 6:
        tries += 1
        world = copy.deepcopy(sim)
        world.search_seat = -1
        _determinize(world, seat, rng)
        try:
            wv = {}
            for a in legal:
                tile = TILE_LIST[a - ACT["Play"]]
                clone = copy.deepcopy(world)
                wv[a] = s6.PIMCVSim._rollout_vcut(clone, seat, tile, sim.k_cutoff)
        except Exception:
            continue
        for a in legal:
            val_sum[a] += wv[a]
        w += 1
    best_key = None; best_act = kd_act
    for a in legal:
        key = (val_sum[a] / w if w else 0.0, 1 if a == kd_act else 0)
        if best_key is None or key > best_key:
            best_key = key; best_act = a
    return val_sum, best_act, w


def run_deceq(seed=9800000, seat=0, nW=8, k=6, ncmp=3):
    init_infer(mode="exact", null=False, device="cpu")
    sim = G6Sim([None] * 4, seed=seed, quan=0, learner_seats=[], cnn=True)
    sim.search_seat = seat; sim.null = False; sim.true_state = False
    sim.n_worlds = nW; sim.k_cutoff = k
    sim._rng = np.random.RandomState((seed * 4 + seat) % (2**31 - 1))

    orig = G6Sim._batched_search
    state = {"n": 0, "fail": 0}

    def wrapped(self, s_, legal, kd_act, n_):
        # snapshot the rng so both searches consume the SAME determinization draws
        rng_a = copy.deepcopy(self._rng)
        rng_b = copy.deepcopy(self._rng)
        # s6 ORIGINAL sequential rollout (Sim._loop / Sim._resolve_claims / _rollout_vcut)
        vs_s6, ba_s6, w_s6 = s6_seq_search(self, s_, legal, kd_act, n_, rng_a)
        # g6 LOCKSTEP generator port
        vs_g6, ba_g6 = _g6_valsum(self, s_, legal, kd_act, k, n_, rng_b)
        # advance the real rng exactly as the search would (rng_a was advanced identically)
        self._rng = rng_a

        match_vs = all(vs_s6[a] == vs_g6[a] for a in legal)
        match_ba = (ba_s6 == ba_g6)
        state["n"] += 1
        print(f"[deceq #{state['n']}] seat={s_} nlegal={len(legal)} w_s6={w_s6} "
              f"kd_act={kd_act} best_s6={ba_s6} best_g6={ba_g6} "
              f"val_match={match_vs} best_match={match_ba}", flush=True)
        if not (match_vs and match_ba):
            state["fail"] += 1
            for a in legal:
                if vs_s6[a] != vs_g6[a]:
                    print(f"    MISMATCH a={a}: s6={vs_s6[a]!r} g6={vs_g6[a]!r} d={vs_s6[a]-vs_g6[a]:.6g}")
        if state["n"] >= ncmp:
            raise _Stop()
        return ba_g6

    G6Sim._batched_search = wrapped
    try:
        sim.play()
    except _Stop:
        pass
    finally:
        G6Sim._batched_search = orig
    print(f"DECEQ_RESULT compared={state['n']} failures={state['fail']}")
    return state["fail"] == 0


def _g6_valsum(sim, seat, legal, kd_act, k, nW, rng):
    """Run g6 lockstep exact and return (val_sum dict, best_act) mirroring _batched_search."""
    worlds = []
    for _w in range(nW):
        world = copy.deepcopy(sim); world.search_seat = -1
        _determinize(world, seat, rng); worlds.append(world)
    rolls = []
    for wi, world in enumerate(worlds):
        for a in legal:
            tile = TILE_LIST[a - ACT["Play"]]
            clone = copy.deepcopy(world)
            g = clone._rollout_vcut_g(seat, tile, k)
            rolls.append({"wi": wi, "a": a, "gen": g, "send": None, "done": False, "val": None})
    while True:
        preqs = []; pidx = []; vreqs = []; vidx = []; any_live = False
        for r in rolls:
            if r["done"]:
                continue
            any_live = True
            try:
                req = r["gen"].send(r["send"])
            except StopIteration as e:
                r["done"] = True; r["val"] = e.value; continue
            if req[0] == "policy":
                pidx.append(r); preqs.append((req[1], req[2]))
            else:
                vidx.append(r); vreqs.append(req[1])
        if not any_live or (not preqs and not vreqs):
            break
        if preqs:
            for r, a in zip(pidx, g6_engine.eval_policy_exact(preqs)):
                r["send"] = a
        if vreqs:
            for r, v in zip(vidx, g6_engine.eval_value_exact(vreqs)):
                r["send"] = v
    val_sum = {a: 0.0 for a in legal}
    by_world = {}
    for r in rolls:
        by_world.setdefault(r["wi"], {})[r["a"]] = r["val"]
    w = 0
    for wi in range(nW):
        wv = by_world.get(wi, {})
        if any(wv.get(a) is None for a in legal):
            continue
        for a in legal:
            val_sum[a] += wv[a]
        w += 1
    best_key = None; best_act = kd_act
    for a in legal:
        key = (val_sum[a] / w if w else 0.0, 1 if a == kd_act else 0)
        if best_key is None or key > best_key:
            best_key = key; best_act = a
    return val_sum, best_act


def run_smoke(nW=2, k=3, seed=9800000):
    init_infer(mode="exact", null=False, device="cpu")
    t0 = time.time()
    sim = G6Sim([None] * 4, seed=seed, quan=0, learner_seats=[], cnn=True)
    sim.search_seat = 0; sim.n_worlds = nW; sim.k_cutoff = k
    sim._rng = np.random.RandomState(seed * 4)
    sim.play()
    print(f"SMOKE ok placement(seat0)={_placement(sim.scores,0):.3f} scores={sim.scores} "
          f"decisions={getattr(sim,'_decisions',0)} overrides={getattr(sim,'_override',0)} "
          f"bad_world={getattr(sim,'_bad_world',0)} time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke")
    ap.add_argument("--seed", type=int, default=9800000)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--nW", type=int, default=8)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--ncmp", type=int, default=3)
    a = ap.parse_args()
    if a.mode == "smoke":
        run_smoke(nW=a.nW, k=a.k, seed=a.seed)
    elif a.mode == "deceq":
        ok = run_deceq(seed=a.seed, seat=a.seat, nW=a.nW, k=a.k, ncmp=a.ncmp)
        print("DECEQ", "PASS" if ok else "FAIL")
