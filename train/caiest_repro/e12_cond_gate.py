"""
e12_cond_gate.py — placement gate for SOURCE-CONDITIONED candidates (39-plane input).

The conditioned model (e11_cond_train.py) takes 38 game planes + 1 source plane
(0=base corpus, 1=Final2 finalists). At deploy we set the source plane to
--src (default 1.0 = "play like the Final2 finalists"). The gate machinery is
otherwise byte-identical to e12_ens_gate.py: mean-softmax-over-legal ensemble,
paired duplicate placement, argmax with legal-repair.

Calibration self-test (--selftest): cand==ref, both conditioned same path/src,
must give EXACTLY 2.500 (identical policy in all seats).
"""
import os, sys, json, argparse, time
sys.path.insert(0, "/root/caiest_repro")
import numpy as np, torch, multiprocessing as mp
import models_explore
from sim_cnn import Sim

torch.set_num_threads(1)
_CACHE = {}


def _load(path, in_planes):
    key = (path, in_planes)
    if key not in _CACHE:
        models_explore.IN_PLANES = in_planes            # BEFORE build
        from models_explore import build
        m = build("resbn_fused", channels=128, blocks=40)
        m.load_state_dict(torch.load(path, map_location="cpu")); m.eval()
        _CACHE[key] = m
    return _CACHE[key]


def _logits(m, obs, mask, src=None):
    """obs is (1,38,4,9). If src is not None, append a constant (1,1,4,9) plane."""
    ob = np.ascontiguousarray(obs).astype(np.float32)
    if src is not None:
        pl = np.full((ob.shape[0], 1, 4, 9), float(src), dtype=np.float32)
        ob = np.concatenate([ob, pl], axis=1)           # (1,39,4,9)
    with torch.no_grad():
        lg = m({"is_training": False, "obs": {
            "observation": torch.from_numpy(ob),
            "action_mask": torch.from_numpy(np.ascontiguousarray(mask))}})
    return lg.numpy().flatten()


def _single_fn(m, src=None):
    def fn(obs, mask, return_logits=False):
        lg = _logits(m, obs, mask, src)
        return lg if return_logits else [int(lg.argmax())]
    return fn


def _ens_fn(models, src=None):
    def fn(obs, mask, return_logits=False):
        mk = mask.flatten().astype(np.float32)
        acc = None
        for m in models:
            lg = _logits(m, obs, mask, src)
            lg = np.where(mk > 0, lg, -1e30)
            lg = lg - lg.max()
            p = np.exp(lg) * (mk > 0)
            s = p.sum()
            p = p / s if s > 0 else (mk / max(1.0, mk.sum()))
            acc = p if acc is None else acc + p
        avg = acc / len(models)
        out = np.log(np.where(avg > 0, avg, 1e-30))
        return out if return_logits else [int(out.argmax())]
    return fn


class PSim(Sim):
    def _ask(self, seat):
        from sim_cnn import ACT
        obs, mask = self._obs_mask(seat)
        if mask.sum() == 0:
            return ACT["Pass"]
        lg = self.policies[seat](obs[None, :], mask[None, :], return_logits=True)
        act = int(lg.argmax())
        if not mask[act]:
            act = int(np.flatnonzero(mask)[0])
        return act


# module-level config filled by main (so _work is picklable via fork)
CFG = {}


def _mk_cand():
    ms = [_load(p, CFG["cand_planes"]) for p in CFG["cand"]]
    return _ens_fn(ms, CFG["cand_src"])


def _mk_ref():
    m = _load(CFG["ref"][0], CFG["ref_planes"])
    if len(CFG["ref"]) == 1:
        return _single_fn(m, CFG["ref_src"])
    return _ens_fn([_load(p, CFG["ref_planes"]) for p in CFG["ref"]], CFG["ref_src"])


def _work(seed):
    fc = _mk_cand(); fr = _mk_ref()
    placement_sum = 0.0
    for cs in range(4):
        pols = [fr] * 4; pols[cs] = fc
        sim = PSim(pols, seed=seed, quan=0, learner_seats=[], cnn=True)
        sim.play()
        sc = sim.scores; c = sc[cs]
        greater = sum(1 for j in range(4) if sc[j] > c)
        equal = sum(1 for j in range(4) if sc[j] == c)
        placement_sum += 5.0 - (greater + (equal + 1) / 2.0)
    return placement_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)          # comma-sep pkls
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand_planes", type=int, default=39)
    ap.add_argument("--ref_planes", type=int, default=38)
    ap.add_argument("--cand_src", type=float, default=1.0)
    ap.add_argument("--ref_src", type=float, default=None)  # None => don't append (38-plane)
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=200000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    global CFG
    CFG = dict(cand=a.cand.split(","), ref=a.ref.split(","),
               cand_planes=a.cand_planes, ref_planes=a.ref_planes,
               cand_src=a.cand_src,
               ref_src=(None if a.ref_planes == 38 else (a.ref_src if a.ref_src is not None else a.cand_src)))
    if a.selftest:
        CFG = dict(cand=a.cand.split(","), ref=a.cand.split(",")[:1],
                   cand_planes=a.cand_planes, ref_planes=a.cand_planes,
                   cand_src=a.cand_src, ref_src=a.cand_src)
        # cand ensemble vs ref=single-first-of-same: only 2.500 if cand is single too
        CFG["cand"] = a.cand.split(",")[:1]
    t0 = time.time()
    args = list(range(a.seed0, a.seed0 + a.seeds))
    with mp.Pool(a.workers) as p:
        res = p.map(_work, args, chunksize=4)
    pts = sum(res) / (len(res) * 4)
    out = dict(cand=a.cand, ref=a.ref, cand_src=a.cand_src, seeds=a.seeds,
               seed0=a.seed0, placement_pts=round(pts, 4),
               seconds=round(time.time() - t0, 1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
