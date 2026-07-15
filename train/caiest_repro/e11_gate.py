"""
e11_gate.py — calibrated duplicate-format placement gate for the aug/TTA study.
Reuses the exact duplicate scoring of e8_gate (4-seat rotation; base-vs-base raw = 2.500),
but lam=0 (no value model) and adds optional SUIT-PERMUTATION TEST-TIME AUGMENTATION on the
candidate policy: average the policy logits over a set of suit-permutations of the state
(each permuted logit vector realigned to the original action space via fwd_action_perm), then
argmax. --tta-perms selects the subset (e.g. "0,3,4" = identity + 2 cyclic rotations = C3).

  python3 e11_gate.py --cand ckpt/aug/aug_128x40_s0.pkl --ref ckpt/e1b/full_128x40_s1.pkl \
      --seeds 500 --workers 40 --seed0 500000 --out cell.json                 # plain net gate
  python3 e11_gate.py --cand ckpt/e1b/full_128x40_s1.pkl --ref ckpt/e1b/full_128x40_s1.pkl \
      --cand-tta 1 --tta-perms 0,1,2,3,4,5 --seeds 500 --workers 40 --seed0 500000 --out cell.json  # TTA gate
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, multiprocessing as mp
from sim_cnn import Sim
from models_explore import build
import suit_aug

torch.set_num_threads(1)
_CACHE = {}

# precompute suit-perm remaps (numpy) once per process
_PERMS = suit_aug.PERMS
_ROWS = [np.array([p[0], p[1], p[2], 3]) for p in _PERMS]
_A = [suit_aug.action_perm(p) for p in _PERMS]        # A[new]=old : new_mask = old_mask[A]
_F = [suit_aug.fwd_action_perm(p) for p in _PERMS]     # F[old]=new : realign logits via lg[F]


def _parse_cfg(s):
    cfg = {}
    for kv in s.split(","):
        kv = kv.strip()
        if not kv: continue
        k, v = kv.split("="); cfg[k] = int(v)
    return cfg


def _load_policy(path, kind, cfg):
    key = (path, kind, tuple(sorted(cfg.items())))
    if key not in _CACHE:
        m = build(kind, **cfg)
        if path.endswith(".npz"):
            z = np.load(path); sd = {k: torch.from_numpy(z[k]) for k in z.keys() if k != "meta_blocks"}
        else:
            sd = torch.load(path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd and not any(
                    k.startswith(("stem", "body", "foot", "head")) for k in sd):
                sd = sd["state_dict"]
        m.load_state_dict(sd); m.eval()
        _CACHE[key] = m
    return _CACHE[key]


def _logits(m, obs, mask):
    with torch.no_grad():
        lg = m({"is_training": False, "obs": {
            "observation": torch.from_numpy(np.ascontiguousarray(obs)),
            "action_mask": torch.from_numpy(np.ascontiguousarray(mask))}})
    return lg.numpy().flatten()


def _greedy_lg(m):
    def fn(obs, mask):
        return [int(_logits(m, obs, mask).argmax())]
    return fn


def _tta_lg(m, perm_idxs):
    def fn(obs, mask):
        acc = None
        for pi in perm_idxs:
            po = obs[:, :, _ROWS[pi], :]
            pm = mask[:, _A[pi]]
            lg = _logits(m, po, pm)          # permuted action space
            aligned = lg[_F[pi]]             # realign to original action indexing
            acc = aligned if acc is None else acc + aligned
        acc /= len(perm_idxs)
        return [int(acc.argmax())]
    return fn


def _work(arg):
    seed, cand, ck, ccfg, ref, rk, rcfg, tta, perm_idxs = arg
    mc = _load_policy(cand, ck, ccfg); mr = _load_policy(ref, rk, rcfg)
    fc = _tta_lg(mc, perm_idxs) if tta else _greedy_lg(mc)
    fr = _greedy_lg(mr)
    placement_sum = 0.0; pts = [0, 0, 0, 0]; micro = 0; pg_rank = []
    for cs in range(4):
        pols = [fr, fr, fr, fr]; pols[cs] = fc
        sim = Sim(pols, seed=seed, quan=0, learner_seats=[], cnn=True)
        sim.play()
        sc = sim.scores; cand_score = sc[cs]
        greater = sum(1 for j in range(4) if sc[j] > cand_score)
        equal = sum(1 for j in range(4) if sc[j] == cand_score)
        avg_rank = greater + (equal + 1) / 2.0
        placement_sum += 5.0 - avg_rank; micro += cand_score
        r = max(0, min(3, int(round(avg_rank)) - 1)); pts[r] += 1
        pg_rank.append(float(avg_rank))
    return placement_sum, pts, micro, pg_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True); ap.add_argument("--cand-kind", default="resbn_fused")
    ap.add_argument("--cand-cfg", default="channels=128,blocks=40")
    ap.add_argument("--ref", required=True); ap.add_argument("--ref-kind", default="resbn_fused")
    ap.add_argument("--ref-cfg", default="channels=128,blocks=40")
    ap.add_argument("--cand-tta", type=int, default=0)
    ap.add_argument("--tta-perms", default="0,1,2,3,4,5")
    ap.add_argument("--seeds", type=int, default=500)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=500000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ccfg = _parse_cfg(a.cand_cfg); rcfg = _parse_cfg(a.ref_cfg)
    perm_idxs = [int(x) for x in a.tta_perms.split(",")] if a.cand_tta else [0]
    args = [(a.seed0 + i, a.cand, a.cand_kind, ccfg, a.ref, a.ref_kind, rcfg,
             bool(a.cand_tta), perm_idxs) for i in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers) as p:
        res = p.map(_work, args, chunksize=1)
    ngames = len(res) * 4
    tot_pts = sum(r[0] for r in res); dist = [0, 0, 0, 0]; micro = 0; pg = []
    for r in res:
        for i in range(4): dist[i] += r[1][i]
        micro += r[2]; pg.extend(r[3])
    pg = np.asarray(pg, np.float64)
    out = dict(cand=os.path.basename(a.cand), ref=os.path.basename(a.ref),
               cand_tta=a.cand_tta, tta_perms=(perm_idxs if a.cand_tta else []),
               games=ngames, seeds=len(res),
               placement_pts=round(tot_pts / ngames, 4) if ngames else 0.0,
               dist_1234=dist,
               first_pct=round(100 * dist[0] / ngames, 2) if ngames else 0.0,
               fourth_pct=round(100 * dist[3] / ngames, 2) if ngames else 0.0,
               sg_win_rate=round(float((pg <= 1.5).mean()), 4),
               sg_fourth_rate=round(float((pg >= 3.5).mean()), 4),
               micro_cand_per_game=round(micro / ngames, 3) if ngames else 0.0,
               seconds=round(time.time() - t0, 1), seed0=a.seed0)
    with open(a.out, "w") as f: json.dump(out, f, indent=2)
    print(json.dumps(out), flush=True)
    if ngames == 0: sys.exit(2)


if __name__ == "__main__":
    main()
