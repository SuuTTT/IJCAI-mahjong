"""
tta_time.py — measure per-move inference latency of bn128s1 with/without suit-perm TTA,
on CPU single-thread (mirrors the Botzone runtime). Reports ms/move for single forward,
3-perm C3 TTA, and 6-perm full TTA. Used to check TTA stays under the ~1s/move TLE budget.
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from models_explore import build
import suit_aug

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
_PERMS = suit_aug.PERMS
_ROWS = [np.array([p[0], p[1], p[2], 3]) for p in _PERMS]
_A = [suit_aug.action_perm(p) for p in _PERMS]
_F = [suit_aug.fwd_action_perm(p) for p in _PERMS]

BN = "ckpt/e1b/full_128x40_s1.pkl"
m = build("resbn_fused", channels=128, blocks=40)
sd = torch.load(BN, map_location="cpu"); m.load_state_dict(sd); m.eval()

d = np.load(os.path.join(HERE, "data", "cooked_single.npz"))
rng = np.random.RandomState(0)
idx = np.sort(rng.choice(len(d["act"]), 2500, replace=False))
OBS = np.ascontiguousarray(d["obs"][idx]).astype(np.int8)
MASK = np.ascontiguousarray(d["mask"][idx])


def _lg(obs, mask):
    with torch.no_grad():
        return m({"is_training": False, "obs": {
            "observation": torch.from_numpy(np.ascontiguousarray(obs)).float()[None],
            "action_mask": torch.from_numpy(np.ascontiguousarray(mask)).float()[None]}}).numpy().flatten()


def single(o, mk):
    return int(_lg(o, mk).argmax())


def tta(o, mk, perm_idxs):
    acc = None
    for pi in perm_idxs:
        po = o[:, _ROWS[pi], :]           # (38,4,9)
        pm = mk[_A[pi]]
        lg = _lg(po, pm)
        aligned = lg[_F[pi]]
        acc = aligned if acc is None else acc + aligned
    return int((acc / len(perm_idxs)).argmax())


def bench(fn, n=2000, warm=200):
    for i in range(warm):
        fn(OBS[i], MASK[i])
    t0 = time.time()
    for i in range(n):
        fn(OBS[i % len(OBS)], MASK[i % len(MASK)])
    return 1000.0 * (time.time() - t0) / n


res = {
    "single_ms_per_move": round(bench(lambda o, mk: single(o, mk)), 3),
    "tta3_C3_ms_per_move": round(bench(lambda o, mk: tta(o, mk, [0, 3, 4])), 3),
    "tta6_full_ms_per_move": round(bench(lambda o, mk: tta(o, mk, [0, 1, 2, 3, 4, 5])), 3),
    "threads": torch.get_num_threads(),
    "note": "CPU single-thread, batch-1 per move (Botzone-like). TLE budget ~1000 ms/move.",
}
with open(os.path.join(HERE, "tta_time.json"), "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
