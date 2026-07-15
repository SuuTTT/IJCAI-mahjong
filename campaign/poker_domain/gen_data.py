"""Generate a behavioral-cloning dataset by self-play under the reference
(coherent near-equilibrium) strategy. Each decision encountered is recorded as
(infoset features, sampled action, legal mask). Reach-weighting is natural:
info-states are visited in proportion to how often the coherent policy reaches
them. Saves data.npz with feats[n,27], y[n], legal[n,3]."""
import argparse, json, random, time
import numpy as np
import leduc as L
from model import encode_key
from cfr import json_to_strat


def sample_action(probs, legal, rng):
    r = rng.random()
    cum = 0.0
    for a in legal:
        cum += probs[a]
        if r <= cum:
            return a
    return legal[-1]


def selfplay(strat, rng, records):
    # random physical deal
    idx = list(range(6))
    rng.shuffle(idx)
    p0, p1 = L.DECK[idx[0]], L.DECK[idx[1]]
    s = L.new_game(p0, p1)
    while s.kind != "terminal":
        if s.kind == "chance_pub":
            s = L.sample_public(s, rng)
            continue
        key = L.infoset_key(s)
        legal = L.legal_actions(s)
        probs = strat[key]
        a = sample_action(probs, legal, rng)
        mask = [1 if x in legal else 0 for x in range(3)]
        records.append((key, a, mask))
        s = L.apply_action(s, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="reference_strategy.json")
    ap.add_argument("--games", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data.npz")
    a = ap.parse_args()
    t0 = time.time()
    with open(a.ref) as f:
        strat = json_to_strat(json.load(f)["strategy"])
    rng = random.Random(a.seed)
    records = []
    for _ in range(a.games):
        selfplay(strat, rng, records)
    feats = np.stack([encode_key(k) for k, _, _ in records]).astype(np.float32)
    y = np.array([a_ for _, a_, _ in records], dtype=np.int64)
    legal = np.array([m for _, _, m in records], dtype=np.int8)
    np.savez_compressed(a.out, feats=feats, y=y, legal=legal)
    meta = {"games": a.games, "n_samples": len(records), "seed": a.seed,
            "action_counts": [int((y == i).sum()) for i in range(3)]}
    with open(a.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[gen] games={a.games} samples={len(records)} "
          f"action_counts={meta['action_counts']} secs={time.time()-t0:.1f} "
          f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
