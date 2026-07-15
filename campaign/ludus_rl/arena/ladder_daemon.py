"""Continuous rated ladder: the platform's round-robin engine.

Loop forever: pick the least-played pair from the rated pool (league champion,
recent generations, every uploaded user bot, scripted anchors), play a block of
seat-swapped mirrored matches via baselines.eval_pair (the same CI-honest
protocol as league gates), update OpenSkill ratings per block, publish
standings atomically for /api/standings.

Run (GPU box, niced):  python -m arena.ladder_daemon --out /root/ludus_ladder_live
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from openskill.models import PlackettLuce

REPO = Path(__file__).resolve().parents[1]
LEAGUE = Path("/root/ludus_train/league")
BOTS = Path("/root/ludus_bots")


GRAPHS = Path("/root/ludus_graphs")


def pool() -> dict[str, str]:
    """bot display-id -> agent spec (eval_pair specs + graph:<name>)."""
    out = {"random_v0": "random", "rule_v0": "rule"}
    if (LEAGUE / "champion.msgpack").exists():
        out["champion"] = f"ppo:{LEAGUE / 'champion.msgpack'}"
    gens = sorted(LEAGUE.glob("gen_*.msgpack"),
                  key=lambda p: int(p.stem.split("_")[1]))
    for p in gens[-3:]:                       # recent generations only
        out[p.stem] = f"ppo:{p}"
    for p in sorted(BOTS.glob("*.msgpack")):
        out[f"user:{p.stem}"] = f"ppo:{p}"
    for p in sorted(GRAPHS.glob("*.json")):
        out[f"graph:{p.stem}"] = f"graph:{p.stem}"
    return out


# ---------------- sequential runner for python-side (graph) policies --------
def _seq_policy(spec: str, seat: int):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from boom import engine, graphs, vec
    if spec.startswith("graph:"):
        return graphs.make_policy(graphs.load_rules(spec[6:]), seat)
    if spec == "random":
        rnd = jax.jit(lambda k, s: vec.flat_to_triple(
            vec.random_legal_action(k, s, seat)))
        return lambda key, state, tick: (
            [int(v) for v in np.asarray(rnd(key, state))]
            if tick % 5 == 0 else [4, 0, 0])
    if spec == "rule":
        from baselines.rule_bot import rule_v0_action
        fn = jax.jit(lambda s: rule_v0_action(s, seat))
        return lambda key, state, tick: (
            [int(v) for v in np.asarray(fn(state))]
            if tick % 5 == 0 else [4, 0, 0])
    if spec.startswith("ppo:"):
        from flax.serialization import from_bytes

        from baselines.ppo_selfplay import ActorCritic
        from boom.engine import H, OBS_C, OBS_VEC, W
        net = ActorCritic()
        tmpl = net.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, H, W, OBS_C), jnp.float32),
                        jnp.zeros((1, OBS_VEC), jnp.float32))
        params = from_bytes(tmpl, Path(spec[4:]).read_bytes())

        @jax.jit
        def greedy(state, key):
            obs = engine.observe(state, seat)
            mask = vec.flat_legal(state, seat)
            logits, _ = net.apply(params, obs.spatial[None], obs.vector[None])
            return vec.flat_to_triple(jax.random.categorical(
                key, jnp.where(mask, logits[0], -1e9) / 0.6))
        return lambda key, state, tick: [int(v) for v in
                                         np.asarray(greedy(state, key))]
    raise ValueError(spec)


def play_block_seq(spec_a: str, spec_b: str, matches: int) -> dict | None:
    """Seat-swapped mirrored pairs, one env at a time (for graph policies)."""
    import jax
    import jax.numpy as jnp
    from boom import engine
    step = jax.jit(engine.step)
    wins_a = draws = 0
    pairs = matches // 2
    for i in range(pairs):
        for flip in (0, 1):
            p0 = _seq_policy(spec_b if flip else spec_a, 0)
            p1 = _seq_policy(spec_a if flip else spec_b, 1)
            state = engine.reset(jax.random.PRNGKey(1000 + i), None)
            key = jax.random.PRNGKey(2000 + i)
            res = -1
            while res == -1:
                key, k0, k1 = jax.random.split(key, 3)
                t = int(state.tick)
                state = step(state, jnp.asarray(
                    [p0(k0, state, t), p1(k1, state, t)], jnp.int32), None)
                res = int(engine.result(state))
            if res == 2:
                draws += 1
            elif (res == 0) != bool(flip):
                wins_a += 1
    return {"a_win_rate": wins_a / max(pairs * 2, 1),
            "draw_rate": draws / max(pairs * 2, 1)}


def play_block(spec_a: str, spec_b: str, matches: int) -> dict | None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = f.name
    r = subprocess.run([sys.executable, "-m", "baselines.eval_pair",
                        "--a", spec_a, "--b", spec_b,
                        "--matches", str(matches), "--out", out],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"eval failed {spec_a} vs {spec_b}: {r.stderr[-300:]}", flush=True)
        return None
    return json.loads(Path(out).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/ludus_ladder_live")
    ap.add_argument("--block", type=int, default=24)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    model = PlackettLuce()

    ratings: dict = {}
    games: dict = {}
    pair_counts: dict = {}
    state_f = outdir / "state.json"
    if state_f.exists():
        st = json.loads(state_f.read_text())
        ratings = {k: model.rating(mu=v["mu"], sigma=v["sigma"], name=k)
                   for k, v in st["ratings"].items()}
        games = st.get("games", {})
        pair_counts = {tuple(k.split("|")): v
                       for k, v in st.get("pairs", {}).items()}

    while True:
        P = pool()
        for b in P:
            ratings.setdefault(b, model.rating(name=b))
            games.setdefault(b, 0)
        pairs = list(itertools.combinations(sorted(P), 2))
        random.shuffle(pairs)
        pair = min(pairs, key=lambda pr: pair_counts.get(pr, 0))
        a, b = pair
        if P[a].startswith("graph:") or P[b].startswith("graph:"):
            n = min(args.block, 12)
            res = play_block_seq(P[a], P[b], n)
        else:
            n = args.block
            res = play_block(P[a], P[b], n)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if res is None:
            time.sleep(10)
            continue
        wa = round(res["a_win_rate"] * n)
        draws = round(res.get("draw_rate", 0) * n)
        wb = n - wa - draws
        for _ in range(wa):
            [[ratings[a]], [ratings[b]]] = model.rate(
                [[ratings[a]], [ratings[b]]])
        for _ in range(wb):
            [[ratings[b]], [ratings[a]]] = model.rate(
                [[ratings[b]], [ratings[a]]])
        games[a] += n
        games[b] += n

        rows = sorted(({"bot": k, "mu": r.mu, "sigma": r.sigma,
                        "rating": r.mu - 2 * r.sigma, "games": games[k]}
                       for k, r in ratings.items() if k in P),
                      key=lambda x: -x["rating"])
        tmp = outdir / "standings.tmp"
        tmp.write_text(json.dumps({"updated": time.time(), "rows": rows,
                                   "protocol": "seat-swapped mirrored pairs, "
                                   "OpenSkill, rating = mu - 2*sigma"}, indent=1))
        tmp.replace(outdir / "standings.json")
        state_f.write_text(json.dumps({
            "ratings": {k: {"mu": r.mu, "sigma": r.sigma}
                        for k, r in ratings.items()},
            "games": games,
            "pairs": {f"{k[0]}|{k[1]}": v for k, v in pair_counts.items()}}))
        print(json.dumps({"pair": pair, "a_wr": res["a_win_rate"],
                          "blocks": pair_counts[pair]}), flush=True)


if __name__ == "__main__":
    main()
