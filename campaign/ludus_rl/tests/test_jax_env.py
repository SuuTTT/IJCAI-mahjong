"""Acceptance tests + replay-equivalence + throughput benchmark for the
vmappable JAX MCR env (mahjong/jax_env.py).

Run the pytest bits:      python -m pytest tests/test_jax_env.py
Run full validation:      python tests/test_jax_env.py --golden /root/mcr_testset/mcr_final2026_golden.jsonl
Throughput only:          python tests/test_jax_env.py --bench --batch 4096
"""
import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mahjong import jax_env as J


# --------------------------------------------------------------- pytest tests
def _demo_wall():
    from mahjong.engine import build_wall
    return J.wall_to_codes(build_wall(0))


def test_reset_step_jittable_vmappable():
    walls = jnp.stack([jnp.asarray(_demo_wall()) for _ in range(8)])
    states = jax.vmap(J.reset, in_axes=(0, None))(walls, 0)
    assert states.hands.shape == (8, 4, 34)
    acts = jnp.zeros((8, 4, 2), jnp.int32).at[:, :, 0].set(2)  # everyone "Play W1"
    step_vj = jax.jit(jax.vmap(J.step))
    ns = step_vj(states, acts)
    assert ns.hands.shape == (8, 4, 34)
    assert ns.phase.shape == (8,)


def test_determinism():
    walls = jnp.stack([jnp.asarray(_demo_wall()) for _ in range(16)])
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 16)
    s1, n1 = J.v_run(walls, 0, keys)
    s2, n2 = J.v_run(walls, 0, keys)
    assert jnp.array_equal(n1, n2)
    assert jnp.array_equal(s1.hands, s2.hands)
    # identical walls+keys across lanes -> identical trajectories
    kk = jnp.broadcast_to(keys[0], (16, 2))
    ww = jnp.broadcast_to(jnp.asarray(_demo_wall()), (16, 136))
    s3, n3 = J.v_run(ww, 0, kk)
    assert int(n3.min()) == int(n3.max())


# ------------------------------------------------------------------ full run
def test_illegal_hu_forfeit():
    """issue #4 regression: declaring Hu on a non-winning hand must FORFEIT
    (declarer penalised), never be paid as a fan-0 win."""
    wall = jnp.asarray(J.wall_to_codes(_wall_seed(1000)))
    st = J.reset(wall, 0)
    acts = jnp.zeros((4, 2), jnp.int32).at[0, 0].set(J.A_HU)
    st2 = J.step(st, acts)
    fan, scores = J.host_score(jax.device_get(st2))
    assert fan == -1, f"illegal Hu should report fan=-1, got {fan}"
    assert scores[0] < 0, f"declarer must be penalised, got scores={scores}"
    assert sum(scores) == 0, f"forfeit must stay zero-sum, got {scores}"


def test_meld_cap():
    """issue #3 regression: a seat already holding 4 melds is never offered a
    5th (Chi/Peng/Gang/AnGang). BuGang (in-place upgrade) is unaffected."""
    import numpy as np
    wall = jnp.asarray(J.wall_to_codes(_wall_seed(7)))
    st = J.reset(wall, 0)._replace(meld_cnt=jnp.array([4, 4, 4, 4], jnp.int32))
    actor = int(st.seat_to_act)
    # draw phase: a concealed 4-of-a-kind must NOT offer AnGang at meld_cnt=4
    st_d = st._replace(hands=st.hands.at[actor, 10].set(4))
    draw = np.asarray(J._draw_mask_row(st_d, st_d.seat_to_act))
    assert not draw[J.A_ANGANG + 10], "5th AnGang offered at meld_cnt=4"
    # claim phase: a seat holding 3 of a just-discarded tile must NOT offer
    # Peng/Gang/Chi at meld_cnt=4
    st_c = st._replace(last_discard_seat=jnp.int32(0),
                       last_discard_tile=jnp.int32(5),
                       hands=st.hands.at[1, 5].set(3))
    claim = np.asarray(J._claim_mask_row(st_c, 1))
    assert not claim[J.A_CHI:J.A_ANGANG].any(), "5th Peng/Gang/Chi at meld_cnt=4"


def iter_records(path, limit=0):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def run_replay(path, limit=0):
    n = ok = 0
    fails = []
    tagfail = {}
    for rec in iter_records(path, limit):
        n += 1
        good, msg = J.replay_game(rec)
        if good:
            ok += 1
        else:
            fails.append((rec["game_id"], msg, rec.get("tags", [])))
            for t in rec.get("tags", []) or ["untagged"]:
                tagfail[t] = tagfail.get(t, 0) + 1
    print("REPLAY EQUIVALENCE: %d/%d golden games reproduced exactly" % (ok, n))
    if fails:
        print("  failing tag histogram:", tagfail)
        for gid, msg, tags in fails[:25]:
            print("  FAIL %s [%s]: %s" % (gid, ",".join(tags), msg))
    return ok, n


def run_bench(batch=4096, device=None, seed=0):
    dev = device or jax.default_backend()
    walls = jnp.stack([jnp.asarray(J.wall_to_codes(_wall_seed(s)))
                       for s in range(batch)])
    keys = jax.random.split(jax.random.PRNGKey(seed), batch)
    # warmup / compile
    s, n = J.v_run(walls, 0, keys)
    n.block_until_ready()
    t0 = time.time()
    s, n = J.v_run(walls, 0, keys)
    n.block_until_ready()
    dt = time.time() - t0
    total = int(np.asarray(n).sum())
    print("THROUGHPUT [%s] batch=%d: %d game-steps in %.4fs = %.0f steps/s "
          "(env-only, random-legal policy; %.1f avg steps/game)"
          % (dev, batch, total, dt, total / dt, total / batch))
    return total / dt


def _wall_seed(s):
    from mahjong.engine import build_wall
    return build_wall(s)


def check_callback():
    """Demonstrate the jax.pure_callback fan scorer inside jit on a real Hu."""
    # find a zimo/ron golden game, replay to terminal, score via jit(score_state)
    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="/root/mcr_testset/mcr_final2026_golden.jsonl")
    ap.add_argument("--full", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()

    print("jax", jax.__version__, "backend", jax.default_backend())
    print("=" * 70)
    # acceptance 1: jittable + vmappable
    test_reset_step_jittable_vmappable()
    walls = jnp.stack([jnp.asarray(_demo_wall()) for _ in range(4)])
    states = jax.vmap(J.reset, in_axes=(0, None))(walls, 0)
    print("ACCEPT-1 jit(vmap(step)) ok. State e.g. hands", states.hands.shape,
          states.hands.dtype, "phase", states.phase.shape, states.phase.dtype)
    print("=" * 70)

    if args.bench:
        run_bench(args.batch)
        sys.exit(0)

    # acceptance 4: determinism
    test_determinism()
    print("ACCEPT-4 determinism ok (same wall+key -> identical trajectory)")
    print("=" * 70)

    # regression: issue #4 illegal-Hu forfeit
    test_illegal_hu_forfeit()
    print("REGRESS  illegal-Hu forfeit ok (fan=-1, declarer penalised, zero-sum)")
    test_meld_cap()
    print("REGRESS  meld cap ok (no 5th Chi/Peng/Gang/AnGang at meld_cnt=4)")
    print("=" * 70)

    # acceptance 2: replay equivalence
    path = args.full or args.golden
    run_replay(path, args.limit)
    print("=" * 70)

    # acceptance 3: throughput
    run_bench(args.batch)
