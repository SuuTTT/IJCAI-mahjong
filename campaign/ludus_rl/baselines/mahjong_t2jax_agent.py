"""JAX-T2 Mahjong agent: load a T2-JAX flax policy checkpoint and play MCR.

Loads EITHER a T2-JAX training checkpoint (``latest.msgpack`` / ``best_by_eval.
msgpack``, written by ``baselines/mahjong_t2_jax.py:Trainer.save_ckpt`` as a flax
msgpack blob ``{"pparams","vparams","popt","vopt","meta"}``) OR a raw champion
``.npz`` (the frozen SL anchor ``kdens_s0_fp16.npz``) into the *identical* flax
kdens3 forward (``baselines/mahjong_jax_policy.py``), and exposes it as a
Botzone-protocol agent via ``mahjong.rl_env.PolicyAgent`` (the champion
``feature.py`` encoder -> obs (38,4,9) + a masked argmax over the 235 actions).

Because the msgpack blob is a plain nested dict of arrays, we restore it with
``flax.serialization.msgpack_restore`` (no need to reconstruct the optimizer-state
template) and pull out ``["pparams"]`` -- the trainable policy pytree, whose keys
(``stem_w``, ``body_i_c1_w`` ...) match ``mahjong_jax_policy.load_params``'s tree.

CPU-only by contract (the GPU belongs to the trainer): callers must
``export JAX_PLATFORMS=cpu`` before importing jax.  A single jitted forward is
built per checkpoint path and CACHED, so every game of an eval shares one
compiled policy (compile once, not per game).

Verify (0-illegal through the oracle-exact MyEngine, >=50 games):
    JAX_PLATFORMS=cpu python -m baselines.mahjong_t2jax_agent \
        --verify /root/ludus_train/mahjong_t2_jax/latest.msgpack --games 50
MyEngine (validate_adapter) applies discards with ``hand.remove(tile)`` and has
NO legal-fallback, so any illegal action raises -- N clean games == 0 illegal.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, "/root/ludus")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/root/ludus/baselines")
os.environ.setdefault("MCR_CHAMPION_DIR", "/root/mcr_champion")

import numpy as np

DEFAULT_ANCHOR = os.environ.get("MCR_SL_ANCHOR",
                                "/root/mcr_champion/kdens_s0_fp16.npz")

# path -> act(obs, mask) callable, so an eval reuses one compiled forward.
_ACT_CACHE = {}
_META_CACHE = {}


def _restore_pparams(path):
    """Return (pparams_dict_of_np, blocks, env_steps) for a .msgpack or .npz."""
    import jax.numpy as jnp
    from mahjong_jax_policy import load_params
    if path.endswith(".npz"):
        bundle = load_params(path)              # {"params": {...}, "blocks": int}
        _META_CACHE.setdefault(path, {"env_steps": None})
        return bundle["params"], bundle["blocks"], None
    import flax.serialization as fser
    with open(path, "rb") as f:
        blob = fser.msgpack_restore(f.read())
    pp = blob["pparams"]
    meta = dict(blob.get("meta", {}) or {})
    _META_CACHE[path] = meta                     # cache the FULL meta (env_steps, update, ...)
    env_steps = meta.get("env_steps")
    blocks = sum(1 for k in pp if k.endswith("_c1_w") and k.startswith("body_"))
    params = {k: jnp.asarray(np.asarray(v), dtype=jnp.float32) for k, v in pp.items()}
    return params, int(blocks), env_steps


def load_meta(path):
    """Checkpoint meta (env_steps/update/...) without building a forward."""
    if path in _META_CACHE:
        return _META_CACHE[path]
    if path.endswith(".npz"):
        m = {"env_steps": None}
    else:
        import flax.serialization as fser
        with open(path, "rb") as f:
            blob = fser.msgpack_restore(f.read())
        m = dict(blob.get("meta", {}) or {})
    _META_CACHE[path] = m
    return m


def make_act(path):
    """Build (and cache) a masked-argmax ``act(obs, mask) -> int`` for a ckpt."""
    if path in _ACT_CACHE:
        return _ACT_CACHE[path]
    import jax
    import jax.numpy as jnp
    from mahjong_jax_policy import forward
    params, blocks, env_steps = _restore_pparams(path)   # also populates _META_CACHE
    bundle = {"params": params, "blocks": blocks}

    @jax.jit
    def _fwd(o, m):
        # forward() already applies the hard legal mask (-1e9 on illegal)
        return forward(bundle, o, m, jnp.float32)

    def act(obs, mask):
        o = jnp.asarray(np.asarray(obs, np.float32)[None])       # (1,38,4,9)
        m = jnp.asarray(np.asarray(mask, bool)[None])            # (1,235)
        logits = np.asarray(_fwd(o, m))[0]
        # belt-and-suspenders: never pick outside the legal set
        a = int(np.where(np.asarray(mask, bool), logits, -1e30).argmax())
        return a

    _ACT_CACHE[path] = act
    return act


def make_agent_factory(path):
    """Zero-arg factory -> fresh Botzone agent (shared compiled policy)."""
    from mahjong.rl_env import PolicyAgent
    act = make_act(path)
    return lambda: PolicyAgent(act)


# ------------------------------------------------------------------- verify CLI
def verify(path, games=50, seed0=0, opponent="anchor"):
    """Drive ``games`` matches with the t2jax agent at seat 0 through MyEngine.

    Any illegal action would raise inside MyEngine (``hand.remove`` with no
    fallback), so completing all games == 0 illegal actions.  Opponent field is
    the SL anchor (same net) by default; ``--opp efficiency`` for a mixed field."""
    from mahjong.engine import build_wall
    from mahjong.match import run_match
    from mahjong.bots import EfficiencyBot, RandomLegalBot

    cand_factory = make_agent_factory(path)
    if opponent == "anchor":
        opp_factory = make_agent_factory(DEFAULT_ANCHOR)
        mkopp = lambda s: opp_factory()
    elif opponent == "efficiency":
        mkopp = lambda s: EfficiencyBot(seed=s)
    else:
        mkopp = lambda s: RandomLegalBot(seed=s)

    print(f"[verify] {games} games, seat0=t2jax({os.path.basename(path)}), "
          f"opponents={opponent}", flush=True)
    t0 = time.time()
    ok = 0
    wins = 0
    scoresum = 0.0
    for i in range(games):
        seed = seed0 + i
        wall = build_wall(seed)
        agents = [cand_factory()] + [mkopp(seed * 10 + s) for s in (1, 2, 3)]
        try:
            res = run_match(agents, wall, quan=0, srand=seed)
        finally:
            for a in agents:
                c = getattr(a, "close", None)
                if callable(c):
                    try:
                        c()
                    except Exception:
                        pass
        ok += 1
        scoresum += res["scores"][0]
        wins += int(res["winner"] == 0)
        if (i + 1) % 10 == 0:
            dt = time.time() - t0
            print(f"    {i + 1}/{games} clean  {dt:.1f}s ({dt / (i + 1):.2f}s/game)",
                  file=sys.stderr, flush=True)
    dt = time.time() - t0
    print(f"[verify] RESULT {ok}/{games} games completed with ZERO illegal "
          f"actions ({dt:.1f}s, {dt / games:.2f}s/game). seat0 win_rate="
          f"{wins / games:.3f} mean_score={scoresum / games:.3f}", flush=True)
    meta = load_meta(path)
    print(f"[verify] ckpt meta env_steps={meta.get('env_steps')} "
          f"update={meta.get('update')}", flush=True)
    return ok == games


def main():
    ap = argparse.ArgumentParser(description="JAX-T2 Mahjong agent")
    ap.add_argument("--verify", type=str, help="ckpt (.msgpack/.npz) to legality-verify")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--opp", type=str, default="anchor",
                    choices=["anchor", "efficiency", "random"])
    a = ap.parse_args()
    if a.verify:
        good = verify(a.verify, a.games, a.seed, a.opp)
        sys.exit(0 if good else 1)
    ap.error("give --verify <ckpt>")


if __name__ == "__main__":
    main()
