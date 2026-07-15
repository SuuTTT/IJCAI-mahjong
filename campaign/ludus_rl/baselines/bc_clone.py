"""My Clone v0: behavior-clone a player from their recorded matches.

Replays are (env_version, seed, decks, action_log) — deterministic records, so
re-simulating them regenerates every observation the player acted on. The
clone fine-tunes the league champion's prior on the player's actions (a few
hundred human card-plays can steer a competent policy; they cannot teach a
blank network to play), then registers as an uploadable bot.

Run:  python baselines/bc_clone.py --name su --replays /root/ludus_replays \
          --init /root/ludus_train/league/champion.msgpack \
          --out /root/ludus_bots/clone_su.msgpack
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.serialization import from_bytes, to_bytes

from baselines.ppo_selfplay import ActorCritic
from boom import engine, vec
from boom.engine import H, OBS_C, OBS_VEC, W


def build_dataset(replay_dir: str, seat: int, noop_ratio: float, rng):
    """Re-simulate replays; collect (spatial, vector, flat_action) for `seat`."""
    step = jax.jit(engine.step)
    obs_fn = jax.jit(lambda s: engine.observe(s, seat))
    S, V, A = [], [], []
    n_replays = 0
    for path in sorted(glob.glob(f"{replay_dir}/*.json")):
        rec = json.loads(Path(path).read_text())
        decks = None if rec.get("decks") is None \
            else jnp.asarray(rec["decks"], jnp.int32)
        state = engine.reset(jax.random.PRNGKey(rec["seed"]), decks)
        n_replays += 1
        for a0, a1 in rec["action_log"]:
            mine = a0 if seat == 0 else a1
            is_play = mine[0] < 4
            if is_play or rng.random() < noop_ratio:
                o = obs_fn(state)
                S.append(np.asarray(o.spatial))
                V.append(np.asarray(o.vector))
                flat = 0 if not is_play else \
                    1 + mine[0] * (W * H) + mine[2] * W + mine[1]
                A.append(flat)
            state = step(state, jnp.asarray([a0, a1], jnp.int32), None)
    return (np.stack(S), np.stack(V), np.asarray(A, np.int32), n_replays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--replays", default="/root/ludus_replays")
    ap.add_argument("--seat", type=int, default=0, help="which seat is the human")
    ap.add_argument("--init", default=None,
                    help="champion checkpoint to fine-tune (omit = from scratch)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--noop-ratio", type=float, default=0.03,
                    help="fraction of no-op ticks kept (plays are ~1:60 ticks)")
    ap.add_argument("--play-weight", type=float, default=4.0)
    cfg = ap.parse_args()

    rng = np.random.default_rng(0)
    S, V, A, n = build_dataset(cfg.replays, cfg.seat, cfg.noop_ratio, rng)
    plays = int((A != 0).sum())
    print(json.dumps({"event": "dataset", "replays": n, "samples": len(A),
                      "card_plays": plays}), flush=True)

    net = ActorCritic()
    params = net.init(jax.random.PRNGKey(0),
                      jnp.zeros((1, H, W, OBS_C), jnp.float32),
                      jnp.zeros((1, OBS_VEC), jnp.float32))
    if cfg.init:
        params = from_bytes(params, Path(cfg.init).read_bytes())
        print(json.dumps({"event": "init_from", "path": cfg.init}), flush=True)
    tx = optax.adam(cfg.lr)
    opt = tx.init(params)
    Sj, Vj, Aj = jnp.asarray(S), jnp.asarray(V), jnp.asarray(A)
    w = jnp.where(Aj != 0, cfg.play_weight, 1.0)
    w = w / w.sum()

    @jax.jit
    def train_step(params, opt):
        def loss_fn(p):
            logits, _ = net.apply(p, Sj, Vj)
            ce = optax.softmax_cross_entropy_with_integer_labels(logits, Aj)
            return (ce * w).sum()
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt = tx.update(grads, opt, params)
        return optax.apply_updates(params, updates), opt, loss

    for i in range(cfg.steps):
        params, opt, loss = train_step(params, opt)
        if (i + 1) % 100 == 0:
            logits, _ = net.apply(params, Sj, Vj)
            acc = float((jnp.argmax(logits, -1) == Aj)[Aj != 0].mean())
            print(json.dumps({"event": "train", "step": i + 1,
                              "loss": float(loss), "play_top1_acc": acc}),
                  flush=True)

    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out).write_bytes(to_bytes(params))
    print(json.dumps({"event": "saved", "out": cfg.out,
                      "bot": f"user:{Path(cfg.out).stem}"}), flush=True)


if __name__ == "__main__":
    main()
