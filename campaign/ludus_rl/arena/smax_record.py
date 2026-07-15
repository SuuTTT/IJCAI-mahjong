"""Record a SMAX episode with the trained MAPPO actor -> scene-frame replay JSON.

Frames follow the docs/09 schema spirit: entities + shot events per tick.
Run on the box:  python -m arena.smax_record --map 2s3z \
    --params /root/ludus_train/smax_mappo/actor_params.msgpack \
    --out /root/ludus_replays_smax/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/root/JaxMARL/baselines/MAPPO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="2s3z")
    ap.add_argument("--params", default="/root/ludus_train/smax_mappo/actor_params.msgpack")
    ap.add_argument("--out", default="/root/ludus_replays_smax")
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()

    from flax.serialization import from_bytes
    from jaxmarl import make
    from jaxmarl.environments.smax import map_name_to_scenario
    from jaxmarl.wrappers.baselines import SMAXLogWrapper
    from mappo_rnn_smax import ActorRNN, ScannedRNN

    # LogWrapper: envs auto-reset inside step, so the terminal state is
    # unobservable — the win flag must come from info["returned_won_episode"]
    env = SMAXLogWrapper(make("HeuristicEnemySMAX",
               scenario=map_name_to_scenario(cfg.map),
               see_enemy_actions=True, walls_cause_death=True,
               attack_mode="closest"))
    key = jax.random.PRNGKey(cfg.seed)
    obs, state = env.reset(key)

    n_agents = len(env.agents)          # allies only (num_agents counts all units)
    actor = ActorRNN(env.action_space(env.agents[0]).n,
                     config={"FC_DIM_SIZE": 128, "GRU_HIDDEN_DIM": 128})
    # template init to load params
    obs_dim = obs[env.agents[0]].shape[-1]   # env already includes agent id
    n_act = env.action_space(env.agents[0]).n
    h = ScannedRNN.initialize_carry(n_agents, 128)
    dummy = (jnp.zeros((1, n_agents, obs_dim)), jnp.zeros((1, n_agents), bool),
             jnp.ones((1, n_agents, n_act)))
    params = actor.init(key, h, dummy)
    params = from_bytes(params, Path(cfg.params).read_bytes())

    frames = []
    done = False
    t = 0
    agent_ids = jnp.eye(n_agents)
    won = False
    while not done and t < 400:
        st = state
        for attr in ("env_state", "state"):
            st = getattr(st, attr, st)
        units = []
        pos = np.asarray(st.unit_positions)
        hp = np.asarray(st.unit_health)
        alive = np.asarray(st.unit_alive)
        teams = np.asarray(st.unit_teams)
        types = np.asarray(st.unit_types)
        atk = np.asarray(st.prev_attack_actions)
        cd = np.asarray(st.unit_weapon_cooldowns)
        for i in range(len(hp)):
            units.append({"id": i, "x": float(pos[i][0]), "y": float(pos[i][1]),
                          "hp": float(hp[i]), "alive": int(alive[i]),
                          "team": int(teams[i]), "type": int(types[i])})
        shots = []
        if frames:
            prev_cd = frames[-1]["_cd"]
            for i in range(len(hp)):
                if alive[i] and cd[i] > prev_cd[i] and atk[i] > 0:
                    tgt = int(atk[i])
                    if tgt < len(hp):
                        shots.append({"from": i, "to": tgt})
        frames.append({"t": t, "units": units, "shots": shots,
                       "_cd": [float(c) for c in cd]})

        obs_in = jnp.stack([obs[a] for a in env.agents])[None]
        dones_in = jnp.zeros((1, n_agents), bool)
        avail = env.get_avail_actions(getattr(state, "env_state", state))
        avail_in = jnp.stack([avail[a] for a in env.agents])[None].astype(jnp.float32)
        h, pi = actor.apply(params, h, (obs_in, dones_in, avail_in))
        key, ka, ks = jax.random.split(key, 3)
        acts = pi.mode()[0]                  # greedy: match eval protocol
        actions = {a: acts[i] for i, a in enumerate(env.agents)}
        obs, state, rew, dones, info = env.step(ks, state, actions)
        done = bool(dones["__all__"])
        if done:
            won = bool(np.asarray(info["returned_won_episode"]).ravel()[0] > 0)
        t += 1

    for f in frames:
        f.pop("_cd")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"game": "smax", "map": cfg.map, "seed": cfg.seed, "ticks": t,
           "ally_won": won, "n_allies": n_agents, "frames": frames}
    p = out / f"smax_{cfg.map}_{int(time.time())}.json"
    p.write_text(json.dumps(rec))
    print(json.dumps({"saved": str(p), "ticks": t, "ally_won": won}))


if __name__ == "__main__":
    main()
