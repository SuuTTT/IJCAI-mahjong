# SMAX Challenge — build a micro-battle agent

StarCraft-like multi-agent micro on JaxMARL's SMAX (Apache-2.0). You control
the **ally squad**; the enemy runs the built-in heuristic (closest-attack).
The leaderboard is **per-scenario win rate** — cooperative MARL, not
head-to-head (that comes with self-play divisions later).

## Scenarios (the ladder)

| scenario | allies | enemies | flavor |
|---|---|---|---|
| 2s3z | 2 stalkers + 3 zealots | same | mixed melee/ranged (house baseline: 1.00 win) |
| 3s5z | 3 stalkers + 5 zealots | same | bigger mixed brawl (house run in flight) |
| 5m_vs_6m | 5 marines | 6 marines | outnumbered — focus-fire discipline |
| 10m_vs_11m | 10 marines | 11 marines | large outnumbered fight |
| MMM | marines+marauders+medivac | same | composition micro |

## Interface (JaxMARL)

```python
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
env = make("HeuristicEnemySMAX", scenario=map_name_to_scenario("2s3z"),
           see_enemy_actions=True, walls_cause_death=True, attack_mode="closest")
obs, state = env.reset(key)              # obs: dict per ally, 127-dim on 2s3z
acts = {a: policy(obs[a]) for a in env.agents}
obs, state, rew, dones, info = env.step(key, state, acts)
```

- **Obs** (per ally): own features + visible allies/enemies (pos, hp, type,
  cooldown) — dim varies per scenario; the env bakes the agent id in.
- **Actions** (discrete): 4 moves + stop + one attack slot per enemy
  (2s3z: 10 actions). Use `env.get_avail_actions(state)` and mask.
- **Episode end**: one side wiped or timeout. **Gotcha we hit ourselves:**
  envs auto-reset inside `step`, so read wins from
  `SMAXLogWrapper`'s `info["returned_won_episode"]`, never from the
  post-done state.
- **Determinism**: everything is JAX — (scenario, seed, policy) reproduces
  the episode bit-for-bit. Replays are recorded exactly that way.

## Submission (upload tier)

Same policy as Boom bots: submit msgpack Flax params for the reference
MAPPO-RNN architecture (`FC 128 / GRU 128`, `mappo_rnn_smax.ActorRNN`) via
the bots page with `game=smax`; we roll N=32 eval episodes per scenario and
post win rate + Wilson CI. Custom architectures land with the per-game arch
registry (roadmap).

## Watching

`/smax` replays recorded episodes in 3D — every unit, shot, and death is a
faithful playback of the deterministic sim.
