"""
csm_env.py — vectorized JAX environment for Chinese Standard Mahjong (R1 foundation).

GOAL: run B parallel games entirely on GPU (state as fixed arrays, jit+vmap transitions, batched
policy forward) to escape the ~246 g/s CPU self-play wall and enable scaled self-play RL.

SCOPE / MILESTONES (be honest about what's built):
  [x] Phase 1 (this file): state arrays, reset/deal, the draw→discard round-robin core, obs encoding,
      a fixed-size action space, and a THROUGHPUT benchmark (bench.py). Reward is a placeholder
      (wall-exhaustion draw) — enough to validate steps/s, not yet to train.
  [ ] Phase 2: win detection (agari/shanten) — the algorithmic core; reward = win/lose.
  [ ] Phase 3: claim mechanics (Chi/Peng/Gang/Hu priority after a discard) — turn order becomes
      non-round-robin; the genuinely tricky transition.
  [ ] Phase 4: 81-fan scoring (the multi-week part) — for the real duplicate-score reward.

Design notes:
  * State is a flat pytree of (B,...) arrays so jax.jit/vmap apply over the batch with no Python loop.
  * 34 tiles: W1-9=0-8, T1-9=9-17, B1-9=18-26, F1-4=27-30 (winds), J1-3=31-33 (dragons). 4 of each = 136.
  * Phase 1 is DISCARD-ONLY (each turn: draw 1, discard 1, advance). Claims/win added in Phase 2-3.
"""
import jax, jax.numpy as jnp
from jax import lax
from functools import partial

NT = 34          # tile types
WALL = 136       # 4 * 34
HAND0 = 13       # tiles per player at deal


def init_state(key, B):
    """Deal B games: shuffle a 136-tile wall, give 13 to each of 4 players, set player 0 to draw."""
    keys = jax.random.split(key, B)
    deck = jnp.repeat(jnp.arange(NT), 4)[None, :]                      # (1,136) tile ids, 4 of each
    deck = jnp.broadcast_to(deck, (B, WALL))
    perm = jax.vmap(lambda k: jax.random.permutation(k, WALL))(keys)   # (B,136) shuffle order
    wall = jnp.take_along_axis(deck, perm, axis=1)                     # (B,136) shuffled tile ids
    hands = jnp.zeros((B, 4, NT), jnp.int8)
    # deal first 52 tiles: 13 to each player (player p gets wall[13p:13p+13])
    def deal(hands, args):
        b = args
        return hands, None
    for p in range(4):
        idx = wall[:, p * HAND0:(p + 1) * HAND0]                        # (B,13) tile ids
        oh = jax.nn.one_hot(idx, NT, dtype=jnp.int8).sum(1)             # (B,34) counts
        hands = hands.at[:, p, :].set(oh)
    ptr = jnp.full((B,), 4 * HAND0, jnp.int32)                          # next undealt wall index
    state = {
        "wall": wall, "ptr": ptr, "hands": hands,
        "discards": jnp.zeros((B, 4, NT), jnp.int8),
        "cur": jnp.zeros((B,), jnp.int32),                              # current player to act
        "last": jnp.full((B,), -1, jnp.int32),                         # last discarded tile
        "done": jnp.zeros((B,), bool),
        "turn": jnp.zeros((B,), jnp.int32),
        "drawn": jnp.full((B,), -1, jnp.int32),                        # tile just drawn (in hand, awaiting discard)
    }
    # player 0 draws their first tile
    return _draw(state)


def _draw(state):
    """Current player draws the next wall tile (if any). If wall exhausted -> done (draw game)."""
    B = state["cur"].shape[0]; bi = jnp.arange(B)
    empty = state["ptr"] >= WALL
    safe_ptr = jnp.minimum(state["ptr"], WALL - 1)
    tile = state["wall"][bi, safe_ptr]                                  # (B,)
    add = jax.nn.one_hot(tile, NT, dtype=jnp.int8)                      # (B,34)
    hands = state["hands"].at[bi, state["cur"], :].add(jnp.where(empty[:, None], 0, add))
    return {**state, "hands": hands, "ptr": state["ptr"] + 1,
            "drawn": jnp.where(empty, -1, tile),
            "done": state["done"] | empty}


def legal_discards(state):
    """(B,34) bool mask: a tile is discardable iff the current player holds >=1."""
    B = state["cur"].shape[0]; bi = jnp.arange(B)
    return state["hands"][bi, state["cur"], :] > 0


def step(state, action):
    """action: (B,) tile index 0..33 to discard. Phase-1 round-robin: discard -> next player draws."""
    B = state["cur"].shape[0]; bi = jnp.arange(B)
    rem = jax.nn.one_hot(action, NT, dtype=jnp.int8)
    hands = state["hands"].at[bi, state["cur"], :].add(-rem)            # remove discarded tile
    discards = state["discards"].at[bi, state["cur"], :].add(rem)
    nxt = (state["cur"] + 1) % 4
    state = {**state, "hands": hands, "discards": discards, "last": action,
             "cur": nxt, "turn": state["turn"] + 1}
    state = _draw(state)
    reward = jnp.zeros((B,), jnp.float32)                              # Phase-1 placeholder (no win yet)
    return state, reward, state["done"]


def encode_obs(state):
    """Minimal obs for the policy: (B, P, 34) planes — current hand (4 count levels) + the 4 discard
    piles. Enough to drive a small conv policy for the throughput benchmark (full feature later)."""
    B = state["cur"].shape[0]; bi = jnp.arange(B)
    hand = state["hands"][bi, state["cur"], :]                         # (B,34) counts 0..4
    hand_planes = jnp.stack([(hand >= k).astype(jnp.float32) for k in (1, 2, 3, 4)], 1)  # (B,4,34)
    disc = state["discards"].astype(jnp.float32)                       # (B,4,34)
    return jnp.concatenate([hand_planes, disc], 1)                     # (B,8,34)


@partial(jax.jit, static_argnums=(2,))
def rollout(key, state, n_steps, policy_params=None):
    """Run n_steps of greedy-random play over the batch. Returns final state + total reward.
    (Benchmark harness uses this; a real policy net plugs into the action selection.)"""
    def body(carry, _):
        state, key = carry
        key, sk = jax.random.split(key)
        mask = legal_discards(state)
        # placeholder policy: uniform over legal discards (replace with net forward in bench.py)
        logits = jnp.where(mask, 0.0, -1e9)
        action = jax.random.categorical(sk, logits, axis=1)
        state, r, done = step(state, action)
        return (state, key), r
    (state, key), rs = lax.scan(body, (state, key), None, length=n_steps)
    return state, rs.sum(0)
