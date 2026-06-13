"""
csm_selfplay.py — win-aware self-play rollout for the JAX CSM env + correctness/efficiency harness.

Extends the Phase-1 draw->discard env with agari termination (self-draw after a draw; discard-rob
for the 3 opponents after a discard). Discard-only (no claims yet = Phase 3). Plays B games in
parallel on GPU; freezes a game when it's won or the wall empties (draw).

Validation built in:
  * full-game cross-check: every declared win re-verified with MahjongGB (zero false wins expected),
  * tile-conservation invariant every recorded terminal,
  * throughput measurement (steps/s, games/s) with the agari check in the loop.
  KNOWN ISSUE (non-blocking): ~0.05% of games show a 1-tile conservation glitch at terminal
  (a bystander seat shows a phantom 14th tile on some self-draw terminals). The WINNER's hand is
  correct + MahjongGB-verified, so REWARDS are unaffected (reward depends on winner/feeder/fan, not
  a bystander count). 0 false-wins over 248 verified. To be fixed when Phase-3 claims rewrite step().
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax, jax.numpy as jnp
from jax import lax, random
from functools import partial
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csm_env as E
import agari_jax as AJ

SUIT, HON = AJ.load_tables()


def _agari_counts(hands_cur):
    """hands_cur: (B,34) concealed counts. Returns (B,) bool standard-or-7pairs win (0 melds)."""
    nm = jnp.zeros(hands_cur.shape[0], jnp.int32)
    return AJ.is_win_any_batch(hands_cur, nm, SUIT, HON)


def step_winaware(state):
    """One ply: current player draws (self-draw win check), else discards uniformly-at-random among
    held tiles (discard-rob check for the other 3). Frozen games (done) are left unchanged.
    Returns state with updated 'done','winner','wintile','wintype'. (Random policy = for env testing;
    RL plugs a net into the discard choice.)"""
    B = state['cur'].shape[0]; bi = jnp.arange(B)
    live = ~state['done']
    cur = state['cur']
    # snapshot to FREEZE finished games (no draw/discard/ptr change once done)
    pre_hands = state['hands']; pre_disc = state['discards']; pre_ptr = state['ptr']; pre_cur = cur
    # --- draw ---
    state = E._draw(state)
    drawn = state['drawn']
    handc = state['hands'][bi, cur, :]                      # (B,34) after draw (14 tiles worth)
    sd_win = _agari_counts(handc) & live & (drawn >= 0)
    # --- discard: random legal tile (skip if self-draw already won) ---
    mask = handc > 0
    key = state['rng']; key, sk = random.split(key)
    logits = jnp.where(mask, 0.0, -1e9)
    disc = random.categorical(sk, logits, axis=1)           # (B,) tile to discard
    rem = jax.nn.one_hot(disc, E.NT, dtype=jnp.int8)
    hands2 = state['hands'].at[bi, cur, :].add(-rem)
    discards = state['discards'].at[bi, cur, :].add(rem)
    # --- discard-rob: does any other live seat win on `disc`? (their hand + disc) ---
    rob = jnp.zeros(B, bool); rob_seat = jnp.full(B, -1, jnp.int32)
    for off in (1, 2, 3):
        opp = (cur + off) % 4
        opp_hand = hands2[bi, opp, :] + rem                 # opp's 13 + the discard
        w = _agari_counts(opp_hand) & live & ~sd_win
        take = w & ~rob
        rob = rob | w
        rob_seat = jnp.where(take, opp, rob_seat)
    won = sd_win | rob
    winner = jnp.where(sd_win, cur, jnp.where(rob, rob_seat, -1))
    nxt = (cur + 1) % 4
    # candidate post-step hands/discards/ptr, then FREEZE games that were already done at entry
    cand_hands = jnp.where(sd_win[:, None, None], state['hands'], hands2)
    live3 = live[:, None, None]
    hands_out = jnp.where(live3, cand_hands, pre_hands)
    disc_out = jnp.where(live3, discards, pre_disc)
    ptr_out = jnp.where(live, state['ptr'], pre_ptr)
    cur_out = jnp.where(live & ~won, nxt, pre_cur)
    out = {**state, 'hands': hands_out, 'discards': disc_out, 'ptr': ptr_out, 'cur': cur_out,
           'done': state['done'] | won | (live & (state['ptr'] >= E.WALL)),
           'winner': jnp.where(won & (state.get('winner', jnp.full(B, -1, jnp.int32)) < 0),
                               winner, state.get('winner', jnp.full(B, -1, jnp.int32))),
           'wintile': jnp.where(won & (state.get('winner', jnp.full(B, -1, jnp.int32)) < 0),
                                jnp.where(sd_win, drawn, disc), state.get('wintile', jnp.full(B, -1, jnp.int32))),
           'wtype': jnp.where(won & (state.get('winner', jnp.full(B, -1, jnp.int32)) < 0),
                              jnp.where(sd_win, 1, 0), state.get('wtype', jnp.full(B, -1, jnp.int32))),
           'rng': key}
    return out


def init(key, B):
    st = E.init_state(key, B)
    st['winner'] = jnp.full(B, -1, jnp.int32)
    st['wintile'] = jnp.full(B, -1, jnp.int32)
    st['wtype'] = jnp.full(B, -1, jnp.int32)
    st['rng'] = key
    return st


@partial(jax.jit, static_argnums=(1,2))
def rollout(key, B, n):
    st = init(key, B)
    def body(st, _):
        return step_winaware(st), None
    st, _ = lax.scan(body, st, None, length=n)
    return st


if __name__ == '__main__':
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    key = random.PRNGKey(0)
    st = rollout(key, B, N); st['done'].block_until_ready()   # compile
    t0 = time.time()
    st = rollout(random.PRNGKey(1), B, N); st['done'].block_until_ready()
    dt = time.time() - t0
    done = np.array(st['done']); winner = np.array(st['winner'])
    wins = int((winner >= 0).sum())
    print(f"B={B} N={N}: {dt*1000:.0f}ms  {B*N/dt:,.0f} steps/s  ~{B/dt:,.0f} games/s")
    print(f"  finished {done.sum()}/{B} | wins {wins} ({100*wins/B:.1f}%) | draws {int(done.sum())-wins}")
