"""
agari_jax.py — JAX-vectorized winning-hand detection for the self-play env.

Uses the precomputed per-group feasibility tables (build_agari_tables.py):
  suit table (5^9 keys) and honor table (5^7 keys), each a uint16 bitmask over (n_sets 0..4, pair 0..1):
  bit (s*2 + p) set  <=>  that group's counts can decompose into s sets + p pair(s).
A hand wins iff, across the 4 groups (3 number suits + honors), some choice sums to `need` sets and
exactly 1 pair total. We compute that with a tiny fixed DP over (sets 0..4, pairs 0..1), fully
vmap-able over the batch. Validated against the numpy reference (agari.is_win_any) and MahjongGB.
"""
import os
import numpy as np
import jax, jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
POW5_9 = (5 ** np.arange(9)).astype(np.int64)
POW5_7 = (5 ** np.arange(7)).astype(np.int64)


def load_tables(suit_path=None, honor_path=None):
    suit = np.load(suit_path or os.path.join(_HERE, 'agari_suit.npy'))    # (5^9,) uint16
    honor = np.load(honor_path or os.path.join(_HERE, 'agari_honor.npy'))  # (5^7,) uint16
    return jnp.asarray(suit.astype(np.int32)), jnp.asarray(honor.astype(np.int32))


def _mask_to_sp(mask):
    """uint16 mask -> (5,2) bool feasibility over (sets 0..4, pair 0..1)."""
    bits = (mask[..., None, None] >> (jnp.arange(5)[:, None] * 2 + jnp.arange(2)[None, :])) & 1
    return bits.astype(bool)


def is_win_batch(hands, n_melds, suit_tab, honor_tab):
    """hands: (B,34) int counts (concealed). n_melds: (B,) int. Returns (B,) bool standard-win.
    Vectorized: pack keys -> gather masks -> DP-combine for (need sets, 1 pair)."""
    B = hands.shape[0]
    need = 4 - n_melds                                                  # (B,) sets required
    keys = [jnp.dot(hands[:, 9 * s:9 * s + 9].astype(jnp.int64), POW5_9) for s in range(3)]
    hk = jnp.dot(hands[:, 27:34].astype(jnp.int64), POW5_7)
    group_sp = [_mask_to_sp(suit_tab[keys[s]]) for s in range(3)] + [_mask_to_sp(honor_tab[hk])]  # 4 x (B,5,2)
    # DP over groups: R[b, s, p] reachable; start R[b,0,0]=True
    R = jnp.zeros((B, 5, 2), bool).at[:, 0, 0].set(True)
    for g in group_sp:
        newR = jnp.zeros_like(R)
        for sg in range(5):
            for pg in range(2):
                feas = g[:, sg, pg]                                     # (B,)
                if sg == 0 and pg == 0:
                    newR = newR | (R & feas[:, None, None])
                    continue
                shifted = jnp.zeros_like(R)
                # add (sg,pg): R[s,p] -> [s+sg, p+pg] within bounds
                s_src = jnp.arange(5)
                valid_s = s_src + sg <= 4
                p_src = jnp.arange(2)
                valid_p = p_src + pg <= 1
                src = R[:, :5, :]
                # build shifted via static slicing (sg,pg are python ints here)
                tmp = jnp.zeros_like(R)
                tmp = tmp.at[:, sg:5, pg:2].set(R[:, :5 - sg, :2 - pg])
                shifted = tmp
                newR = newR | (shifted & feas[:, None, None])
        R = newR
    bi = jnp.arange(B)
    return R[bi, need, 1]


def is_seven_pairs_batch(hands, n_melds):
    counts_ok = jnp.all((hands == 0) | (hands == 2) | (hands == 4), axis=1)
    pairs = jnp.sum(hands // 2, axis=1)
    return (n_melds == 0) & (jnp.sum(hands, axis=1) == 14) & counts_ok & (pairs == 7)


def is_win_any_batch(hands, n_melds, suit_tab, honor_tab):
    return is_win_batch(hands, n_melds, suit_tab, honor_tab) | is_seven_pairs_batch(hands, n_melds)


def reach_and_phi(hands, n_melds, suit_tab, honor_tab):
    """Returns (is_win(B,), phi(B,)). phi = max over reachable (sets,pair) of (sets + 0.5*pair),
    a dense 'completion potential' in [0,4.5] for reward shaping (4 sets + pair = 4.5 = a win)."""
    B = hands.shape[0]; need = 4 - n_melds
    keys = [jnp.dot(hands[:, 9 * s:9 * s + 9].astype(jnp.int64), POW5_9) for s in range(3)]
    hk = jnp.dot(hands[:, 27:34].astype(jnp.int64), POW5_7)
    groups = [_mask_to_sp(suit_tab[keys[s]]) for s in range(3)] + [_mask_to_sp(honor_tab[hk])]
    R = jnp.zeros((B, 5, 2), bool).at[:, 0, 0].set(True)
    for g in groups:
        newR = jnp.zeros_like(R)
        for sg in range(5):
            for pg in range(2):
                feas = g[:, sg, pg]
                if sg == 0 and pg == 0:
                    newR = newR | (R & feas[:, None, None]); continue
                tmp = jnp.zeros_like(R).at[:, sg:5, pg:2].set(R[:, :5 - sg, :2 - pg])
                newR = newR | (tmp & feas[:, None, None])
        R = newR
    bi = jnp.arange(B)
    val = (jnp.arange(5)[:, None] + 0.5 * jnp.arange(2)[None, :])           # (5,2) value of each (s,p)
    phi = jnp.max(jnp.where(R, val[None], -1.0), axis=(1, 2))
    return R[bi, need, 1], phi
