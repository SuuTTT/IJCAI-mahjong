"""
obs38.py — JAX-batched 38-plane CAIEST observation for the self-play env (matches feature.py /
pimc_obs.py byte-for-byte). This is what lets us WARM-START the RL policy from the SL net
(lad_chunjiandu) instead of training from random — the missing piece (per Tjong: SL first, then RL).

Layout (38,4,9) flattened over 34 tiles (+2 pad): plane0 SEAT_WIND, 1 PREVALENT_WIND,
2-5 HAND thermometer (cur), 6-21 DISCARD (4 planes/seat, relative 0=cur), 22-37 HALF_FLUSH (melds,
zero in discard-only env). Winds: seat wind = seat index; prevalent = 0 (round East).
"""
import jax, jax.numpy as jnp

NT = 34


def encode_obs38(hands, discards, cur):
    """hands (B,4,34) int counts, discards (B,4,34) int counts, cur (B,) int -> obs (B,38,4,9) float32.
    Discard-only env: no melds (HALF_FLUSH planes stay 0)."""
    B = cur.shape[0]; bi = jnp.arange(B)
    obs = jnp.zeros((B, 38, 36), jnp.float32)
    # SEAT_WIND (plane 0): F(cur+1) -> tile index 27+cur ; PREVALENT (plane1): F1 = 27
    obs = obs.at[bi, 0, 27 + cur].set(1.0)
    obs = obs.at[:, 1, 27].set(1.0)
    # HAND thermometer planes 2..5
    hand = hands[bi, cur, :]                                  # (B,34)
    for k in range(4):
        obs = obs.at[:, 2 + k, :34].set((hand >= (k + 1)).astype(jnp.float32))
    # DISCARD planes 6..21: 4 planes per relative seat j (0=cur)
    for j in range(4):
        src = (cur + j) % 4
        dpile = discards[bi, src, :]                          # (B,34) counts
        for k in range(4):
            obs = obs.at[:, 6 + 4 * j + k, :34].set((dpile >= (k + 1)).astype(jnp.float32))
    # HALF_FLUSH planes 22..37 stay 0 (no claims yet)
    return obs.reshape(B, 38, 4, 9)
