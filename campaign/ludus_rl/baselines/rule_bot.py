"""rule_v0: scripted heuristic bot (WO-P0-02) — a ladder calibration anchor.

Pure JAX, jit/vmap-safe, deterministic given (state, player):
  1. DEFEND: if enemy units are on our half, play the highest-dps affordable unit
     card on top of the biggest threat.
  2. PUSH: if energy is nearly full, play the highest-hp affordable unit at the
     bridge of the lane whose enemy turret is weakest.
  3. Otherwise no-op (bank energy).

Acts at most once per BOT_PERIOD ticks (callers gate the cadence). Returns a
PLAYER-CENTRIC [slot, x, y] triple like every agent."""

from __future__ import annotations

import jax.numpy as jnp

from boom import engine
from boom.engine import C, E_UNIT, FP, H, W

RIVER_HI_FP_ = 17 * FP
PUSH_ENERGY = 9 * E_UNIT
NOOP = jnp.array([4, 0, 0], jnp.int32)


def _to_player_frame(player, x_tile, y_tile):
    px = jnp.where(player == 1, W - 1 - x_tile, x_tile)
    py = jnp.where(player == 1, H - 1 - y_tile, y_tile)
    return px, py


def rule_v0_action(state: engine.State, player) -> jnp.ndarray:
    """(3,) int32 player-centric action triple."""
    hand = state.hand[player]                        # (4,)
    energy = state.energy[player]
    afford = energy >= C.cost[hand] * E_UNIT
    is_unit = C.is_spell[hand] == 0
    playable = afford & is_unit & (jnp.sum(state.u_hp <= 0) >= C.count[hand] + 8)

    # --- threat assessment (engine frame) ---
    alive = state.u_hp > 0
    enemy = alive & (state.u_owner != player)
    on_our_half = jnp.where(player == 0,
                            state.u_y < 15 * FP,
                            state.u_y >= 17 * FP)
    threat = enemy & on_our_half
    threat_hp = jnp.where(threat, state.u_hp, 0)
    biggest = jnp.argmax(threat_hp)                  # 0 if none; gated below
    has_threat = threat.any()

    # defender: highest-dps affordable unit card
    dps = jnp.where(C.period[hand] > 0, C.dmg[hand] * 5 // jnp.maximum(C.period[hand], 1), 0)
    def_slot = jnp.argmax(jnp.where(playable, dps, -1))
    can_defend = playable.any() & has_threat

    tx = jnp.clip(state.u_x[biggest] // FP, 0, W - 1)
    ty_raw = state.u_y[biggest] // FP
    # clamp placement into our legal half (engine frame), then convert
    ty = jnp.where(player == 0, jnp.clip(ty_raw, 0, 14), jnp.clip(ty_raw, 17, H - 1))
    dx, dy = _to_player_frame(player, tx, ty)
    defend = jnp.stack([def_slot.astype(jnp.int32), dx, dy])

    # --- push (engine frame): weakest enemy turret's lane bridge ---
    enemy_turrets = jnp.where(player == 0,
                              state.tower_hp[3:5], state.tower_hp[0:2])
    # prefer a damaged-but-alive turret lane; if one is dead, push the other
    lane = jnp.where(enemy_turrets[0] == 0, 1,
                     jnp.where(enemy_turrets[1] == 0, 0,
                               jnp.argmin(enemy_turrets)))
    bx = jnp.where(lane == 0, 4, 13)
    by = jnp.where(player == 0, 14, 17)              # own edge of the bridge
    px, py = _to_player_frame(player, bx, by)

    # start a push with the beefiest card at (near) full energy; while own units
    # are already on the enemy half, SUSTAIN the push with dps cards — CR-strength
    # towers shred lone attackers, so single-unit pushes just donate elixir
    on_enemy_half = alive & (state.u_owner == player) & jnp.where(
        player == 0, state.u_y >= RIVER_HI_FP_, state.u_y < 15 * FP)
    pushing = on_enemy_half.any()
    start_slot = jnp.argmax(jnp.where(playable, C.hp[hand], -1))
    sustain_slot = jnp.argmax(jnp.where(playable, dps, -1))
    push_slot = jnp.where(pushing, sustain_slot, start_slot)
    push = jnp.stack([push_slot.astype(jnp.int32), px, py])
    can_push = playable.any() & (
        (energy >= PUSH_ENERGY) | (pushing & (energy >= 5 * E_UNIT)))

    return jnp.where(can_defend, defend, jnp.where(can_push, push, NOOP)).astype(jnp.int32)
