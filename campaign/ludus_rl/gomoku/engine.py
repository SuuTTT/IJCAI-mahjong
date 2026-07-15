"""Gomoku (freestyle five-in-a-row), 15x15 — Ludus game #2.

Same contract as boom: pure-functional JAX, int32 state, deterministic,
jit/vmap-safe. Turn-based: one flat action (y*15+x) per step by the player to
move. Replay = the move list.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

N = 15
N_ACTIONS = N * N
ENV_VERSION = "gomoku/v1"


class State(NamedTuple):
    board: jax.Array      # (N, N) int32: 0 empty, 1 black, 2 white
    to_move: jax.Array    # int32: 0 black, 1 white (black plays first)
    moves: jax.Array      # int32 move count
    result: jax.Array     # int32: -1 ongoing, 0 black wins, 1 white wins, 2 draw


def reset(_key=None) -> State:
    return State(board=jnp.zeros((N, N), jnp.int32),
                 to_move=jnp.int32(0),
                 moves=jnp.int32(0),
                 result=jnp.int32(-1))


def legal(state: State) -> jax.Array:
    """(225,) bool — empty cells while the game is ongoing."""
    open_cells = (state.board == 0).reshape(-1)
    return open_cells & (state.result == -1)


def _line_win(board: jax.Array, stone: jax.Array) -> jax.Array:
    """Any run of 5+ `stone` in a row/col/diag/anti-diag."""
    m = (board == stone).astype(jnp.int32)

    def count_dir(dy: int, dx: int):
        acc = m
        cur = m
        for _ in range(4):
            cur = jnp.roll(cur, (dy, dx), (0, 1))
            # zero the wrapped edge
            if dy > 0:
                cur = cur.at[:dy, :].set(0)
            if dx > 0:
                cur = cur.at[:, :dx].set(0)
            acc = acc * cur
        return acc.any()

    return (count_dir(0, 1) | count_dir(1, 0) |
            count_dir(1, 1) | _anti_diag(m))


def _anti_diag(m: jax.Array) -> jax.Array:
    acc = m
    cur = m
    for _ in range(4):
        cur = jnp.roll(cur, (1, -1), (0, 1))
        cur = cur.at[:1, :].set(0)
        cur = cur.at[:, -1:].set(0)
        acc = acc * cur
    return acc.any()


def step(state: State, action: jax.Array) -> State:
    """action: flat int in [0, 225). Illegal or post-game actions are no-ops."""
    a = jnp.clip(action, 0, N_ACTIONS - 1)
    y, x = a // N, a % N
    ok = (state.board[y, x] == 0) & (state.result == -1)
    stone = state.to_move + 1
    board = jnp.where(ok, state.board.at[y, x].set(stone), state.board)
    won = _line_win(board, stone) & ok
    moves = state.moves + ok.astype(jnp.int32)
    full = moves >= N_ACTIONS
    result = jnp.where(won, state.to_move.astype(jnp.int32),
                       jnp.where(full, jnp.int32(2), jnp.int32(-1)))
    result = jnp.where(state.result != -1, state.result, result)
    to_move = jnp.where(ok, 1 - state.to_move, state.to_move)
    return State(board=board, to_move=to_move, moves=moves, result=result)


def observe(state: State, player: int) -> jax.Array:
    """(N, N, 3) float32: [my stones, their stones, to-move flag plane]."""
    mine = (state.board == player + 1).astype(jnp.float32)
    theirs = (state.board == 2 - player).astype(jnp.float32)
    turn = jnp.full((N, N), (state.to_move == player).astype(jnp.float32))
    return jnp.stack([mine, theirs, turn], axis=-1)


def result(state: State) -> jax.Array:
    return state.result
