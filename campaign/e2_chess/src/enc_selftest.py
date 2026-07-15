#!/usr/bin/env python3
"""Round-trip self-test for chess_enc: every legal move in random games must
map move->index->move identically, and indices must be unique per position."""
import random, sys
import chess
from chess_enc import move_to_index, index_to_move, encode_board, N_ACTIONS

random.seed(0)
positions = 0
moves_checked = 0
for g in range(200):
    board = chess.Board()
    while not board.is_game_over() and board.ply() < 120:
        legal = list(board.legal_moves)
        seen = set()
        for mv in legal:
            idx = move_to_index(mv)
            assert 0 <= idx < N_ACTIONS, (idx, mv)
            assert idx not in seen, ("collision", mv, board.fen())
            seen.add(idx)
            back = index_to_move(idx, board)
            assert back == mv, ("roundtrip", mv, back, board.fen())
            moves_checked += 1
        x = encode_board(board)
        assert x.shape == (18, 8, 8) and x.dtype.name == "uint8"
        assert int(x[:12].sum()) == len(board.piece_map())
        positions += 1
        board.push(random.choice(legal))
print(f"OK positions={positions} moves_checked={moves_checked}")
