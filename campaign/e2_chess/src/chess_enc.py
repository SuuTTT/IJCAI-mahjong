"""E2 chess BC: board/move encoding (shared by encoder, ladder, models).

OBS: 8x8x18 uint8 planes, ABSOLUTE orientation (rank 0 = rank 1 / a1 row; no flipping).
  0-5   white P,N,B,R,Q,K
  6-11  black P,N,B,R,Q,K
  12    side to move (all-ones if white to move, zeros if black)
  13-16 castling rights: white-K, white-Q, black-K, black-Q (all-ones planes)
  17    en-passant target square (one-hot, zeros if none)

ACTION: AlphaZero-style 4672 = from_square(64) * 73 + move_plane(73), absolute coords.
  planes 0-55  queen-like slide: dir(8: N,NE,E,SE,S,SW,W,NW) * 7 + (distance-1)
               (queen promotions are encoded as ordinary pawn slides)
  planes 56-63 knight moves, fixed (df,dr) order
  planes 64-72 underpromotions: (df+1: capture-left/push/capture-right) * 3 + {N,B,R}
"""
import chess
import numpy as np

N_PLANES = 18
N_ACTIONS = 4672

_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]  # N NE E SE S SW W NW
_KNIGHT = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_KNIGHT_IDX = {d: i for i, d in enumerate(_KNIGHT)}
_DIR_IDX = {d: i for i, d in enumerate(_DIRS)}
_UNDERPROMO = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}


def encode_board(board: chess.Board) -> np.ndarray:
    """Board -> (18,8,8) uint8. Layout: planes[c, rank, file]."""
    x = np.zeros((N_PLANES, 8, 8), dtype=np.uint8)
    for sq, piece in board.piece_map().items():
        c = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        x[c, chess.square_rank(sq), chess.square_file(sq)] = 1
    if board.turn == chess.WHITE:
        x[12] = 1
    if board.has_kingside_castling_rights(chess.WHITE):
        x[13] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        x[14] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        x[15] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        x[16] = 1
    if board.ep_square is not None:
        x[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1
    return x


def move_to_index(move: chess.Move) -> int:
    f, t = move.from_square, move.to_square
    df = chess.square_file(t) - chess.square_file(f)
    dr = chess.square_rank(t) - chess.square_rank(f)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        plane = 64 + (df + 1) * 3 + _UNDERPROMO[move.promotion]
    elif (df, dr) in _KNIGHT_IDX:
        plane = 56 + _KNIGHT_IDX[(df, dr)]
    else:
        dist = max(abs(df), abs(dr))
        d = (0 if df == 0 else df // abs(df), 0 if dr == 0 else dr // abs(dr))
        plane = _DIR_IDX[d] * 7 + (dist - 1)
    return f * 73 + plane


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Inverse map; adds queen promotion when a pawn slide reaches the last rank."""
    f, plane = divmod(idx, 73)
    fr, ff = chess.square_rank(f), chess.square_file(f)
    if plane < 56:
        d, dist = divmod(plane, 7)
        df, dr = _DIRS[d][0] * (dist + 1), _DIRS[d][1] * (dist + 1)
        promo = None
    elif plane < 64:
        df, dr = _KNIGHT[plane - 56]
        promo = None
    else:
        u = plane - 64
        df, dr = u // 3 - 1, (1 if fr == 6 else -1)
        promo = [chess.KNIGHT, chess.BISHOP, chess.ROOK][u % 3]
    tr, tf = fr + dr, ff + df
    if not (0 <= tr < 8 and 0 <= tf < 8):
        return chess.Move.null()
    t = chess.square(tf, tr)
    if promo is None and board.piece_type_at(f) == chess.PAWN and tr in (0, 7):
        promo = chess.QUEEN
    return chess.Move(f, t, promotion=promo)


def legal_action_mask(board: chess.Board) -> np.ndarray:
    m = np.zeros(N_ACTIONS, dtype=bool)
    for mv in board.legal_moves:
        m[move_to_index(mv)] = True
    return m
