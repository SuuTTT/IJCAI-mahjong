"""Bitboard Othello (Reversi) on a 6x6 board.

We use 6x6 (a legitimate, well-studied Reversi variant) rather than 8x8 because
the teacher is a pure-Python alpha-beta minimax: on 8x8 a depth-6 search costs
~5.5 s, which makes the depth-6 data-generation AND the depth-5 opponent eval
ladder (hundreds of games) exceed the multi-hour budget. On 6x6 the full depth
ladder D in {1,2,4,6} stays fast, preserving the coherence-gradient design.

Squares indexed 0..35, index = row*6 + col, row 0 = top, col 0 = left.
Two ints `me` (player to move) and `opp`. Logic is from the mover's perspective;
apply_move returns the position already swapped to the next player.

Cross-validated against an independent array reference engine (test_othello.py).
"""

N = 6
FULL = (1 << 36) - 1
# per-row 6-bit group. col!=0 -> 0b111110=0x3E ; col!=5 -> 0b011111=0x1F
_ROWA = 0x3E  # not a-file (drop col 0)
_ROWH = 0x1F  # not h-file (drop col 5)
NOT_A = sum(_ROWA << (6 * r) for r in range(6))
NOT_H = sum(_ROWH << (6 * r) for r in range(6))


def _e(x):  return (x << 1) & NOT_A & FULL   # col +1
def _w(x):  return (x >> 1) & NOT_H          # col -1
def _n(x):  return (x >> 6)                  # row -1
def _s(x):  return (x << 6) & FULL           # row +1
def _ne(x): return (x >> 5) & NOT_A          # row-1 col+1
def _nw(x): return (x >> 7) & NOT_H          # row-1 col-1
def _se(x): return (x << 7) & NOT_A & FULL   # row+1 col+1
def _sw(x): return (x << 5) & NOT_H & FULL   # row+1 col-1

_DIRS = (_e, _w, _n, _s, _ne, _nw, _se, _sw)


def bit(r, c):
    return 1 << (r * 6 + c)


def initial():
    """Standard 6x6 start; returns (me=Black, opp=White). Black to move.
    White on (2,2),(3,3); Black on (2,3),(3,2)."""
    black = bit(2, 3) | bit(3, 2)
    white = bit(2, 2) | bit(3, 3)
    return black, white


def popcount(x):
    return bin(x).count("1")


def legal_moves(me, opp):
    empty = ~(me | opp) & FULL
    moves = 0
    for shift in _DIRS:
        t = shift(me) & opp
        t |= shift(t) & opp
        t |= shift(t) & opp
        t |= shift(t) & opp   # max flippable run on 6-wide board is 4
        moves |= shift(t) & empty
    return moves


def legal_move_list(me, opp):
    m = legal_moves(me, opp)
    out = []
    while m:
        lsb = m & (-m)
        out.append(lsb.bit_length() - 1)
        m ^= lsb
    return out


def apply_move(me, opp, sq):
    """Play square `sq` for `me`; return (new_me, new_opp) swapped to next player."""
    m = 1 << sq
    flips = 0
    for shift in _DIRS:
        t = shift(m) & opp
        t |= shift(t) & opp
        t |= shift(t) & opp
        t |= shift(t) & opp
        if shift(t) & me:
            flips |= t
    new_me = me | m | flips
    new_opp = opp & ~flips
    return new_opp, new_me  # swap perspective


def has_move(me, opp):
    return legal_moves(me, opp) != 0


def is_terminal(me, opp):
    if legal_moves(me, opp):
        return False
    if legal_moves(opp, me):
        return False
    return True


def winner_score(me, opp):
    """Final disc difference from `me` perspective. Positive => me wins."""
    return popcount(me) - popcount(opp)
