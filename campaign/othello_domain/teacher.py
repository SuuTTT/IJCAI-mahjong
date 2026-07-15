"""Alpha-beta minimax Othello teacher. Depth D controls strength/coherence.

Heuristic = positional weight matrix (corners high, X/C squares negative)
+ mobility term. Near a full board it shifts toward pure disc difference.
All evaluations are from the mover's perspective (negamax convention).
"""
import othello as oth

# positional weights for 6x6, indexed by square 0..35. Corners high, X/C sq neg.
_W = [
    100, -20,  10,  10, -20, 100,
    -20, -40,  -5,  -5, -40, -20,
     10,  -5,   3,   3,  -5,  10,
     10,  -5,   3,   3,  -5,  10,
    -20, -40,  -5,  -5, -40, -20,
    100, -20,  10,  10, -20, 100,
]

INF = 10 ** 9
_TOTAL = 36


def evaluate(me, opp):
    """Static heuristic from `me` perspective."""
    filled = oth.popcount(me) + oth.popcount(opp)
    # positional
    pos = 0
    x = me
    while x:
        lsb = x & (-x); i = lsb.bit_length() - 1; pos += _W[i]; x ^= lsb
    x = opp
    while x:
        lsb = x & (-x); i = lsb.bit_length() - 1; pos -= _W[i]; x ^= lsb
    # mobility
    my_mob = oth.popcount(oth.legal_moves(me, opp))
    op_mob = oth.popcount(oth.legal_moves(opp, me))
    mob = 0
    if my_mob + op_mob != 0:
        mob = 100 * (my_mob - op_mob) // (my_mob + op_mob)
    # disc difference (weighted up as board fills / endgame)
    disc = oth.popcount(me) - oth.popcount(opp)
    if filled >= 30:  # endgame (36 squares): discs dominate
        return 100 * disc + pos + 5 * mob
    return pos + 8 * mob + disc


def _order(moves, me, opp):
    # order by static square weight desc for better alpha-beta pruning
    return sorted(moves, key=lambda s: _W[s], reverse=True)


def _negamax(me, opp, depth, alpha, beta):
    if depth == 0:
        return evaluate(me, opp)
    moves = oth.legal_move_list(me, opp)
    if not moves:
        # pass, unless terminal
        if not oth.legal_moves(opp, me):
            # terminal: exact disc score, scaled huge to dominate heuristic
            d = oth.popcount(me) - oth.popcount(opp)
            return (1 if d > 0 else -1 if d < 0 else 0) * 100000 + d
        # pass: swap, negate
        return -_negamax(opp, me, depth - 1, -beta, -alpha)
    best = -INF
    for s in _order(moves, me, opp):
        nme, nopp = oth.apply_move(me, opp, s)
        val = -_negamax(nme, nopp, depth - 1, -beta, -alpha)
        if val > best:
            best = val
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def best_move(me, opp, depth):
    """Proper alpha-beta root search -> best move square (None if must pass)."""
    moves = oth.legal_move_list(me, opp)
    if not moves:
        return None
    best = -INF
    best_mv = moves[0]
    alpha, beta = -INF, INF
    for s in _order(moves, me, opp):
        nme, nopp = oth.apply_move(me, opp, s)
        val = -_negamax(nme, nopp, depth - 1, -beta, -alpha)
        if val > best:
            best = val
            best_mv = s
        if best > alpha:
            alpha = best
    return best_mv


def search(me, opp, depth):
    """Return (best_move, {move: value}); full-window per root move (for sampling).
    Slower than best_move; only used when soft/temperature values are needed."""
    moves = oth.legal_move_list(me, opp)
    if not moves:
        return None, {}
    vals = {}
    best = -INF
    best_mv = moves[0]
    for s in _order(moves, me, opp):
        nme, nopp = oth.apply_move(me, opp, s)
        val = -_negamax(nme, nopp, depth - 1, -INF, INF)
        vals[s] = val
        if val > best:
            best = val
            best_mv = s
    return best_mv, vals
