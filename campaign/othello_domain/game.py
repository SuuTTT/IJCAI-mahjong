"""Game play + match harness. Policies are callables policy(me, opp) -> square,
operating from the MOVER's perspective (me = current player's discs)."""
import random
import othello as oth


def random_opening(rng, n_plies):
    """Return a list of legal opening squares (played from the start) of length
    up to n_plies. Regenerated deterministically from rng."""
    me, opp = oth.initial()
    seq = []
    for _ in range(n_plies):
        moves = oth.legal_move_list(me, opp)
        if not moves:
            me, opp = opp, me
            if not oth.legal_moves(me, opp):
                break
            moves = oth.legal_move_list(me, opp)
        sq = rng.choice(moves)
        seq.append(sq)
        me, opp = oth.apply_move(me, opp, sq)
    return seq


def play_game(black_policy, white_policy, opening=None):
    """Play one game. `opening` = forced opening squares (applied to whoever is to
    move, in order). Returns (result, black_discs, white_discs) with result
    +1 black win, 0 draw, -1 white win."""
    black, white = oth.initial()
    btm = True
    ply = 0
    opening = opening or []
    while True:
        me, opp = (black, white) if btm else (white, black)
        if oth.is_terminal(me, opp):
            break
        moves = oth.legal_move_list(me, opp)
        if not moves:
            btm = not btm
            continue
        if ply < len(opening):
            sq = opening[ply]
            if sq not in moves:      # opening desync (rare) -> fall back
                sq = (black_policy if btm else white_policy)(me, opp)
        else:
            sq = (black_policy if btm else white_policy)(me, opp)
        ret_me, ret_opp = oth.apply_move(me, opp, sq)  # (nextplayer_me, nextplayer_opp)
        if btm:
            black, white = ret_opp, ret_me   # mover(black) discs = ret_opp
        else:
            white, black = ret_opp, ret_me
        btm = not btm
        ply += 1
    bc, wc = oth.popcount(black), oth.popcount(white)
    res = 1 if bc > wc else (-1 if wc > bc else 0)
    return res, bc, wc


def match(policy_a, policy_b, n_pairs, open_plies=4, seed=0):
    """Play n_pairs paired games (a=black then a=white on the same opening).
    Returns dict with a_winrate (draws=0.5), wins/draws/losses for A, n_games."""
    rng = random.Random(seed)
    a_pts = 0.0
    w = d = l = 0
    ng = 0
    for i in range(n_pairs):
        opening = random_opening(random.Random((seed << 20) ^ i), open_plies)
        # game 1: A black, B white
        res, _, _ = play_game(policy_a, policy_b, opening)
        if res == 1: a_pts += 1; w += 1
        elif res == 0: a_pts += 0.5; d += 1
        else: l += 1
        ng += 1
        # game 2: A white, B black (same opening)
        res, _, _ = play_game(policy_b, policy_a, opening)
        if res == -1: a_pts += 1; w += 1     # white(A) wins
        elif res == 0: a_pts += 0.5; d += 1
        else: l += 1
        ng += 1
    return {"a_winrate": a_pts / ng, "wins": w, "draws": d, "losses": l,
            "n_games": ng}
