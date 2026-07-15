"""Validate the 6x6 bitboard engine against an INDEPENDENT array reference
engine, plus known-position checks. Loud failure on any mismatch."""
import random
import othello as bb

N = 6
DIRS8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


# ---------- independent array reference engine ----------
def ref_initial():
    b = [[0] * N for _ in range(N)]  # 0 empty, 1 me, -1 opp (mover perspective)
    b[2][2] = -1  # white=opp
    b[3][3] = -1
    b[2][3] = 1   # black=me
    b[3][2] = 1
    return b


def ref_legal(b):
    moves = set()
    for r in range(N):
        for c in range(N):
            if b[r][c] != 0:
                continue
            for dr, dc in DIRS8:
                rr, cc = r + dr, c + dc
                seen_opp = False
                while 0 <= rr < N and 0 <= cc < N and b[rr][cc] == -1:
                    seen_opp = True
                    rr += dr
                    cc += dc
                if seen_opp and 0 <= rr < N and 0 <= cc < N and b[rr][cc] == 1:
                    moves.add(r * N + c)
                    break
    return moves


def ref_apply(b, sq):
    r, c = divmod(sq, N)
    nb = [row[:] for row in b]
    nb[r][c] = 1
    for dr, dc in DIRS8:
        rr, cc = r + dr, c + dc
        line = []
        while 0 <= rr < N and 0 <= cc < N and nb[rr][cc] == -1:
            line.append((rr, cc))
            rr += dr
            cc += dc
        if line and 0 <= rr < N and 0 <= cc < N and nb[rr][cc] == 1:
            for (fr, fc) in line:
                nb[fr][fc] = 1
    return [[-nb[r][c] for c in range(N)] for r in range(N)]  # swap perspective


def board_to_bb(b):
    me = opp = 0
    for r in range(N):
        for c in range(N):
            if b[r][c] == 1:
                me |= 1 << (r * N + c)
            elif b[r][c] == -1:
                opp |= 1 << (r * N + c)
    return me, opp


def test_known_start():
    me, opp = bb.initial()
    mv = set(bb.legal_move_list(me, opp))
    ref = ref_legal(ref_initial())
    assert mv == ref, f"start moves {sorted(mv)} != ref {sorted(ref)}"
    assert len(mv) == 4, f"expected 4 opening moves, got {len(mv)}"
    assert bb.popcount(me) == 2 and bb.popcount(opp) == 2
    print(f"OK  known 6x6 start: black has 4 moves {sorted(mv)}")


def test_cross_random(n_games=3000, seed=1):
    rng = random.Random(seed)
    total = 0
    for g in range(n_games):
        me, opp = bb.initial()
        rb = ref_initial()
        passes = 0
        while True:
            mvs = set(bb.legal_move_list(me, opp))
            rmvs = ref_legal(rb)
            assert mvs == rmvs, f"g{g}: legal bb={sorted(mvs)} ref={sorted(rmvs)}"
            rme, ropp = board_to_bb(rb)
            assert (rme, ropp) == (me, opp), f"g{g}: board mismatch"
            total += 1
            if not mvs:
                me, opp = opp, me
                rb = [[-rb[r][c] for c in range(N)] for r in range(N)]
                passes += 1
                if passes >= 2:
                    break
                continue
            passes = 0
            sq = rng.choice(sorted(mvs))
            me, opp = bb.apply_move(me, opp, sq)
            rb = ref_apply(rb, sq)
        assert bb.popcount(me) + bb.popcount(opp) == sum(
            1 for r in range(N) for c in range(N) if rb[r][c] != 0)
    print(f"OK  cross-validated {n_games} random games, {total} states: "
          f"legal-move sets + boards + terminals all match")


def test_terminal_scoring():
    me = (1 << 20) - 1
    opp = bb.FULL & ~me
    assert bb.popcount(me) == 20 and bb.popcount(opp) == 16
    assert bb.is_terminal(me, opp)
    assert bb.winner_score(me, opp) == 4
    print("OK  terminal scoring (full board 20-16 -> +4)")


if __name__ == "__main__":
    test_known_start()
    test_terminal_scoring()
    test_cross_random()
    print("ALL ENGINE TESTS PASSED")
