"""
agari.py — winning-hand (和牌) structure detection for the JAX CSM env (Phase 2).

A standard win = 4 melds (sets) + 1 pair, formed from the 14 tiles (concealed hand + declared melds).
Detecting this fast & vectorized is the algorithmic core of a trainable self-play env.

APPROACH (the standard table trick, JAX-friendly):
  * A hand splits into 4 independent groups: 3 number suits (W,T,B, 9 ranks each) + honors (7 types).
  * For ONE number suit with rank-counts c[0..8], precompute every (n_sets, has_pair) it can form
    (runs i,i+1,i+2 and triplets i,i,i; at most the global one pair). Key = base-5 pack of counts.
  * Honors: triplets + pair only (no runs) — same idea, simpler.
  * A 14-tile hand wins iff some split across the 4 groups sums to need_sets sets + exactly one pair.
  * The per-suit feasibility table is a static numpy array -> in JAX the check becomes a gather + combine.

This module: numpy reference (`is_win`) validated against MahjongGB, plus the precomputed tables that
the JAX env will gather into. Special hands (七对/十三幺/...) handled separately (cheap count checks).
"""
import numpy as np
from functools import lru_cache

SUIT_LEN = 9
HONOR_LEN = 7


@lru_cache(maxsize=None)
def _suit_feasible(counts, allow_runs):
    """Return a frozenset of (n_sets, has_pair) achievable from this suit's counts.
    counts: tuple of 9 (or 7) ints. Recursively peel runs/triplets/one pair. has_pair in {0,1}."""
    c = list(counts)
    # find first non-zero rank
    i = next((k for k, v in enumerate(c) if v > 0), None)
    if i is None:
        return frozenset({(0, 0)})
    out = set()
    # option A: triplet at i
    if c[i] >= 3:
        c[i] -= 3
        for s, p in _suit_feasible(tuple(c), allow_runs):
            out.add((s + 1, p))
        c[i] += 3
    # option B: run i,i+1,i+2
    if allow_runs and i + 2 < len(c) and c[i] >= 1 and c[i + 1] >= 1 and c[i + 2] >= 1:
        c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
        for s, p in _suit_feasible(tuple(c), allow_runs):
            out.add((s + 1, p))
        c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
    # option C: pair at i (only if we still have the global pair to spend — tracked by caller via has_pair)
    if c[i] >= 2:
        c[i] -= 2
        for s, p in _suit_feasible(tuple(c), allow_runs):
            if p == 0:                     # at most one pair total within this group
                out.add((s, 1))
        c[i] += 2
    # must fully consume tiles for a valid decomposition: if any leftover, this branch is invalid
    # (handled because leftover single tiles produce no (s,p) — _suit_feasible only returns when empty)
    return frozenset(out) if out else frozenset()


def is_win(hand34, n_melds):
    """hand34: length-34 int counts of CONCEALED tiles. n_melds: # already-declared melds (each a set).
    Standard win: concealed tiles decompose into (4 - n_melds) sets + exactly 1 pair. Returns bool."""
    need = 4 - n_melds
    suits = [tuple(hand34[0:9]), tuple(hand34[9:18]), tuple(hand34[18:27])]
    honor = tuple(hand34[27:34]) + (0, 0)            # pad to 9 for the (run-disabled) solver shape
    groups = [_suit_feasible(s, True) for s in suits] + [_suit_feasible(honor[:7] + (0, 0), False)]
    # combine: total sets == need, total pairs == 1
    def combine(idx, sets, pairs):
        if idx == 4:
            return sets == need and pairs == 1
        for s, p in groups[idx]:
            if sets + s <= need and pairs + p <= 1:
                if combine(idx + 1, sets + s, pairs + p):
                    return True
        return False
    if any(len(g) == 0 for g in groups):
        return False
    return combine(0, 0, 0)


def is_seven_pairs(hand34, n_melds):
    # a 4-of-a-kind counts as TWO pairs (豪华七对); count pairs as v//2
    return n_melds == 0 and sum(hand34) == 14 and all(v in (0, 2, 4) for v in hand34) and \
        sum(v // 2 for v in hand34) == 7


def is_win_any(hand34, n_melds):
    """Standard OR seven-pairs. (13-orphans / knitted are rarer specials — add if RL needs them.)"""
    return is_win(hand34, n_melds) or is_seven_pairs(hand34, n_melds)


if __name__ == '__main__':
    # quick self-checks (no MahjongGB needed): known wins/non-wins
    def h(*pairs):
        a = [0] * 34
        for idx, c in pairs: a[idx] = c
        return a
    # 123 123 123 123 + 11 in W (4 runs + pair) -> WIN with 0 melds
    w = [0] * 34
    for i in range(3):
        for r in range(3): w[r] += 1
    # simpler: W: 1112345678999? construct a real one: 11 123 123 123 123 across W
    hand = [0]*34
    # W1x2 (pair) + W123 + W123 + ... need 4 sets+pair in one suit: 11 234 234 ... not trivial; use mixed
    # pair W1W1 + run W2W3W4 + run T2T3T4 + run B2B3B4 + triplet J1J1J1
    hand=[0]*34; hand[0]=2; hand[1]=hand[2]=hand[3]=1; hand[10]=hand[11]=hand[12]=1
    hand[19]=hand[20]=hand[21]=1; hand[31]=3
    print("expect WIN :", is_win_any(hand, 0))
    hand2=list(hand); hand2[0]=1; hand2[4]=1  # break the pair -> no pair, 14 tiles but invalid
    print("expect FALSE:", is_win_any(hand2, 0))
    sp=[0]*34
    for i in [0,5,10,15,20,25,30]: sp[i]=2
    print("expect 7pairs WIN:", is_win_any(sp,0))
