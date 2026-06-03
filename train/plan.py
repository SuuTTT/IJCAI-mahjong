"""
plan.py — fan-potential rollout planner (test-time search) for Mahjong conversion.

The deployed greedy policy minimizes shanten, so it races to *tenpai* — but a fast
2-fan tenpai scores ZERO (8-fan floor). That is exactly why the bot draws ~90% of
games. This planner instead scores each candidate discard by the EXPECTED FAN of the
hands it can still complete, via Monte-Carlo self-draw rollouts against the unseen
tile pool, with completion validated by the real fan calculator (the 8-fan gate is
respected by construction).

It is a *solitaire* planner: it uses only legitimate information (my own hand + public
melds + public discards) and models my future draws from the unseen pool. It ignores
opponent claims/rong (conservative), which keeps it deployable (no hidden state) and
cheap. Opponent danger/defense is left to the base policy / future work.

Plugs into sim.py as a per-seat policy via a back-reference to the Sim (it reads only
its own seat's legitimate state). Falls back to the base NN policy for non-discard
decisions (claims, HU).
"""
import os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from data.feature_agent import ACT, TILE_LIST, TILE_INDEX
from train.sim import _fan, _full_wall
try:
    from MahjongGB import MahjongShanten
except Exception:
    MahjongShanten = None

SUITS = "WBT"
def _full_multiset():
    m = {}
    for s in SUITS:
        for n in range(1, 10): m[f"{s}{n}"] = 4
    for n in range(1, 5): m[f"F{n}"] = 4
    for n in range(1, 4): m[f"J{n}"] = 4
    return m

def _shan(pk, hand):
    try:
        return MahjongShanten(pack=pk, hand=tuple(hand))
    except Exception:
        return 99

def _playout_discard(h, pk):
    """Shanten-guided discard: among isolated/low-value candidates, drop the one whose
    removal keeps the lowest shanten. Only tests a few candidates (cheap) so the playout
    progresses toward a winnable hand without an O(distinct) shanten sweep every draw."""
    c = Counter(h)
    def conn(t):
        s = t[0]
        if s in SUITS:
            n = int(t[1]); v = (c[t] - 1) * 3
            for d in (-2, -1, 1, 2):
                v += c.get(f"{s}{n+d}", 0) * (2 if abs(d) == 1 else 1)
            return v
        return (c[t] - 1) * 3
    cand = sorted(set(h), key=conn)[:3]                 # 3 least-connected distinct tiles
    best, bs, bnew = cand[0], 99, 99
    for t in cand:
        rem = list(h); rem.remove(t)
        s = _shan(pk, rem)
        if s < bs: bs, best = s, t
    return best, bs

def expected_fan(concealed, packs, unseen, seat, quan, rng, n_mc=16, max_draws=18):
    """MC estimate of (mean_fan, win_rate) reachable by self-draw from this concealed hand.
    Shanten-guided playout; the real fan calculator is the win check (8-fan gate), gated by
    a tenpai test so it is only called when a win is actually possible."""
    pk = tuple((ty, tl, 1) for ty, tl in packs)
    total_fan = 0; wins = 0
    for _ in range(n_mc):
        h = list(concealed)
        pool = list(unseen); rng.shuffle(pool)
        sh = _shan(pk, h)
        for _i in range(min(max_draws, len(pool))):
            t = pool.pop(); h.append(t)
            if sh <= 0:                                  # was tenpai -> may have completed
                f = _fan(h, packs, t, seat, quan, True, False)
                if f >= 8:
                    wins += 1; total_fan += f; break
            d, sh = _playout_discard(h, pk)              # improve & discard; sh = new shanten
            h.remove(d)
    return total_fan / n_mc, wins / n_mc


class PlanningPolicy:
    """Per-seat policy: fan-potential planning on discards, base NN policy otherwise.
    Set .sim (the outer Sim) after the Sim is constructed; reads only own-seat state."""
    def __init__(self, seat, base_policy, n_mc=16, max_draws=16, top_k=6, seed=0):
        self.seat = seat; self.base = base_policy
        self.n_mc = n_mc; self.max_draws = max_draws; self.top_k = top_k
        self.rng = random.Random(1000 + seat + seed)
        self.sim = None

    def _unseen(self):
        """Tiles this seat cannot see = (4 - publicly shown) - my concealed hand.
        Uses the agent's own `shown` tally (discards + all melds), so it excludes my
        hand/melds, every discard, and opponents' melds — leaving opponents' concealed
        tiles + the wall: exactly the pool to determinize future draws from."""
        a = self.sim.agents[self.seat]
        myc = Counter(self.sim.hand[self.seat])
        out = []
        for t in TILE_LIST:
            cnt = max(0, 4 - a.shown.get(t, 0)) - myc.get(t, 0)
            if cnt > 0: out += [t] * cnt
        return out

    def __call__(self, obs, mask):
        m = mask[0]
        play_lo, play_hi = ACT["Play"], ACT["Chi"]
        # defer all decisions to the base policy; only REFINE a plain discard.
        # (Hu is listed as valid on every draw turn, so we must let the base decide
        #  Hu/claim/discard first, then plan only when it actually wants to discard.)
        a0 = int(self.base(obs, mask)[0])
        if self.sim is None or not (play_lo <= a0 < play_hi):
            return np.array([a0])
        hand = list(self.sim.hand[self.seat])
        packs = list(self.sim.melds[self.seat])
        unseen = self._unseen()
        probs, _ = self.base.m.forward(obs[0], m) if hasattr(self.base, "m") else (None, None)
        cands = []
        for a in range(play_lo, play_hi):
            if m[a]:
                t = TILE_LIST[a - play_lo]
                if t in hand: cands.append((a, t))
        if not cands:
            return np.array([a0])
        if probs is not None:
            cands.sort(key=lambda at: -probs[at[0]])
        cands = cands[: self.top_k]
        best_a, best_score = cands[0][0], -1.0
        for a, t in cands:
            rem = list(hand); rem.remove(t)
            mf, wr = expected_fan(rem, packs, unseen, self.seat, self.sim.quan,
                                  self.rng, self.n_mc, self.max_draws)
            if mf > best_score:
                best_score, best_a = mf, a
        return np.array([best_a])
