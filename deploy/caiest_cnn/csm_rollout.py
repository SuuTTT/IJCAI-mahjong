# csm_rollout.py — opponent-aware determinized rollout for test-time PIMC (opt-in CAIEST_PIMC).
#
# WHY THIS, NOT plan.py: the prior fan-potential planner HURT because it was SOLITAIRE — it let our
# hand draw freely and ignored that opponents win first. This engine plays ALL FOUR seats forward
# from a determinized world (determinize.sample): each turn draw->discard, and after every draw a
# self-draw Hu check, after every discard a ROBBING Hu check for the other three seats. So an
# opponent can complete and end the game before us — the exact dynamic plan.py missed. Reward is the
# real MCR duplicate score for OUR seat. Rollout discard policy is cheap shanten-greedy (MCTS-style
# light playout); the EXPENSIVE 40-block net only chooses the ROOT candidate discards.
#
# Claims (Chi/Peng/Gang) are OFF in the playout (spike scope): draw->discard->Hu captures the
# first-order who-completes-first and deal-in dynamics. Add claims later if the held-out signal is +.
import os, random
from collections import Counter

try:
    from MahjongGB import MahjongFanCalculator, MahjongShanten
except Exception:
    MahjongFanCalculator = None
    MahjongShanten = None

SUITS = "WTB"
ROLL = os.environ.get('CAIEST_PIMC_ROLL', 'fan')   # 'fan' = conversion-biased playout, 'greedy' = shanten-only


def _shanten(pack, hand):
    try:
        return MahjongShanten(pack=tuple(pack), hand=tuple(hand))
    except Exception:
        return 99


def _fan_count(pack, hand, winTile, isSelfDrawn, seatWind, prevalentWind, is4th, isWallLast):
    """Return total fan if (pack,hand)+winTile is a legal >=8-fan win, else 0. hand EXCLUDES winTile
    for a robbed (discard) win; for self-draw winTile is the just-drawn tile already in hand -> we
    pass hand WITHOUT winTile to match feature.py._check_mahjong convention."""
    if MahjongFanCalculator is None:
        return 0
    try:
        fans = MahjongFanCalculator(
            pack=tuple(pack), hand=tuple(hand), winTile=winTile, flowerCount=0,
            isSelfDrawn=isSelfDrawn, is4thTile=is4th, isAboutKong=False,
            isWallLast=isWallLast, seatWind=seatWind, prevalentWind=prevalentWind, verbose=True)
        fc = sum(fp * cnt for fp, cnt, _, _ in fans)
        return fc if fc >= 8 else 0
    except Exception:
        return 0


def _convert_discard(hand, pack, rng):
    """Fan-aware playout policy: once one suit dominates the concealed hand, commit to a flush by
    shedding off-suit/honour tiles — so rollouts actually reach >=8-fan hands (Hun/Qing Yi Se,
    Peng Peng Hu) instead of stalling at cheap sub-8-fan tenpai. Below the commitment threshold it
    falls back to plain shanten-greedy. This is the ROLLOUT policy only (cheap, no net)."""
    c = Counter(hand)
    suit = {s: sum(v for t, v in c.items() if t[0] == s) for s in SUITS}
    maj = max(SUITS, key=lambda s: suit[s])
    if suit[maj] >= 6:
        # candidates to shed: off-suit number tiles (least connected), then lone honours
        offsuit = [t for t in set(hand) if t[0] in SUITS and t[0] != maj and c[t] < 3]
        lone_honor = [t for t in set(hand) if t[0] in 'FJ' and c[t] == 1]
        pool = sorted(offsuit, key=lambda t: c[t]) + sorted(lone_honor)
        if pool:
            best, bs = pool[0], 99
            for t in pool[:3]:
                rem = list(hand); rem.remove(t)
                s = _shanten(pack, rem)
                if s < bs:
                    bs, best = s, t
            return best
    return _greedy_discard(hand, pack, rng)


def _greedy_discard(hand, pack, rng):
    """Cheap shanten-guided discard: among the 3 least-connected distinct tiles, drop the one whose
    removal keeps the lowest shanten (ported from plan.py._playout_discard)."""
    c = Counter(hand)

    def conn(t):
        s = t[0]
        if s in SUITS:
            n = int(t[1]); v = (c[t] - 1) * 3
            for d in (-2, -1, 1, 2):
                v += c.get('%s%d' % (s, n + d), 0) * (2 if abs(d) == 1 else 1)
            return v
        return (c[t] - 1) * 3

    cands = sorted(set(hand), key=conn)[:3]
    best, bs = cands[0], 99
    for t in cands:
        rem = list(hand); rem.remove(t)
        s = _shanten(pack, rem)
        if s < bs:
            bs, best = s, t
    return best


def _scores_discard_win(winner, loser, fan):
    """MCR point-win (点和): winner gets (8+fan) from discarder + 8 from each other. Zero-sum."""
    s = [0, 0, 0, 0]
    for p in range(4):
        if p == winner:
            continue
        pay = (8 + fan) if p == loser else 8
        s[p] -= pay
        s[winner] += pay
    return s


def _scores_self_draw(winner, fan):
    """MCR self-draw (自摸): winner gets (8+fan) from EACH of the other three."""
    s = [0, 0, 0, 0]
    for p in range(4):
        if p == winner:
            continue
        s[p] -= (8 + fan)
        s[winner] += (8 + fan)
    return s


def rollout_once(hands, wall, packs, seatwinds, prevalent, start, max_turns, rng):
    """Play forward from `start` player's turn (they draw first). hands[p] lists, wall list (draw
    from end), packs[p] fixed. Returns the 4-vector MCR score (our seat is index 0). Draw if the
    wall empties or max_turns hit (score all-zero)."""
    hands = [list(h) for h in hands]
    wall = list(wall)
    shown = Counter()
    for p in range(4):
        for t in hands[p]:
            pass  # concealed, not shown
    cur = start
    for _turn in range(max_turns):
        if not wall:
            return [0, 0, 0, 0]
        wallLast = (len(wall) <= 1)
        # --- draw ---
        t = wall.pop()
        hands[cur].append(t)
        is4th = shown[t] == 4
        # self-draw Hu?
        hand_excl = list(hands[cur]); hand_excl.remove(t)
        fan = _fan_count(packs[cur], hand_excl, t, True, seatwinds[cur], prevalent, is4th, wallLast)
        if fan:
            return _scores_self_draw(cur, fan)
        # --- discard ---
        d = _convert_discard(hands[cur], packs[cur], rng) if ROLL == 'fan' else _greedy_discard(hands[cur], packs[cur], rng)
        hands[cur].remove(d)
        shown[d] += 1
        # robbing Hu by any other seat?
        is4d = shown[d] == 4
        for r in range(4):
            if r == cur:
                continue
            fan = _fan_count(packs[r], hands[r], d, False, seatwinds[r], prevalent, is4d, wallLast)
            if fan:
                return _scores_discard_win(r, cur, fan)
        cur = (cur + 1) % 4
    return [0, 0, 0, 0]


def evaluate_discard(my_hand_after, packs, hands_rest, wall, seatwinds, prevalent,
                     k_det, max_turns, rng, determ_sample):
    """EV of OUR seat after we have already discarded (my_hand_after = our concealed hand post-discard).
    determ_sample() returns a fresh {hands, wall} world each call (opponents+wall resampled).
    We fix our own hand; opponents/wall come from the determinization. Returns mean our-score."""
    total = 0.0
    for _ in range(k_det):
        world = determ_sample()
        hands = [list(my_hand_after), world['hands'][1], world['hands'][2], world['hands'][3]]
        s = rollout_once(hands, world['wall'], packs, seatwinds, prevalent,
                         start=1, max_turns=max_turns, rng=rng)  # after our discard, player 1 acts
        total += s[0]
    return total / max(k_det, 1)
