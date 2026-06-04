"""
fan_mask.py — 8-fan look-ahead for Chinese Standard Mahjong (research §8-fan masking).
Deploy-time conversion lever: avoid discards that lock the hand into a DEAD-END tenpai
(tenpai whose only winning tiles score < 8 fan). Pick, among the policy's top-K discard
candidates, the highest-prob one that keeps a >=8-fan path; fall back to the policy's top
if none qualifies. No retraining; usable in the bot and the RL sim.
"""
from MahjongGB import RegularShanten, SevenPairsShanten, ThirteenOrphansShanten, MahjongFanCalculator

def _shanten_and_wins(concealed, packs):
    """Min shanten over the main patterns + the set of winning tiles (only meaningful at tenpai)."""
    best = 99; wins = set()
    for fn in (RegularShanten, SevenPairsShanten, ThirteenOrphansShanten):
        try:
            s, useful = fn(tuple(concealed))
        except Exception:
            continue
        if s < best:
            best = s; wins = set(useful)
        elif s == best:
            wins |= set(useful)
    return best, wins

def max_fan_at_tenpai(concealed, packs, seat, quan):
    """If the 13-tile hand is tenpai, return the MAX fan over its winning tiles (rong, exposed-meld
    lower bound). Returns None if not tenpai. >=8 => '8-fan reachable now'; <8 => dead-end tenpai."""
    sh, wins = _shanten_and_wins(concealed, packs)
    if sh > 0:
        return None                      # not committed yet -> not a dead end
    pk = tuple((ty, tl, 1) for ty, tl in packs)
    best = -1
    for w in wins:
        try:
            r = MahjongFanCalculator(pack=pk, hand=tuple(concealed), winTile=w, flowerCount=0,
                                     isSelfDrawn=False, is4thTile=False, isAboutKong=False,
                                     isWallLast=False, seatWind=seat, prevalentWind=quan)
            f = sum(x for x, _ in r)
            if f > best: best = f
        except Exception:
            continue
    return best

def choose_discard(hand, packs, seat, quan, ranked_tiles, top_k=5):
    """ranked_tiles: legal discard tiles in the policy's preference order (best first).
    Returns the chosen discard: the highest-ranked candidate that is NOT a dead-end tenpai
    (i.e. keeps a >=8-fan path or is not yet tenpai). Falls back to ranked_tiles[0]."""
    cands = ranked_tiles[:top_k]
    for t in cands:
        if t not in hand:
            continue
        rem = list(hand); rem.remove(t)
        mf = max_fan_at_tenpai(rem, packs, seat, quan)
        if mf is None or mf >= 8:        # not committed, or a legal 8-fan tenpai -> keep
            return t
    return ranked_tiles[0]               # all top-K are dead-end tenpai -> no better option

if __name__ == '__main__':
    # discrimination test: a sub-8-fan tenpai vs an 8-fan-capable tenpai
    # all-chows, no fan source (dead-end, ~<8): W123 W456 B234 T567 + pair T9? build a clean low-fan tenpai
    low = ['W1','W2','W3','W4','W5','W6','B2','B3','B4','T5','T6','T7','T9']  # waiting T8 -> mostly low fan
    print('low-fan hand max tenpai fan:', max_fan_at_tenpai(low, [], 0, 0))
    # flush-ish (清一色 potential, high fan)
    hi = ['W1','W2','W3','W4','W5','W6','W7','W8','W9','W1','W2','W3','W9']
    print('flush hand max tenpai fan:', max_fan_at_tenpai(hi, [], 0, 0))
