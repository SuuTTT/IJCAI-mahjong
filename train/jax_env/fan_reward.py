"""
fan_reward.py — terminal reward for the self-play env (hybrid: JAX detects the win, MahjongGB scores
the fan exactly, MCR duplicate rule converts fan -> 4-seat points). Validated: the MCR rule reproduces
50/52 real collected game finishes (the 2 misses are log-parse feeder ambiguity, not the rule; in the
env the feeder is known exactly).

Design: win-detection (agari_jax) runs every step on GPU. Only at the RARE terminal does fan scoring
happen on CPU via MahjongGB — exact, and cheap because terminals are infrequent. The 8-fan floor is
enforced here: a structural win below 8 fan scores ZERO (the core difficulty the RL must learn around).
"""
from MahjongGB import MahjongFanCalculator

TILES = [f'{s}{n}' for s in 'WTB' for n in range(1, 10)] + [f'F{n}' for n in range(1, 5)] + [f'J{n}' for n in range(1, 4)]


def fan_of(pack, hand_tiles, win_tile, self_drawn, seat_wind, prevalent, is_last=False, is_kong=False):
    """Exact fan via MahjongGB. pack: tuple of (type,tile,offer) melds; hand_tiles: concealed tile
    strings WITHOUT win_tile. Returns total fan (int) or 0 if not a legal win / below... (raw fan;
    8-floor applied by caller)."""
    try:
        fans = MahjongFanCalculator(pack=tuple(pack), hand=tuple(hand_tiles), winTile=win_tile,
                                    flowerCount=0, isSelfDrawn=self_drawn, is4thTile=False,
                                    isAboutKong=is_kong, isWallLast=is_last,
                                    seatWind=seat_wind, prevalentWind=prevalent, verbose=False)
        return sum(fp * cnt for fp, cnt, *_ in fans)
    except Exception:
        return 0


def mcr_scores(winner, feeder, fan, self_drawn):
    """MCR duplicate points (sum zero). self-draw: each of the 3 pays 8+fan to winner.
    discard win: feeder pays 8+fan, the other two pay 8 each. fan must already be >=8 (floor)."""
    s = [0, 0, 0, 0]
    if self_drawn:
        for p in range(4):
            if p != winner:
                s[p] -= (8 + fan); s[winner] += (8 + fan)
    else:
        for p in range(4):
            if p == winner:
                continue
            pay = (8 + fan) if p == feeder else 8
            s[p] -= pay; s[winner] += pay
    return s


def terminal_reward(winner, feeder, fan, self_drawn, our_seat=0):
    """Reward for OUR seat at a terminal. Enforces the 8-fan floor: fan<8 => not a scoring win => 0
    (the hand continues / draws in a real game; here we treat a sub-8 'win' as no-win)."""
    if fan < 8:
        return 0, None
    sc = mcr_scores(winner, feeder, fan, self_drawn)
    return sc[our_seat], sc
