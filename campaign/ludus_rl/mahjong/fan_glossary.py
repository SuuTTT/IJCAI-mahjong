"""Human-readable glossary for Chinese Standard (MCR / 国标) mahjong fans (番).

The fan *values* and the fan *names* are produced by PyMahjongGB's
``MahjongFanCalculator`` — this module never invents point values.  What it adds
is, for each of the 81 standard fans (plus the combined kong fan), a short
one-line explanation of *what the pattern is*, so a player can learn why a hand
scored the way it did.

The Chinese keys and the English names are copied verbatim from PyMahjongGB's
own ``fan_name`` / ``fan_name_en`` tables
(``mahjong-algorithm/fan_calculator.h``), so ``en_for`` returns exactly the
string the calculator emits in verbose mode.

Public helpers:
    ``en_for(cn)``   -> English name for a Chinese fan name ("" if unknown)
    ``desc_for(cn)`` -> one-line description                ("" if unknown)
    ``entry(cn)``    -> {"en": ..., "desc": ...}
"""

# cn -> (english name, one-line description).  English strings are the exact
# PyMahjongGB verbose-mode names.
FANS = {
    # ---- 88 ----
    "大四喜": ("Big Four Winds",
             "Pungs (or kongs) of all four winds: East, South, West and North."),
    "大三元": ("Big Three Dragons",
             "Pungs (or kongs) of all three dragons: Red, Green and White."),
    "绿一色": ("All Green",
             "Hand built only from green tiles: bamboo 2,3,4,6,8 and the Green Dragon."),
    "九莲宝灯": ("Nine Gates",
              "A concealed one-suit hand of 1112345678999 waiting on any tile of that suit."),
    "四杠": ("Four Kongs", "Four kongs declared in a single hand."),
    "连七对": ("Seven Shifted Pairs",
             "Seven pairs in one suit on seven consecutive numbers."),
    "十三幺": ("Thirteen Orphans",
             "One of every terminal and honor (1/9 of each suit, all winds and dragons) plus a duplicate."),
    # ---- 64 ----
    "清幺九": ("All Terminals",
             "Every set is a pung or pair of pure terminals — only 1s and 9s, no honors."),
    "小四喜": ("Little Four Winds",
             "Pungs of three winds plus a pair of the fourth wind."),
    "小三元": ("Little Three Dragons",
             "Pungs of two dragons plus a pair of the third dragon."),
    "字一色": ("All Honors", "The whole hand is honor tiles — winds and dragons only."),
    "四暗刻": ("Four Concealed Pungs",
             "Four pungs (or kongs), every one concealed (none claimed from a discard)."),
    "一色双龙会": ("Pure Terminal Chows",
               "One suit: 123 and 789 each made twice, with a pair of 5s."),
    # ---- 48 ----
    "一色四同顺": ("Quadruple Chow", "Four identical chows in the same suit."),
    "一色四节高": ("Four Pure Shifted Pungs",
               "Four pungs in one suit, each shifted up by one number."),
    # ---- 32 ----
    "一色四步高": ("Four Pure Shifted Chows",
               "Four chows in one suit, each shifted up by the same 1 or 2 steps."),
    "三杠": ("Three Kongs", "Three kongs declared in one hand."),
    "混幺九": ("All Terminals and Honors",
             "Every set is a pung or pair of terminals (1/9) or honor tiles."),
    # ---- 24 ----
    "七对": ("Seven Pairs", "Seven pairs, fully concealed."),
    "七星不靠": ("Greater Honors and Knitted Tiles",
              "All seven honors as singles plus a knitted straight across the suits, no sets."),
    "全双刻": ("All Even Pungs",
             "Four pungs of even-numbered suited tiles (2,4,6,8) with an even pair."),
    "清一色": ("Full Flush", "The entire hand is a single suit, with no honor tiles."),
    "一色三同顺": ("Pure Triple Chow", "Three identical chows in the same suit."),
    "一色三节高": ("Pure Shifted Pungs",
               "Three pungs in one suit, each shifted up by one number."),
    "全大": ("Upper Tiles", "Every tile is a 7, 8 or 9."),
    "全中": ("Middle Tiles", "Every tile is a 4, 5 or 6."),
    "全小": ("Lower Tiles", "Every tile is a 1, 2 or 3."),
    # ---- 16 ----
    "清龙": ("Pure Straight", "A 1-to-9 run in one suit: 123-456-789."),
    "三色双龙会": ("Three-Suited Terminal Chows",
               "123 and 789 in two suits, with a pair of 5s in the third suit."),
    "一色三步高": ("Pure Shifted Chows",
               "Three chows in one suit, each shifted up by the same 1 or 2 steps."),
    "全带五": ("All Five", "Every set and the pair contains a 5."),
    "三同刻": ("Triple Pung", "The same-numbered pung in all three suits."),
    "三暗刻": ("Three Concealed Pungs", "Three concealed pungs (or kongs)."),
    # ---- 12 ----
    "全不靠": ("Lesser Honors and Knitted Tiles",
             "A knitted straight across suits plus honor singles — no sets and no pair."),
    "组合龙": ("Knitted Straight",
             "1-9 spread across the three suits in the fixed 147 / 258 / 369 knit."),
    "大于五": ("Upper Four", "Every tile is a 6, 7, 8 or 9."),
    "小于五": ("Lower Four", "Every tile is a 1, 2, 3 or 4."),
    "三风刻": ("Big Three Winds", "Pungs of three of the four wind tiles."),
    # ---- 8 ----
    "花龙": ("Mixed Straight",
           "A 1-9 run split across the three suits (e.g. 123 / 456 / 789 each a different suit)."),
    "推不倒": ("Reversible Tiles",
             "Only tiles that look the same upside down: dots 1,2,3,4,5,8,9, bamboo 2,4,5,6,8,9 and White Dragon."),
    "三色三同顺": ("Mixed Triple Chow", "The same chow in all three suits."),
    "三色三节高": ("Mixed Shifted Pungs",
               "Three pungs, one per suit, each shifted up by one number."),
    "无番和": ("Chicken Hand", "A legal win that scores no other fan at all."),
    "妙手回春": ("Last Tile Draw", "Winning by self-draw on the very last tile of the wall."),
    "海底捞月": ("Last Tile Claim", "Winning on the last discard of the hand."),
    "杠上开花": ("Out with Replacement Tile",
              "Winning on the replacement tile drawn right after declaring a kong."),
    "抢杠和": ("Robbing The Kong",
             "Winning on a tile an opponent adds to an existing pung to make a kong."),
    # ---- 6 ----
    "碰碰和": ("All Pungs", "Four pungs or kongs plus a pair — no chows."),
    "混一色": ("Half Flush", "One suit plus honor tiles only."),
    "三色三步高": ("Mixed Shifted Chows",
               "Three chows, one in each suit, each shifted up by the same amount."),
    "五门齐": ("All Types",
             "All five tile groups present: the three suits, winds and dragons."),
    "全求人": ("Melded Hand",
             "Every set is claimed from others and the win is on a discard — a fully melded hand."),
    "双暗杠": ("Two Concealed Kongs", "Two concealed kongs."),
    "双箭刻": ("Two Dragons Pungs", "Pungs of two different dragon tiles."),
    # ---- 4 ----
    "全带幺": ("Outside Hand",
             "Every set and the pair contains a terminal (1/9) or an honor tile."),
    "不求人": ("Fully Concealed Hand", "A fully concealed hand won by self-draw."),
    "双明杠": ("Two Melded Kongs", "Two exposed (melded) kongs."),
    "和绝张": ("Last Tile",
             "Winning on the fourth and final copy of a tile — the other three already showing."),
    # ---- 2 ----
    "箭刻": ("Dragon Pung", "A pung (or kong) of dragon tiles."),
    "圈风刻": ("Prevalent Wind", "A pung of the round's prevailing wind."),
    "门风刻": ("Seat Wind", "A pung of your own seat wind."),
    "门前清": ("Concealed Hand", "A fully concealed hand won on someone's discard."),
    "平和": ("All Chows", "Four chows and a plain-tile pair — all sequences, no honors."),
    "四归一": ("Tile Hog", "Holding all four copies of a suited tile without kong-ing them."),
    "双同刻": ("Double Pung", "Two pungs of the same number in two different suits."),
    "双暗刻": ("Two Concealed Pungs", "Two concealed pungs."),
    "暗杠": ("Concealed Kong", "A concealed (self-drawn) kong."),
    "断幺": ("All Simples", "No terminals or honors — every tile is a 2 through 8."),
    # ---- 1 ----
    "一般高": ("Pure Double Chow", "Two identical chows in the same suit."),
    "喜相逢": ("Mixed Double Chow", "The same chow in two different suits."),
    "连六": ("Short Straight", "Two consecutive chows in one suit (e.g. 123 + 456)."),
    "老少副": ("Two Terminal Chows", "123 and 789 in the same suit."),
    "幺九刻": ("Pung of Terminals or Honors", "A pung of a terminal (1/9) or honor tile."),
    "明杠": ("Melded Kong", "An exposed (melded) kong."),
    "缺一门": ("One Voided Suit", "The hand is missing one of the three suits entirely."),
    "无字": ("No Honors", "The hand contains no wind or dragon tiles."),
    "边张": ("Edge Wait",
           "Winning on the edge tile of a run — the 3 for a 1-2, or the 7 for an 8-9."),
    "嵌张": ("Closed Wait", "Winning on the middle tile that fills a gap, e.g. 1_3 waiting on 2."),
    "单钓将": ("Single Wait", "Winning by completing the lone pair — a single-tile pair wait."),
    "自摸": ("Self-Drawn", "Completing the winning hand on your own draw."),
    "花牌": ("Flower Tiles", "Bonus flower / season tiles (not dealt in this ruleset's wall)."),
    "明暗杠": ("Concealed Kong and Melded Kong",
             "One concealed kong together with one melded kong."),
}

# Reverse lookup by English name (for callers that only have the en string).
_BY_EN = {en: (cn, desc) for cn, (en, desc) in FANS.items()}


def entry(cn: str) -> dict:
    """{'en', 'desc'} for a Chinese fan name; blanks if unknown."""
    en, desc = FANS.get(cn, ("", ""))
    return {"en": en, "desc": desc}


def en_for(cn: str) -> str:
    return FANS.get(cn, ("", ""))[0]


def desc_for(cn: str = None, en: str = None) -> str:
    """Description by Chinese name (preferred) or English name."""
    if cn and cn in FANS:
        return FANS[cn][1]
    if en and en in _BY_EN:
        return _BY_EN[en][1]
    return ""
